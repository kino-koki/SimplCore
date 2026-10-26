# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 56B MoE 训练主循环 (FSDP + 梯度检查点 + 余弦 LR + 辅助损失)。

可在 CPU 上做小尺度冒烟 (``--smoke``)，也可经 ``torchrun`` 在 GPU 集群上扩展。
仅依赖 PyTorch 与标准库。
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Iterable, Iterator, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.seeding import set_global_seed
from simplcore_pg.model_moe import SimplCoreMoEModel
from simplcore_pg.distributed import shard_model_for_distributed, plan_parallelism
from simplcore_pg.training.data import (
    SimpleChineseTokenizer, IterableCorpus, PackedDataset,
)

# 长上下文流式训练 (可选)
try:
    from simplcore_pg.model_stream import SimplCoreStreamingModel
except Exception:  # pragma: no cover
    SimplCoreStreamingModel = None


# ----------------------------------------------------------------------
# 合成中文语料 (冒烟用；真实训练替换为 --data-dir)
# ----------------------------------------------------------------------
_SAMPLE_CN = [
    "中华文明源远流长，汉字是其中最重要的载体之一。",
    "人工智能正在改变世界，大模型让机器理解了人类语言。",
    "子曰：学而时习之，不亦说乎？有朋自远方来，不亦乐乎？",
    "江南水乡的小桥流水，承载着千年的诗意与乡愁。",
    "量子计算利用叠加与纠缠，有望在特定问题上超越经典计算机。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "深度学习通过多层非线性变换，从数据中自动学习层次化表示。",
    "古道西风瘦马，夕阳西下，断肠人在天涯。",
]


def synthetic_corpus(n: int = 2000, seed: int = 0) -> Iterator[str]:
    rng = random.Random(seed)
    for _ in range(n):
        base = rng.choice(_SAMPLE_CN)
        # 随机拼接以产生长度变化
        k = rng.randint(1, 3)
        yield " ".join([base] * k)


# ----------------------------------------------------------------------
# 学习率调度 (warmup + cosine)
# ----------------------------------------------------------------------
def _lr_at_step(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
def train(config: SimplCoreConfig, corpus: Iterable[str], args) -> dict:
    # 全局随机种子：旧版本全包没有任何 manual_seed，导致权重初始化与数据顺序
    # 都不可复现（连续两次构造的模型权重不同）。这里统一收口。
    seed = int(getattr(args, "seed", 0) or 0)
    set_global_seed(seed, deterministic=bool(getattr(args, "deterministic", False)))

    tokenizer = SimpleChineseTokenizer(vocab_limit=config.vocab_size)
    dataset = PackedDataset(corpus, tokenizer, seq_len=args.seq_len, min_seq=32)

    model = SimplCoreMoEModel(config)
    if args.longctx:
        if SimplCoreStreamingModel is None:
            raise RuntimeError("长上下文流式模型不可用 (model_stream 导入失败)。")
        if not config.use_rfp or not config.use_dmc:
            print("[warn] 长上下文流式核心依赖 RFP/DMC 自研子层，"
                  "但 config.use_rfp/use_dmc 为 False；已强制开启。")
            config.use_rfp = config.use_dmc = True
        model = SimplCoreStreamingModel(config)
    if args.grad_ckpt:
        model.gradient_checkpointing = True

    # 分布式分片
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        model = shard_model_for_distributed(
            model, num_experts=config.num_experts, world_size=world_size,
            mp_policy=args.mp_policy,
        )

    rank = int(os.environ.get("RANK", 0))
    # 设备选择：CUDA > ROCm(HIP) > CPU；Windows 下 AMD 集显不可见，回退 CPU。
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    elif getattr(torch.version, "hip", None) is not None and torch.cuda.is_available():
        device = torch.device("cuda", rank)  # ROCm 暴露为 cuda 设备
    else:
        if world_size > 1:
            print("[warn] 分布式训练需要 GPU (CUDA/ROCm)，当前环境不可用，回退单进程 CPU。")
        device = torch.device("cpu")
    model.to(device)

    optimizer = model.configure_optimizers(lr=args.lr)
    use_amp = args.mp_policy in ("bf16", "fp16") and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.mp_policy == "fp16")

    max_steps = args.max_steps or args.epochs * max(1, args.max_steps or 1000)
    warmup = max(1, int(0.03 * max_steps))
    loader = iter(dataset)
    micro_bs = max(1, args.micro_batch)

    def _next_batch():
        ids, labs = [], []
        for _ in range(micro_bs):
            try:
                s = next(loader)
            except StopIteration:
                return None
            ids.append(s["input_ids"])
            labs.append(s["labels"])
        return {
            "input_ids": torch.stack(ids, dim=0),
            "labels": torch.stack(labs, dim=0),
        }

    log: dict = {"steps": 0, "tokens": 0, "loss_sum": 0.0, "aux_sum": 0.0}
    model.train()
    t0 = time.time()
    for step in range(1, max_steps + 1):
        batch = _next_batch()
        if batch is None:
            loader = iter(dataset)
            batch = _next_batch()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        lr = _lr_at_step(step, warmup, max_steps, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(input_ids, training=True)
            aux = model.last_aux_loss
            ce = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100
            )
            loss = ce + (args.aux_coeff * aux if aux is not None else 0.0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        log["steps"] += 1
        log["tokens"] += int(input_ids.numel())
        log["loss_sum"] += float(ce)
        log["aux_sum"] += float(aux) if aux is not None else 0.0

        if rank == 0 and step % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"[step {step}/{max_steps}] loss={float(ce):.4f} "
                f"aux={float(aux) if aux is not None else 0:.4f} "
                f"lr={lr:.2e} tok/s={log['tokens']/max(1e-6,dt):.0f}",
                flush=True,
            )

        if rank == 0 and args.ckpt_dir and step % args.ckpt_every == 0:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(
                {"step": step, "config": config.as_dict(),
                 "state": model.state_dict() if world_size == 1 else None},
                os.path.join(args.ckpt_dir, f"ckpt_{step}.pt"),
            )
    return log


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _main() -> None:
    p = argparse.ArgumentParser(description="SimplCore 56B MoE 训练")
    p.add_argument("--preset", default="56b-moe")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="0=按 epochs 推算；冒烟可设小值")
    p.add_argument("--seed", type=int, default=0,
                   help="全局随机种子 (Python/NumPy/torch)；0 也可复现")
    p.add_argument("--deterministic", action="store_true",
                   help="请求确定性算子 (更慢，但同机可逐位复现)")
    p.add_argument("--data-dir", default="", help="语料目录 (.jsonl/.txt)；留空使用合成中文语料")
    p.add_argument("--ckpt-dir", default="")
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--grad-ckpt", action="store_true", help="启用激活重计算 (省显存)")
    p.add_argument("--longctx", action="store_true",
                   help="启用长上下文流式训练 (固定状态，翻倍不爆显存)")
    p.add_argument("--chunk-size", type=int, default=1024, help="长上下文流式块大小")
    p.add_argument("--local-kernel", type=int, default=7, help="局部交互卷积核大小 (无注意力)")
    p.add_argument("--aux-coeff", type=float, default=1.0, help="辅助损失总系数")
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--mp-policy", default="none", choices=["none", "bf16", "fp16"])
    p.add_argument("--smoke", action="store_true", help="CPU 小尺度冒烟 (极小配置)")
    args = p.parse_args()

    if args.smoke:
        cfg = SimplCoreConfig(
            vocab_size=512, hidden_size=64, num_layers=2, num_experts=2,
            experts_per_tok=2, moe_intermediate_size=128, max_sequence_length=args.seq_len,
            dmc_num_heads=4,
            # 生产级默认启用自研架构 (RFP/DMC)，冒烟同样验证自研链路
            use_rfp=True, use_dmc=True, dropout=0.0,
        )
        corpus = synthetic_corpus(n=200)
        args.max_steps = args.max_steps or 20
        args.seq_len = min(args.seq_len, 128)
        if args.longctx:
            cfg.chunk_size = min(args.chunk_size, args.seq_len)
            cfg.local_kernel_size = args.local_kernel
    else:
        cfg = SimplCoreConfig.from_preset(args.preset)
        if args.data_dir:
            corpus = IterableCorpus(args.data_dir)
        else:
            corpus = synthetic_corpus(n=5000)
        args.max_steps = args.max_steps or (args.epochs * 1000)
        if args.longctx:
            cfg.chunk_size = args.chunk_size
            cfg.local_kernel_size = args.local_kernel

    log = train(cfg, corpus, args)
    print("训练完成:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in log.items()})


if __name__ == "__main__":
    _main()

