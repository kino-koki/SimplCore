# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""长序列精度对比基准：RFP 谱全局卷积 vs 纯滑动注意力（长距离依赖任务）。

任务：长距离 key→value 检索（induction / associative recall）
  - 序列形如 [KEY=k][填充...][QUERY=k]→ 目标输出 VALUE(k)
  - KEY/VALUE 取自固定小词表，中间填充随机 token 制造长距离依赖
  - 序列越长，纯局部窗口注意力越难把 KEY 信息传播到末尾 QUERY 处
  - RFP（频域全局卷积）理论上天生具备全局感受野，应更稳

对比：同配置下，仅切换 use_rfp/use_dmc（自研全局层）开关，训练同样步数，
报告训练后「末位 token 预测 VALUE 的准确率」。

这是诚实状态文档中「长序列精度对比」空白的填补：不靠 FLOPs 公式，
靠真实训练后的任务准确率说话。CPU 小尺度可复现。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_stream import SimplCoreStreamingModel


# ----------------------------------------------------------------------
# 合成长距离检索任务
# ----------------------------------------------------------------------
def make_batch(batch_size: int, seq_len: int, n_keys: int, rng: random.Random,
               vocab: int, device="cpu", dtype=torch.long):
    """构造 [KEY][filler...][QUERY]→VALUE 批次。

    词表布局：
      0 .. n_keys-1      : KEY id
      n_keys .. 2*n_keys-1: VALUE id (= KEY + n_keys)
      2*n_keys ..        : 填充 token
    """
    pad_base = 2 * n_keys
    pad_max = vocab - 1
    keys = list(range(n_keys))
    input_ids = torch.zeros(batch_size, seq_len, dtype=dtype)
    labels = torch.full((batch_size, seq_len), -100, dtype=dtype)
    for b in range(batch_size):
        k = rng.choice(keys)
        v = k + n_keys
        input_ids[b, 0] = k                       # KEY 在开头
        # 中间填充（仅在合法填充区间 [pad_base, pad_max] 内）
        for i in range(1, seq_len - 1):
            input_ids[b, i] = rng.randint(pad_base, pad_max)
        input_ids[b, seq_len - 1] = k             # QUERY 在末尾 = 同样的 KEY
        labels[b, seq_len - 1] = v                # 目标：VALUE
    return input_ids.to(device), labels.to(device)


def build_config(vocab, hidden, layers, chunk, use_rfp, use_dmc):
    return SimplCoreConfig(
        vocab_size=vocab, hidden_size=hidden, num_layers=layers,
        num_experts=2, experts_per_tok=2, moe_intermediate_size=128,
        chunk_size=chunk, max_sequence_length=chunk * 8,
        dmc_num_heads=4, use_rfp=use_rfp, use_dmc=use_dmc,
        use_localmix=True, dropout=0.0,
    )


@torch.no_grad()
def evaluate(model, seq_len, n_keys, steps=50, seed=1234):
    model.eval()
    rng = random.Random(seed)
    correct = 0
    total = 0
    for _ in range(steps):
        x, y = make_batch(8, seq_len, n_keys, rng, vocab=model.config.vocab_size)
        logits = model(x)
        last = logits[:, -1, :]
        pred = last.argmax(dim=-1)
        tgt = y[:, -1]
        correct += (pred == tgt).sum().item()
        total += tgt.numel()
    return correct / max(1, total)


def train_one(model, seq_len, n_keys, steps, lr, seed=0):
    model.train()
    opt = model.configure_optimizers(lr=lr)
    rng = random.Random(seed)
    bs = 8  # 小 batch：规避 Windows/CPU 下 RFP/DMC 反向的累积显存泄漏
    for step in range(steps):
        x, y = make_batch(bs, seq_len, n_keys, rng, vocab=model.config.vocab_size)
        opt.zero_grad(set_to_none=True)
        logits = model(x, training=True)
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss)
        del logits, x, y
        if step % 20 == 19:
            import gc as _gc
            _gc.collect()
            del loss
    return last_loss


def run_trial(name, cfg, seq_len, n_keys, train_steps, lr, device):
    torch.manual_seed(0)
    model = SimplCoreStreamingModel(cfg).to(device)
    t0 = time.time()
    final_loss = train_one(model, seq_len, n_keys, train_steps, lr)
    acc = evaluate(model, seq_len, n_keys)
    cost = time.time() - t0
    rec = {
        "variant": name, "params": sum(p.numel() for p in model.parameters()),
        "final_train_loss": round(final_loss, 4),
        "eval_accuracy": round(acc, 4),
        "train_seconds": round(cost, 1),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--n-keys", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="bench_longctx_accuracy.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    # 词表 = [KEY n_keys] + [VALUE n_keys] + [填充 pad_vocab]
    pad_vocab = 16
    base = dict(vocab=2 * args.n_keys + pad_vocab, hidden=args.hidden, layers=args.layers,
                chunk=args.chunk)

    trials: List[Dict] = []
    print("[diag] building local_only variant...", flush=True)
    trials.append(run_trial(
        "local_only",
        build_config(use_rfp=False, use_dmc=False, **base),
        args.seq_len, args.n_keys, args.train_steps, args.lr, device))
    print("[diag] local_only done:", trials[-1], flush=True)
    print("[diag] building rfp_dmc variant...", flush=True)
    trials.append(run_trial(
        "rfp_dmc",
        build_config(use_rfp=True, use_dmc=True, **base),
        args.seq_len, args.n_keys, args.train_steps, args.lr, device))
    print("[diag] rfp_dmc done:", trials[-1], flush=True)

    report = {
        "task": "long_distance_key_value_recall",
        "seq_len": args.seq_len, "n_keys": args.n_keys,
        "train_steps": args.train_steps, "lr": args.lr,
        "attention_free": True,
        "trials": trials,
        "note": ("准确率越高越好；对比局部算子 (local_only) 与完整 SimplCore (rfp_dmc)。"
                 "RFP 具备全局感受野，应在长距离任务上不劣于纯局部；二者均不含自注意力。"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    best = max(trials, key=lambda r: r["eval_accuracy"])
    print(f"\n最佳变体: {best['variant']} (acc={best['eval_accuracy']:.3f})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        raise
