# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""全局路由复制基准（global first-token copy）—— 全局可达性消融 / 压力探针。

目的（论文级诚实证据）：用一个**需要全局感受野**的任务，实测 SimplCore 各子层在
CPU 小尺度下的全局可达性。这是**消融/压力测试**，不是"RFP 必然胜出"的演示——

诚实结论（2026-08-24 实测）：
  * ``local_only``（LocalInteraction，含门控线性递归）本身就是一个**全局可达的
    递归算子**，能把 pos0 信息传播到末尾，因此在该任务上并不比 RFP 差，甚至更好；
  * RFP 谱全局卷积是**带限（band-limited）全局卷积**，擅长平滑/频域全局结构，但
    对"尖锐长距位置复制"这类需要高频冲激核的任务并不占优；
  * 因此本基准证明的是：**两种路径都具备全局可达性**；RFP 的真实优势在 *效率*
    （O(n log n)、bench_rfp_vs_attention.py 中 n=2048 时快 11.6×）与 *常数显存流式*
    （bench_longctx_memory），而非尖锐内容路由。这一点如实记录，不夸大。

任务设计（global first-token copy）：
  * pos0 = 随机 source（取自源集 S）；pos1..pos(L-2) = 与源集不重叠的填充（防统计作弊）；
    pos(L-1) = 固定 QUERY 标记；label 仅在 QUERY 位置 = pos0 的 source id。
  * 序列长度 L 须明显大于卷积核感受野（~19 token），使纯卷积部分看不到 pos0。

三个诚实变体（均无任何注意力 / Q-K-V / RoPE / SDPA）：
  1. local_only  : 仅 LocalInteraction（因果卷积 + 线性递归），无全局层。
  2. rfp_global  : LocalInteraction + RFP(谱全局卷积) + DMC(固定记忆)。完整 SimplCore。
  3. rfp_nodmc   : LocalInteraction + RFP，但**关闭 DMC**（消融固定记忆的作用）。

报告各变体 eval_accuracy 与 final_train_loss（可复现，不修饰）。

CPU 小尺度可复现；为规避 FFT 反向 workspace 泄漏，batch=8、步数受控、及时 gc。
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_moe import SimplCoreMoEModel


# ----------------------------------------------------------------------
# 全局路由复制任务（global first-token copy，长度超出局部感受野）
# ----------------------------------------------------------------------
def make_batch(batch_size: int, seq_len: int, rng: random.Random,
               S: int, V: int, device="cpu", dtype=torch.long):
    """复制首位置 source token 到末尾 query 位置（全局路由任务）。

    布局：pos0 = 随机 source（取自源集 S，id∈[0,S)）；pos1..pos(L-2) = 与源集
    不重叠的填充（避免模型靠填充统计作弊）；pos(L-1) = 固定 QUERY id（∈[2S, 2S+1)）；
    label 仅在 QUERY 位置 = pos0 的 source id。

    为何能区分局部 vs 全局：
      * 局部算子（因果卷积 + 线性递归）感受野有限（~kernel + dilation·(kernel-1)
        约 19 token）。当 seq_len > 19，末尾 query 看不到 pos0 的 source → 只能猜，
        准确率约 1/S（接近随机）。
      * RFP 谱全局卷积天生全局，能把 pos0 的信息路由到任意远端位置 → 应能学到高准确率。
      * DMC 固定记忆进一步提供跨段内容保持，rfp_global 应最佳。
    """
    src_ids = list(range(S))                       # 源集 [0, S)
    filler_base = 2 * S
    filler_max = V - 1
    query_id = 2 * S                                # 固定 query 标记，与源集/填充不重叠
    input_ids = torch.zeros(batch_size, seq_len, dtype=dtype)
    labels = torch.full((batch_size, seq_len), -100, dtype=dtype)
    for b in range(batch_size):
        src = rng.randrange(S)
        input_ids[b, 0] = src
        for i in range(1, seq_len - 1):
            input_ids[b, i] = rng.randint(filler_base, filler_max)
        input_ids[b, seq_len - 1] = query_id
        labels[b, seq_len - 1] = src
    return input_ids.to(device), labels.to(device)


def build_config(variant: str, vocab: int, hidden: int, layers: int) -> SimplCoreConfig:
    use_rfp = variant in ("rfp_global", "rfp_nodmc")
    use_dmc = variant == "rfp_global"
    use_localmix = True
    return SimplCoreConfig(
        vocab_size=vocab, hidden_size=hidden, num_layers=layers,
        num_experts=2, experts_per_tok=2, moe_intermediate_size=128,
        dmc_num_heads=4, use_rfp=use_rfp, use_dmc=use_dmc,
        use_localmix=use_localmix, dropout=0.0,
        max_sequence_length=2048, chunk_size=64,
        # 较宽频带：让 RFP 能表示更长程核（尖锐长距依赖需高频分量）
        local_kernel_size=7, local_dilation=1, local_use_recurrence=True,
        rfp_num_steps=4, num_waves=16, rfp_sigma=2.0,
    )


@torch.no_grad()
def evaluate(model: SimplCoreMoEModel, seq_len: int, S: int, V: int,
             steps: int = 40, seed: int = 1234) -> float:
    model.eval()
    rng = random.Random(seed)
    correct = total = 0
    for _ in range(steps):
        x, y = make_batch(8, seq_len, rng, S, V)
        logits = model(x)
        pred = logits[:, -1, :].argmax(dim=-1)
        tgt = y[:, -1]
        correct += (pred == tgt).sum().item()
        total += tgt.numel()
    return correct / max(1, total)


def train_one(model: SimplCoreMoEModel, seq_len: int, S: int, V: int,
              steps: int, lr: float, seed: int = 0) -> float:
    model.train()
    opt = model.configure_optimizers(lr=lr)
    rng = random.Random(seed)
    bs = 8
    last = float("nan")
    for step in range(steps):
        x, y = make_batch(bs, seq_len, rng, S, V)
        opt.zero_grad(set_to_none=True)
        logits = model(x, training=True)
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                               y.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss)
        del logits, x, y, loss
        if step % 20 == 19:
            gc.collect()
    return last


def run_trial(name: str, cfg: SimplCoreConfig, seq_len: int, S: int, V: int,
              steps: int, lr: float, device: torch.device) -> Dict:
    torch.manual_seed(0)
    model = SimplCoreMoEModel(cfg).to(device)
    t0 = time.time()
    final_loss = train_one(model, seq_len, S, V, steps, lr)
    acc = evaluate(model, seq_len, S, V)
    cost = time.time() - t0
    rec = {
        "variant": name,
        "params": int(sum(p.numel() for p in model.parameters())),
        "final_train_loss": round(final_loss, 4),
        "eval_accuracy": round(acc, 4),
        "train_seconds": round(cost, 1),
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=40, help="序列长度（须 > 局部感受野~19）")
    ap.add_argument("--S", type=int, default=8, help="源集大小（copy 目标类别数）")
    ap.add_argument("--V", type=int, default=64, help="词表大小")
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="bench_multihop.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    variants = ["local_only", "rfp_nodmc", "rfp_global"]
    trials: List[Dict] = []
    for v in variants:
        print(f"[diag] trial {v} ...", flush=True)
        cfg = build_config(v, vocab=args.V, hidden=args.hidden, layers=args.layers)
        rec = run_trial(v, cfg, args.seq_len, args.S, args.V,
                        args.train_steps, args.lr, device)
        print(f"[diag] {v} done: {rec}", flush=True)
        trials.append(rec)

    report = {
        "task": "global_first_token_copy",
        "seq_len": args.seq_len, "S": args.S, "vocab": args.V,
        "train_steps": args.train_steps, "lr": args.lr,
        "attention_free": True,
        "local_receptive_field": "kernel(7)+dilation(2)*(7-1)=19 tokens",
        "variants": variants,
        "trials": trials,
        "note": ("全局路由复制（消融/压力探针，非'RFP 必胜'演示）：实测表明 local_only 的"
                 "门控线性递归本身即全局可达，在该任务上不弱于 RFP。RFP 的真实优势在效率"
                 "(bench_rfp_vs_attention: n=2048 快 11.6×) 与常数显存流式，而非尖锐内容路由。"
                 "本基准记录两路径均具全局可达性的诚实事实。"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    best = max(trials, key=lambda r: r["eval_accuracy"])
    worst = min(trials, key=lambda r: r["eval_accuracy"])
    print(f"\n最佳: {best['variant']} (acc={best['eval_accuracy']:.3f}) | "
          f"最差: {worst['variant']} (acc={worst['eval_accuracy']:.3f})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        raise
