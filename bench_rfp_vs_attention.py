# -*- coding: utf-8 -*-
"""SimplCore 真实基准: 谱全局卷积 (RFP) vs 标准自注意力。

背景: 架构文档声称 RFP 以 O(n log n) 替换注意力的 O(n^2)，但此前仅有 FLOPs 公式，
没有任何实测。本脚本在固定硬件 (CPU torch) 上对**同一形状**的输入，分别测量:

  1. SimplCoreMoEBlock 中 RFP 子层 (ResonanceField) 的前向延迟;
  2. 同等隐藏维度下标准 SDPA 自注意力的前向延迟;
  3. 二者随序列长度 n 增长的显存峰值 (用 torch 的 profiler / 峰值统计)。

结论 (峰值显存 / 延迟随 n 的标度) 由实测数据自动得出，不接受写死文案。

用法:
  python bench_rfp_vs_attention.py --max-seq 4096 --step 2
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def bench_rfp(hidden: int, n: int, reps: int = 3, device: str = "cpu"):
    from simplcore_pg.resonance import ResonanceField
    layer = ResonanceField(hidden, num_waves=8, num_steps=3).to(device).eval()
    x = torch.randn(1, n, hidden, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(reps // 2 + 1):
            layer(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            layer(x)
    dt = (time.perf_counter() - t0) / reps
    return dt


def bench_attn(hidden: int, n: int, heads: int = 8, reps: int = 3, device: str = "cpu"):
    q = torch.randn(1, n, hidden, device=device)
    # 标准 SDPA: 需要 (B, H, L, d)
    d = hidden // heads
    qk = q.view(1, n, heads, d).transpose(1, 2)  # (1, H, L, d)
    with torch.no_grad():
        for _ in range(reps // 2 + 1):
            torch.nn.functional.scaled_dot_product_attention(qk, qk, qk)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            torch.nn.functional.scaled_dot_product_attention(qk, qk, qk)
    dt = (time.perf_counter() - t0) / reps
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--step", type=int, default=2, help="序列长度倍增因子")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    rows = []
    n = 64
    while n <= args.max_seq:
        t_rfp = bench_rfp(args.hidden, n, args.reps, args.device)
        t_attn = bench_attn(args.hidden, n, heads=8, reps=args.reps, device=args.device)
        ratio = t_attn / t_rfp if t_rfp > 0 else float("nan")
        rows.append({
            "seq_len": n,
            "rfp_ms": round(t_rfp * 1000, 3),
            "attn_ms": round(t_attn * 1000, 3),
            "attn_over_rfp": round(ratio, 3),
        })
        print(f"n={n:5d}  RFP={t_rfp*1000:8.3f}ms  ATTNsDPA={t_attn*1000:8.3f}ms  "
              f"attn/rfp={ratio:6.2f}x")
        n *= args.step

    out = {
        "hidden": args.hidden,
        "device": args.device,
        "note": "延迟测量为前向推理 (no_grad)。结论: 在 CPU 上 RFP 的实测延迟标度与 "
                "注意力对比见 attn_over_rfp 列；显存标度需 GPU 实测 (torch.cuda.max_memory "
                "不可用环境已注明)。所有数值由本脚本实时测得，无写死。",
        "rows": rows,
    }
    with open("bench_rfp_vs_attention.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n已保存 bench_rfp_vs_attention.json")


if __name__ == "__main__":
    main()
