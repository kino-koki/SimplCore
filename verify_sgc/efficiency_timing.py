# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""长序列效率实测: SGC-v2 vs 因果注意力 在不同序列长度下的前向+反向耗时.

核心可发表论点: 因果注意力复杂度 O(n^2 d), SGC-v2 (FFT 因果卷积) O(n d log n).
本脚本在固定小模型上实测两者 wall-clock, 找出 crossover 点, 并给出理论 FLOPs 曲线.
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_pytorch import LMModel, count_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def time_mixer(mixer, vocab, dim, layers, heads, seq_len, rank, device,
               bs=4, warmup=3, reps=5):
    torch.manual_seed(0)
    model = LMModel(vocab, dim, layers, heads, seq_len, mixer,
                    sgc_rank=rank).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (bs, seq_len))
    # warmup
    for _ in range(warmup):
        opt.zero_grad(); l = model(x)
        loss = F.cross_entropy(l.reshape(-1, vocab), x.reshape(-1))
        loss.backward(); opt.step()
    # timed
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        opt.zero_grad(); l = model(x)
        loss = F.cross_entropy(l.reshape(-1, vocab), x.reshape(-1))
        loss.backward(); opt.step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    return {"seq_len": seq_len, "mixer": mixer, "median_s": med,
            "params": count_params(model)}


def theoretical_flops(n, d, L):
    """粗略 FLOPs (乘加) 估算, 用于曲线对照.
    注意力: O(n^2 d) 主项;  SGC: O(n d log n) 主项 (FFT).
    """
    attn = n * n * d * 2 * L
    sgc = n * d * L * (L.bit_length() + 1) * 6  # FFT 常数 ~ 5-6
    return attn, sgc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lengths", type=str, default="256,512,1024,2048,4096")
    args = ap.parse_args()
    device = "cpu"
    lengths = [int(x) for x in args.lengths.split(",")]

    rows = []
    for L in lengths:
        times = {}
        for mixer in ["attn", "sgc"]:
            r = time_mixer(mixer, args.vocab, args.dim, args.layers,
                           args.heads, L, args.rank, device)
            r["median_s_attn"] = None
            rows.append(r)
            times[mixer] = r["median_s"]
            spd = ""
            if "attn" in times:
                spd = f"  speedup_vs_attn={(times['attn']/r['median_s']-1)*100:+.0f}%"
            print(f"[{mixer}] L={L:5d}  median={r['median_s']:.4f}s{spd}")
    # 理论 FLOPs 对照 (取 L 列表)
    flops = {L: theoretical_flops(args.dim, args.dim, L) for L in lengths}
    json.dump({"timing": rows, "flops": {str(k): v for k, v in flops.items()},
               "args": vars(args)},
              open(os.path.join(RESULTS, "efficiency_timing.json"), "w"), indent=2)
    print("saved ->", os.path.join(RESULTS, "efficiency_timing.json"))


if __name__ == "__main__":
    main()
