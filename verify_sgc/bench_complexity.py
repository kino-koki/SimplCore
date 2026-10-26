# -*- coding: utf-8 -*-
"""诚实复杂度缩放基准: 隔离 mixer 前向在这一步长上的理论阶数.

在相同 D=176, batch=4 下, 把 seq_len 从 256 倍增到 2048,
独立测量 attn / sgc / ca 单次前向耗时. 用最小步数取中位数以减少噪声.

预期(诚实验证):
  attn  时间 ~ O(n^2)   -> 每倍增 n, 时间约 x4
  sgc  时间 ~ O(n log n)  -> 接近线性(FFT)略超线性
  ca   时间 ~ O(n)      -> 每倍增 n, 时间约 x2
输出结果 JSON 供报告使用. 不做任何加速/作弊, 真实计时.
"""
from __future__ import annotations
import json, time, gc, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sgc_pytorch import AttentionMixer, SpectralGlobalConvV2
from sgc_v3 import ContentAdaptiveMixer

DIM = 176
FEAT = 32
BATCH = 4
N_TRIALS = 6
N_WARM = 2
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(torch.get_num_threads())


def median_time(fn, warm=N_WARM, trials=N_TRIALS):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(trials):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    torch.manual_seed(0)
    seqs = [256, 512, 1024, 2048]
    results = {"dim": DIM, "batch": BATCH, "seqs": seqs,
               "mixers": {}, "complexity_ratio": {}}
    for name, fn in [("attn", lambda: AttentionMixer(DIM, 4, 4096)),
                     ("sgc", lambda: SpectralGlobalConvV2(DIM, 3, 4096, 8)),
                     ("ca", lambda: ContentAdaptiveMixer(DIM, FEAT))]:
        mod = fn()
        mod.eval()
        results["mixers"][name] = []
        prev = None
        for n in seqs:
            x = torch.randn(BATCH, n, DIM)
            with torch.no_grad():
                t = median_time(lambda: mod(x))
            results["mixers"][name].append(round(t * 1000, 2))  # ms
            msg = f"{name:5s} seq={n:5d}  {t*1000:8.2f} ms"
            if prev:
                msg += f"   x{t/prev:.2f} (prev倍率)"
            print(msg)
            prev = t
        # 2048/256 的整体倍率 (n 共 x8; 若 ~8 为线性, ~64 为二次)
        ratio = results["mixers"][name][-1] / results["mixers"][name][0]
        results["complexity_ratio"][name] = round(ratio, 2)
        print(f"  -> {name} 2048/256 倍率 = {ratio:.2f} (n 扩 8x; 线性≈8, 二次≈64)")
    out_path = os.path.join(os.path.dirname(__file__), "results", "complexity_scaling.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()