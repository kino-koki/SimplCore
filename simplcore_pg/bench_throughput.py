# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 长序列吞吐基准 (诚实版)。

背景：本机为 Windows + AMD Radeon 680M，PyTorch 官方不支持 AMD 训练；
此处仅用 **CPU** 做相对吞吐测量，目的是量化「谱全局卷积(RFP) vs 原生局部交互
(LocalInteraction，无注意力) vs 标准全注意力(仅作复杂度基准)」在长序列下的
*复杂度阶数差异*，而非给出绝对 GPU 吞吐数字 (那需要在真机/WSL2+ROCm 上另测)。

测量量：
  * 单层在不同序列长度下的 wall-clock 前向时间 (CPU, fp32)；
  * 以 n=1024 归一化的相对时间，直观看到 RFP/Local 接近 O(n) 而
    全注意力接近 O(n^2)；
  * 每层理论 FLOPs 估算 (用于交叉验证阶数)。

自注意力 (SelfAttention) 仅作为外部复杂度基准参与对比，不属于 SimplCore 架构；
架构自身算子为 RFP (谱全局卷积) 与 LocalInteraction (因果卷积+线性递归)。

所有数字均来自真实 ``time.perf_counter`` 测量，不写死。

运行：
    python simplcore_pg/bench_throughput.py
输出：
    bench_throughput_result.json
"""
from __future__ import annotations

import json
import math
import time
import torch

from .config import SimplCoreConfig
from .resonance import ResonanceField
from .localmix import LocalInteraction
from .attention import SelfAttention  # 仅作为复杂度基准(外部对比)，非架构组成部分


def measure(fn, warmup: int = 1, reps: int = 3):
    """返回中位数耗时 (秒)。"""
    ts = []
    for _ in range(warmup):
        fn()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def layer_flops(n: int, H: int, heads: int, window: int, num_waves: int, num_steps: int, chunk: int) -> dict:
    """单层各组件理论 FLOPs 估算 (乘加各算 1 FLOP，近似)。"""
    hd = H // heads
    # 全注意力: 2 * B*heads*n*window*... 用全局 n^2 形式
    full_attn = 2 * heads * n * n * hd + heads * n * n
    # 窗口注意力: n 个 query 各看 window 个 key
    win_attn = 2 * heads * n * window * hd + heads * n * window
    # RFP 谱卷积: 每块 FFT O(c log c) * (n/c) 块 * W 通道 + 投影线性
    blocks = max(1, math.ceil(n / chunk))
    fft_per_block = 2 * chunk * math.log2(chunk) * 5  # rfft+irfft 近似常数
    rfp_fft = blocks * num_waves * fft_per_block
    rfp_linear = n * H * num_waves * 2 + n * num_waves * H * 2  # in/out proj
    rfp = rfp_fft + rfp_linear * num_steps
    return {
        "full_attention": int(full_attn),
        "windowed_attention": int(win_attn),
        "rfp_spectral": int(rfp),
    }


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    H = 256
    heads = 8
    num_waves = 8
    chunk = 256
    window = 256
    max_seq = 4096

    # 构造可测的纯组件 (不依赖完整模型)
    # 架构自身算子：RFP (谱全局卷积) 与 LocalInteraction (原生局部卷积+递归，无注意力)
    rfp = ResonanceField(H, num_waves=num_waves, num_steps=3, max_seq_len=chunk).to(device).eval()
    local = LocalInteraction(H, kernel_size=window, dilation=1, use_recurrence=True).to(device).eval()
    # 仅作复杂度基准的外部自注意力 (不参与架构)
    full_attn = SelfAttention(H, num_heads=heads, max_seq_len=max_seq, dropout=0.0, eps=1e-6).to(device).eval()

    lengths = [256, 512, 1024, 2048, 4096]
    results = {
        "device": "cpu",
        "dtype": "fp32",
        "config": {"H": H, "heads": heads, "num_waves": num_waves,
                    "chunk": chunk, "window": window, "max_seq": max_seq},
        "per_length": [],
        "note": "CPU 相对测量，量化复杂度阶数；非 GPU 绝对吞吐。",
    }

    # 预构建 RFP 固定状态与窗口 attn 历史 KV (模拟流式已建立)
    for n in lengths:
        x = torch.randn(1, n, H, device=device, dtype=dtype)

        # RFP 分块谱卷积 (单层)
        t_rfp = measure(lambda: rfp(x))
        # 原生局部交互 (LocalInteraction，无注意力)
        t_local = measure(lambda: local(x))
        # 全注意力 (单层，仅作基准，模块内部自建因果掩码)
        t_full = measure(lambda: full_attn(x, causal=True, seq_len=n)[0])

        flops = layer_flops(n, H, heads, window, num_waves, 3, chunk)
        rec = {
            "seq_len": n,
            "time_rfp_s": round(t_rfp, 4),
            "time_local_s": round(t_local, 4),
            "time_full_attn_s": round(t_full, 4),
            "flops_est": flops,
        }
        results["per_length"].append(rec)
        print(f"n={n:5d} | RFP {t_rfp*1000:7.1f}ms | Local {t_local*1000:7.1f}ms | "
              f"FullAttn {t_full*1000:7.1f}ms")

    # 归一化到 n=1024，验证阶数
    base = next(r for r in results["per_length"] if r["seq_len"] == 1024)
    for r in results["per_length"]:
        n = r["seq_len"]
        r["norm_rfp"] = round(r["time_rfp_s"] / base["time_rfp_s"], 3)
        r["norm_local"] = round(r["time_local_s"] / base["time_local_s"], 3)
        r["norm_full"] = round(r["time_full_attn_s"] / base["time_full_attn_s"], 3)
        # 理论倍率: RFP/Local ~ n/1024 (线性), Full ~ (n/1024)^2
        r["theory_linear_ratio"] = round(n / 1024, 3)
        r["theory_quadratic_ratio"] = round((n / 1024) ** 2, 3)

    with open("bench_throughput_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n已写出 bench_throughput_result.json")
    # 简要结论
    print("\n=== 复杂度阶数验证 (相对 n=1024) ===")
    for r in results["per_length"]:
        print(f"n={r['seq_len']:5d}: RFP实测×{r['norm_rfp']} (理论线性×{r['theory_linear_ratio']}) | "
              f"Local实测×{r['norm_local']} | Full实测×{r['norm_full']} (理论二次×{r['theory_quadratic_ratio']})")


if __name__ == "__main__":
    main()
