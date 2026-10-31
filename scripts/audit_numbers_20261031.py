# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""外部审计数字的数值复现（2026-10-31）。

逐项复现外部核对给出的四个数字，按分档判据（<0.1% 通过 / >1% 报错 /
量级不符硬失败）输出判定。验证对象：

1. fp32 sigmoid 饱和阈值：论文口径 17.3 vs 审计实测 16.636；
2. theta 网格公共零点：legacy / logfreq 网格在 tau=j(M+1)-1 处的
   min_m |sin((tau+1)*theta_m)|；
3. 单层流式状态字节：M=64/D=1024/K=4，论文 "512 KiB" vs 实现口径；
4. 命题 5 误差界的 1/|sin theta| 因子：实测误差对旧界
   max|delta|/(1-rho) 的违反倍数 vs 1/|sin theta| 预测。

运行：python scripts/audit_numbers_20261031.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

BAND = lambda dev, ref: abs(dev) / abs(ref) * 100 if ref else 0.0


def band(dev, ref):
    """相对偏差分档。"""
    if ref == 0:
        return "PASS" if dev == 0 else "FAIL(magnitude)"
    p = abs(dev - ref) / abs(ref) * 100
    if p < 0.1:
        return f"PASS(<0.1%, {p:.3f}%)"
    if p <= 1.0:
        return f"ERROR-REPORT({p:.2f}%)"
    return f"HARD-FAIL({p:.1f}%)"


# ---------------------------------------------------------------------------
# 1. fp32 sigmoid saturation threshold
# ---------------------------------------------------------------------------
def check_sigmoid():
    # scan: smallest fp32 x (as float32 grid) with float32 sigmoid(x) == 1.0
    lo, hi = 10.0, 20.0
    for _ in range(200):  # bisection on the exact rounding boundary
        mid = np.float32((lo + hi) / 2)
        if torch.sigmoid(torch.tensor(mid, dtype=torch.float32)).item() >= 1.0:
            hi = float(mid)
        else:
            lo = float(mid)
    x_sat = hi
    analytic_24 = 24.0 * math.log(2.0)   # 16.6355
    analytic_25 = 25.0 * math.log(2.0)   # 17.3287
    print(f"[1] sigmoid fp32 saturation: measured x*={x_sat:.4f}")
    print(f"    audit 16.636  -> {band(x_sat, 16.636)}")
    print(f"    paper 17.3    -> {band(x_sat, 17.3)}")
    print(f"    analytic 24ln2={analytic_24:.4f}, 25ln2={analytic_25:.4f}")
    # verify: sigmoid(16.0) < 1.0 in fp32 (below threshold it is not yet 1.0)
    below = torch.sigmoid(torch.tensor(16.0, dtype=torch.float32)).item()
    print(f"    sigmoid(16.0) fp32 = {below!r} (<1.0 expected: {below < 1.0})")
    return x_sat


# ---------------------------------------------------------------------------
# 2. common zeros of the legacy theta grid
# ---------------------------------------------------------------------------
def check_grid_zeros():
    """公共零点判定口径：tau 处 max_m |sin((tau+1)*theta_m)| —— 全体基
    同时为零（max≈0）才是"核恒为 0 且训练修不了"；min 只给出"最不幸
    单基"的 1/|sin| 放大因子（命题 5 的界）。两个口径都报告。"""
    print("[2] grid zeros at tau=j(M+1)-1: "
          "common-zero iff max_m|sin|==0; min_m|sin| -> prop5 1/|sin| factor")
    # 代数：theta_m=pi(m+0.5)/M 型均匀网格在 j(M+1) ≡ 0 (mod 2M) 处
    # 公共零点，最小 j = 2M/gcd(M+1,2)（M 偶 -> j=2M，M 奇 -> j=M）。
    for mode in ("legacy", "logfreq"):
        for M in (64, 128):
            if mode == "legacy":
                theta = math.pi * (np.arange(M) + 0.5) / M
            else:
                lam = np.exp(np.linspace(0, math.log(8192), M))
                th_lo = math.pi / 8192
                theta = np.exp(np.linspace(math.log(th_lo),
                                           math.log(math.pi / 2 * 0.9), M))
            jmax = 2 * M if mode == "legacy" else 16
            worst_min = (None, np.inf)
            common_zero = []
            for j in range(1, jmax + 1):
                tau1 = j * (M + 1)
                s = np.abs(np.sin(tau1 * theta))
                if s.max() == 0.0:
                    common_zero.append(j)
                if s.min() < worst_min[1]:
                    worst_min = (j, s.min())
            j, smin = worst_min
            print(f"    {mode:7s} M={M:4d}: worst single-basis j={j}, "
                  f"min|sin|={smin:.3e}, 1/|sin|={1.0/smin if smin else np.inf:.3e}; "
                  f"common-zero j: {common_zero if common_zero else 'none'}")


# ---------------------------------------------------------------------------
# 3. per-layer streaming state bytes
# ---------------------------------------------------------------------------
def check_state_bytes():
    from simplcore.core.sgc import SpectralGlobalConv
    m = SpectralGlobalConv(dim=1024, num_bases=64, local_kernel=4,
                           evidence_gate=True)
    b = m.streaming_state_bytes(1, dtype=torch.float32)
    kib = b / 1024
    print(f"[3] stream_state_bytes M=64 D=1024 K=4 fp32: {b} B = {kib:.2f} KiB")
    print(f"    paper '512 KiB'  -> {band(kib, 512.0)}")
    print(f"    audit '528 KiB'  -> {band(kib, 528.0)}")
    # breakdown
    parts = {
        "2*M*D": 2 * 64 * 1024 * 4,
        "(K-1)*D": 3 * 1024 * 4,
        "energy_sum": 4,
    }
    print(f"    breakdown: {parts} (sum={sum(parts.values())} B)")
    print(f"    + CMT memory (B,D)={1024*4} B -> {(b + 4096)/1024:.2f} KiB "
          f"(model-level incl. memory: matches 528.0?)")
    return b


# ---------------------------------------------------------------------------
# 4. Prop-5 bound: measured violation factor vs 1/|sin theta|
# ---------------------------------------------------------------------------
def check_prop5():
    from simplcore.core.sgc import SpectralGlobalConv
    torch.manual_seed(0)
    torch.set_printoptions(precision=8)
    M, L, T = 8, 4096, 512
    m = SpectralGlobalConv(dim=16, num_bases=M, local_kernel=1,
                           content_gate=False, evidence_gate=False,
                           uncertainty_gate=False, out_gate=False)
    with torch.no_grad():
        m.coef.normal_(0, 0.3)
        m.in_proj.weight.normal_(0, 0.2)
    rho, theta = m.oscillator_params()
    x = torch.randn(2, L, 16)

    # streaming path in float64 (reference) vs float32 (measured error)
    def stream(dtype):
        y1 = torch.zeros(2, M, 16, dtype=dtype)
        y2 = torch.zeros(2, M, 16, dtype=dtype)
        outs = []
        a = (2.0 * rho * torch.cos(theta)).to(dtype)
        b = (-(rho ** 2)).to(dtype)
        c = torch.sin(theta).to(dtype)
        for t in range(L):
            u = x[:, t].to(dtype).unsqueeze(1)          # (B,1,D)
            y_new = a.view(1, M, 1) * y1 + b.view(1, M, 1) * y2 \
                + c.view(1, M, 1) * u
            y2, y1 = y1, y_new
            outs.append(y_new)                          # (B,M,D)
        return torch.stack(outs, dim=1)                 # (B,L,M,D)

    y32 = stream(torch.float32).float()
    y64 = stream(torch.float64).float()
    e = (y32 - y64).abs()                               # (B,L,M,D)
    per_m = e.view(2, L, M, 16).amax(dim=(0, 1, 3))     # (M,) max err per basis
    u = 2.0 ** -24
    # old bound: u / (1 - rho); new bound: u / ((1-rho) * |sin theta|)
    old = u / (1 - rho.double())
    new = u / ((1 - rho.double()) * torch.sin(theta).double())
    ratio_old = (per_m.double() / old).max().item()
    ratio_new = (per_m.double() / new).max().item()
    one_over_sin = (1.0 / torch.sin(theta).double()).max().item()
    print(f"[4] prop5 per-basis max error vs bounds (M={M}, L={L}):")
    print(f"    max measured violation of OLD bound: {ratio_old:.3f}x")
    print(f"    max 1/|sin theta| on this grid:      {one_over_sin:.3f}")
    print(f"    new bound headroom (>=1 expected):   {ratio_new:.3f}x")
    print(f"    old-bound violated: {ratio_old > 1.0}; "
          f"1/sin explains: {abs(ratio_old - one_over_sin)/one_over_sin*100:.1f}%")


if __name__ == "__main__":
    check_sigmoid()
    print()
    check_grid_zeros()
    print()
    check_state_bytes()
    print()
    check_prop5()