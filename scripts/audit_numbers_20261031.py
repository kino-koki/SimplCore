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
    """公共零点判定口径（0.1-1 统一，与 sgc._assert_no_common_zero 一致）：
    **全 τ 扫描**（τ ∈ [1, 4M+16]，对齐真实零点位置 τ=2Mk-1；首个零点
    τ=2M-1、周期 2M）——旧脚本只扫 τ+1=j(M+1)，与真实零点位置不对齐
    （外部审计 2026-10-31 指出）。legacy 均匀网格同时受公式级拒绝
    （fp32 值级无法分辨代数零点，见 test_audit_guards.py）。两个报告口径：
    max_m|sin|（公共零点判定）与 min_m|sin|（命题 5 的 1/|sin| 放大因子）。"""
    print("[2] grid zeros, FULL-tau sweep (tau in [1, 4M+16]); "
          "common-zero iff max_m|sin|==0")
    for mode in ("legacy", "logfreq"):
        for M in (64, 128):
            if mode == "legacy":
                theta = math.pi * (np.arange(M) + 0.5) / M
            else:
                lam = np.exp(np.linspace(0, math.log(8192), M))
                th_lo = math.pi / 8192
                theta = np.exp(np.linspace(math.log(th_lo),
                                           math.log(math.pi / 2 * 0.9), M))
            T = 4 * M + 16
            tau = np.arange(1, T + 1)
            sins = np.abs(np.sin(tau[:, None] * theta[None, :]))  # (T, M)
            row_max = sins.max(axis=1)
            zero_taus = tau[row_max == 0.0]
            r = sins.max(axis=1) / np.sqrt(M)
            first = tau[row_max < 1e-6]
            print(f"  {mode:>7} M={M:>3}: min_tau max_m|sin| = "
                  f"{row_max.min():.3e} @ tau={tau[row_max.argmin()]}; "
                  f"exact common zeros: {list(zero_taus[:4])}"
                  f"{'...' if len(zero_taus) > 4 else ''}; "
                  f"near-zero(tau): {list(first[:4])}; "
                  f"min_tau ||Psi||/sqrt(M) = {r.min():.3e}")
    print("  legacy 均匀网格在 sgc 守护下被公式级拒绝（fp32 值级无法分辨"
          "代数零点）——论文 5.79x/5.13 的来源网格在 v1.2 默认路径不可构造。")


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


def check_prop5():
    """命题 5 双口径对照（0.1-α 统一，与 test_audit_guards._stream_error_vs_bound
    同口径）：

    * **口径 B（主口径，正确的界）**：δ ≤ k·u·max_t(|a||ỹ1|+|b||ỹ2|+c|u|)，
      k=5、u=2^-24——随 |ỹ| 增长，从 fp64 参考流重构。**只有此口径的界
      是真正的界**（误差注入随信号幅度缩放，见命题 5 注入系数分析）。
    * **口径 A（裸 u，错误口径，仅存档对照）**：δ ≤ u 常数——只在
      |ỹ|≤1 时成立，超出后"违反"数字是口径错误的产物，不是界的问题。

    外部审计（2026-10-31）双口径实测：B 口径旧界最坏 1.69x（违反）→
    新界 headroom 45.7x（成立）；A 口径旧界 106.4x、新界 1.38x（"违反"
    恰证明裸 u 不是界）。论文 5.79x/5.13 来自 legacy M=8 网格 + 未标口径
    （该网格已被守护拒绝，v1.2 默认路径不可构造）；logfreq M=64 默认
    网格的 1/|sinθ| = 1/sin(π/8192) ≈ 2607.6（与 §5.4 保守性警告的
    decay_max/π 口径一致——此前正文 5.13 与之矛盾，已勘误）。
    """
    from simplcore.core.sgc import SpectralGlobalConv
    M, L = 64, 512
    m = SpectralGlobalConv(dim=16, num_bases=M, local_kernel=1,
                           content_gate=False, evidence_gate=False,
                           uncertainty_gate=False, out_gate=False)
    with torch.no_grad():
        m.coef.normal_(0, 0.3)
    rho, theta = m.oscillator_params()
    x = torch.randn(2, L, 16)

    def stream(dtype):
        y1 = torch.zeros(2, M, 16, dtype=dtype)
        y2 = torch.zeros(2, M, 16, dtype=dtype)
        outs = []
        a = (2.0 * rho * torch.cos(theta)).to(dtype)
        b = (-(rho ** 2)).to(dtype)
        c = torch.sin(theta).to(dtype)
        for t in range(L):
            u = x[:, t].to(dtype).unsqueeze(1)
            y_new = a.view(1, M, 1) * y1 + b.view(1, M, 1) * y2                 + c.view(1, M, 1) * u
            y2, y1 = y1, y_new
            outs.append(y_new)
        return torch.stack(outs, dim=1)

    y64 = stream(torch.float64)
    y32 = stream(torch.float32).float()
    e = (y32.double() - y64).abs()
    per_m = e.view(2, L, M, 16).amax(dim=(0, 1, 3))          # (M,) 实测最坏

    # 口径 B：从 fp64 参考流重构 per-step 扰动（k=5，命题 5 定义）
    u = 2.0 ** -24
    a = (2.0 * rho * torch.cos(theta)).double().view(1, 1, M, 1)
    b = (-(rho ** 2)).double().view(1, 1, M, 1)
    c = torch.sin(theta).double().view(1, 1, M, 1)
    u_inj = x.double().unsqueeze(2)
    zeros1 = torch.zeros(2, 1, M, 16, dtype=torch.float64)
    zeros2 = torch.zeros(2, 2, M, 16, dtype=torch.float64)
    y_prev1 = torch.cat([zeros1, y64[:, :-1]], dim=1)
    y_prev2 = torch.cat([zeros2, y64[:, :-2]], dim=1)
    pert = (a.abs() * y_prev1 + b.abs() * y_prev2
            + c.abs() * u_inj).amax(dim=(0, 1, 3))            # (M,)
    kmax = 5.0
    bound_old_B = kmax * u * pert / (1 - rho.double())         # 缺 1/sin（旧界）
    bound_new_B = kmax * u * pert / ((1 - rho.double())
                                     * torch.sin(theta).double())  # 修正界
    ratio_old_B = (per_m.double() / bound_old_B).max().item()
    ratio_new_B = (per_m.double() / bound_new_B).max().item()
    # 口径 A（裸 u，错误口径对照）
    bound_old_A = u / (1 - rho.double())
    bound_new_A = u / ((1 - rho.double()) * torch.sin(theta).double())
    ratio_old_A = (per_m.double() / bound_old_A).max().item()
    ratio_new_A = (per_m.double() / bound_new_A).max().item()
    one_over_sin = (1.0 / torch.sin(theta).double()).max().item()
    print(f"[4] prop5 dual-caliber (M={M}, L={L}, logfreq default grid):")
    print(f"    1/|sin theta| on grid:               {one_over_sin:.1f} "
          f"(= decay_max/pi 口径; 论文旧值 5.13 来自 legacy M=8，已勘误)")
    print(f"    B(k*u*pert, 主口径)  old={ratio_old_B:.3f}x  "
          f"new={ratio_new_B:.3f}x  (new<1 => 修正界成立)")
    print(f"    A(裸 u, 错误口径)    old={ratio_old_A:.3f}x  "
          f"new={ratio_new_A:.3f}x  (『违反』恰证明裸 u 不是界——存档对照)")


if __name__ == "__main__":
    check_sigmoid()
    print()
    check_grid_zeros()
    print()
    check_state_bytes()
    print()
    check_prop5()