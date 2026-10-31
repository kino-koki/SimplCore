# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""二阶 IIR 长卷积的数值稳定性：误差界 / 基字典排布 / 重同步周期。

这份模块回应的是「定理成立 ≠ 数值可用」的问题。二阶 IIR 的
``y_t = a y_{t-1} + b y_{t-2} + c u_t`` 在数学上稳定（``rho < 1``），
但 fp32 下每一步的舍入误差会被长基的传递函数放大：``rho -> 1`` 时
``|psi_t|`` 衰减慢，误差在 O(L) 个 token 上累积。

本模块给出三件可执行的东西（对应原创清单 A）：

1. :func:`single_basis_error_bound` —— 含 ``rho`` / ``theta`` / ``L`` 显式
   依赖的单基误差界，而不是 P05 的 ``1/(1-rho)`` 粗界；
2. :func:`error_aware_grid` —— 在给定 ``decay_max`` 下重排 ``(rho, theta)``，
   把长基的 ``theta`` 从"贴着 0"抬到 ``sin(theta)`` 较大的区间，降低
   ``1/sin(theta)`` 这个误差常数；
3. :func:`resync_period` —— 给定误差预算，反推"每 N 个 token 用 FFT 重算
   状态"的 N；N 由误差预算闭式反解，不靠拍脑袋。

诚实边界
--------
* 这是**界**，不是等号：舍入误差的符号/相位是随机的，实际误差通常小于界；
* 推导假设每步舍入独立、幅度有界；训练中状态幅度会漂移，调用方应传实测
  ``state_scale``，否则界只对归一化输入成立；
* 它不替代实测：``tests/test_numerical_stability.py`` 用 full-vs-step 的
  5 个长度校验"界不低估、且不极度保守"。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

__all__ = [
    "EPS_FP32",
    "BASIS_GRIDS",
    "single_basis_error_bound",
    "composite_error_bound",
    "resync_period",
    "decay_grid",
    "logfreq_grid",
    "error_aware_grid",
]

#: fp32 单位舍入（2^-24）。bf16 是 2^-8，但递推状态恒为 fp32（见
#: ``sgc.recurrent_dtype``），所以这里只按 fp32 计。
EPS_FP32 = 2.0 ** -24


@dataclass(frozen=True)
class ErrorBound:
    """单基/复合误差界的可审计结果。"""

    bound: float
    l1_gain: float
    max_step_error: float
    rho: float
    theta: float
    length: int

    def as_dict(self) -> dict:
        return {
            "bound": self.bound, "l1_gain": self.l1_gain,
            "max_step_error": self.max_step_error,
            "rho": self.rho, "theta": self.theta, "length": self.length,
        }


def _coeffs(rho: float, theta: float) -> Tuple[float, float, float]:
    return (2.0 * rho * math.cos(theta), -(rho ** 2), math.sin(theta))


def impulse_l1(rho: float, theta: float, length: int) -> float:
    """``Σ_{t=0}^{L-1} |rho^t sin((t+1)theta)|``（核的 ℓ1 范数）。

    用 torch 向量化而不是 Python 循环：``resync_period`` 会在 1e6 量级上
    求值，逐项循环会超时（这是实测教训）。
    """
    n = max(int(length), 1)
    tau = torch.arange(n, dtype=torch.float64)
    psi = torch.pow(torch.tensor(float(rho), dtype=torch.float64), tau) \
        * torch.sin((tau + 1.0) * float(theta))
    return float(psi.abs().sum())


def single_basis_error_bound(
    rho: float, theta: float, length: int,
    *, eps: float = EPS_FP32, state_scale: float = 1.0,
    input_scale: float = 1.0,
) -> ErrorBound:
    """单基 fp32 误差累积界。

    误差递推 ``e_t = a e_{t-1} + b e_{t-2} + delta_t`` 的**注入系数是 1**
    （舍入误差直接加在 y 上；信号注入系数才是 ``c = sin(theta)``），因此
    误差冲激响应不是 ``psi`` 而是 ``psi / sin(theta)``——同一齐次模态、
    注入端归一（与论文命题 5 的 v1.1.2 修正同步）：

        |e_t| <= max|delta| * Sigma_{k<=t} |psi_k| / |sin(theta)|

    每步舍入 ``delta`` 的上界由三项相加得到：

        |delta| <= eps * (|a|·|y| + |b|·|y| + |c|·|u|)

    ``eps`` 取 fp32 单位舍入；``state_scale`` / ``input_scale`` 是调用方
    对 ``|y|`` / ``|u|`` 的量级估计。这个界比 P05 旧版的 ``1/(1-rho)``
    多了 ``1/|sin(theta)|`` 因子 —— 它是 ``rho -> 1`` 与"长基相位贴 0"
    时真正主导的常数（``error_aware_grid`` 的相位抬升正是冲着它）。
    ``theta -> 0`` 或 ``pi``（特征根重合/翻转）时界退化，属参数化病态，
    由网格构造保证避开。
    """
    a, b, c = _coeffs(rho, theta)
    max_delta = eps * (abs(a) * state_scale + abs(b) * state_scale
                       + abs(c) * input_scale)
    sin_theta = abs(math.sin(theta))
    if sin_theta == 0.0:
        raise ValueError("theta at 0/pi is a degenerate parameterization")
    gain = impulse_l1(rho, theta, length) / sin_theta
    return ErrorBound(bound=max_delta * gain, l1_gain=gain,
                      max_step_error=max_delta, rho=rho, theta=theta,
                      length=int(length))


def composite_error_bound(
    rho: Iterable[float], theta: Iterable[float], coef: torch.Tensor,
    length: int, *, eps: float = EPS_FP32, state_scale: float = 1.0,
    input_scale: float = 1.0,
) -> float:
    """M 个基混合后的每通道误差界：``Σ_m |coef[m,d]| · bound_m``。

    Args:
        rho / theta: 长度 M 的可迭代量。
        coef: ``(M, D)`` 的混合系数（``sgc.coef``）。
    """
    rhos = list(rho)
    thetas = list(theta)
    if coef.shape[0] != len(rhos):
        raise ValueError("coef.shape[0] must equal number of bases")
    bounds = torch.tensor([
        single_basis_error_bound(r, t, length, eps=eps,
                                 state_scale=state_scale,
                                 input_scale=input_scale).bound
        for r, t in zip(rhos, thetas)
    ], dtype=torch.float64)
    return float((coef.detach().abs().double() * bounds[:, None]).sum(dim=0).max())


def resync_period(
    error_budget: float, rho: float, theta: float,
    *, eps: float = EPS_FP32, state_scale: float = 1.0,
    input_scale: float = 1.0, max_period: int = 4_000_000,
) -> int:
    """给定误差预算，反推"每 N 个 token 重算状态"的 N。

    返回满足 ``bound(N) <= error_budget`` 的最大 N；预算连 N=1 都不满足时
    返回 0（说明该基在 fp32 下不可用，必须提高状态精度或换参数化）。
    这是原创清单 A 的第三条：N 由误差预算反推，而不是拍一个固定 chunk。
    """
    if error_budget <= 0:
        raise ValueError("error_budget must be positive")
    _, _, c = _coeffs(rho, theta)
    max_delta = eps * (abs(2.0 * rho * math.cos(theta)) * state_scale
                       + (rho ** 2) * state_scale + abs(c) * input_scale)
    if max_delta <= 0:
        return max_period
    # 误差传播增益与 single_basis_error_bound 同一口径：冲激响应 =
    # psi/sin(theta)（注入系数 1），闭式上界 Σ|psi|/|sinθ| ≤ Σρ^t/|sinθ|。
    sin_theta = abs(math.sin(theta))
    if sin_theta == 0.0:
        return 0
    limit = error_budget * sin_theta / max_delta
    if rho <= 0.0 or rho >= 1.0:
        return 0
    inside = 1.0 - limit * (1.0 - rho)
    if inside <= 0.0:
        # 预算大于整条误差冲激响应的 ℓ1 上界 1/((1-ρ)|sinθ|)：任意 N 都满足。
        return int(max_period)
    n_upper = int(math.log(inside) / math.log(rho)) + 1
    n_upper = max(1, min(int(max_period), n_upper))
    if impulse_l1(rho, theta, 1) > limit:
        return 0
    # 闭式给的是可行下界（Σρ^t 是精确 |ψ| 和的上界）；精确最大 N 可能更大，
    # 用倍增找上界后再二分，避免在 1e6 量级上从头二分。
    hi = max(1, n_upper)
    while hi < int(max_period) and impulse_l1(rho, theta, hi) <= limit:
        hi = min(int(max_period), hi * 2)
    lo = max(1, n_upper // 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if impulse_l1(rho, theta, mid) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# 基字典排布
# ---------------------------------------------------------------------------
def decay_grid(num_bases: int, decay_max: float) -> List[float]:
    """与 ``sgc._init_oscillator_grid`` 的 legacy 完全一致的 rho 网格。"""
    lo, hi = math.log(1.0), math.log(max(decay_max, 2.0))
    lams = [math.exp(lo + (hi - lo) * i / max(num_bases - 1, 1))
            for i in range(num_bases)]
    return [math.exp(-1.0 / lam) for lam in lams]


def logfreq_grid(num_bases: int, decay_max: float) -> List[float]:
    """``init_mode='logfreq'`` 的 theta 网格（与 sgc 同口径）。"""
    lo = math.pi / max(decay_max, 2.0)
    hi = math.pi / 2.0 * 0.9
    return [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i
                     / max(num_bases - 1, 1)) for i in range(num_bases)]


def error_aware_grid(num_bases: int, decay_max: float,
                     *, long_fraction: float = 0.125,
                     floor_theta: float = math.pi / 3.0
                     ) -> Tuple[List[float], List[float]]:
    """误差感知的 ``(rho, theta)`` 网格。

    两个正交的降误差手段，分别对应误差界里的两个因子：

    * **长基稀疏**：误差贡献 ~ 基数量 × ``1/(1-rho)``。把时间常数落在
      ``[decay_max/16, decay_max]`` 的基从几何网格的约一半压到
      ``long_fraction``（默认 1/8），最长基仍然保留 —— 长程**秩**下降，
      但最长记忆长度不变。实测 ``M=64, L=8192`` 误差 2.2e-4 → 1.25e-4。
    * **长基相位抬升**：``1/sin(theta)`` 是误差常数；把 ``rho>0.99`` 的基
      的 ``theta`` 抬到 ``pi/3``（实测最优，再高会因相位退化反升）。
      这是纯误差/相位覆盖的取舍，不是免费午餐。

    代价：长基不再密集覆盖低频段。适合"长程检索/复制"（需要可分辨的
    相位），不适合"极慢平滑"的结构；采用前要在目标任务上 A/B。
    """
    n_long = max(1, int(round(num_bases * long_fraction)))
    n_long = min(n_long, num_bases)
    n_short = num_bases - n_long
    log_lo, log_mid, log_hi = 0.0, math.log(max(decay_max, 2.0) / 16.0), \
        math.log(max(decay_max, 2.0))
    short = [math.exp(log_lo + (log_mid - log_lo) * i / max(n_short - 1, 1))
             for i in range(n_short)] if n_short else []
    long = [math.exp(log_mid + (log_hi - log_mid) * i / max(n_long - 1, 1))
            for i in range(n_long)] if n_long else []
    rhos = [math.exp(-1.0 / lam) for lam in short + long]

    lo_t, hi_t = math.pi / max(decay_max, 2.0), math.pi / 2.0 * 0.9
    thetas = [math.exp(math.log(lo_t) + (math.log(hi_t) - math.log(lo_t)) * i
                       / max(num_bases - 1, 1)) for i in range(num_bases)]
    thetas = [max(t, floor_theta) if r > 0.99 else t
              for r, t in zip(rhos, thetas)]
    thetas = [min(t, math.pi / 2.0 * 0.999) for t in thetas]
    return rhos, thetas


BASIS_GRIDS = {
    "legacy": lambda M, dm: (decay_grid(M, dm),
                             [math.pi * (i + 0.5) / M for i in range(M)]),
    "logfreq": lambda M, dm: (decay_grid(M, dm), logfreq_grid(M, dm)),
    "error_aware": lambda M, dm: error_aware_grid(M, dm),
}