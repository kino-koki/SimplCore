# -*- coding: utf-8 -*-
"""数值稳定性理论的验证：界不低估、重同步周期自洽、误差感知网格更优。

对应原创清单 A 的三条判据（`docs/NUMERICAL_STABILITY.md`）：

1. 误差界能在同一数量级预测实测（这里用单基 L=2048 校验）；
2. 误差感知网格相对对数网格降低累积误差（L=2048 实测 ≥2×）；
3. 重同步周期 N 由误差预算反解，且边界自洽。
"""
from __future__ import annotations

import math

import pytest
import torch

from simplcore.core.numerical_stability import (
    BASIS_GRIDS, composite_error_bound, error_aware_grid, impulse_l1,
    resync_period, single_basis_error_bound,
)
from simplcore.core.sgc import SpectralGlobalConv


def _set_grid(m: SpectralGlobalConv, rho, theta) -> None:
    with torch.no_grad():
        m.raw_rho.copy_(torch.logit(torch.tensor(rho, dtype=torch.float32),
                                    eps=1e-9))
        m.raw_theta.copy_(torch.logit(
            torch.tensor(theta, dtype=torch.float32) / math.pi, eps=1e-9))


def _drift(m: SpectralGlobalConv, L: int, *, dim: int = 16):
    x = torch.randn(1, L, dim)
    with torch.no_grad():
        full = m(x)
        st = m.init_state(1, x.device, x.dtype)
        outs, max_y = [], 0.0
        for t in range(L):
            y, st = m.step(x[:, t:t + 1], st)
            outs.append(y)
            max_y = max(max_y, float(st.y1.abs().max()))
        streamed = torch.cat(outs, dim=1)
    return full, streamed, x, max_y


def test_bound_does_not_underestimate_single_basis() -> None:
    rho, theta = 0.9999, math.pi / 3.0
    m = SpectralGlobalConv(dim=16, num_bases=1, local_kernel=4,
                           content_gate=False, evidence_gate=False,
                           uncertainty_gate=False).eval()
    _set_grid(m, [rho], [theta])
    L = 2048
    full, streamed, x, max_y = _drift(m, L)
    err = float((full - streamed).abs().max())
    bound = single_basis_error_bound(
        rho, theta, L, state_scale=max_y, input_scale=float(x.abs().max()))
    assert bound.bound > 0
    # 界必须覆盖实测（允许 20% 的 FFT 侧误差余量）。
    assert bound.bound >= 0.8 * err, (bound.bound, err)


def test_bound_is_monotone_in_length() -> None:
    prev = 0.0
    for L in (16, 64, 256, 1024, 4096):
        b = single_basis_error_bound(0.9999, math.pi / 3, L).bound
        assert b >= prev
        prev = b


def test_resync_period_respects_budget() -> None:
    rho, theta = 0.9999, math.pi / 3
    budget = 1e-5
    n = resync_period(budget, rho, theta)
    assert n >= 1
    delta = single_basis_error_bound(rho, theta, 1).max_step_error
    # 与 single_basis_error_bound 同一口径：增益含 1/|sinθ|（误差冲激
    # 响应 = ψ/sinθ，注入系数 1）。
    sin_theta = abs(math.sin(theta))
    assert delta * impulse_l1(rho, theta, n) / sin_theta <= budget
    assert delta * impulse_l1(rho, theta, n + 1) / sin_theta > budget


def test_bound_covers_small_theta_basis() -> None:
    """小 θ 基（logfreq 网格的慢基）上界必须覆盖实测——修正前的 gain
    （Σ|ψ|，缺 1/|sinθ|）在该情形可低估千倍以上（命题 5 v1.1.2 的
    同源缺口）。"""
    rho, theta = 0.9999, math.pi / 8192.0
    m = SpectralGlobalConv(dim=16, num_bases=1, local_kernel=4,
                           content_gate=False, evidence_gate=False,
                           uncertainty_gate=False).eval()
    _set_grid(m, [rho], [theta])
    L = 1024
    full, streamed, x, max_y = _drift(m, L)
    err = float((full - streamed).abs().max())
    bound = single_basis_error_bound(
        rho, theta, L, state_scale=max_y, input_scale=float(x.abs().max()))
    # 数学关系：修正增益 = 旧 Σ|ψ| 增益 / |sinθ|，且该因子在本参数下 > 100
    factor = 1.0 / abs(math.sin(theta))
    assert factor > 100.0
    # 界必须覆盖实测（允许 20% 的 FFT 侧误差余量）
    assert bound.bound >= 0.8 * err, (bound.bound, err)


def test_resync_period_zero_when_budget_impossible() -> None:
    assert resync_period(1e-12, 0.99999, math.pi / 3) == 0


def test_error_aware_grid_beats_logfreq_by_2x_at_2k() -> None:
    """L=2048 下，误差感知网格相对 logfreq 至少 2×（实测约 2.2×）。"""
    dim, L = 32, 2048
    errs = {}
    for name in ("logfreq", "error_aware"):
        rho, theta = BASIS_GRIDS[name](64, 8192)
        m = SpectralGlobalConv(dim=dim, num_bases=64, local_kernel=4,
                               content_gate=True, evidence_gate=False,
                               uncertainty_gate=False).eval()
        _set_grid(m, rho, theta)
        full, streamed, _, _ = _drift(m, L, dim=dim)
        errs[name] = float((full - streamed).abs().max())
    assert errs["error_aware"] * 2.0 <= errs["logfreq"], errs


def test_error_aware_keeps_longest_memory() -> None:
    rho, _ = error_aware_grid(64, 8192)
    assert max(rho) >= 0.9998          # 最长基仍覆盖 ~8192 token
    assert len(rho) == 64


def test_composite_bound_scales_with_coef_magnitude() -> None:
    rho, theta = BASIS_GRIDS["logfreq"](8, 8192)
    coef = torch.full((8, 4), 0.02)
    b1 = composite_error_bound(rho, theta, coef, 1024)
    b2 = composite_error_bound(rho, theta, coef * 2, 1024)
    assert b2 == pytest.approx(2 * b1, rel=1e-9)