"""频率网格修复的守护测试：`init_mode="linear"`。

背景（详见 docs/frequency_grid_fix.md）
--------------------------------------
默认 ``logfreq`` 把 θ_min 设计在 π/decay_max（decay=8192 时 3.8e-4），
低频端相邻基在有限 τ 窗口内近似共线 → 字典表征容量塌缩。仓库原生守护
口径 ``_grid_rank_condition``（Ψ=sin((τ+1)θ), L=512, SVD）实测：

    M=32  logfreq    数值秩 24/32   cond 1.92e16
    M=32  hybrid     数值秩 32/32   cond 6.52
    M=32  linear     数值秩 32/32   cond 1.04

修复是**零成本**的：只改初始化网格，不增参数、不增 FLOPs、不改 O(1) 状态。
本文件守护三件事：网格质量、零成本、契约不破。

★ 边界（不在本文件宣称被修好）
  正交网格只在周期/连续信号轴兑现收益；key→value 绑定轴（MQAR 类）
  无收益甚至回退。共享叠加状态不可分离是独立的结构限制。
"""
import math

import pytest
import torch

from simplcore.core.arch.sgc import (
    SpectralGlobalConv, _init_oscillator_grid, _grid_rank_condition,
    theta_clamp_floor, THETA_CLAMP_HI,
)


def _mk(mode, **kw):
    kw.setdefault("dim", 64)
    kw.setdefault("num_bases", 32)
    kw.setdefault("decay_max", 8192.0)
    kw.setdefault("local_kernel", 4)
    kw.setdefault("evidence_gate", False)
    kw.setdefault("uncertainty_gate", False)
    kw.setdefault("init_mode", mode)
    return SpectralGlobalConv(**kw)


# ── 1) 网格质量 ────────────────────────────────────────────────
def test_linear_grid_is_full_rank_and_well_conditioned():
    """★ 核心：linear 满秩且良条件，logfreq 秩缺失。"""
    _, th_log = _init_oscillator_grid(32, 8192.0, "logfreq")
    _, th_lin = _init_oscillator_grid(32, 8192.0, "linear", theta_floor=0.3)
    r_log, c_log = _grid_rank_condition(th_log, 512)
    r_lin, c_lin = _grid_rank_condition(th_lin, 512)

    assert r_log < 32, f"logfreq 应秩缺失（复现基准），实测 {r_log}/32"
    assert r_lin == 32, f"linear 应满秩，实测 {r_lin}/32"
    assert c_lin < 10.0, f"linear cond 应接近 1，实测 {c_lin:.3f}"
    assert c_log / c_lin > 1e10, (
        f"改善应 >10 个数量级: {c_log:.3e} -> {c_lin:.3e}")


def test_linear_grid_lifts_theta_min():
    """θ_min 从 π/decay_max 抬到 floor，且上界放开到 π−floor。"""
    lo = math.pi / 8192.0
    _, th_log = _init_oscillator_grid(32, 8192.0, "logfreq")
    _, th_lin = _init_oscillator_grid(32, 8192.0, "linear", 0.3)
    assert float(th_log.min()) < 1e-3
    assert float(th_lin.min()) > 0.3
    assert float(th_lin.max()) > 2.7          # > pi/2，覆盖整个 (0,pi)
    assert float(th_lin.min()) > lo * 100


def test_linear_theta_is_within_clamp_bounds():
    """linear 网格的 θ 全落在 oscillator_params 的合法钳制区间内。"""
    _, th = _init_oscillator_grid(32, 8192.0, "linear", 0.3)
    floor = theta_clamp_floor(8192.0)
    assert float(th.min()) >= floor
    assert float(th.max()) <= THETA_CLAMP_HI


def test_linear_floor_validation():
    import math as _m
    with pytest.raises(ValueError, match="linear_floor"):
        _mk("linear", linear_floor=0.0)
    with pytest.raises(ValueError, match="linear_floor"):
        _mk("linear", linear_floor=_m.pi / 2)
    with pytest.raises(ValueError, match="unknown init_mode"):
        _mk("not_a_mode")


# ── 2) 零成本 ──────────────────────────────────────────────────
def test_linear_costs_nothing():
    """★ 真实收益的前提：参数量、状态字节、输出形状全部不变。"""
    a, b = _mk("logfreq"), _mk("linear")
    assert sum(p.numel() for p in a.parameters()) == \
           sum(p.numel() for p in b.parameters()), "参数量不应增加"
    for batch in (1, 8):
        assert a.streaming_state_bytes(batch) == b.streaming_state_bytes(batch), \
            "O(1) 状态字节不应改变"
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        assert a(x).shape == b(x).shape


def test_default_mode_unchanged():
    """历史默认仍是 logfreq（既有 checkpoint / 实验可复现）。"""
    m = _mk("logfreq")
    assert m.init_mode == "logfreq"
    assert m.linear_floor == 0.3       # 参数存在但 logfreq 走原分支
    _, th = _init_oscillator_grid(32, 8192.0, "logfreq")
    _, th_default = _init_oscillator_grid(32, 8192.0)
    assert torch.allclose(th, th_default), "默认分支必须逐位不变"


# ── 3) 契约不破 ────────────────────────────────────────────────
def test_linear_parallel_matches_streaming():
    """FFT/IIR 对偶性在新网格下保持。"""
    m = _mk("linear").eval()
    B, L, D = 2, 12, 64
    x = torch.randn(B, L, D)
    with torch.no_grad():
        y_par = m(x)
        st = m.init_state(B, x.device, x.dtype)
        outs = []
        for t in range(L):
            y_t, st = m.step(x[:, t:t + 1], st)
            outs.append(y_t if y_t.dim() == 3 else y_t.unsqueeze(1))
        y_str = torch.cat(outs, 1)
    diff = (y_par - y_str).abs().max().item()
    assert diff / max(y_par.abs().max().item(), 1e-12) < 1e-4, (
        f"新网格下并行/流式不一致: {diff:.2e}")


def test_linear_chunk_matches_full():
    """跨块一致性（含自由响应）在新网格下保持。"""
    m = _mk("linear").eval()
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    with torch.no_grad():
        y_full = m(x)
        st = m.init_state(B, x.device, x.dtype)
        y1, st = m.forward_chunk(x[:, :7], st)
        y2, st = m.forward_chunk(x[:, 7:], st)
        y_chunk = torch.cat([y1, y2], 1)
    diff = (y_full - y_chunk).abs().max().item()
    assert diff / max(y_full.abs().max().item(), 1e-12) < 2e-3


def test_linear_build_kernel_is_finite_at_long_length():
    """θ_min 抬高后，长 L 的核不再有 1/|sinθ| 病态放大。"""
    m = _mk("linear").eval()
    with torch.no_grad():
        k = m.build_kernel(4096)
    assert torch.isfinite(k).all()
    assert float(k.abs().max()) < 1e4


def test_linear_is_orthogonal_even_with_rho_decay():
    """含 ρ^τ 的真实字典口径下，linear 仍显著优于 logfreq。

    这是比仓库守护口径更接近实际前向的度量：
      Ψ[τ,m] = ρ_m^τ · sin((τ+1)θ_m)，列归一化后取 Gram 条件数。
    """
    def cond(mode):
        rho, th = _init_oscillator_grid(32, 8192.0, mode, 0.3)
        tau = torch.arange(1024, dtype=torch.float64)[:, None]
        P = (rho.double()[None, :] ** tau) * torch.sin(
            (tau + 1.0) * th.double()[None, :])
        Pn = P / P.norm(dim=0, keepdim=True).clamp_min(1e-300)
        ev = torch.linalg.eigvalsh(Pn.T @ Pn)
        return float(ev[-1] / ev[0].clamp_min(1e-300))
    c_log, c_lin = cond("logfreq"), cond("linear")
    assert c_lin < c_log / 100, (
        f"含 rho 的真实字典也应显著改善: {c_log:.3e} -> {c_lin:.3e}")
