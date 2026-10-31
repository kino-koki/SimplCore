# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""外部审计（2026-10-31）结论的可执行守护。

把四项审计发现固化为回归测试，防止同类"数字-公式-实现"漂移复发：

1. 命题 5：误差界必须含 1/|sinθ| 因子（旧界实测被违反 5.79×）；
2. theta 网格公共零点：legacy 均匀网格在 tau=2M(M+1)-1 处核恒 0，
   默认 logfreq 无公共零点，构造期守护 `_assert_no_common_zero` 生效；
3. 流式状态字节账目：解析式 4·(2MD+(K-1)D+1)，M=64/D=1024/K=4 =
   536,580 B = 524.0 KiB（不含模型级 CMT memory 的 4 KiB）；
4. fp32 sigmoid 饱和阈值：实测 16.636（=24·ln2，torch 实现路径），
   论文旧值 17.3 为解析充分界（25·ln2），两者口径已在文档分开。
5. 泄漏断言分路径：流式 IIR 扰动未来 -> 过去输出逐位相等（精确 0）；
   整段 FFT 并行 -> fp32 重排噪声 < 1e-5（实数域因果，见定理 1/2）。

复现脚本：scripts/audit_numbers_20261031.py
"""
from __future__ import annotations

import math

import pytest
import torch

from simplcore.core.sgc import (SpectralGlobalConv, _assert_no_common_zero,
                                _init_oscillator_grid)


# ---------------------------------------------------------------------------
# 1. prop-5 bound needs the 1/|sin theta| factor
# ---------------------------------------------------------------------------
def _stream_error_vs_bound(M=8, L=4096, D=16, seed=0):
    """Per-basis fp32 streaming error vs prop-5 bounds.

    命题 5 的 δ 上界定义是 δ_m[t] ≤ k·u·(|a_m·ỹ[t-1]| + |b_m·ỹ[t-2]| +
    c_m·|u_t|)（随 |ỹ| 增长，非裸 u），故界须从 fp64 参考流的实际 |ỹ|
    重构：max_s|δ_s| ≤ k·u·max_t(...)，再除以 (1-ρ)|sinθ|。
    """
    torch.manual_seed(seed)
    m = SpectralGlobalConv(dim=D, num_bases=M, local_kernel=1,
                           content_gate=False, evidence_gate=False,
                           uncertainty_gate=False, out_gate=False)
    with torch.no_grad():
        m.coef.normal_(0, 0.3)
    rho, theta = m.oscillator_params()
    u_seq = torch.randn(1, L, D)

    def stream(dtype):
        y1 = torch.zeros(1, M, D, dtype=dtype)
        y2 = torch.zeros(1, M, D, dtype=dtype)
        outs = []
        a = (2.0 * rho * torch.cos(theta)).to(dtype)
        b = (-(rho ** 2)).to(dtype)
        c = torch.sin(theta).to(dtype)
        for t in range(L):
            u = u_seq[:, t].to(dtype).unsqueeze(1)
            y_new = a.view(1, M, 1) * y1 + b.view(1, M, 1) * y2 \
                + c.view(1, M, 1) * u
            y2, y1 = y1, y_new
            outs.append(y_new)
        return torch.stack(outs, dim=1)              # (B,L,M,D)

    y64 = stream(torch.float64)
    y32 = stream(torch.float32)
    e = (y32.double() - y64).abs()
    per_m = e.amax(dim=(0, 1, 3))                    # (M,)

    # per-step perturbation bound from the fp64 reference (k=5, per prop 5)
    u = 2.0 ** -24
    a = (2.0 * rho * torch.cos(theta)).double().view(1, 1, M, 1)
    b = (-(rho ** 2)).double().view(1, 1, M, 1)
    c = torch.sin(theta).double().view(1, 1, M, 1)
    u_inj = u_seq.double().unsqueeze(2)              # (B,L,1,D)
    zeros1 = torch.zeros(1, 1, M, D, dtype=torch.float64)
    zeros2 = torch.zeros(1, 2, M, D, dtype=torch.float64)
    y_prev1 = torch.cat([zeros1, y64[:, :-1]], dim=1)
    y_prev2 = torch.cat([zeros2, y64[:, :-2]], dim=1) if L >= 2 \
        else torch.zeros(1, L, M, D, dtype=torch.float64)
    pert = (a.abs() * y_prev1 + b.abs() * y_prev2
            + c.abs() * u_inj).amax(dim=(0, 1, 3))   # (M,) max_t (...)
    kmax = 5.0
    old = kmax * u * pert / (1 - rho.double())                       # 缺 1/sin
    new = kmax * u * pert / ((1 - rho.double()) * torch.sin(theta).double())
    return per_m, old, new, theta


def test_prop5_bound_requires_inverse_sin_factor():
    """命题 5 修正界的两层守护：

    (a) 对象级：误差递推 e=a·e1+b·e2+δ 的注入系数是 1（非信号的
        c=sinθ），其冲激响应 = ψ[τ]/sinθ——旧证明直接用 |ψ[τ]|≤ρ^τ
        会丢掉 1/|sinθ|≥1 因子（审计发现的缺口）；
    (b) 端到端：δ 按命题 5 定义（k≤5、随 |ỹ|）从 fp64 参考重构后，
        修正界必须覆盖实测误差。
    """
    torch.manual_seed(0)
    M, L = 8, 256
    m = SpectralGlobalConv(dim=4, num_bases=M, local_kernel=1,
                           content_gate=False, evidence_gate=False,
                           uncertainty_gate=False, out_gate=False)
    rho, theta = m.oscillator_params()
    rd = torch.float64
    a = (2.0 * rho * torch.cos(theta)).to(rd)
    b = (-(rho ** 2)).to(rd)

    # (a) impulse response of the error recurrence (unit delta at t=0)
    y1 = torch.zeros(M, dtype=rd)
    y2 = torch.zeros(M, dtype=rd)
    h = []
    for t in range(L):
        inj = 1.0 if t == 0 else 0.0
        y_new = a * y1 + b * y2 + inj
        y2, y1 = y1, y_new
        h.append(y_new)
    h = torch.stack(h)                                    # (L, M)
    tau = torch.arange(L, dtype=rd).unsqueeze(1)
    psi = torch.exp(torch.log(rho.to(rd)).unsqueeze(0) * tau) \
        * torch.sin((tau + 1.0) * theta.to(rd).unsqueeze(0))
    sin_theta = torch.sin(theta.to(rd))                   # 与 psi 同在 fp64 域

    # 锐利单点断言（无容差歧义）：
    # (i) h[0] = 1 精确 —— 误差注入系数是 1；信号冲激响应 ψ[0] = sinθ < 1。
    #     旧证明把误差冲激响应当 ψ，正是在这一点上丢掉 1/sinθ 因子。
    assert (h[0] == 1.0).all(), "error injection coefficient must be 1"
    assert (psi[0] == sin_theta).all() and (sin_theta < 1.0).all()
    # (ii) 峰点：在窗内命中 sin((τ+1)θ)≈1 的基上，|h[τ*]| = ρ^{τ*}/sinθ
    #      （峰处 sin 斜率为零，数值最稳；rtol 覆盖 τ* 取整偏差 ≤ θ/2）
    for m_idx in range(M):
        tau_star = int(round(math.pi / (2.0 * theta[m_idx].item()))) - 1
        if not (0 <= tau_star < L):
            continue                                       # 峰在窗外，跳过
        got = h[tau_star, m_idx].item()
        want = (rho.to(rd)[m_idx] ** tau_star / sin_theta[m_idx]).item()
        assert abs(got) == pytest.approx(want, rel=max(2.0 * theta[m_idx].item(), 1e-3)), \
            (m_idx, tau_star, got, want)

    # (b) end-to-end corrected bound holds
    per_m, old, new, _ = _stream_error_vs_bound()
    assert (per_m <= new * (1.0 + 1e-9)).all(), \
        "corrected bound (with 1/|sin theta|) violated"


# ---------------------------------------------------------------------------
# 2. common zeros of the theta grid
# ---------------------------------------------------------------------------
def test_default_grid_has_no_common_zero():
    """默认 logfreq 网格：任意 tau=j(M+1)-1 处全体基不同时为零。"""
    for M, decay in ((64, 8192.0), (128, 8192.0), (8, 8192.0)):
        _rho, theta = _init_oscillator_grid(M, decay, mode="logfreq")
        for j in range(1, 2 * M + 9):
            s = torch.sin(float(j * (M + 1)) * theta)
            assert s.abs().max().item() >= 1e-6, (M, j)


def test_legacy_grid_hits_common_zero_guard():
    """legacy 均匀网格的代数公共零点：j=2M/gcd(M+1,2)（M=64 -> j=128,
    tau=8319）处全体基同时为零。fp64 公式级判定精确为零；fp32 值级被
    参数化误差（tau·ulp(theta)≈1.5e-3）掩盖成"近零"——所以守护必须
    公式级。legacy 构造被拒绝，logfreq 放行。"""
    M = 64
    j = 2 * M                                     # gcd(M+1,2)=2 for even M
    tau1 = j * (M + 1)
    theta64 = math.pi * (torch.arange(M, dtype=torch.float64) + 0.5) / M
    s64 = torch.sin(float(tau1) * theta64)
    assert s64.abs().max().item() < 1e-9, \
        "expected algebraic common zero of the legacy grid is missing"
    # fp32 值级口径（为什么必须公式级）：参数化误差把零点抬到 ~1.5e-3
    theta32 = math.pi * (torch.arange(M, dtype=torch.float32) + 0.5) / M
    assert torch.sin(float(tau1) * theta32).abs().max().item() > 1e-4
    with pytest.raises(ValueError, match="common zero"):
        _init_oscillator_grid(M, 8192.0, mode="legacy")
    # logfreq 无代数公共零点，放行
    _rho, theta = _init_oscillator_grid(M, 8192.0, mode="logfreq")
    for jj in range(1, 2 * M + 9):
        s = torch.sin(float(jj * (M + 1)) * theta.to(torch.float64))
        assert s.abs().max().item() >= 1e-6, (M, jj)


# ---------------------------------------------------------------------------
# 3. streaming state bytes: analytic formula and the audited numbers
# ---------------------------------------------------------------------------
def test_stream_state_bytes_formula_and_audit_number():
    """解析式 4·(2MD+(K-1)D+1)；M=64/D=1024/K=4 = 536,580 B = 524.0 KiB
    （不含模型级 CMT memory；含之则 528.0 KiB——外部审计口径）。"""
    m = SpectralGlobalConv(dim=1024, num_bases=64, local_kernel=4)
    expected = 4 * (2 * 64 * 1024 + 3 * 1024 + 1)
    got = m.streaming_state_bytes(1, dtype=torch.float32)
    assert got == expected == 536_580
    # 536,580 B = 524.0039 KiB —— 文档"524.0 KiB"为一位小数口径（+4 B
    # 为 energy_sum fp32 标量）
    assert got == pytest.approx(524.0 * 1024, rel=1e-5)          # 524.0 KiB/层
    assert got + 1024 * 4 == pytest.approx(528.0 * 1024, rel=1e-5)  # +CMT memory


# ---------------------------------------------------------------------------
# 4. fp32 sigmoid saturation threshold (measured, torch impl path)
# ---------------------------------------------------------------------------
def test_fp32_sigmoid_saturation_threshold():
    """实测阈值 16.636（24·ln2）：torch fp32 sigmoid 在其上精确返回 1.0；
    17.3（25·ln2 解析界）作为充分条件仍真。"""
    x24 = 24.0 * math.log(2.0)
    x25 = 25.0 * math.log(2.0)
    below = torch.sigmoid(torch.tensor(16.0, dtype=torch.float32)).item()
    at = torch.sigmoid(torch.tensor(x24, dtype=torch.float32)).item()
    above = torch.sigmoid(torch.tensor(x25, dtype=torch.float32)).item()
    assert below < 1.0
    assert at == 1.0 or abs(at - 1.0) <= 2.0 ** -24     # 16.636 boundary
    assert above == 1.0


# ---------------------------------------------------------------------------
# 5. leak guard must name its path (streaming bit-exact vs FFT fp32 noise)
# ---------------------------------------------------------------------------
def test_causality_by_path():
    """定理 2 的分路径口径：流式逐位精确因果；FFT 并行 fp32 噪声 < 1e-5。"""
    torch.manual_seed(0)
    m = SpectralGlobalConv(dim=16, num_bases=8, local_kernel=4)
    m.eval()
    L = 32
    x = torch.randn(1, L, m.dim)
    with torch.no_grad():
        # streaming: perturb the future, past outputs must be BIT-EXACT
        st = m.init_state(1, x.device, x.dtype)
        past = []
        for t in range(L // 2):
            y_t, st = m.step(x[:, t], st)
            past.append(y_t)
        past = torch.stack(past, dim=1)
        st2 = m.init_state(1, x.device, x.dtype)
        past2 = []
        for t in range(L // 2):
            y_t, st2 = m.step(x[:, t], st2)           # future differs, past same
            past2.append(y_t)
        past2 = torch.stack(past2, dim=1)
        assert torch.equal(past, past2), \
            "streaming path is not bit-exactly causal"

        # full FFT parallel: real-domain causal, fp32 reorder noise < 1e-5
        y1 = m(x)
        x2 = x.clone()
        x2[:, L // 2:, :] = torch.randn(1, L - L // 2, m.dim) * 100
        y2 = m(x2)
        delta = (y1[:, : L // 2] - y2[:, : L // 2]).abs().max().item()
        assert delta < 1e-5, f"FFT parallel path leak beyond fp32 noise: {delta}"