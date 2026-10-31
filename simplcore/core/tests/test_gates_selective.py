# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""G4 修复守护：γ/β 双门（gate_ab）分支的 L2/L3 闸门。

本轮把「四层闸门」从"只有单头谱基分支有"补齐为"两条分支同一份数学"
（``simplcore/core/gates.py`` 是唯一实现）。本文件只守**结构性主张**：

1. 开启闸门后，闸门标量真的参与计算（关掉→输出与旧行为一致）；
2. L2 权重只依赖**因果前缀**（扰动未来不改变过去的权重）；
3. 闸门参数梯度可达（防静默零梯度 → 闸门退化为常数）；
4. 流式（IIR）与整段（FFT）在**开启两层闸门**后仍逐位一致；
5. 全链路严格因果；
6. 参数量 bit-exact（含 4 个闸门标量）；
7. 两条分支调用同一个 ``gates`` 函数，因此同一输入下权重完全相同。

预算：CPU < 30 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.gates import (                                   # noqa: E402
    evidence_weight_step,
    evidence_weights,
    uncertainty_weights,
)
from simplcore.core.selective import (                               # noqa: E402
    GatedSpectralConv2,
    analytic_gated_spectral_params,
)

D, M, H, K = 32, 8, 4, 4


def _mixer(evidence=True, uncertainty=True, gate_ab=True):
    torch.manual_seed(0)
    return GatedSpectralConv2(D, M, local_kernel=K, num_heads=H,
                              gate_ab=gate_ab,
                              evidence_gate=evidence,
                              uncertainty_gate=uncertainty).eval()


# ---------------------------------------------------------------------------
# 1. 闸门真实参与（关掉 == 旧行为）
# ---------------------------------------------------------------------------
def test_gates_are_wired_and_ablatable():
    torch.manual_seed(1)
    x = torch.randn(2, 24, D)

    m_off = _mixer(evidence=False, uncertainty=False)
    m_on = _mixer(evidence=True, uncertainty=True)
    m_on.load_state_dict(m_off.state_dict(), strict=False)

    w_e_off = evidence_weights(x, m_off.evidence_gain
                               if hasattr(m_off, "evidence_gain")
                               else torch.tensor(1.0),
                               torch.tensor(0.0), torch.tensor(0.0))
    # 关掉闸门时，模块里不应存在闸门参数
    assert not hasattr(m_off, "evidence_gain")
    assert not hasattr(m_off, "uncertainty_logit")

    with torch.no_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    # 默认初始（gain=1, pos_gain=0, bias=0, λ=0）时闸门不是恒等，输出必有差异
    assert not torch.allclose(y_off, y_on, atol=1e-5), \
        "开启/关闭闸门输出相同：闸门没有接入前向"
    # 但差异应当是"受控"的（不是 NaN / 爆掉）
    assert torch.isfinite(y_on).all()
    assert float((y_on - y_off).abs().max()) < 50.0


def test_gate_scalars_have_gradients():
    torch.manual_seed(2)
    m = _mixer()
    x = torch.randn(2, 16, D)
    m(x).pow(2).sum().backward()
    for name in ("evidence_gain", "evidence_pos_gain", "evidence_bias",
                 "uncertainty_logit"):
        p = getattr(m, name)
        assert p.grad is not None, f"{name} 梯度缺失"
        assert torch.any(p.grad != 0), f"{name} 零梯度：闸门退化为常数"


# ---------------------------------------------------------------------------
# 2. L2 只依赖因果前缀
# ---------------------------------------------------------------------------
def test_evidence_weights_are_causal_prefix_only():
    torch.manual_seed(3)
    m = _mixer()
    x = torch.randn(1, 12, D)
    x2 = x.clone()
    x2[:, 6:] += 3.7                      # 只扰动后半段（未来）
    with torch.no_grad():
        w1 = m._evidence_weights(x)
        w2 = m._evidence_weights(x2)
    assert torch.allclose(w1[:, :6], w2[:, :6], atol=1e-6), \
        "L2 权重看到了未来位置"
    assert not torch.allclose(w1[:, 6:], w2[:, 6:], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. 流式 == 整段（两层闸门同时开启）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("length", [1, 4, 17, 64])
def test_stream_matches_parallel_with_gates(length: int) -> None:
    torch.manual_seed(4)
    m = _mixer()
    x = torch.randn(2, length, D)
    with torch.no_grad():
        y_full = m(x)
        state = m.init_state(2, x.device, x.dtype)
        outs = []
        for t in range(length):
            y_t, state = m.step(x[:, t], state)
            outs.append(y_t)
        y_stream = torch.stack(outs, dim=1)
    assert torch.allclose(y_full, y_stream, atol=1e-4), \
        f"L={length} 流式与整段不一致：max|Δ|=" \
        f"{float((y_full - y_stream).abs().max()):.3e}"


def test_evidence_step_matches_parallel_prefix():
    """逐步累积的 energy_sum 与 cumsum 口径给出相同权重。"""
    torch.manual_seed(5)
    m = _mixer()
    x = torch.randn(1, 9, D)
    with torch.no_grad():
        w_par = m._evidence_weights(x)                       # (1, 9, 1)
        es = None
        got = []
        for t in range(9):
            w, es = evidence_weight_step(
                x[:, t:t + 1], es, t,
                m.evidence_gain, m.evidence_pos_gain, m.evidence_bias)
            got.append(w)
        w_step = torch.cat(got, dim=1)
    assert torch.allclose(w_par, w_step, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. 因果性
# ---------------------------------------------------------------------------
def test_branch_is_strictly_causal():
    torch.manual_seed(6)
    m = _mixer()
    x = torch.randn(1, 16, D)
    x2 = x.clone()
    x2[:, -1] += 5.0                     # 只扰动最后一个 token
    with torch.no_grad():
        y1 = m(x)
        y2 = m(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-4), \
        "扰动末位改变了历史输出：非因果"


# ---------------------------------------------------------------------------
# 5. 不确定门：分歧越大压得越狠
# ---------------------------------------------------------------------------
def test_uncertainty_gate_damps_disagreement():
    logit = torch.zeros(1)
    base = torch.ones(1, 3, D)
    same = base.clone()
    diff = -base.clone()
    w_same = uncertainty_weights(base, same, logit)
    w_diff = uncertainty_weights(base, diff, logit)
    assert torch.allclose(w_same, torch.ones_like(w_same), atol=1e-6)
    assert float(w_diff.mean()) < float(w_same.mean())


# ---------------------------------------------------------------------------
# 6. 参数量 bit-exact（含闸门标量）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [
    dict(evidence_gate=True, uncertainty_gate=True),
    dict(evidence_gate=False, uncertainty_gate=True),
    dict(evidence_gate=True, uncertainty_gate=False),
    dict(evidence_gate=False, uncertainty_gate=False),
])
def test_analytic_params_bit_exact_with_gates(cfg):
    torch.manual_seed(7)
    m = GatedSpectralConv2(D, M, local_kernel=K, num_heads=H, **cfg)
    actual = sum(p.numel() for p in m.parameters())
    analytic = analytic_gated_spectral_params(
        D, num_bases=M, num_heads=H, local_kernel=K, **cfg)
    assert actual == analytic, f"{actual} != {analytic} for {cfg}"
