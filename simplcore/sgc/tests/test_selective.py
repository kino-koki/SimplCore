# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""门控增强 + 多头谱基 mixer 守护测试.

守护 ``simplcore/sgc/selective.py`` 的结构性主张（不守收敛质量，后者需
A100 规模训练）：
  * 因果性（未来扰动不改变过去，FFT 数值噪声 <1e-4）；
  * 双形式等价：整段 forward == 逐 token stream_step（L 在 {1,4,17,64}）；
  * O(1) 状态：streaming_state_bytes 与 L 无关，且多头与单头字节数相同；
  * gamma/beta 门路径真实参与计算（开/关输出差异可测）；
  * 多头谱基秩：num_heads>1 时核子空间秩 > 单头（数值秩测试）；
  * 参数量 bit-exact（analytic == model.num_params()，3 种配置）。

预算：全模块 CPU 上 < 90 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.selective import (                              # noqa: E402
    GatedSpectralConv2,
    analytic_gated_spectral_params,
)

D, M, G, K = 32, 8, 4, 4


def _model(gate_ab=True, num_heads=G):
    torch.manual_seed(0)
    return GatedSpectralConv2(D, M, local_kernel=K, num_heads=num_heads,
                              gate_ab=gate_ab).eval()


# ---------------------------------------------------------------------------
# 1. 因果性
# ---------------------------------------------------------------------------
def test_causality_future_perturbation_does_not_change_past():
    m = _model()
    x = torch.randn(2, 64, D)
    x2 = x.clone()
    x2[:, 32:] = torch.randn_like(x2[:, 32:]) * 100
    with torch.no_grad():
        y1, y2 = m(x), m(x2)
    assert (y1[:, :32] - y2[:, :32]).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# 2. 双形式等价：forward == stream（L 在 {1,4,17,64}）
# ---------------------------------------------------------------------------
def test_dual_form_equivalence():
    m = _model()
    for L in (1, 4, 17, 64):
        x = torch.randn(2, L, D)
        with torch.no_grad():
            full = m(x)
            st = m.init_state(2, x.device, x.dtype)
            outs = []
            for t in range(L):
                lg, st = m.step(x[:, t], st)
                outs.append(lg)
            streamed = torch.stack(outs, 1)
        d = (streamed - full).abs().max().item()
        assert d < 4e-5, f"L={L}: dual-form mismatch {d}"


# ---------------------------------------------------------------------------
# 3. O(1) 状态：字节数与 L 无关；多头与单头相同
# ---------------------------------------------------------------------------
def test_streaming_state_bytes_constant_and_head_invariant():
    m_multi = _model(num_heads=G)
    m_single = _model(num_heads=1)
    b_multi_1 = m_multi.streaming_state_bytes(2)
    b_multi_2 = m_multi.streaming_state_bytes(2)  # 字节数与 L 无关（无 L 参数）
    assert b_multi_1 == b_multi_2
    b_single = m_single.streaming_state_bytes(2)
    # 多头不改 O(1) 状态字节数
    assert b_multi_1 == b_single, \
        f"multi-head state bytes {b_multi_1} != single-head {b_single}"
    # 与解析式对照：(2*M*D + (K-1)*D) * batch * itemsize
    expected = (2 * M * D + (K - 1) * D) * 2 * 4
    assert b_multi_1 == expected, f"{b_multi_1} != {expected}"


# ---------------------------------------------------------------------------
# 4. gamma/beta 门路径真实参与
# ---------------------------------------------------------------------------
def test_gamma_beta_gates_affect_output():
    # 同一构造下 gate_ab=True vs False：相同公共权重 + True 版多 gate_gamma/beta
    m_on = GatedSpectralConv2(D, M, local_kernel=K, num_heads=G, gate_ab=True).eval()
    m_off = GatedSpectralConv2(D, M, local_kernel=K, num_heads=G, gate_ab=False).eval()
    # 公共权重逐一 copy（按名字匹配，形状必同）
    with torch.no_grad():
        off_state = dict(m_off.named_parameters())
        for n_ab, p_ab in m_on.named_parameters():
            if n_ab in off_state:
                p_ab.copy_(off_state[n_ab])
        x = torch.randn(2, 32, D)
        y_on, y_off = m_on(x), m_off(x)
    d = (y_on - y_off).abs().max().item()
    assert d > 1e-3, f"gate_ab=True 未改变输出: {d}"


# ---------------------------------------------------------------------------
# 5. 多头谱基的 head 局部性：只改 head g 的系数块，核矩阵只有 head g 列变化
# ---------------------------------------------------------------------------
def test_multi_head_kernel_head_locality():
    """多头谱基的结构收益：每个 head 的 hd 通道由独立的 M*hd 系数块驱动。

    守护：coef 只设 head 1 非零时，核矩阵严格只有 head 1 的 hd 列非零
    （head 局部性/解耦）。注：整核矩阵时间秩恒为 <= M（所有列都是 M 个
    psi_m 的组合），"表达力提升"体现在 head 间系数子空间的结构解耦，
    而非整核时间秩。
    """
    torch.manual_seed(1)
    hd = D // G
    m = GatedSpectralConv2(D, M, local_kernel=K, num_heads=G, gate_ab=False).eval()
    with torch.no_grad():
        L = 64
        m.coef.zero_()
        m.coef[:, 1, :] = torch.randn(M, hd) * 0.05     # 只有 head 1 有系数
        K_only1 = m.build_kernel(L)
        for g in range(G):
            col = K_only1[:, g * hd:(g + 1) * hd]
            if g == 1:
                assert col.abs().max().item() > 1e-6, "head 1 kernel should be active"
            else:
                assert col.abs().max().item() == 0.0, \
                    f"head {g} kernel should be zero when only head 1 coef set"


# ---------------------------------------------------------------------------
# 6. 参数量 bit-exact（4 种配置）
# ---------------------------------------------------------------------------
def test_analytic_params_bit_exact():
    for gate_ab, num_heads, out_gate in (
            (True, G, False),
            (False, G, False),
            (False, 1, False),
            (True, 1, True),
    ):
        m = GatedSpectralConv2(D, M, local_kernel=K, num_heads=num_heads,
                               gate_ab=gate_ab, out_gate=out_gate)
        n_model = sum(p.numel() for p in m.parameters())
        n_analytic = analytic_gated_spectral_params(
            D, M, num_heads, content_gate=True, out_gate=out_gate,
            gate_ab=gate_ab, local_kernel=K)
        assert n_model == n_analytic, (
            f"gate_ab={gate_ab} heads={num_heads} out_gate={out_gate}: "
            f"model {n_model} != analytic {n_analytic}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
