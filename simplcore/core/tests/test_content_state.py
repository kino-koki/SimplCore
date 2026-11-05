# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""CAS 契约守护：把「少一条」钉成机器可验证的账，防止纯度被悄悄扩大。

锁住：O(1) 状态 / 无 N×N / 无 KV cache / 严格因果 / 双形式；
并**显式断言**「核输入无关」这条已经让出去（不许偷偷改回去冒充纯
SGC），以及 FFT↔IIR 双形式不适用于本模块。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.arch.content_state import ContentAddressedState
from simplcore.core.arch.purity import scan_source_attention_free
from pathlib import Path


def _layer(**kw) -> ContentAddressedState:
    torch.manual_seed(0)
    return ContentAddressedState(dim=16, feat_dim=8, value_dim=16,
                                 zero_init_readout=False, **kw)


def test_o1_state_bytes_independent_of_L() -> None:
    """四长度逐字节同一状态形状——不是「随 L 线性增长但常数小」。"""
    m = _layer()
    b = m.state_bytes(batch=1)
    assert b == (8 * 16 + 8) * 4
    for L in (1, 4, 17, 64, 4096):
        u = torch.randn(1, L, 16)
        st = m.init_state(1, dtype=torch.float32)
        y, st2 = m.step(u[:, -1], st)
        assert st2.S.shape == (1, 8, 16)
        assert st2.z.shape == (1, 8)
        assert m.state_bytes(batch=1) == b
    st = m.init_state(1, dtype=torch.float32)
    assert st.S.numel() + st.z.numel() == 8 * 16 + 8


def test_parallel_matches_streaming_fp64() -> None:
    """prefix-cumsum ↔ 一阶累加：fp64 应精确一致。"""
    torch.manual_seed(1)
    m = _layer().double()
    u = torch.randn(2, 23, 16, dtype=torch.float64)
    with torch.no_grad():
        y_par = m(u)
        st = m.init_state(2, dtype=torch.float64)
        outs = []
        for t in range(u.shape[1]):
            y_t, st = m.step(u[:, t], st)
            outs.append(y_t)
        y_str = torch.stack(outs, dim=1)
    assert torch.max(torch.abs(y_par - y_str)) < 1e-10


def test_parallel_matches_streaming_fp32_within_tolerance() -> None:
    """fp32 是容差内一致，不是逐位（与 SGC 的 FFT 路径同口径）。"""
    torch.manual_seed(2)
    m = _layer()
    u = torch.randn(1, 96, 16)
    with torch.no_grad():
        y_par = m(u)
        st = m.init_state(1, dtype=torch.float32)
        outs = []
        for t in range(u.shape[1]):
            y_t, st = m.step(u[:, t], st)
            outs.append(y_t)
        y_str = torch.stack(outs, dim=1)
    err = float(torch.max(torch.abs(y_par - y_str)))
    assert err < 5e-4, err


def test_strict_causality_future_perturbation() -> None:
    """改未来 token 不改变过去输出（含 prefix cumsum 的并行路径）。"""
    torch.manual_seed(3)
    m = _layer()
    u = torch.randn(1, 12, 16)
    u2 = u.clone(); u2[0, 7:] += 5.0
    with torch.no_grad():
        y1, y2 = m(u), m(u2)
    assert torch.equal(y1[:, :7], y2[:, :7])


def test_contract_spells_out_the_cost() -> None:
    """「少一条」= 核变输入函数；FFT↔IIR 让位给 prefix-sum↔累加。

    这两条必须是显式 False，不许被包装成「纯 SGC 的小改进」。
    """
    c = _layer().contract()
    assert c["state_bytes_L_independent"] is True
    assert c["no_NxN_scores"] is True
    assert c["no_kv_cache"] is True
    assert c["no_position_softmax"] is True
    assert c["strictly_causal"] is True
    assert c["kernel_input_independent"] is False       # 让出去的那一条
    assert c["fft_iir_dual_form"] is False              # 口径变化
    assert c["dual_form"] == "prefix-cumsum <-> first-order accumulator"


def test_low_rank_cuts_state_cost() -> None:
    """feat_dim=r 把状态从 O(D²) 压到 O(r·D)——1.7B 落地的必要条件。"""
    full = ContentAddressedState(dim=64, feat_dim=64, value_dim=64)
    low = ContentAddressedState(dim=64, feat_dim=8, value_dim=64)
    assert low.state_bytes() < full.state_bytes() / 7
    assert low.contract()["state_scalar_count"] == 8 * 64 + 8


def test_kernel_really_depends_on_content() -> None:
    """判据性：同样的位置、不同历史内容 → 不同读出（静态核做不到）。"""
    torch.manual_seed(4)
    m = _layer()
    # 两段前缀长度相同、内容不同，query 相同
    prefix_a = torch.zeros(1, 6, 16); prefix_a[0, 2, 0] = 3.0
    prefix_b = torch.zeros(1, 6, 16); prefix_b[0, 2, 1] = 3.0
    q = torch.zeros(1, 1, 16); q[0, 0, 3] = 1.0
    with torch.no_grad():
        ya = m(torch.cat([prefix_a, q], 1))[0, -1]
        yb = m(torch.cat([prefix_b, q], 1))[0, -1]
    assert float(torch.max(torch.abs(ya - yb))) > 1e-6


def test_source_scan_stays_clean_without_exemption() -> None:
    """不靠定义级豁免、不靠改名绕过——CAS 本身不含被禁标识符。"""
    root = Path(__file__).resolve().parents[3] / "simplcore"
    hits = [h for h in scan_source_attention_free(root)
            if "content_state" in str(h[0])]
    assert hits == [], hits
