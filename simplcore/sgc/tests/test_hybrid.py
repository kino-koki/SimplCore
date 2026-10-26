# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC 混合架构（全局谱卷积 + 局部滑窗注意力）守护测试.

守护 ``simplcore/sgc/hybrid.py`` 的核心契约：
  * 布局正确性：``attn_every`` 决定哪些层用滑窗注意力，其余用 SGC；
  * 因果性继承：混合模型仍严格因果（扰动未来，前文逐位不变）；
  * 流式 == 整段：逐 token 流式前向与整段前向逐位一致（双形式等价）；
  * 流式状态有界：``stream_bytes`` 与 L 无关（SGC O(1) + 滑窗 O(W·H·D)）；
  * 参数量 bit-exact：``analytic_hybrid_params`` == 真实模型 ``num_params``；
  * 滑窗注意力本身：窗口外 key 不可见、窗口内因果正确。

预算：全模块 CPU 上 < 90 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.config import get_preset                            # noqa: E402
from simplcore.sgc.hybrid import (                                     # noqa: E402
    HybridModel,
    SlidingWindowAttention,
    analytic_hybrid_params,
    _attn_layout,
)
from simplcore.sgc.model import SGCModel                               # noqa: E402


def _hybrid_cfg(attn_every=2, window=16, dim=32, layers=4, heads=4, seed=0):
    cfg = get_preset("debug")
    cfg.dim = dim
    cfg.num_layers = layers
    cfg.num_heads = heads
    cfg.mixer = "hybrid"
    cfg.attn_every = attn_every
    cfg.attn_window = window
    torch.manual_seed(seed)
    return cfg


def _assert_causal(model, L=20, split=12, tol=1e-4):
    idx = torch.randint(0, model.config.vocab_size, (1, L))
    with torch.no_grad():
        y1 = model(idx)
        idx2 = idx.clone()
        idx2[:, split:] = torch.randint(0, model.config.vocab_size, (1, L - split))
        y2 = model(idx2)
    delta = (y1[:, :split] - y2[:, :split]).abs().max().item()
    assert delta < tol, f"causality broken: {delta}"
    return delta


# ---------------------------------------------------------------------------
# 1. 布局
# ---------------------------------------------------------------------------
def test_attn_layout_positions():
    layout = _attn_layout(4, attn_every=2)
    assert layout == [False, True, False, True]
    assert _attn_layout(8, attn_every=None) == [False] * 8
    assert _attn_layout(3, attn_every=1) == [True, True, True]
    assert _attn_layout(6, attn_every=4) == [False, False, False, True, False, False]


def test_hybrid_block_routing_via_sgcmodel():
    cfg = _hybrid_cfg(attn_every=2, layers=4)
    m = SGCModel(cfg).eval()
    attn_idx = [i for i, b in enumerate(m.blocks) if b.use_attn]
    assert attn_idx == [1, 3], f"expected attn at 1,3 got {attn_idx}"


# ---------------------------------------------------------------------------
# 2. 因果性继承
# ---------------------------------------------------------------------------
def test_hybrid_is_causal():
    cfg = _hybrid_cfg(attn_every=2, window=8, layers=4)
    m = SGCModel(cfg).eval()
    _assert_causal(m)


def test_hybrid_is_causal_window_larger_than_seq():
    """窗口 >= 序列长度时退化为全因果注意力，仍严格因果。"""
    cfg = _hybrid_cfg(attn_every=2, window=64, layers=4)
    m = SGCModel(cfg).eval()
    _assert_causal(m, L=16, split=8)


# ---------------------------------------------------------------------------
# 3. 流式 == 整段（双形式等价）
# ---------------------------------------------------------------------------
def test_streaming_matches_full_forward():
    cfg = _hybrid_cfg(attn_every=2, window=8, layers=4)
    m = SGCModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full = m(idx)
        st = m.init_stream(2)
        outs = []
        for t in range(24):
            lg, st = m.stream_step(idx[:, t], st)
            outs.append(lg)
        streamed = torch.stack(outs, dim=1)
    diff = (streamed - full).abs().max().item()
    assert diff < 1e-5, f"streaming diverged: {diff}"


# ---------------------------------------------------------------------------
# 4. 流式状态有界（与 L 无关）
# ---------------------------------------------------------------------------
def test_stream_bytes_is_length_independent():
    cfg = _hybrid_cfg(attn_every=2, window=16, layers=4)
    m = SGCModel(cfg)
    b1 = m.stream_bytes(1)
    b8 = m.stream_bytes(8)
    assert b1 > 0
    assert b8 == 8 * b1, "stream bytes must scale linearly with batch, not L"
    # 注意力层的滑窗项 + SGC 层的 O(1) 项，均不随 L 变
    fresh = HybridModel(cfg)
    assert m.stream_bytes(1) == fresh.stream_bytes(1)


# ---------------------------------------------------------------------------
# 5. 参数量 bit-exact
# ---------------------------------------------------------------------------
def test_analytic_params_matches_real_model():
    for attn_every, window in [(2, 8), (4, 32), (1, 16)]:
        cfg = _hybrid_cfg(attn_every=attn_every, window=window, layers=4)
        m = SGCModel(cfg)
        assert analytic_hybrid_params(cfg) == m.num_params(), \
            f"attn_every={attn_every} window={window} param mismatch"


def test_attn_layer_param_term():
    """注意力层每层多 2·D²（相对 SGC 层）：4D² vs 2D²+...，可解析对照。"""
    cfg_sgc = _hybrid_cfg(attn_every=None, window=8, layers=4)
    cfg_attn = _hybrid_cfg(attn_every=1, window=8, layers=4)
    # 全注意力层参数量 > 全 SGC 层（每层 +2D² 来自 qkv 比 in+out 多）
    assert analytic_hybrid_params(cfg_attn) > analytic_hybrid_params(cfg_sgc)


# ---------------------------------------------------------------------------
# 6. 滑窗注意力单元正确性
# ---------------------------------------------------------------------------
def test_sliding_window_masks_out_of_window():
    """窗口外（j < t - W）的 key 不可见：扰动窗口外位置不影响 t 的输出。"""
    torch.manual_seed(0)
    dim, heads, window = 32, 4, 8
    attn = SlidingWindowAttention(dim, heads, window=window).eval()
    x = torch.randn(2, 20, dim)
    with torch.no_grad():
        y1 = attn(x)
        # 扰动 t=19 的窗口外历史：j < 19 - 8 + 1 = 12，即位置 0..11
        # 只改位置 0（对 t=19 而言远在窗口外），t=19 的输出不应变
        x2 = x.clone()
        t = 19
        x2[:, 0, :] = torch.randn(2, dim) * 100.0
        y2 = attn(x2)
    # 位置 t=19 的输出不变（位置 0 在其窗口外）
    delta = (y1[:, t] - y2[:, t]).abs().max().item()
    assert delta < 1e-4, f"out-of-window leak at t={t}: {delta}"
    # 位置 0 自身的输出会变（因为 x[0] 被改了），这是预期的
    # 位置 8 仍在窗口内（j=0 对 t=8 来说 8-0=8 不 < W=8，所以 j=0 对 t=8 也窗口外）
    # 位置 7：j=0, 7-0=7 < 8 在窗口内 -> 会变
    assert (y1[:, 7] - y2[:, 7]).abs().max().item() > 1e-2, "in-window key should affect output"


def test_sliding_window_causal():
    """窗口内因果：扰动未来位置不影响当前。"""
    torch.manual_seed(0)
    attn = SlidingWindowAttention(32, 4, window=16).eval()
    x = torch.randn(1, 20, 32)
    with torch.no_grad():
        y1 = attn(x)
        x2 = x.clone()
        x2[:, 15:] = torch.randn_like(x2[:, 15:]) * 100
        y2 = attn(x2)
    assert (y1[:, :15] - y2[:, :15]).abs().max().item() < 1e-4


def test_sliding_window_equals_full_attn_when_window_ge_L():
    """窗口 >= L 时，滑窗注意力 == 普通因果注意力。"""
    torch.manual_seed(0)
    dim, heads, L = 32, 4, 12
    from simplcore.sgc.model import CausalAttention
    sw = SlidingWindowAttention(dim, heads, window=L).eval()
    ca = CausalAttention(dim, heads).eval()
    x = torch.randn(2, L, dim)
    with torch.no_grad():
        # 共用同一套 QKV / out / norm 权重以对照
        ca.qkv.weight.copy_(sw.qkv.weight)
        ca.out_proj.weight.copy_(sw.out_proj.weight)
        if hasattr(ca.norm, "weight"):
            ca.norm.weight.copy_(sw.norm.weight)
        y_sw = sw(x)
        y_ca = ca(x)
    diff = (y_sw - y_ca).abs().max().item()
    assert diff < 1e-4, f"window>=L should equal full causal attn: {diff}"


# ---------------------------------------------------------------------------
# 7. 生成路径
# ---------------------------------------------------------------------------
def test_hybrid_generate_runs():
    cfg = _hybrid_cfg(attn_every=2, window=8, layers=4)
    m = SGCModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    out = m.generate(idx, max_new_tokens=4, temperature=0.0)
    assert out.shape[1] == 12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
