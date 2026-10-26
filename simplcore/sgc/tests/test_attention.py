# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""GQA 滑窗注意力守护测试.

守护 ``simplcore/sgc/attention.py`` 的 GQA（分组查询）滑窗注意力：
  * MHA 退化（num_kv_heads == num_heads）与 SlidingWindowAttention 同参数同前向；
  * GQA 流式 KV 字节数 = MHA 的 num_kv_heads/num_heads 倍（省 cache 主张）；
  * 流式 == 整段前向逐位一致（MHA 与 GQA 各一例）；
  * 因果性（扰动未来不影响过去）；
  * 滑窗外 key 不可见、窗口内可见；
  * 参数量 bit-exact（analytic == model）。

预算：纯 CPU < 60 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.attention import GQASlidingWindowAttention, analytic_gqa_attn_params  # noqa: E402
from simplcore.sgc.hybrid import SlidingWindowAttention  # noqa: E402

D, H, W = 32, 4, 8
L, B = 24, 2


def _forward_stream_diff(m: torch.nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        full = m(x)
        st = m.init_stream(B, x.device, x.dtype)
        outs = []
        for t in range(L):
            lg, st = m.stream_step(x[:, t], st)
            outs.append(lg)
        s = torch.cat(outs, dim=1)
    return (s - full).abs().max().item()


# ---------------------------------------------------------------------------
# 1. MHA 退化与 SlidingWindowAttention 一致
# ---------------------------------------------------------------------------
def test_mha_matches_sliding_window_attention():
    torch.manual_seed(0)
    gqa = GQASlidingWindowAttention(D, H, W, num_kv_heads=H).eval()
    ref = SlidingWindowAttention(D, H, W).eval()
    # 同布局权重 1:1 对照：qkv (3D) + out + norm
    with torch.no_grad():
        gqa.qkv.weight.copy_(ref.qkv.weight)
        gqa.out_proj.weight.copy_(ref.out_proj.weight)
        gqa.norm.weight.copy_(ref.norm.weight)
    x = torch.randn(B, L, D)
    with torch.no_grad():
        y_gqa, y_ref = gqa(x), ref(x)
    assert torch.equal(y_gqa, y_ref), \
        f"MHA GQA != SlidingWindowAttention: {(y_gqa - y_ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# 2. GQA 省 KV cache
# ---------------------------------------------------------------------------
def test_gqa_reduces_kv_cache():
    mha = GQASlidingWindowAttention(D, H, W, num_kv_heads=H)
    gqa = GQASlidingWindowAttention(D, H, W, num_kv_heads=2)
    assert gqa.stream_bytes(1) == mha.stream_bytes(1) / 2
    assert gqa.stream_bytes(1) == 2 * 1 * 2 * W * (D // H) * 4


# ---------------------------------------------------------------------------
# 3. 流式 == 整段（MHA 与 GQA 各一例）
# ---------------------------------------------------------------------------
def test_streaming_matches_full_mha_and_gqa():
    torch.manual_seed(0)
    x = torch.randn(B, L, D)
    for G in (H, 2):
        m = GQASlidingWindowAttention(D, H, W, num_kv_heads=G).eval()
        d = _forward_stream_diff(m, x)
        assert d < 1e-5, f"GQA(kv={G}) streaming diverged: {d}"


# ---------------------------------------------------------------------------
# 4. 因果性
# ---------------------------------------------------------------------------
def test_causality():
    torch.manual_seed(0)
    m = GQASlidingWindowAttention(D, H, W, num_kv_heads=2).eval()
    x = torch.randn(B, L, D)
    x2 = x.clone()
    x2[:, L // 2:] = torch.randn_like(x2[:, L // 2:]) * 100
    with torch.no_grad():
        y1, y2 = m(x), m(x2)
    assert (y1[:, : L // 2] - y2[:, : L // 2]).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# 5. 滑窗外不可见 / 窗口内可见
# ---------------------------------------------------------------------------
def test_window_visibility():
    torch.manual_seed(0)
    m = GQASlidingWindowAttention(D, H, W, num_kv_heads=2).eval()
    x = torch.randn(B, L, D)
    x2 = x.clone()
    x2[:, 0, :] = torch.randn(1, D) * 100
    with torch.no_grad():
        y1, y2 = m(x), m(x2)
    # 位置 0 相对位置 19 在窗口外（19 - 0 = 19 >= W=8）→ 不变
    assert (y1[:, 19] - y2[:, 19]).abs().max().item() < 1e-6
    # 位置 0 相对位置 7 在窗口内（7 < 8）→ 变化
    assert (y1[:, 7] - y2[:, 7]).abs().max().item() > 1e-3


# ---------------------------------------------------------------------------
# 6. 参数量 bit-exact
# ---------------------------------------------------------------------------
def test_analytic_params_bit_exact():
    for G in (H, 2):
        m = GQASlidingWindowAttention(D, H, W, num_kv_heads=G)
        n_model = sum(p.numel() for p in m.parameters())
        n_analytic = analytic_gqa_attn_params(D, H, G)
        assert n_model == n_analytic, f"G={G}: model {n_model} != analytic {n_analytic}"


# ---------------------------------------------------------------------------
# 7. MHA 退化：GQA(num_kv_heads==H) 与 SlidingWindowAttention 数值等价
#    （同一权重布局下逐位一致 —— 等价性证明，不只是参数量一致）
# ---------------------------------------------------------------------------
def test_mha_equivalence_numeric_identity():
    torch.manual_seed(0)
    gqa = GQASlidingWindowAttention(D, H, W, num_kv_heads=H).eval()
    ref = SlidingWindowAttention(D, H, W).eval()
    with torch.no_grad():
        gqa.qkv.weight.copy_(ref.qkv.weight)
        gqa.out_proj.weight.copy_(ref.out_proj.weight)
        gqa.norm.weight.copy_(ref.norm.weight)
    x = torch.randn(B, L, D)
    with torch.no_grad():
        y_gqa, y_ref = gqa(x), ref(x)
    assert (y_gqa - y_ref).abs().max().item() < 1e-6


def test_gqa_k_squash_equivalence_to_mha():
    """GQA(G) 等价于 MHA 的组内共享 K/V 特例（直接数值证明）：

    构造 MHA 权重使 k/v 段按「组内共享」约束布局（组 g 的 grp 个 query
    头共用同一个 kv 向量，即 MHA 的 k 段 = GQA 的 k_gqa 按组广播回 H 头），
    则 GQA(G) 与约束后的 MHA 前向逐位一致。
    """
    torch.manual_seed(1)
    G, hd, grp = 2, D // H, H // 2
    mha = SlidingWindowAttention(D, H, W).eval()
    gqa = GQASlidingWindowAttention(D, H, W, num_kv_heads=G).eval()

    with torch.no_grad():
        q_w = mha.qkv.weight[:D].clone()              # q 投影（两模块共享）
        k_gqa = gqa.qkv.weight[D:D + G * hd].clone()  # G*hd 行
        v_gqa = gqa.qkv.weight[D + G * hd:].clone()
        # MHA 的 k/v 段（各 D 行）= 组内广播：组 g 内 grp 个 head 共用
        # k_gqa[g]（k_gqa 是 G*hd 行，按组切片即得每头 hd 行）
        mha_k_seg = torch.cat(
            [k_gqa[g * hd:(g + 1) * hd].repeat(grp, 1) for g in range(G)], 0)
        mha_v_seg = torch.cat(
            [v_gqa[g * hd:(g + 1) * hd].repeat(grp, 1) for g in range(G)], 0)
        mha.qkv.weight.data.copy_(
            torch.cat([q_w, mha_k_seg, mha_v_seg], dim=0))
        gqa.qkv.weight.data.copy_(
            torch.cat([q_w, k_gqa, v_gqa], dim=0))
        gqa.out_proj.weight.data.copy_(mha.out_proj.weight)
        gqa.norm.weight.data.copy_(mha.norm.weight)
    x = torch.randn(B, L, D)
    with torch.no_grad():
        y_gqa, y_mha = gqa(x), mha(x)
    d = (y_gqa - y_mha).abs().max().item()
    assert d < 1e-5, f"GQA-as-MHA-special-case diverged: {d}"
