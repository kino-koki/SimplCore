# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""GQA + YaRN 组合注意力守护测试.

守护 ``attention_yarn.py``（GQASlidingWindowAttentionYaRN）的
"省 KV 缓存 + 长程外推"结构组合路径：
  * GQA+YaRN 构建 + 前向有限；
  * 流式 == 整段前向逐位一致（含 YaRN 缩放表）；
  * stream_bytes 按 G 头核算（省 H/G 倍）；
  * 参数量 bit-exact；
  * strategy="none" 逐位等于 GQASlidingWindowAttention（同权重）；
  * MHA 退化（G=H）逐位等于 SlidingWindowAttentionYaRN（同权重）；
  * 因果性（未来扰动不改变过去）；
  * GQA+YaRN preset 端到端（hybrid-gqa-yarn-150m 小版）。

预算：纯 CPU < 60 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.attention import GQASlidingWindowAttention            # noqa: E402
from simplcore.sgc.attention_yarn import (                                # noqa: E402
    GQASlidingWindowAttentionYaRN,
    analytic_gqa_yarn_attn_params,
)
from simplcore.sgc.rope_scales import SlidingWindowAttentionYaRN         # noqa: E402
from simplcore.sgc.hybrid import SlidingWindowAttention                   # noqa: E402
from simplcore.sgc.config import SGCConfig                                # noqa: E402
from simplcore.sgc.model import SGCModel                                   # noqa: E402


def _cfg(**kw) -> SGCConfig:
    base = dict(vocab_size=128, dim=32, num_layers=4, num_heads=4,
                num_bases=8, local_kernel=4, attn_every=2, attn_window=8,
                mixer="hybrid")
    base.update(kw)
    return SGCConfig(**base)


def _ref_gqa(dim, H, G, W, seed=0, strategy="none", factor=1.0):
    """同权重参照：strategy=none 时与 GQASlidingWindowAttention 同布局."""
    torch.manual_seed(seed)
    m = GQASlidingWindowAttentionYaRN(
        dim=dim, num_heads=H, num_kv_heads=G, window=W,
        strategy=strategy, factor=factor)
    return m


# ---------------------------------------------------------------------------
# 1. 构建 + 前向有限
# ---------------------------------------------------------------------------
def test_build_and_forward_finite():
    m = _ref_gqa(dim=32, H=4, G=2, W=8, seed=0)
    x = torch.randn(2, 16, 32)
    out = m(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert m.num_kv_heads == 2
    assert m.head_dim == 8


# ---------------------------------------------------------------------------
# 2. 流式 == 整段前向（含 YaRN 缩放表）
# ---------------------------------------------------------------------------
def test_streaming_matches_full_forward_yarn():
    torch.manual_seed(0)
    m = GQASlidingWindowAttentionYaRN(dim=32, num_heads=4, num_kv_heads=2,
                                      window=8, strategy="yarn", factor=2.0).eval()
    x = torch.randn(1, 24, 32)
    with torch.no_grad():
        full = m(x)
        st = m.init_stream(1, device=x.device, dtype=x.dtype)
        out = []
        for t in range(24):
            y, st = m.stream_step(x[:, t:t + 1], st)
            out.append(y.squeeze(1))
        streamed = torch.stack(out, dim=1)
    diff = (full - streamed).abs().max().item()
    assert diff < 1e-5, f"streaming vs full diff={diff}"


# ---------------------------------------------------------------------------
# 3. 流式 == 整段前向（strategy=none，逐位等于固定 RoPE GQA）
# ---------------------------------------------------------------------------
def test_streaming_matches_full_forward_none():
    torch.manual_seed(0)
    m = GQASlidingWindowAttentionYaRN(dim=32, num_heads=4, num_kv_heads=2,
                                      window=8, strategy="none", factor=1.0).eval()
    x = torch.randn(1, 16, 32)
    with torch.no_grad():
        full = m(x)
        st = m.init_stream(1, device=x.device, dtype=x.dtype)
        out = []
        for t in range(16):
            y, st = m.stream_step(x[:, t:t + 1], st)
            out.append(y.squeeze(1))
        streamed = torch.stack(out, dim=1)
    assert (full - streamed).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# 4. stream_bytes 按 G 头核算（省 H/G 倍）
# ---------------------------------------------------------------------------
def test_stream_bytes_by_g_heads():
    m = GQASlidingWindowAttentionYaRN(dim=32, num_heads=4, num_kv_heads=2,
                                      window=8, strategy="yarn", factor=2.0)
    # G=2, hd=8, W=8: 2 * 1 * 2 * 8 * 8 * 4 = 1024 bytes (fp32)
    assert m.stream_bytes(1, torch.float32) == 2 * 1 * 2 * 8 * 8 * 4
    # MHA 对照：G=4 → 2 * 1 * 4 * 8 * 8 * 4 = 2048
    m_mha = GQASlidingWindowAttentionYaRN(dim=32, num_heads=4, num_kv_heads=4,
                                          window=8)
    assert m_mha.stream_bytes(1, torch.float32) == 2 * 1 * 4 * 8 * 8 * 4
    assert m.stream_bytes(1) * 2 == m_mha.stream_bytes(1)  # G=2 省 1/2


# ---------------------------------------------------------------------------
# 5. 参数量 bit-exact
# ---------------------------------------------------------------------------
def test_analytic_params_bit_exact():
    dim, H, G, hd = 32, 4, 2, 8
    analytic = analytic_gqa_yarn_attn_params(dim, H, G)
    m = GQASlidingWindowAttentionYaRN(dim=dim, num_heads=H, num_kv_heads=G,
                                      window=8, strategy="yarn", factor=2.0)
    real = sum(p.numel() for p in m.parameters())
    assert real == analytic, f"real={real} analytic={analytic}"
    # MHA 退化
    analytic_mha = analytic_gqa_yarn_attn_params(dim, H, H)
    m_mha = GQASlidingWindowAttentionYaRN(dim=dim, num_heads=H, num_kv_heads=H,
                                          window=8)
    assert sum(p.numel() for p in m_mha.parameters()) == analytic_mha
    assert analytic < analytic_mha  # GQA 省参数


# ---------------------------------------------------------------------------
# 6. strategy="none" 逐位等于 GQASlidingWindowAttention（同权重）
# ---------------------------------------------------------------------------
def test_none_strategy_equals_plain_gqa():
    dim, H, G, W = 32, 4, 2, 8
    torch.manual_seed(42)
    m_yarn = GQASlidingWindowAttentionYaRN(dim=dim, num_heads=H, num_kv_heads=G,
                                           window=W, strategy="none", factor=1.0)
    m_plain = GQASlidingWindowAttention(dim=dim, num_heads=H, num_kv_heads=G,
                                         window=W)
    # 同权重布局（qkv = D + 2*G*hd）→ 直接搬
    m_plain.qkv.weight.data.copy_(m_yarn.qkv.weight.data)
    m_plain.out_proj.weight.data.copy_(m_yarn.out_proj.weight.data)
    m_plain.norm.weight.data.copy_(m_yarn.norm.weight.data)
    m_plain.eval(); m_yarn.eval()
    x = torch.randn(2, 12, dim)
    with torch.no_grad():
        out_plain = m_plain(x)
        out_yarn = m_yarn(x)
    assert (out_plain - out_yarn).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 7. MHA 退化（G=H）逐位等于 SlidingWindowAttentionYaRN（同权重）
# ---------------------------------------------------------------------------
def test_mha_degenerate_equals_sliding_window_yarn():
    dim, H, W = 32, 4, 8
    torch.manual_seed(7)
    m_gqa_yarn = GQASlidingWindowAttentionYaRN(dim=dim, num_heads=H,
                                               num_kv_heads=H, window=W,
                                               strategy="yarn", factor=2.0)
    m_ref = SlidingWindowAttentionYaRN(dim=dim, num_heads=H, window=W,
                                        strategy="yarn", factor=2.0)
    # MHA 退化时 qkv 布局 = 3*D（与 SlidingWindowAttentionYaRN 一致）
    m_ref.qkv.weight.data.copy_(m_gqa_yarn.qkv.weight.data)
    m_ref.out_proj.weight.data.copy_(m_gqa_yarn.out_proj.weight.data)
    m_ref.norm.weight.data.copy_(m_gqa_yarn.norm.weight.data)
    m_ref.eval(); m_gqa_yarn.eval()
    x = torch.randn(2, 12, dim)
    with torch.no_grad():
        out_ref = m_ref(x)
        out_gqa = m_gqa_yarn(x)
    assert (out_ref - out_gqa).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 8. 因果性（未来扰动不改变过去输出）
# ---------------------------------------------------------------------------
def test_causality():
    torch.manual_seed(0)
    m = GQASlidingWindowAttentionYaRN(dim=32, num_heads=4, num_kv_heads=2,
                                      window=8, strategy="yarn", factor=2.0).eval()
    x = torch.randn(1, 16, 32)
    x2 = x.clone()
    x2[:, 12:] += 1.0   # 扰动未来位置
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x2)
    # 前 12 个位置输出不变
    assert (out1[:, :12] - out2[:, :12]).abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 9. GQA+YaRN preset 端到端（小版模型）
# ---------------------------------------------------------------------------
def test_gqa_yarn_preset_end_to_end():
    cfg = _cfg(num_kv_heads=2, rope_scale_strategy="yarn", rope_scale_factor=2.0)
    m = SGCModel(cfg).eval()
    # 注意力层应为 GQASlidingWindowAttentionYaRN
    attn_blks = [b for b in m.blocks if hasattr(b.mixer, "scaler")]
    assert len(attn_blks) >= 1
    for b in attn_blks:
        assert type(b.mixer).__name__ == "GQASlidingWindowAttentionYaRN"
        assert b.mixer.num_kv_heads == 2
        assert b.mixer.strategy == "yarn"
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits, loss = m(x, x)
    assert torch.isfinite(loss).all()
    # 流式（单 batch 元素；init_stream(1) 对应 batch=1）
    st = m.init_stream(1, device=x.device, dtype=x.dtype)
    with torch.no_grad():
        for t in range(4):
            logits, st = m.stream_step(x[:1, t], st)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 10. HF 桥接 GQA warmstart 路径对 GQA+YaRN 层生效
# ---------------------------------------------------------------------------
def test_hf_bridge_gqa_yarn_warmstart():
    from simplcore.sgc.hf_bridge import attention_gqa_warmstart
    cfg = _cfg(num_kv_heads=2, rope_scale_strategy="yarn")
    m = SGCModel(cfg).eval()
    # 构造 Llama 风格 state_dict
    sd = {}
    for li in range(cfg.num_layers):
        p = f"model.layers.{li}."
        sd[p + "self_attn.q_proj.weight"] = torch.randn(cfg.dim, cfg.dim)
        sd[p + "self_attn.k_proj.weight"] = torch.randn(2 * 8, cfg.dim)
        sd[p + "self_attn.v_proj.weight"] = torch.randn(2 * 8, cfg.dim)
        sd[p + "self_attn.o_proj.weight"] = torch.randn(cfg.dim, cfg.dim)
        sd[p + "input_layernorm.weight"] = torch.rand(cfg.dim)
        sd[p + "post_attention_layernorm.weight"] = torch.rand(cfg.dim)
    sd["model.embed_tokens.weight"] = torch.randn(128, cfg.dim)
    sd["norm.weight"] = torch.rand(cfg.dim)
    rep = attention_gqa_warmstart(m, sd)
    # 注意力层 qkv k 段应被搬运
    blk1 = m.blocks[1]
    assert hasattr(blk1.mixer, "qkv")
    moved_kv = [r for r in rep.moved if "k_proj" in r or "v_proj" in r]
    assert len(moved_kv) >= 4, f"expected >=4 kv moves, got {len(moved_kv)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
