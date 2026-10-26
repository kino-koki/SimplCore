# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC Hybrid 集成守护：SGCModel 路由 GQA / gate_ab / YaRN + 参数量 bit-exact + 流式==整段.

守护 3 个新 mixer/注意力模块（``attention.py`` / ``selective.py`` /
``rope_scales.py``）接入 ``SGCModel``（``mixer="hybrid"`` + 新 config 字段）
后的**集成不变量**：
  * ``SGCModel(cfg)`` 在 num_kv_heads / gate_ab / rope_scale_strategy 各组合下
    能构建、前向、流式；
  * 流式 == 整段前向逐位一致（混合堆叠，max|Δ|<1e-5）；
  * 参数量 bit-exact：``SGCModel(cfg).num_params() == analytic_params(cfg)``
    （覆盖 GQA 省 kv 项 / gate_ab 双门 2D² 项 / YaRN MHA 布局项）。

预算：全模块 CPU 上 < 90 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.config import SGCConfig, analytic_params          # noqa: E402
from simplcore.sgc.model import SGCModel                             # noqa: E402


def _cfg(**kw) -> SGCConfig:
    base = dict(vocab_size=1024, dim=32, num_layers=4, num_heads=4,
                num_bases=8, local_kernel=4, attn_every=2, attn_window=8,
                mixer="hybrid")
    base.update(kw)
    return SGCConfig(**base)


def _stream_diff(m: SGCModel, x: torch.Tensor, L: int = 16) -> float:
    with torch.no_grad():
        full = m(x)
        st = m.init_stream(x.shape[0])
        outs = []
        for t in range(L):
            lg, st = m.stream_step(x[:, t], st)
            outs.append(lg)
        s = torch.stack(outs, 1)
    return (s - full).abs().max().item()


# ---------------------------------------------------------------------------
# 1. 各组合可构建 + 前向有限
# ---------------------------------------------------------------------------
def test_all_configs_build_and_forward_finite():
    combos = [
        {},
        dict(num_kv_heads=2),
        dict(gate_ab=True),
        dict(num_kv_heads=2, gate_ab=True),
        dict(rope_scale_strategy="yarn", rope_scale_factor=2.0),
        dict(rope_scale_strategy="linear", rope_scale_factor=2.0),
        # GQA + YaRN 组合层（GQASlidingWindowAttentionYaRN）
        dict(num_kv_heads=2, rope_scale_strategy="yarn", rope_scale_factor=2.0),
        dict(num_kv_heads=2, gate_ab=True, rope_scale_strategy="yarn",
             rope_scale_factor=2.0),
    ]
    x = torch.randint(0, 1024, (2, 16))
    for kw in combos:
        m = SGCModel(_cfg(**kw)).eval()
        with torch.no_grad():
            logits, _ = m(x, x)
        assert torch.isfinite(logits).all(), f"non-finite logits for {kw}"
        # GQA+YaRN 组合层应被正确路由
        if kw.get("num_kv_heads") and kw.get("rope_scale_strategy") in ("yarn", "linear"):
            attn_blks = [b for b in m.blocks if hasattr(b.mixer, "scaler")]
            assert len(attn_blks) >= 1
            for b in attn_blks:
                assert type(b.mixer).__name__ == "GQASlidingWindowAttentionYaRN", \
                    f"unexpected mixer type {type(b.mixer).__name__} for {kw}"


# ---------------------------------------------------------------------------
# 2. 流式 == 整段（混合 GQA + gate_ab + YaRN 堆叠）
# ---------------------------------------------------------------------------
def test_streaming_matches_full_forward_all_combos():
    x = torch.randint(0, 1024, (2, 16))
    for kw in (
        dict(num_kv_heads=2),
        dict(gate_ab=True),
        dict(num_kv_heads=2, gate_ab=True),
        dict(rope_scale_strategy="yarn", rope_scale_factor=2.0),
        dict(num_kv_heads=2, gate_ab=True, rope_scale_strategy="yarn",
             rope_scale_factor=2.0),
    ):
        m = SGCModel(_cfg(**kw)).eval()
        d = _stream_diff(m, x)
        assert d < 1e-5, f"streaming diverged for {kw}: {d}"


# ---------------------------------------------------------------------------
# 3. 参数量 bit-exact（6 种组合）
# ---------------------------------------------------------------------------
def test_analytic_params_bit_exact_all_combos():
    for kw in (
        {},
        dict(num_kv_heads=2),
        dict(gate_ab=True),
        dict(num_kv_heads=2, gate_ab=True),
        dict(rope_scale_strategy="yarn", rope_scale_factor=2.0),
        dict(num_kv_heads=2, gate_ab=True, rope_scale_strategy="yarn",
             rope_scale_factor=2.0),
    ):
        cfg = _cfg(**kw)
        n_model = SGCModel(cfg).num_params()
        n_analytic = analytic_params(cfg)
        assert n_model == n_analytic, \
            f"{kw}: model {n_model} != analytic {n_analytic}"


# ---------------------------------------------------------------------------
# 4. GQA 省 KV 体现在 stream_bytes
# ---------------------------------------------------------------------------
def test_gqa_reduces_stream_bytes():
    mha = SGCModel(_cfg()).eval()
    gqa = SGCModel(_cfg(num_kv_heads=2)).eval()
    # GQA 把注意力层 KV 缓存缩半（SGC 层状态不变）→ 总字节严格更少
    assert gqa.stream_bytes(1) < mha.stream_bytes(1)
    # 差值 = 2 个注意力层 × (4→2 kv 头) 的 KV 省量 = 2 × 2·W·hd·4 = 2048
    assert mha.stream_bytes(1) - gqa.stream_bytes(1) == 2 * 1024


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
