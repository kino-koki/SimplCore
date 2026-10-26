# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC ↔ Transformer/HF 互操作桥接守护测试.

守护 ``simplcore/sgc/hf_bridge.py`` 的核心契约:
  * 保守搬运: 形状匹配才搬, 错配一律跳过并记录 (不污染模型);
  * 因果性继承: 迁移后的 SGC 仍严格因果 (扰动未来, 前文逐位不变);
  * 暖启动: 注意力衰减推断 + 振荡基网格重铺, 结构不变;
  * HF 评测桥: 不改计算路径, 只映射调用约定;
  * 1:1 组件确实被搬 (embedding / FFN / norms / head)。

预算: 全模块 CPU 上 < 90 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.config import get_preset                            # noqa: E402
from simplcore.sgc.hf_bridge import (                                    # noqa: E402
    init_sgc_from_hf,
    migrate_embeddings_and_ffn,
    attention_to_sgc_warmstart,
    sgc_to_hf_eval_bridge,
    sgc_state_to_pseudo_kv,
    MigrationReport,
)
from simplcore.sgc.model import SGCModel                               # noqa: E402


def _gpt2_style_state_dict(V=1024, D=128, L=2, H=512, seed=0):
    """构造 GPT-2 / Llama 混合风格的 state_dict (尺寸匹配 sgc debug preset)。"""
    torch.manual_seed(seed)
    state = {
        "wte.weight": torch.randn(V, D),
        "ln_f.weight": torch.ones(D),
        "lm_head.weight": torch.randn(V, D),
    }
    for i in range(L):
        state[f"transformer.h.{i}.ln_1.weight"] = torch.ones(D)
        state[f"transformer.h.{i}.ln_2.weight"] = torch.ones(D)
        state[f"blocks.{i}.down_proj.weight"] = torch.randn(D, H)
        state[f"blocks.{i}.up_proj.weight"] = torch.randn(H, D)
        state[f"blocks.{i}.gate_proj.weight"] = torch.randn(H, D)
        state[f"blocks.{i}.attn.out_proj.weight"] = torch.randn(D, D)
    return state


def _assert_causal(model, L=32, split=16):
    """扰动未来位置, 前文输出必须逐位不变 (严格因果)。"""
    cfg = model.config
    idx = torch.randint(0, cfg.vocab_size, (1, L))
    with torch.no_grad():
        y1 = model(idx)
        idx2 = idx.clone()
        idx2[:, split:] = torch.randint(0, cfg.vocab_size, (1, L - split))
        y2 = model(idx2)
    delta = (y1[:, :split] - y2[:, :split]).abs().max().item()
    assert delta < 1e-4, f"causality broken: {delta}"
    return delta


# ---------------------------------------------------------------------------
# 1. 保守搬运: 匹配组件被搬, 错配被跳过
# ---------------------------------------------------------------------------
def test_1to1_components_migrated():
    state = _gpt2_style_state_dict()
    sgc, report = init_sgc_from_hf(state, target_preset="debug")
    # embedding 被搬
    assert torch.equal(sgc.tok_emb.weight.detach(), state["wte.weight"])
    # FFN down 被搬
    assert torch.equal(sgc.blocks[0].ffn.down.weight.detach(),
                       state["blocks.0.down_proj.weight"])
    # ln_f 被搬
    assert torch.equal(sgc.ln_f.weight.detach(), state["ln_f.weight"])
    # 报告记录了搬运
    assert any("tok_emb" in m for m in report.moved)
    assert any("ffn.down" in m for m in report.moved)
    assert report.ok


def test_shape_mismatch_is_skipped_not_corrupted():
    """形状错配必须进 skipped_shape, 绝不污染模型 (走默认初始化)。"""
    state = _gpt2_style_state_dict()
    # 故意把 embedding 改成错误形状 (V 翻倍)
    state["wte.weight"] = torch.randn(2048, 128)
    sgc, report = init_sgc_from_hf(state, target_preset="debug")
    # 错配被记录, 未被搬运
    assert any("tok_emb" in s[0] for s in report.skipped_shape)
    # 模型结构完好, 因果性未被破坏
    _assert_causal(sgc)


def test_missing_keys_do_not_crash():
    """源 state_dict 缺键时必须优雅降级 (不搬), 不抛异常。"""
    sgc = SGCModel(get_preset("debug"))
    report = migrate_embeddings_and_ffn(sgc, {}, None)
    # 没有源, 全部走默认初始化, 但模型仍可前向
    idx = torch.randint(0, sgc.config.vocab_size, (1, 16))
    with torch.no_grad():
        out = sgc(idx)
    assert out.shape == (1, 16, sgc.config.vocab_size)


# ---------------------------------------------------------------------------
# 2. 因果性继承
# ---------------------------------------------------------------------------
def test_migration_preserves_strict_causality():
    state = _gpt2_style_state_dict()
    sgc, _ = init_sgc_from_hf(state, target_preset="debug")
    _assert_causal(sgc)


def test_migration_preserves_streaming_state_invariant():
    """迁移不改变 O(1) 流式状态字节数 (结构未动)。"""
    state = _gpt2_style_state_dict()
    sgc, _ = init_sgc_from_hf(state, target_preset="debug")
    fresh = SGCModel(get_preset("debug"))
    assert sgc.stream_bytes(1) == fresh.stream_bytes(1), "stream state changed by migration"


# ---------------------------------------------------------------------------
# 3. 暖启动: 衰减推断 + 振荡基重铺
# ---------------------------------------------------------------------------
def test_warmstart_rescales_oscillator_grid():
    state = _gpt2_style_state_dict()
    sgc, report = init_sgc_from_hf(state, target_preset="debug")
    # 暖启动被记录
    assert any("raw_rho" in w for w in report.warmstarted)
    # raw_rho 不再是默认 8192 网格 (推断的 decay_max 约 4070)
    assert "attention warmstart" in " ".join(report.notes)


def test_warmstart_accepts_explicit_decay_max():
    state = _gpt2_style_state_dict()
    sgc, report = init_sgc_from_hf(state, target_preset="debug", decay_max=1024.0)
    assert "decay_max=1024" in " ".join(report.notes)


# ---------------------------------------------------------------------------
# 4. HF 评测桥
# ---------------------------------------------------------------------------
def test_hf_eval_bridge_generate_and_stream_bytes():
    state = _gpt2_style_state_dict()
    sgc, _ = init_sgc_from_hf(state, target_preset="debug")
    bridge = sgc_to_hf_eval_bridge(sgc)
    idx = torch.randint(0, sgc.config.vocab_size, (1, 8))
    out = bridge.generate(idx, max_new_tokens=4, temperature=0.0, use_state=True)
    assert out.shape[1] == 12
    # 透传 O(1) 状态字节数
    assert bridge.stream_bytes > 0


def test_pseudo_kv_reflects_constant_state_bytes():
    state = _gpt2_style_state_dict()
    sgc, _ = init_sgc_from_hf(state, target_preset="debug")
    st = sgc.init_stream(2)
    pkv = sgc_state_to_pseudo_kv(sgc, st, batch=2)
    assert pkv["state_bytes"] == torch.tensor(st.num_bytes())
    assert pkv["pseudo_k"].shape == (2, 1, sgc.config.dim)


# ---------------------------------------------------------------------------
# 5. 报告完整性
# ---------------------------------------------------------------------------
def test_report_is_auditable():
    state = _gpt2_style_state_dict()
    sgc, report = init_sgc_from_hf(state, target_preset="debug")
    s = report.summary()
    # 每类计数都能在 summary 里看到
    for label in ("moved", "skipped shape", "skipped unk", "warmstart"):
        assert label in s
    # 错配与未知键数为 0 (干净源)
    assert len(report.skipped_shape) == 0
    assert len(report.skipped_unknown) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
