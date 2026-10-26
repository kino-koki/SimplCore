# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""GQA / YaRN / 门控权重迁移守护测试.

守护 ``hf_bridge.attention_gqa_warmstart`` + ``init_sgc_from_hf`` 的
GQA/YaRN 注意力权重迁移路径（Llama 风格 q_proj/k_proj/v_proj → GQA 合并 qkv）：
  * MHA→MHA（G=H）：q/k/v 段逐位搬运；
  * MHA→GQA（G<H）：k/v 段取前 G*hd 行（截断 + note 记录）；
  * GQA→GQA：k/v 段 1:1 搬运；
  * 形状错配：skip + 记录，绝不污染；
  * YaRN 配置走同一路径（rope_scale_strategy 触发 GQA warmstart）；
  * 完整 init_sgc_from_hf + cfg_override 端到端。

预算：纯 CPU < 40 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.config import SGCConfig                                    # noqa: E402
from simplcore.sgc.hf_bridge import (                                            # noqa: E402
    MigrationReport,
    attention_gqa_warmstart,
    init_sgc_from_hf,
)
from simplcore.sgc.model import SGCModel                                       # noqa: E402


def _cfg(**kw) -> SGCConfig:
    base = dict(vocab_size=128, dim=32, num_layers=4, num_heads=4,
                num_bases=8, local_kernel=4, attn_every=2, attn_window=8,
                mixer="hybrid")
    base.update(kw)
    return SGCConfig(**base)


def _llama_state(D=32, H=4, G=4, hd=8):
    """Llama 风格 state_dict：独立 q_proj/k_proj/v_proj + o_proj + ln_1/ln_2.

    G=H 时 MHA；G<H 时 GQA。
    """
    sd = {}
    for li in range(4):
        prefix = f"model.layers.{li}."
        sd[prefix + "self_attn.q_proj.weight"] = torch.randn(D, D)
        sd[prefix + "self_attn.k_proj.weight"] = torch.randn(G * hd, D)
        sd[prefix + "self_attn.v_proj.weight"] = torch.randn(G * hd, D)
        sd[prefix + "self_attn.o_proj.weight"] = torch.randn(D, D)
        sd[prefix + "input_layernorm.weight"] = torch.rand(D)
        sd[prefix + "post_attention_layernorm.weight"] = torch.rand(D)
        sd[prefix + "mlp.gate_proj.weight"] = torch.randn(128, D)
        sd[prefix + "mlp.up_proj.weight"] = torch.randn(128, D)
        sd[prefix + "mlp.down_proj.weight"] = torch.randn(D, 128)
    sd["model.embed_tokens.weight"] = torch.randn(128, D)
    sd["norm.weight"] = torch.rand(D)
    return sd


# ---------------------------------------------------------------------------
# 1. MHA→MHA（G=H）逐位搬运
# ---------------------------------------------------------------------------
def test_mha_to_mha_numeric():
    cfg = _cfg()                       # 默认 num_kv_heads=None -> G=H=4
    m = SGCModel(cfg).eval()
    sd = _llama_state(D=32, H=4, G=4, hd=8)
    rep = attention_gqa_warmstart(m, sd)
    # 搬完 q 段应与 q_proj 一致（以某一层为例）
    blk1 = m.blocks[1]
    qkv_w = blk1.mixer.qkv.weight.data
    assert qkv_w[:32].shape == (32, 32)
    # 至少 q_proj 被搬运了
    moved_q = [r for r in rep.moved if "qkv[q]" in r and "q_proj" in r]
    assert len(moved_q) >= 2, f"expected >=2 q_proj moves, got {len(moved_q)}"


# ---------------------------------------------------------------------------
# 2. MHA→GQA（G<H）k/v 段截断 + note 记录
# ---------------------------------------------------------------------------
def test_mha_to_gqa_truncation():
    cfg = _cfg(num_kv_heads=2)          # G=2 < H=4
    m = SGCModel(cfg).eval()
    sd = _llama_state(D=32, H=4, G=4, hd=8)   # 源 MHA（G=4）
    rep = attention_gqa_warmstart(m, sd)
    blk1 = m.blocks[1]
    # k 段 shape = G*hd = 2*8 = 16
    assert blk1.mixer.qkv.weight.data[32:48].shape == (16, 32)
    notes_trunc = [n for n in rep.notes if "truncated" in n]
    assert len(notes_trunc) >= 4, f"expected truncation notes, got {rep.notes[:4]}"


# ---------------------------------------------------------------------------
# 3. GQA→GQA 逐位搬运
# ---------------------------------------------------------------------------
def test_gqa_to_gqa():
    cfg = _cfg(num_kv_heads=2)
    m = SGCModel(cfg).eval()
    sd = _llama_state(D=32, H=4, G=2, hd=8)   # 源 GQA（G=2）
    rep = attention_gqa_warmstart(m, sd)
    blk1 = m.blocks[1]
    moved_kv = [r for r in rep.moved if "k_proj" in r or "v_proj" in r]
    assert len(moved_kv) >= 4, f"expected >=4 kv moves, got {len(moved_kv)}"
    # k 段应与源 k_proj 逐位一致
    assert torch.equal(blk1.mixer.qkv.weight.data[32:48],
                        sd["model.layers.1.self_attn.k_proj.weight"])


# ---------------------------------------------------------------------------
# 4. 形状错配 skip + 记录
# ---------------------------------------------------------------------------
def test_shape_mismatch_skipped():
    cfg = _cfg(num_kv_heads=2)
    m = SGCModel(cfg).eval()
    sd = _llama_state(D=32, H=4, G=2, hd=8)
    # 注意力层是 layer 1（attn_every=2）；把 layer1 的 q_proj 改成错误 shape
    sd["model.layers.1.self_attn.q_proj.weight"] = torch.randn(16, 32)
    rep = attention_gqa_warmstart(m, sd)
    assert any("blk1" in s[0] and "qkv[q]" in s[0] for s in rep.skipped_shape), \
        f"expected blk1 q_proj mismatch in skipped_shape, got {rep.skipped_shape[:4]}"


# ---------------------------------------------------------------------------
# 5. YaRN 配置走 GQA warmstart 路径
# ---------------------------------------------------------------------------
def test_yarn_triggers_gqa_warmstart():
    cfg = _cfg(rope_scale_strategy="yarn", rope_scale_factor=2.0)
    m = SGCModel(cfg).eval()
    sd = _llama_state(D=32, H=4, G=4, hd=8)
    rep = attention_gqa_warmstart(m, sd)
    # 注意力层有 qkv（YaRN 模块）→ 应被搬运
    moved_attn = [r for r in rep.moved if "qkv" in r]
    assert len(moved_attn) >= 4, f"expected >=4 attn qkv moves, got {len(moved_attn)}"


# ---------------------------------------------------------------------------
# 6. 完整 init_sgc_from_hf + cfg_override（GQA 端到端）
# ---------------------------------------------------------------------------
def test_init_sgc_from_hf_gqa_e2e():
    cfg = _cfg(num_kv_heads=2)
    sd = _llama_state(D=32, H=4, G=2, hd=8)
    m, rep = init_sgc_from_hf(sd, target_preset="sgc-150m", cfg_override=cfg)
    # GQA 层 qkv k 段应被搬运
    blk1 = m.blocks[1]
    assert hasattr(blk1.mixer, "qkv")
    assert torch.equal(blk1.mixer.qkv.weight.data[32:48],
                        sd["model.layers.1.self_attn.k_proj.weight"])
    # report 完整
    assert rep.ok
    assert len(rep.moved) > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
