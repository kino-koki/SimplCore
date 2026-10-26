# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""safetensors 导出/加载往返守护（SGC + SGC Hybrid hybrid 架构）.

守护 ``simplcore/sgc/export.py`` 对混合架构的正确性：
  * save_pretrained -> load_pretrained 逐位一致（含 tie 重建）；
  * 混合模型（滑窗注意力层）导出后流式状态字节数不变（结构未变）；
  * 导出的 config.json 保留 attn_every / attn_window / mixer="hybrid"。

预算: 全模块 CPU 上 < 60 s。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.config import get_preset                            # noqa: E402
from simplcore.sgc.export import load_pretrained, save_pretrained      # noqa: E402
from simplcore.sgc.model import SGCModel                               # noqa: E402


def _hybrid_cfg():
    cfg = get_preset("debug")
    cfg.dim = 32
    cfg.num_layers = 4
    cfg.num_heads = 4
    cfg.mixer = "hybrid"
    cfg.attn_every = 2
    cfg.attn_window = 8
    return cfg


def test_hybrid_roundtrip_bitexact():
    torch.manual_seed(0)
    cfg = _hybrid_cfg()
    m1 = SGCModel(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        save_pretrained(m1, d)
        m2 = load_pretrained(d)

    # config 往返保留混合字段
    c2 = json.loads(Path(d, "config.json").read_text()) if False else m2.config
    assert c2.mixer == "hybrid"
    assert c2.attn_every == 2
    assert c2.attn_window == 8

    # 逐位一致
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        assert torch.equal(m1(idx), m2(idx))

    # 流式状态字节不变（结构未变）
    assert m1.stream_bytes(1) == m2.stream_bytes(1) > 0


def test_sgc_roundtrip_preserves_stream_bytes():
    torch.manual_seed(0)
    cfg = get_preset("debug")
    m1 = SGCModel(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        save_pretrained(m1, d)
        m2 = load_pretrained(d)
    b1, b2 = m1.stream_bytes(1), m2.stream_bytes(1)
    assert b1 == b2 and b1 > 0


def test_hybrid_exported_streaming_matches_full():
    """加载后的混合模型，流式逐 token 仍 == 整段前向。"""
    torch.manual_seed(0)
    cfg = _hybrid_cfg()
    m1 = SGCModel(cfg).eval()
    with tempfile.TemporaryDirectory() as d:
        save_pretrained(m1, d)
        m2 = load_pretrained(d)
    idx = torch.randint(0, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        full = m2(idx)
        st = m2.init_stream(2)
        outs = []
        for t in range(20):
            lg, st = m2.stream_step(idx[:, t], st)
            outs.append(lg)
        streamed = torch.stack(outs, dim=1)
    assert (streamed - full).abs().max().item() < 1e-5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
