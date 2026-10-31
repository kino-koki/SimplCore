# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""思考加速模块的测试。"""
from __future__ import annotations

import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.accel import AdaptiveDepth, ComputeLedger


def _tiny():
    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                         decay_max=128, local_kernel=3, num_heads=2,
                         ffn_ratio=2.0, mixer="sgc")
    return SimplCoreModel(cfg).eval()


def test_noop_is_bit_identical_to_model():
    """skip_threshold=-inf、min_depth=层数 时退化为恒全算力，与模型逐位一致。"""
    m = _tiny()
    acc = AdaptiveDepth(m, skip_threshold=float("-inf"), min_depth=m.config.num_layers)
    idx = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        g1 = m.generate(idx, max_new_tokens=6, temperature=0, top_k=1)
        g2 = acc.generate(idx, max_new_tokens=6, temperature=0, top_k=1)
    assert torch.equal(g1, g2)


def test_noop_ledger_speedup_is_one():
    m = _tiny()
    acc = AdaptiveDepth(m, skip_threshold=float("-inf"), min_depth=m.config.num_layers)
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        acc.generate(idx, max_new_tokens=6, temperature=0, top_k=1)
    assert acc.ledger.speedup == 1.0
    assert isinstance(acc.ledger, ComputeLedger)


def test_aggressive_skip_gives_speedup_and_finite():
    m = _tiny()
    acc = AdaptiveDepth(m, skip_threshold=1e9, min_depth=1)
    idx = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        g = acc.generate(idx, max_new_tokens=8, temperature=0, top_k=1)
    assert acc.ledger.speedup > 1.0
    assert torch.isfinite(g).all()


def test_accelerator_has_no_attention():
    m = _tiny()
    acc = AdaptiveDepth(m, skip_threshold=1e9, min_depth=1)
    assert acc.attention_like_modules() == []


def test_invalid_depths_rejected():
    m = _tiny()
    try:
        AdaptiveDepth(m, min_depth=0)
        assert False, "min_depth<1 应被拒绝"
    except ValueError:
        pass
    try:
        AdaptiveDepth(m, min_depth=10, max_depth=4)
        assert False, "min_depth>max_depth 应被拒绝"
    except ValueError:
        pass
