# -*- coding: utf-8 -*-
"""SimplCoreModel.snapshot / restore 的测试（DreamingLoop 权重回滚依赖它）。"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel


def _tiny() -> SimplCoreConfig:
    return SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_heads=2,
                     num_bases=4, ffn_ratio=2.0)


def test_snapshot_restore_roundtrip_is_bit_exact() -> None:
    m = SimplCoreModel(_tiny())
    snap = m.snapshot()

    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)

    m.restore(snap)
    now = m.state_dict()
    for k, v in snap.items():
        assert torch.equal(now[k].cpu(), v), f"参数 {k} 未被精确恢复"


def test_restore_rejects_empty_snapshot() -> None:
    m = SimplCoreModel(_tiny())
    with pytest.raises(ValueError):
        m.restore({})
