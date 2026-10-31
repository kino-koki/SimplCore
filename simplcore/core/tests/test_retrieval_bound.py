# -*- coding: utf-8 -*-
"""检索容量账的契约测试（原创清单 B 的最小可证伪框架）。

核心断言：**1M needle 的瓶颈不是状态比特数**，而是固定维线性读出的干扰。
如果这个账算错了，"O(1) 状态检索下界"的整条论证都会站不住。
"""
from __future__ import annotations

import pytest

from simplcore.core.config import get_preset
from simplcore.core.retrieval_bound import (
    capacity_ratio, ledger, linear_readout_snr,
)
from simplcore.core.scaling import stream_state_bytes


def test_tuan_1b_capacity_is_not_the_bottleneck() -> None:
    """30.4 MiB 状态对 1M×V=32000 的容量比 > 1：信息论上够存。"""
    cfg = get_preset("tuan-1b")
    floats = stream_state_bytes(cfg, batch=1) // 4
    ratio = capacity_ratio(floats, 1_048_576, 32000)
    assert ratio > 1.0, (floats, ratio)
    # 但线性读出的干扰仍然存在：SNR 在 O(1) 量级，不是"无限可检索"。
    snr = linear_readout_snr(floats, 1_048_576)
    assert 1.0 < snr < 10.0, snr


def test_ledger_keeps_both_views_separate() -> None:
    cfg = get_preset("tuan-1b")
    floats = stream_state_bytes(cfg, batch=1) // 4
    row = ledger(floats, 1_048_576, 32000)
    assert row.capacity_ratio > 1.0
    assert row.readout_snr < 10.0
    d = row.as_dict()
    assert d["assumptions"]
    assert d["required_bits"] == pytest.approx(1_048_576 * 2 * 14.966, rel=1e-3)


def test_snr_is_monotone_in_state_and_sequence() -> None:
    assert linear_readout_snr(8_000_000, 1_000_000) > linear_readout_snr(1_000_000, 1_000_000)
    assert linear_readout_snr(8_000_000, 2_000_000) < linear_readout_snr(8_000_000, 1_000_000)


def test_small_state_hits_interference_regime() -> None:
    """单层状态（2·M·D）对 1M token：SNR < 1，检索会被干扰淹没。"""
    cfg = get_preset("tuan-1b")
    per_layer = 2 * cfg.num_bases * cfg.dim  # y1/y2
    assert linear_readout_snr(per_layer, 1_048_576) < 1.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        capacity_ratio(100, 0, 32000)
    with pytest.raises(ValueError):
        linear_readout_snr(0, 1000)