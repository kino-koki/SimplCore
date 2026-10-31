# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""部署侧 HBM 账本守护测试（纯解析，秒级）。

守护用户硬指标：
1. 1M 上下文完整放下（含权重）≤ 2 GB；
2. 上下文翻倍（1M→2M→4M→8M）增量 ≤ 1 GB —— 且结构性为 0；
3. 增量为 0 的来源是：四项占用（权重/流式状态/chunk workspace/logits）全部与 L 无关。
"""
import pytest

from simplcore.core.config import get_preset
from simplcore.core.deploy_budget import (
    DEPLOY_DOUBLING_CAP,
    DEPLOY_HBM_CAP_1M,
    DEPLOY_LADDER,
    deploy_bytes,
    deploy_verdict,
    deployment_report,
    weight_bytes,
)
from simplcore.core.scaling import embedding_params, non_embedding_params


@pytest.fixture(scope="module")
def cfg():
    return get_preset("tuan-1b")


@pytest.fixture(scope="module")
def rows(cfg):
    return deployment_report(cfg)


@pytest.fixture(scope="module")
def verdict(rows):
    return deploy_verdict(rows)


def test_1m_fits_in_2gb(cfg, rows, verdict):
    """1M 上下文完整放下 ≤ 2 GiB（含 1B bf16 权重 ~1.89 GiB）。"""
    assert verdict["fits_2gb_at_1m"] is True
    assert rows[0].total_bytes <= DEPLOY_HBM_CAP_1M
    # 权重是预算大头，必须 < 2GB
    assert rows[0].weights_bytes < DEPLOY_HBM_CAP_1M


def test_doubling_delta_within_1gb_and_structurally_zero(rows, verdict):
    """翻倍增量 ≤ 1 GB，且四项占用与 L 无关 → 增量恒为 0。"""
    assert verdict["doubling_cap_ok"] is True
    assert verdict["max_doubling_delta_gib"] == 0.0
    for r in rows[1:]:
        assert r.delta_bytes == 0


def test_total_invariant_across_ladder(cfg):
    """1M 与 8M 的部署占用逐字节相同 —— deploy_bytes 根本不接受 L 参数。"""
    one_m = deployment_report(cfg)[0].total_bytes
    eight_m = deployment_report(cfg)[-1].total_bytes
    assert one_m == eight_m


def test_claim_holds(verdict):
    assert verdict["claim_holds"] is True


def test_workspace_scales_with_chunk_and_batch(cfg):
    """调参如实入账：batch/chunk 变大 → 占用线性抬高（但与 L 依旧无关）。"""
    base = deploy_bytes(cfg, batch=1, chunk=512)
    assert deploy_bytes(cfg, batch=1, chunk=1024) > base
    assert deploy_bytes(cfg, batch=2, chunk=512) > base
    # batch=2 权重不变、激活翻倍，可能超 2GB —— 但与 L 无关的不变式必须保持
    r2 = deployment_report(cfg, batch=2)
    assert r2[0].total_bytes == r2[-1].total_bytes
    assert all(r.delta_bytes == 0 for r in r2[1:])


def test_weight_accounting(cfg):
    params = embedding_params(cfg) + non_embedding_params(cfg)
    assert weight_bytes(params) == params * 2  # bf16
    assert 0.9e9 <= params <= 1.2e9  # 1B 口径


def test_ladder_starts_at_1m():
    """阶梯必须从 1M 起（用户口径："是否真正 1M 起"）。"""
    assert DEPLOY_LADDER[0] == 1024 * 1024
    assert DEPLOY_LADDER[-1] == 8 * 1024 * 1024
