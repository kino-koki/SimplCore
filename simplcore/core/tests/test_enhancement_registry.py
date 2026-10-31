# -*- coding: utf-8 -*-
"""训练增强编排的契约测试。

守护三件事：
1. 未知增强名必须报错（旧实现静默忽略 → "以为开了，其实逐位不变"）；
2. 同位置 wrapper 的替代关系必须显式记录（不再静默 last-wins）；
3. ``bio`` 消费 ``unity`` 的合法组合不能被误判成替代。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.training_integration import (
    ENHANCEMENT_SPECS,
    build_enhanced_model,
    resolve_enhancements,
)


def _cfg() -> SimplCoreConfig:
    return SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=4,
                           ffn_ratio=2.0)


# ---------------------------------------------------------------------------
# 1. 输入解析
# ---------------------------------------------------------------------------
def test_unknown_enhancement_raises() -> None:
    with pytest.raises(ValueError, match="unknown enhancement"):
        resolve_enhancements(["pondeer"])


def test_empty_is_noop() -> None:
    active, superseded, consumed = resolve_enhancements([])
    assert active == [] and superseded == {} and consumed == []


def test_eval_and_post_keep_declaration_order() -> None:
    active, _, _ = resolve_enhancements(["dream", "accel", "smm", "reasoning"])
    assert active == ["accel", "smm", "dream", "reasoning"]


# ---------------------------------------------------------------------------
# 2. wrapper 替代关系显式化
# ---------------------------------------------------------------------------
def test_envlearn_supersedes_ponder() -> None:
    active, superseded, _ = resolve_enhancements(["ponder", "envlearn"])
    assert active == ["envlearn"]
    assert superseded == {"ponder": "envlearn"}


def test_unity_supersedes_envlearn() -> None:
    active, superseded, _ = resolve_enhancements(["envlearn", "unity"])
    assert active == ["unity"]
    assert superseded == {"envlearn": "unity"}


def test_bio_consumes_unity_instead_of_superseding_it() -> None:
    active, superseded, consumed = resolve_enhancements(["unity", "bio"])
    assert active == ["unity", "bio"]
    assert superseded == {}
    assert consumed == ["unity"]


def test_full_chain_records_every_replacement() -> None:
    active, superseded, consumed = resolve_enhancements(
        ["ponder", "envlearn", "unity", "bio"])
    assert active == ["unity", "bio"]
    assert superseded == {"ponder": "envlearn", "envlearn": "unity"}
    assert consumed == ["unity"]


# ---------------------------------------------------------------------------
# 3. 构建结果与 spec 一致
# ---------------------------------------------------------------------------
def test_default_off_leaves_base_model_bit_identical() -> None:
    base = SimplCoreModel(_cfg())
    em = build_enhanced_model(base, _cfg(), [])
    assert em.model is base
    assert em.active == [] and em.superseded == {}
    x = torch.randint(0, 64, (2, 8))
    _, loss = em.train_forward(x, x)
    assert torch.isfinite(loss)


def test_registry_covers_every_spec_and_has_no_orphan() -> None:
    from simplcore.core.training_integration import _BUILDERS

    assert set(ENHANCEMENT_SPECS) == set(_BUILDERS)
    for name, spec in ENHANCEMENT_SPECS.items():
        assert spec.name == name
        assert spec.kind in {"wrapper", "eval", "post"}


def test_build_records_superseded_in_result() -> None:
    base = SimplCoreModel(_cfg())
    em = build_enhanced_model(base, _cfg(), ["ponder", "envlearn"])
    assert em.active == ["envlearn"]
    assert em.superseded == {"ponder": "envlearn"}
    assert "envlearn" in em.extras and "ponder" not in em.extras
    assert em.describe()["superseded"] == {"ponder": "envlearn"}