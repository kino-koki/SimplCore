# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""P0-1 / P0-3 回归守护：频率网格修复在主路径可用、默认路径有可审计警告。

上游待修清单（`docs/upstream_fixlist.md`）：

* P0-1 ``init_mode="linear"`` 在算子层已实现且有 10 项守护，但
  ``SimplCoreConfig.validate()`` 的历史白名单漏了它，导致
  ``SimplCoreConfig → SimplCoreModel`` 主路径根本启用不了修复。
* P0-3 默认 ``logfreq`` 被构造期数值秩守护显式豁免（只对
  hybrid/linear 跑 ``_assert_numerical_rank``），静默放行塌缩字典。

本文件锁住这两条修复，防止回退。
"""
from __future__ import annotations

import warnings

import pytest

from simplcore.core.model.config import SimplCoreConfig
from simplcore.core.model.model import SimplCoreModel
from simplcore.core.arch.sgc import _init_oscillator_grid


def _tiny_cfg(**kw):
    return SimplCoreConfig(vocab_size=32, dim=32, num_layers=1,
                           num_bases=8, local_kernel=4,
                           tie_embeddings=False, **kw)


def test_linear_reaches_model_mixer_via_config() -> None:
    """P0-1 核心回归：config → model 主路径能真正启用 linear 网格。"""
    cfg = _tiny_cfg(sgc_init_mode="linear")
    cfg.validate()                       # 旧版在这里抛 ValueError
    model = SimplCoreModel(cfg)
    mixer = model.blocks[0].mixer
    assert getattr(mixer, "init_mode", None) == "linear", (
        "sgc_init_mode='linear' 没有传到 mixer")


def test_linear_validate_accepts_all_four_modes() -> None:
    """白名单四项齐全，getattr 默认值与字段默认一致。"""
    for mode in ("legacy", "logfreq", "hybrid", "linear"):
        # legacy 会 warn（代数公共零点），但 validate 不应拒绝
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _tiny_cfg(sgc_init_mode=mode).validate()
    with pytest.raises(ValueError, match="sgc_init_mode"):
        _tiny_cfg(sgc_init_mode="bogus").validate()


def test_legacy_validate_warns_about_common_zero() -> None:
    """legacy 在 config 层提前说明「构造会被拒绝」，不给用户间接报错。"""
    with pytest.warns(RuntimeWarning, match="common zero"):
        _tiny_cfg(sgc_init_mode="legacy").validate()


def test_logfreq_grid_warns_about_rank_deficiency() -> None:
    """P0-3：默认 logfreq 不再静默放行 cond~1e16 / 秩缺失的字典。"""
    with pytest.warns(RuntimeWarning, match="rank-deficient"):
        _init_oscillator_grid(32, 8192.0, mode="logfreq")


def test_linear_grid_does_not_warn() -> None:
    """linear 过构造期数值质量门，不该触发塌缩警告。"""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _init_oscillator_grid(32, 8192.0, mode="linear")
    assert not [w for w in rec if "rank-deficient" in str(w.message)]
