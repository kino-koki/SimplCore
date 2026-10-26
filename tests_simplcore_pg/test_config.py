# -*- coding: utf-8 -*-
"""SimplCoreConfig 测试。"""
import pytest

from simplcore_pg.config import SimplCoreConfig


def test_default_config_valid():
    cfg = SimplCoreConfig()
    assert cfg.hidden_size == 768
    assert cfg.num_layers == 12


@pytest.mark.parametrize("patch", [
    {"vocab_size": 0},
    {"hidden_size": -1},
    {"min_depth": 10, "max_depth": 4},
    {"tar_topk": 10, "num_paths": 4},
    {"hidden_size": 64, "dmc_num_heads": 5},
    {"dropout": 1.5},
    {"tar_dropout": -0.1},
    {"rfp_dt": 0.0},
    {"rms_norm_eps": 0.0},
    {"modality_dims": (768, 768), "num_modalities": 3},
    {"pad_token_id": 999999, "vocab_size": 1000},
])
def test_invalid_configs_raise(patch):
    with pytest.raises((ValueError, TypeError)):
        SimplCoreConfig(**patch)


def test_presets():
    for name in ["nano", "small", "150m", "7b", "14b", "72b", "tiny"]:
        cfg = SimplCoreConfig.from_preset(name)
        cfg.validate()


def test_unknown_preset():
    with pytest.raises(ValueError):
        SimplCoreConfig.from_preset("999b")


def test_estimate_parameters_sum():
    cfg = SimplCoreConfig()
    est = cfg.estimate_parameters()
    # total 必须等于所有组件参数量之和 (含 attn 等可选组件)
    components = sum(v for k, v in est.items() if k != "total")
    assert est["total"] == components
    assert est["total"] > 0


def test_compat_accessors():
    cfg = SimplCoreConfig.get_nano_config()
    assert cfg.max_seq_len == cfg.max_sequence_length
    assert cfg.layer_norm_eps == cfg.rms_norm_eps
    assert cfg.num_heads == cfg.dmc_num_heads
    assert cfg.num_rollout_steps == 0  # use_rollout=False


def test_as_dict_serializable():
    cfg = SimplCoreConfig.get_nano_config()
    d = cfg.as_dict()
    assert isinstance(d["modality_dims"], list)
    assert d["vocab_size"] == cfg.vocab_size
