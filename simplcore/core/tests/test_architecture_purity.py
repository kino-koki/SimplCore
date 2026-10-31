# -*- coding: utf-8 -*-
"""架构纯洁性守卫的测试：默认 SimplCore 必须无注意力，注意力开关必须被拒。

注意：SimplCore 已彻底移除所有注意力模块（CausalAttention / ReasoningAttention /
SlidingWindowAttention / GQA-YaRN / RoPE 等，见 ``.trash/attn_removed/``），
``SimplCoreConfig.validate()`` 现在只允许 ``sgc`` / ``sgc-nogate`` / ``sgc-nolocal`` /
``srf`` 四种非注意力 mixer。本测试守卫两道防线：
  1. ``assert_pure_config`` —— 在建模前挡掉会引入注意力的配置；
  2. ``assert_attention_free`` + ``SimplCoreModel(cfg).validate()`` —— 建模时若仍有
     注意力苗头，构造直接抛错。
"""
from __future__ import annotations

import types

import pytest

from simplcore.core.config import SimplCoreConfig, get_preset
from simplcore.core.model import SimplCoreModel
from simplcore.core.purity import (
    assert_attention_free,
    assert_pure_config,
    assert_registry_pure,
    attention_like_modules,
)


def _tiny(**over) -> SimplCoreConfig:
    kw = dict(vocab_size=64, dim=32, num_layers=2, num_heads=2,
              num_bases=4, ffn_ratio=2.0)
    kw.update(over)
    return SimplCoreConfig(**kw)


def _fake(**over) -> object:
    """构造一个只带指定属性的轻量对象，用于直接测 purity 守卫（绕过 config 校验）。"""
    ns = types.SimpleNamespace()
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# 1. 干净配置必须放行
# ---------------------------------------------------------------------------
def test_default_sgc_is_attention_free() -> None:
    cfg = _tiny()
    assert_pure_config(cfg)
    model = SimplCoreModel(cfg)
    assert attention_like_modules(model) == []
    assert assert_attention_free(model) is True


def test_sgc_1b_preset_is_pure() -> None:
    cfg = get_preset("tuan-1b")
    assert_pure_config(cfg)  # 只查配置，不实例化 1B


def test_srf_is_attention_free() -> None:
    cfg = _tiny(mixer="srf", srf_heads=2, srf_chunk=8)
    assert_pure_config(cfg)
    assert assert_attention_free(SimplCoreModel(cfg)) is True


def test_gated_spectral_conv2_is_attention_free() -> None:
    """GLA/Mamba2 风格通道门（GatedSpectralConv2）非 softmax 注意力，应放行。"""
    cfg = _tiny(mixer="sgc-nogate", num_heads=2)
    # 该 mixer 的实际多头变体由 selective.py 提供；这里只确认纯 SGC 变体通过守卫。
    assert_pure_config(cfg)
    assert assert_attention_free(SimplCoreModel(cfg)) is True


# ---------------------------------------------------------------------------
# 2. 注意力开关必须被守卫抓住
# ---------------------------------------------------------------------------
def test_purity_guard_rejects_attn_mixer_string() -> None:
    with pytest.raises(AssertionError):
        assert_pure_config(_fake(mixer="attn"))


def test_purity_guard_rejects_hybrid_mixer_string() -> None:
    with pytest.raises(AssertionError):
        assert_pure_config(_fake(mixer="hybrid"))


def test_purity_guard_rejects_attn_every_switch() -> None:
    with pytest.raises(AssertionError):
        assert_pure_config(_fake(mixer="sgc", attn_every=2))


def test_purity_guard_rejects_reason_attn_switch() -> None:
    with pytest.raises(AssertionError):
        assert_pure_config(_fake(mixer="sgc", reason_attn_every=1))


# ---------------------------------------------------------------------------
# 3. 第二道防线：SimplCoreModel 构造时 validate() 直接拒绝注意力配置
# ---------------------------------------------------------------------------
def test_model_rejects_attn_mixer_at_construction() -> None:
    """config.validate() 只允许非注意力 mixer，传 attn 应在建模时抛 ValueError。"""
    cfg = _tiny(mixer="attn")
    with pytest.raises(AssertionError):
        assert_pure_config(cfg)  # 守卫先抓
    with pytest.raises(ValueError):
        SimplCoreModel(cfg)


def test_model_rejects_hybrid_mixer_at_construction() -> None:
    cfg = _tiny(mixer="hybrid")
    with pytest.raises(ValueError):
        SimplCoreModel(cfg)


# ---------------------------------------------------------------------------
# 4. mixer 注册表：单一事实源的一致性与纯度
# ---------------------------------------------------------------------------
def test_registry_is_pure() -> None:
    """注册表里登记的实现类必须全部通过注意力类名检查."""
    assert assert_registry_pure() is True


def test_validate_whitelist_matches_registry() -> None:
    """validate() 的白名单与注册表必须同源（防两处漂移）."""
    from simplcore.core.registry import mixer_kinds
    for kind in mixer_kinds():
        cfg = _tiny(mixer=kind, srf_heads=2, srf_chunk=8)
        cfg.validate()                    # 注册表内合法取值必须全部放行
    with pytest.raises(ValueError):
        _tiny(mixer="attn").validate()    # 注册表外取值必须被拒


def test_registry_class_paths_cover_all_kinds() -> None:
    """purity 用的类路径表必须覆盖注册表全部条目（新增 mixer 漏登记会被抓）."""
    from simplcore.core.registry import _MIXER_CLASS_PATHS, mixer_kinds
    assert set(_MIXER_CLASS_PATHS) == set(mixer_kinds())


def test_every_registered_mixer_builds_and_streams() -> None:
    """注册表每种 mixer 都必须能构造出模型并支持 O(1) 流式."""
    for kind in ("sgc", "sgc-nogate", "sgc-nolocal", "srf"):
        cfg = _tiny(mixer=kind, srf_heads=2, srf_chunk=8)
        m = SimplCoreModel(cfg)
        assert m.supports_streaming, f"mixer {kind!r} 不支持 O(1) 流式"

# ---------------------------------------------------------------------------
# 5. 源码级守卫：改名不能绕开"无注意力"
# ---------------------------------------------------------------------------
def test_source_scan_is_clean_on_product_line() -> None:
    from simplcore.core.purity import assert_source_attention_free
    assert assert_source_attention_free() is True


def test_source_scan_catches_renamed_attention_module(tmp_path) -> None:
    """把类名改成 MixerA、但保留 q_proj/k_proj/v_proj 的实现必须被抓。"""
    from simplcore.core.purity import scan_source_attention_free

    (tmp_path / "sneaky.py").write_text(
        "import torch.nn as nn\n"
        "class MixerA(nn.Module):\n"
        "    def __init__(self, d):\n"
        "        super().__init__()\n"
        "        self.q_proj = nn.Linear(d, d)\n"
        "        self.k_proj = nn.Linear(d, d)\n"
        "        self.v_proj = nn.Linear(d, d)\n",
        encoding="utf-8")
    hits = scan_source_attention_free(tmp_path)
    idents = {h[2] for h in hits}
    assert {"q_proj", "k_proj", "v_proj"} <= idents


def test_source_scan_catches_attention_api_in_forward(tmp_path) -> None:
    from simplcore.core.purity import scan_source_attention_free

    (tmp_path / "sneaky.py").write_text(
        "import torch.nn.functional as F\n"
        "class MixerB:\n"
        "    def forward(self, q, k, v):\n"
        "        return F.scaled_dot_product_attention(q, k, v)\n",
        encoding="utf-8")
    idents = {h[2] for h in scan_source_attention_free(tmp_path)}
    assert "scaled_dot_product_attention" in idents


def test_source_scan_ignores_auditor_names(tmp_path) -> None:
    """守卫自身的函数名引用 attention 是为了检查它不存在，不能算实现。"""
    from simplcore.core.purity import scan_source_attention_free

    (tmp_path / "audit.py").write_text(
        "def assert_attention_free(model):\n"
        "    return True\n"
        "def _check_no_attention(adapter):\n"
        "    return True\n",
        encoding="utf-8")
    assert scan_source_attention_free(tmp_path) == []
