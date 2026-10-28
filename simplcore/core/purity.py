# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""架构纯洁性守卫 —— 把"脱离 Transformer"变成**可测试**的硬约束。

为什么需要它
------------
"这套架构不属于 Transformer 框架"如果只写在文档里，就会在后续迭代中悄悄失效：
某次为了追指标又把 ``mixer`` 设成注意力族、或新加一个 ``*Attention*`` 模块，
注意力就回来了，而 loss 曲线完全正常。本模块提供两道断言，让 CI 在合并前
直接失败：

* :func:`assert_pure_config` —— 在建模前挡掉会引入注意力的配置开关；
* :func:`assert_attention_free` —— 在建模后遍历 ``named_modules()``，
  确认实例化出来的模型里真的没有任何 ``*Attention*`` / ``*attn*`` 模块。

当前状态
--------
SimplCore 已**彻底移除**所有注意力模块（``CausalAttention`` / ``ReasoningAttention`` /
``SlidingWindowAttention`` / GQA-YaRN 注意力 / RoPE 等均已删除，见 ``.trash/attn_removed/``）。
``SimplCoreConfig.validate()`` 现在只允许 ``sgc`` / ``sgc-nogate`` / ``sgc-nolocal`` / ``srf``
四种非注意力 mixer。本守卫作为**回归防火墙**保留：万一将来有人重新引入注意力
模块或配置开关，CI 会立刻失败，而不是等 loss 曲线骗过所有人。

判定口径
--------
SGC（谱全局卷积）、SRF（谱振流）、SSM（选择性状态空间）、GatedSpectralConv2
（GLA/Mamba2 风格通道门，非 softmax 注意力）都通过；``attn`` / ``hybrid*``
族以及任何类名含 ``attention`` / ``attn`` 的 ``nn.Module`` 都会被抓出来。
``SimplCoreConfig.validate()`` 的 mixer 白名单来自 ``registry.mixer_kinds``，
``assert_registry_pure()`` 额外保证注册表里登记的实现类本身不含注意力。

用法::

    from simplcore.core.purity import assert_pure_config, assert_attention_free
    assert_pure_config(cfg)          # 先挡配置
    model = SimplCoreModel(cfg)
    assert_attention_free(model)     # 再挡实例
"""
from __future__ import annotations

from typing import List, Tuple

__all__ = [
    "ATTENTION_MIXERS",
    "attention_like_modules",
    "assert_attention_free",
    "assert_pure_config",
    "assert_registry_pure",
]

#: 会引入注意力子模块的 mixer 取值。
ATTENTION_MIXERS = frozenset({
    "attn", "hybrid", "hybrid-gqa", "hybrid-yarn", "hybrid-gqa-yarn",
})

#: 模块类名里出现任一子串即视为注意力模块。
_ATTENTION_TOKENS = ("attention", "attn")


def attention_like_modules(model) -> List[Tuple[str, str]]:
    """返回 ``[(模块名, 类名), ...]``；空列表表示模型无注意力。"""
    hits: List[Tuple[str, str]] = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        low = cls.lower()
        if any(tok in low for tok in _ATTENTION_TOKENS):
            hits.append((name or "<root>", cls))
    return hits


def assert_attention_free(model) -> bool:
    """断言模型不含任何注意力模块；否则抛 :class:`AssertionError`。"""
    hits = attention_like_modules(model)
    if hits:
        listing = ", ".join(f"{n}:{c}" for n, c in hits[:8])
        more = "" if len(hits) <= 8 else f" 等 {len(hits)} 个"
        raise AssertionError(
            f"模型包含注意力模块，违反“脱离 Transformer”约束：{listing}{more}"
        )
    return True


def assert_pure_config(config) -> bool:
    """断言配置不会引入注意力；否则抛 :class:`AssertionError`。"""
    problems: List[str] = []
    mixer = getattr(config, "mixer", "sgc")
    if mixer in ATTENTION_MIXERS:
        problems.append(f"mixer={mixer!r}")
    if getattr(config, "attn_every", None) is not None:
        problems.append(f"attn_every={config.attn_every}")
    if getattr(config, "reason_attn_every", None) is not None:
        problems.append(f"reason_attn_every={config.reason_attn_every}")
    if problems:
        raise AssertionError(
            "配置会引入注意力，违反纯洁性：" + ", ".join(problems)
        )
    return True


def assert_registry_pure() -> bool:
    """断言 mixer 注册表里登记的**实现类**全部不是注意力模块.

    ``validate()`` 的白名单来自注册表（``registry.mixer_kinds``），所以
    "往注册表里塞一个注意力实现" 等于给注意力开后门。本断言遍历注册表
    的类路径做类名检查，与 :func:`assert_attention_free` 同一判定口径，
    在 CI 里堵住这条后门。
    """
    from .registry import iter_mixer_classes
    hits: List[str] = []
    for name, cls in iter_mixer_classes():
        low = cls.__name__.lower()
        if any(tok in low for tok in _ATTENTION_TOKENS):
            hits.append(f"{name}:{cls.__name__}")
    if hits:
        raise AssertionError(
            "mixer 注册表包含注意力实现，违反纯洁性：" + ", ".join(hits)
        )
    return True
