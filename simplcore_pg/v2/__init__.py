# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""[实验性子包 · 非 SimplCore 主线架构 · 含滑动窗口注意力]

⚠️ 本子包是独立的实验性变体（无词表连续符号接口 GlyphStream + TriMemory 三级
记忆）。其中 ``swa`` (SlidingWindowAttention) 使用滑动窗口自注意力 + KV 缓存，
**属于 Transformer 路线**，与主线 ``simplcore_pg`` (v2.0 起完全无注意力：
LocalInteraction + ResonanceField + DynamicMemoryChannels + MoE) 不同。

主线架构已「完全放弃 Transformer/自注意力影响」，本 ``v2`` 子包仅作为历史
实验保留，**不计入主线架构、不参与主线可复现性链路**。新开发请在顶层
``simplcore_pg`` 模块进行。

SimplCore v2：无词表连续符号接口 (GlyphStream) + TriMemory 三级记忆。

子模块：
  * ``config_v2``  — v2 专用配置 (V2Config)；
  * ``deltanet``   — Gated DeltaNet 线性循环层 (L1，单步 O(1) 状态)；
  * ``swa``        — 滑动窗口注意力 (L0，KV 缓存恒定)；
  * ``octave``     — 八度分层压缩记忆 (L2，每翻倍 +1 个固定预算层)；
  * ``glyph``      — 无词表前端 (字形编码 + 熵驱动合并 + 原型解码)；
  * ``block_v2``   — TriMemoryBlock (L0/L1/L2 + FFN/MoE)；
  * ``model_v2``   — SimplCoreV2Model 完整无词表模型。

设计文档见仓库根目录 ``SimplCore_v2_无词表长上下文重构方案.md``。
"""
from __future__ import annotations

from .config_v2 import V2Config
from .deltanet import GatedDeltaNet, DeltaState
from .swa import SlidingWindowAttention, WindowKVCache
from .octave import OctaveMemory, OctaveBankState
from .glyph import GlyphStreamEncoder, EntropyDrivenPooler, PrototypeDecoder
from .block_v2 import TriMemoryBlock, TriMemoryState
from .model_v2 import SimplCoreV2Model

__all__ = [
    "V2Config",
    "GatedDeltaNet", "DeltaState",
    "SlidingWindowAttention", "WindowKVCache",
    "OctaveMemory", "OctaveBankState",
    "GlyphStreamEncoder", "EntropyDrivenPooler", "PrototypeDecoder",
    "TriMemoryBlock", "TriMemoryState",
    "SimplCoreV2Model",
]
