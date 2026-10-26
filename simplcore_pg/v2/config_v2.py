# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore v2 专用配置 (V2Config)。

与 v1 ``SimplCoreConfig`` 解耦：v2 不再有 vocab_size / max_sequence_length
(无词表、上下文长度理论无界)，代之以 TriMemory 三级记忆与 GlyphStream 参数。

显存公式 (推理)：
    M(L) = swa_kv (恒定) + delta_state (恒定) + octave_levels(L) × level_budget
    octave_levels(L) = ceil(log2(L / level0_coverage)) + 1  → 每翻倍 +1 层
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class V2Config:
    """TriMemory + GlyphStream 全部超参数。"""

    # ---- 推理核 ----
    hidden_size: int = 256
    num_layers: int = 4

    # ---- L0 滑动窗口注意力 (SWA) ----
    swa_num_heads: int = 4
    swa_num_kv_heads: int = 2          # GQA
    swa_window: int = 512              # 生产目标 8192；测试用小窗口
    rope_base: float = 10_000.0

    # ---- L1 Gated DeltaNet ----
    delta_num_heads: int = 4

    # ---- L2 Octave Memory ----
    octave_every: int = 2              # 每 k 个 block 插入一次八度记忆读取
    octave_slots_per_level: int = 64   # 每层固定槽数 (生产 4096)
    octave_segment_tokens: int = 32    # 每多少 token 压缩一次 (生产 4096)
    octave_mem_tokens: int = 4         # 每段压成的 memory token 数 (生产 32)
    octave_promote_tokens: int = 4     # 层满向上层推入的 token 数 (生产 32)
    octave_max_levels: int = 16
    octave_num_heads: int = 4

    # ---- FFN / MoE ----
    ffn_intermediate_ratio: float = 4.0
    num_experts: int = 1               # 1 = 稠密 FFN；>1 启用 MoE (复用 v1 MoELayer)
    experts_per_tok: int = 2
    moe_intermediate_size: int = 0

    # ---- GlyphStream 前端 ----
    glyph_size: int = 32
    glyph_in_channels: int = 1
    glyph_patch_size: int = 8
    glyph_hidden: int = 128
    glyph_encoder_layers: int = 2
    glyph_encoder_heads: int = 4
    pool_boundary_threshold: float = 0.5   # 熵驱动合并的边界概率阈值
    max_chars_per_unit: int = 8            # 单个语义单元最多合并字符数

    # ---- 原型解码 ----
    num_prototypes: int = 256          # 可学习字形原型数 (生产 ~6000)

    # ---- 通用 ----
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        H = self.hidden_size
        checks = [
            (H > 0, "hidden_size 必须为正"),
            (self.num_layers > 0, "num_layers 必须为正"),
            (H % self.swa_num_heads == 0, "hidden_size 必须能被 swa_num_heads 整除"),
            (H % self.delta_num_heads == 0, "hidden_size 必须能被 delta_num_heads 整除"),
            (H % self.octave_num_heads == 0, "hidden_size 必须能被 octave_num_heads 整除"),
            (self.swa_num_heads % self.swa_num_kv_heads == 0,
             "swa_num_heads 必须能被 swa_num_kv_heads 整除 (GQA)"),
            (self.swa_window > 0, "swa_window 必须为正"),
            (self.octave_every > 0, "octave_every 必须为正"),
            (self.octave_slots_per_level >= self.octave_mem_tokens,
             "octave_slots_per_level 不能小于 octave_mem_tokens"),
            (self.octave_slots_per_level % self.octave_mem_tokens == 0,
             "octave_slots_per_level 必须能被 octave_mem_tokens 整除"),
            (self.octave_segment_tokens > 0, "octave_segment_tokens 必须为正"),
            (0 < self.octave_promote_tokens <= self.octave_slots_per_level,
             "octave_promote_tokens 必须在 (0, slots] 内"),
            (self.glyph_size % self.glyph_patch_size == 0,
             "glyph_size 必须能被 glyph_patch_size 整除"),
            (self.glyph_hidden % self.glyph_encoder_heads == 0,
             "glyph_hidden 必须能被 glyph_encoder_heads 整除"),
            (self.num_prototypes > 0, "num_prototypes 必须为正"),
            (self.num_experts >= 1, "num_experts 必须 >= 1"),
            (self.num_experts == 1 or 1 <= self.experts_per_tok <= self.num_experts,
             "MoE 启用时 experts_per_tok 越界"),
            (0.0 <= self.dropout < 1.0, "dropout 必须在 [0,1)"),
            (0.0 < self.pool_boundary_threshold < 1.0, "pool 阈值必须在 (0,1)"),
            (self.max_chars_per_unit >= 1, "max_chars_per_unit 必须 >= 1"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(f"V2Config 非法: {msg}。")

    # ------------------------------------------------------------------
    # 显存账目 (字节)
    # ------------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.swa_num_heads

    def swa_kv_bytes(self, dtype_bytes: int = 2) -> int:
        """L0 KV 缓存恒定大小：W × 2(K,V) × layers × kv_heads × head_dim。"""
        return (self.swa_window * 2 * self.num_layers
                * self.swa_num_kv_heads * self.head_dim * dtype_bytes)

    def delta_state_bytes(self, dtype_bytes: int = 2) -> int:
        """L1 循环状态恒定大小：layers × heads × (dv × dk)。"""
        d = self.hidden_size // self.delta_num_heads
        return self.num_layers * self.delta_num_heads * d * d * dtype_bytes

    def octave_level_bytes(self, dtype_bytes: int = 1) -> int:
        """L2 单层预算 (FP8 存储)：slots × H × 读取层数(每 octave_every 层共享一份)。"""
        n_read_layers = math.ceil(self.num_layers / self.octave_every)
        return self.octave_slots_per_level * self.hidden_size * dtype_bytes * n_read_layers

    def octave_coverage_tokens(self, level: int) -> int:
        """第 level 层一个 memory token 覆盖的原始 token 数 (指数增长)。"""
        seg = self.octave_segment_tokens
        fan = self.octave_slots_per_level // self.octave_promote_tokens
        return seg * (fan ** level)

    def octave_levels_for(self, context_tokens: int) -> int:
        """覆盖 context_tokens 所需的八度层数 (对数增长)。"""
        l0 = self.octave_slots_per_level // self.octave_mem_tokens * self.octave_segment_tokens
        if context_tokens <= l0:
            return 1
        fan = self.octave_slots_per_level // self.octave_promote_tokens
        return min(self.octave_max_levels,
                   1 + math.ceil(math.log(context_tokens / l0, fan)))

    def total_inference_memory(self, context_tokens: int, dtype_bytes: int = 2) -> dict:
        """给定上下文长度的推理显存账目 (不含模型权重)。"""
        lv = self.octave_levels_for(context_tokens)
        swa = self.swa_kv_bytes(dtype_bytes)
        delta = self.delta_state_bytes(dtype_bytes)
        octb = lv * self.octave_level_bytes(dtype_bytes=1)  # FP8
        return {
            "context_tokens": context_tokens,
            "swa_kv_bytes": swa,
            "delta_state_bytes": delta,
            "octave_levels": lv,
            "octave_bytes": octb,
            "total_bytes": swa + delta + octb,
        }
