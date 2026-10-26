# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""TriMemoryBlock：L0 SWA + L1 DeltaNet + (可选) L2 Octave 读取 + FFN/MoE。

推理期状态容器 ``TriMemoryState`` 汇总整个模型的全部运行时状态：
  * 每层一个 ``WindowKVCache`` (容量恒定 W)；
  * 每层一个 ``DeltaState`` (dv×dk，恒定)；
  * 模型级共享一个 ``OctaveBankState`` (层数随 log L 增长)。

**单步解码显存与上下文长度无关** (前两项恒定)，总状态随 log L 增长
(第三项) —— 即设计文档的 M(L) = 常数 + log2(L/1M) × 单层预算。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..layers import RMSNorm, FFN
from ..moe import MoELayer
from .swa import SlidingWindowAttention, WindowKVCache
from .deltanet import GatedDeltaNet, DeltaState
from .octave import OctaveMemory, OctaveBankState


class TriMemoryState:
    """模型级推理状态 (恒定预算的 L0/L1 + 对数增长的 L2)。"""

    def __init__(self) -> None:
        self.kv_caches: List[WindowKVCache] = []
        self.delta_states: List[DeltaState] = []
        self.octave_bank: Optional[OctaveBankState] = None

    def nbytes(self) -> int:
        total = sum(c.nbytes() for c in self.kv_caches)
        total += sum(s.nbytes() for s in self.delta_states)
        if self.octave_bank is not None:
            total += self.octave_bank.nbytes()
        return total

    def reset(self) -> None:
        for c in self.kv_caches:
            c.reset()
        for s in self.delta_states:
            s.reset()
        if self.octave_bank is not None:
            self.octave_bank.reset()


class TriMemoryBlock(nn.Module):
    """TriMemory 计算块。

    Args:
        config: V2Config。
        has_octave: 本块是否插入八度记忆读取 (每 octave_every 层一次)。
        octave: 共享的 OctaveMemory 模块 (全模型单例，读写权重共享)。
    """

    def __init__(self, config, has_octave: bool,
                 octave: Optional[OctaveMemory] = None) -> None:
        super().__init__()
        self.config = config
        self.has_octave = has_octave
        H = config.hidden_size

        self.swa = SlidingWindowAttention(
            H, config.swa_num_heads, config.swa_num_kv_heads,
            config.swa_window, rope_base=config.rope_base, eps=config.rms_norm_eps,
        )
        self.delta = GatedDeltaNet(H, config.delta_num_heads, eps=config.rms_norm_eps)
        self.octave = octave if has_octave else None

        if config.num_experts > 1:
            inter = config.moe_intermediate_size or int(round(config.ffn_intermediate_ratio * H))
            self.moe = MoELayer(
                H, num_experts=config.num_experts, experts_per_tok=config.experts_per_tok,
                intermediate_size=inter, dropout=config.dropout, eps=config.rms_norm_eps,
            )
        else:
            self.moe = None
            self.ffn_norm = RMSNorm(H, eps=config.rms_norm_eps)
            self.ffn = FFN(H, intermediate_ratio=config.ffn_intermediate_ratio,
                           dropout=config.dropout)

    # ------------------------------------------------------------------
    def _ffn_path(self, x: torch.Tensor, training: bool = False):
        """FFN/MoE 分支。Returns: (output, aux 或 None)。"""
        if self.moe is not None:
            return self.moe(x, training=training)
        return x + self.ffn(self.ffn_norm(x)), None

    # ------------------------------------------------------------------
    # 训练前向 (可微; 八度记忆读自传入的 detached 快照)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
        octave_tokens: Optional[torch.Tensor] = None,
        octave_level_ids: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns: (output (B,L,H), aux_loss 或 None)。"""
        x = self.swa(x, pos_offset=pos_offset)
        x, _S = self.delta(x)
        if self.octave is not None:
            x = self.octave.read(x, memory_tokens=octave_tokens,
                                 level_ids=octave_level_ids)
        x, aux = self._ffn_path(x, training=training)
        return x, aux

    # ------------------------------------------------------------------
    # 推理前向 (运行时状态, 原地更新)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_runtime(
        self,
        x: torch.Tensor,
        kv: WindowKVCache,
        delta_state: DeltaState,
        bank: Optional[OctaveBankState],
    ) -> torch.Tensor:
        """x: (B, n, H) 新 token 段；各状态原地更新。"""
        x = self.swa.step(x, kv)
        x = self.delta.forward_runtime(x, delta_state)
        if self.octave is not None:
            x = self.octave.read(x, bank=bank)
        x, _ = self._ffn_path(x, training=False)
        return x

    # ------------------------------------------------------------------
    def init_layer_states(self, batch: int, device: torch.device,
                          dtype: torch.dtype = torch.float32):
        """为本层创建推理状态 (kv, delta)。"""
        kv = WindowKVCache(batch, self.config.swa_num_kv_heads,
                           self.config.swa_window,
                           self.config.hidden_size // self.config.swa_num_heads,
                           device, dtype)
        ds = self.delta.init_state(batch, device, dtype)
        return kv, ds

    def extra_repr(self) -> str:
        kind = "MoE" if self.moe is not None else "FFN"
        return f"octave={self.has_octave}, ffn={kind}"
