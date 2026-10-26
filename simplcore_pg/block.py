# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCoreBlock：SimplCore 的核心计算单元（无注意力）。

将 局部交互层 (LocalInteraction)、共振场 (RFP)、动态记忆 (DMC)、拓扑路由
(TAR) 与 FFN 以 pre-norm + 残差方式串联。自 v2.0 起，**自注意力已彻底移除**，
局部依赖由 LocalInteraction（因果卷积 + 线性递归）承担。每个子层内部已含归一化
与残差，Block 仅需按顺序调用，保证梯度畅通、训练稳定。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .localmix import LocalInteraction
from .resonance import ResonanceField
from .memory import DynamicMemoryChannels, MemoryState
from .routing import TopologyAwareRouter
from .layers import FFN, RMSNorm


class SimplCoreBlock(nn.Module):
    """SimplCore 计算块（无注意力）。"""

    def __init__(self, config) -> None:
        super().__init__()
        H = config.hidden_size
        # 局部交互层（替代自注意力）：因果卷积 + 可选线性递归
        self.localmix = LocalInteraction(
            H, kernel_size=config.local_kernel_size, dilation=config.local_dilation,
            use_conv=getattr(config, "local_use_conv", True),
            use_recurrence=config.local_use_recurrence, eps=config.rms_norm_eps,
            max_seq_len=config.max_sequence_length,
        ) if getattr(config, "use_localmix", True) else None
        self.rfp = ResonanceField(
            H, num_waves=config.num_waves, num_steps=config.rfp_num_steps,
            damping=config.rfp_damping, dt=config.rfp_dt, sigma=config.rfp_sigma,
            eps=config.rms_norm_eps, max_seq_len=config.max_sequence_length,
            freeze_evolution=config.rfp_freeze_evolution,
            use_checkpoint=config.rfp_use_checkpoint,
        ) if getattr(config, "use_rfp", True) else None
        self.dmc = DynamicMemoryChannels(
            H, num_channels=config.num_channels, channel_size=config.channel_size,
            num_heads=config.dmc_num_heads, eps=config.rms_norm_eps,
        ) if getattr(config, "use_dmc", True) else None
        self.tar = TopologyAwareRouter(
            H, num_paths=config.num_paths, topk=config.tar_topk,
            dropout=config.tar_dropout, eps=config.rms_norm_eps,
        )
        self.ffn_norm = RMSNorm(H, eps=config.rms_norm_eps)
        self.ffn = FFN(H, intermediate_ratio=config.ffn_intermediate_ratio, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        memory: Optional[MemoryState] = None,
        training: bool = False,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        if self.localmix is not None:
            x = self.localmix(x)
        if self.rfp is not None:
            x = self.rfp(x, seq_len=seq_len)
        if self.dmc is not None:
            x = self.dmc(x, runtime=memory)
        x = self.tar(x, training=training)
        x = x + self.ffn(self.ffn_norm(x))
        return x
