# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""动态记忆通道 (DynamicMemoryChannels, DMC)。

以可学习的记忆库 (memory bank) 作为外部记忆，通过跨注意力从隐状态读取内容；
并提供运行时 ``MemoryState`` 用于自回归推演时的写入 (detached)，保持训练期
完全可微、无状态。

设计：
  * 记忆库形状 (num_channels, channel_size, head_dim)，head_dim = H // dmc_num_heads；
  * 读取：query(x) 对展平的 (C*S, head_dim) 记忆做缩放点积注意力；
  * 写入：推演时将新槽位写入运行时记忆 (不进入计算图)，按 LRU 覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .layers import RMSNorm
from .exceptions import MemoryStateError


class MemoryState:
    """DMC 运行时记忆 (自回归推演使用，不进入训练计算图)。

    维护一个可写的记忆库副本与写指针，提供 LRU 覆盖策略。
    """

    def __init__(self, num_channels: int, channel_size: int, head_dim: int, device: torch.device) -> None:
        self.num_channels = num_channels
        self.channel_size = channel_size
        self.head_dim = head_dim
        self.slots = num_channels * channel_size
        self.bank = torch.zeros(self.slots, head_dim, device=device)
        self.write_ptr = 0

    def write(self, values: torch.Tensor) -> None:
        """写入若干记忆槽 (values: (k, head_dim))，按 LRU 覆盖。"""
        k = values.shape[0]
        if values.shape[1] != self.head_dim:
            raise MemoryStateError("写入维度与 head_dim 不符。",
                                   details={"got": tuple(values.shape), "head_dim": self.head_dim})
        for i in range(k):
            self.bank[self.write_ptr] = values[i].detach()
            self.write_ptr = (self.write_ptr + 1) % self.slots

    def reset(self) -> None:
        self.bank.zero_()
        self.write_ptr = 0

    def as_tensor(self) -> torch.Tensor:
        return self.bank


@dataclass
class _MemoryShapes:
    num_channels: int
    channel_size: int
    head_dim: int


class DynamicMemoryChannels(nn.Module):
    """动态记忆通道层。"""

    def __init__(
        self,
        hidden_size: int,
        num_channels: int = 64,
        channel_size: int = 128,
        num_heads: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除。")
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.channel_size = channel_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.slots = num_channels * channel_size

        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.output_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.memory_bank = nn.Parameter(
            torch.zeros(num_channels, channel_size, self.head_dim)
        )
        self.norm = RMSNorm(hidden_size, eps=eps)
        self._init_memory()

    def _init_memory(self) -> None:
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.02)

    def _bank_for_read(self, runtime: Optional[MemoryState]) -> torch.Tensor:
        if runtime is None:
            bank = self.memory_bank  # (C, S, hd)
        else:
            if runtime.slots != self.slots or runtime.head_dim != self.head_dim:
                raise MemoryStateError("运行时记忆形状与本层不匹配。")
            bank = runtime.as_tensor().view(self.num_channels, self.channel_size, self.head_dim)
        return bank

    def read(self, x: torch.Tensor, runtime: Optional[MemoryState] = None) -> torch.Tensor:
        """从记忆库读取内容并残差融合。

        Args:
            x: (B, L, H) 隐状态。
            runtime: 可选运行时记忆 (推演期)。

        Returns:
            (B, L, H) 记忆增强后的隐状态。
        """
        torch._assert(x.dim() == 3, "DMC 期望 3D 输入 (B, L, H)")
        B, L, H = x.shape
        q = self.query_proj(x)  # (B, L, H)

        bank = self._bank_for_read(runtime)  # (C, S, hd)
        mem = bank.reshape(self.slots, self.head_dim)  # (C*S, hd)
        k = self.key_proj(mem)     # (slots, hd)
        v = self.value_proj(mem)   # (slots, hd)

        # 缩放点积注意力 (无 mask)
        scale = self.head_dim ** -0.5
        # q: (B*L, H) -> reshape heads
        qh = q.view(B * L, self.num_heads, self.head_dim).transpose(0, 1)  # (heads, B*L, hd)
        kh = k.view(1, self.slots, self.head_dim).expand(self.num_heads, -1, -1)  # (heads, slots, hd)
        vh = v.view(1, self.slots, self.head_dim).expand(self.num_heads, -1, -1)

        scores = torch.einsum("hbl,hsd->hbs", qh, kh) * scale  # (heads, B*L, slots)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hbs,hsd->hbd", attn, vh)  # (heads, B*L, hd)
        out = out.transpose(0, 1).reshape(B * L, H)
        out = self.output_proj(out).view(B, L, H)
        return self.norm(x + out)

    def forward(self, x: torch.Tensor, runtime: Optional[MemoryState] = None) -> torch.Tensor:
        return self.read(x, runtime=runtime)

    def _make_runtime(self, batch: int, device: torch.device) -> MemoryState:
        """为流式推演创建运行时记忆 (固定槽位，与序列长度无关)。"""
        return MemoryState(
            num_channels=self.num_channels, channel_size=self.channel_size,
            head_dim=self.head_dim, device=device,
        )

    def extra_repr(self) -> str:
        return (f"H={self.hidden_size}, channels={self.num_channels}, "
                f"channel_size={self.channel_size}, heads={self.num_heads}")
