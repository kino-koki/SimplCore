# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 原生局部交互层 (LocalInteraction) —— 完全替代自注意力。

设计动机（论文级重构，2026-08-24）：
  SimplCore 正式放弃 Transformer/自注意力路线。自注意力的 Q/K/V/RoPE/SDPA
  全部移除，改由 **纯卷积 + 线性递归** 的局部交互层承担"邻近 token 混合"职责：

    * 全局依赖：交给 ResonanceField (RFP，谱全局卷积，O(n log n)，天生全局)；
    * 局部依赖：由本层的因果深度可分离卷积 + 可选门控线性递归承担，O(n) 且
      完全无注意力机制。

两种局部算子（均不需要位置编码，因果性由左填充保证）：

  1) CausalDepthwiseConv1d：逐通道一维因果卷积（可带空洞 dilation 扩大感受野），
     权重在 (W_conv,) 上共享跨通道（depthwise），参数量 O(kernel) 与 H 无关，
     是轻量、稳定、易并行的局部混合器。
  2) GatedLinearRecurrence (可选)：token 级线性递归 h_t = a_t ⊙ h_{t-1} + (1-a_t)⊙x_t，
     与卷积并联，提供跨 token 的单向信息流（类线性注意力/线性 RNN，但无 softmax
     归一化、无全局 token 交互，故不是注意力）。

二者经门控融合，输出 (B,L,H)。整个层纯张量运算、无数据相关控制流，
兼容 torch.compile / FX。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .layers import RMSNorm


class CausalDepthwiseConv1d(nn.Module):
    """因果深度可分离一维卷积（逐通道，可带空洞）。

    因果性：左填充 kernel_size-1 个时间步（含 dilation），使位置 t 的卷积
    仅依赖 t 及更早的位置，不含未来信息。权重 (out_channels, 1, kernel_size)
    跨通道共享（groups=out_channels），参数量极小。
    """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        if kernel_size < 1 or dilation < 1:
            raise ValueError("kernel_size/dilation 必须 >= 1。")
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation,
            groups=channels, bias=bias, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, L, C)。"""
        # (B, C, L) 用于 Conv1d
        x_t = x.transpose(1, 2)
        x_pad = torch.nn.functional.pad(x_t, (self.pad, 0))  # 左填充，因果
        y = self.conv(x_pad)                                  # (B, C, L)
        return y.transpose(1, 2)


class GatedLinearRecurrence(nn.Module):
    """门控线性递归（可选局部算子，非注意力）。

    h_t = a_t ⊙ h_{t-1} + (1 - a_t) ⊙ x_t，a_t = sigmoid(W_a x_t + b)
    纯递归、无全局 token 交互、无 softmax 归一化，复杂度 O(n·H)。

    并行实现（论文级，FX 可追踪）：一阶线性递归 h_t = a_t h_{t-1} + b_t 的
    前缀计算是**可结合**的复合运算
      combine((A1,B1),(A2,B2)) = (A1·A2, A2·B1 + B2)
    用 Hillis-Steele 关联扫描在 O(log C) 深度内并行算出全部 h_t（无 Python
    顺序循环），数值稳定（中间量全程有界：A ∈ (0,1]，B 为带衰减的加权和，
    无上溢/下溢），且与顺序展开逐位一致（仅受 float32 舍入影响，~1e-7）。

    长序列按 ``chunk_size`` 分块：块内并行扫描、块间顺序传播固定状态，
    总顺序深度 O(L/C + log C)。块内扫描的循环界为常量（chunk_size），
    兼容 torch.compile / FX 图模式追踪。
    """

    def __init__(self, hidden_size: int, bias: bool = True,
                 chunk_size: int = 128, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.n_chunks = (max_seq_len + chunk_size - 1) // chunk_size  # 常量（静态循环界）
        self.gate = nn.Linear(hidden_size, hidden_size, bias=bias)
        # 预计算关联扫描各轮掩码（常量 buffer，避免追踪期 arange/device 依赖）
        n_levels = (chunk_size - 1).bit_length()
        masks = []
        for level in range(n_levels):
            d = 1 << level
            masks.append((torch.arange(chunk_size) >= d).view(1, chunk_size, 1))
        self.register_buffer("_scan_masks", torch.stack(masks))  # (n_levels, 1, C, 1)

    def _assoc_scan(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hillis-Steele 关联扫描：h_t = a_t h_{t-1} + b_t, h_{-1} = 0。

        Args:
            a: (B, C, H) 衰减门 ∈ (0, 1)。
            b: (B, C, H) 输入。
        Returns:
            (A, B)：A[t] = ∏_{k≤t} a_k（前缀积），B[t] = h_t。
        """
        A, B = a, b
        for level in range(self._scan_masks.shape[0]):
            d = 1 << level
            mask = self._scan_masks[level]  # (1, C, 1) 常量
            A_prev = torch.roll(A, d, dims=1)
            B_prev = torch.roll(B, d, dims=1)
            A_new = torch.where(mask, A_prev * A, A)
            B_new = torch.where(mask, A * B_prev + B, B)  # 用旧 A（当前段衰减 A2）
            A, B = A_new, B_new
        return A, B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, H) -> (B, L, H)，要求 L <= max_seq_len。"""
        B, L, H = x.shape
        torch._assert(
            L <= self.max_seq_len,
            "序列长度超过 max_seq_len；静默裁剪会破坏因果性，已拒绝。",
        )
        a = torch.sigmoid(self.gate(x))              # (B, L, H) 遗忘门 ∈ (0,1)
        b = (1.0 - a) * x                            # 输入加权
        C = self.chunk_size
        M = self.n_chunks
        total = M * C
        # 填充到固定总长（静态循环界，FX 可追踪）；填充位置 b=0、a=sigmoid(bias)，
        # 因果性保证其不影响 L 之前的输出，最后切片丢弃。
        a = torch.nn.functional.pad(a, (0, 0, 0, total - L))
        b = torch.nn.functional.pad(b, (0, 0, 0, total - L))
        a_c = a.view(B, M, C, H)
        b_c = b.view(B, M, C, H)
        state = torch.zeros_like(a_c[:, 0, 0])  # (B, H)，zeros_like 兼容 FX 追踪
        outs = []
        for j in range(M):  # 静态循环：块内并行扫描 + 块间状态传播
            A_j, B_j = self._assoc_scan(a_c[:, j], b_c[:, j])
            outs.append(B_j + A_j * state.unsqueeze(1))
            state = A_j[:, -1] * state + B_j[:, -1]
        h = torch.cat(outs, dim=1)
        return h[:, :L]


class LocalInteraction(nn.Module):
    """SimplCore 原生局部交互层：卷积 + (可选)线性递归，门控融合。替代自注意力。"""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 7,
        dilation: int = 1,
        use_conv: bool = True,
        use_recurrence: bool = True,
        eps: float = 1e-6,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_conv = use_conv
        self.use_recurrence = use_recurrence
        self.conv = CausalDepthwiseConv1d(hidden_size, kernel_size=kernel_size, dilation=dilation, bias=False) if use_conv else None
        if use_recurrence:
            self.recur = GatedLinearRecurrence(hidden_size, bias=True, max_seq_len=max_seq_len)
        else:
            self.recur = None
        n_parts = int(use_conv) + int(use_recurrence)
        if n_parts == 0:
            raise ValueError("LocalInteraction 至少需要卷积或递归之一。")
        self.fuse = nn.Linear(hidden_size * n_parts, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, H) -> (B, L, H)。"""
        torch._assert(x.dim() == 3, "LocalInteraction 期望 3D 输入 (B, L, H)")
        parts = []
        if self.conv is not None:
            parts.append(self.conv(x))
        if self.recur is not None:
            parts.append(self.recur(x))
        cat = torch.cat(parts, dim=-1)
        out = self.fuse(cat)
        return self.norm(x + out)

    def extra_repr(self) -> str:
        return (f"H={self.hidden_size}, kernel={self.kernel_size}, dilation={self.dilation}, "
                f"conv={self.use_conv}, recurrence={self.use_recurrence}")
