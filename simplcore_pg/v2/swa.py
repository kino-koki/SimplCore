# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""L0 滑动窗口注意力 (Sliding Window Attention, TriMemory 近程精确记忆)。

  * 窗口 W 固定 → KV 缓存恒定 O(W)，与上下文总长无关；
  * RoPE 仅使用窗口内的**绝对位置** (环形缓存中按写入时的绝对位置编码)，
    相对性质保证 q_i·k_j 只依赖 (i−j)，因此无需任何位置外推；
  * GQA (num_kv_heads < num_heads) 进一步压缩 KV 体积；
  * 训练前向用带状因果掩码 + SDPA；推理用固定大小的环形 KV 缓存
    (``WindowKVCache``)，每步 O(W)。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import RMSNorm


class WindowKVCache:
    """固定窗口环形 KV 缓存 (推理期)。容量 W 恒定 → 显存 O(1)。

    缓存的 k 已在写入时完成 RoPE (按绝对位置)，读取时直接参与点积。
    """

    def __init__(self, batch: int, kv_heads: int, window: int, head_dim: int,
                 device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.window = window
        self.k = torch.zeros(batch, kv_heads, window, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch, kv_heads, window, head_dim, device=device, dtype=dtype)
        self.write_ptr = 0
        self.length = 0          # 已写入的绝对位置总数

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """写入 n 个新位置的 k/v (k_new/v_new: (B, kv, n, hd))，环形覆盖最旧位置。"""
        n = k_new.shape[2]
        for i in range(n):
            self.k[:, :, self.write_ptr] = k_new[:, :, i]
            self.v[:, :, self.write_ptr] = v_new[:, :, i]
            self.write_ptr = (self.write_ptr + 1) % self.window
            self.length += 1

    def ordered(self) -> tuple[torch.Tensor, torch.Tensor]:
        """按时间顺序返回有效 k/v：(B, kv, m, hd)，m = min(length, W)。"""
        m = min(self.length, self.window)
        if self.length < self.window:
            return self.k[:, :, :m], self.v[:, :, :m]
        # 环形已满：最旧位置在 write_ptr
        k = torch.cat([self.k[:, :, self.write_ptr:], self.k[:, :, :self.write_ptr]], dim=2)
        v = torch.cat([self.v[:, :, self.write_ptr:], self.v[:, :, :self.write_ptr]], dim=2)
        return k, v

    def nbytes(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()

    def reset(self) -> None:
        self.k.zero_(); self.v.zero_()
        self.write_ptr = 0
        self.length = 0


class SlidingWindowAttention(nn.Module):
    """滑动窗口多头注意力 (GQA + RoPE, pre-norm + 残差内置)。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        window: int,
        rope_base: float = 10_000.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除。")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads 必须能被 num_kv_heads 整除 (GQA)。")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.window = window

        self.norm = RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        inv_freq = 1.0 / (rope_base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------
    def _rope(self, positions: torch.Tensor, dtype: torch.dtype, device: torch.device):
        """positions: (L,) 绝对位置 -> cos/sin (L, head_dim)。"""
        freqs = torch.outer(positions.to(torch.float32), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).to(device), emb.sin().to(dtype).to(device)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x: (B, heads, L, hd)；cos/sin: (L, hd)。"""
        x1, x2 = x.chunk(2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def _repeat_kv(self, t: torch.Tensor) -> torch.Tensor:
        """(B, kv, L, hd) -> (B, heads, L, hd) GQA 展开。"""
        return t.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """训练/全量前向：带状因果掩码 (j ≤ i 且 i−j < W)。

        Args:
            x: (B, L, H)。
            pos_offset: 本段在整个序列中的绝对起始位置 (RoPE 用)。
        """
        torch._assert(x.dim() == 3, "SWA 期望 3D 输入 (B, L, H)")
        B, L, H = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        pos = torch.arange(pos_offset, pos_offset + L, dtype=torch.float32, device=x.device)
        cos, sin = self._rope(pos, x.dtype, x.device)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        k, v = self._repeat_kv(k), self._repeat_kv(v)

        # 带状因果掩码: 允许 j<=i 且 i-j < W
        i = torch.arange(L, device=x.device).unsqueeze(1)
        j = torch.arange(L, device=x.device).unsqueeze(0)
        allowed = (j <= i) & ((i - j) < self.window)
        mask = torch.zeros(L, L, dtype=x.dtype, device=x.device)
        mask = mask.masked_fill(~allowed, float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, L, H)
        return x + self.o_proj(out)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, cache: WindowKVCache) -> torch.Tensor:
        """推理单步/小段前向：KV 写入环形缓存，注意力只看缓存内 ≤W 个位置。

        Args:
            x_t: (B, n, H) 新 token (n 通常为 1，也支持小段)。
            cache: 固定容量的环形缓存 (原地更新)。
        """
        B, n, H = x_t.shape
        h = self.norm(x_t)
        q = self.q_proj(h).view(B, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, n, self.num_kv_heads, self.head_dim).transpose(1, 2)

        abs_pos = torch.arange(cache.length, cache.length + n,
                               dtype=torch.float32, device=x_t.device)
        cos, sin = self._rope(abs_pos, x_t.dtype, x_t.device)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        cache.append(k, v)
        k_all, v_all = cache.ordered()          # (B, kv, m, hd)
        k_all, v_all = self._repeat_kv(k_all), self._repeat_kv(v_all)

        # 新段内部仍需因果 (第 i 个新 token 不能看同段更后的)，缓存部分全部可见
        m = k_all.shape[2]
        ii = torch.arange(n, device=x_t.device).unsqueeze(1)
        jj = torch.arange(m, device=x_t.device).unsqueeze(0)
        # 缓存位置全局索引 < 新段位置全局索引 (m-n+jj 对应当前缓存序)
        allowed = jj <= (m - n + ii)
        mask = torch.zeros(n, m, dtype=x_t.dtype, device=x_t.device)
        mask = mask.masked_fill(~allowed, float("-inf"))

        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, n, H)
        return x_t + self.o_proj(out)

    def extra_repr(self) -> str:
        return (f"H={self.hidden_size}, heads={self.num_heads}, "
                f"kv={self.num_kv_heads}, W={self.window}")
