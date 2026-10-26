# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""GQA + YaRN 组合注意力：省 KV 缓存 + 长程外推的结构保证.

设计动机
========
:class:`GQASlidingWindowAttention`（attention.py）省 KV 缓存（按 kv 头
G 存，省 H/G 倍），但用固定 RoPE（rope_theta=10000）——外推倍率 >1 时
高频分量失配。:class:`SlidingWindowAttentionYaRN`（rope_scales.py）用
NTK 平滑位置缩放表，但 qkv 布局是 MHA（3*D²），不省 KV。

本模块 :class:`GQASlidingWindowAttentionYaRN` 把两者**组合**：
* GQA 的合并 qkv 布局（[q|k|v] = D + 2*G*hd，KV 缓存只按 G 头存，
  流式字节省 H/G 倍）；
* YaRN/NTK 的逐分量位置缩放表（RopeScaler，strategy/factor 可调），
  流式路径按**绝对位置**取缩放表——绝对位置超过训练长度时表仍数学
  良定义，流式 == 整段前向逐位一致。

这是 Llama3 系列的标配组合（GQA kv=8/q=32 + 128K YaRN 外推）在
SGC 混合架构里的落地。

诚实边界
--------
* 这是**结构保证**（KV 省量精确核算、流式==整段逐位、因果、参数量
  bit-exact、NTK 表有界有限），具体困惑度收益需训练实测。
* strategy="none" 且 factor=1.0 时逐位等于 GQASlidingWindowAttention。
* 纯 torch，零第三方。

API
---
* :class:`GQASlidingWindowAttentionYaRN`
* :func:`analytic_gqa_yarn_attn_params` —— 该层参数量（不实例化，bit-exact）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixer import make_norm
from ._rope import _apply_rope
from .rope_scales import RopeScaler


class GQASlidingWindowAttentionYaRN(nn.Module):
    """因果多头（GQA）滑窗自注意力 + YaRN/NTK 位置缩放外推.

    与 GQASlidingWindowAttention 同骨架，差异仅在 cos/sin 表由
    RopeScaler 生成（strategy/factor 可调），流式路径按**绝对位置**
    取缩放表。

    Args:
        dim / num_heads / window / num_kv_heads / norm / dropout / eps /
            rope_theta: 与 GQASlidingWindowAttention 一致。
        strategy: "none" | "linear" | "yarn"
            （strategy="none" 且 factor=1.0 时逐位等于 GQASlidingWindowAttention）。
        factor: 缩放倍率（strategy="none" 时忽略；factor <= 1 时退化为标准 RoPE）。
    """

    def __init__(self, dim: int, num_heads: int, window: int = 256,
                 num_kv_heads: Optional[int] = None, norm: str = "rmsnorm",
                 dropout: float = 0.0, eps: float = 1e-6,
                 rope_theta: float = 10000.0,
                 strategy: str = "none", factor: float = 1.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        num_kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
        assert 1 <= num_kv_heads <= num_heads and num_heads % num_kv_heads == 0, \
            "num_kv_heads must divide num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.window = max(1, int(window))
        self.rope_theta = float(rope_theta)
        self.strategy = strategy
        self.factor = float(factor)
        self.head_group = num_heads // num_kv_heads
        self.scaler = RopeScaler(strategy=strategy, factor=factor, base=rope_theta)

        self.norm = make_norm(norm, dim, eps=eps)
        self.qkv = nn.Linear(dim, dim + 2 * self.kv_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # -- 投影/重组 -------------------------------------------------------------
    def _project(self, h: torch.Tensor):
        B, L, _ = h.shape
        qkv = self.qkv(h)
        q, k, v = qkv.split([self.dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor):
        if self.num_kv_heads == self.num_heads:
            return k, v
        B, G, L, hd = k.shape
        k = k.unsqueeze(2).expand(B, G, self.head_group, L, hd).reshape(
            B, self.num_heads, L, hd)
        v = v.unsqueeze(2).expand(B, G, self.head_group, L, hd).reshape(
            B, self.num_heads, L, hd)
        return k, v

    # -- 掩码 -----------------------------------------------------------------
    def _windowed_causal_mask(self, L: int, device, dtype) -> torch.Tensor:
        idx = torch.arange(L, device=device)
        j = idx.view(1, -1)
        t = idx.view(-1, 1)
        visible = (j <= t) & ((t - j) < self.window)
        neg = torch.finfo(dtype).min / 4
        mask = torch.zeros(L, L, device=device, dtype=dtype)
        return mask.masked_fill(~visible, neg)

    # -- 前向 -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm(x)
        q, k, v = self._project(h)
        cos, sin = self.scaler.tables(self.head_dim, L, x.device, q.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        ke, ve = self._expand_kv(k, v)
        attn = torch.matmul(q, ke.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = self._windowed_causal_mask(L, q.device, q.dtype)
        attn = F.softmax(attn + mask.unsqueeze(0).unsqueeze(0), dim=-1)
        y = torch.matmul(self.drop(attn), ve)
        y = y.transpose(1, 2).reshape(B, L, D)
        return x + self.out_proj(y)

    # -- 流式（环形 KV 缓存，只按 kv 头存，按绝对位置重套缩放 RoPE）------------
    def init_stream(self, batch: int, device, dtype) -> Dict:
        G = self.num_kv_heads
        return {
            "k": torch.zeros(batch, G, self.window, self.head_dim,
                              device=device, dtype=dtype),
            "v": torch.zeros(batch, G, self.window, self.head_dim,
                              device=device, dtype=dtype),
            "n": 0,
        }

    @torch.no_grad()
    def stream_step(self, x_t: torch.Tensor, state: Dict) -> Tuple[torch.Tensor, Dict]:
        """Consume one token (B,1,D) or (B,D); return (B,1,D) + new state.

        环形缓存存**无 RoPE 的原始 k/v**（按 kv 头 G 存，GQA 省 H/G 倍），
        每步按**绝对位置**重新套缩放 RoPE，与整段前向逐位一致。
        """
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        B, one, D = x_t.shape
        assert one == 1
        h = self.norm(x_t)
        q, k, v = self._project(h)
        q = q[:, :, 0, :]                     # (B, H, hd)
        k = k[:, :, 0, :]                     # (B, G, hd)
        v = v[:, :, 0, :]                     # (B, G, hd)

        pos = state["n"]
        dev = x_t.device
        W = self.window
        cos_q, sin_q = self.scaler.tables(self.head_dim, pos + 1, dev, q.dtype)
        q_rot = _apply_rope(q.unsqueeze(2), cos_q[pos:pos + 1],
                            sin_q[pos:pos + 1])[:, :, 0]      # (B, H, hd)

        slot = pos % W
        state["k"][:, :, slot] = k
        state["v"][:, :, slot] = v
        state["n"] = pos + 1
        n = state["n"]

        w = min(W, n)
        start_abs = n - w
        slots = torch.arange(start_abs, n, device=dev) % W
        Ks = state["k"][:, :, slots]         # (B, G, w, hd)
        Vs = state["v"][:, :, slots]
        cos_k, sin_k = self.scaler.tables(self.head_dim, n, dev, Ks.dtype)
        Ks_rot = _apply_rope(Ks, cos_k[start_abs:n], sin_k[start_abs:n])

        if self.num_kv_heads == self.num_heads:
            qe = q_rot.unsqueeze(2)                          # (B, H, 1, hd)
            Ke = Ks_rot                                      # (B, H, w, hd)
            Ve = Vs
        else:
            qe = q_rot.reshape(B, self.num_kv_heads, self.head_group, self.head_dim)
            qe = qe.unsqueeze(3)                              # (B, G, group, 1, hd)
            Ke = Ks_rot.unsqueeze(2).expand(
                B, self.num_kv_heads, self.head_group, w, self.head_dim)
            Ve = Vs.unsqueeze(2).expand(
                B, self.num_kv_heads, self.head_group, w, self.head_dim)

        # 保证 dtype 一致（Ks_rot 可能因 scaler.tables 的 dtype 参数与 qe 不一致）
        Ke = Ke.to(qe.dtype)
        Ve = Ve.to(qe.dtype)
        attn = torch.matmul(qe, Ke.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, Ve)
        out = out.squeeze(3).reshape(B, self.num_heads * self.head_dim)
        y = x_t.squeeze(1) + self.out_proj(out)
        return y.unsqueeze(1), state

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        """流式 KV 字节数：2 * batch * G * W * hd（G = num_kv_heads）。"""
        return 2 * batch * self.num_kv_heads * self.window * self.head_dim * dtype.itemsize

    streaming_state_bytes = stream_bytes

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.num_heads}, "
                f"kv_heads={self.num_kv_heads}, window={self.window}, "
                f"rope_theta={self.rope_theta}, strategy={self.strategy}, "
                f"factor={self.factor}")


def analytic_gqa_yarn_attn_params(dim: int, num_heads: int,
                                   num_kv_heads: Optional[int] = None) -> int:
    """:class:`GQASlidingWindowAttentionYaRN` 的精确参数量（不实例化）。

    与 analytic_gqa_attn_params 完全相同（qkv + out_proj + norm 的
    param 口径一致，YaRN 缩放表是**非参数**计算——cos/sin 表由
    base/factor 解析生成，不引入可学习参数）：

    即 = 2*dim**2 + 2*dim*G*head_dim + dim（G = num_kv_heads）。
    """
    from .attention import analytic_gqa_attn_params
    return analytic_gqa_attn_params(dim, num_heads, num_kv_heads)


__all__ = [
    "GQASlidingWindowAttentionYaRN",
    "analytic_gqa_yarn_attn_params",
]
