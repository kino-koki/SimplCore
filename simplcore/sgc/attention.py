# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC Hybrid 高效注意力：GQA（分组查询）+ 因果滑窗 + 可选 SDPA/Flash 路径.

设计动机（当前最强 LLM 注意力体系的标配组合）
================================================
* **GQA（Grouped-Query Attention）**：K/V 头数 ``num_kv_heads < num_heads``，
  多组 query 共享同一组 K/V。Llama2-70B（kv=8/q=64）、Llama3-70B（kv=8/q=64）
  即此结构。直接收益：流式 KV 缓存缩小 ``num_heads / num_kv_heads`` 倍
  （本包所有"有界流式状态"主张的关键——缓存只按 kv 头存），而权重只少
  ``2*D*(H-G)*head_dim`` 个。``num_kv_heads == num_heads`` 时退化为标准 MHA，
  与 :class:`simplcore.sgc.hybrid.SlidingWindowAttention` 1:1 同骨架、同布局
  （单个合并 ``qkv = Linear(dim, dim + 2*kv_dim)``，MHA 时即 3*D，与现有
  SlidingWindowAttention 的 qkv 权重一一对应），便于消融对照与权重迁移。
* **因果滑窗**：每个 query 只看 ``window`` 个 key，O(L·W)，无 O(L²)，无随 L
  增长的缓存。流式推理用环形缓存（只存**无 RoPE 的原始 k/v**，每步按绝对
  位置重套 RoPE），保证 流式 == 整段前向 逐位一致（与 ``test_hybrid`` 同
  守护协议）。
* **SDPA/Flash 路径**（``use_sdpa=True``）：训练前向改用
  ``F.scaled_dot_product_attention`` + banded 因果滑窗 bool 掩码，
  由 cuDNN/Flash 内核处理（GPU 上 O(L·W) 显存）；默认 ``False`` 走
  显式 matmul+softmax，与流式路径同一数值实现，守护测试才做逐位对照。

诚实边界
--------
* GQA 的"质量 vs 省 cache"权衡是**结构保证**（KV 字节数精确核算、流式==
  整段逐位一致、因果性），具体困惑度收益需训练实测；
* 纯 torch，零第三方。

API
---
* :class:`GQASlidingWindowAttention`
* :func:`analytic_gqa_attn_params` —— 该层参数量（不实例化，bit-exact）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixer import make_norm
from ._rope import _apply_rope, _rope_tables


# =============================================================================
# GQA + 滑窗因果注意力
# =============================================================================
class GQASlidingWindowAttention(nn.Module):
    """因果多头（GQA）滑窗自注意力。

    * ``num_kv_heads == num_heads``（默认，或 None 表示取 num_heads）：
      标准 MHA，骨架/权重布局与
      :class:`~simplcore.sgc.hybrid.SlidingWindowAttention` 完全一致
      （单个 ``qkv = Linear(D, 3D, bias=False)``），可 1:1 对照/迁移；
    * ``num_kv_heads < num_heads`` 且整除：GQA——q 投影 D->H*hd，
      k/v 各 D->G*hd（G = num_kv_heads），推理时每个 kv 头服务
      ``H//G`` 个 query 头（KV 缓存只按 G 存，流式字节省 H/G 倍）。

    前向骨架：前置 RMSNorm/LayerNorm -> qkv -> RoPE(q,k) -> 滑窗因果掩码
    （有限大负数，bf16 安全）-> softmax -> 残差 + out_proj。
    """

    def __init__(self, dim: int, num_heads: int, window: int = 256,
                 num_kv_heads: Optional[int] = None, norm: str = "rmsnorm",
                 dropout: float = 0.0, eps: float = 1e-6,
                 rope_theta: float = 10000.0, use_sdpa: bool = False) -> None:
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
        self.use_sdpa = bool(use_sdpa)
        self.head_group = num_heads // num_kv_heads

        self.norm = make_norm(norm, dim, eps=eps)
        # 合并 qkv 布局 [q | k | v]：MHA 时恰为 3*D（与 SlidingWindowAttention
        # 的 qkv 权重一一对应，便于 1:1 对照）；GQA 时 k/v 段各为 kv_dim。
        self.qkv = nn.Linear(dim, dim + 2 * self.kv_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # -- 投影/重组 -------------------------------------------------------------
    def _project(self, h: torch.Tensor):
        """h: (B, L, D) -> q (B,H,L,hd), k (B,G,L,hd), v (B,G,L,hd)。"""
        B, L, _ = h.shape
        qkv = self.qkv(h)
        q, k, v = qkv.split([self.dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor):
        """MHA 时直接返回；GQA 时把每个 kv 头复制到其服务的 query 组。

        k/v: (B, G, L, hd) -> (B, H, L, hd)。
        """
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
        """(L, L) 有限大负数掩码：j <= t 且 t - j < window 时可见。"""
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
        cos, sin = _rope_tables(self.head_dim, L, x.device, q.dtype,
                                 base=self.rope_theta)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        if self.use_sdpa:
            ke, ve = self._expand_kv(k, v)
            bool_mask = self._windowed_causal_mask(L, x.device, q.dtype).bool()
            y = F.scaled_dot_product_attention(q, ke, ve,
                                                attn_mask=~bool_mask)
        else:
            ke, ve = self._expand_kv(k, v)
            attn = torch.matmul(q, ke.transpose(-2, -1)) * (self.head_dim ** -0.5)
            mask = self._windowed_causal_mask(L, q.device, q.dtype)
            attn = F.softmax(attn + mask.unsqueeze(0).unsqueeze(0), dim=-1)
            y = torch.matmul(self.drop(attn), ve)
        y = y.transpose(1, 2).reshape(B, L, D)
        return x + self.out_proj(y)

    # -- 流式（环形 KV 缓存，只按 kv 头存）--------------------------------------
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
        """Consume one token (B, 1, D) or (B, D); return (B, 1, D) + new state.

        环形缓存存**无 RoPE 的原始 k/v**（按 kv 头 G 存，GQA 省 H/G 倍），
        每步按绝对位置重新套 RoPE，与整段前向逐位一致。
        返回 per-layer hidden (B, 1, D)，与 :class:`SlidingWindowAttention` 流式
        协议相同（调用方自己堆叠成 (B, L, D)）。
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
        cos_q, sin_q = _rope_tables(self.head_dim, pos + 1, dev, q.dtype,
                                    base=self.rope_theta)
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
        cos_k, sin_k = _rope_tables(self.head_dim, n, dev, Ks.dtype,
                                    base=self.rope_theta)
        Ks_rot = _apply_rope(Ks, cos_k[start_abs:n], sin_k[start_abs:n])  # (B, G, w, hd)

        # GQA：query 头 (B, H, hd) 扩展到 (B, G, group, 1, hd) 与每个 kv 头配对
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

        attn = torch.matmul(qe, Ke.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, Ve)
        # out: (B, G, group, 1, hd)；head h = g*group + i 在 D 中占通道
        # h*hd..h*hd+hd，故按 C 顺序展平 (G, group) -> H（与 q_rot.reshape 一致）
        out = out.squeeze(3).reshape(B, self.num_heads * self.head_dim)
        y = x_t.squeeze(1) + self.out_proj(out)
        return y.unsqueeze(1), state

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        """流式 KV 字节数：2 * batch * G * W * hd（G = num_kv_heads）。"""
        return 2 * batch * self.num_kv_heads * self.window * self.head_dim * dtype.itemsize

    # SGCModel.stream_bytes 统一探测 streaming_state_bytes，提供别名
    streaming_state_bytes = stream_bytes

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
                f"window={self.window}, sdpa={self.use_sdpa}")


# =============================================================================
# 参数量解析（bit-exact）
# =============================================================================
def analytic_gqa_attn_params(dim: int, num_heads: int, num_kv_heads: Optional[int] = None) -> int:
    """:class:`GQASlidingWindowAttention` 的精确参数量（不实例化）。

    每一项（bias 全部为 False）：

    * ``q`` 投影:   ``dim * dim``
    * ``k`` 投影:   ``dim * (num_kv_heads * head_dim)``，head_dim = dim // num_heads
    * ``v`` 投影:   同 ``k``
    * ``out`` 投影: ``dim * dim``
    * 前置 norm:    ``dim``（RMSNorm weight；LayerNorm 则 2*dim，本实现按
      rmsnorm 口径，与 ``SlidingWindowAttention`` 一致）

    MHA 退化（num_kv_heads == num_heads）时上式 = 3*D² + D² + D = 4*D² + D，
    与 ``simplcore.sgc.hybrid.SlidingWindowAttention`` 的参数口径一致
    （其 qkv = Linear(D, 3D) + out = Linear(D, D) + norm weight D）。

    由 ``tests/test_attention.py::test_analytic_params_bit_exact`` 守护。
    """
    kv_heads = num_heads if num_kv_heads is None else int(num_kv_heads)
    head_dim = dim // num_heads
    return dim * dim + 2 * dim * kv_heads * head_dim + dim * dim + dim


__all__ = [
    "GQASlidingWindowAttention",
    "analytic_gqa_attn_params",
]
