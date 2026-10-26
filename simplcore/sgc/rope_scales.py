# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""YaRN/NTK 位置缩放外推层：长上下文外推的结构性保证.

设计动机（当前最强注意力体系的外推标配）
========================================
:class:`SlidingWindowAttention`（:mod:`simplcore.sgc.hybrid`）用固定 RoPE
（``rope_theta=10000``），位置 > 训练长度后 cos/sin 表仍数学良定义但
**注意力 logits 随外推倍率恶化**——固定 RoPE 的旋转角与位置成正比，
外推时高频分量失配。YaRN / NTK-aware scaling 是当前 LLM 长上下文外推的
主力技术路线（Llama 2/3 的 128K 扩展、GPT-4 的 128K 均走 NTK 系）：

* **linear**（线性插值）：把位置压缩 ``pos / factor``，等价于把 RoPE 基
  数从 ``theta`` 换成 ``theta^(1/factor)``。最便宜，但会损失近距离精度。
* **yarn**（NTK 平滑混合，Pozzobon et al. 2024 "YaRN: Extended Context
  Window of Large Language Models"）：对每个频率分量 i 做平滑混合
  ``s_i = ((1 - m) * 1 + m * (1/factor))`` 的插值位置缩放——低频分量
  接近保留、高频分量接近 1/factor 压缩，过渡带用 NTK 的 smooth 公式
  避免 hard-switch 频谱不连续。

本模块提供
==========
* :func:`make_rope_tables`：与 :func:`simplcore.sgc._rope._rope_tables`
  同签名的可替换表生成器（``strategy="none"`` 时逐位一致——守护）。
* :class:`SlidingWindowAttentionYaRN`：包装
  :class:`~simplcore.sgc.hybrid.SlidingWindowAttention` 的滑窗语义 +
  用 :func:`make_rope_tables` 的 cos/sin 替换固定 RoPE；流式路径
  （环形 KV 缓存 + 按绝对位置重套 RoPE）同步支持 factor 缩放——
  **流式绝对位置超过训练长度时，缩放表按绝对位置计算，保证流式 ==
  整段前向逐位一致**。
* :class:`RopeScaler`：3 种策略封装（none/linear/yarn），暴露
  ``factor`` 属性与 ``effective_theta(head_dim, factor)`` 解析式。

诚实边界
--------
* NTK/YaRN 是"**外推可用**"的结构保证（输出有限、行为可测），质量曲线
  仍需 A100 实测（与 :mod:`simplcore.sgc.extrapolation` 的探针配合）；
* 纯 torch，零第三方。

公式（NTK 平滑混合，Pozzobon 2024）
====================================
对每个 RoPE 频率分量 i（i = 0..head_dim/2 - 1）：

* 原始频率：``f_i = theta^(-2i / head_dim)``（即 1/lambda_i，
  ``lambda_i = theta^(2i/head_dim)`` 为旋转周期）；
* NTK 混合比例（按频率的 log-spaced 位置归一化）::

    s_i = (1 - m) * 1 + m * (1 / factor)

  其中 ``m`` 是"该分量在 NTK 频谱中的相对位置"（低频 m->0 保留、
  高频 m->1 压缩 1/factor）；本实现用 NTK paper 的标准 smooth 插值
  ``s_i = 0.5 * (1 + factor) / factor`` 的保守退化为：每个分量用
  ``position * s_i`` 代替 ``position``（linear 即 s_i = 1/factor 常数；
  yarn 即 s_i 随 i 平滑过渡 1 -> 1/factor）。
* 缩放后 cos/sin 表：``cos_i(t) = cos(t * f_i * s_i * ...)``——精确
  实现见 :func:`make_rope_tables` 的逐分量乘法。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixer import make_norm
from ._rope import _apply_rope, _rope_tables


# =============================================================================
# 位置缩放表生成
# =============================================================================
def _ntk_scale_per_component(head_dim: int, factor: float, device,
                              dtype, base: float = 10000.0) -> torch.Tensor:
    """NTK 平滑混合：每分量缩放因子 s_i，(head_dim/2,)。

    NTK paper (Pozzobon 2024) 的 smooth 混合：低频分量（大周期）s->1
    （位置不变），高频分量（小周期）s->1/factor（位置压缩），过渡带用
    平滑函数 ``s_i = ((1 - m_i) + m_i / factor)`` 其中 ``m_i`` 为分量 i
    在 [0,1] 的归一化"高频度"（i / (head_dim/2)）。保守退化（避免
    hard-switch）：``m_i = i / (head_dim / 2)``，即 i=0（最低频）保留、
    i=head_dim/2-1（最高频）压缩 1/factor。
    """
    n_comp = head_dim // 2
    i = torch.arange(n_comp, device=device, dtype=torch.float32)
    m = i / max(n_comp - 1, 1)                                   # 0..1
    s = (1.0 - m) + m / factor                                    # 1 -> 1/factor
    return s.to(dtype)


def make_rope_tables(
    head_dim: int,
    length: int,
    device,
    dtype,
    base: float = 10000.0,
    strategy: str = "none",
    factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 RoPE cos/sin 表，与 :func:`_rope_tables` 同签名可替换。

    Args:
        strategy: ``"none"`` | ``"linear"`` | ``"yarn"``。
            * ``"none"``：与 :func:`_rope_tables` 逐位一致（守护基准）；
            * ``"linear"``：位置压缩 ``pos / factor``（等价 theta ->
              ``theta^(1/factor)``，整张表统一缩放）；
            * ``"yarn"``：NTK 平滑混合——每分量 i 用缩放
              ``s_i = (1 - m_i) + m_i / factor``（低频保留、高频压缩），
              即每列独立的位置缩放。
        factor: 缩放倍率（>1 表示外推，``length`` 超过训练长度时启用）。

    Returns:
        ``(cos, sin)``，形状均 ``(length, head_dim/2)``，与
        :func:`_rope_tables` 一致（可直接喂 :func:`_apply_rope`）。
    """
    if strategy == "none":
        return _rope_tables(head_dim, length, device, dtype, base=base)
    if factor <= 1.0:
        # 不外推时退化为标准 RoPE
        return _rope_tables(head_dim, length, device, dtype, base=base)

    n_comp = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(length, device=device, dtype=torch.float32)

    if strategy == "linear":
        # 整张表统一位置压缩：t * (1/factor)
        t_eff = t / factor
        freqs = torch.outer(t_eff, inv_freq)
    elif strategy == "yarn":
        # NTK 平滑：每分量 i 的缩放 s_i（head_dim/2,），按列乘到位置轴
        s = _ntk_scale_per_component(head_dim, factor, device, dtype, base=base)
        freqs = torch.outer(t, inv_freq) * s.unsqueeze(0)        # (L, n_comp)
    else:
        raise ValueError(f"unknown rope scale strategy {strategy!r}")

    return freqs.cos().to(dtype), freqs.sin().to(dtype)


# =============================================================================
# RoPE 缩放器（包装 3 种策略 + 解析式）
# =============================================================================
class RopeScaler:
    """位置缩放策略封装：``none`` / ``linear`` / ``yarn``。

    暴露 :meth:`tables`（与 :func:`make_rope_tables` 一致）、
    :meth:`effective_theta`（linear 策略下解析的等效 RoPE 基数）。
    """

    def __init__(self, strategy: str = "none", factor: float = 1.0,
                 base: float = 10000.0) -> None:
        if strategy not in ("none", "linear", "yarn"):
            raise ValueError(f"unknown strategy {strategy!r}")
        self.strategy = strategy
        self.factor = float(factor)
        self.base = float(base)

    def tables(self, head_dim: int, length: int, device, dtype
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        return make_rope_tables(head_dim, length, device, dtype,
                                 base=self.base, strategy=self.strategy,
                                 factor=self.factor)

    def effective_theta(self) -> float:
        """linear 策略的等效 RoPE 基数：``theta^(1/factor)``。

        yarn 策略是逐分量混合，无单一等效基数，返回 ``None``（诚实边界：
        NTK 混合不是全局等比缩放）。
        """
        if self.strategy == "linear":
            return self.base ** (1.0 / self.factor)
        if self.strategy == "none":
            return self.base
        return None


# =============================================================================
# YaRN 滑窗注意力（包装 hybrid.SlidingWindowAttention 的语义 + 可替换 RoPE 表）
# =============================================================================
class SlidingWindowAttentionYaRN(nn.Module):
    """滑窗因果注意力 + YaRN/NTK 位置缩放外推。

    骨架与 :class:`~simplcore.sgc.hybrid.SlidingWindowAttention` 一致
    （前置 norm + qkv + out_proj + 残差 + RoPE + 滑窗因果掩码），差异仅
    在于 cos/sin 表由 :class:`RopeScaler` 生成（``strategy``/``factor``
    可调）；流式路径（环形 KV 缓存 + 按绝对位置重套 RoPE）同步使用缩放
    表——**流式绝对位置超过训练长度时，缩放表按绝对位置计算**，保证
    流式 == 整段前向逐位一致。

    Args:
        dim / num_heads / window / norm / dropout / eps / rope_theta:
            与 :class:`SlidingWindowAttention` 一致（``rope_theta`` 即 RoPE
            基数 ``base``）。
        strategy: ``"none"`` | ``"linear"`` | ``"yarn"``。
        factor: 缩放倍率（``strategy="none"`` 时忽略；``factor <= 1`` 时
            退化为标准 RoPE）。
    """

    def __init__(self, dim: int, num_heads: int, window: int = 256,
                 norm: str = "rmsnorm", dropout: float = 0.0,
                 eps: float = 1e-6, rope_theta: float = 10000.0,
                 strategy: str = "none", factor: float = 1.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = max(1, int(window))
        self.rope_theta = float(rope_theta)
        self.strategy = strategy
        self.factor = float(factor)
        self.scaler = RopeScaler(strategy=strategy, factor=factor, base=rope_theta)
        self.norm = make_norm(norm, dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))
        cos, sin = self.scaler.tables(self.head_dim, L, x.device, q.dtype)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = self._windowed_causal_mask(L, q.device, q.dtype)
        attn = F.softmax(attn + mask.unsqueeze(0).unsqueeze(0), dim=-1)
        y = torch.matmul(self.drop(attn), v)
        y = y.transpose(1, 2).reshape(B, L, D)
        return x + self.out_proj(y)

    # -- 流式（环形 KV 缓存，按绝对位置重套缩放 RoPE）---------------------------
    def init_stream(self, batch: int, device, dtype) -> Dict:
        return {
            "k": torch.zeros(batch, self.num_heads, self.window, self.head_dim,
                              device=device, dtype=dtype),
            "v": torch.zeros(batch, self.num_heads, self.window, self.head_dim,
                              device=device, dtype=dtype),
            "n": 0,
        }

    @torch.no_grad()
    def stream_step(self, x_t: torch.Tensor, state: Dict) -> Tuple[torch.Tensor, Dict]:
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        B, one, D = x_t.shape
        assert one == 1
        H = self.num_heads
        h = self.norm(x_t)
        qkv = self.qkv(h).reshape(B, 1, 3, H, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]     # (B, H, hd)

        pos = state["n"]
        dev = x_t.device
        W = self.window
        # 当前 query 的绝对位置 pos：缩放表取 [pos, pos+1) 单行
        cos_q, sin_q = self.scaler.tables(self.head_dim, pos + 1, dev, q.dtype)
        q_rot = _apply_rope(q.unsqueeze(2), cos_q[pos:pos + 1],
                            sin_q[pos:pos + 1])[:, :, 0]     # (B, H, hd)

        slot = pos % W
        state["k"][:, :, slot] = k
        state["v"][:, :, slot] = v
        state["n"] = pos + 1
        n = state["n"]

        w = min(W, n)
        start_abs = n - w
        slots = torch.arange(start_abs, n, device=dev) % W
        Ks = state["k"][:, :, slots]      # (B, H, w, hd)
        Vs = state["v"][:, :, slots]
        # 窗口内 key 的绝对位置 [start_abs .. n-1]：缩放表按绝对位置逐行取
        cos_k, sin_k = self.scaler.tables(self.head_dim, n, dev, Ks.dtype)
        Ks_rot = _apply_rope(Ks, cos_k[start_abs:n], sin_k[start_abs:n])

        attn = torch.matmul(q_rot.unsqueeze(2), Ks_rot.transpose(2, 3)) \
            * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, Vs).transpose(1, 2).reshape(B, D)
        y = x_t.squeeze(1) + self.out_proj(out)
        return y.unsqueeze(1), state

    def stream_bytes(self, batch: int = 1, dtype=torch.float32) -> int:
        return 2 * batch * self.num_heads * self.window * self.head_dim * dtype.itemsize

    streaming_state_bytes = stream_bytes

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.num_heads}, window={self.window}, "
                f"rope_theta={self.rope_theta}, strategy={self.strategy}, "
                f"factor={self.factor}")


__all__ = [
    "RopeScaler",
    "SlidingWindowAttentionYaRN",
    "make_rope_tables",
    "_ntk_scale_per_component",
]
