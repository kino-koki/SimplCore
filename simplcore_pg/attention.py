# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""[基准对比模块 · 非架构组成部分]

自注意力子层 (带 RoPE 与 Flash/SDPA 后端)。

⚠️ 本模块 **不属于** SimplCore 架构。SimplCore v2.0 起已完全放弃
Transformer/自注意力路线，主干算子为 LocalInteraction + ResonanceField +
DynamicMemoryChannels + MoE，**无 Q/K/V/RoPE/SDPA、无 KV Cache**。

本文件仅作为「复杂度阶数对比」的外部基准，被 ``bench_throughput.py`` 用来
量化标准全注意力相对 RFP/局部算子的 O(n²) 劣势。它已从 ``simplcore_pg``
公开 API 中移除，不会被任何生产模型/流式推理路径导入。请勿在新架构代码中使用。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import RMSNorm


class SelfAttention(nn.Module):
    """带 RoPE 的多头自注意力。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除。")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = RMSNorm(hidden_size, eps=eps)

        # RoPE 逆频率表 (head_dim/2,)；cos/sin 在前向中按实际 L 动态计算，
        # 既避免对缓冲区做动态切片，又兼容 torch.compile / FX 的符号形状。
        inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        seq_len: int | None = None,
        causal: bool = False,
        past_key_value: tuple | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple:
        """自注意力。

        Args:
            x: (B, L, H)。
            attention_mask: 可选 (B, L) bool，True 为有效位置；用于 padding 屏蔽。
            seq_len: 可选显式序列长度。若提供 (例如在 FX symbolic_trace 的
                ``concrete_args`` 下由具体 ``input_ids`` 推导出的常量)，则直接用于
                RoPE 表计算，避免对符号形状调用 ``torch.arange``；缺省时由 ``x.shape``
                推导，兼容 eager / ``torch.compile`` (符号 SymInt)。
            causal: 是否施加因果掩码 (自回归训练/生成)。
            past_key_value: KV 缓存 (k, v)，用于增量生成 (KV Cache)。
            use_cache: 返回 (out, (k, v)) 以便上层串联缓存。

        Returns:
            默认 (B, L, H)；当 ``use_cache`` 或 ``past_key_value`` 时返回
            ``(out, (k_full, v_full))``。
        """
        torch._assert(x.dim() == 3, "SelfAttention 期望 3D 输入 (B, L, H)")
        B, L, H = x.shape
        if seq_len is None:
            seq_len = L
        torch._assert(seq_len <= self.max_seq_len,
                      f"序列长度 {seq_len} 超过预计算 max_seq_len {self.max_seq_len}")

        h = self.norm(x)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE：增量生成时按绝对位置偏移计算，保证与全量前向一致
        offset = past_key_value[0].shape[2] if past_key_value is not None else 0
        cos, sin = self._rope_table(seq_len, x.dtype, x.device, offset=offset)
        q, k = apply_rotary(q, k, cos, sin)

        # KV 缓存拼接
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v)

        # 构造 SDPA 的 attn_mask
        q_len = q.shape[2]
        sdpa_mask = self._build_mask(x, attention_mask, causal, k.shape[2], q_len)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=self.attn_dropout.p if isinstance(self.attn_dropout, nn.Dropout) else 0.0
        )
        out = out.transpose(1, 2).reshape(B, L, H)
        out = self.o_proj(out)
        if use_cache or past_key_value is not None:
            return x + out, present
        return x + out

    def _build_mask(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
        causal: bool,
        full_len: int,
        q_len: int | None = None,
    ):
        """构造 SDPA 的 attn_mask (float, -inf 为屏蔽)，支持 padding + 因果。

        当 ``q_len`` 与 ``full_len`` 不同 (局部窗口注意力：query 为当前块、
        key 含历史窗口) 时，构造 (B, 1, q_len, full_len) 因果掩码：
        当前块的 query 全局位置 = ``full_len - q_len + i``，仅允许 attend
        到 key 全局位置 <= 该值。
        """
        if not causal and attention_mask is None:
            return None
        B, L, H = x.shape
        if q_len is None:
            q_len = full_len
        dtype = x.dtype
        device = x.device
        # 当前块 query 的全局起始位置
        q_start = full_len - q_len
        mask = torch.zeros(1, 1, q_len, full_len, dtype=dtype, device=device)
        if causal:
            # query 全局位置 = q_start + i；key 全局位置 = j；屏蔽 j > q_start + i
            q_pos = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(1) + q_start
            k_pos = torch.arange(full_len, dtype=torch.long, device=device).unsqueeze(0)
            causal_m = torch.where(k_pos > q_pos, float("-inf"), 0.0).to(dtype)
            mask = mask + causal_m.unsqueeze(0)
        if attention_mask is not None:
            torch._assert(attention_mask.shape == (B, L), "attention_mask 形状应为 (B, L)")
            # pad: (B,1,1,L)，与 (1,1,q_len,full_len) 相加广播为 (B,1,q_len,full_len)
            pad = torch.where(
                attention_mask.unsqueeze(1).unsqueeze(2),
                torch.zeros(1, dtype=dtype, device=device),
                torch.full((1,), float("-inf"), dtype=dtype, device=device),
            )
            mask = mask + pad
        return mask

    def _rope_table(self, L: int, dtype: torch.dtype, device: torch.device, offset: int = 0):
        """按实际序列长度 L 计算 RoPE 的 cos/sin 表 (L, head_dim)。

        设计要点 (FX / torch.compile 兼容)：
          * ``L`` 在 symbolic_trace 的 ``concrete_args`` 下为具体 ``int``，
            在 ``torch.compile(dynamic=True)`` 下为 ``SymInt``；二者均支持
            ``torch.arange``，因此不再对 ``L`` 做任何数据相关分支。
          * ``arange`` 使用具体 ``float32`` 类型，避免把符号 ``dtype``/``device``
            (Proxy) 作为 ``arange`` 的关键字参数 (否则 symbolic_trace 报错)；
            末尾通过 ``.to(dtype).to(device)`` 对齐到输入张量的实际类型/设备。
          * ``offset`` 为增量生成时的绝对位置起点。
        """
        t = torch.arange(offset, offset + L, dtype=torch.float32)  # 具体类型，符号形状安全
        freqs = torch.outer(t, self.inv_freq)          # (L, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)         # (L, head_dim)
        cos = emb.cos().to(dtype).to(device)
        sin = emb.sin().to(dtype).to(device)
        return cos, sin


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """对 (B, heads, L, D) 的 q/k 施加 RoPE。"""
    def _rot(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin
    return _rot(q), _rot(k)
