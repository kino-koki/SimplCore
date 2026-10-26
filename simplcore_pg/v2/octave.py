# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""L2 Octave Memory 八度分层压缩记忆 (TriMemory 远程记忆)。

核心机制：
  * **写入**：每 ``segment_tokens`` 个隐状态经段压缩器 (perceiver) 压成
    ``mem_tokens`` 个 memory token，追加到第 0 层；
  * **升格 (promote)**：第 k 层槽满 (``slots_per_level``) 时，将其全部内容
    再压缩为 ``promote_tokens`` 个 token 推入第 k+1 层，第 k 层清空滚动。
    第 k 层单个 token 覆盖 ``segment × (slots/promote)^k`` 个原始位置 ——
    **越久远的历史分辨率越低**，层数随 log(上下文长度) 增长；
  * **预算恒定**：每层槽数固定 → 总显存 = 层数 × 单层预算，
    即"上下文每翻倍，+1 层固定预算"的 O(log L) 增长；
  * **读取**：query 对所有层拼接内容做 cross-attention，层级偏置
    (近层优先) 可学习，模拟记忆衰退曲线。

训练态读取对 query 投影可微；记忆库内容在块边界写入并 detach (TBPTT)，
与 v1 ``MemoryState`` 的"训练可微 / 推理运行时"分离设计一致。

**诚实声明**：远程记忆是有损压缩检索。固定预算意味着超远程信息的召回
存在理论上界；显著性保护段 (不压缩) 属于后续工程项 (见设计文档 §4.5)。
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import RMSNorm, FFN, init_weights


# ----------------------------------------------------------------------
# 段压缩器：S 个隐状态 -> M 个 memory token
# ----------------------------------------------------------------------
class SegmentCompressor(nn.Module):
    """perceiver 式段压缩：cross-attn (M 个可学习 query <- 段内 token) + FFN。"""

    def __init__(self, hidden_size: int, mem_tokens: int, num_heads: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.mem_tokens = mem_tokens
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.queries = nn.Parameter(torch.zeros(mem_tokens, hidden_size))
        nn.init.normal_(self.queries, std=0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(hidden_size, eps=eps)
        self.f_norm = RMSNorm(hidden_size, eps=eps)
        self.ffn = FFN(hidden_size, intermediate_ratio=2.0)
        self.ffn_norm = RMSNorm(hidden_size, eps=eps)
        self.apply(lambda m: init_weights(m, std=0.02))

    def forward(self, segment: torch.Tensor) -> torch.Tensor:
        """segment: (B, S, H) -> (B, M, H)。"""
        B, S, H = segment.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        qh = self.q_proj(self.q_norm(q)).view(B, self.mem_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        kh = self.k_proj(self.f_norm(segment)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        vh = self.v_proj(self.f_norm(segment)).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        ca = F.scaled_dot_product_attention(qh, kh, vh)
        ca = ca.transpose(1, 2).reshape(B, self.mem_tokens, H)
        q = q + self.o_proj(ca)
        q = q + self.ffn(self.ffn_norm(q))
        return q


# ----------------------------------------------------------------------
# 运行时记忆库 (推理期，不进计算图)
# ----------------------------------------------------------------------
class OctaveBankState:
    """八度分层记忆库的运行时状态。

    每层固定 ``slots_per_level`` 个槽 → 单层显存恒定；层数随 log(L) 增长。

    Attributes:
        levels: 每层内容张量列表，形状 (slots, H)（未写满时尾部为零）。
        counts: 每层已写入槽数。
        total_tokens_seen: 已压缩的原始 token 总数。
    """

    def __init__(self, slots_per_level: int, hidden_size: int, max_levels: int,
                 device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.slots = slots_per_level
        self.hidden_size = hidden_size
        self.max_levels = max_levels
        self.levels: List[torch.Tensor] = [
            torch.zeros(slots_per_level, hidden_size, device=device, dtype=dtype)
        ]
        self.counts: List[int] = [0]
        self.total_tokens_seen = 0

    @property
    def num_active_levels(self) -> int:
        return sum(1 for c in self.counts if c > 0) or 1

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.levels[: len(self.counts)])

    def reset(self) -> None:
        self.levels = [torch.zeros_like(self.levels[0])]
        self.counts = [0]
        self.total_tokens_seen = 0


class OctaveMemory(nn.Module):
    """八度分层压缩记忆层。

    Args:
        hidden_size: 模型维度 H。
        slots_per_level: 每层固定槽数。
        mem_tokens: 每段压成的 memory token 数。
        promote_tokens: 层满向上层推入的 token 数。
        segment_tokens: 每多少原始 token 压缩为一段。
        max_levels: 最大层数。
        num_heads: 读取注意力头数。
        eps: RMSNorm 稳定常数。
    """

    def __init__(
        self,
        hidden_size: int,
        slots_per_level: int = 64,
        mem_tokens: int = 4,
        promote_tokens: int = 4,
        segment_tokens: int = 32,
        max_levels: int = 16,
        num_heads: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除。")
        self.hidden_size = hidden_size
        self.slots = slots_per_level
        self.mem_tokens = mem_tokens
        self.promote_tokens = promote_tokens
        self.segment_tokens = segment_tokens
        self.max_levels = max_levels
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.compressor = SegmentCompressor(hidden_size, mem_tokens, num_heads, eps)
        self.promoter = SegmentCompressor(hidden_size, promote_tokens, num_heads, eps)

        # 读取: cross-attention (q <- 所有层拼接的记忆 token)
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.mem_norm = RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # 层级偏置 (近层优先, 可学习): 初始化为随层级温和衰减
        self.level_bias = nn.Parameter(torch.linspace(0.0, -1.0, max_levels))

        self.apply(lambda m: init_weights(m, std=0.02))

    # ------------------------------------------------------------------
    def init_bank(self, device: torch.device,
                  dtype: torch.dtype = torch.float32) -> OctaveBankState:
        return OctaveBankState(self.slots, self.hidden_size, self.max_levels, device, dtype)

    # ------------------------------------------------------------------
    # 写入 (运行时, 不进计算图)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def write_segment(self, segment: torch.Tensor, bank: OctaveBankState) -> None:
        """把一段隐状态 (S, H) 或 (B=1, S, H) 压缩写入第 0 层，必要时逐级升格。"""
        if segment.dim() == 2:
            segment = segment.unsqueeze(0)
        toks = self.compressor(segment)[0]          # (M, H)
        self._append(bank, 0, toks)
        bank.total_tokens_seen += segment.shape[1]

    def _append(self, bank: OctaveBankState, level: int, toks: torch.Tensor) -> None:
        """向第 level 层追加 token；槽满则升格到上一层后清空本层再写入。

        顶层 (max_levels-1) 溢出时执行**自压缩**：将全部内容再压为
        ``promote_tokens`` 个 token 写回本层 —— 固定预算的必然代价是
        最久远历史的分辨率持续降低 (有损，见模块 docstring 诚实声明)。
        """
        if level >= self.max_levels:
            level = self.max_levels - 1
        while len(bank.levels) <= level:
            bank.levels.append(torch.zeros_like(bank.levels[0]))
            bank.counts.append(0)
        n = toks.shape[0]
        if bank.counts[level] + n > self.slots:
            content = bank.levels[level][: bank.counts[level]]
            squeezed = self.promoter(content.unsqueeze(0))[0]   # (P, H)
            bank.levels[level].zero_()
            bank.counts[level] = 0
            if level < self.max_levels - 1:
                self._append(bank, level + 1, squeezed)         # 升格到上一层
            else:
                self._append(bank, level, squeezed)             # 顶层自压缩回写
        ptr = bank.counts[level]
        bank.levels[level][ptr: ptr + n] = toks.detach()
        bank.counts[level] = ptr + n

    # ------------------------------------------------------------------
    # 读取 (query 侧可微)
    # ------------------------------------------------------------------
    def read(
        self,
        x: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None,
        level_ids: Optional[torch.Tensor] = None,
        bank: Optional[OctaveBankState] = None,
    ) -> torch.Tensor:
        """cross-attention 读取八度记忆并残差融合。

        两种用法：
          * 训练/TBPTT：显式传入 ``memory_tokens`` (M_total, H) 与 ``level_ids``
            (M_total,) —— 通常来自上一块的 detached bank 快照；
          * 推理：传入 ``bank``，自动取其全部有效内容 (detached)。

        Args:
            x: (B, L, H)。
        Returns:
            (B, L, H)。
        """
        torch._assert(x.dim() == 3, "OctaveMemory 期望 3D 输入 (B, L, H)")
        B, L, H = x.shape
        if bank is not None:
            toks, lids = self._bank_snapshot(bank)
        elif memory_tokens is not None:
            toks, lids = memory_tokens, level_ids
        else:
            toks, lids = None, None

        if toks is None or toks.shape[0] == 0:
            return x  # 记忆为空：恒等 (零开销)

        toks = toks.to(device=x.device, dtype=x.dtype)
        M = toks.shape[0]
        h = self.norm(x)
        mh = self.mem_norm(toks)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(mh).view(M, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(mh).view(M, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.unsqueeze(0).expand(B, -1, -1, -1)              # (B, heads, M, hd)
        v = v.unsqueeze(0).expand(B, -1, -1, -1)

        # 层级偏置 (logit 单位): 近层 (level 小) 加分, 随层级温和衰减
        if lids is not None:
            bias = self.level_bias.to(x.dtype)[lids.long().clamp(0, self.max_levels - 1)]
            bias = bias.view(1, 1, 1, M)
        else:
            bias = 0.0
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5) + bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                           # (B, heads, L, hd)
        out = out.transpose(1, 2).reshape(B, L, H)
        return x + self.o_proj(out)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.read(x, **kwargs)

    # ------------------------------------------------------------------
    def _bank_snapshot(self, bank: OctaveBankState):
        """取记忆库全部有效内容: (tokens (M,H), level_ids (M,))。"""
        toks, lids = [], []
        for lv, cnt in enumerate(bank.counts):
            if cnt > 0:
                toks.append(bank.levels[lv][:cnt])
                lids.append(torch.full((cnt,), lv, dtype=torch.long,
                                       device=bank.levels[lv].device))
        if not toks:
            empty = torch.zeros(0, self.hidden_size)
            return empty, torch.zeros(0, dtype=torch.long)
        return torch.cat(toks, dim=0), torch.cat(lids, dim=0)

    # ------------------------------------------------------------------
    def estimate_level_bytes(self, dtype_bytes: int = 1) -> int:
        """单层显存预算 (字节)。"""
        return self.slots * self.hidden_size * dtype_bytes

    def extra_repr(self) -> str:
        return (f"H={self.hidden_size}, slots={self.slots}, M={self.mem_tokens}, "
                f"promote={self.promote_tokens}, seg={self.segment_tokens}")
