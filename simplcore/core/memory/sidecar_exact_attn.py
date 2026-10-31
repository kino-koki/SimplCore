# Copyright 2026 (C) 六安市青奈科技有限责任公司
# purity: allow-attention-reference
# -*- coding: utf-8 -*-
"""L2 取回的**局部精确注意力**实验件（EXPERIMENTAL，不进默认产品线）。

本文件是「指针 → top-k → 局部精确 cross-attention」的完整版实现：对取回的
k 个块（共 k·B 个 token）做标准 scaled dot-product cross-attention。

**为什么是独立文件 + 豁免标记**：架构纯洁性守卫（``arch/purity.py`` 三层
扫描）禁止产品线内出现注意力模块——这是 SimplCore 最硬的架构性质，不能为
一个特性开后门。因此：

* 默认通路走 ``sidecar.py`` 的**纯净注入**（指针加权池化 + 零初始化加性
  投影，与 CMT memory_in 同纪律）——信息精确性来自 L1 档案（逐 token
  精确副本），不来自注意力算子，通过 purity；
* 本文件带 ``purity: allow-attention-reference`` 标记，被源码扫描豁免，
  仅供「局部注意力 vs 池化注入」的对照实验（协议 X-sidecar-attn）。
  它不是产品线组件，不构成架构主张，未在真实任务上实测。

诚实边界：两版的检索质量差异（池化 vs 精确注意力）未实测；在对照协议
落地之前，不得宣称任一版"实现精确引用"。
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .sidecar import SidecarRetriever

__all__ = ["LocalExactCrossAttention"]


class LocalExactCrossAttention(nn.Module):
    """只对 top-k 块的精确 token 做 cross-attention（O(k·B) 键，与 L 无关）。

    Args:
        retriever: 已接好 L1 档案与指针头的 :class:`SidecarRetriever`。
        num_heads: 注意力头数（dim 必须整除）。

    与全量注意力的区别：键值集不是全部 L 个位置，而是指针选出的 k·B 个
    **精确 token**（从 L1 档案逐 token 读回、经模型 tok_emb 编码）——
    「精确到第几个字」的信息来自档案，注意力只负责在极小的键集上做对齐。
    """

    def __init__(self, retriever: SidecarRetriever, num_heads: int = 4) -> None:
        super().__init__()
        d = int(retriever.gist.embedding.embedding_dim)
        if d % num_heads:
            raise ValueError(f"dim {d} not divisible by num_heads {num_heads}")
        self.retriever = retriever
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, h: torch.Tensor, k: Optional[int] = None,
                need_weights: bool = False
                ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """h: (..., dim) 当前 O(1) 状态 → (..., dim) 取回向量。

        键集 = 指针 top-k 块的精确 token 嵌入（k·B 个），与 L 无关。
        """
        scores, ids = self.retriever.retrieve(h, k)
        emb = self.retriever.gist.embedding
        keys: List[torch.Tensor] = []
        for b in ids.reshape(-1).tolist():
            lo, hi = self.retriever.archive.block_span(int(b))
            t = self.retriever.archive.read(lo, hi).to(h.device).long()
            keys.append(emb(t).to(h.dtype))            # (B_i, dim)
        kv = torch.cat(keys, dim=0)                    # (k·B, dim)

        q = self.q_proj(h).unsqueeze(-2)               # (..., 1, dim)
        kd = self.k_proj(kv).unsqueeze(0)              # (1, k·B, dim)
        vd = self.v_proj(kv).unsqueeze(0)              # (1, k·B, dim)
        lead = q.dim() - 1
        for _ in range(lead):
            kd = kd.unsqueeze(0)
            vd = vd.unsqueeze(0)
        q2 = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim) \
              .transpose(-3, -2)                       # (..., H, 1, hd)
        k2 = kd.reshape(*kd.shape[:-1], self.num_heads, self.head_dim) \
               .transpose(-3, -2)
        v2 = vd.reshape(*vd.shape[:-1], self.num_heads, self.head_dim) \
               .transpose(-3, -2)
        # 手写 SDPA（本文件在 purity 定义级白名单内；nn.functional 的 SDPA
        # 不返回权重，而对照协议需要逐键权重做审计）
        att = q2 @ k2.transpose(-2, -1) / math.sqrt(self.head_dim)
        w = torch.softmax(att, dim=-1)                 # (..., H, 1, k·B)
        out = (w @ v2).transpose(-3, -2).reshape(*q.shape[:-1], -1)
        out = self.out_proj(out).squeeze(-2)
        return (out, w) if need_weights else out