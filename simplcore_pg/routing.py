# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""拓扑感知路由 (TopologyAwareRouter, TAR)。

轻量级条件计算 / 稀疏路由：维护 P 条可学习的"计算路径"变换，按输入动态选择
top-k 条路径并以归一化权重组合，实现近似 MoE 的表征容量提升而保持低成本。

特性：
  * 训练期对路由 logits 做 dropout 以缓解路径坍缩；
  * 温度可学习，软化路由分布；
  * 无数据相关控制流，兼容 ``torch.compile`` / FX。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm


class TopologyAwareRouter(nn.Module):
    """拓扑感知路由层。"""

    def __init__(
        self,
        hidden_size: int,
        num_paths: int = 4,
        topk: int = 2,
        temperature_init: float = 1.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not (1 <= topk <= num_paths):
            raise ValueError("topk 必须在 [1, num_paths]。")
        self.hidden_size = hidden_size
        self.num_paths = num_paths
        self.topk = topk
        self.eps = eps

        self.router = nn.Linear(hidden_size, num_paths, bias=True)
        self.paths = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_paths)]
        )
        self.temperature = nn.Parameter(torch.tensor(float(temperature_init)))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = RMSNorm(hidden_size, eps=eps)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.router.bias)  # 初始路由均匀
        for p in self.paths:
            nn.init.normal_(p.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, *, training: bool = False) -> torch.Tensor:
        """路由并组合路径。

        Args:
            x: (B, L, H)。
            training: 是否训练 (启用路由 dropout)。

        Returns:
            (B, L, H) 路由增强后的隐状态。
        """
        torch._assert(x.dim() == 3, "TAR 期望 3D 输入 (B, L, H)")
        h = self.norm(x)

        logits = self.router(h)                       # (B, L, P)
        if training and not isinstance(self.dropout, nn.Identity):
            logits = self.dropout(logits)
        temp = F.softplus(self.temperature).clamp_min(1e-2)  # > 0

        # top-k 路由
        topk_logits, topk_idx = torch.topk(logits, self.topk, dim=-1)  # (B,L,k)
        topk_w = torch.softmax(topk_logits / temp, dim=-1)            # (B,L,k)

        # 计算所有路径输出并收集 top-k (P, B, L, H)
        all_paths = torch.stack([p(h) for p in self.paths], dim=0)
        gathered = self._gather_paths(all_paths, topk_idx)          # (B,L,k,H)
        out = (gathered * topk_w.unsqueeze(-1)).sum(dim=2)          # (B,L,H)
        return x + out

    def _gather_paths(self, all_paths: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """按 top-k 索引从各路径输出中收集 (FX/compile 安全，无 Python 循环/arange)。

        Args:
            all_paths: (P, B, L, H) 所有路径输出。
            idx: (B, L, k) top-k 路径索引。

        Returns:
            (B, L, k, H) 收集后的路径输出。
        """
        # all_paths: (P, B, L, H)。将 idx 转置为 (k, B, L)，沿路径维 (dim=0) gather，
        # 再转置回 (B, L, k, H)。整个流程无 Python 循环 / arange，符号形状安全。
        idx_perm = idx.permute(2, 0, 1)                                  # (k, B, L)
        idx_exp = idx_perm.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)  # (k, B, L, H)
        gathered = torch.gather(all_paths, 0, idx_exp)                  # (k, B, L, H)
        return gathered.permute(1, 2, 0, 3)                             # (B, L, k, H)

    def extra_repr(self) -> str:
        return f"H={self.hidden_size}, paths={self.num_paths}, topk={self.topk}"
