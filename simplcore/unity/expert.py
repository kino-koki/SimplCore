# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity 专家：per-token 小 MLP（域专属，数据必须不同）。

每个专家是一个 O(1) 状态、非注意力的小参数模块，与 SGC 哲学一致：
不引入注意力、不破坏 ``purity.py`` 的无注意力约束，4M 超长上下文下状态仍 O(1)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.sgc import make_norm


class ExpertMLP(nn.Module):
    """单层 gated MLP（norm → silu(gate)⊙up → down + 残差）。"""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0,
                 norm: str = "rmsnorm", eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = make_norm(norm, dim, eps=eps)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.out_scale = nn.Parameter(torch.ones(1))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        act = F.silu(self.gate(x)) * self.up(x)
        act = self.drop(act)
        return h + self.out_scale * self.down(act)


class Expert(nn.Module):
    """一个小参数专家：``ExpertMLP`` 栈。

    域专属，训练数据必须与其他专家不同。这是「数个小模型合体」中的「一个小模型」。
    """

    def __init__(self, dim: int, hidden: int, layers: int = 2,
                 dropout: float = 0.0, norm: str = "rmsnorm",
                 eps: float = 1e-6, name: str = "expert") -> None:
        super().__init__()
        self.dim = dim
        self.name = name
        self.layers = nn.ModuleList(
            [ExpertMLP(dim, hidden, dropout, norm, eps) for _ in range(max(1, int(layers)))])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h
        for layer in self.layers:
            x = layer(x)
        return x

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
