# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 基础层：RMSNorm、FFN 与初始化工具 (无注意力 / 无 RoPE)。

全部基于 PyTorch 2.1 公共 API 实现，并兼容 ``torch.compile`` / FX 符号追踪
(不依赖 ``torch.Tensor.backward`` 之外的动态控制流，避免图断裂)。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """手写 RMSNorm (数值稳定，避免引入 mean 偏置)。

    Args:
        x: (..., H) 输入。
        weight: (H,) 可学习缩放。
        eps: 稳定常数。

    Returns:
        (..., H) 归一化结果。
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


class RMSNorm(nn.Module):
    """RMSNorm 层。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在 float32 下计算统计量以提升稳定性，结果还原原 dtype
        input_dtype = x.dtype
        x = x.to(torch.float32)
        out = rms_norm(x, self.weight.float(), self.eps)
        return out.to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


# ----------------------------------------------------------------------
# FFN
# ----------------------------------------------------------------------
class FFN(nn.Module):
    """SwiGLU 风格前馈网络 (更稳的激活，便于 compile)。

    hidden -> up -> activation -> down
    """

    def __init__(self, hidden_size: int, intermediate_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inter = int(round(hidden_size * intermediate_ratio))
        self.fc1 = nn.Linear(hidden_size, inter, bias=False)
        self.fc2 = nn.Linear(hidden_size, inter, bias=False)  # gate
        self.fc3 = nn.Linear(inter, hidden_size, bias=False)  # down
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc2(x)
        act = F.silu(self.fc1(x)) * gate
        act = self.dropout(act)
        return self.fc3(act)


# ----------------------------------------------------------------------
# 初始化
# ----------------------------------------------------------------------
def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """统一的参数初始化 (正态初始化线性/嵌入，norm 置 1)。"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
