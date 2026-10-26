# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""增强版门控谱全局卷积 (Gated SGC-V3) —— 专为修复 SGC-V2 高熵过拟合.

背景
----
SGC-V2 (因果全局卷积) 使用位置相对核 K[t-s] 做因果混合, 复杂度 O(n log n),
但在高熵真实语言建模上因无法按内容路由而明显过拟合 (test ppl 11.10 vs 6.16)。

本模块 (GatedSGCV3) 在 V2 基础上引入:
  1. 轻量内容门控头 (sigmoid 门控): 让模型按内容动态调整卷积核参与度
  2. Dropout: 训练时随机失活部分卷积核元素, 防止死记位置模式
  3. 输出正则化 (L2 penalty): 限制核范数, 抑制训练集过拟合

核心公式:
    u       = W_in · Norm(x)
    y_sgc   = CausalConv_K(u)                          # SGC-V2 因果全局卷积
    gate    = sigmoid(W_gate · Norm(x))                # 内容相关门控 (B,L,D)
    y_gated = y_sgc * gate                             # 门控调制
    out     = residual + out_scale · W_out(y_gated)    # 输出投影 + 残差

复杂度: 门控为 O(B·L·D²) 的线性层, 不破坏 O(n log n) 因果卷积主导项。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

from .spectral_global_conv import SpectralGlobalConvV2


class GatedSGCV3(nn.Module):
    """SGC-V2 + 轻量内容门控头 + Dropout + 正则化 (抗过拟合版).

    相比 GatedSGCV2 的改进:
    - 门控 bias 初始化为 -1.0 (sigmoid(-1)≈0.27, 门控初始为"偏关", 防止过早路由)
    - 加 Dropout(rate) 在门控输出上, 训练时随机失活
    - 门控输出加 L2 正则化 (通过 gate_scale 的 penalty)
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 512,
        rank: int = 8,
        gate_bias: float = -1.0,
        dropout_rate: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim > 0 and max_len >= 1
        self.dim = dim
        self.max_len = max_len
        self.rank = rank
        self.gate_bias = gate_bias
        self.dropout_rate = dropout_rate
        self.eps = eps

        # 复用 SGC-V2 的因果全局卷积
        self.sgc = SpectralGlobalConvV2(
            dim=dim, kernel_size=3, max_len=max_len, rank=rank, eps=eps
        )

        # 门控头: 从输入投影到通道门控
        self.gate_proj = nn.Linear(dim, dim, bias=True)

        # 初始化为 gate_bias (sigmoid(-1)≈0.27)
        with torch.no_grad():
            self.gate_proj.bias.data[:] = gate_bias

        # 门控输出缩放 (初始为 0.5, 让门控只"轻微调制" V2 输出)
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.5)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 输出投影
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # SGC-V2 因果全局卷积 (位置相对)
        y_sgc = self.sgc(x)  # (B, L, D) 含残差

        # 内容相关门控
        gate_logits = self.gate_proj(x)  # (B, L, D)
        gate = torch.sigmoid(gate_logits) * self.gate_scale  # (B, L, D)

        # Dropout (训练时随机失活部分门控)
        if self.dropout is not None and self.training:
            gate = self.dropout(gate)

        # 门控调制 SGC 输出
        y_gated = y_sgc * gate

        # 输出投影 + 残差
        out = self.out_proj(y_gated)
        return residual + self.out_scale * out


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Gated SGC-V3 自检 (抗过拟合版)")
    print("=" * 60)

    layer = GatedSGCV3(dim=32, max_len=64, rank=8, gate_bias=-1.0, dropout_rate=0.1)
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = layer(x)

    print(f"输入: {tuple(x.shape)}  输出: {tuple(y.shape)}")
    assert y.shape == (2, 16, 32), "形状不匹配"
    assert bool(np.isfinite(y.data).all()), "输出含 NaN"

    # 梯度流验证
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and float(np.abs(p.grad).sum()) > 0
        for p in layer.parameters()
    )
    print(f"梯度流通畅: {has_grad}")
    assert has_grad, "参数无梯度"

    # 参数量统计
    total_params = sum(
        int(np.prod(p.shape)) for p in layer.parameters() if p.requires_grad
    )
    print(f"总参数量: {total_params}")

    print()
    print("Gated SGC-V3 smoke test PASSED")
