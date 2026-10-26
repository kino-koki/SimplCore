# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""门控谱全局卷积 (Gated SGC) —— 为 SGC-V2 补偿内容相关路由的轻量扩展.

背景
----
SGC-V2 (因果全局卷积) 使用位置相对核 K[t-s] 做因果混合, 复杂度 O(n log n),
但在高熵真实语言建模上因无法按内容路由而明显过拟合 (test ppl 11.10 vs 6.16)。

本模块在 V2 基础上引入一个轻量门控头, 使模型能按内容动态调整卷积核的
"有效参与度" (类似 Mamba 的选择性机制, 但保持位置相对核的复杂度优势):

    gate_t = sigmoid(W_g x_t)          # (B, L, D) 内容相关门控
    y_t    = gate_t * y_t^sgc           # 门控后的输出

复杂度: 门控为 O(B·L·D²) 的线性层, 不破坏 O(n log n) 因果卷积主导项。

这是论文 4.5 节"门控消融"的核心实现, 用于验证: 能否通过轻量门控
缓解 SGC-V2 在高熵真实语料上的过拟合 (引理3 的出路)。

修复说明
--------
原实现 gate 作用于 y_sgc (含残差), 导致门控效果被残差冲淡。
修复: 让 gate 作用于 SGC 输出与输入的差值 (即门控只调制"混合贡献"),
残差恒等保留, 这样门控才能真正发挥内容路由作用。

    y_sgc_diff = y_sgc - residual            # 差值 (混合贡献)
    y_gated    = y_sgc_diff * gate           # 门控调制混合贡献
    out        = residual + out_scale * W_out(y_gated)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

from .spectral_global_conv import SpectralGlobalConvV2


class GatedSGCV2(nn.Module):
    """SGC-V2 + 轻量内容门控头 (修复版).

    核心公式:
        u       = W_in · Norm(x)                          # 输入投影
        y_sgc   = CausalConv_K(u)                         # SGC-V2 因果全局卷积 (含残差)
        diff    = y_sgc - x                               # 混合贡献差值
        gate    = sigmoid(W_gate · Norm(x))               # 内容相关门控 (B,L,D)
        out     = residual + out_scale · W_out(diff * gate)

    门控使用 sigmoid 将每个位置/通道的值压到 (0,1), 使模型能按内容
    动态决定哪些位置/通道"参与"卷积输出, 补偿位置相对核无法路由的缺陷。

    初始化: gate_bias=0 (即 sigmoid(0)=0.5, 门控初始为"半开半关"),
    让训练起点接近 V2 但留出学习空间。
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 512,
        rank: int = 8,
        gate_bias: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim > 0 and max_len >= 1
        self.dim = dim
        self.max_len = max_len
        self.rank = rank
        self.gate_bias = gate_bias
        self.eps = eps

        # 复用 SGC-V2 的因果全局卷积
        self.sgc = SpectralGlobalConvV2(
            dim=dim, kernel_size=3, max_len=max_len, rank=rank, eps=eps
        )

        # 门控头: 从输入投影到通道门控
        self.gate_proj = nn.Linear(dim, dim, bias=True)

        # 初始化为 gate_bias (sigmoid(gate_bias) = 0.5 当 bias=0)
        with torch.no_grad():
            self.gate_proj.bias.data[:] = gate_bias

        # 门控输出缩放 (初始为 1, 让门控几乎不影响 V2 输出)
        self.gate_scale = nn.Parameter(torch.ones(1))

        # 输出投影 (复用 V2 的)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # SGC-V2 因果全局卷积 (位置相对, 含残差)
        y_sgc = self.sgc(x)  # (B, L, D) 含残差

        # 差值: 混合贡献 (去掉残差)
        diff = y_sgc - residual  # (B, L, D)

        # 内容相关门控
        gate_logits = self.gate_proj(x)  # (B, L, D)
        gate = torch.sigmoid(gate_logits) * self.gate_scale  # (B, L, D)

        # 门控调制混合贡献 (只调制差值, 残差恒等保留)
        y_gated = diff * gate

        # 输出投影 + 残差
        out = self.out_proj(y_gated)
        return residual + self.out_scale * out


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("Gated SGC-V2 自检 (修复版)")
    print("=" * 60)

    layer = GatedSGCV2(dim=32, max_len=64, rank=8)
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

    print("\nGated SGC-V2 smoke test PASSED")
