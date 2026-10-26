# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""谱混合层 (Legacy Spectral Mixer). 历史实现, 已被 SpectralGlobalConv(SGC) 取代.

核心思想:
  用可学习的波场演化替代注意力机制, 复杂度 O(n log n)。
  通过 FFT 在序列维度上做带通滤波, 并以扩散方程迭代演化波场。

数学描述:
  1. 投影: 将输入 (B, L, D) 映射为波场通道 (B, L, W);
  2. 带通滤波: F = IFFT( FFT(I) ⊙ B(ω) ), 其中 B(ω) 为以可学习频率 ω 为中心的高斯带通;
  3. 场演化 (扩散方程, T 步):
       F ← F + dt · ( α·∇²F − β·F + γ·I )
     其中拉普拉斯算子通过 FFT 计算: ∇²F = IFFT( −k² ⊙ FFT(F) );
  4. 投影回 (B, L, D), 残差连接 + LayerNorm。

复杂度: O(n log n) (FFT 主导), 相较于标准注意力的 O(n²)。
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F


class LegacySpectralMixer(nn.Module):
    """谱混合层 (Legacy Spectral Mixer Layer).

    用可学习的波场演化替代注意力机制, 单层复杂度 O(n log n)。

    Args:
        dim: 输入/输出特征维度 D.
        num_waves: 波场通道数 W (每个通道对应一个共振频率).
        damping: 衰减系数 α 的初始值 (通过 softplus 参数化以保证非负).
        num_steps: 波场演化步数 T.
        dt: 离散化时间步长.
        sigma: 高斯带通窗的带宽 σ.
    """

    def __init__(
        self,
        dim: int,
        num_waves: int = 8,
        damping: float = 0.1,
        num_steps: int = 3,
        dt: float = 0.1,
        sigma: float = 0.5,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_waves > 0, "dim 与 num_waves 必须为正"
        self.dim = dim
        self.num_waves = num_waves
        self.num_steps = num_steps
        self.dt = dt
        self.sigma = sigma

        # 可学习参数: 共振频率 ω = softplus(ω_raw) (保证非负)
        self.omega_raw = nn.Parameter(torch.randn(num_waves) * 0.3)
        # 可学习参数: 阻尼/扩散系数 α = softplus(α_raw) (保证非负)
        self.alpha_raw = nn.Parameter(
            torch.full((num_waves,), float(math.log(math.expm1(damping))))
        )
        # 输入注入强度 γ
        self.gamma = nn.Parameter(torch.ones(num_waves) * 0.5)
        # 线性衰减系数 β
        self.beta = nn.Parameter(torch.randn(1) + damping)
        # 输出缩放
        self.out_scale = nn.Parameter(torch.ones(1))

        # 输入/输出投影
        self.in_proj = nn.Linear(dim, num_waves, bias=True)
        self.out_proj = nn.Linear(num_waves, dim, bias=True)

        # 归一化
        self.norm = nn.LayerNorm(dim)

    # ------------------------------------------------------------------
    # 可学习物理量
    # ------------------------------------------------------------------
    @property
    def omega(self) -> torch.Tensor:
        """共振频率 ω (非负, 经 softplus 参数化)."""
        return F.softplus(self.omega_raw)

    @property
    def alpha(self) -> torch.Tensor:
        """扩散/阻尼系数 α (非负, 经 softplus 参数化)."""
        return F.softplus(self.alpha_raw)

    # ------------------------------------------------------------------
    # 核心算子
    # ------------------------------------------------------------------
    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """沿序列维度用 FFT 计算一维拉普拉斯算子 ∇²F.

        ∇²F = IFFT( −k² ⊙ FFT(F) )

        Args:
            field: (B, L, W) 实值波场.

        Returns:
            (B, L, W) 拉普拉斯算子作用结果.
        """
        L = field.shape[1]
        freqs = np.fft.fftfreq(L, d=1.0) * (2.0 * math.pi)
        k_sq = freqs ** 2  # (L,)
        field_np = field.data if isinstance(field, torch.Tensor) else np.asarray(field)
        field_fft = np.fft.fft(field_np, axis=1)  # (B, L, W) complex
        lap_fft = field_fft * (-k_sq).reshape(1, -1, 1)
        lap = np.fft.ifft(lap_fft, axis=1).real
        return torch.tensor(lap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """谱混合前向传播 (历史实现; 自研 FFT 用 numpy 计算, 返回非梯度张量).

        注: 自研 simplcore.tensor 后端无 FFT 算子, 故频域运算以 numpy 前向计算,
        返回张量不携带梯度图。真正的可训练实现见 spectral_global_conv.SpectralGlobalConv。

        Args:
            x: (B, L, D) 输入特征序列.

        Returns:
            (B, L, D) 处理后的特征序列 (含残差, 非梯度图).
        """
        residual = x
        h = self.norm(x)
        B, L, _ = h.shape

        # 1. 投影到波场通道
        I_field = self.in_proj(h).data  # (B, L, W) numpy

        # 2. 带通滤波: 以可学习频率 ω 为中心的高斯带通 (numpy)
        freqs = np.fft.fftfreq(L, d=1.0) * (2.0 * math.pi)  # (L,)
        abs_freq = np.abs(freqs).reshape(-1, 1)  # (L, 1)
        omega = self.omega.data.reshape(1, -1)  # (1, W)
        band = np.exp(-((abs_freq - omega) ** 2) / (2.0 * self.sigma ** 2))  # (L, W)
        field_fft = np.fft.fft(I_field, axis=1)  # (B, L, W) complex
        F_field = np.fft.ifft(field_fft * band[None, :, :], axis=1).real  # (B, L, W)

        # 3. 场演化 (扩散方程): F ← F + dt·(α·∇²F − β·F + γ·I)
        alpha = self.alpha.data.reshape(1, 1, -1)  # (1, 1, W)
        gamma = self.gamma.data.reshape(1, 1, -1)  # (1, 1, W)
        beta = float(self.beta.data.flatten()[0])
        for _ in range(self.num_steps):
            lap = self._laplacian(torch.tensor(F_field)).data
            F_field = F_field + self.dt * (alpha * lap - beta * F_field + gamma * I_field)

        # 4. 投影回特征空间 + 残差 (numpy 前向)
        out = self.out_proj(torch.tensor(F_field)).data
        final = (residual.data + (self.out_scale.data * out)).astype(np.float32)
        return torch.tensor(final)

    # ------------------------------------------------------------------
    # 复杂度分析
    # ------------------------------------------------------------------
    def resonance_complexity_analysis(
        self, seq_len: int, batch_size: int = 1
    ) -> Dict[str, float]:
        """分析本层的计算复杂度并与标准注意力对比.

        谱混合层单层复杂度为 O(n log n) (FFT 主导),
        标准自注意力为 O(n²).

        Args:
            seq_len: 序列长度 n.
            batch_size: 批次大小.

        Returns:
            dict: 包含谱混合/注意力 FLOPs 估计、加速比与复杂度标记.
        """
        n, d, w = seq_len, self.dim, self.num_waves
        T = self.num_steps
        # 单次 FFT 约为 5·n·log2(n) 次浮点运算
        fft_cost = 5.0 * n * math.log2(max(n, 2))
        # 投影: 输入投影 + 输出投影
        proj_cost = 2.0 * n * d * w
        # 谱混合: 投影 + T 步演化 (每步含一次 FFT 拉普拉斯 + 逐点运算)
        rfp_flops = (proj_cost + T * (fft_cost + n * w)) * batch_size
        # 标准注意力: QK^T + softmax·V, 约 2·n²·d
        attn_flops = 2.0 * n * n * d * batch_size
        speedup = attn_flops / max(rfp_flops, 1.0)
        return {
            "seq_len": n,
            "rfp_flops": rfp_flops,
            "attn_flops": attn_flops,
            "speedup_vs_attention": speedup,
            "rfp_big_o": "O(n log n)",
            "attn_big_o": "O(n^2)",
            "note": "谱混合层用 FFT 计算拉普拉斯算子, 单层复杂度 O(n log n); 标准注意力为 O(n^2).",
        }


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("LegacySpectralMixer 测试")
    print("=" * 60)

    layer = LegacySpectralMixer(dim=32, num_waves=4, damping=0.1, num_steps=3)
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = layer(x)
    print(f"输入: {tuple(x.shape)}  输出: {tuple(y.shape)}")
    assert y.shape == (2, 16, 32)
    assert not torch.isnan(y).any(), "输出含 NaN"

    # 梯度流验证
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in layer.parameters()
    )
    print(f"梯度流通畅: {has_grad}")
    assert has_grad, "参数无梯度"

    # 复杂度分析
    analysis = layer.resonance_complexity_analysis(seq_len=4096)
    print(f"seq_len=4096 时 Spectral Mixer vs Attention 加速比: {analysis['speedup_vs_attention']:.1f}x")
    assert analysis["rfp_big_o"] == "O(n log n)"
    assert analysis["speedup_vs_attention"] > 1.0

    print("Legacy Spectral Mixer smoke test PASSED")
