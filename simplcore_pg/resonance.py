# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""共振场处理 (ResonanceField, RFP) —— 可训练的**因果**谱全局卷积。

.. warning::

   本文件在 2026 年审计中发现一个致命缺陷并已修复: 旧实现用
   ``rfft -> 乘以实值带通窗 -> irfft`` 做"谱滤波"。一个**实值(零相位)频率响应**
   对应的冲激响应必然关于 t=0 偶对称, 因此该算子**必然看到未来 token**。
   实测: 只扰动最后一个 token, 前面 511/511 个位置的输出全部改变
   (``max|Δ| = 2.74``); 关闭 RFP 后该泄漏恰好为 0.0。

   后果: 用旧 RFP 训练的自回归语言模型交叉熵是被"偷看答案"压低的,
   其困惑度数字(例如 ``val_ppl=1.28``)无效, 生成过程也不合法。

   修复方式: 把目标幅频响应 ``band(k)`` 通过逆 FFT 变回时域, 取出 **因果半边**
   (丢弃负时间半轴并补回增益), 得到因果 FIR 核, 再用 2L 零填充的 FFT 做
   **因果卷积**。这样既保留带通选择性与 O(L log L) 复杂度, 又严格因果
   (见 ``tests_simplcore_pg/test_causality.py``)。

   旧的零相位行为仍可用 ``causal=False`` 复现, 但明确标注为**非因果**,
   仅用于对照实验, 不得用于任何自回归训练或生成。

数学（因果版）：
  u = in_proj(x)                        # (B, L, W)
  h = causal_fir(band(omega, sigma))    # (L, W) 因果 FIR 核
  for s in 1..S:
      u = causal_conv(u, h) * damp
      u = (1-beta) * u + beta * u_0
  out = out_proj(u) * out_scale
  return norm(x + out)
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _grad_checkpoint

from .layers import RMSNorm


def band_to_causal_kernel(band: torch.Tensor, length: int) -> torch.Tensor:
    """Convert a real, non-negative magnitude response into a **causal** FIR kernel.

    A real frequency response corresponds to a symmetric (zero-phase, hence
    non-causal) impulse response. To obtain a causal filter with the same
    passband we use the classical window method:

    1. ``h = irfft(band, n=length)`` -- a real, circularly even signal whose
       index 0 is time 0, indices ``1..length//2`` are negative times and the
       remaining are the matching positive times;
    2. reorder so index ``m`` is time ``+m`` and keep only ``m >= 0``;
    3. double the amplitude, because the discarded negative-time half carried
       half of the (symmetric) energy, so the passband gain would otherwise halve.

    Args:
        band: (Lf, W) real non-negative magnitude response, ``Lf == length // 2 + 1``.
        length: sequence length L.
    Returns:
        (L, W) causal FIR kernel, differentiable w.r.t. ``band``.
    """
    h = torch.fft.irfft(band, n=length, dim=0)          # (L, W), even-symmetric
    if length == 1:
        return h
    # positive-time ordering: h[0], h[L-1], h[L-2], ..., h[1]
    pos = torch.cat([h[:1], h.flip(0)[: length - 1]], dim=0)
    return 2.0 * pos


def causal_spectral_conv(field: torch.Tensor, kernel: torch.Tensor,
                         length: int | None = None) -> torch.Tensor:
    """Depthwise **causal** convolution of ``field`` with a length-L kernel bank.

    ``y[b, t, w] = sum_{s <= t} kernel[t - s, w] * field[b, s, w]`` computed exactly
    with a zero-padded FFT of size 2L (no circular aliasing for a causal kernel of
    length <= L). O(L log L), and strictly future-blind.

    Args:
        field: (B, L, W).
        kernel: (L, W) causal FIR kernel bank.
        length: explicit sequence length. Must be a concrete ``int`` when the module
            is traced with ``torch.fx`` -- ``field.shape[1]`` is a Proxy there, and
            ``torch.fft.rfft`` requires a real integer ``n``.
    """
    B, L, W = field.shape
    if length is None:
        length = L
    N = 2 * length
    cX = torch.fft.rfft(field, n=N, dim=1)                     # (B, N/2+1, W)
    cK = torch.fft.rfft(kernel, n=N, dim=0).unsqueeze(0)       # (1, N/2+1, W)
    y = torch.fft.irfft(cX * cK, n=N, dim=1)                   # (B, N, W)
    return y[:, :length, :]


def _evolve_one_step(field: torch.Tensor, band: torch.Tensor, damp: torch.Tensor,
                     L: int, causal: bool = True) -> torch.Tensor:
    """单步谱演化（纯函数，供 checkpoint 重算）。

    Args:
        field: (B, L, W) 当前波场。
        band:  (Lf, W) 可学习带通幅频窗。
        damp:  标量阻尼系数（buffer，不参与梯度）。
        L:     序列长度（显式传入：FX 追踪下为具体常量，避免 Proxy 进入 rfft 的 n）。
        causal: True 时构造因果 FIR 并做因果卷积；False 时退化为旧的零相位滤波
                （**非因果**，仅供对照）。
    Returns:
        (B, L, W) 演化后波场。
    """
    if not causal:
        spec = torch.fft.rfft(field, n=L, dim=1)
        spec = spec * band.unsqueeze(0) * damp
        return torch.fft.irfft(spec, n=L, dim=1)
    kernel = band_to_causal_kernel(band, L)                    # (L, W)
    return causal_spectral_conv(field, kernel, length=L) * damp


class ResonanceField(nn.Module):
    """共振场处理层（可训练的**因果**谱全局卷积）。

    Args:
        hidden_size: 隐层宽度 H。
        num_waves: 波通道数 W。
        num_steps: 谱演化步数 S。
        damping/dt: 固定物理超参（**不参与梯度**，见 docstring 说明）。
        sigma: 带通窗初始带宽。
        max_seq_len: 频率轴上界（用于初始化 omega）。
        freeze_evolution: True 时演化走 no_grad、仅投影可学（环境回退用）。
        use_checkpoint: 训练时用梯度检查点包裹单步，规避 Windows/CPU FFT 反向
            workspace 累积。
        causal: 默认 True。**必须保持 True 才能用于自回归语言建模**；
            置 False 会得到非因果算子，仅用于复现历史对照结果。
    """

    def __init__(
        self,
        hidden_size: int,
        num_waves: int = 8,
        num_steps: int = 3,
        damping: float = 0.10,
        dt: float = 0.10,
        sigma: float = 0.50,
        eps: float = 1e-6,
        max_seq_len: int = 2048,
        freeze_evolution: bool = False,
        use_checkpoint: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or num_waves <= 0 or num_steps <= 0:
            raise ValueError("hidden_size/num_waves/num_steps 必须为正。")
        self.hidden_size = hidden_size
        self.num_waves = num_waves
        self.num_steps = num_steps
        self.eps = eps
        self.freeze_evolution = bool(freeze_evolution)
        self.use_checkpoint = bool(use_checkpoint)
        self.causal = bool(causal)

        self.in_proj = nn.Linear(hidden_size, num_waves, bias=True)
        self.out_proj = nn.Linear(num_waves, hidden_size, bias=True)

        # 每波通道可学习的中心频率与带通 / 输出缩放
        nyq = max(1.0, max_seq_len / 2.0)
        self.omega = nn.Parameter(torch.linspace(1.0, nyq * 0.8, num_waves))
        self.alpha = nn.Parameter(torch.ones(num_waves))
        self.gamma = nn.Parameter(torch.zeros(num_waves))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.out_scale = nn.Parameter(torch.tensor(1.0))

        # 固定物理超参（buffer，**不参与梯度**）：damping / dt / sigma 是设计超参，
        # 不是可学参数。此前 docstring 声称它们"梯度可达"，与实现不符，已更正。
        self.register_buffer("damping", torch.tensor(float(damping)), persistent=False)
        self.register_buffer("dt", torch.tensor(float(dt)), persistent=False)
        self.register_buffer("sigma", torch.tensor(float(sigma)), persistent=False)
        self.register_buffer("nyq", torch.tensor(nyq), persistent=False)

        self.norm = RMSNorm(hidden_size, eps=eps)

    @staticmethod
    def _band_pass(k: torch.Tensor, omega: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """高斯带通幅频窗 (Lf, W)。"""
        k = k.view(-1, 1)
        om = omega.view(1, -1)
        sg = sigma.view(1, -1).clamp_min(1e-3)
        return torch.exp(-((k - om) ** 2) / (2.0 * sg ** 2))

    def _evolve_loop(self, u: torch.Tensor, L: int, band: torch.Tensor,
                     damp: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """多步演化；freeze_evolution 时 FFT 路径走 no_grad，否则真正可训练。"""
        field = u
        if self.freeze_evolution:
            with torch.no_grad():
                for _ in range(self.num_steps):
                    f = _evolve_one_step(field, band, damp, L, self.causal)
                    field = f * (1.0 - beta) + u * beta
            return field.detach()

        use_ckpt = self.use_checkpoint and self.training and torch.is_grad_enabled()
        for _ in range(self.num_steps):
            if use_ckpt:
                f = _grad_checkpoint(
                    _evolve_one_step, field, band, damp, L, self.causal, use_reentrant=False
                )
            else:
                f = _evolve_one_step(field, band, damp, L, self.causal)
            field = f * (1.0 - beta) + u * beta
        return field

    def forward(self, x: torch.Tensor, seq_len: int | None = None) -> torch.Tensor:
        """谱演化前向。

        Args:
            x: (B, L, H) 输入隐状态。
            seq_len: 可选显式序列长度（FX concrete_args 下为常量）。
        Returns:
            (B, L, H) 共振场增强后的隐状态。
        """
        torch._assert(x.dim() == 3, "ResonanceField 期望 3D 输入 (B, L, H)")
        L = seq_len if seq_len is not None else x.shape[1]
        torch._assert(L >= 1, "序列长度必须 >= 1")

        u = self.in_proj(x)
        Lf = L // 2 + 1
        k = torch.arange(Lf, dtype=torch.float32).to(x.dtype).to(x.device)

        omega = self.omega.clamp(0.5, self.nyq)
        beta = torch.sigmoid(self.beta)
        damp = torch.exp(-self.damping * self.dt)
        band = self._band_pass(k, omega, self.sigma)

        field = self._evolve_loop(u, L, band, damp, beta)
        out = self.out_proj(field) * self.out_scale
        return self.norm(x + out)

    def extra_repr(self) -> str:
        return (f"H={self.hidden_size}, W={self.num_waves}, steps={self.num_steps}, "
                f"sigma={float(self.sigma):.2f}, causal={self.causal}, "
                f"freeze_evo={self.freeze_evolution}")
