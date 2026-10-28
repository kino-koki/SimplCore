# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Selective spectral state space kernel (the SSM half of SRFN).

Why this file exists
--------------------
``SpectralGlobalConv`` is already a state-space model in disguise: M damped
oscillator bases, an O(1) recurrence for streaming, and an FFT convolution for
training. What it is *not* is **selective** -- ``rho`` / ``theta`` / ``coef``
are global constants, so the model cannot decide per token how long to remember.

S4 -> Mamba's whole contribution is that one change: make the step size
``Delta_t`` a function of the input. The recurrence becomes time-varying and the
FFT trick dies with it (a convolution kernel only exists for a time-invariant
system). This module pays that price:

* ``A`` stays diagonal with complex eigenvalues ``lambda_m = rho_m e^{i theta_m}``
  (damped oscillators, not pure decay -- oscillators hold *periodic* memory,
  which decay-only SSMs like S4D/Mamba cannot represent);
* ``Delta_t`` is predicted from the input, one scalar per token;
* the time-varying convolution is evaluated **chunk-wise with a lower-triangular
  Toeplitz matrix**, so a chunk is one batched matmul instead of C Python steps;
* ``step()`` remains O(1) in sequence length and agrees with the parallel path
  (enforced by ``tests/test_srf.py``).

Design notes worth keeping
--------------------------
**Why ``Delta`` is one scalar per token, not per head.**
The first implementation used per-head ``Delta`` (Mamba-2 style). On CPU it ran
13x slower than the FFT path, and profiling showed the cost was not the matmul
but the *size of the intermediates*: ``(B, H, M, C, C)`` decay and phase tensors
with ``cos``/``sin`` applied to them, four times per layer. Dropping the head
axis shrinks every intermediate by ``H`` (8x here) without changing the MAC
count. A scalar-per-token ``Delta`` still answers the question that matters --
"should this position be remembered longer than the last one" -- it just cannot
specialise by channel group. That trade is worth 8x, and it is reversible later
if a GPU makes the head axis cheap.

**Why ``matmul`` and not ``einsum``.**
``torch.einsum("bhmij,bhmjd->bhmid", ...)`` is 5-D and falls back to a generic
contraction path. ``torch.matmul`` on the same operands dispatches to batched
GEMM. Same math, ~2x faster on CPU, and it keeps autograd simple.

**Why the causal mask is a cached buffer.**
``torch.tril`` was being rebuilt per chunk per layer per step; the mask shape
only depends on ``chunk_size``, which is fixed at construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 振荡器基初始化与 SGC 共用同一实现（单一事实源）：
# SGC 与 SRF 的 A/B 对照实验（srf-30m vs sgc-30m）要求两边初始化分布逐位一致，
# 这里曾逐字复制一份，一旦只改其中一份，对照基线即被静默污染。
from .sgc import _init_oscillator_grid, _logit


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
@dataclass
class SSMState:
    """Constant-size recurrent state of :class:`SelectiveSpectralSSM`.

    Attributes:
        h_re: (B, M, D) real part of the per-base state.
        h_im: (B, M, D) imaginary part of the per-base state.
        length: tokens consumed so far (diagnostics only).
    """

    h_re: torch.Tensor
    h_im: torch.Tensor
    length: int = 0

    def num_bytes(self) -> int:
        return (self.h_re.numel() * self.h_re.element_size()
                + self.h_im.numel() * self.h_im.element_size())


# ----------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------
class SelectiveSpectralSSM(nn.Module):
    """Diagonal complex SSM with an input-dependent step size.

    Recurrence (per channel ``d``, per base ``m``)::

        lambda_m  = rho_m * exp(i * theta_m)
        Delta_t   = softplus(W_dt h_t)                    # per token, scalar
        h_t[m,d]  = lambda_m^{Delta_t} * h_{t-1}[m,d] + b[m,d] * u_t[d]
        y_t[d]    = Re( sum_m coef[m,d] * h_t[m,d] ) + skip[d] * u_t[d]

    Args:
        dim: channel width D.
        num_bases: number of oscillator bases M (the state dimension).
        chunk_size: C of the chunked scan. Larger = fewer sequential steps but
            a bigger (C, C) matmul; 32 is the CPU compromise.
        decay_max: longest decay time constant used to initialise ``rho``.
        dt_bias: initial ``Delta`` in log space (negative = start near the
            time-invariant regime instead of at ``softplus(0) ~ 0.69``).
    """

    def __init__(
        self,
        dim: int,
        num_bases: int = 16,
        chunk_size: int = 32,
        decay_max: float = 1024.0,
        dt_bias: float = -4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_bases <= 0:
            raise ValueError("dim and num_bases must be positive")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        self.dim = dim
        self.num_bases = num_bases
        self.chunk_size = chunk_size
        self.eps = eps

        # --- oscillator bank ------------------------------------------------
        rho0, theta0 = _init_oscillator_grid(num_bases, decay_max)
        self.raw_rho = nn.Parameter(_logit(rho0))
        self.raw_theta = nn.Parameter(_logit(theta0 / math.pi))
        # 输入耦合 b[m,d] 与输出耦合 coef[m,d]
        self.b_in = nn.Parameter(torch.randn(num_bases, dim) * 0.02)
        # 量级 1/sqrt(M)：M 个基相干叠加时输出不随 M 线性膨胀
        self.coef = nn.Parameter(torch.randn(num_bases, dim) / math.sqrt(num_bases))
        # 直通项 D（SSM 标准配置）：init=1 让 block 起点接近恒等，
        # 否则初始输出只有 1e-3 量级，前几百步几乎没梯度。
        self.skip = nn.Parameter(torch.ones(dim))

        # --- selective step size --------------------------------------------
        self.dt_proj = nn.Linear(dim, 1, bias=True)
        with torch.no_grad():
            self.dt_proj.weight.mul_(0.01)
            self.dt_proj.bias.fill_(dt_bias)

        # 因果掩码只依赖 C，构造时缓存，避免每 chunk 每层每步重建 tril。
        self.register_buffer("_causal", torch.tril(torch.ones(chunk_size, chunk_size)),
                             persistent=False)

    # -- parameters -----------------------------------------------------------
    def oscillator_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rho = torch.sigmoid(self.raw_rho)
        theta = torch.sigmoid(self.raw_theta) * math.pi
        return rho, theta

    def step_size(self, h: torch.Tensor) -> torch.Tensor:
        """``Delta`` in (0, inf), shape (B, T, 1)."""
        return F.softplus(self.dt_proj(h))

    # -- decay matrix ---------------------------------------------------------
    def _chunk_kernel(self, delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lower-triangular decay factors for one chunk.

        Args:
            delta: (B, C, 1) step sizes of one chunk.
        Returns:
            (k_re, k_im) each (B, M, C, C) with ``k[i,j] = lambda_m**(S_i - S_j)``
            for ``i >= j`` and 0 above the diagonal.
        """
        rho, theta = self.oscillator_params()
        # reshape 而不是 squeeze：C=1 时 squeeze(-1) 会把 (B,1) 压成 (B,)，
        # 后面的 cumsum(dim=1) 直接越界。
        d = delta.reshape(delta.shape[0], -1)                        # (B,C)
        s = torch.cumsum(d, dim=1)                                   # (B,C)
        diff = s.unsqueeze(-1) - s.unsqueeze(-2)                     # (B,C,C)
        mask = self._causal
        # 上三角（未来位置）先归零，否则 S_i - S_j < 0 会让 exp 爆炸；
        # 但**不能只靠这一步就当掩码**——exp(0) 是 1，上三角会变成有效权重。
        # 真正的置零在最后乘一次 mask。踩过：位置 3（chunk 末）对、中间位全错。
        diff = diff * mask
        diff = diff.unsqueeze(1)                                     # (B,1,C,C)

        log_rho = torch.log(rho).clamp_max(-1e-6).view(1, -1, 1, 1)   # (1,M,1,1)
        th = theta.view(1, -1, 1, 1)                                  # (1,M,1,1)
        amp = torch.exp(diff * log_rho) * mask                        # (B,M,C,C)
        phase = diff * th
        return amp * torch.cos(phase), amp * torch.sin(phase)

    # -- parallel form --------------------------------------------------------
    def forward(self, u: torch.Tensor,
                state: Optional[SSMState] = None,
                return_state: bool = False):
        """Chunked parallel scan.

        Args:
            u: (B, T, D) already projected / gated input.
            state: optional incoming state (streaming a long sequence chunk by
                chunk). ``None`` = start from zero.
            return_state: also return the final :class:`SSMState`.
        """
        B, T, D = u.shape
        M, C = self.num_bases, self.chunk_size
        dev, dt = u.device, u.dtype

        delta = self.step_size(u)                                   # (B,T,1)
        pad = (-T) % C
        if pad:
            u_p = F.pad(u, (0, 0, 0, pad))
            d_p = F.pad(delta, (0, 0, 0, pad))
        else:
            u_p, d_p = u, delta
        Tp = T + pad
        nc = Tp // C

        # (B,nc,C,D) -> (B,M,nc,C,D)：每个基对输入做一次因果卷积
        # 先把 b[m,d] 调制和 chunk 分组合并做，避免出现 (B,H,M,...) 五维张量
        # —— 那是第一版慢 13 倍的主因。
        x = u_p.view(B, nc, C, D).unsqueeze(1) * self.b_in.view(1, M, 1, 1, D)

        dch = d_p.view(B, nc, C, 1).permute(0, 3, 1, 2)              # (B,1,nc,C)

        if state is None:
            h_re = torch.zeros(B, M, D, device=dev, dtype=dt)
            h_im = torch.zeros(B, M, D, device=dev, dtype=dt)
        else:
            h_re, h_im = state.h_re, state.h_im

        rho, theta = self.oscillator_params()
        log_rho = torch.log(rho).clamp_max(-1e-6).view(1, M, 1)
        th = theta.view(1, M, 1)

        outs = []
        for c in range(nc):
            k_re, k_im = self._chunk_kernel(dch[:, 0, c])             # (B,M,C,C)
            xc = x[:, :, c]                                          # (B,M,C,D)

            # chunk 内从零状态出发的输入贡献（下三角因果卷积）
            y_re = torch.matmul(k_re, xc)
            y_im = torch.matmul(k_im, xc)

            # 上一 chunk 末状态按 lambda^{S_i} 衰减后叠加（复数乘）
            s_cum = torch.cumsum(dch[:, 0, c].reshape(B, -1), dim=1)  # (B,C)
            amp = torch.exp(s_cum.unsqueeze(1) * log_rho)            # (B,M,C)
            ph = s_cum.unsqueeze(1) * th
            car_re = (amp * torch.cos(ph)).unsqueeze(-1)             # (B,M,C,1)
            car_im = (amp * torch.sin(ph)).unsqueeze(-1)
            y_re = y_re + car_re * h_re.unsqueeze(2) - car_im * h_im.unsqueeze(2)
            y_im = y_im + car_im * h_re.unsqueeze(2) + car_re * h_im.unsqueeze(2)

            h_re = y_re[:, :, -1]
            h_im = y_im[:, :, -1]
            outs.append(y_re)

        # (B,M,nc,C,D) -> (B,nc,C,M,D) -> 与 coef 收缩
        y = torch.stack(outs, dim=2).permute(0, 2, 3, 1, 4)          # (B,nc,C,M,D)
        y = (y * self.coef.view(1, 1, 1, M, D)).sum(3)               # (B,nc,C,D)
        y = y.reshape(B, Tp, D)[:, :T]
        y = y + u * self.skip.view(1, 1, D)

        if return_state:
            return y, SSMState(h_re=h_re, h_im=h_im,
                               length=(state.length if state else 0) + T)
        return y

    # -- streaming form -------------------------------------------------------
    def init_state(self, batch: int, device, dtype) -> SSMState:
        return SSMState(
            h_re=torch.zeros(batch, self.num_bases, self.dim, device=device, dtype=dtype),
            h_im=torch.zeros(batch, self.num_bases, self.dim, device=device, dtype=dtype),
        )

    def step(self, u_t: torch.Tensor, state: SSMState) -> Tuple[torch.Tensor, SSMState]:
        """One token, O(1) state and compute."""
        squeeze = u_t.dim() == 2
        if squeeze:
            u_t = u_t.unsqueeze(1)
        B, one, D = u_t.shape
        assert one == 1, "step() consumes exactly one position"
        M = self.num_bases

        delta = self.step_size(u_t).view(B, 1)                       # (B,1)
        rho, theta = self.oscillator_params()

        amp = torch.exp(delta * torch.log(rho).clamp_max(-1e-6).view(1, M))   # (B,M)
        ph = delta * theta.view(1, M)
        c, s = torch.cos(ph), torch.sin(ph)

        xb = u_t.squeeze(1).unsqueeze(1) * self.b_in.view(1, M, D)    # (B,M,D)
        ar = (amp * c).unsqueeze(-1)
        ai = (amp * s).unsqueeze(-1)

        new_re = ar * state.h_re - ai * state.h_im + xb
        new_im = ai * state.h_re + ar * state.h_im

        y = (new_re * self.coef.view(1, M, D)).sum(1).unsqueeze(1)     # (B,1,D)
        y = y + u_t * self.skip.view(1, 1, D)

        state.h_re, state.h_im = new_re, new_im
        state.length += 1
        return (y.squeeze(1) if squeeze else y), state

    # -- introspection --------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, bases={self.num_bases}, chunk={self.chunk_size}")

    def streaming_state_bytes(self, batch: int, dtype=torch.float32) -> int:
        itemsize = torch.empty(0, dtype=dtype).element_size()
        return 2 * self.num_bases * self.dim * batch * itemsize
