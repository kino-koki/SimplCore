# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC mixer: content-gated causal spectral global convolution.

See package docstring in ``simplcore/sgc/__init__.py`` and ``docs/SGC_SPEC.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Cheaper than LayerNorm and the de-facto standard in modern LMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def make_norm(kind: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm kind: {kind!r}")


# ----------------------------------------------------------------------------
# Streaming state
# ----------------------------------------------------------------------------
@dataclass
class LayerStreamState:
    """Constant-size recurrent state of one SGC mixer (independent of sequence length).

    Attributes:
        y1: (B, M, D) most recent oscillator state.
        y2: (B, M, D) previous oscillator state.
        conv_buf: (B, local_kernel - 1, D) depthwise-conv history, may be None.
        length: number of tokens consumed so far.
        energy_sum: (B, 1) running sum of the per-position input energy,
            used by the L2 evidence gate. Constant size.
    """

    y1: torch.Tensor
    y2: torch.Tensor
    conv_buf: Optional[torch.Tensor]
    length: int = 0
    energy_sum: Optional[torch.Tensor] = None


@dataclass
class SGCStreamState:
    """Per-layer streaming state bundle for a whole :class:`SGCModel`."""

    layers: list
    length: int = 0
    memory: Optional[torch.Tensor] = None    # 投影后的 CMT 压缩记忆 (B, D)

    def detach(self) -> "SGCStreamState":
        for st in self.layers:
            st.y1 = st.y1.detach()
            st.y2 = st.y2.detach()
            if st.conv_buf is not None:
                st.conv_buf = st.conv_buf.detach()
        return self

    def num_bytes(self) -> int:
        total = 0
        for st in self.layers:
            total += st.y1.numel() * st.y1.element_size()
            total += st.y2.numel() * st.y2.element_size()
            if st.conv_buf is not None:
                total += st.conv_buf.numel() * st.conv_buf.element_size()
            if st.energy_sum is not None:
                total += st.energy_sum.numel() * st.energy_sum.element_size()
        return total


# ----------------------------------------------------------------------------
# FFT helpers
# ----------------------------------------------------------------------------
def fft_causal_conv(u: torch.Tensor, kernel: torch.Tensor, fft_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Causal (depthwise) convolution of ``u`` with a length-L kernel bank.

    Computes ``y[b, t, d] = sum_{s <= t} kernel[t - s, d] * u[b, s, d]`` exactly,
    in ``O(L log L)``, by zero-padding to ``N = 2L`` and multiplying in the
    frequency domain (circular convolution of length 2L has no aliasing for a
    causal kernel of length <= L).

    Args:
        u: (B, L, D) input.
        kernel: (L, D) kernel bank (shared across the batch).
        fft_dtype: dtype used for the transform; ``float32`` is recommended.
    """
    B, L, D = u.shape
    N = 2 * L
    work_dtype = fft_dtype if fft_dtype is not None else u.dtype
    xw = u.to(work_dtype)
    kw = kernel.to(work_dtype)
    cX = torch.fft.rfft(xw, n=N, dim=1)
    cK = torch.fft.rfft(kw, n=N, dim=0).unsqueeze(0)
    cY = cX * cK
    y = torch.fft.irfft(cY, n=N, dim=1)[:, :L, :]
    return y.to(u.dtype)


def depthwise_causal_conv(u: torch.Tensor, weight: torch.Tensor, buf: Optional[torch.Tensor]):
    """Depthwise causal 1-D convolution with ``weight`` of shape (D, 1, K).

    Returns ``(y, new_buf)`` where ``new_buf`` holds the last ``K-1`` inputs, so a
    single implementation serves both the parallel path (``buf=None``) and the
    streaming path (one position at a time, with ``buf`` carried in the state).

    In the streaming case the convolution is evaluated over ``cat(buf, u)`` and
    only the outputs belonging to the **new** positions are returned; otherwise the
    buffered positions would be emitted again.
    """
    K = weight.shape[-1]
    if K == 1:
        return u * weight.view(1, 1, -1), u[:, -0:, :]
    if buf is not None and buf.shape[1] > 0:
        uu = torch.cat([buf, u], dim=1)
    else:
        uu = u
    n_new = u.shape[1]
    pad = K - 1
    up = F.pad(uu.transpose(1, 2), (pad, 0))          # (B, D, T+pad), left pad => causal
    y = F.conv1d(up, weight, groups=weight.shape[0])  # (B, D, T)
    y = y.transpose(1, 2)[:, -n_new:, :]              # keep only the new positions
    if uu.shape[1] >= pad:
        new_buf = uu[:, uu.shape[1] - pad:, :]
    else:
        new_buf = uu
    return y, new_buf


# ----------------------------------------------------------------------------
# The mixer
# ----------------------------------------------------------------------------
class SpectralGlobalConv(nn.Module):
    """SGC mixer: content-gated causal spectral global convolution.

    Forward (parallel / training form)::

        h  = norm(x)
        u  = in_proj(h) * sigmoid(gate_proj(h))        # content gating (selective)
        y  = fft_causal_conv(u, K)                     # long range, O(L log L)
           + depthwise_causal_conv(u, W_loc)           # short range, O(L*K)
           + feed * u                                  # per-channel instantaneous term
        out = out_proj(y)
        return x + out_scale * out

    where the global kernel bank is::

        K[tau, d] = sum_m coef[m, d] * psi_m[tau]
        psi_m[tau] = rho_m^tau * sin((tau + 1) * theta_m)

    ``step()`` evaluates the same operator one token at a time through the
    equivalent 2nd-order IIR recurrence, with O(1) state in the sequence length.

    Args:
        dim: channel width D.
        num_bases: number of damped-oscillator bases M (default 64).
        decay_max: longest decay time constant used to initialise rho (default 8192).
        local_kernel: depthwise causal conv width K (default 4).
        rank: rank r of the short-range depthwise conv parameterisation. The conv
            is depthwise so this is unused unless ``low_rank_local`` is set.
        content_gate: enable input-dependent gating (default True).
        out_gate: enable an output gate (default False).
        norm: "rmsnorm" or "layernorm".
        fft_dtype: dtype for the FFT path; ``torch.float32`` by default.
        eps: norm epsilon.
    """

    def __init__(
        self,
        dim: int,
        num_bases: int = 64,
        decay_max: float = 8192.0,
        local_kernel: int = 4,
        content_gate: bool = True,
        out_gate: bool = False,
        evidence_gate: bool = True,
        uncertainty_gate: bool = True,
        norm: str = "rmsnorm",
        fft_dtype: Optional[torch.dtype] = torch.float32,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")
        if local_kernel < 1:
            raise ValueError("local_kernel must be >= 1")

        self.dim = dim
        self.num_bases = num_bases
        self.local_kernel = local_kernel
        self.content_gate = content_gate
        self.out_gate = out_gate
        self.fft_dtype = fft_dtype

        self.norm = make_norm(norm, dim, eps=eps)

        # --- content / projections -------------------------------------------
        self.in_proj = nn.Linear(dim, dim, bias=False)
        if content_gate:
            self.gate_proj = nn.Linear(dim, dim, bias=False)
        if out_gate:
            self.out_gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # --- long-range oscillator bank --------------------------------------
        rho0, theta0 = _init_oscillator_grid(num_bases, decay_max)
        self.raw_rho = nn.Parameter(_logit(rho0))
        self.raw_theta = nn.Parameter(_logit(theta0 / math.pi))
        # Small non-zero init so that d/d(rho), d/d(theta) are live from step 0
        # (with coef == 0 the kernel does not depend on rho/theta at all).
        self.coef = nn.Parameter(torch.randn(num_bases, dim) * 0.02)

        # Per-channel instantaneous (diagonal) term; init 1 keeps the block close
        # to identity at the start of training.
        self.feed = nn.Parameter(torch.ones(dim))

        # --- short-range depthwise conv --------------------------------------
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=local_kernel,
                                    groups=dim, bias=False)
        with torch.no_grad():
            self.local_conv.weight.zero_()
            self.local_conv.weight[:, 0, -1] = 1.0   # identity at init

        self.out_scale = nn.Parameter(torch.ones(1))

        # --- four-layer gate: L2 evidence / L3 uncertainty -------------------
        # L2 证据门: per-position evidence derived from the *causal prefix* energy
        # of the gated input; damps the long-range path when the context carries
        # no usable signal. Because the statistic is a causal-prefix quantity,
        # the parallel (cumsum) form and the streaming (running sum) form agree,
        # which is what preserves the FFT / IIR dual form.
        self.evidence_gate = evidence_gate
        # 用标量参数而非 Linear：闸门响应形状是**结构性**的（证据能量越高越开），
        # 只有强度/偏置可学。这样做有两个好处：
        #   1) 避免全零权重导致闸门退化成常数（对输入不敏感，形同虚设）；
        #   2) 不受上层模型 `apply(self._init_weights)` 对所有 nn.Linear 重施
        #      normal(0, init_std) 的影响。
        self.evidence_gain = nn.Parameter(torch.tensor(1.0))       # 证据项强度
        self.evidence_pos_gain = nn.Parameter(torch.tensor(0.0))   # 位置计数项强度
        self.evidence_bias = nn.Parameter(torch.tensor(0.0))        # 偏置
        # L3 不确定门: the long-range and local paths are two independent
        # estimates of the same context; their disagreement is an
        # uncertainty proxy. High disagreement damps the long path so the
        # model falls back on local evidence.
        self.uncertainty_gate = uncertainty_gate
        self.uncertainty_logit = nn.Parameter(torch.zeros(1))
        self.gate_eps = 1e-6

    # -- kernel ---------------------------------------------------------------
    def oscillator_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (rho, theta), both of shape (M,), with rho in (0,1), theta in (0,pi)."""
        rho = torch.sigmoid(self.raw_rho)
        theta = torch.sigmoid(self.raw_theta) * math.pi
        return rho, theta

    def build_kernel(self, length: int, device=None, dtype=None) -> torch.Tensor:
        """Materialise the causal kernel bank K of shape (length, D).

        Defined for **any** ``length``: there is no ``max_len`` truncation. This
        removes the SGC v2 defect where sequences longer than ``max_len``
        silently degraded into a fixed-window convolution.
        """
        rho, theta = self.oscillator_params()
        device = device if device is not None else self.coef.device
        dtype = dtype if dtype is not None else self.coef.dtype
        tau = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)      # (1,L)
        log_rho = torch.log(rho.to(dtype)).unsqueeze(1)                          # (M,1)
        psi = torch.exp(log_rho * tau) * torch.sin((tau + 1.0) * theta.to(dtype).unsqueeze(1))
        kernel = self.coef.to(dtype).t() @ psi                                   # (D, M) @ (M, L)
        return kernel.t().contiguous()                                           # (L, D)

    # -- forward --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, L, D = x.shape
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        kernel = self.build_kernel(L, device=u.device, dtype=u.dtype)
        long = fft_causal_conv(u, kernel, self.fft_dtype)
        local = self._local_conv_full(u)

        if self.evidence_gate:
            long = long * self._evidence_weights(u)
        if self.uncertainty_gate:
            long = long * self._uncertainty_weights(long, local)

        y = long + local + self.feed * u

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        return residual + self.out_scale * out

    def _local_conv_full(self, u: torch.Tensor) -> torch.Tensor:
        K = self.local_kernel
        if K == 1:
            return u * self.local_conv.weight.view(1, 1, -1)
        up = F.pad(u.transpose(1, 2), (K - 1, 0))
        return F.conv1d(up, self.local_conv.weight, groups=self.dim).transpose(1, 2)

    # -- L2 evidence gate -----------------------------------------------------
    def _evidence_weights(self, u: torch.Tensor) -> torch.Tensor:
        """Per-position evidence weight in (0, 1), shape (B, L, 1).

        Evidence is the causal prefix average energy of the gated input::

            c_t = (1/t) * sum_{s<=t} ||u_s||^2

        ``log1p`` features of ``(c_t, t)`` feed a 2->1 linear squashed by a
        sigmoid, so a low-energy prefix (nothing usable in the context)
        shrinks the long-range contribution, while a high-energy prefix
        opens it.

        The statistic is a *causal prefix* quantity, hence the parallel
        (cumsum) form and the streaming (running sum) form agree -- this is
        what keeps the FFT / IIR dual form intact.
        """
        B, L, _ = u.shape
        energy = (u.float() ** 2).sum(dim=-1)                 # (B, L)
        csum = torch.cumsum(energy, dim=1)                   # (B, L)
        t = torch.arange(1, L + 1, device=u.device,
                         dtype=energy.dtype).unsqueeze(0)    # (1, L)
        c = csum / t                                         # (B, L) prefix mean energy
        logit = (self.evidence_gain * torch.log1p(c)
                 + self.evidence_pos_gain * torch.log1p(t.expand_as(c))
                 + self.evidence_bias)
        return torch.sigmoid(logit).unsqueeze(-1)            # (B, L, 1)

    # -- L3 uncertainty gate --------------------------------------------------
    def _uncertainty_weights(self, long: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Per-position damping factor in (0, 1], shape (B, L, 1).

        The long-range and local paths are two independent estimates of the
        same context. Their disagreement is a disagreement-style uncertainty
        proxy: when they diverge, the long-range contribution is damped and
        the model falls back on local evidence.

            d_t = ||long_t - local_t||_1 / (||long_t||_1 + ||local_t||_1 + eps)
            w_t = 1 - sigmoid(gamma) * d_t
        """
        num = (long - local).abs().sum(dim=-1, keepdim=True)
        den = (long.abs().sum(dim=-1, keepdim=True)
               + local.abs().sum(dim=-1, keepdim=True) + self.gate_eps)
        d = num / den
        return 1.0 - torch.sigmoid(self.uncertainty_logit) * d

    # -- L2 evidence gate, streaming form -------------------------------------
    def _evidence_step(self, u: torch.Tensor, state: LayerStreamState) -> torch.Tensor:
        """Streaming counterpart of :meth:`_evidence_weights` for one position.

        The running prefix energy sum lives in the (constant-size) stream
        state, so the streamed weight matches the parallel one.
        """
        energy = (u.float() ** 2).sum(dim=-1)                 # (B, 1)
        if state.energy_sum is None:
            state.energy_sum = energy
        else:
            state.energy_sum = state.energy_sum + energy
        t = torch.tensor(float(state.length + 1), device=u.device, dtype=torch.float32)
        c = state.energy_sum.squeeze(-1) / t                 # (B,)
        logit = (self.evidence_gain * torch.log1p(c)
                 + self.evidence_pos_gain * torch.log1p(t).expand_as(c)
                 + self.evidence_bias)
        return torch.sigmoid(logit)[:, None, None]           # (B, 1, 1)

    # -- streaming ------------------------------------------------------------
    def init_state(self, batch: int, device, dtype) -> LayerStreamState:
        M, D = self.num_bases, self.dim
        y1 = torch.zeros(batch, M, D, device=device, dtype=dtype)
        y2 = torch.zeros(batch, M, D, device=device, dtype=dtype)
        buf = None
        if self.local_kernel > 1:
            buf = torch.zeros(batch, self.local_kernel - 1, D, device=device, dtype=dtype)
        energy_sum = torch.zeros(batch, 1, device=device, dtype=torch.float32)
        return LayerStreamState(y1=y1, y2=y2, conv_buf=buf, length=0,
                                 energy_sum=energy_sum)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: LayerStreamState) -> Tuple[torch.Tensor, LayerStreamState]:
        """One token, O(1) state, O(1) compute in the sequence length.

        Args:
            x_t: (B, 1, D) or (B, D) input for the next position.
            state: persistent state from :meth:`init_state` or a previous call.
        Returns:
            (y_t, new_state) with ``y_t`` of shape (B, 1, D).
        """
        squeeze = x_t.dim() == 2
        if squeeze:
            x_t = x_t.unsqueeze(1)
        B, one, D = x_t.shape
        M = self.num_bases
        assert one == 1, "step() consumes exactly one position"

        h = self.norm(x_t)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        rho, theta = self.oscillator_params()
        a = (2.0 * rho * torch.cos(theta)).to(u.dtype)                 # (M,)
        b = (-(rho ** 2)).to(u.dtype)
        c = torch.sin(theta).to(u.dtype)

        um = u.squeeze(1).unsqueeze(1)                                 # (B,1,D)
        y_new = a.view(1, M, 1) * state.y1 + b.view(1, M, 1) * state.y2 + c.view(1, M, 1) * um
        state.y2, state.y1 = state.y1, y_new

        nxt = torch.einsum("bmd,md->bd", y_new, self.coef.to(u.dtype)).unsqueeze(1)   # (B,1,D)
        loc, state.conv_buf = depthwise_causal_conv(u, self.local_conv.weight, state.conv_buf)

        if self.evidence_gate:
            nxt = nxt * self._evidence_step(u, state)
        if self.uncertainty_gate:
            nxt = nxt * self._uncertainty_weights(nxt, loc)

        y = nxt + loc + self.feed * u

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        state.length += 1
        y_t = x_t + self.out_scale * out
        return (y_t.squeeze(1) if squeeze else y_t), state

    # -- introspection --------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, num_bases={self.num_bases}, "
                f"local_kernel={self.local_kernel}, content_gate={self.content_gate}, "
                f"out_gate={self.out_gate}, evidence_gate={self.evidence_gate}, "
                f"uncertainty_gate={self.uncertainty_gate}")

    def streaming_state_bytes(self, batch: int, dtype=torch.float32) -> int:
        """Bytes of constant streaming state, independent of sequence length."""
        itemsize = torch.empty(0, dtype=dtype).element_size()
        n = 2 * self.num_bases * self.dim * batch
        n += max(self.local_kernel - 1, 0) * self.dim * batch
        extra = 0
        if self.evidence_gate:
            # 运行前缀能量和：每条序列一个 fp32 标量（与序列长度无关）
            extra += batch * 4
        return n * itemsize + extra

    def effective_kernel_rank(self, length: int) -> int:
        """Numerical rank of the kernel bank; shows the expressivity of the basis."""
        with torch.no_grad():
            K = self.build_kernel(length)
            return int(torch.linalg.matrix_rank(K.t().float()))


# ----------------------------------------------------------------------------
# Initialisation helpers
# ----------------------------------------------------------------------------
def _init_oscillator_grid(num_bases: int, decay_max: float):
    """Log-spaced decay time constants spanning [1, decay_max] and spread frequencies.

    Returns (rho0, theta0) both of shape (M,), with rho0 in (0,1) and theta0 in (0,pi).
    Decay time constants are geometrically spaced so the bank covers every
    timescale from a single token up to ``decay_max``.
    """
    lo, hi = math.log(1.0), math.log(max(decay_max, 2.0))
    lam = torch.exp(torch.linspace(lo, hi, num_bases))
    rho0 = torch.exp(-1.0 / lam)
    theta0 = math.pi * (torch.arange(num_bases, dtype=torch.float32) + 0.5) / num_bases
    return rho0, theta0


def _logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)
