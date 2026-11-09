# -*- coding: utf-8 -*-
"""Real-corpus evolution mixer -- STEP 0 (pure static SGC baseline).

Pure NumPy. No torch, no synthetic task, no analytic witness. This module is
imported by ``realtrain/train_real_lm.py`` and trained with Adam on a real
character-level language corpus.

STEP 0 exposes only ``StaticSGCMixer``: a strictly causal, time-invariant
(LTI / Toeplitz) mixer with an O(d) recurrent state

    s_t = rho * s_{t-1} + W_x e_t ,      y_t = s_t .

The kernel K(lag) = rho^lag is a fixed positive-real pole; it carries NO
input-dependent information. Under Theorem A this is exactly the regime in
which content addressing must fail unless the required bits are supplied on
the input side. We train it on real text to MEASURE that failure rather than
to assume it.

Invariants checked by ``invariants()`` are properties of the forward map, not
claims about language-model quality.
"""
from __future__ import annotations

import numpy as np

__all__ = ["StaticSGCMixer", "ARCH_STEP", "MIXER_CLASSES"]

ARCH_STEP = 0
MIXER_CLASSES = ("StaticSGCMixer",)


class StaticSGCMixer:
    """Fixed positive-real-pole LTI mixer. O(d) state, no N x N score."""

    def __init__(self, d: int, rho: float = 0.9, seed: int = 0):
        rng = np.random.default_rng(seed + 1000)
        self.d = int(d)
        self.rho = float(rho)
        self.W_x = rng.standard_normal((d, d)) * (1.0 / np.sqrt(d))
        # cache for backward
        self._cache = {}

    def params(self):
        return {"W_x": self.W_x}

    def forward(self, e: np.ndarray):
        """e: (B, L, d) -> y: (B, L, d), strictly causal LTI."""
        B, L, d = e.shape
        u = e @ self.W_x
        s = np.zeros((B, d), dtype=np.float64)
        y = np.empty_like(u)
        for t in range(L):
            s = self.rho * s + u[:, t, :]
            y[:, t, :] = s
        self._cache = dict(e=e, u=u, y=y)
        return y

    def backward(self, dy: np.ndarray):
        """Analytic adjoint of the IIR recurrence (rho fixed, not learned)."""
        e = self._cache["e"]
        B, L, d = e.shape
        r = np.zeros((B, d), dtype=np.float64)
        du = np.empty_like(dy)
        for t in range(L - 1, -1, -1):
            r = dy[:, t, :] + self.rho * r
            du[:, t, :] = r
        gW = np.einsum("bld,blm->dm", e, du) / 1.0
        de = du @ self.W_x.T
        return {"W_x": gW}, de

    def invariants(self, Ls=(9, 65, 257)):
        """Toeplitz / FFT-IIR / causality / constant-state checks."""
        L = max(Ls)
        K = self.rho ** np.arange(L)
        A = np.zeros((L, L))
        for i in range(L):
            for j in range(i + 1):
                A[i, j] = K[i - j]
        toe = max(abs(A[i, j] - A[i - 1, j - 1])
                  for i in range(1, L) for j in range(1, i + 1))
        rng = np.random.default_rng(123)
        x = rng.standard_normal(L)
        y0 = A @ x
        y1 = np.fft.ifft(np.fft.fft(K, 2 * L) * np.fft.fft(x, 2 * L))[:L].real
        err = float(np.linalg.norm(y0 - y1) / (np.linalg.norm(y0) + 1e-12))
        xin = rng.standard_normal((1, L, self.d))
        y = self.forward(xin)
        xin2 = xin.copy(); xin2[:, L // 2:, :] = 0.0
        y2 = self.forward(xin2)
        causal = int(np.count_nonzero(y[:, :L // 2] != y2[:, :L // 2]))
        return dict(arch_step=ARCH_STEP, kernel_static=True,
                    toeplitz_dev=float(toe), fft_iir_rel_err=err,
                    future_perturbation_changed=causal,
                    state_is_O_d=True, has_NxN_score=False,
                    has_kv_cache=False,
                    all_pass=bool(toe < 1e-12 and err < 1e-10 and causal == 0))
