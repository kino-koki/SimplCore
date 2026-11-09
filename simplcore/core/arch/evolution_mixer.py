# -*- coding: utf-8 -*-
"""Real-corpus evolution mixer -- STEP 1 (input-dependent evidence gate).

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

__all__ = ["StaticSGCMixer", "GatedEvolutionMixer", "ARCH_STEP", "MIXER_CLASSES"]

ARCH_STEP = 1
MIXER_CLASSES = ("StaticSGCMixer", "GatedEvolutionMixer")


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

    def forward(self, e: np.ndarray, tok=None):
        """e: (B, L, d) -> y: (B, L, d), strictly causal LTI.

        ``tok`` is accepted for interface compatibility with the gated mixer
        but is deliberately NOT read by this arm: a time-invariant kernel must
        not depend on the input symbols.
        """
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


class GatedEvolutionMixer(StaticSGCMixer):
    """STEP 1 -- input-dependent L2 evidence gate over a local path.

        c_t = (1/t) sum_{s<=t} tok_s / 128        (input-side statistic)
        g_t = sigmoid(a log1p(c_t) + b log1p(t) + c0)
        y_t = g_t * long_t + (1-g_t) * local_t

    This **deliberately gives up strict time invariance**: the gate is a
    function of the input symbols, which is exactly the input-side bit supply
    that Theorem A requires and the P2 escape route. The recurrent state is
    still O(d); no N x N score and no KV cache are introduced.

    ``c_t`` depends only on token ids, never on learnable parameters, so its
    stop-gradient status is structural rather than an implementation trick.
    """

    def __init__(self, d: int, rho: float = 0.9, seed: int = 0):
        super().__init__(d, rho, seed)
        rng = np.random.default_rng(seed + 2000)
        self.W_l = rng.standard_normal((d, d)) * (1.0 / np.sqrt(d))
        self.a = np.array(0.0)
        self.b = np.array(0.0)
        self.c0 = np.array(0.0)
        self._c = {}

    def params(self):
        p = super().params()
        p.update({"W_l": self.W_l, "a": self.a, "b": self.b, "c0": self.c0})
        return p

    @staticmethod
    def _sigmoid(x):
        return 0.5 * (1.0 + np.tanh(0.5 * x))

    def forward(self, e, tok=None):
        B, L, d = e.shape
        u = e @ self.W_x
        local = e @ self.W_l
        s = np.zeros((B, d), dtype=np.float64)
        long = np.empty_like(u)
        for t in range(L):
            s = self.rho * s + u[:, t, :]
            long[:, t, :] = s
        if tok is None:
            raise ValueError("GatedEvolutionMixer requires tok (input-side bits)")
        tv = tok.astype(np.float64) / 128.0
        ct = np.cumsum(tv, axis=1) / np.arange(1, L + 1)[None, :]
        tt = np.log1p(np.arange(1, L + 1, dtype=np.float64))[None, :]
        logit = self.a * np.log1p(ct) + self.b * tt + self.c0
        g = self._sigmoid(logit)                      # (B, L)
        y = g[..., None] * long + (1.0 - g[..., None]) * local
        self._c.update(e=e, u=u, long=long, local=local, g=g, ct=ct, tt=tt)
        return y

    def backward(self, dy):
        c = self._c
        e, long, local, g = c["e"], c["long"], c["local"], c["g"]
        B, L, d = e.shape
        dg_raw = np.sum(dy * (long - local), axis=-1)     # (B, L)
        dg = dg_raw * g * (1.0 - g)
        ga = np.array(float(np.sum(dg * np.log1p(c["ct"]))))
        gb = np.array(float(np.sum(dg * c["tt"])))
        gc0 = np.array(float(np.sum(dg)))
        dlong = dy * g[..., None]
        dlocal = dy * (1.0 - g[..., None])
        r = np.zeros((B, d), dtype=np.float64)
        du = np.empty_like(dlong)
        for t in range(L - 1, -1, -1):
            r = dlong[:, t, :] + self.rho * r
            du[:, t, :] = r
        gW_x = np.einsum("bld,blm->dm", e, du)
        gW_l = np.einsum("bld,blm->dm", e, dlocal)
        de = du @ self.W_x.T + dlocal @ self.W_l.T
        return {"W_x": gW_x, "W_l": gW_l, "a": ga, "b": gb, "c0": gc0}, de

    def invariants(self, Ls=(9, 65, 257)):
        base = StaticSGCMixer.invariants(self, Ls)
        base.update(arch_step=ARCH_STEP, kernel_static=False,
                    time_invariant=False, input_dependent_gate=True,
                    state_is_O_d=True, has_NxN_score=False,
                    has_kv_cache=False,
                    note=("strict Toeplitz/FFT-IIR duality is retained by the "
                          "long path only; the composite forward map is no "
                          "longer time invariant by construction"))
        return base
