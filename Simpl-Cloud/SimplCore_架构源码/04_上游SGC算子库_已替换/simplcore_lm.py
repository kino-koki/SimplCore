# -*- coding: utf-8 -*-
"""SimplCore LM -- ARCH_STEP 7. A complete, end-to-end trainable language model.

This is a REAL upgrade of the architecture, not a renamed checkpoint. The six
previous steps were a single mixer followed by a linear read-out; that model
cannot learn language, which is why every paired CI contained zero. Step 7 adds
depth, normalisation, a feed-forward block, local content addressing and a
gated consolidating state, while keeping the SimplCore invariants:

  * global channel = static multi-pole Toeplitz kernel (time invariant,
    FFT/IIR equivalent, no input-dependent kernel);
  * consolidation = O(d^2) outer-product recurrence with an input-dependent
    write gate (the "self-evolving" path), state size independent of L;
  * local channel = fixed-W window attention (W constant, not a growing KV
    cache, no full N x N score over the whole sequence);
  * no attention weights are cached across generation steps.

Pure NumPy fp64 with hand-written analytic adjoints (see nn.py). No torch.

All three fused operators below are verified against an explicit O(L) or O(L^2)
reference recurrence in verify.py; the matrix form is used only for speed.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn  # noqa: E402  (local minimal autodiff)

__all__ = ["SimplCoreLM", "SimplCoreBlock", "ARCH_STEP", "params_of",
           "window_attn", "toeplitz_mix", "decayed_attn",
           "window_attn_reference", "decayed_attn_reference"]

ARCH_STEP = 7


# ---------------------------------------------------------------------------
# fused operators (analytic adjoints)
# ---------------------------------------------------------------------------
def window_attn(q, k, v, W, pos):
    """Causal fixed-window content addressing. W is a CONSTANT, not L.

    Returns (B,L,d). j ranges over t-W+1 .. t; padded positions are masked.
    """
    B, L, d = q.data.shape
    pad = W - 1
    inv = 1.0 / math.sqrt(d)
    kp = np.concatenate([np.zeros((B, pad, d)), k.data], axis=1)
    vp = np.concatenate([np.zeros((B, pad, d)), v.data], axis=1)
    # kp[t + pad] = k_t ; window w at position t reads kp[t + w] = k_{t+w-pad}
    idx = np.arange(L)[:, None] + np.arange(W)[None, :]      # (L,W)
    kw = kp[:, idx, :]                                       # (B,L,W,d)
    vw = vp[:, idx, :]
    score = np.einsum("bld,blwd->blw", q.data, kw) * inv
    score = score + pos.data[None, None, :]
    j = np.arange(L)[:, None] - pad + np.arange(W)[None, :]
    invalid = j < 0
    score = np.where(invalid[None, :, :], -1e9, score)
    z = score - score.max(axis=-1, keepdims=True)
    ex = np.exp(z)
    a = ex / ex.sum(axis=-1, keepdims=True)
    y = np.einsum("blw,blwd->bld", a, vw)

    def back(g):
        da = np.einsum("bld,blwd->blw", g, vw)
        dvw = np.einsum("bld,blw->blwd", g, a)
        ds = a * (da - (a * da).sum(axis=-1, keepdims=True))
        ds = np.where(invalid[None, :, :], 0.0, ds)
        dq = np.einsum("blw,blwd->bld", ds, kw) * inv
        dkw = np.einsum("blw,bld->blwd", ds, q.data) * inv
        dpos = ds.sum(axis=(0, 1))
        dkp = np.zeros_like(kp)
        dvp = np.zeros_like(vp)
        dkpt = dkp.transpose(1, 0, 2)
        dvpt = dvp.transpose(1, 0, 2)
        ii = idx.ravel()
        np.add.at(dkpt, ii, dkw.transpose(1, 2, 0, 3).reshape(L * W, B, d))
        np.add.at(dvpt, ii, dvw.transpose(1, 2, 0, 3).reshape(L * W, B, d))
        dk = dkp[:, pad:pad + L, :]
        dv = dvp[:, pad:pad + L, :]
        nn._acc(q, dq)
        nn._acc(k, dk)
        nn._acc(v, dv)
        nn._acc(pos, dpos)

    return nn.custom(y, (q, k, v, pos), back)


def toeplitz_mix(u, wlogits, rhos, L):
    """Static multi-pole SGC channel  y = sum_k softmax(w)_k (A_k @ u).

    A_k[i,j] = rho_k^(i-j) for j<=i else 0. The kernel is time invariant;
    only the mixture weights are learned, and they do not depend on t.
    """
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    As = [np.where(j <= i, r ** (i - j), 0.0) for r in rhos]
    K = len(As)
    z = wlogits.data - wlogits.data.max()
    e = np.exp(z)
    w = e / e.sum()
    Au = np.stack([A @ u.data for A in As])          # (K,B,L,d)
    y = np.einsum("k,kbld->bld", w, Au)

    def back(g):
        du = np.zeros_like(u.data)
        raw = np.empty(K)
        for k in range(K):
            du += w[k] * (As[k].T @ g)
            raw[k] = np.sum(g * (As[k] @ u.data))
        dw = w * (raw - np.dot(w, raw))
        nn._acc(u, du)
        nn._acc(wlogits, dw)
    return nn.custom(y, (u, wlogits), back)


def decayed_attn(q, k, v, beta, lam):
    """Gated consolidating state, written in its O(d^2) recurrence form.

    S_t = lam*S_{t-1} + beta_t * outer(k_t, v_t)
    y_t = q_t S_t / sqrt(d)

    The matrix evaluation below is algebraically identical (verified in
    verify.py). No attention matrix is stored at generation time.
    """
    B, L, d = q.data.shape
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    D = np.where(j <= i, lam ** (i - j), 0.0)
    c = 1.0 / math.sqrt(d)
    qkt = np.einsum("bld,bmd->blm", q.data, k.data)
    S = qkt * D[None, :, :] * beta.data[:, None, :] * c
    y = np.einsum("blm,bmd->bld", S, v.data)

    def back(g):
        dS = np.einsum("bld,bmd->blm", g, v.data)
        dv = np.einsum("blm,bld->bmd", S, g)
        dqkt = dS * D[None, :, :] * beta.data[:, None, :] * c
        dbeta = (dS * D[None, :, :] * qkt * c).sum(axis=1)
        dq = np.einsum("blm,bmd->bld", dqkt, k.data)
        dk = np.einsum("blm,bld->bmd", dqkt, q.data)
        nn._acc(q, dq)
        nn._acc(k, dk)
        nn._acc(v, dv)
        nn._acc(beta, dbeta)
    return nn.custom(y, (q, k, v, beta), back)


# ---------------------------------------------------------------------------
# explicit references used by verify.py (not used in training)
# ---------------------------------------------------------------------------
def window_attn_reference(q, k, v, W, pos):
    B, L, d = q.shape
    y = np.zeros((B, L, d))
    for t in range(L):
        lo = max(0, t - W + 1)
        ks, vs = k[:, lo:t + 1, :], v[:, lo:t + 1, :]
        s = np.einsum("bd,bwd->bw", q[:, t, :], ks) / math.sqrt(d)
        s = s + pos[-(t - lo + 1):]
        z = s - s.max(axis=-1, keepdims=True)
        a = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
        y[:, t, :] = np.einsum("bw,bwd->bd", a, vs)
    return y


def decayed_attn_reference(q, k, v, beta, lam):
    B, L, d = q.shape
    S = np.zeros((B, d, d))
    y = np.zeros((B, L, d))
    c = 1.0 / math.sqrt(d)
    for t in range(L):
        S = lam * S + beta[:, t, None, None] * np.einsum(
            "bd,be->bde", k[:, t, :], v[:, t, :])
        y[:, t, :] = np.einsum("bd,bde->be", q[:, t, :], S) * c
    return y


# ---------------------------------------------------------------------------
# block + model
# ---------------------------------------------------------------------------
class SimplCoreBlock:
    def __init__(self, d, W, rhos, lam, seed):
        rng = np.random.default_rng(seed)
        sc = 1.0 / math.sqrt(d)
        p = lambda *s: nn.param(rng.standard_normal(s) * sc)
        self.Wq, self.Wk, self.Wv, self.Wu, self.Wo = (p(d, d) for _ in range(5))
        self.W1, self.W2 = p(d, 4 * d), p(4 * d, d)
        self.w_beta = nn.param(np.zeros(d))
        self.pos = nn.param(np.zeros(W))
        self.wlogits = nn.param(np.zeros(len(rhos)))
        self.alpha = nn.param(np.array(0.0))
        self.gamma = nn.param(np.array(0.0))
        self.ln1_g, self.ln1_b = nn.param(np.ones(d)), nn.param(np.zeros(d))
        self.ln2_g, self.ln2_b = nn.param(np.ones(d)), nn.param(np.zeros(d))
        self.W, self.rhos, self.lam = W, rhos, lam

    def params(self):
        return {k: v for k, v in self.__dict__.items() if isinstance(v, nn.Tensor)}

    def forward(self, x):
        L = x.shape[1]
        h = nn.layernorm(x, self.ln1_g, self.ln1_b)
        q, k, v = (nn.matmul(h, m) for m in (self.Wq, self.Wk, self.Wv))
        a = window_attn(q, k, v, self.W, self.pos)
        u = nn.matmul(h, self.Wu)
        g = toeplitz_mix(u, self.wlogits, self.rhos, L)
        wb = nn.reshape(self.w_beta, (self.w_beta.shape[0], 1))
        beta3 = nn.sigmoid(nn.matmul(h, wb))
        beta = nn.reshape(beta3, (beta3.shape[0], beta3.shape[1]))
        m = decayed_attn(q, k, v, beta, self.lam)
        y = nn.add(a, nn.add(nn.mul(g, self.alpha), nn.mul(m, self.gamma)))
        x = nn.add(x, nn.matmul(y, self.Wo))
        h2 = nn.layernorm(x, self.ln2_g, self.ln2_b)
        f = nn.matmul(nn.gelu(nn.matmul(h2, self.W1)), self.W2)
        return nn.add(x, f)


class SimplCoreLM:
    """Decoder-only LM. Every parameter below is inside the forward graph."""

    def __init__(self, V, d=96, n_layer=3, W=12,
                 rhos=(0.6, 0.9, 0.99), lam=0.9, maxL=512, seed=0):
        rng = np.random.default_rng(seed + 77)
        self.V, self.d, self.W, self.rhos, self.lam = V, d, W, rhos, lam
        self.E = nn.param(rng.standard_normal((V, d)) * 0.08)
        self.P = nn.param(rng.standard_normal((maxL, d)) * 0.02)
        self.blocks = [SimplCoreBlock(d, W, rhos, lam, seed + 1000 + i)
                       for i in range(n_layer)]
        self.lnf_g, self.lnf_b = nn.param(np.ones(d)), nn.param(np.zeros(d))
        self.Wh = nn.param(rng.standard_normal((d, V)) / math.sqrt(d))
        self.bh = nn.param(np.zeros(V))
        self.maxL = maxL

    def params(self):
        out = {"E": self.E, "P": self.P, "lnf_g": self.lnf_g,
               "lnf_b": self.lnf_b, "Wh": self.Wh, "bh": self.bh}
        for i, b in enumerate(self.blocks):
            for k, v in b.params().items():
                out["b%d.%s" % (i, k)] = v
        return out

    def forward(self, tok):
        B, L = tok.shape
        assert L <= self.maxL
        pos_idx = np.broadcast_to(np.arange(L)[None, :], (B, L))
        e = nn.add(nn.embedding(self.E, tok), nn.embedding(self.P, pos_idx))
        for b in self.blocks:
            e = b.forward(e)
        e = nn.layernorm(e, self.lnf_g, self.lnf_b)
        return nn.add(nn.matmul(e, self.Wh), self.bh)

    def loss(self, tok, tgt):
        return nn.cross_entropy(self.forward(tok), tgt)

    def logits(self, tok):
        return self.forward(tok).data


def params_of(model):
    return model.params()
