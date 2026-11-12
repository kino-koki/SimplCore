#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimplCore ARCH_STEP 8 -- O(1)-state streaming inference.

This is the *same* model as ``simplcore_lm.SimplCoreLM`` (whose weights are
``model.npz`` and whose 1500-step real-corpus training record is in
``train_record.json``), rewritten as a per-token recurrence.  The three fused
matrix operators are replaced by their exact constant-size states:

  window_attn  W x W softmax      -> ring buffer of the last W (k,v) pairs
  toeplitz_mix L x L Toeplitz     -> K first-order IIR states  (FFT/IIR dual)
  decayed_attn L x L gated scores -> one d x d outer-product state

No KV cache that grows with L, no N x N score, no attention matrix at
generation time.  ``verify_final.py`` checks stream_step against the full
unroll on the *trained* weights to ~1e-12.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.special import erf

import nn
import simplcore_lm as A

ARCH_STEP = 8
HERE = os.path.dirname(os.path.abspath(__file__))


def _gelu(x):
    z = x / math.sqrt(2.0)
    return 0.5 * x * (1.0 + erf(z))


def _ln(x, g, b, eps=1e-5):
    """x: (..., d), identical to nn.layernorm's forward pass."""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return (xc / np.sqrt(var + eps)) * g + b


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class StreamState:
    """Constant-size state for one block."""

    __slots__ = ("kh", "vh", "iir", "S")

    def __init__(self, d, K):
        self.kh = []          # last W keys   (each (d,))
        self.vh = []          # last W values (each (d,))
        self.iir = np.zeros((K, d), dtype=np.float64)
        self.S = np.zeros((d, d), dtype=np.float64)

    def n_bytes(self):
        n = self.iir.nbytes + self.S.nbytes
        for a in self.kh:
            n += a.nbytes
        for a in self.vh:
            n += a.nbytes
        return int(n)


class StreamSimplCoreLM:
    """Per-token recurrence, numerically equal to SimplCoreLM.logits(prefix)."""

    def __init__(self, model: A.SimplCoreLM):
        self.m = model
        self.d = model.d
        self.W = model.W
        self.K = len(model.rhos)
        self.rhos = np.asarray(model.rhos, dtype=np.float64)
        self.lam = model.lam
        self.P = model.params()

    def new_state(self):
        return [StreamState(self.d, self.K) for _ in self.m.blocks]

    # -------------------------------------------------------------- one token
    def step(self, tok_id, t, states):
        """Return (logits (V,), updated states). t is the absolute position."""
        p = self.P
        x = p["E"].data[tok_id][None, :] + p["P"].data[t][None, :]
        d, W, inv = self.d, self.W, 1.0 / math.sqrt(self.d)

        for bi, blk in enumerate(self.m.blocks):
            st = states[bi]
            pre = "b%d." % bi
            h = _ln(x, p[pre + "ln1_g"].data, p[pre + "ln1_b"].data)  # (1,d)
            q = (h @ p[pre + "Wq"].data)[0]
            k = (h @ p[pre + "Wk"].data)[0]
            v = (h @ p[pre + "Wv"].data)[0]

            # ---- local fixed-window content addressing (ring buffer) ----
            st.kh.append(k)
            st.vh.append(v)
            if len(st.kh) > W:
                st.kh.pop(0)
                st.vh.pop(0)
            ell = len(st.kh)
            ks = np.stack(st.kh, axis=0)                  # (ell,d)
            vs = np.stack(st.vh, axis=0)
            s = (ks @ q) * inv + p[pre + "pos"].data[W - ell:]
            z = s - s.max()
            aw = np.exp(z) / np.exp(z).sum()
            a = aw @ vs                                    # (d,)

            # ---- static multi-pole Toeplitz kernel, IIR form ----
            u = (h @ p[pre + "Wu"].data)[0]                # (d,)
            st.iir = self.rhos[:, None] * st.iir + u[None, :]
            w = _softmax(p[pre + "wlogits"].data)
            g = w @ st.iir                                  # (d,)

            # ---- gated self-evolving outer-product state ----
            beta = float(_sigmoid(float(h[0] @ p[pre + "w_beta"].data)))
            st.S = self.lam * st.S + beta * np.outer(k, v)
            mm = (q @ st.S) * inv                           # (d,)

            alpha = float(p[pre + "alpha"].data)
            gamma = float(p[pre + "gamma"].data)
            y = (a[None, :] + alpha * g[None, :] + gamma * mm[None, :])
            x = x + y @ p[pre + "Wo"].data
            h2 = _ln(x, p[pre + "ln2_g"].data, p[pre + "ln2_b"].data)
            f = _gelu(h2 @ p[pre + "W1"].data) @ p[pre + "W2"].data
            x = x + f

        x = _ln(x, p["lnf_g"].data, p["lnf_b"].data)
        logits = (x @ p["Wh"].data + p["bh"].data)[0]
        return logits, states

    def logits_of_prefix(self, ids):
        """Replay a prefix through the O(1) state; return the last logits."""
        st = self.new_state()
        for t, i in enumerate(ids):
            lg, st = self.step(int(i), t, st)
        return lg, st

    # ------------------------------------------------------------ generation
    def generate(self, prompt_ids, n_new=200, temp=0.85, top_p=0.92, seed=7,
                 maxL=None):
        """Pure O(1)-state generation.  Re-seeds the state if t reaches maxL
        (identical to the full-unroll sliding-window convention)."""
        maxL = maxL or self.m.maxL
        rng = np.random.default_rng(seed)
        ids = [int(i) for i in prompt_ids]
        st = self.new_state()
        t = 0
        # prime the state with the prompt
        for i in ids:
            lg, st = self.step(i, t, st)
            t += 1
        for _ in range(n_new):
            if t >= maxL:                       # re-base positions, keep last maxL
                ids = ids[-maxL:]
                st = self.new_state()
                t = 0
                for i in ids:
                    lg, st = self.step(i, t, st)
                    t += 1
            z = lg / temp
            z = z - z.max()
            pr = np.exp(np.clip(z, -60, 60))
            pr /= pr.sum()
            o = np.argsort(pr)[::-1]
            keep = o[:int(np.searchsorted(np.cumsum(pr[o]), top_p)) + 1]
            pk = pr[keep] / pr[keep].sum()
            nxt = int(rng.choice(keep, p=pk))
            ids.append(nxt)
            lg, st = self.step(nxt, t, st)
            t += 1
        return ids


def load_trained(here=HERE):
    cfg = json.load(open(os.path.join(here, "config.json"), encoding="utf-8"))
    m = A.SimplCoreLM(cfg["V"], d=cfg["d"], n_layer=cfg["layers"], W=cfg["W"],
                      rhos=tuple(cfg["rhos"]), lam=cfg["lam"],
                      maxL=cfg["maxL"], seed=cfg["seed"])
    z = np.load(os.path.join(here, "model.npz"))
    for n, p in m.params().items():
        np.copyto(p.data, z["p:" + n])
    return m, cfg


# ---------------------------------------------------------------------------
# exact matrix-form references, used only by the white-box checker
# ---------------------------------------------------------------------------
def toeplitz_iir_reference(u, wlogits, rhos):
    """Explicit O(L^2) matrix form (the operator used in training)."""
    L = u.shape[1]
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    As = [np.where(j <= i, r ** (i - j), 0.0) for r in rhos]
    w = _softmax(wlogits)
    Au = np.stack([A_ @ u[0] for A_ in As])
    return np.einsum("k,kld->ld", w, Au)          # (L,d)


def toeplitz_iir_stream(u, wlogits, rhos):
    """The O(L) recurrence actually used by StreamSimplCoreLM."""
    K, d = len(rhos), u.shape[-1]
    w = _softmax(wlogits)
    s = np.zeros((K, d))
    out = np.empty((u.shape[1], d))
    for t in range(u.shape[1]):
        s = np.asarray(rhos)[:, None] * s + u[0, t][None, :]
        out[t] = w @ s
    return out
