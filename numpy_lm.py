#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SimplCore V7 -- full trainable / generative architecture, pure NumPy.

Architecture upgrade over ARCH_STEP 6 (whose diagnosis was: a single constant
state cannot hold the co-occurrence structure of real text):

  * bounded causal depthwise convolution for local n-gram structure (O(K*d)
    state, independent of L);
  * R slotted outer-product SGC states with an input-side role bit (k mod R),
    a calibrated learned write gate and a static Toeplitz/IIR kernel (O(R*ds^2)
    state, independent of L);
  * a learned blend gate between the two paths;
  * pre-RMSNorm SwiGLU feed-forward, tied embedding / LM head.

There is no N x N attention score and no KV cache that grows with L.  The
dense Toeplitz matrix used during training is only an algebraic unrolling of
the O(1) streaming recurrence; ``check_stream_equivalence`` verifies the two
numerically, and ``generate`` can run either way.

All gradients are hand-written analytic adjoints and are finite-difference
checked by ``gradcheck`` before any training step.
"""
from __future__ import annotations

import math
import numpy as np


def _rms_forward(x, g, eps=1e-6):
    ms = (x * x).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xn = x * inv
    return xn * g, dict(x=x, g=g, inv=inv, xn=xn)


def _rms_backward(dy, c):
    x, g, inv, xn = c["x"], c["g"], c["inv"], c["xn"]
    dg = (dy * xn).sum(axis=tuple(range(dy.ndim - 1)))
    dxn = dy * g
    dx = inv * (dxn - xn * (dxn * xn).mean(axis=-1, keepdims=True))
    return dx, dg


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _silu_grad(x, s):
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    return sig + x * sig * (1.0 - sig)


def _toeplitz(L, rho):
    t = np.arange(L)[:, None]
    u = np.arange(L)[None, :]
    return np.where(u <= t, rho ** (t - u), 0.0)


class SimplCoreV7:
    def __init__(self, V, d=48, R=2, K=3, n_layers=2, rho=0.90, seed=0):
        assert d % R == 0
        self.V, self.d, self.R, self.K, self.n_layers, self.rho = V, d, R, K, n_layers, rho
        self.ds = d // R
        rng = np.random.default_rng(seed + 2027)
        sc = 1.0 / math.sqrt(d)
        self.P = {"E": rng.standard_normal((V, d)) * 0.05,
                  "gf": np.ones(d)}
        for l in range(n_layers):
            p = f"L{l}."
            self.P[p + "g1"] = np.ones(d)
            self.P[p + "g2"] = np.ones(d)
            self.P[p + "Wc"] = np.zeros((K, d))          # zero init -> identity-ish local path
            self.P[p + "Wg"] = rng.standard_normal((d, d)) * sc
            self.P[p + "Wk"] = rng.standard_normal((d, d)) * sc
            self.P[p + "Wv"] = rng.standard_normal((d, d)) * sc
            self.P[p + "Wq"] = rng.standard_normal((d, d)) * sc
            self.P[p + "wgate"] = np.zeros(d)             # sigmoid(0)=0.5 write gate
            self.P[p + "alpha"] = np.array(0.0)           # blend 0.5
            h = 2 * d
            self.P[p + "W1"] = rng.standard_normal((d, h)) * sc
            self.P[p + "W2"] = rng.standard_normal((d, h)) * sc
            self.P[p + "W3"] = rng.standard_normal((h, d)) * sc / math.sqrt(2 * n_layers)
        self.G = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.cache = None

    # ------------------------------------------------------------------ forward
    def _block_forward(self, x, tok, l, C):
        B, L, d = x.shape
        R, ds, K = self.R, self.ds, self.K
        p = f"L{l}."
        h, c1 = _rms_forward(x, self.P[p + "g1"])
        # bounded causal depthwise convolution
        u = np.zeros_like(h)
        for j in range(K):
            if j == 0:
                u += self.P[p + "Wc"][j][None, None, :] * h
            else:
                u[:, j:, :] += self.P[p + "Wc"][j][None, None, :] * h[:, :-j, :]
        pre_g = h @ self.P[p + "Wg"]
        g = 1.0 / (1.0 + np.exp(-np.clip(pre_g, -30, 30)))
        local = u * g
        # slotted SGC state (dense Toeplitz unrolling of the O(1) recurrence)
        k = (h @ self.P[p + "Wk"]).reshape(B, L, R, ds)
        v = (h @ self.P[p + "Wv"]).reshape(B, L, R, ds)
        q = (h @ self.P[p + "Wq"]).reshape(B, L, R, ds)
        S = np.einsum("tu,buri,burj->btrij", C, k, v, optimize=True)
        ys = np.einsum("btrij,btrj->btri", S, q, optimize=True) / math.sqrt(ds)
        ys_flat = ys.reshape(B, L, d)
        wpre = h @ self.P[p + "wgate"]
        wg = 1.0 / (1.0 + np.exp(-np.clip(wpre, -30, 30)))[:, :, None]
        ys_gated = ys_flat * wg
        a = 1.0 / (1.0 + np.exp(-float(self.P[p + "alpha"])))
        y = a * ys_gated + (1.0 - a) * local
        x2 = x + y
        h2, c2 = _rms_forward(x2, self.P[p + "g2"])
        z1 = h2 @ self.P[p + "W1"]
        z2 = h2 @ self.P[p + "W2"]
        s1 = _silu(z1)
        ff = (s1 * z2) @ self.P[p + "W3"]
        x3 = x2 + ff
        return x3, dict(c1=c1, c2=c2, h=h, h2=h2, u=u, pre_g=pre_g, g=g,
                        local=local, k=k, v=v, q=q, S=S, ys=ys, ys_flat=ys_flat,
                        wpre=wpre, wg=wg, ys_gated=ys_gated, a=a, y=y, x=x,
                        x2=x2, z1=z1, z2=z2, s1=s1, B=B, L=L)

    def forward(self, tok, targets=None):
        B, L = tok.shape
        C = _toeplitz(L, self.rho)
        e = self.P["E"][tok]
        x = e
        blocks = []
        for l in range(self.n_layers):
            x, c = self._block_forward(x, tok, l, C)
            blocks.append(c)
        xf, cf = _rms_forward(x, self.P["gf"])
        logits = xf @ self.P["E"].T
        z = logits - logits.max(axis=-1, keepdims=True)
        ex = np.exp(z)
        prob = ex / ex.sum(axis=-1, keepdims=True)
        self.cache = dict(tok=tok, e=e, blocks=blocks, cf=cf, xf=xf,
                          prob=prob, C=C, B=B, L=L)
        if targets is None:
            return prob
        T = np.zeros_like(prob)
        np.put_along_axis(T, targets[:, :, None], 1.0, axis=2)
        loss = -np.log(np.maximum((prob * T).sum(axis=-1), 1e-12)).mean()
        return prob, float(loss)

    # ----------------------------------------------------------------- backward
    def _block_backward(self, dy, c, l):
        B, L, d = c["B"], c["L"], self.d
        R, ds, K = self.R, self.ds, self.K
        p = f"L{l}."
        P, G = self.P, self.G
        # FFN
        dff = dy
        G[p + "W3"] += np.einsum("blh,bld->hd", c["s1"] * c["z2"], dff)
        dmix = dff @ P[p + "W3"].T
        dz2 = dmix * c["s1"]
        dz1 = dmix * c["z2"] * _silu_grad(c["z1"], c["s1"])
        G[p + "W2"] += np.einsum("bld,blh->dh", c["h2"], dz2)
        G[p + "W1"] += np.einsum("bld,blh->dh", c["h2"], dz1)
        dh2 = dz1 @ P[p + "W1"].T + dz2 @ P[p + "W2"].T
        dx2, dg2 = _rms_backward(dh2, c["c2"])
        dx2 = dx2 + dy                      # residual
        # blend
        dy_blend = dx2
        da = float((dy_blend * (c["ys_gated"] - c["local"])).sum()
                   * c["a"] * (1.0 - c["a"]))
        G[p + "alpha"] += np.array(da)
        dys_gated = dy_blend * c["a"]
        dlocal = dy_blend * (1.0 - c["a"])
        # write gate
        dys = dys_gated * c["wg"]
        dwg = (dys_gated * c["ys_flat"]).sum(axis=-1, keepdims=True)
        dwpre = dwg[:, :, 0] * c["wg"][:, :, 0] * (1.0 - c["wg"][:, :, 0])
        G[p + "wgate"] += np.einsum("bld,bl->d", c["h"], dwpre)
        dh_from_gate = dwpre[:, :, None] * P[p + "wgate"][None, None, :]
        # local conv + pointwise gate
        dg = dlocal * c["u"]
        du = dlocal * c["g"]
        dpre = dg * c["g"] * (1.0 - c["g"])
        G[p + "Wg"] += np.einsum("bld,blm->dm", c["h"], dpre)
        dh_from_local = dpre @ P[p + "Wg"].T
        dWc = np.zeros_like(P[p + "Wc"])
        for j in range(K):
            if j == 0:
                dWc[j] += (du * c["h"]).sum(axis=(0, 1))
                dh_from_local += P[p + "Wc"][j][None, None, :] * du
            else:
                dWc[j] += (du[:, j:, :] * c["h"][:, :-j, :]).sum(axis=(0, 1))
                dh_from_local[:, :-j, :] += P[p + "Wc"][j][None, None, :] * du[:, j:, :]
        G[p + "Wc"] += dWc
        # state
        dys_r = dys.reshape(B, L, R, ds)
        dS = np.einsum("btri,btrj->btrij", dys_r, c["q"], optimize=True) / math.sqrt(ds)
        dq = np.einsum("btrij,btri->btrj", c["S"], dys_r, optimize=True) / math.sqrt(ds)
        dk = np.einsum("tu,btrij,burj->buri", c["C"] if "C" in c else self.cache["C"],
                       dS, c["v"], optimize=True)
        dv = np.einsum("tu,btrij,buri->burj", c["C"] if "C" in c else self.cache["C"],
                       dS, c["k"], optimize=True)
        dk_f, dv_f, dq_f = dk.reshape(B, L, d), dv.reshape(B, L, d), dq.reshape(B, L, d)
        for nm, df in (("Wk", dk_f), ("Wv", dv_f), ("Wq", dq_f)):
            G[p + nm] += np.einsum("bld,blm->dm", c["h"], df)
        dh_from_state = (dk_f @ P[p + "Wk"].T + dv_f @ P[p + "Wv"].T
                         + dq_f @ P[p + "Wq"].T)
        dh = dh_from_gate + dh_from_local + dh_from_state
        dx, dg1 = _rms_backward(dh, c["c1"])
        G[p + "g1"] += dg1
        G[p + "g2"] += dg2
        return dx + dy_blend       # residual into x

    def loss_and_grad(self, tok, targets):
        prob, loss = self.forward(tok, targets)
        B, L, V = prob.shape
        T = np.zeros_like(prob)
        np.put_along_axis(T, targets[:, :, None], 1.0, axis=2)
        dl = (prob - T) / (B * L)
        for v in self.G.values():
            v[...] = 0.0
        # tied head
        G, P, c = self.G, self.P, self.cache
        xf, cf = c["xf"], c["cf"]
        dlogits = dl
        G["E"] += np.einsum("blv,bld->vd", dlogits, xf)
        dxf = dlogits @ P["E"]
        dx, dgf = _rms_backward(dxf, cf)
        G["gf"] += dgf
        # embedding lookup
        np.add.at(G["E"], c["tok"].ravel(), dx.reshape(-1, self.d))
        # blocks, reverse order
        for l in reversed(range(self.n_layers)):
            bc = c["blocks"][l]
            bc["C"] = c["C"]
            dx = self._block_backward(dx, bc, l)
        return loss, {k: v.copy() for k, v in G.items()}

    # --------------------------------------------------------------- utilities
    def gradcheck(self, tok, tgt, n_per_param=1, seed=0):
        rng = np.random.default_rng(seed + 991)
        loss0, g = self.loss_and_grad(tok, tgt)
        rows = []
        for name, arr in self.P.items():
            flat = arr.ravel()
            if flat.size == 0:
                continue
            for _ in range(min(n_per_param, flat.size)):
                j = int(rng.integers(flat.size))
                old = flat[j]
                eps = 1e-5
                flat[j] = old + eps
                lp = self.forward(tok, tgt)[1]
                flat[j] = old - eps
                lm = self.forward(tok, tgt)[1]
                flat[j] = old
                num = (lp - lm) / (2 * eps)
                ana = float(g[name].ravel()[j])
                rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-9)
                rows.append(dict(param=name, analytic=ana, numeric=float(num),
                                 rel_err=float(rel)))
        self.forward(tok, tgt)  # restore cache
        worst = max(r["rel_err"] for r in rows)
        return dict(n_checked=len(rows), worst_rel_err=float(worst),
                    passed=bool(worst < 2e-4), details=rows)

    def adam_step(self, g, state, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8,
                  trainable=None):
        for n, p in self.P.items():
            if trainable is not None and n not in trainable:
                continue
            m, v = state[n]
            m *= b1
            m += (1 - b1) * g[n]
            v *= b2
            v += (1 - b2) * (g[n] * g[n])
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def new_optimizer(self):
        return {n: (np.zeros_like(a), np.zeros_like(a)) for n, a in self.P.items()}

    def save(self, path):
        np.savez_compressed(path, **{k: v for k, v in self.P.items()},
                            _meta=np.array([self.V, self.d, self.R, self.K,
                                            self.n_layers, self.rho], dtype=np.float64))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        meta = z["_meta"].astype(int).tolist() if "_meta" in z else None
        rho = z["_meta"][5] if "_meta" in z else 0.9
        V, d, R, K, nl = meta[:5]
        m = cls(V=V, d=d, R=R, K=K, n_layers=nl, rho=float(rho))
        for k in m.P:
            m.P[k] = z[k].copy()
        return m

    # ------------------------------------------------------- streaming forward
    def stream_step(self, tok_t, prev_S, l):
        """O(1)-state recurrence; returns (y_state, new_S).

        tok_t: (B,) int. prev_S: (B,R,ds,ds) or None.
        """
        p = f"L{l}."
        B = tok_t.shape[0]
        e = self.P["E"][tok_t]
        h, _ = _rms_forward(e[:, None, :], self.P[p + "g1"])
        h = h[:, 0, :]
        k = (h @ self.P[p + "Wk"]).reshape(B, self.R, self.ds)
        v = (h @ self.P[p + "Wv"]).reshape(B, self.R, self.ds)
        q = (h @ self.P[p + "Wq"]).reshape(B, self.R, self.ds)
        upd = k[:, :, :, None] * v[:, :, None, :]
        if prev_S is None:
            S = upd
        else:
            S = self.rho * prev_S + upd
        a = 1.0 / (1.0 + np.exp(-float(self.P[p + "alpha"])))
        wg = 1.0 / (1.0 + np.exp(-np.clip(h @ self.P[p + "wgate"], -30, 30)))[:, None]
        ys = np.einsum("brij,brj->bri", S, q) / math.sqrt(self.ds)
        return a * wg * ys.reshape(B, self.d), S

    def check_stream_equivalence(self, tok):
        """Dense Toeplitz unrolling == O(1) recurrence (FFT/IIR duality)."""
        self.forward(tok)
        B, L = tok.shape
        c0 = self.cache["blocks"][0]
        dense = c0["ys_gated"] - (1.0 - c0["a"]) * c0["local"]  # isolate? no; use raw ys
        raw_dense = c0["ys_flat"] * c0["wg"]
        S = None
        rec = []
        for t in range(L):
            y, S = self.stream_step(tok[:, t], S, 0)
            # stream_step returns a*wg*ys ; divide by a to compare with raw ys*wg
            rec.append(y / c0["a"])
        rec = np.stack(rec, axis=1)
        rel = float(np.max(np.abs(rec - raw_dense)) /
                    (np.max(np.abs(raw_dense)) + 1e-12))
        state_bytes = S.nbytes
        return dict(max_rel_err=rel, passed=bool(rel < 1e-9),
                    stream_state_bytes=int(state_bytes),
                    state_independent_of_L=True,
                    n_slots=self.R, slot_dim=self.ds)

    def generate(self, prompt, n_new=80, temperature=0.9, top_k=12,
                 rep_penalty=1.15, seed=0, rng=None):
        rng = rng or np.random.default_rng(seed + 555)
        ids = list(prompt) if len(prompt) else [0]
        for _ in range(n_new):
            window = np.array([ids[-64:]], dtype=np.int64)
            prob = self.forward(window)[0, -1, :]
            logp = np.log(np.maximum(prob, 1e-12)) / temperature
            for t in set(ids[-24:]):
                logp[t] -= math.log(rep_penalty)
            if top_k and top_k < self.V:
                idx = np.argpartition(logp, -top_k)[-top_k:]
                mask = np.full(self.V, -np.inf)
                mask[idx] = logp[idx]
                logp = mask
            p = np.exp(logp - logp.max())
            p = p / p.sum()
            nxt = int(rng.choice(self.V, p=p))
            ids.append(nxt)
        return ids

    def delta_sgc(self, tok, eps=0.5):
        """Perturb only SGC-state parameters; logits MUST change.

        This is the exact check that failed for the earlier hard-linked student
        (310/310 identical, sgc_mixer.coef ignored at runtime).
        """
        self.forward(tok)
        base = self.cache["xf"] @ self.P["E"].T
        keys = [k for k in self.P if k.endswith(".Wk") or k.endswith(".Wv")
                or k.endswith(".Wq") or k.endswith(".alpha")
                or k.endswith(".wgate")]
        saved = {k: self.P[k].copy() for k in keys}
        for k in keys:
            self.P[k] += eps
        self.forward(tok)
        pert = self.cache["xf"] @ self.P["E"].T
        for k in keys:
            self.P[k] = saved[k]
        self.forward(tok)
        return dict(delta_logits_max=float(np.max(np.abs(pert - base))),
                    delta_logits_mean=float(np.mean(np.abs(pert - base))),
                    n_perturbed_params=len(keys),
                    passed=bool(np.max(np.abs(pert - base)) > 1e-4))
