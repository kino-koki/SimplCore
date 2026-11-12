#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""REAL TRAINING, not a controlled synthetic task.

Corpus : tiny Shakespeare (real human text, 1,115,394 bytes), character level.
Split  : chronological (first 90% train, last 10% validation), never shuffled.
Task   : next-character prediction with a cross-entropy loss, trained by Adam.
        No key/value binding task, no analytic witness, no injected oracle.

Arms are loaded dynamically from
``_cloud_repo/SimplCore-main/simplcore/core/arch/evolution_mixer.py`` so that
each ARCHITECTURE STEP is a real code change pushed to the repository.

Everything is pure NumPy fp64 with hand-written analytic adjoints; every run
is preceded by a finite-difference gradcheck. No torch is available here, and
no result below is a simulated number.
"""
from __future__ import annotations

import argparse, importlib.util, inspect, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARCH = os.path.join(ROOT, "_cloud_repo", "SimplCore-main",
                    "simplcore", "core", "arch", "evolution_mixer.py")
CORPUS = os.path.join(HERE, "data", "tinyshakespeare.txt")
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

ARM_CLASS = {
    "A_static":        "StaticSGCMixer",
    "A_static_wide":   "StaticSGCMixer",
    "B_gated":         "GatedEvolutionMixer",
    "C_fused":         "FusedEvolutionMixer",
    "D_calibrated":    "CalibratedFusedMixer",
    "F_expanded":      "ExpandedCodebookMixer",
    "G_slotted":      "SlottedFusedMixer",
    "H_isolated":     "CreditIsolatedMixer",
    "H_bigram":       "BigramBiasMixer",
    "E_linear_attn":   "LinearAttentionReference",
}
ARM_D = {
    "A_static": 32, "A_static_wide": 56, "B_gated": 32,
    "C_fused": 32, "D_calibrated": 32, "F_expanded": 32,
    "G_slotted": 32, "H_isolated": 32, "H_bigram": 32,
    "E_linear_attn": 32,
}


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def load_corpus():
    text = open(CORPUS, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = len(data)
    cut = int(0.9 * n)
    return data[:cut], data[cut:], len(chars), text


def batch(data, B, L, rng):
    hi = len(data) - L - 1
    idx = rng.integers(0, hi, size=B)
    X = np.stack([data[i:i + L + 1] for i in idx])
    return X[:, :-1], X[:, 1:], idx


# ---------------------------------------------------------------------------
# reference arm: outer-product linear attention (NOT a SimplCore claim)
# ---------------------------------------------------------------------------
class LinearAttentionReference:
    """Order-2 outer-product state, content addressed. Reference only."""

    def __init__(self, d, rho=0.9, seed=0):
        rng = np.random.default_rng(seed + 1000)
        self.d = d
        sc = 1.0 / np.sqrt(d)
        self.W_q = rng.standard_normal((d, d)) * sc
        self.W_k = rng.standard_normal((d, d)) * sc
        self.W_v = rng.standard_normal((d, d)) * sc
        self.c = None

    def params(self):
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v}

    def forward(self, e):
        B, L, d = e.shape
        q = e @ self.W_q; k = e @ self.W_k; v = e @ self.W_v
        S = np.zeros((B, L, d, d)); y = np.empty_like(v)
        cur = np.zeros((B, d, d))
        for t in range(L):
            cur = cur + np.einsum("bd,be->bde", k[:, t, :], v[:, t, :])
            S[:, t, :, :] = cur
            y[:, t, :] = np.einsum("bd,bde->be", q[:, t, :], cur) / math.sqrt(d)
        self.c = dict(e=e, q=q, k=k, v=v, S=S, y=y)
        return y

    def backward(self, dy):
        c = self.c; e, q, k, v, S = c["e"], c["q"], c["k"], c["v"], c["S"]
        B, L, d = e.shape
        dq = np.zeros_like(q); dk = np.zeros_like(k); dv = np.zeros_like(v)
        dS = np.zeros((B, d, d))
        inv = 1.0 / math.sqrt(d)
        for t in range(L - 1, -1, -1):
            dS = dS + np.einsum("bd,be->bde", q[:, t, :], dy[:, t, :]) * inv
            dq[:, t, :] = np.einsum("bde,be->bd", S[:, t, :, :], dy[:, t, :]) * inv
            dk[:, t, :] = np.einsum("bde,be->bd", dS, v[:, t, :])
            dv[:, t, :] = np.einsum("bde,bd->be", dS, k[:, t, :])
        gW = {
            "W_q": np.einsum("bld,blm->dm", e, dq),
            "W_k": np.einsum("bld,blm->dm", e, dk),
            "W_v": np.einsum("bld,blm->dm", e, dv),
        }
        de = dq @ self.W_q.T + dk @ self.W_k.T + dv @ self.W_v.T
        return gW, de

    def invariants(self, Ls=(9, 65, 257)):
        return dict(arch_step="reference", kernel_static=False,
                    state_is_O_d2=True, has_NxN_score=False,
                    has_kv_cache=False, all_pass=True)


# ---------------------------------------------------------------------------
# model + analytic adjoint
# ---------------------------------------------------------------------------
class LM:
    def __init__(self, V, d, mixer, seed):
        rng = np.random.default_rng(seed + 77)
        self.E = rng.standard_normal((V, d)) * 0.1
        self.W_o = rng.standard_normal((d, V)) * (1.0 / math.sqrt(d))
        self.b = np.zeros(V)
        self.mixer = mixer
        self.V = V

    def params(self):
        p = {"E": self.E, "W_o": self.W_o, "b": self.b}
        p.update(self.mixer.params())
        return p

    def forward(self, tok):
        e = self.E[tok]
        y = self.mixer.forward(e, tok)
        logits = y @ self.W_o + self.b
        # STEP 6 credit-isolated read-out. Old arms have no logit_bias()
        # attribute, so this branch is absent for them and their numerical
        # behaviour is bit-identical to STEP 5.
        bias_raw = None
        bias_fn = getattr(self.mixer, "logit_bias", None)
        if bias_fn is not None:
            bias_raw = bias_fn(tok)
            logits = logits + np.asarray(self.mixer.gamma) * bias_raw
        z = logits - logits.max(axis=-1, keepdims=True)
        ex = np.exp(z)
        p = ex / ex.sum(axis=-1, keepdims=True)
        self.c = dict(tok=tok, e=e, y=y, prob=p, bias_raw=bias_raw)
        return p

    def loss_and_grad(self, tok, tgt):
        p = self.forward(tok)
        B, L, V = p.shape
        T = np.eye(V, dtype=np.float64)[tgt]
        loss = -np.log(np.maximum(np.sum(p * T, axis=-1), 1e-12)).mean()
        dl = (p - T) / (B * L)
        gWo = np.einsum("bld,blv->dv", self.c["y"], dl)
        gb = dl.sum(axis=(0, 1))
        dy = dl @ self.W_o.T
        gm, de = self.mixer.backward(dy)
        gE = np.zeros_like(self.E)
        np.add.at(gE, tok.ravel(), de.reshape(-1, de.shape[-1]))
        g = {"E": gE, "W_o": gWo, "b": gb}
        g.update(gm)
        if self.c["bias_raw"] is not None:
            # bias_raw depends only on token ids and the zero-order state:
            # no adjoint through the consolidation path, one scalar gradient.
            g["gamma"] = np.array(float(np.sum(dl * self.c["bias_raw"])))
        return float(loss), g


def gradcheck(lm, tok, tgt, seed=0):
    rng = np.random.default_rng(seed + 999)
    loss0, g = lm.loss_and_grad(tok, tgt)
    names = [n for n in g if g[n].size > 0]
    out = []
    for name in names:
        a = g[name]
        flat = a.ravel()
        if flat.size == 0:
            continue
        j = int(rng.integers(flat.size))
        p = lm.params()[name]
        pf = p.ravel()
        old = pf[j]
        eps = 1e-5
        pf[j] = old + eps; lp = lm.loss_and_grad(tok, tgt)[0]
        pf[j] = old - eps; lmn = lm.loss_and_grad(tok, tgt)[0]
        pf[j] = old
        num = (lp - lmn) / (2 * eps)
        rel = abs(num - flat[j]) / (abs(num) + abs(flat[j]) + 1e-9)
        out.append(dict(param=name, analytic=float(flat[j]),
                        numeric=float(num), rel_err=float(rel)))
    # restore cache state
    lm.loss_and_grad(tok, tgt)
    worst = max(o["rel_err"] for o in out)
    return dict(n_checked=len(out), worst_rel_err=worst,
                passed=bool(worst < 2e-4), details=out)


def adam_step(params, grads, state, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    for n, p in params.items():
        g = grads[n]
        m, v = state[n]
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        p -= lr * mh / (np.sqrt(vh) + eps)


# ---------------------------------------------------------------------------
# evaluation: overall bpc and long/short-range split
# ---------------------------------------------------------------------------
def distances(val, tok):
    """For each target position, distance to the most recent previous
    occurrence of that same character in the full validation stream."""
    # tok is a batch of windows; compute on the flat validation array instead
    out = []
    for t in range(1, len(val)):
        c = val[t]
        prev = np.nonzero(val[:t] == c)[0]
        out.append(int(t - prev[-1]) if len(prev) else t)
    return np.array(out)


def char_distances(seq, cap=512):
    """Distance to the most recent previous occurrence of the same character.
    Computed once on the real validation stream, not on a synthetic task."""
    last = {}
    d = np.empty(len(seq), dtype=np.int64)
    for t, c in enumerate(seq):
        i = int(c)
        d[t] = t - last.get(i, -10 ** 9)
        last[i] = t
        if d[t] > cap:
            d[t] = cap
    return d


def evaluate(lm, val, B, L, seed, dist=None):
    rng = np.random.default_rng(seed + 4242)
    toks, tgts, idx = batch(val, B, L, rng)
    p = lm.forward(toks)
    T = np.eye(lm.V)[tgts]
    nll = -np.log(np.maximum(np.sum(p * T, axis=-1), 1e-12)) / math.log(2.0)
    res = dict(bpc=float(nll.mean()), bpc_std=float(nll.std()),
               n_positions=int(nll.size))
    if dist is not None:
        # target at window position l has absolute index idx[b] + 1 + l
        pos = idx[:, None] + 1 + np.arange(L)[None, :]
        dm = dist[np.clip(pos, 0, len(dist) - 1)]
        long_m = dm > 16
        short_m = dm <= 4
        res["bpc_long"] = float(nll[long_m].mean()) if long_m.any() else None
        res["bpc_short"] = float(nll[short_m].mean()) if short_m.any() else None
        res["n_long"] = int(long_m.sum())
        res["n_short"] = int(short_m.sum())
    return res


# ---------------------------------------------------------------------------
def load_arch_module():
    spec = importlib.util.spec_from_file_location("evolution_mixer", ARCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_arm(arm, seed, steps, B, L, lr, mixer_kwargs=None):
    mod = load_arch_module()
    cname = ARM_CLASS[arm]
    if cname == "LinearAttentionReference":
        Mix = LinearAttentionReference
    else:
        Mix = getattr(mod, cname, None)
        if Mix is None:
            return dict(arm=arm, seed=seed, error=f"class {cname} absent at ARCH_STEP {mod.ARCH_STEP}")
    tr, va, V, _ = load_corpus()
    dist = char_distances(va)
    d = ARM_D[arm]
    sig = inspect.signature(Mix.__init__)
    kw = dict(d=d, seed=seed)
    if "n_symbols" in sig.parameters:
        kw["n_symbols"] = V          # real alphabet, never folded
    if mixer_kwargs:
        kw.update(mixer_kwargs)
    mixer = Mix(**kw)
    lm = LM(V, d, mixer, seed)
    params = lm.params()
    state = {n: (np.zeros_like(a), np.zeros_like(a)) for n, a in params.items()}
    rng = np.random.default_rng(seed + 31337)
    tk, tg, _ = batch(tr, B, L, rng)
    if hasattr(mixer, "observe"):
        mixer.observe(tk, tg)   # one real batch of experience, zero-order
    gc = gradcheck(lm, tk, tg, seed)
    if not gc["passed"]:
        return dict(arm=arm, seed=seed, error="gradcheck FAIL", gradcheck=gc)
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        tk, tg, _ = batch(tr, B, L, rng)
        if hasattr(mixer, "observe"):
            mixer.observe(tk, tg)
        loss, g = lm.loss_and_grad(tk, tg)
        adam_step(params, g, state, step, lr=lr)
        if step % max(1, steps // 10) == 0 or step == 1:
            ev = evaluate(lm, va, B, L, seed, dist)
            curve.append(dict(step=step, train_bpc=float(loss / math.log(2.0)),
                              val_bpc=ev["bpc"], bpc_long=ev.get("bpc_long"),
                              bpc_short=ev.get("bpc_short")))
    final = evaluate(lm, va, B, L, seed, dist)
    nparam = sum(a.size for a in params.values())
    try:
        inv = mixer.invariants()
    except Exception as ex:
        inv = dict(error=str(ex))
    return dict(arm=arm, seed=seed, d=d, V=V, steps=steps, B=B, L=L, lr=lr,
                n_params=int(nparam), final_val_bpc=final["bpc"],
                bpc_long=final.get("bpc_long"), bpc_short=final.get("bpc_short"),
                n_long=final.get("n_long"), n_short=final.get("n_short"),
                final_train_bpc=curve[-1]["train_bpc"], curve=curve,
                gradcheck=dict(n_checked=gc["n_checked"],
                               worst_rel_err=gc["worst_rel_err"],
                               passed=gc["passed"]),
                invariants=inv,
                param_count_note="arm parameter counts are reported, not equalized",
                elapsed_s=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A_static,A_static_wide,E_linear_attn")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--L", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    for arm in arms:
        for sd in seeds:
            r = run_arm(arm, sd, a.steps, a.B, a.L, a.lr)
            rows.append(r)
            tag = "OK" if "final_val_bpc" in r else "FAIL"
            print(f"  {tag:4s} {arm:16s} s={sd}  " +
                  (f"val_bpc={r['final_val_bpc']:.4f} params={r['n_params']} "
                   f"gc={r['gradcheck']['worst_rel_err']:.1e} {r['elapsed_s']}s"
                   if tag == "OK" else str(r.get("error"))), flush=True)
    out = a.out or os.path.join(OUT, "realtrain_" + "_".join(arms) + ".json")
    json.dump(dict(corpus="tiny_shakespeare_char_level",
                   corpus_bytes=os.path.getsize(CORPUS),
                   split="chronological 90/10, no shuffle",
                   optimizer="Adam", steps=a.steps, B=a.B, L=a.L, lr=a.lr,
                   arch_step=load_arch_module().ARCH_STEP,
                   rows=rows), open(out, "w"), indent=2)
    print("  ->", out, flush=True)


if __name__ == "__main__":
    main()
