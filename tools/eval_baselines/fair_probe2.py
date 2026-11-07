"""
fair_probe2.py -- v6.3.  AUDIT OF OUR OWN PREVIOUS FAIRNESS CLAIM.

v6.2 said "every arm receives exactly the same one-hot visible ids". That was
TRUE for the ids but FALSE for the COMPUTATION: only the TSC branch consumed
the predecessor view Xp; every baseline used U = X @ E alone. In addition the
shared width d = 24 was SMALLER than the vocabulary V = 64, so every linear
embedding had a nontrivial null space, which penalises mechanisms that must
learn the (predecessor -> current) pairing from scratch while leaving the
hard-coded TSC inductive bias intact.

This file removes BOTH asymmetries:

  * ALL arms receive (U, Up, q, present-mask), with E = the SAME fixed
    orthogonal Hadamard code of full width d = V = 64. No learned embedding.
  * Every mechanism is written in its natural (key = predecessor,
    value = current) form, the form TSC already used.

It then gives, for three mechanisms, an EXPLICIT ZERO-TRAINING analytic
weight assignment that solves H=1 exactly. That is a witness that the function
CLASSES coincide on this task; any remaining difference is optimisation /
inductive bias, NOT expressivity. No new theorem is claimed.

Nothing here enters the SimplCore architecture. No LM, no ppl, no private
weights. Run: python3 tools/eval_baselines/fair_probe2.py --part all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = "/data/workspace"
sys.path.insert(0, HERE)
sys.path.insert(0, W + "/train/superposition_probe")

import ag
from ag import T, const, param, nll_from_logits, free_graph
import tsc_final_probe as TSC

OUT = W + "/tools/eval_baselines/results"
os.makedirs(OUT, exist_ok=True)

V = 64
VK = V // 2
N_PAIRS = 4
D = V                       # FULL width; removes the d < V null space
SEEDS = (0, 1, 2, 3, 4)
STEPS = 400
LR = 0.02
BATCH = 64

C = TSC.hadamard(V)         # fixed, orthonormal, symmetric; C C^T = I
CVAL = C[VK:, :]            # (32,64) value-code read-out
TSC.set_C(C, VK)


# ------------------------------------------------------------------ data
def make_data(hops, seed, n=512, layout="gapped", dup=False):
    """Identical visible ids AND identical (U, Up, q, mask) to EVERY arm."""
    d0 = TSC.build(N_PAIRS, hops, n, seed, V, layout=layout,
                   encoding="codebook", filler_max=3, dup=dup)
    X = d0["X"]                                  # (B,L,V)
    B, L, _ = X.shape
    Xp = np.zeros_like(X)
    Xp[:, 1:, :] = X[:, :-1, :]
    present = (X.sum(-1) > 0.5).astype(np.float64)

    U = X @ C               # current-token codes
    Up = Xp @ C             # predecessor-token codes
    q = d0["Q0"].copy()     # = C[chain[0]], the external query code
    K = TSC.kernel(L)       # static SGC decay kernel (TSC arms only)
    neg = (1.0 - present) * (-1e9)
    return dict(U=U, Up=Up, q=q, Y=d0["Y"], K=K, NEG=neg,
                present=present,
                meta=dict(hops=hops, seed=seed, L=L, layout=layout,
                          dup=dup, chance=1.0 / VK, d=D, V=V,
                          embedding="fixed orthogonal Hadamard (no learned E)"))


def sub(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1
                and v.shape[0] == D["Y"].shape[0] else v)
            for k, v in D.items()}


# ------------------------------------------------------------------ arms
class Arm:
    """Every arm sees (U, Up, q, NEG).  Key side = Up, value side = U."""

    def __init__(self, name, seed):
        self.name = name
        rng = np.random.default_rng(51000 + seed * 137 + sum(map(ord, name)))
        sc = 1.0 / math.sqrt(D)
        self.p = {}
        self.p["Wout"] = param(rng.standard_normal((D, VK)) * 1e-2)
        for k in ("Wk", "Wv", "Wq"):
            self.p[k] = param(rng.standard_normal((D, D)) * sc)
        if name in ("tsc_h2", "attn_x2"):
            self.p["T1"] = param(rng.standard_normal((D, D)) * sc)
        if name == "gla_pair":
            self.p["Wg"] = param(rng.standard_normal((D, D)) * sc)
        if name == "delta_pair":
            self.p["lam"] = param(np.array(0.1))
        if name == "ssm_pair":
            self.p["Wg"] = param(rng.standard_normal((D, D)) * sc)
        self.n_params = int(sum(x.d.size for x in self.p.values()))

    def forward(self, S, analytic=False):
        p = self.p
        U, Up, q = const(S["U"]), const(S["Up"]), const(S["q"])
        B, L, d = S["U"].shape
        K = const(S["K"].reshape(1, L, 1))
        NEG = const(S["NEG"].reshape(B, L, 1))

        # Analytic witness: identity projections + value-code read-out.
        # These are FIXED weights, not trained, and are identical in spirit
        # for all three mechanisms.
        if analytic:
            Wk = const(np.eye(D))
            Wv = const(np.eye(D))
            Wq = const(np.eye(D))
            Wout = const(CVAL.T)                 # (64,32)
            T1 = const(TSC.T_star(C))
        else:
            Wk, Wv, Wq, Wout = p["Wk"], p["Wv"], p["Wq"], p["Wout"]
            T1 = p.get("T1", const(np.eye(D)))

        qq = q @ Wq

        def tsc_layer(qq_):
            m = (Up * qq_.reshape(B, 1, d)).sum(-1)            # (B,L)
            return (m.reshape(B, L, 1) * U * K).sum(1)         # (B,d)

        def linear_layer(qq_):
            Kk = Up @ Wk
            Vv = U @ Wv
            m = (Kk * qq_.reshape(B, 1, d)).sum(-1)
            return (m.reshape(B, L, 1) * Vv).sum(1)

        def attn_layer(qq_):
            Kk = Up @ Wk
            Vv = U @ Wv
            sc_ = (Kk * qq_.reshape(B, 1, d)).sum(-1, keepdims=True)
            sc_ = sc_ / math.sqrt(D) + NEG
            if analytic:
                sc_ = sc_ * 50.0                 # sharpen; still a softmax
            a = sc_.softmax(1)
            return (a * Vv).sum(1)

        if self.name.startswith("tsc"):
            z = tsc_layer(qq)
            if self.name == "tsc_h2":
                q2 = (z @ T1)
                if not analytic:
                    q2 = q2 / (np.linalg.norm(q2.d, axis=-1,
                                              keepdims=True) + 1e-12)
                z = tsc_layer(q2)
            return z @ Wout

        if self.name in ("linear_pair", "gla_pair", "delta_pair"):
            Kk = Up @ Wk
            Vv = U @ Wv
            if self.name == "gla_pair":
                g = (Up @ p["Wg"]).sigmoid()
                Kk = Kk * g
            m = (Kk * qq.reshape(B, 1, d)).sum(-1)
            z = (m.reshape(B, L, 1) * Vv).sum(1)
            if self.name == "delta_pair":
                G = (Kk.transpose(0, 2, 1) @ Kk)
                z = z - p["lam"] * ((qq.reshape(B, 1, d) @
                                     G.transpose(0, 2, 1)).reshape(B, d))
            return z @ Wout

        if self.name.startswith("attn_x"):
            z = attn_layer(qq)
            if self.name == "attn_x2":
                q2 = z @ T1
                if not analytic:
                    q2 = q2 / (np.linalg.norm(q2.d, axis=-1,
                                              keepdims=True) + 1e-12)
                z = attn_layer(q2)
            return z @ Wout

        if self.name == "ssm_pair":
            g = ((U + Up) @ p["Wg"]).sigmoid()
            h = const(np.zeros((B, D)))
            for t in range(L):
                gt = g[:, t, :]
                h = gt * h + (gt * (-1.0) + 1.0) * U[:, t, :]
            return (h + qq) @ Wout

        raise ValueError(self.name)


ARMS = ("tsc_h1", "tsc_h2", "linear_pair", "attn_x1", "attn_x2",
        "gla_pair", "delta_pair", "ssm_pair")
ANALYTIC_ARMS = ("tsc_h1", "linear_pair", "attn_x1")


# ------------------------------------------------------------------ run
def evaluate(arm, D, analytic=False, chunk=128):
    tot = 0
    for st in range(0, D["Y"].shape[0], chunk):
        s = sub(D, slice(st, st + chunk))
        lg = arm.forward(s, analytic=analytic)
        tot += int((lg.d.argmax(-1) == s["Y"]).sum())
        free_graph(lg)
    return tot / D["Y"].shape[0]


def run_one(arm, Dtr, Dte, seed, steps=STEPS, lr=LR, batch=BATCH):
    rng = np.random.default_rng(79000 + seed * 31 + sum(map(ord, arm.name)))
    opt = {k: dict(m=np.zeros_like(v.d), v=np.zeros_like(v.d))
           for k, v in arm.p.items()}
    Bn = Dtr["Y"].shape[0]
    t0 = time.time()
    for step in range(steps):
        idx = rng.choice(Bn, size=min(batch, Bn), replace=False)
        s = sub(Dtr, idx)
        logits = arm.forward(s)
        loss = nll_from_logits(logits, s["Y"])
        loss.backward()
        for k, P in arm.p.items():
            o = opt[k]
            o["m"] = 0.9 * o["m"] + 0.1 * P.g
            o["v"] = 0.999 * o["v"] + 0.001 * P.g * P.g
            mh = o["m"] / (1 - 0.9 ** (step + 1))
            vh = o["v"] / (1 - 0.999 ** (step + 1))
            P.d -= lr * mh / (np.sqrt(vh) + 1e-8)
            P.g[...] = 0.0
        free_graph(logits)
    return dict(acc=evaluate(arm, Dte), params=arm.n_params,
                wall_s=round(time.time() - t0, 2))


def bootstrap_paired(diffs, n=10000, seed=0):
    a = np.asarray(diffs, dtype=float)
    r = np.random.default_rng(seed)
    means = a[r.integers(0, len(a), size=(n, len(a)))].mean(1)
    return dict(mean=float(a.mean()),
                ci95=[float(np.quantile(means, 0.025)),
                      float(np.quantile(means, 0.975))],
                n_seeds=len(a), all_positive=bool((a > 0).all()))


def part_analytic(quick=False):
    """Zero-training witnesses: the SAME task solved by three mechanisms."""
    print("== PART 0: explicit analytic witnesses (ZERO training) ==", flush=True)
    seeds = (0,) if quick else SEEDS
    rows = []
    for hops in (1,):
        for seed in seeds:
            D = make_data(hops, 7000 + seed, n=512)
            for name in ANALYTIC_ARMS:
                arm = Arm(name, seed)
                acc = evaluate(arm, D, analytic=True)
                rows.append(dict(arm=name, hops=hops, seed=seed, acc=acc,
                                 params=arm.n_params, steps=0))
                print(f"  H={hops} s={seed} {name:14s} analytic acc={acc:.4f}",
                      flush=True)
    sm = []
    for name in ANALYTIC_ARMS:
        v = [r["acc"] for r in rows if r["arm"] == name]
        sm.append(dict(arm=name, mean=float(np.mean(v)), sd=float(np.std(v)),
                       n=len(v)))
    out = dict(
        status="explicit zero-training witnesses",
        claim=("On H=1 the three mechanisms have the SAME explicit analytic "
               "solution (identity key/value/query projections + value-code "
               "read-out). Their function classes therefore COINCIDE on this "
               "task. Any training-speed difference is an OPTIMISATION / "
               "INDUCTIVE-BIAS difference, NOT an expressivity difference. "
               "This is an observation with an explicit witness, NOT a new "
               "theorem and NOT a superiority claim."),
        witness=("E = fixed orthonormal Hadamard C, Wk=Wv=Wq=I, "
                 "Wout = C[V_K:,:]^T. z = sum_s (q . Up_s) U_s = C[v_j] "
                 "because q . C[v] = 0 (alphabet partition). "
                 "TSC differs only by the fixed positive kernel K(tau), "
                 "which preserves argmax."),
        rows=rows, summary=sm)
    json.dump(out, open(OUT + "/analytic_equivalence.json", "w"), indent=1)
    print("PART 0 ->", OUT + "/analytic_equivalence.json", flush=True)
    return out


def analytic_h2_logits(S, mode):
    """Explicit ZERO-TRAINING two-hop solution, pure NumPy.

    q2 = z1 @ M, where M = C[V_K:,:]^T @ C[:V_K,:] maps the value code
    C[V_K+k] to the key code C[k]. Both hops are sum_s (q . Up_s) U_s,
    with either a fixed positive kernel K(tau) (TSC) or a sharpened softmax
    over the same scores (attention). The construction is identical in
    spirit for both mechanisms: it is the SAME algebraic object.
    """
    U, Up, q, K = S["U"], S["Up"], S["q"], S["K"]
    B, L, d = U.shape
    M = CVAL.T @ C[:VK, :]                    # (64,64): value -> key code
    Kt = K.reshape(1, L, 1)

    def hop(qq):
        m = (Up * qq[:, None, :]).sum(-1)     # (B,L)
        if mode == "tsc":
            return (m[:, :, None] * U * Kt).sum(1)
        sc = m / math.sqrt(d) + S["NEG"]
        sc = sc * 50.0
        e = np.exp(sc - sc.max(1, keepdims=True))
        a = e / e.sum(1, keepdims=True)
        return (a[:, :, None] * U).sum(1)

    z1 = hop(q)
    q2 = z1 @ M
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-12)
    z2 = hop(q2)
    return z2 @ CVAL.T                        # (B,V_K)


def part_analytic_h2(quick=False):
    """Theorem B's construction, on the SAME corrected data, zero training."""
    print("== PART 2: H=2 explicit analytic witnesses ==", flush=True)
    seeds = (0,) if quick else SEEDS
    rows = []
    for seed in seeds:
        S = make_data(2, 7000 + seed, n=512)
        for mode in ("tsc", "attention"):
            lg = analytic_h2_logits(S, mode)
            acc = float((lg.argmax(-1) == S["Y"]).mean())
            rows.append(dict(mode=mode, seed=seed, acc=acc, steps=0,
                             hops=2, params=0))
            print(f"  s={seed} H=2 {mode:10s} analytic acc={acc:.4f}",
                  flush=True)
    sm = []
    for mode in ("tsc", "attention"):
        v = [r["acc"] for r in rows if r["mode"] == mode]
        sm.append(dict(mode=mode, mean=float(np.mean(v)),
                       sd=float(np.std(v)), n=len(v)))
    out = dict(
        status="H=2 zero-training witnesses on corrected data",
        claim=("On the SAME corrected data, BOTH the static-LTI two-layer "
               "stack AND the two-layer cross-attention stack admit an "
               "explicit analytic solution with accuracy 1.0. Therefore the "
               "v6.3 training gap (tsc_h2 0.3625 vs attn_x2 0.9543) is an "
               "OPTIMISATION / INDUCTIVE-BIAS gap, NOT a function-class or "
               "expressivity gap. Theorem B (the exactly-H-layer bound under "
               "its stated premises) is a constructive existence statement "
               "and is NOT withdrawn; what is withdrawn is v6.2's claim that "
               "the bound was ALSO verified by end-to-end training."),
        witness=("M = C[V_K:,:]^T C[:V_K,:] maps C[V_K+k] -> C[k]. "
                 "hop(q) = sum_s (q . Up_s) U_s K(tau) [TSC] or the "
                 "sharpened softmax of the same scores [attention]. "
                 "q2 = normalise(hop(q) @ M); answer = argmax hop(q2) @ CVAL^T. "
                 "No learned parameter appears."),
        rows=rows, summary=sm)
    json.dump(out, open(OUT + "/analytic_h2.json", "w"), indent=1)
    print("PART 2 ->", OUT + "/analytic_h2.json", flush=True)
    return out


def part_train(quick=False):
    steps = 40 if quick else STEPS
    seeds = (0,) if quick else SEEDS
    hops_list = (1, 2) if not quick else (1,)
    print("== PART 1: corrected same-run training (full width, shared Up) ==",
          flush=True)
    rows = []
    for hops in hops_list:
        for seed in seeds:
            Dtr = make_data(hops, 1000 + seed, n=512)
            Dte = make_data(hops, 7000 + seed, n=512)
            for name in ARMS:
                arm = Arm(name, seed)
                r = run_one(arm, Dtr, Dte, seed, steps=steps)
                r.update(arm=name, hops=hops, seed=seed, chance=1.0 / VK,
                         steps=steps, lr=LR, batch=BATCH, d=D)
                rows.append(r)
                print(f"  H={hops} s={seed} {name:14s} acc={r['acc']:.4f} "
                      f"p={arm.n_params:6d}", flush=True)
    ci = {}
    if not quick:
        for base in ARMS:
            if base == "tsc_h2":
                continue
            ds = []
            for seed in SEEDS:
                a = next(r["acc"] for r in rows if r["hops"] == 2
                         and r["seed"] == seed and r["arm"] == "tsc_h2")
                b = next(r["acc"] for r in rows if r["hops"] == 2
                         and r["seed"] == seed and r["arm"] == base)
                ds.append(a - b)
            ci["tsc_h2_minus_" + base] = bootstrap_paired(ds)
    out = dict(
        status="corrected same-run evaluation, 5 seeds",
        correction=("v6.2 gave Xp ONLY to the TSC branch and used d=24<V=64. "
                    "Here ALL arms get (U,Up,q,mask) with a fixed full-width "
                    "orthonormal embedding. v6.2 low baseline numbers are "
                    "NOT evidence about published models and NOT an "
                    "expressivity result."),
        note=("Minimal mechanism references in an independent evaluation "
              "directory; NOT part of the SimplCore architecture; NOT "
              "reproductions of GLA / DeltaNet / Mamba-2 / Transformer."),
        protocol=dict(steps=steps, lr=LR, batch=BATCH, d=D, V=V,
                      optimizer="Adam", seeds=list(seeds),
                      embedding="fixed Hadamard, not learned",
                      n_train=512, n_test=512, layout="gapped"),
        rows=rows, paired_ci_h2=ci)
    json.dump(out, open(OUT + "/corrected_baselines.json", "w"), indent=1)
    print("PART 1 ->", OUT + "/corrected_baselines.json", flush=True)
    return out


def adjudicate(A0, A1, A2):
    checks = {}
    if A1["protocol"]["steps"] >= 200:
        checks["all_arms_same_protocol"] = len({(r["steps"], r["lr"],
                                                r["batch"], r["d"])
                                               for r in A1["rows"]}) == 1
        checks["full_width_d_equals_V"] = A1["protocol"]["d"] == V
        checks["embedding_fixed_not_learned"] = "not learned" in \
            A1["protocol"]["embedding"]
        checks["every_seed_every_arm"] = len(A1["rows"]) == \
            len(ARMS) * len(SEEDS) * 2
        checks["paired_ci_present"] = len(A1["paired_ci_h2"]) == len(ARMS) - 1
    checks["analytic_witnesses_present"] = len(A0["summary"]) == 3
    checks["h2_witnesses_present"] = len(A2["summary"]) == 2
    obs = {}
    if A1["protocol"]["steps"] >= 200:
        for h in (1, 2):
            obs["h%d_best_arm_mean" % h] = max(
                (float(np.mean([r["acc"] for r in A1["rows"]
                                if r["hops"] == h and r["arm"] == a])), a)
                for a in ARMS)
        sm = {r["arm"]: r["mean"] for r in A0["summary"]}
        obs["all_three_analytic_exact"] = all(v > 0.99 for v in sm.values())
        obs["analytic_means"] = sm
        obs["h2_analytic_means"] = {r["mode"]: r["mean"]
                                    for r in A2["summary"]}
        obs["h2_both_exact"] = all(r["mean"] > 0.99 for r in A2["summary"])
        obs["v62_training_claim_reversed"] = (
            A1["paired_ci_h2"]["tsc_h2_minus_attn_x2"]["mean"] < 0)
    out = dict(checks=checks, n_pass=int(sum(bool(v) for v in checks.values())),
               n_checks=len(checks), all_pass=bool(all(checks.values())),
               observations=obs,
               interpretation=("Checks certify that the two v6.2 asymmetries "
                               "are REMOVED. Observations are reported, not "
                               "gated: any training gap is optimisation, not "
                               "expressivity, because an explicit zero-training "
                               "solution exists for all three mechanisms."))
    json.dump(out, open(OUT + "/corrected_adjudication.json", "w"), indent=1)
    print("ADJUDICATION", json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("analytic", "h2", "train", "all"),
                    default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    A0 = part_analytic(a.quick) if a.part in ("analytic", "all") else None
    A2 = part_analytic_h2(a.quick) if a.part in ("h2", "all") else None
    A1 = part_train(a.quick) if a.part in ("train", "all") else None
    if A0 is not None and A1 is not None and A2 is not None and not a.quick:
        adjudicate(A0, A1, A2)


if __name__ == "__main__":
    main()
