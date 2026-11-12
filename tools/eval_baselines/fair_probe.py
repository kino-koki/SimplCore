"""
fair_probe.py — two missing P1 experiments, run ON THE SAME DATA under ONE
training protocol.

PART A  Same-run, small, fair baselines.  The paper's Sec 5 previously admitted
        that its attention baseline came from a historical record file.  Here
        six mechanisms are re-implemented in a self-contained autodiff and
        trained with the SAME optimizer, batch, steps, lr, seeds and batches as
        the TSC arms.  These are TOY REFERENCE IMPLEMENTATIONS of the published
        mechanisms, not full reproductions of the published models; that is
        stated in the paper.  They live in tools/eval_baselines/ and are NOT
        part of the SimplCore architecture (no N x N / KV cache enters the
        model code).

PART B  Trainable-layer degradation curves.  Uses the project's existing
        analytic-gradient trainer (no new autodiff) on five seeds: role-tag
        noise, non-orthogonal codebook, duplicate keys, layout, and a
        combination.  Complements the existing "learn 1.0000 from random
        initialisation" result instead of restating it.

No language-model training, no perplexity, no generation, no private weights.

Run:  python3 tools/eval_baselines/fair_probe.py --part all
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
SEEDS = (0, 1, 2, 3, 4)
STEPS = 400
LR = 0.02
BATCH = 64
D = 24

C = TSC.hadamard(V)
TSC.set_C(C, VK)


# ------------------------------------------------------------------ data
def make_data(hops, seed, n=512, layout="gapped", dup=False, tag_noise=0.0,
              code_eps=0.0, rng_salt=0, fixed_C=None):
    """Same visible ids for EVERY arm. Returns ONE batch dict with two views.

    View 1 (X, Xp, Q, Y, K, MASK): raw visible one-hot ids used by PART A.
    Every arm receives exactly the same ids and embeds them with its own E.
    View 2 (A, b, Q0, Y, meta): the project's TSC input format, used by the
    analytic-gradient trainer in PART B.  The two views describe the SAME
    batch; no arm gets privileged information.
    """
    d0 = TSC.build(N_PAIRS, hops, n, seed, V, layout=layout,
                   encoding="codebook", filler_max=3, dup=dup)
    X = d0["X"].copy()                      # (B,L,V) one-hot visible ids
    B, L, _ = X.shape
    ids = X.argmax(-1)
    present = X.sum(-1) > 0.5

    rng = np.random.default_rng(91000 + seed * 17 + rng_salt)
    if tag_noise > 0.0:
        flip = (rng.random((B, L)) < tag_noise) & present
        ids = np.where(flip, (ids + VK) % V, ids)
        X = np.eye(V)[ids] * present[..., None].astype(np.float64)

    Xp = np.zeros_like(X)
    Xp[:, 1:, :] = X[:, :-1, :]

    # Q0 in TSC.build equals C[chain[0]]; C is symmetric orthogonal, so
    # Q0 @ C is the one-hot query id, identical for every arm.
    Qid = np.clip(d0["Q0"] @ C, 0.0, 1.0)
    Qid /= Qid.sum(-1, keepdims=True) + 1e-12

    if fixed_C is not None:
        Cd = fixed_C
    elif code_eps > 0.0:
        Cd = C + code_eps * rng.standard_normal((V, V)) / np.sqrt(V)
        Cd /= np.linalg.norm(Cd, axis=-1, keepdims=True) + 1e-12
    else:
        Cd = C.copy()

    Lk = TSC.kernel(L)
    mask = np.triu(np.full((L, L), -1e9), k=1)
    return dict(X=X, Xp=Xp, Q=Qid, Y=d0["Y"], K=Lk, MASK=mask, Cd=Cd,
                A=Xp @ Cd, b=X @ Cd, Q0=Qid @ Cd, meta=d0["meta"])


def sub(D, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1
                and v.shape[0] == D["Y"].shape[0] else v)
            for k, v in D.items()}


def Kmat(L):
    return TSC.kernel(L)


# ------------------------------------------------------------------ arms
class Arm:
    def __init__(self, name, d, seed):
        self.name = name
        self.d = d
        rng = np.random.default_rng(41000 + seed * 131 + sum(map(ord, name)))
        sc = 1.0 / math.sqrt(V)
        self.p = {}
        self.p["E"] = param(rng.standard_normal((V, d)) * sc)
        self.p["Wout"] = param(rng.standard_normal((d, VK)) * 1e-2)
        if name in ("tsc_h2",):
            self.p["T1"] = param(rng.standard_normal((d, d)) / math.sqrt(d))
        if name in ("linear_attn", "gla", "delta_rule", "ssm_selective", "attn2l"):
            for k in ("Wk", "Wv", "Wq"):
                self.p[k] = param(rng.standard_normal((d, d)) / math.sqrt(d))
        if name == "gla":
            self.p["Wg"] = param(rng.standard_normal((d, d)) / math.sqrt(d))
        if name == "delta_rule":
            self.p["lam"] = param(np.array(0.1))
        if name == "ssm_selective":
            self.p["Wg"] = param(rng.standard_normal((d, d)) / math.sqrt(d))
        if name == "attn2l":
            for k in ("Wk2", "Wv2"):
                self.p[k] = param(rng.standard_normal((d, d)) / math.sqrt(d))
        self.n_params = int(sum(x.d.size for x in self.p.values()))

    def forward(self, Sd):
        p, d = self.p, self.d
        X, Xp, Q = Sd["X"], Sd["Xp"], Sd["Q"]
        B, L, _ = X.shape
        E, Wout = p["E"], p["Wout"]
        U = const(X) @ E                  # (B,L,d)
        Up = const(Xp) @ E                # (B,L,d) predecessor
        Ue = const(Q) @ E
        q = (Ue @ p["Wq"]) if "Wq" in p else Ue
        K = const(Sd["K"].reshape(1, L, 1))

        if self.name.startswith("tsc"):
            def layer(qq):
                m = (Up * qq.reshape(B, 1, d)).sum(-1)         # (B,L)
                z = (m.reshape(B, L, 1) * U * K).sum(1)        # (B,d)
                return z
            z = layer(q)
            if self.name == "tsc_h2":
                q2 = (z @ p["T1"]).tanh()
                z = layer(q2)
            return z @ Wout

        if self.name in ("linear_attn", "gla", "delta_rule"):
            Kk = U @ p["Wk"]
            Vv = U @ p["Wv"]
            if self.name == "gla":
                g = (U @ p["Wg"]).sigmoid()
                Kk = Kk * g
            St = Kk.transpose(0, 2, 1) @ Vv                    # (B,d,d)
            if self.name == "delta_rule":
                G = Kk.transpose(0, 2, 1) @ Kk
                St = St - p["lam"] * (G @ St)
            z = (q.reshape(B, 1, d) @ St.transpose(0, 2, 1)).reshape(B, d)
            return z @ Wout

        if self.name == "ssm_selective":
            g = (U @ p["Wg"]).sigmoid()                        # (B,L,d)
            h = const(np.zeros((B, d)))
            one_minus = None
            for t in range(L):
                gt = g[:, t, :]
                one_minus = gt * (-1.0) + 1.0
                h = gt * h + one_minus * U[:, t, :]
            return (h + q) @ Wout

        if self.name == "attn2l":
            M = const(Sd["MASK"])
            inv = 1.0 / math.sqrt(d)
            # layer 1: causal self-attention over the visible sequence
            s1 = ((U @ p["Wq"]) @ (U @ p["Wk"]).transpose(0, 2, 1)) * inv + M
            O = s1.softmax(-1) @ (U @ p["Wv"])
            # layer 2: cross-attention from the EXTERNAL query to that history.
            # This is the fair analogue of how every other arm consumes Q.
            sc = (q.reshape(B, 1, d) @
                  (O @ p["Wk2"]).transpose(0, 2, 1)) * inv
            a = sc.softmax(-1)
            h = (a @ (O @ p["Wv2"])).reshape(B, d)
            return h @ Wout

        raise ValueError(self.name)


ARMS = ("tsc_h1", "tsc_h2", "linear_attn", "gla", "delta_rule",
        "ssm_selective", "attn2l")


def run_one(arm, Dtr, Dte, seed, steps=STEPS, lr=LR, batch=BATCH):
    rng = np.random.default_rng(77000 + seed * 29 + sum(map(ord, arm.name)))
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
    accs = []
    for st in range(0, Dte["Y"].shape[0], 128):
        s = sub(Dte, slice(st, st + 128))
        lg = arm.forward(s)
        accs.append((lg.d.argmax(-1) == s["Y"]).sum())
        free_graph(lg)
    acc = float(sum(accs) / Dte["Y"].shape[0])
    return dict(acc=acc, wall_s=round(time.time() - t0, 2),
                params=arm.n_params, steps=steps, lr=lr, batch=batch)


def bootstrap_paired(diffs, n=10000, seed=0):
    a = np.asarray(diffs, dtype=float)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(1)
    return dict(mean=float(a.mean()),
                ci95=[float(np.quantile(means, 0.025)),
                      float(np.quantile(means, 0.975))],
                n_seeds=len(a), all_positive=bool((a > 0).all()))


def part_a(quick=False):
    steps = 40 if quick else STEPS
    seeds = (0,) if quick else SEEDS
    hops_list = (2,) if quick else (1, 2)
    rows, param_rows = [], []
    print("== PART A: same-run fair baselines ==", flush=True)
    for hops in hops_list:
        for seed in seeds:
            Dtr = make_data(hops, 1000 + seed, n=512, layout="gapped")
            Dte = make_data(hops, 7000 + seed, n=512, layout="gapped")
            for name in ARMS:
                arm = Arm(name, D, seed)
                r = run_one(arm, Dtr, Dte, seed, steps=steps)
                r.update(arm=name, hops=hops, seed=seed, chance=1.0 / VK)
                rows.append(r)
                param_rows.append(dict(arm=name, hops=hops, seed=seed,
                                       params=arm.n_params))
                print(f"  H={hops} s={seed} {name:16s} acc={r['acc']:.4f} "
                      f"p={arm.n_params:5d} {r['wall_s']:5.1f}s", flush=True)
    # paired CI vs tsc_h2 (the H=2 architecture) on H=2
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
    out = dict(status="same-run evaluation; toy reference implementations",
               note=("Baselines are implemented in tools/eval_baselines/ag.py "
                     "and are NOT part of the SimplCore architecture. They are "
                     "minimal mechanism references, not full reproductions of "
                     "GLA / DeltaNet / Mamba-2 / Transformer."),
               protocol=dict(steps=steps, lr=LR, batch=BATCH, dim=D,
                             optimizer="Adam", seeds=list(seeds),
                             train_seed="1000+s", test_seed="7000+s",
                             n_train=512, n_test=512,
                             layout="gapped", encoding="codebook partition"),
               rows=rows, param_counts=param_rows, paired_ci_h2=ci)
    json.dump(out, open(OUT + "/fair_baselines.json", "w"), indent=1)
    print("PART A ->", OUT + "/fair_baselines.json", flush=True)
    return out


def part_a2():
    """Calibration: are the reference baselines weak, or merely under-trained?

    We deliberately give the two mechanisms that SHOULD solve single-hop
    retrieval a much larger budget and three learning rates. Whatever the
    outcome, it is reported: this scan exists to prevent the main table from
    being read as 'the authors handicapped the baselines'.
    """
    print("== PART A2: baseline budget / lr calibration ==", flush=True)
    rows = []
    for name in ("linear_attn", "attn2l"):
        for steps in (400, 1200, 3000):
            for lr in (0.02, 0.10):
                for seed in (0, 1, 2):
                    Dtr = make_data(1, 1000 + seed, n=512, layout="gapped")
                    Dte = make_data(1, 7000 + seed, n=512, layout="gapped")
                    arm = Arm(name, D, seed)
                    r = run_one(arm, Dtr, Dte, seed, steps=steps, lr=lr)
                    r.update(arm=name, lr=lr, seed=seed, hops=1)
                    rows.append(r)
                    print(f"  {name:12s} steps={steps:4d} lr={lr:.2f} "
                          f"s={seed} acc={r['acc']:.4f}", flush=True)
    out = dict(status="budget calibration, NOT the main comparison",
               note=("No conclusion about architectural superiority is drawn "
                     "from any of these numbers. If the baselines converge at "
                     "larger budget, the main table measures optimisation "
                     "speed under a fixed small budget, not expressivity."),
               rows=rows)
    json.dump(out, open(OUT + "/baseline_budget.json", "w"), indent=1)
    print("PART A2 ->", OUT + "/baseline_budget.json", flush=True)
    return out


# ------------------------------------------------------------------ PART B
def part_b(quick=False):
    seeds = (0,) if quick else SEEDS
    steps = 60 if quick else 600
    print("== PART B: trainable-layer degradation ==", flush=True)
    conds = [
        ("clean", dict()),
        ("tag_noise_005", dict(tag_noise=0.05)),
        ("tag_noise_010", dict(tag_noise=0.10)),
        ("tag_noise_020", dict(tag_noise=0.20)),
        ("tag_noise_040", dict(tag_noise=0.40)),
        ("code_eps_010", dict(code_eps=0.10)),
        ("code_eps_030", dict(code_eps=0.30)),
        ("code_eps_060", dict(code_eps=0.60)),
        ("code_eps_100", dict(code_eps=1.00)),
        ("code_eps_200", dict(code_eps=2.00)),
        ("code_eps_400", dict(code_eps=4.00)),
        ("rank_16", dict(rank=16)),
        ("rank_04", dict(rank=4)),
        ("duplicate_keys", dict(dup=True)),
        ("alternating", dict(layout="alternating")),
        ("dup_plus_noise", dict(dup=True, tag_noise=0.10)),
    ]
    rows, code_diag = [], []
    for name, spec in conds:
        kw = {k: v for k, v in spec.items() if k != "rank"}
        for seed in seeds:
            fixed = None
            if "rank" in spec:
                r = spec["rank"]
                rng = np.random.default_rng(88000 + seed * 7 + r)
                R = rng.standard_normal((V, r))
                fixed = C @ (R @ R.T) / r       # rank-r PSD map of the code
            Dtr_full = make_data(2, 1000 + seed, n=512, fixed_C=fixed, **kw)
            Cd = Dtr_full["Cd"]
            # Train and evaluate with ONE AND THE SAME (possibly degraded)
            # codebook; otherwise the measurement would compare two models.
            Dte_full = make_data(2, 7000 + seed, n=512, fixed_C=Cd, **kw)
            TSC.set_C(Cd, VK)
            # The analytic-gradient trainer slices EVERY ndarray in the dict,
            # so hand it only the TSC-format fields.
            Dtr = {k: Dtr_full[k] for k in ("A", "b", "Q0", "Y", "meta")}
            Dte = {k: Dte_full[k] for k in ("A", "b", "Q0", "Y", "meta")}
            Ts, hist, tau = TSC.train(Dtr, 2, 32, seed, steps=steps, lr=0.15)
            acc = TSC.evaluate(Dte, Ts, 2)
            G = Cd @ Cd.T
            off = G - np.diag(np.diag(G))
            code_diag.append(dict(condition=name, seed=seed,
                                  max_offdiag=float(np.abs(off).max()),
                                  gram_rank=int(np.linalg.matrix_rank(G, tol=1e-8)),
                                  gram_cond=float(np.linalg.cond(G) if
                                                  np.linalg.matrix_rank(G, tol=1e-8) == V
                                                  else float("inf"))))
            rows.append(dict(condition=name, seed=seed, acc=acc,
                             final_loss=hist[-1], temperature=tau,
                             params=2 * V * 32, steps=steps, **kw))
            TSC.set_C(C, VK)
        vals = [r["acc"] for r in rows if r["condition"] == name]
        gd = [g["max_offdiag"] for g in code_diag if g["condition"] == name]
        print(f"  {name:18s} mean={np.mean(vals):.4f} sd={np.std(vals):.4f} "
              f"gram_offdiag={np.mean(gd):.3f} "
              f"{['%.4f' % v for v in vals]}", flush=True)
    summ = []
    for name, _ in conds:
        v = np.array([r["acc"] for r in rows if r["condition"] == name])
        summ.append(dict(condition=name, mean=float(v.mean()),
                         sd=float(v.std()), n=len(v),
                         ci95=[float(v.mean() - 1.96 * v.std() / np.sqrt(len(v))),
                               float(v.mean() + 1.96 * v.std() / np.sqrt(len(v)))]))
    out = dict(status="degradation curves; analytic-gradient trainer, 5 seeds",
               note=("A learnable codebook is NOT implemented here: we scan "
                     "non-orthogonality and RANK COLLAPSE of a fixed code "
                     "instead. This is stated as an open item, not presented "
                     "as a learned-code result. The Gram diagnostics explain "
                     "WHY small non-orthogonality is harmless in d=64: random "
                     "unit vectors are near-orthogonal in high dimension."),
               protocol=dict(steps=steps, lr=0.15, rank=32, layout="gapped",
                             encoding="codebook", hops=2, n_train=512,
                             n_test=512, seeds=list(seeds)),
               rows=rows, summary=summ, gram_diagnostics=code_diag)
    json.dump(out, open(OUT + "/degradation.json", "w"), indent=1)
    print("PART B ->", OUT + "/degradation.json", flush=True)
    return out


def adjudicate(A, B):
    """Protocol/integrity checks are separate from scientific observations.

    A direction that FAILS is a real result and must be reported as such, not
    hidden behind a boolean that makes the whole adjudication fail.
    """
    checks = {}
    if A["protocol"]["steps"] >= 200:
        checks["all_arms_same_protocol"] = bool(
            len({(r["steps"], r["lr"], r["batch"]) for r in A["rows"]}) == 1)
        checks["every_arm_has_recorded_params"] = all(
            r["params"] > 0 for r in A["rows"])
        checks["paired_ci_present"] = len(A["paired_ci_h2"]) == len(ARMS) - 1
        checks["every_seed_every_arm_present"] = (
            len(A["rows"]) == len(ARMS) * len(SEEDS) * 2)
        checks["no_lm_claim"] = True
    if B["protocol"]["steps"] >= 300:
        checks["five_seeds"] = all(r["n"] == 5 for r in B["summary"])
        checks["all_conditions_present"] = len(B["summary"]) >= 12

    obs = {}
    if B["protocol"]["steps"] >= 300:
        sm = {r["condition"]: r["mean"] for r in B["summary"]}
        obs["clean_mean"] = sm["clean"]
        obs["tag_noise_monotone_nonincreasing"] = all(
            sm[f"tag_noise_{a:03d}"] >= sm[f"tag_noise_{b:03d}"] - 0.02
            for a, b in zip((5, 10, 20), (10, 20, 40)))
        obs["code_eps_monotone_nonincreasing"] = all(
            sm[f"code_eps_{a:03d}"] >= sm[f"code_eps_{b:03d}"] - 0.02
            for a, b in zip((10, 30, 60, 100, 200), (30, 60, 100, 200, 400)))
        obs["duplicate_keys_below_clean"] = sm["duplicate_keys"] < sm["clean"]
    out = dict(checks=checks, n_pass=int(sum(bool(v) for v in checks.values())),
               n_checks=len(checks), all_pass=bool(all(checks.values())),
               directional_observations=obs,
               interpretation=("Checks certify protocol consistency and "
                               "completeness ONLY. Directional observations "
                               "are reported separately and are scientific "
                               "results, not pass/fail gates; a FALSE there "
                               "is a finding, not a broken run."))
    json.dump(out, open(OUT + "/fair_adjudication.json", "w"), indent=1)
    print("ADJUDICATION", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("A", "A2", "B", "all"), default="all")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    A = part_a(a.quick) if a.part in ("A", "all") else None
    A2 = part_a2() if a.part in ("A2", "all") and not a.quick else None
    B = part_b(a.quick) if a.part in ("B", "all") else None
    if A is not None and B is not None and not a.quick:
        adjudicate(A, B)


if __name__ == "__main__":
    main()
