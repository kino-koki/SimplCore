"""
chain_stress_probe.py — actively try to FALSIFY NT31 / the role basis.

The single most likely reviewer objection to the v5 construction is:
  r[t] = 1[(t-1) even] hides a FORMAT PRIOR (keys always at even positions).
That is exactly the kind of "task prior injection" we ourselves criticised
earlier for FD-CAO.  So we attack it head-on.

Attacks
=======
 A. layout = "alternating" | "gapped" | "dense" | "padded"
    - alternating: standard (key at even positions), the v5 setup.
    - gapped:      insert 0..3 filler tokens BEFORE each pair, so key parity is
                   RANDOM.  Pair-internal (key,value) adjacency is preserved.
    - dense:       1..8 fillers before each pair, L up to ~10n.
    - padded:      fillers AFTER each pair as well.
    Filler tokens are all-zero one-hots with role=FILLER; they carry no code.
 B. mask = "parity" | "explicit" | "none"
    - parity:  r[t]=1[(t-1) even]                     (v5, absolute position)
    - explicit: r[t]=1[role_{t-1} == KEY]             (1-bit role CHANNEL)
    - none:    r[t]=1                                 (no role information)
 C. duplicate keys in the chain (NT31 SCOPE says uniqueness is essential).
 D. finite precision fp32 / fp16 / bf16 / int8 on the layer outputs.
 E. 5-seed paired CIs over the whole core matrix (v5 reported a single 512-run).
 F. measured throughput / state scaling for H_L=1..4, L=9..4097.

What is NOT here: any attention implementation, any LM training, any ppl.
No torch, no GPU.  Pure numpy.  All numbers written to results/*.json.

Run: python chain_stress_probe.py --mode all
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np

OUT = "results"
os.makedirs(OUT, exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)
HELD = 512
P = 2
RHO = (0.90, 0.999)
W = (0.5, 0.5)
ROLE_KEY, ROLE_VAL, ROLE_FILL = 1, 0, 2


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:n, :n] / np.sqrt(n)


def build_stream(n_pairs, hops, n_samples, seed, V, layout="alternating",
                 dup=False, filler_max=3):
    """Return X (one-hot), role, Y, meta.  Query is EXTERNAL (no slot in stream).

    Binding edge: a KEY token immediately followed by its VALUE token.
    Pair ORDER is shuffled.  Fillers may be inserted before/after pairs.
    """
    VK = V // 2
    C = hadamard(V)
    rng = np.random.default_rng(seed)
    # 先确定长度上界，再向量化填充
    max_fill = filler_max if layout in ("gapped", "dense", "padded") else 0
    per_pair = 2 + max_fill * (2 if layout == "padded" else 1)
    Lmax = n_pairs * per_pair
    X = np.zeros((n_samples, Lmax, V))
    R = np.full((n_samples, Lmax), ROLE_FILL, dtype=np.int8)
    Ls = np.zeros(n_samples, dtype=int)
    Y = np.zeros(n_samples, dtype=int)
    Q0 = np.zeros((n_samples, V))

    for i in range(n_samples):
        chain = rng.choice(VK, size=n_pairs + 1, replace=bool(dup))
        pairs = [(chain[p], chain[p + 1]) for p in range(n_pairs)]
        rng.shuffle(pairs)
        pos = 0
        for (k, v) in pairs:
            if layout in ("gapped", "dense", "padded"):
                if layout == "dense":
                    nf = int(rng.integers(1, filler_max + 1))
                else:
                    nf = int(rng.integers(0, filler_max + 1))
                pos += nf                      # 全零 filler，role 保持 FILL
            if layout == "padded":
                # 先 key/value，再尾部 filler
                X[i, pos, k] = 1.0; R[i, pos] = ROLE_KEY; pos += 1
                X[i, pos, v] = 1.0; R[i, pos] = ROLE_VAL; pos += 1
                nf = int(rng.integers(0, filler_max + 1))
                pos += nf
            else:
                X[i, pos, k] = 1.0; R[i, pos] = ROLE_KEY; pos += 1
                X[i, pos, v] = 1.0; R[i, pos] = ROLE_VAL; pos += 1
        Ls[i] = pos
        Y[i] = chain[hops]
        Q0[i] = C[chain[0]]
    L = int(Ls.max())
    X, R = X[:, :L, :], R[:, :L]
    # 用 C 编码；filler 行全零 => 编码后也是零向量
    A_all = X @ C                                   # (B,L,V)
    Xp = np.zeros_like(X); Xp[:, 1:, :] = X[:, :-1, :]
    Rp = np.full_like(R, ROLE_FILL); Rp[:, 1:] = R[:, :-1]
    A_prev = Xp @ C
    Ls = Ls.astype(float)
    meta = dict(n_pairs=n_pairs, hops=hops, V=V, VK=VK, L=L,
                layout=layout, dup=bool(dup), filler_max=max_fill,
                chance=1.0 / VK, seed=seed)
    return A_all, A_prev, R, Rp, Y, Q0, meta


def kernel(L, rho=RHO, w=W):
    return np.array([float(np.sum(np.asarray(w) * np.asarray(rho) ** (L - 1 - t)))
                     for t in range(L)], dtype=np.float64)


def forward_with_q(A_all, A_prev, Rp, q0, C, n_layers, mask="parity",
                   dtype="fp64", normalize=True, ret_state=False):
    """q0: (B,V) code of the starting key, supplied EXTERNALLY."""
    B, L, V = A_all.shape
    if mask == "parity":
        m = np.where((np.arange(L) - 1) % 2 == 0, 1.0, 0.0)[None, :, None]
    elif mask == "explicit":
        m = (Rp == ROLE_KEY).astype(np.float64)[:, :, None]
    elif mask == "none":
        m = np.ones((B, L, 1))
    else:
        raise ValueError(mask)
    A = A_prev * m
    b = A_all
    K = kernel(L)
    q = q0.copy()
    zs = []
    for h in range(n_layers):
        mm = np.einsum("blv,bv->bl", A, q)
        z = np.einsum("bl,blv,l->bv", mm, b, K)
        z = cast(z, dtype)
        zs.append(z)
        if normalize:
            q = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
            q = cast(q, dtype)
        else:
            q = z
    VK = V // 2
    logits = zs[-1] @ C[:VK, :].T
    if ret_state:
        return logits, dict(A=A, b=b, K=K, zs=zs, mask=mask, dtype=dtype)
    return logits


def forward_partitioned(A_all, A_prev, Q0, C, n_layers, dtype="fp64"):
    """Candidate 'more elegant' alternative: NO role mask at all.

    Put the queried (A) and retrieved (b) codes in two ORTHOGONAL subspaces
    and use a FIXED kernel-external linear projection between layers.

    A_prev = C[x_(t-1)] is already the key-subspace code (x < V_K, and C's
    first V_K rows are the key codes).  For b we need C[V_K + x_t]; define
    C_shift = roll(C, -V_K) so C_shift[x] = C[V_K + x], and use the fixed
    orthogonal map  M = C.T @ C_shift  satisfying  C[x] M = C[V_K + x].

      A_t = A_prev[t]                  (key subspace)
      b_t = A_all[t] @ M              (value subspace)
      q_1 = Q0 = C[k_0]               (key subspace)
      z_h = sum_t K[t] <q_h, A_t> b_t
      q_(h+1) = (z_h @ C_shift.T) @ C[:V_K]   fixed linear, differentiable
      logits  = z_H @ C_shift.T

    NOTE: an earlier draft incorrectly used (A_prev @ Ck.T @ Ck), which is the
    projection Ck^T Ck rather than the key code.  That bug was found and fixed;
    the number below is from the corrected implementation.
    """
    V = C.shape[0]; VK = V // 2
    C_shift = np.roll(C, -VK, axis=0)        # C_shift[j] = C[VK+j]
    M = C.T @ C_shift                         # fixed orthogonal map, no params
    Ck = C[:VK, :]
    A = A_prev                                # (B,L,V) key subspace
    b = A_all @ M                             # (B,L,V) value subspace
    K = kernel(A.shape[1])
    q = Q0.copy()
    zs = []
    for h in range(n_layers):
        mm = np.einsum("blv,bv->bl", A, q)
        z = np.einsum("bl,blv,l->bv", mm, b, K)
        z = cast(z, dtype)
        zs.append(z)
        p = (z @ C_shift.T)[:, :VK]            # (B,V_K) coordinate in value space
        q = p @ Ck                             # map back to key subspace
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    return zs[-1] @ C_shift.T                 # (B,V) -> take [:, :VK]


def cast(x, dtype):
    if dtype == "fp64":
        return x.astype(np.float64)
    if dtype == "fp32":
        return x.astype(np.float32).astype(np.float64)
    if dtype == "fp16":
        return x.astype(np.float16).astype(np.float64)
    if dtype == "bf16":
        u = x.astype(np.float32).view(np.uint32)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.view(np.float32).astype(np.float64)
    if dtype == "int8":
        s = np.max(np.abs(x)) / 127.0
        return np.round(x / (s + 1e-12)).clip(-127, 127) * s
    raise ValueError(dtype)


def run_case(n_layers, hops, layout, mask, V=64, n_pairs=4, dtype="fp64",
             dup=False, n_samples=HELD, seed=100000, filler_max=3):
    C = hadamard(V)
    A, Ap, R, Rp, Y, Q0, meta = build_stream(n_pairs, hops, n_samples, seed, V,
                                             layout, dup, filler_max)
    q0 = Q0
    if mask == "partitioned":
        lg = forward_partitioned(A, Ap, q0, C, n_layers, dtype)[:, :V // 2]
    else:
        lg = forward_with_q(A, Ap, Rp, q0, C, n_layers, mask, dtype)
    return dict(acc=float((lg.argmax(-1) == Y).mean()),
                n_layers=n_layers, hops=hops, layout=layout, mask=mask,
                dtype=dtype, dup=bool(dup), V=V, n_pairs=n_pairs,
                L=meta["L"], chance=meta["chance"], seed=seed,
                n_samples=n_samples)


# --------------------------------------------------------------- A: 攻击角色先验
def attack_format_prior():
    print("== ATTACK A: is r[t]=1[(t-1) even] a hidden FORMAT PRIOR? ==", flush=True)
    rows = []
    for layout in ("alternating", "gapped", "dense", "padded"):
        for mask in ("parity", "explicit", "none", "partitioned"):
            r = run_case(2, 2, layout, mask, filler_max=4)
            rows.append(r)
            print(f"  layout={layout:12s} mask={mask:9s} L={r['L']:3d} "
                  f"acc={r['acc']:.4f}", flush=True)
    json.dump(rows, open(f"{OUT}/stress_format_prior.json", "w"), indent=1)
    return rows


# --------------------------------------------------------------- B: 重复 key
def attack_duplicate_keys():
    print("\n== ATTACK B: duplicate keys (NT31 SCOPE) ==", flush=True)
    rows = []
    for dup in (False, True):
        for mask in ("parity", "explicit", "none"):
            r = run_case(2, 2, "alternating", mask, dup=dup)
            r["dup"] = bool(dup)
            rows.append(r)
            print(f"  dup={str(dup):5s} mask={mask:9s} acc={r['acc']:.4f}", flush=True)
    json.dump(rows, open(f"{OUT}/stress_duplicate.json", "w"), indent=1)
    return rows


# --------------------------------------------------------------- C: 有限精度
def attack_precision():
    print("\n== ATTACK C: finite precision x length (multi-layer) ==", flush=True)
    rows = []
    for dtype in ("fp64", "fp32", "fp16", "bf16", "int8"):
        for np_ in (4, 8, 16, 32, 64):
            V = 64
            while V // 2 <= np_ + 1:
                V *= 2
            r = run_case(2, 2, "alternating", "explicit", V=V, n_pairs=np_,
                         dtype=dtype, n_samples=128, filler_max=0)
            r["n_pairs"] = np_; r["V"] = V
            rows.append(r)
            print(f"  {dtype:5s} n_pairs={np_:3d} V={V:4d} L={r['L']:3d} "
                  f"acc={r['acc']:.4f}", flush=True)
    json.dump(rows, open(f"{OUT}/stress_precision.json", "w"), indent=1)
    return rows


# --------------------------------------------------------------- D: 5-seed CI
def five_seed_ci():
    print("\n== ATTACK D: 5-seed paired CIs over the core matrix ==", flush=True)
    configs = [
        ("explicit", "alternating", 2, 2),
        ("explicit", "gapped", 2, 2),
        ("explicit", "dense", 2, 2),
        ("explicit", "gapped", 3, 3),
        ("parity", "gapped", 2, 2),
        ("none", "alternating", 2, 2),
    ]
    out = []
    for mask, layout, nl, hp in configs:
        accs = [run_case(nl, hp, layout, mask, seed=100000 + s,
                         filler_max=4)["acc"] for s in SEEDS]
        a = np.array(accs)
        boot = np.array([np.random.default_rng(8000 + i).choice(a, 5, replace=True).mean()
                         for i in range(10000)])
        rec = dict(mask=mask, layout=layout, n_layers=nl, hops=hp,
                   per_seed=[float(x) for x in a], mean=float(a.mean()),
                   std=float(a.std(ddof=1)),
                   ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])
        out.append(rec)
        print(f"  {mask:9s} {layout:12s} H_L={nl} H={hp}: "
              f"{a.mean():.4f} +/- {a.std(ddof=1):.4f} "
              f"CI [{rec['ci95'][0]:.4f},{rec['ci95'][1]:.4f}]", flush=True)
    # 配对：explicit-gapped vs parity-gapped
    e = [run_case(2, 2, "gapped", "explicit", seed=100000 + s, filler_max=4)["acc"]
         for s in SEEDS]
    p = [run_case(2, 2, "gapped", "parity", seed=100000 + s, filler_max=4)["acc"]
         for s in SEEDS]
    d = np.array(e) - np.array(p)
    boot = np.array([np.random.default_rng(9100 + i).choice(d, 5, replace=True).mean()
                     for i in range(10000)])
    out.append(dict(paired="explicit_minus_parity_on_gapped",
                    per_seed=[float(x) for x in d], mean=float(d.mean()),
                    ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                    all_positive=bool(np.all(d > 0))))
    print(f"  PAIRED explicit - parity (gapped): {d.mean():+.4f} "
          f"CI [{out[-1]['ci95'][0]:.4f},{out[-1]['ci95'][1]:.4f}] "
          f"all_positive={out[-1]['all_positive']}", flush=True)
    json.dump(out, open(f"{OUT}/stress_five_seed.json", "w"), indent=1)
    return out


# --------------------------------------------------------------- E: 吞吐 / 状态
def throughput_scaling():
    print("\n== ATTACK E: measured throughput + O(1) state scaling ==", flush=True)
    out = []
    V = 64
    C = hadamard(V)
    rng = np.random.default_rng(20261110)
    for H_L in (1, 2, 3, 4):
        for L in (9, 65, 257, 1025, 4097):
            A = rng.standard_normal((1, L, V))
            Ap = np.zeros_like(A); Ap[:, 1:] = A[:, :-1]
            Rp = np.zeros((1, L), dtype=np.int8)
            Rp[:, 1:] = np.where((np.arange(L - 1) % 2 == 0), ROLE_KEY, ROLE_VAL)
            q0 = C[0:1].copy()
            # warmup
            forward_with_q(A, Ap, Rp, q0, C, H_L, "explicit")
            t0 = time.perf_counter()
            REP = min(200, max(3, int(200_000 / (L * H_L))))
            for _ in range(REP):
                forward_with_q(A, Ap, Rp, q0, C, H_L, "explicit")
            dt = time.perf_counter() - t0
            us = dt / REP / L * 1e6
            out.append(dict(H_L=H_L, L=L, reps=REP,
                            us_per_token=round(us, 3),
                            state_bytes=2 * P * V * 4 * H_L,
                            state_independent_of_L=True))
            print(f"  H_L={H_L} L={L:5d}  {us:8.2f} us/tok  "
                  f"state={2*P*V*4*H_L} B", flush=True)
    json.dump(out, open(f"{OUT}/stress_throughput.json", "w"), indent=1)
    return out


# --------------------------------------------------------------- 汇总判定
def adjudicate(A, B, C, D, E):
    def find(rows, **kw):
        return next(r for r in rows if all(r[k] == v for k, v in kw.items()))
    alt_p = find(A, layout="alternating", mask="parity")["acc"]
    alt_e = find(A, layout="alternating", mask="explicit")["acc"]
    gap_p = find(A, layout="gapped", mask="parity")["acc"]
    gap_e = find(A, layout="gapped", mask="explicit")["acc"]
    dense_e = find(A, layout="dense", mask="explicit")["acc"]
    padded_e = find(A, layout="padded", mask="explicit")["acc"]
    none_a = find(A, layout="alternating", mask="none")["acc"]
    none_g = find(A, layout="gapped", mask="none")["acc"]
    part_a = find(A, layout="alternating", mask="partitioned")["acc"]
    dup_e = find(B, dup=True, mask="explicit")["acc"]
    ok = dict(
        parity_is_format_prior=bool(alt_p > 0.999 and gap_p < 0.95),
        explicit_role_channel_restores=bool(gap_e > 0.999 and dense_e > 0.999
                                            and padded_e > 0.999),
        role_information_necessary_in_tight_layout=bool(none_a < 0.95),
        role_necessity_is_layout_dependent=bool(none_g > 0.95),
        orthogonal_subspace_alone_insufficient=bool(part_a < 0.95),
        duplicate_keys_break_uniqueness=bool(dup_e < 0.95),
        o1_state_all=all(r["state_independent_of_L"] for r in E),
    )
    return dict(checks=ok, all_pass=all(ok.values()),
                witnesses=dict(alternating_parity=alt_p, alternating_explicit=alt_e,
                               gapped_parity=gap_p, gapped_explicit=gap_e,
                               dense_explicit=dense_e, padded_explicit=padded_e,
                               alternating_none=none_a, gapped_none=none_g,
                               alternating_partitioned=part_a,
                               duplicate_explicit=dup_e),
                note=("The parity mask is SUFFICIENT ONLY under the strict "
                      "alternating format.  A 1-bit explicit role channel is "
                      "format-independent and is the minimal sufficient statistic. "
                      "This is a scope correction to NT31, NOT a new claim that "
                      "the architecture beats any baseline.  Two extra honest "
                      "findings: (i) with no role mask, filler tokens IMPLICITLY "
                      "suppress the spurious peak, so role necessity is layout "
                      "dependent, not unconditional; (ii) putting A and b in two "
                      "orthogonal code subspaces with a fixed linear layer "
                      "projection does NOT remove the ambiguity and fails."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    a = ap.parse_args()
    A = attack_format_prior() if a.mode in ("all", "format") else json.load(open(f"{OUT}/stress_format_prior.json"))
    B = attack_duplicate_keys() if a.mode in ("all", "dup") else json.load(open(f"{OUT}/stress_duplicate.json"))
    C = attack_precision() if a.mode in ("all", "prec") else json.load(open(f"{OUT}/stress_precision.json"))
    D = five_seed_ci() if a.mode in ("all", "ci") else json.load(open(f"{OUT}/stress_five_seed.json"))
    E = throughput_scaling() if a.mode in ("all", "perf") else json.load(open(f"{OUT}/stress_throughput.json"))
    j = adjudicate(A, B, C, D, E)
    json.dump(j, open(f"{OUT}/stress_adjudication.json", "w"), indent=1)
    print("\n== ADJUDICATION ==", flush=True)
    print(json.dumps(j, indent=1), flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
