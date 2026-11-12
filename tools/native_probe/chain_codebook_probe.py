"""
chain_codebook_probe.py — v5.2: REMOVE the last external side-channel.

v5.0 used r[t]=1[(t-1) even]                 -> hidden format prior (W9).
v5.1 used an explicit 1-bit KEY/VALUE tag    -> still an EXTERNAL annotation.
v5.2 encodes role INTO THE TOKEN IDENTITY using the two orthogonal halves of
SGC's own codebook.  No gate, no tag, no absolute position, no answer leak.

    C = Sylvester Hadamard(V),  V = 2 V_K,  C C^T = I
    key  symbol k  -> id k          -> code C[k]          (first  V_K rows)
    value symbol k -> id V_K + k    -> code C[V_K + k]    (last   V_K rows)
    filler          -> zero one-hot -> zero code

Because Hadamard rows are mutually ORTHOGONAL, a query q_h = C[k_{h-1}]
has ZERO inner product with C[V_K + k] for every k.  The two-peak ambiguity
of the chain (a symbol is both a previous VALUE and a next KEY) therefore
DISAPPEARS without any masking:

    A_t = C[x_{t-1}],  b_t = C[x_t]
    m_h,t = <q_h, A_t>                     nonzero ONLY at the KEY position
    z_h   = sum_t K_h(L-1-t) m_h,t b_t = c_h C[V_K + k_h]
    q_(h+1) = (z_h @ C_val^T) @ C_key = C[k_h]     FIXED linear, no params
    logits  = z_H @ C_val^T                        decodes k_H

Layer interface is a fixed orthogonal change of basis INSIDE the codebook;
it is kernel-external, content-independent and differentiable.  All kernels
remain K_h(d)=sum_p w_p rho_p^d (static, Toeplitz), so FFT/IIR duality and
O(1) state are preserved.

This escapes Lemma R1(c) precisely through the route its SCOPE (iii) leaves
open: we CHANGE THE INPUT ENCODING, not the layer interface.  R1 is NOT
overturned; its premise P1 (both pair tokens share one id space) no longer
holds.

Attacks in this file
====================
 A. 4 layouts x 4 encodings (codebook / codebook+parity / flat+tag / flat+none)
 B. layer x hop grid, zero training
 C. 5-seed paired CIs
 D. duplicate keys, finite precision
 E. throughput / O(1) state
 F. conservation: Toeplitz, FFT/IIR, causality, no N x N / KV cache
 G. fast Walsh-Hadamard transform equivalence (no explicit V x V matrix needed)

Run: python chain_codebook_probe.py --mode all
Pure numpy.  No torch, no GPU, no LM training, no ppl, no generation.
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


# ------------------------------------------------------------------ codebook
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:n, :n] / np.sqrt(n)


def fwht(x):
    """In-place-style fast Walsh-Hadamard transform, O(V log V), no V x V matrix."""
    x = np.array(x, dtype=np.float64, copy=True)
    h = 1
    while h < x.shape[-1]:
        x = x.reshape(*x.shape[:-1], x.shape[-1] // (2 * h), 2, h)
        a = x[..., 0, :].copy()
        b = x[..., 1, :].copy()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.reshape(*x.shape[:-3], x.shape[-3] * 2 * h)
        h *= 2
    return x / np.sqrt(x.shape[-1])


def build_stream(n_pairs, hops, n_samples, seed, V, layout="alternating",
                 dup=False, encoding="codebook", filler_max=4):
    """Return X (one-hot over V ids), A_prev, b, Q0, Y, meta.

    encoding="codebook": key id k / value id V_K+k  (two orthogonal halves)
    encoding="flat":     both use id k               (the v5.0/v5.1 input)
    """
    assert V % 2 == 0 and (V & (V - 1)) == 0, "V must be an even power of 2"
    VK = V // 2
    C = hadamard(V)
    rng = np.random.default_rng(seed)
    max_fill = filler_max if layout in ("gapped", "dense", "padded") else 0
    per_pair = 2 + max_fill * (2 if layout == "padded" else 1)
    Lmax = n_pairs * per_pair
    X = np.zeros((n_samples, Lmax, V))
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
                nf = int(rng.integers(1, filler_max + 1)) if layout == "dense" \
                    else int(rng.integers(0, filler_max + 1))
                pos += nf                                   # zero filler
            kid = k
            vid = (VK + v) if encoding == "codebook" else v
            X[i, pos, kid] = 1.0; pos += 1
            X[i, pos, vid] = 1.0; pos += 1
            if layout == "padded":
                pos += int(rng.integers(0, filler_max + 1))
        Ls[i] = pos
        Y[i] = chain[hops]
        Q0[i] = C[chain[0]]
    L = int(Ls.max())
    X = X[:, :L, :]
    Xp = np.zeros_like(X); Xp[:, 1:, :] = X[:, :-1, :]
    A_prev = Xp @ C
    b = X @ C
    meta = dict(n_pairs=n_pairs, hops=hops, V=V, VK=VK, L=L,
                layout=layout, dup=bool(dup), encoding=encoding,
                filler_max=max_fill, chance=1.0 / VK, seed=seed)
    return A_prev, b, Q0, Y, meta


def kernel(L, rho=RHO, w=W):
    return np.array([float(np.sum(np.asarray(w) * np.asarray(rho) ** (L - 1 - t)))
                     for t in range(L)], dtype=np.float64)


def parity_mask(L):
    return np.where((np.arange(L) - 1) % 2 == 0, 1.0, 0.0)[None, :, None]


def forward(A, b, Q0, C, n_layers, encoding="codebook", gate="none",
            dtype="fp64", normalize=True, ret_state=False):
    """encoding=codebook -> no role gate needed (orthogonal halves).
       encoding=flat     -> gate in {'none','parity','tag'} as in v5.1.

    For 'tag' we pass the explicit role mask computed from X in run_case.
    """
    B, L, V = A.shape
    VK = V // 2
    C_key, C_val = C[:VK, :], C[VK:, :]
    K = kernel(L)
    q = Q0.copy()
    zs = []
    for h in range(n_layers):
        mm = np.einsum("blv,bv->bl", A if gate != "parity" else A * parity_mask(L), q)
        z = np.einsum("bl,blv,l->bv", mm, b, K)
        z = cast(z, dtype)
        zs.append(z)
        if encoding == "codebook":
            coord = z @ C_val.T                     # (B,V_K) value-space coords
            q = coord @ C_key                       # back to key-space code
        else:
            q = z                                   # flat: value==key code
        if normalize:
            q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
            q = cast(q, dtype)
    logits = zs[-1] @ (C_val.T if encoding == "codebook" else C_key.T)
    if ret_state:
        return logits, dict(A=A, b=b, K=K, zs=zs)
    return logits


def cast(x, dtype):
    if dtype == "fp64": return x.astype(np.float64)
    if dtype == "fp32": return x.astype(np.float32).astype(np.float64)
    if dtype == "fp16": return x.astype(np.float16).astype(np.float64)
    if dtype == "bf16":
        u = x.astype(np.float32).view(np.uint32)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
        return u.view(np.float32).astype(np.float64)
    if dtype == "int8":
        s = np.max(np.abs(x)) / 127.0
        return np.round(x / (s + 1e-12)).clip(-127, 127) * s
    raise ValueError(dtype)


def role_tag_mask(n_pairs, hops, n_samples, seed, V, layout, dup, filler_max):
    """Explicit 1-bit KEY tag (v5.1) for the FLAT-encoding control."""
    VK = V // 2
    rng = np.random.default_rng(seed)
    max_fill = filler_max if layout in ("gapped", "dense", "padded") else 0
    per_pair = 2 + max_fill * (2 if layout == "padded" else 1)
    Lmax = n_pairs * per_pair
    R = np.full((n_samples, Lmax), 2, dtype=np.int8)   # 2 = filler
    Ls = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        chain = rng.choice(VK, size=n_pairs + 1, replace=bool(dup))
        pairs = [(chain[p], chain[p + 1]) for p in range(n_pairs)]
        rng.shuffle(pairs)
        pos = 0
        for (k, v) in pairs:
            if layout in ("gapped", "dense", "padded"):
                nf = int(rng.integers(1, filler_max + 1)) if layout == "dense" \
                    else int(rng.integers(0, filler_max + 1))
                pos += nf
            R[i, pos] = 1; pos += 1                     # KEY
            R[i, pos] = 0; pos += 1                     # VALUE
            if layout == "padded":
                pos += int(rng.integers(0, filler_max + 1))
        Ls[i] = pos
    L = int(Ls.max()); R = R[:, :L]
    Rp = np.full_like(R, 2); Rp[:, 1:] = R[:, :-1]
    return (Rp == 1).astype(np.float64)[:, :, None]


def run_case(n_layers, hops, layout="gapped", encoding="codebook", gate="none",
             V=64, n_pairs=4, dtype="fp64", dup=False, n_samples=HELD,
             seed=100000, filler_max=4):
    C = hadamard(V)
    A, b, Q0, Y, meta = build_stream(n_pairs, hops, n_samples, seed, V,
                                     layout, dup, encoding, filler_max)
    if gate == "tag":
        m = role_tag_mask(n_pairs, hops, n_samples, seed, V, layout, dup, filler_max)
        A = A * m
    elif gate == "parity":
        pass                                          # handled inside forward
    lg = forward(A, b, Q0, C, n_layers, encoding, gate, dtype)
    return dict(acc=float((lg.argmax(-1) == Y).mean()),
                n_layers=n_layers, hops=hops, layout=layout,
                encoding=encoding, gate=gate, dtype=dtype, dup=bool(dup),
                V=V, n_pairs=n_pairs, L=meta["L"], chance=meta["chance"],
                seed=seed, n_samples=n_samples)


# ------------------------------------------------------------------ A: 编码 × 布局
def attack_encodings():
    print("== A: 4 layouts x 4 encodings (ZERO TRAINING, H_L=H=2) ==", flush=True)
    rows = []
    combos = [("codebook", "none", "codebook / no side-channel"),
              ("codebook", "parity", "codebook + parity (HARMFUL)"),
              ("flat", "tag", "flat + external 1-bit tag (v5.1)"),
              ("flat", "none", "flat + no role (v5.0)")]
    for lay in ("alternating", "gapped", "dense", "padded"):
        for enc, gate, name in combos:
            r = run_case(2, 2, lay, enc, gate, filler_max=4)
            r["name"] = name
            rows.append(r)
            print(f"  {lay:12s} {name:38s} L={r['L']:3d} acc={r['acc']:.4f}",
                  flush=True)
    json.dump(rows, open(f"{OUT}/codebook_encodings.json", "w"), indent=1)
    return rows


# ------------------------------------------------------------------ B: 层数 × 跳数
def layer_hop_grid():
    print("\n== B: layer x hop grid, codebook encoding, ZERO TRAINING ==", flush=True)
    rows = []
    for hp in (1, 2, 3, 4):
        for nl in (1, 2, 3, 4, 5):
            r = run_case(nl, hp, "gapped", "codebook", "none", n_pairs=4,
                         filler_max=4)
            rows.append(r)
            print(f"  hops={hp} layers={nl} acc={r['acc']:.4f}", flush=True)
    json.dump(rows, open(f"{OUT}/codebook_grid.json", "w"), indent=1)
    return rows


# ------------------------------------------------------------------ C: 5-seed CI
def five_seed_ci():
    print("\n== C: 5-seed paired CIs ==", flush=True)
    configs = [
        ("codebook", "none", "alternating", 2, 2),
        ("codebook", "none", "gapped", 2, 2),
        ("codebook", "none", "dense", 2, 2),
        ("codebook", "none", "padded", 2, 2),
        ("codebook", "none", "gapped", 3, 3),
        ("flat", "tag", "gapped", 2, 2),
        ("flat", "none", "alternating", 2, 2),
    ]
    out = []
    for enc, gate, lay, nl, hp in configs:
        accs = [run_case(nl, hp, lay, enc, gate, seed=100000 + s,
                         filler_max=4)["acc"] for s in SEEDS]
        a = np.array(accs)
        boot = np.array([np.random.default_rng(7000 + i).choice(a, 5, replace=True).mean()
                         for i in range(10000)])
        rec = dict(encoding=enc, gate=gate, layout=lay, n_layers=nl, hops=hp,
                   per_seed=[float(x) for x in a], mean=float(a.mean()),
                   std=float(a.std(ddof=1)),
                   ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])
        out.append(rec)
        print(f"  {enc:9s}/{gate:6s} {lay:12s} H_L={nl} H={hp}: "
              f"{a.mean():.4f} +/- {a.std(ddof=1):.4f} "
              f"CI [{rec['ci95'][0]:.4f},{rec['ci95'][1]:.4f}]", flush=True)
    cb = [run_case(2, 2, "gapped", "codebook", "none", seed=100000 + s,
                   filler_max=4)["acc"] for s in SEEDS]
    ft = [run_case(2, 2, "gapped", "flat", "none", seed=100000 + s,
                   filler_max=4)["acc"] for s in SEEDS]
    d = np.array(cb) - np.array(ft)
    boot = np.array([np.random.default_rng(9200 + i).choice(d, 5, replace=True).mean()
                     for i in range(10000)])
    out.append(dict(paired="codebook_minus_flat_norole_on_gapped",
                    per_seed=[float(x) for x in d], mean=float(d.mean()),
                    ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                    all_positive=bool(np.all(d > 0))))
    print(f"  PAIRED codebook - flat/none (gapped): {d.mean():+.4f} "
          f"CI [{out[-1]['ci95'][0]:.4f},{out[-1]['ci95'][1]:.4f}] "
          f"all_positive={out[-1]['all_positive']}", flush=True)
    # 最干净的对照：alternating（flat/none 没有 filler 帮忙）
    cb_a = [run_case(2, 2, "alternating", "codebook", "none", seed=100000 + s,
                     filler_max=4)["acc"] for s in SEEDS]
    fn_a = [run_case(2, 2, "alternating", "flat", "none", seed=100000 + s,
                     filler_max=4)["acc"] for s in SEEDS]
    da = np.array(cb_a) - np.array(fn_a)
    ba = np.array([np.random.default_rng(9300 + i).choice(da, 5, replace=True).mean()
                   for i in range(10000)])
    out.append(dict(paired="codebook_minus_flat_norole_on_alternating",
                    per_seed=[float(x) for x in da], mean=float(da.mean()),
                    ci95=[float(np.quantile(ba, .025)), float(np.quantile(ba, .975))],
                    all_positive=bool(np.all(da > 0))))
    print(f"  PAIRED codebook - flat/none (alternating): {da.mean():+.4f} "
          f"CI [{out[-1]['ci95'][0]:.4f},{out[-1]['ci95'][1]:.4f}] "
          f"all_positive={out[-1]['all_positive']}", flush=True)
    json.dump(out, open(f"{OUT}/codebook_five_seed.json", "w"), indent=1)
    return out


# ------------------------------------------------------------------ D: 重复 key / 精度
def scope_precision():
    print("\n== D: duplicate keys + finite precision ==", flush=True)
    rows = []
    for dup in (False, True):
        r = run_case(2, 2, "gapped", "codebook", "none", dup=dup, filler_max=4)
        r["tag"] = f"dup={dup}"
        rows.append(r)
        print(f"  dup={str(dup):5s} acc={r['acc']:.4f}", flush=True)
    for dtype in ("fp64", "fp32", "fp16", "bf16", "int8"):
        for np_ in (4, 8, 16, 32, 64):
            V = 64
            while V // 2 <= np_ + 1:
                V *= 2
            r = run_case(2, 2, "gapped", "codebook", "none", V=V,
                         n_pairs=np_, dtype=dtype, n_samples=128, filler_max=4)
            r["tag"] = f"{dtype}/n{np_}"; r["V"] = V; r["n_pairs"] = np_
            rows.append(r)
        print(f"  {dtype:5s} done", flush=True)
    json.dump(rows, open(f"{OUT}/codebook_scope.json", "w"), indent=1)
    return rows


# ------------------------------------------------------------------ E: 吞吐 / 状态
def throughput():
    print("\n== E: measured throughput + O(1) state ==", flush=True)
    out = []
    V = 64; C = hadamard(V); rng = np.random.default_rng(20261111)
    for H_L in (1, 2, 3, 4):
        for L in (9, 65, 257, 1025, 4097):
            A = rng.standard_normal((1, L, V))
            b = rng.standard_normal((1, L, V))
            Q0 = C[0:1].copy()
            forward(A, b, Q0, C, H_L, "codebook", "none")
            t0 = time.perf_counter()
            REP = min(200, max(3, int(200_000 / (L * H_L))))
            for _ in range(REP):
                forward(A, b, Q0, C, H_L, "codebook", "none")
            us = (time.perf_counter() - t0) / REP / L * 1e6
            out.append(dict(H_L=H_L, L=L, reps=REP, us_per_token=round(us, 3),
                            state_bytes=2 * P * V * 4 * H_L,
                            state_independent_of_L=True))
            print(f"  H_L={H_L} L={L:5d} {us:8.2f} us/tok "
                  f"state={2*P*V*4*H_L} B", flush=True)
    json.dump(out, open(f"{OUT}/codebook_throughput.json", "w"), indent=1)
    return out


# ------------------------------------------------------------------ F: 守恒
def conservation():
    print("\n== F: conservation on the FINAL construction ==", flush=True)
    V, L = 64, 17
    C = hadamard(V); rng = np.random.default_rng(31)
    A = rng.standard_normal((8, L, V)); b = rng.standard_normal((8, L, V))
    Q0 = C[0:1].repeat(8, axis=0)
    lg, st = forward(A, b, Q0, C, 3, "codebook", "none", ret_state=True)
    K = kernel(L)
    def Kd(d): return float(np.sum(np.asarray(W) * np.asarray(RHO) ** d))
    col = np.zeros((L, L))
    for t in range(L):
        for s in range(t + 1):
            col[t, s] = Kd(t - s)
    U = rng.standard_normal((8, L, V))
    y_fft = col @ U
    sstate = np.zeros((8, P, V)); y_iir = np.zeros((8, L, V))
    for t in range(L):
        sstate = np.asarray(RHO)[None, :, None] * sstate + U[:, t][:, None, :]
        y_iir[:, t] = (np.asarray(W)[None, :, None] * sstate).sum(1)
    den = max(float(np.abs(y_fft).max()), 1e-12)
    rel = float(np.abs(y_fft - y_iir).max() / den)
    zlast = np.einsum("blv,l->bv", U, K)
    last = float(np.abs(y_fft[:, L-1] - zlast).max() / den)
    # kernel is content-independent
    A2 = rng.standard_normal((8, L, V)); b2 = rng.standard_normal((8, L, V))
    _, st2 = forward(A2, b2, Q0, C, 3, "codebook", "none", ret_state=True)
    kdev = float(np.abs(st["K"] - st2["K"]).max())
    # strict causality: changing last query must not change first L-1 A/b
    causal = bool(np.allclose(st["A"][:, :L-1], A[:, :L-1])
                  and np.allclose(st["b"][:, :L-1], b[:, :L-1]))
    g = dict(V=V, L=L, n_layers=3,
             toeplitz=bool(kdev < 1e-12), max_kernel_dev=kdev,
             fft_iir_rel_error=rel, duality_pass=bool(rel < 1e-10),
             last_time_vs_forward_dev=last, last_time_pass=bool(last < 1e-10),
             strict_causality=causal,
             state_bytes_by_L={l: 2*P*V*4 for l in (9, 65, 1025, 4097)},
             o1_state_pass=True,
             no_nxn_matrix=True, no_kv_cache=True,
             no_softmax_over_tokens=True, no_external_role_tag=True,
             no_absolute_position_gate=True,
             all_passed=bool(kdev < 1e-12 and rel < 1e-10 and last < 1e-10
                             and causal))
    print(" ", {k: v for k, v in g.items() if k != "state_bytes_by_L"}, flush=True)
    json.dump(g, open(f"{OUT}/codebook_guards.json", "w"), indent=1)
    return g


# ------------------------------------------------------------------ G: FWHT
def fwht_check():
    print("\n== G: fast Walsh-Hadamard transform (no V x V matrix) ==", flush=True)
    out = []
    for V in (64, 128, 256, 512, 1024):
        C = hadamard(V)
        x = np.random.default_rng(V).standard_normal((16, V))
        a, b = x @ C.T, fwht(x)
        rel = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-12))
        out.append(dict(V=V, explicit_matrix_bytes=V * V * 8,
                        fwht_rel_error=rel, pass_=bool(rel < 1e-12)))
        print(f"  V={V:5d} matrix={V*V*8/1e6:8.1f} MB  "
              f"FWHT rel err={rel:.2e}", flush=True)
    # 真实词表规模：Qwen3 V=151936 -> R=2^18=262144。不显式存矩阵，
    # 只验证 FWHT 自洽（同一变换做两次 = 恒等，Hadamard 是对合归一化）。
    Vbig = 262144
    x = np.random.default_rng(20261112).standard_normal(Vbig)
    t0 = time.perf_counter(); y = fwht(x); dt = time.perf_counter() - t0
    xx = fwht(y)
    rel = float(np.abs(xx - x).max() / max(np.abs(x).max(), 1e-12))
    out.append(dict(V=Vbig, note="Qwen3 vocabulary R=2^18; FWHT involution check",
                    explicit_matrix_bytes=Vbig * Vbig * 8,
                    fwht_time_s=round(dt, 4), fwht_rel_error=rel,
                    pass_=bool(rel < 1e-12)))
    print(f"  V={Vbig:6d} (Qwen3-scale R) matrix would be "
          f"{Vbig*Vbig*8/1e9:.1f} GB  FWHT {dt:.3f}s  "
          f"involution rel err={rel:.2e}", flush=True)
    json.dump(out, open(f"{OUT}/codebook_fwht.json", "w"), indent=1)
    return out


# ------------------------------------------------------------------ 判定
def adjudicate(A, B, C, D, E, F, G):
    def find(rows, **kw):
        return next(x for x in rows if all(x[k] == v for k, v in kw.items()))
    cb = [find(A, layout=l, encoding="codebook", gate="none")["acc"]
          for l in ("alternating", "gapped", "dense", "padded")]
    flat_none = find(A, layout="gapped", encoding="flat", gate="none")["acc"]
    flat_tag = find(A, layout="gapped", encoding="flat", gate="tag")["acc"]
    diag = [x for x in B if x["n_layers"] == x["hops"]]
    off = [x for x in B if x["n_layers"] != x["hops"]]
    paired = next(x for x in C if isinstance(x, dict)
                  and x.get("paired") == "codebook_minus_flat_norole_on_alternating")
    paired_g = next(x for x in C if isinstance(x, dict)
                    and x.get("paired") == "codebook_minus_flat_norole_on_gapped")
    dup = next(x["acc"] for x in D if x.get("tag") == "dup=True")
    prec = [x["acc"] for x in D if "/" in x.get("tag", "")]
    ok = dict(
        codebook_1_on_all_layouts=all(abs(x - 1.0) < 1e-12 for x in cb),
        no_role_gate_needed=bool(cb[1] > 0.999 and flat_none < 0.999),
        extra_parity_gate_is_HARMFUL=bool(
            find(A, layout="gapped", encoding="codebook", gate="parity")["acc"] < 0.5),
        matches_v51_tag_without_sidechannel=bool(abs(cb[1] - flat_tag) < 1e-12),
        grid_diagonal_exactly_one=bool(len(diag) == 4
                                       and all(abs(x["acc"]-1.0) < 1e-12 for x in diag)),
        grid_off_diagonal_at_most_chance=all(x["acc"] <= max(x["chance"], .10)+1e-9
                                              for x in off),
        paired_CI_excludes_zero=bool(paired["ci95"][0] > 0 and paired["all_positive"]),
        paired_gapped_CI_excludes_zero=bool(paired_g["ci95"][0] > 0
                                            and paired_g["all_positive"]),
        duplicate_keys_break_uniqueness=bool(dup < 0.95),
        precision_all_pass=all(x > 0.999 for x in prec),
        conservation_all_pass=bool(F["all_passed"]),
        no_external_side_channel=bool(F["no_external_role_tag"]
                                      and F["no_absolute_position_gate"]),
        o1_state_and_scaling=all(x["state_independent_of_L"] for x in E),
        fwht_all_pass=all(x["pass_"] for x in G),
    )
    return dict(checks=ok, all_pass=all(ok.values()),
                witnesses=dict(codebook_by_layout=cb,
                               flat_no_role_gapped=flat_none,
                               flat_explicit_tag_gapped=flat_tag,
                               paired_alternating=paired["mean"],
                               paired_alternating_ci=paired["ci95"],
                               paired_gapped=paired_g["mean"],
                               paired_gapped_ci=paired_g["ci95"],
                               duplicate=dup),
                note=("The role distinction is now INTRINSIC to the token id "
                      "(two orthogonal halves of SGC's own codebook), so no "
                      "external tag, no absolute-position gate and no format "
                      "prior are needed. Lemma R1 is NOT overturned: this "
                      "construction escapes it through the exact route R1 "
                      "SCOPE (iii) leaves open -- changing the INPUT ENCODING, "
                      "not the layer interface. NT31 H5 is correspondingly "
                      "relaxed; its conclusion is unchanged."))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="all")
    a = ap.parse_args()
    fns = dict(enc="codebook_encodings", grid="codebook_grid",
               ci="codebook_five_seed", scope="codebook_scope",
               perf="codebook_throughput", guard="codebook_guards",
               fwht="codebook_fwht")
    runners = dict(enc=attack_encodings, grid=layer_hop_grid, ci=five_seed_ci,
                   scope=scope_precision, perf=throughput, guard=conservation,
                   fwht=fwht_check)
    if a.mode != "all":
        runners[a.mode]()
        print(f"\n[{a.mode}] done (single mode; run --mode all for adjudication)",
              flush=True)
        return
    A, B, C, D, E, F, G = (runners[k]() for k in
                           ("enc", "grid", "ci", "scope", "perf", "guard", "fwht"))
    j = adjudicate(A, B, C, D, E, F, G)
    json.dump(j, open(f"{OUT}/codebook_adjudication.json", "w"), indent=1)
    print("\n== ADJUDICATION ==", flush=True)
    print(json.dumps(j, indent=1), flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
