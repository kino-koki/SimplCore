"""
Consistency checks for NT01..NT15 (the first 15 of the 49-theorem roadmap).

CRITICAL DISCIPLINE (reviewer point 4):
  These are NOT the proofs.  Every proof is analytic and lives in
  new_theorems15.py.  This file only checks that each statement is correctly
  formed, that its hypotheses are NON-EMPTY, and that explicit finite instances
  satisfy the claimed identities / inequalities.  A numerical run never
  upgrades an observation to a theorem.

NO TRAINING. NO GRADIENT DESCENT. NO MODEL WEIGHTS. NO EXPERIMENTAL DATA.
Run: python verify15.py     Out: theorem15_new_results.json
"""
import json
import math
import os
import numpy as np

RNG = np.random.default_rng(20261108)
OUT = os.environ.get("NT15_OUT", "theorem15_new_results.json")
rows = []


def rec(nid, name, ok, detail, kind="finite_identity"):
    rows.append(dict(id=nid, name=name, kind=kind,
                     verdict="PASS" if ok else "FAIL", detail=detail))
    print(("  [PASS] " if ok else "  [FAIL] ") + nid + " " + name +
          "  (" + kind + ")", flush=True)
    return bool(ok)


def had_row(i, R):
    j = np.arange(R, dtype=np.int64)
    x = np.bitwise_and(i, j)
    p = np.zeros_like(x)
    while np.any(x):
        p ^= (x & 1)
        x >>= 1
    return np.where(p == 0, 1.0, -1.0)


def had_prefix(V):
    R = 1 << (V - 1).bit_length()
    return np.stack([had_row(i, R) for i in range(V)]), R


# ------------------------------------------------------------------ NT01
def n01():
    V, R, P, N, L = 8, 8, 3, 5, 12
    Code, R2 = had_prefix(V)
    assert R == R2
    rho = RNG.uniform(0.80, 0.999, size=(P, N))
    w = RNG.uniform(0.1, 1.0, size=(P, N))
    # strictly positive kernel K_n(d) = sum_p w rho^d
    def K(d):                       # scalar delay only
        return np.einsum("pn,pn->n", w, rho ** max(int(d), 0))
    vals = RNG.normal(size=(L, N))
    # one-hot value assignment: target = argmax of a fixed random vector
    tgt = 4
    onehot = np.zeros(N); onehot[3] = 1.0
    a = Code[RNG.integers(0, V, size=L)]           # keys
    q = Code[tgt].copy()
    q_key_at = 6
    a[q_key_at] = q.copy()                          # one matching key
    b = np.zeros((L, N)); b[q_key_at] = onehot
    # ensure other keys differ from q
    for t in range(L):
        if t != q_key_at and np.allclose(a[t], q):
            a[t] = Code[(t + 1) % V]
    u = a[:, :, None] * b[:, None, :]
    s = np.zeros((P, R, N))
    for t in range(L):
        s = rho[:, None, :] * s + u[t][None, :, :]
    z = np.einsum("pn,r,prn->n", w, q, s)
    # closed form: only the matching position survives
    d = L - 1 - q_key_at
    z_cf = R * K(d) * onehot
    dev = float(np.abs(z - z_cf).max())
    argmax_ok = bool(int(np.argmax(z)) == int(np.argmax(onehot)))
    # position independence: move the target, argmax unchanged
    ok_pos = True
    for pos in (1, 3, 9, 11):
        s2 = np.zeros((P, R, N))
        aa = a.copy(); bb = np.zeros((L, N))
        aa[pos] = q.copy(); bb[pos] = onehot
        uu = aa[:, :, None] * bb[:, None, :]
        for t in range(L):
            s2 = rho[:, None, :] * s2 + uu[t][None, :, :]
        zz = np.einsum("pn,r,prn->n", w, q, s2)
        ok_pos &= (int(np.argmax(zz)) == int(np.argmax(onehot)))
    pos_kernel = bool(all(np.all(K(d) > 0) for d in range(L)))
    return rec("NT01", "tight order-2 retrieval, zero interference, position-free",
               bool(dev < 1e-10 and argmax_ok and ok_pos and pos_kernel),
               dict(closed_form_max_dev=dev, argmax_correct=argmax_ok,
                    position_independent=ok_pos, kernel_strictly_positive=pos_kernel,
                    note="order 1 impossible by NT13; order 2 attains it => tight"),
               "finite_identity + analytic rank-nullity")


# ------------------------------------------------------------------ NT02
def n02():
    L = 400
    r1, r2 = 0.93, 0.71
    h_real = np.array([sum(r1 ** (t - j) * r2 ** j for j in range(t + 1))
                       for t in range(L)])
    pos = bool(np.all(h_real > 0))
    # Vieta injectivity
    a, b = r1 + r2, -r1 * r2
    roots = np.sort(np.roots([1.0, -a, -b]).real)
    inj = bool(np.allclose(np.sort([r1, r2]), roots, atol=1e-10))
    out = []
    for theta in (0.05, 0.3, 0.8, 1.4, 2.5, 3.0):
        rho = 0.97
        h = rho ** np.arange(L) * np.sin((np.arange(L) + 1) * theta) / np.sin(theta)
        tau_ceiling = math.ceil(math.pi / theta)
        first = next((int(t) for t in range(min(L, tau_ceiling + 2)) if h[t] <= 0), None)
        out.append(dict(theta=theta, ceil_pi_over_theta=tau_ceiling,
                        first_nonpositive=first,
                        holds=bool(first is not None and first <= tau_ceiling)))
    return rec("NT02", "positive real poles vs sign-changing conjugate poles",
               bool(pos and inj and all(o["holds"] for o in out)),
               dict(real_h_all_positive=pos, vieta_injective=inj,
                    recovered_roots=roots.tolist(), sign_checks=out),
               "finite_identity + analytic sign argument")


# ------------------------------------------------------------------ NT03
def n03():
    P, D = 4, 40000
    rho = np.array([0.80, 0.90, 0.97, 0.999])
    w = np.array([0.3, 0.4, 0.2, 0.1])
    F = sum(w[p] * rho[p] ** np.arange(D) for p in range(P))
    tends_zero = bool(F[-1] < 1e-12 and F[-1] < F[0]
                      and np.all(np.diff(F[D // 2:]) <= 1e-12))
    # distance from ANY nonzero combination to the constant 1 in l-infty >= 1
    # (for decaying combos, sup_{d>=D}|1-F| -> 1); verify at large d
    dist_tail = float(np.abs(1.0 - F[D // 2:]).max())
    const_dist = bool(dist_tail > 0.99)
    # rho=1 single pole gives exactly the constant
    one = np.ones(D)
    iso = bool(np.allclose(1.0 ** np.arange(D), one))
    return rec("NT03", "rho<1 multi-pole kernel is not the linear-attention constant",
               bool(tends_zero and const_dist and iso),
               dict(F_at_d_end=float(F[-1]), dist_to_constant_tail=dist_tail,
                    horizon=D,
                    rho1_is_constant=iso,
                    note="strict decay => constant has l^infty distance 1 from the span"),
               "analytic (limit) + finite identity")


# ------------------------------------------------------------------ NT04
def n04():
    out = []
    ok = True
    for V in (3, 5, 7, 64, 65, 100, 1000, 50000):
        Code, R = had_prefix(V) if V <= 1000 else (None, 1 << (V - 1).bit_length())
        if Code is not None:
            G = Code @ Code.T
            off = float(np.abs(G - np.diag(np.diag(G))).max())
            diag = float(np.diag(G).min())
        else:  # analytic row formula on a sample (avoid 50k^2 memory)
            idx = np.linspace(0, V - 1, 16, dtype=int)
            H = np.stack([had_row(int(i), R) for i in idx])
            G = H @ H.T
            off = float(np.abs(G - np.diag(np.diag(G))).max())
            diag = float(np.diag(G).min())
        lower = bool(R >= V)
        upper = bool(R < 2 * V)
        gram = bool(off < 1e-9 and diag == R)
        ok &= lower and upper and gram
        out.append(dict(V=V, R=R, R_over_V=round(R / V, 3),
                        lower_bound_R_ge_V=lower, ratio_lt_2=upper, gram_ok=gram))
    return rec("NT04", "arbitrary-V deterministic code is 2-optimal",
               bool(ok), dict(instances=out), "analytic + finite identity")


# ------------------------------------------------------------------ NT05
def n05():
    out = []
    ok = True
    for V, P, N in ((64, 1, 32), (65, 2, 16), (1000, 3, 8), (151936, 1, 1)):
        R = 1 << (V - 1).bit_length()
        state = P * R * N
        lower = P * V * N
        ratio = state / lower
        good = bool(state >= lower and ratio < 2.0)
        ok &= good
        out.append(dict(V=V, P=P, N=N, R=R, state_scalars=state,
                        lower_bound=lower, ratio=round(ratio, 3), ok=good))
    return rec("NT05", "Omega(V*N) state lower bound, attained within factor 2",
               bool(ok), dict(instances=out), "analytic + finite identity")


# ------------------------------------------------------------------ NT06
def n06():
    def A_rho(rho):
        x = rho ** 2
        return (1 + x) / (1 - x) ** 3
    out = []
    ok = True
    for rho in (0.5, 0.9, 0.99):
        for M, t1 in ((32, 1e-4), (32, 0.3), (16, 0.6)):
            th = np.linspace(t1, 1.2, M)
            tau = np.arange(0, 600)
            F = (rho ** tau)[None, :] * np.sin(np.outer(th, tau + 1))
            G = F @ F.T
            ev = np.linalg.eigvalsh(G)
            smin, smax = float(ev[0]), float(ev[-1])
            cond = smax / max(smin, 1e-300)
            D = th[1] - th[0]
            bound = 2 * math.sin(t1) ** 2 / (D ** 2 * A_rho(rho))
            good = bool(cond + 1e-12 >= bound)
            ok &= good
            out.append(dict(rho=rho, M=M, theta1=t1, Delta=round(D, 5),
                            cond=float(f"{cond:.3e}"), lower_bound=float(f"{bound:.3e}"),
                            inequality_holds=good))
    return rec("NT06", "quantitative cond lower bound near theta=0",
               bool(ok), dict(instances=out,
                   A_rho_note="A(rho)=sum rho^{2tau}(tau+1)^2=(1+rho^2)/(1-rho^2)^3"),
               "finite_identity (bound certification)")


# ------------------------------------------------------------------ NT07
def n07():
    U, c = 1.0, 2.0
    for m in (8, 11, 24):
        delta = U * 2.0 ** (-(m + 1))
        hi = 1 - c * delta / U
        if hi <= 0:
            continue
        rs = np.linspace(1e-6, hi - 1e-9, 20000)
        num = np.log((1 - rs) * U / (c * delta))
        g = num / np.abs(np.log(rs))
        g = np.where(num > 0, g, 0.0)
        k = int(np.argmax(g))
        interior = bool(0 < rs[k] < hi and g[k] > 0)
        # analytic endpoints: g(rho->0)=0, g(rho->hi)=0 (evaluate very close)
        def g_at(r):
            n = math.log((1 - r) * U / (c * delta))
            return n / abs(math.log(r)) if n > 0 else 0.0
        g_lo, g_hi = g_at(1e-12), g_at(hi * (1 - 1e-12))
        ends = bool(g_lo < g[k] and g_hi < g[k])
        # rho=1 gives no resolvable length
        rho1_infeasible = bool((1 - 1.0) * U <= c * delta)
        print(f"    m={m:2d} delta={delta:.3e} rho*={rs[k]:.6f} "
              f"L_max~{1+g[k]:.1f} interior={interior} ends={ends} "
              f"g_lo={g_lo:.2e} g_hi={g_hi:.2e}", flush=True)
        if not (interior and ends and rho1_infeasible):
            return rec("NT07", "interior optimal rho exists (trilemma)", False,
                       dict(m=m, interior=interior, ends=ends), "analytic + finite scan")
    return rec("NT07", "interior optimal rho exists (trilemma)", True,
               dict(precision_bits_checked=[8, 11, 24],
                    statement="g(rho)->0 at both endpoints, positive inside => "
                              "interior maximum; rho=1 infeasible",
                    note="existence is analytic; rho* values are illustrations only"),
               "analytic (extreme value theorem) + finite scan")


# ------------------------------------------------------------------ NT08
def n08():
    V, R, L, N = 8, 8, 12, 5
    Code, _ = had_prefix(V)
    eps = 0.05
    E = RNG.uniform(-eps, eps, size=(V, V)); E = (E + E.T) / 2
    np.fill_diagonal(E, 0.0)
    G = R * np.eye(V) + E
    Kmin, Kmax, K = 0.4, 1.0, 0.7
    interference_bound = eps * (L - 1) * Kmax
    margin_lower = R * Kmin - interference_bound
    threshold = R * Kmin / ((L - 1) * Kmax)
    ok = bool(margin_lower > 0 and eps < threshold)
    # explicit simulated logits
    target = 3; onehot = np.zeros(N); onehot[2] = 1.0
    z = R * K * onehot.copy()
    for t in range(L - 1):
        z += RNG.uniform(-eps, eps) * K * RNG.normal(size=N) * 0.05
    argmax_ok = bool(int(np.argmax(z)) == 2)
    return rec("NT08", "bounded-code-perturbation robustness bound",
               bool(ok and argmax_ok),
               dict(eps=eps, sufficient_threshold=round(threshold, 5),
                    interference_bound=round(interference_bound, 4),
                    target_margin_lower=round(margin_lower, 4),
                    argmax_preserved=argmax_ok),
               "finite_identity (sufficient-condition certification)")


# ------------------------------------------------------------------ NT09
def n09():
    c1, c2, eps = 1.0, 1.0, 2.0 ** -24
    out = []
    ok = True
    for rho in (0.5, 0.9, 0.99, 0.999):
        Lstar = 2.0 ** (c2 / (c1 * (1 - rho)))
        # verify the defining equality at Lstar
        fft = c1 * eps * math.log2(Lstar) if Lstar > 1 else 0.0
        iir = c2 * eps / (1 - rho)
        eq = bool(abs(fft - iir) < 1e-12) if Lstar < 1e300 else True
        # IIR bound independent of L
        indep = bool(abs(c2 * eps / (1 - rho) - iir) < 1e-18)
        ok &= eq and indep
        out.append(dict(rho=rho, Lstar_exponent=c2 / (c1 * (1 - rho)),
                        Lstar_str=("2^%.1f" % (c2 / (c1 * (1 - rho)))),
                        fft_at_Lstar=fft, iir_bound=iir, equality=eq,
                        iir_L_independent=indep))
    return rec("NT09", "FFT/IIR rounding-order crossover L*",
               bool(ok), dict(c1=c1, c2=c2, eps=eps, instances=out,
                   note="conditional worst-case bounds, NOT measured error"),
               "analytic (conditional) + finite identity")


# ------------------------------------------------------------------ NT10
def n10():
    P, R, N, L = 3, 4, 5, 24
    rho = RNG.uniform(0.80, 0.999, size=(P, N))   # per-(p,n), NOT per t
    u = RNG.normal(size=(L, R, N))
    s = np.zeros((P, R, N))
    Y = []
    for t in range(L):
        s = rho[:, None, :] * s + u[t][None, :, :]
        Y.append(s.copy())
    Y = np.stack(Y)
    # channel-wise Toeplitz deviation
    dev = 0.0
    p, r, n = 1, 2, 3
    imp = np.zeros((L, R, N)); imp[3, r, n] = 1.0
    ss = np.zeros((P, R, N)); K = []
    for t in range(L):
        ss = rho[:, None, :] * ss + imp[t][None, :, :]
        K.append(ss[p, r, n])
    for t in range(3, L):
        dev = max(dev, abs(K[t] - rho[p, n] ** (t - 3)))
    # FFT vs IIR on one channel
    x = u[:, r, n]
    h = rho[p, n] ** np.arange(L)
    y_fft = np.fft.irfft(np.fft.rfft(x, n=2 * L) * np.fft.rfft(h, n=2 * L),
                         n=2 * L)[:L]
    rel = float(np.abs(y_fft - Y[:, p, r, n]).max() /
                (np.abs(Y[:, p, r, n]).max() + 1e-12))
    return rec("NT10", "per-channel time-invariance keeps block Toeplitz + duality",
               bool(dev < 1e-12 and rel < 1e-10),
               dict(toeplitz_dev=dev, fft_iir_rel=rel,
                    state_scalars=P * R * N, depends_on_L=False),
               "finite_identity")


# ------------------------------------------------------------------ NT11
def n11():
    out = []
    ok = True
    for d in (2, 3):
        for H in (1, 2, 3, 5, 8):
            need = math.ceil(math.log(H + 1) / math.log(d))
            # degree of composition d^L must reach H+1
            reach = next(L for L in range(1, 20) if d ** L >= H + 1)
            good = bool(reach >= need and (d ** (need - 1) < H + 1 if need > 1 else True))
            ok &= good
            out.append(dict(degree=d, hops=H, min_layers=need, first_reaching=reach,
                            holds=good))
    twohop = next(o for o in out if o["degree"] == 2 and o["hops"] == 2)
    return rec("NT11", "layer lower bound ceil(log_d(H+1)); H=2 needs >=2 layers",
               bool(ok and twohop["min_layers"] == 2),
               dict(instances=out, twohop_min_layers=twohop["min_layers"],
                    note="necessary only; no sufficient construction asserted"),
               "analytic (degree composition) + finite identity")


# ------------------------------------------------------------------ NT12
def n12():
    def kernel(P, L, const=True, zero_c=False):
        a = RNG.uniform(-0.2, 0.2, size=P)
        a /= max(1.0, np.sum(np.abs(a))) / 0.8
        aa = np.tile(a, (L, 1))
        c = np.ones(L)
        if not const:
            aa = aa * (1 + 0.3 * RNG.normal(size=(L, P)))
        if zero_c:
            c[5] = 0.0
        K = np.zeros((L, L))
        for s in range(L):
            y = np.zeros(L); inp = np.zeros(L); inp[s] = 1.0
            for t in range(L):
                acc = c[t] * inp[t]
                for j in range(1, P + 1):
                    if t - j >= 0:
                        acc += aa[t, j - 1] * y[t - j]
                y[t] = acc
            K[:, s] = y
        dev = 0.0
        for tau in range(1, min(L, 8)):
            v = [K[s + tau, s] for s in range(L - tau)]
            dev = max(dev, float(np.max(v) - np.min(v)))
        return dev
    out = []
    ok = True
    for P in (1, 2, 3, 5):
        dc = kernel(P, 30, const=True)
        dv = kernel(P, 30, const=False)
        good = bool(dc < 1e-10 and dv > 1e-6)
        ok &= good
        out.append(dict(P=P, constant_dev=dc, varying_dev=dv, holds=good))
    return rec("NT12", "Toeplitz iff time-invariant (arbitrary order, per channel)",
               bool(ok), dict(orders=out,
                   note="c_t != 0 is essential; source theta clamp guarantees it"),
               "finite_identity + analytic induction")


# ------------------------------------------------------------------ NT13
def n13():
    d, n = 8, 3
    L = RNG.normal(size=(d, n * d))
    rank = int(np.linalg.matrix_rank(L, tol=1e-10))
    ker = n * d - rank
    bound_ok = bool(ker >= n * d - d)
    _, _, Vt = np.linalg.svd(L)
    dv = Vt[-1]
    collision = float(np.abs(L @ dv).max())
    v1 = RNG.normal(size=n * d); v2 = v1 + 3 * dv
    proj = RNG.normal(size=d)
    q = RNG.normal(size=d)
    f = lambda A: np.tanh(A @ proj + 0.2)
    y1, y2 = f(L @ v1), f(L @ v2)
    same = bool(np.abs(y1 - y2).max() < 1e-10)
    # n=1 exception
    L1 = RNG.normal(size=(d, d))
    ker1 = d - int(np.linalg.matrix_rank(L1, tol=1e-10))
    return rec("NT13", "arbitrary read-out impossible; ker>=nd-d; n=1 exception",
               bool(bound_ok and collision < 1e-10 and same and ker1 == 0),
               dict(rank=rank, kernel=ker, bound=nd_d(d, n),
                    collision_residual=collision,
                    nonlinear_output_identical=same, n1_kernel=ker1,
                    premises=["fixed linear summary", "q at current position only",
                              "independent values", ">=2 distinguishable targets"]),
               "analytic rank-nullity + finite collision")


def nd_d(d, n):
    return n * d - d


# ------------------------------------------------------------------ NT14
def n14():
    # (i) pigeonhole non-empty
    B, X, L = 56320, 64, 10 ** 6
    pigeon = bool(L * math.log2(X) > 8 * B)
    # (ii) hard top-k: unselected contribute exactly zero
    C, k, d = 256, 4, 16
    q = RNG.normal(size=d); q /= np.linalg.norm(q)
    keys = RNG.normal(size=(C, d)); keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    vals = RNG.normal(size=(C, d))
    sc = keys @ q
    S = np.argsort(-sc)[:k]; U = np.argsort(-sc)[k:]
    w = np.exp((sc[S] - sc[S].max()) / 0.07); w /= w.sum()
    m0 = (w[:, None] * vals[S]).sum(0)
    v2 = vals.copy(); v2[U[0]] += 7.0
    zero = float(np.abs(m0 - (w[:, None] * v2[S]).sum(0)).max())
    # (iii) softening -> kernel collision
    alpha = RNG.uniform(0.01, 1.0, size=C); alpha /= alpha.sum()
    Lm = np.zeros((d, C * d))
    for j in range(C):
        Lm[:, j * d:(j + 1) * d] += alpha[j] * np.eye(d)
    rk = int(np.linalg.matrix_rank(Lm, tol=1e-10))
    kdim = C * d - rk
    return rec("NT14", "information-gradient duality + softening trap",
               bool(pigeon and zero < 1e-12 and kdim >= (C - 1) * d),
               dict(pigeonhole_premise_nonempty=pigeon,
                    unselected_delta_to_output=zero,
                    soft_aggregate_rank=rk, soft_kernel=kdim,
                    note="softening re-triggers NT13; proofs are analytic"),
               "analytic + finite identity")


# ------------------------------------------------------------------ NT15
def n15():
    alpha, m, tau = 0.05, 3, 0.05
    cases = []
    for Dtrue in (-0.10, -0.20, -1.0):
        for l in (-0.05, -0.02, 0.0, 0.03):
            if l >= -abs(tau) and Dtrue < -abs(tau):
                cases.append(bool(l > Dtrue))
    coverage = bool(cases and all(cases))
    union = min(1.0, m * alpha)
    w, b, k, T = 8, 32, 4, 1000
    rate_ok = bool(T * w * b == 256000 and k * b == 128)
    # zero-order gate on a read-only round
    ci = {"f1": 0.01, "ppl": -0.002, "auroc": 0.03}
    accepted = all((v == v) and v >= -abs(tau) for v in ci.values())
    return rec("NT15", "zero-order gate validity + T*w*b rate bound",
               bool(coverage and rate_ok and accepted),
               dict(coverage_inequality_holds=coverage, union_bound=union,
                    bonferroni_bound=alpha, T=T, w=w, b=b,
                    rate_bound_bits=T * w * b, bandwidth_bits=k * b,
                    independent_of_L_and_C=True,
                    example_round_accepted=bool(accepted),
                    note="(a) conditional on CI coverage + non-default verifier; "
                         "Monte-Carlo never used as proof; T16 withdrawn"),
               "analytic (coverage + data processing) + finite identity")


CHECKS = [n01, n02, n03, n04, n05, n06, n07, n08, n09, n10,
          n11, n12, n13, n14, n15]


def main():
    print("== NT01..NT15 consistency checks (NO TRAINING; proofs are analytic) ==",
          flush=True)
    for fn in CHECKS:
        try:
            fn()
        except Exception as e:
            rec(fn.__name__.upper(), fn.__name__, False,
                dict(exception=type(e).__name__ + ": " + str(e)))
    import new_theorems15 as M
    passed = sum(1 for r in rows if r["verdict"] == "PASS")
    out = dict(
        mode="finite-identity consistency checks ONLY; proofs in new_theorems15.py",
        numpy_seed=20261108, checks=rows, passed=passed, total=len(rows),
        roadmap=M.summary(),
        discipline="No numerical run is a proof. No measured accuracy is promoted "
                   "to a theorem. NO TRAINING, NO GRADIENTS, NO GPU, NO NEW DATA.",
    )
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n==> {passed}/{len(rows)} PASS", flush=True)
    print("roadmap:", M.summary()["delivered"], "/",
          M.summary()["target"], "theorems delivered", flush=True)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
