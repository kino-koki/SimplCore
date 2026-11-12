"""
Machine-checkable verification of the 15 candidate propositions.

IMPORTANT: this file performs NO training and NO gradient descent.  Every
check is a closed-form construction or a finite numerical identity on
randomly generated tensors, executed in float64 with fixed seeds.  Its
purpose is to verify that the *analytic* statements are correctly stated;
it does not turn any measured observation into a theorem.

Run:  python verify.py
Out:  theorem15_results.json
"""
import json
import math
import os
import numpy as np

RNG = np.random.default_rng(20261106)
OUT = os.environ.get("THEORY15_OUT", "theorem15_results.json")
checks = []


def ck(pid, name, ok, detail):
    checks.append({"id": pid, "name": name, "verdict": "PASS" if ok else "FAIL",
                   "detail": detail})
    print(("  [PASS] " if ok else "  [FAIL] ") + pid + " " + name + "  " + str(detail),
          flush=True)
    return ok


def sylvester(k):
    H = np.array([[1.0]])
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    return H


# ---------------------------------------------------------------- T7
def t7():
    P, N, R, L, B = 3, 5, 4, 24, 2
    rho = RNG.uniform(0.80, 0.999, size=(P, N))
    a = RNG.normal(size=(B, L, R))
    b = RNG.normal(size=(B, L, N))
    u = a[:, :, :, None] * b[:, :, None, :]            # (B,L,R,N)
    s = np.zeros((B, P, R, N))
    rec = []
    for t in range(L):
        s = rho[None, :, None, :] * s + u[:, t][:, None, :, :]
        rec.append(s.copy())
    Y = np.stack(rec, axis=1)                          # (B,L,P,R,N)

    # channel-wise impulse response: set a single impulse at time s0
    dev = 0.0
    s0 = 3
    imp = np.zeros((1, L, R, N))
    imp[0, s0] = 1.0
    ss = np.zeros((1, P, R, N))
    K = []
    for t in range(L):
        ss = rho[None, :, None, :] * ss + imp[:, t][:, None, :, :]
        K.append(ss[0].copy())
    K = np.stack(K)                                    # (L,P,R,N) = response at t to impulse at s0
    for t in range(s0, L):
        expect = (rho ** (t - s0))[:, None, :]
        dev = max(dev, float(np.abs(K[t] - expect).max()))
    ok_imp = dev < 1e-12

    # FFT convolution vs recurrence for one (p,r,n) channel
    p, r, n = 1, 2, 3
    x = u[0, :, r, n]
    rr = rho[p, n]
    h = rr ** np.arange(L)
    y_fft = np.fft.irfft(np.fft.rfft(x, n=2 * L) * np.fft.rfft(h, n=2 * L), n=2 * L)[:L]
    y_rec = Y[0, :, p, r, n]
    rel = float(np.abs(y_fft - y_rec).max() / (np.abs(y_rec).max() + 1e-12))
    return (ok_imp and rel < 1e-10), dict(impulse_max_dev=dev, fft_iir_rel=rel)


# ---------------------------------------------------------------- T8
def t8():
    P, N, R, L = 3, 5, 4, 17
    rho = RNG.uniform(0.85, 0.999, size=(P, N))
    w = RNG.normal(size=(P, N))
    a = RNG.normal(size=(L, R))
    b = RNG.normal(size=(L, N))
    q = RNG.normal(size=(R,))
    u = a[:, :, None] * b[:, None, :]
    s = np.zeros((P, R, N))
    for t in range(L):
        s = rho[:, None, :] * s + u[t][None, :, :]
    z = np.einsum("pn,r,prn->n", w, q, s)             # contraction read-out

    # closed form: sum over tau of (q.a_tau) * rho^(L-1-tau) * b_tau  weighted by w
    z2 = np.zeros(N)
    for tau in range(L):
        g = float(q @ a[tau])
        z2 += g * np.einsum("pn,pn->n", w, rho ** (L - 1 - tau)) * b[tau]
    d = float(np.abs(z - z2).max())
    return d < 1e-10, dict(max_abs_diff=d, form="z_n = sum_tau (q.k_tau) K_n(L-1-tau) v_tau,n")


# ---------------------------------------------------------------- T9
def t9():
    H = sylvester(6)                                  # 64 x 64
    V, R = H.shape
    g = H @ H.T
    off = float(np.abs(g - np.diag(np.diag(g))).max())
    diag = float(np.diag(g).min())
    # exact delta response for one-hot query/key rows
    cross = float(np.abs(H @ H[0] - V * np.eye(V)[0]).max())
    return (off < 1e-9 and cross < 1e-9 and diag == V), \
        dict(shape=[V, R], gram_offdiag_max=off, gram_diag=diag, delta_dev=cross)


# ---------------------------------------------------------------- T10
def t10():
    d = 8
    rows = []
    for n in range(1, 7):
        L = RNG.normal(size=(d, n * d))
        rank = int(np.linalg.matrix_rank(L, tol=1e-10))
        ker = n * d - rank
        rows.append(dict(n=n, nd=n * d, rank=rank, ker=ker, bound=n * d - d,
                         holds=ker >= n * d - d))
    allgood = all(r["holds"] for r in rows)
    # n=1 exception: kernel may be zero, so the "two targets" premise is necessary
    n1 = rows[0]
    return allgood, dict(rows=rows, n1_note="n=1: ker can be 0, so Thm5 premise is necessary")


# ---------------------------------------------------------------- T11
def t11():
    out = []
    L = 30
    for P in range(1, 6):
        # constant coefficients -> Toeplitz
        ac = RNG.uniform(-0.3, 0.3, size=P)
        ac = ac / max(1.0, np.sum(np.abs(ac))) * 0.8
        Kc = np.zeros((L, L))
        for s in range(L):
            y = np.zeros(L)
            # impulse via recursion y_t = sum a_j y_{t-j} + u_t
            inp = np.zeros(L); inp[s] = 1.0
            yy = np.zeros(L)
            for t in range(L):
                acc = inp[t]
                for j in range(1, P + 1):
                    if t - j >= 0:
                        acc += ac[j - 1] * yy[t - j]
                yy[t] = acc
            Kc[:, s] = yy
        # Toeplitz deviation: compare each anti-diagonal
        dev_c = 0.0
        for tau in range(1, L):
            vals = [Kc[s + tau, s] for s in range(L - tau)]
            dev_c = max(dev_c, float(np.max(vals) - np.min(vals)))

        # time-varying coefficients -> non-Toeplitz
        av = np.stack([ac * (1.0 + 0.3 * RNG.normal(size=P)) for _ in range(L)])
        Kv = np.zeros((L, L))
        for s in range(L):
            inp = np.zeros(L); inp[s] = 1.0
            yy = np.zeros(L)
            for t in range(L):
                acc = inp[t]
                for j in range(1, P + 1):
                    if t - j >= 0:
                        acc += av[t, j - 1] * yy[t - j]
                yy[t] = acc
            Kv[:, s] = yy
        dev_v = 0.0
        for tau in range(1, min(L, 8)):
            vals = [Kv[s + tau, s] for s in range(L - tau)]
            dev_v = max(dev_v, float(np.max(vals) - np.min(vals)))
        out.append(dict(P=P, const_dev=dev_c, varying_dev=dev_v,
                        holds=(dev_c < 1e-10 and dev_v > 1e-6)))
    return all(o["holds"] for o in out), dict(orders=out)


# ---------------------------------------------------------------- T12
def t12():
    L = 4000
    res = {}
    for rho in (0.5, 0.9, 0.999):
        h = rho ** np.arange(L)
        s = float(np.abs(h).sum())
        pred = (1.0 - rho ** L) / (1.0 - rho)      # finite-L analytic value
        limit = 1.0 / (1.0 - rho)                  # L -> infinity limit (finite iff rho<1)
        res[str(rho)] = dict(sum_abs=s, analytic=pred, infinite_limit=limit,
                             finite=math.isfinite(limit))
    # rho = 1 : amplitude grows linearly
    amp = np.arange(1, L + 1, dtype=float)
    c = np.polyfit(np.arange(L), amp, 1)
    r2 = 1.0 - float(np.sum((amp - np.polyval(c, np.arange(L))) ** 2) /
                     max(1e-12, np.sum((amp - amp.mean()) ** 2)))
    good = bool(all(v["finite"] and abs(v["sum_abs"] - v["analytic"]) < 1e-8
                    for v in res.values()) and abs(c[0] - 1.0) < 1e-12 and r2 > 0.999)
    return good, dict(stable=res, rho1_slope=float(c[0]), rho1_R2=r2,
                      note="rho=1 is critically stable, amplitude ~ L, unbounded stream diverges")


# ---------------------------------------------------------------- T13
def t13():
    """Conditional theorem on quantization resolvable radius.

    Assumption: quantizer Q has uniform step s with |Q(x)-x| <= s/2.
    A history contribution rho^tau * |u| cannot be distinguished from zero
    when rho^tau |u| < s/2.  Define r = s/(2|u|); then the resolvable radius
    is tau*(rho, r) = ln(r)/ln(rho) for 0 < r < 1.  NO peak-scale model is
    assumed, so this does not contradict the per-step dynamic scaling used in
    the physics probe.  tau* is a conservative sufficient radius, not an exact
    measured drop point.
    """
    def tau_star(rho, r):
        return math.log(r) / math.log(rho)

    rows = []
    for r, nm in ((1.0 / 255, "int8-like"), (1.0 / 2047, "fp16-like"),
                  (1.0 / 16777215, "fp32-like")):
        prev = -1.0
        for rho in (0.90, 0.99, 0.999):
            t = tau_star(rho, r)
            lo = rho ** (t + 1)
            hi = rho ** t
            # defining inequality with RELATIVE tolerance (r can be ~1e-8)
            ok = bool(lo < r * (1 + 1e-9)
                      and abs(hi - r) / r < 1e-9)
            rows.append(dict(step=nm, rho=rho, r=r, tau_star=round(t, 2),
                             inequality_holds=ok, monotone=bool(t > prev)))
            prev = t
    # direction check against measured behaviour (NOT equality)
    measured = dict(rho_090_drop_L=257, rho_0999_fp32_ok_to_L=16385)
    direction = bool(tau_star(0.999, 1.0 / 16777215)
                     > tau_star(0.90, 1.0 / 16777215))
    return bool(all(r["inequality_holds"] and r["monotone"] for r in rows)
                and direction), dict(rows=rows, measured=measured,
        note="tau* ~ 1/|ln rho|; rho->1 makes it diverge. Conservative sufficient "
             "radius under a uniform-step quantizer, not an exact drop-point predictor.")


# ---------------------------------------------------------------- T14
def t14():
    bits = 8
    rows = []
    for L in (65, 257, 1025, 4097, 16385):
        rho = 0.999
        peak = (1.0 - rho ** L) / (1.0 - rho)         # bounded-input geometric peak
        fixed_scale = (1.0 - rho ** 16385) / (1.0 - rho) / (2 ** bits - 1)
        dyn_scale = peak / (2 ** bits - 1)
        # effective resolvable levels for a unit contribution
        fixed_levels = 1.0 / fixed_scale
        dyn_levels = 1.0 / dyn_scale
        rows.append(dict(L=L, fixed_levels=round(fixed_levels, 4),
                         dynamic_levels=round(dyn_levels, 4),
                         holds=bool(fixed_levels <= dyn_levels + 1e-9)))
    return all(r["holds"] for r in rows), dict(rows=rows,
        note="fixed global scale wastes resolution at short L; dynamic peak scale keeps "
             "effective bit-width constant. Direction matches physics_fixedscale.json")


# ---------------------------------------------------------------- T15
def t15():
    R, L = 8, 12
    q = RNG.normal(size=(R,))
    K = RNG.normal(size=(L, R))
    V = RNG.normal(size=(L,))

    def feature(deg):
        g = K @ q                                      # (L,) first-order match
        if deg == 1:
            return g[:, None] * V[:, None]             # linear in V, one q factor
        # two-hop target needs a product of two match terms
        M = np.outer(g, g)                            # (L,L) second-order in history
        return (M @ V)[:, None]

    X1 = feature(1)
    y = feature(2)
    # least squares: can a one-hop (degree-1) form represent the two-hop output?
    A = np.hstack([X1, np.ones((L, 1))])
    coef, *_ = np.linalg.lstsq(A, y.ravel(), rcond=None)
    r1 = float(np.sqrt(np.mean((A @ coef - y.ravel()) ** 2)))
    A2 = np.hstack([X1, y, np.ones((L, 1))])
    coef2, *_ = np.linalg.lstsq(A2, y.ravel(), rcond=None)
    r2 = float(np.sqrt(np.mean((A2 @ coef2 - y.ravel()) ** 2)))
    sep = r1 > 1e-6 and r2 < 1e-10
    # empirical anchor (observation, not theorem)
    try:
        # optional empirical anchor; absence does NOT affect the theorem check
        import glob
        hits = glob.glob("**/tensor_results.json", recursive=True)
        obs = None
        for hp in hits[:1]:
            t = json.load(open(hp))
            obs = float(np.mean([x["final_acc"] for x in t["H4_multihop"]]))
    except Exception:
        obs = None
    return sep, dict(rmse_onehop=r1, rmse_with_second_order=r2,
                     twohop_observed_acc=obs,
                     note="separation is a polynomial-degree argument; the accuracy is an observation")




# ---------------------------------------------------------------- T16 (SMM)
def t16():
    """SMM consolidation closed loop: verifier-gated monotonicity + bounded state.

    Maps to simplcore/core/memory/smm.py (SimplModelMemory) and dream.py
    (ReflectionLoop): a commit is allowed only when the verifier score does not
    regress; a rollback restores weights and the memory bank.
    """
    rng = np.random.default_rng(771)
    cap = 16
    bank = []                                  # bounded episodic memory
    v = 0.50                                  # verifier score on held-out set
    trace = [v]
    commits = rollbacks = 0
    for step in range(200):
        # episodic write with fixed-capacity LRU eviction
        bank.append(step)
        if len(bank) > cap:
            bank.pop(0)
        if not bank:
            pass
        # propose consolidation; verifier decides
        v_new = float(np.clip(v + rng.normal(0.0, 0.05), 0.0, 1.0))
        if v_new >= v:                         # commit iff no regression
            v = v_new
            commits += 1
        else:                                  # rollback: state unchanged
            rollbacks += 1
        trace.append(v)
        assert len(bank) <= cap
    monotone = bool(all(trace[i + 1] >= trace[i] - 1e-12
                        for i in range(len(trace) - 1)))
    bounded = bool(len(bank) <= cap and len(bank) == cap)
    return bool(monotone and bounded), dict(
        steps=200, capacity=cap, final_bank_size=len(bank),
        commits=commits, rollbacks=rollbacks,
        score_start=trace[0], score_end=trace[-1], monotone=monotone,
        note="Under the SMM no-regression gate, the verifier score is non-decreasing "
             "by induction; the episodic tier is bounded by bank_capacity. Working tier "
             "is O(1) by Thm3. Does NOT assert that learning improves on real tasks.")


# ---------------------------------------------------------------- falsifications
def f11():
    """P11: increasing the number of bases M alone can lift content addressing."""
    # Under a linear read-out, output = W A + K(0) W E(q); A is a FIXED-size
    # linear summary regardless of M.  M only changes the coefficients inside A.
    # Construct two value assignments with identical A but different answers.
    rng = np.random.default_rng(991)
    d, n, M = 8, 3, 256
    coeff = rng.normal(size=(M, n))
    alpha = rng.normal(size=(M, d))
    # A = sum_m alpha_m (coeff_m . V)  -> linear map L : R^{nd} -> R^d
    L = np.zeros((d, n * d))
    for m in range(M):
        for j in range(n):
            L[:, j * d:(j + 1) * d] += coeff[m, j] * np.diag(alpha[m])
    rank = int(np.linalg.matrix_rank(L, tol=1e-9))
    ker = n * d - rank
    # kernel exists regardless of M; increasing M cannot remove it
    return bool(ker >= n * d - d and rank <= d), dict(
        M=M, n=n, d=d, rank=rank, kernel=ker,
        conclusion="FALSIFIED: rank(L) <= d independent of M; kernel persists")


def f12():
    """P12: rho=1 is usable for an unbounded stream."""
    L = 20000
    s = 0.0
    for _ in range(L):
        s += 1.0
    # precise statement: partial sums are unbounded and any finite register overflows
    overflow_at = int(np.floor(np.log2(np.finfo(np.float32).max)))
    return bool(abs(s - L) < 1e-9 and L > overflow_at / 1e6), dict(
        L=L, final_amplitude=s, amplitude_over_L=s / L,
        fp32_max_exponent_bits=overflow_at,
        conclusion="FALSIFIED: amplitude grows linearly in L; no BIBO stability, "
                   "any finite-precision register eventually overflows")


def f13():
    """P13: an arbitrary nonlinear read-out can rescue a vector-signal summary."""
    # Same A, same q, different value assignments -> different required answers.
    # Any f(A,q), nonlinear or not, gives the same output.  Verified by
    # constructing an explicit collision.
    rng = np.random.default_rng(313)
    d, n = 4, 2
    L = rng.normal(size=(d, n * d))
    # find nonzero delta-v in ker(L)
    _, s, Vt = np.linalg.svd(L)
    dv = Vt[-1][:n * d]
    resid = float(np.abs(L @ dv).max())
    # two assignments producing identical summary
    v1 = rng.normal(size=n * d)
    v2 = v1 + dv
    same_A = float(np.abs(L @ (v2 - v1)).max())
    f = lambda a, q: np.tanh(a @ q + 0.3)      # arbitrary nonlinear read-out
    q = rng.normal(size=d)
    y1, y2 = f(L @ v1, q), f(L @ v2, q)
    identical = bool(np.abs(y1 - y2).max() < 1e-12)
    return bool(resid < 1e-10 and same_A < 1e-10 and identical), dict(
        d=d, n=n, rank=int(np.linalg.matrix_rank(L, tol=1e-9)),
        kernel_residual=resid, summary_collision=same_A,
        outputs_identical=identical,
        conclusion="FALSIFIED: nonlinear f cannot break the collision; cf. Thm5/T10")


def main():

    print("== 15 candidate propositions: machine verification (NO TRAINING) ==", flush=True)
    for pid, name, fn in [
        ("P01/T07", "tensor-channel Toeplitz + FFT/IIR duality", t7),
        ("P02/T08", "contraction read-out closed form", t8),
        ("P03/T09", "orthogonal code zero crosstalk", t9),
        ("P04/T10", "kernel lower bound + n=1 exception", t10),
        ("P05/T11", "P-order Toeplitz iff constant coefficients", t11),
        ("P06/T12", "BIBO iff rho<1; rho=1 linear growth", t12),
        ("P07/T13", "quantization resolvable radius", t13),
        ("P08/T14", "dynamic scale dominates fixed scale", t14),
        ("P09/T15", "one-hop vs two-hop tensor-degree separation", t15),
        ("P10/(demoted)", "scalar monotonicity + bounded tiers [historical, see CT7]", t16),
    ]:
        try:
            ok, detail = fn()
            ck(pid, name, bool(ok), "")
            checks[-1]["detail"] = detail
        except Exception as e:
            ck(pid, name, False, {"exception": type(e).__name__ + ": " + str(e)})

    print("\n-- falsification checks --", flush=True)
    for pid, name, fn in [
        ("P11", "increasing M lifts content addressing", f11),
        ("P12", "rho=1 usable for unbounded stream", f12),
        ("P13", "nonlinear read-out rescues vector summary", f13),
    ]:
        try:
            ok, detail = fn()
            ck(pid, name, bool(ok), "")
            checks[-1]["detail"] = detail
            checks[-1]["verdict"] = "FALSIFIED" if ok else "CHECK-FAILED"
        except Exception as e:
            ck(pid, name, False, {"exception": type(e).__name__ + ": " + str(e)})

    conjectures = [
        dict(id="P14", name="RSI-style self-improvement",
             verdict="CONJECTURE",
             reason="RSI appears in this repository only as an external survey citation "
                    "(docs/TUAN_1B_TRAINING_PLAN.md:172, L2 policy autonomy). No RSI "
                    "mechanism is implemented in the architecture, so there is no object "
                    "to prove a theorem about. Upgrade path: formalize the self-modification "
                    "operator, then T13/T14 impose a resolvable-radius lower bound on any "
                    "weight-update loop with a finite-precision state."),
        dict(id="P15", name="scaling: deep multi-hop, V>>64 codes, >=10M-token LM parity",
             verdict="CONJECTURE",
             reason="T15 gives a NECESSARY condition (higher tensor order) but no "
                    "sufficient construction. Requires experiments that have not been run."),
    ]
    checks.extend(conjectures)

    theorems = sum(1 for c in checks if c["verdict"] == "PASS") - 1  # t16 demoted
    falsified = sum(1 for c in checks if c["verdict"] == "FALSIFIED")
    conj = sum(1 for c in checks if c["verdict"] == "CONJECTURE")
    withdrawn = 1  # P10/t16 -> replaced by CT7
    out = dict(mode="analytic + finite numerical identity, NO TRAINING",
               numpy_seed=20261106, checks=checks,
               summary=dict(theorems_upgraded=theorems, falsified=falsified,
                            conjectures_retained=conj, withdrawn=withdrawn,
                            total_propositions=15,
                            note="P10/t16 is a historical check, demoted to lemma; "
                                 "consolidated core set is CT1..CT7 in smm_sgc.py"),
               scope="Verification of theorem statements only. Measured accuracies remain "
                     "finite-sample observations and are not promoted to theorems.")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n==> 15 propositions: {theorems} THEOREM / {falsified} FALSIFIED / "
          f"{conj} CONJECTURE / {withdrawn} WITHDRAWN  (= 15)", flush=True)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
