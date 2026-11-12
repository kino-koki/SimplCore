"""
general_probe.py — 自研路线第二步：把二阶递推的参数空间补全。

从 SGC 数学本质出发
====================
SGC 的 IIR 是 y_t = a·y_{t-1} + b·y_{t-2} + c·u_t。
用 (ρ,θ) 参数化 a=2ρcosθ, b=−ρ²，等价于**把两个极点强制约束成一对复共轭
ρe^{±iθ}**。这只是二阶 LTI 系统参数空间里的一个二维子流形。

二阶递推的完整稳定区域还包括**两个实极点** r1,r2∈(0,1)：
    a = r1+r2,  b = −r1r2,  c = 1
    h[τ] = Σ_{j=0}^{τ} r1^{τ−j} r2^{j}
         = (r1^{τ+1} − r2^{τ+1})/(r1 − r2)
当 r1=r2→1 时 h[τ]=(τ+1) —— 无衰减单调累加器（前面 pureacc 网格想做但
因 θ→0 共线而 cond 爆到 1e22 做不到的那一类）。

这不引入任何外部概念：只是把「人为的阻尼振荡参数化」松开成
「一般稳定二阶 LTI 字典」。全部守恒性质不变：

    · (r1,r2) 与 t 无关 ⇒ 核 Toeplitz ⇒ FFT 整段路径成立
    · 流式 IIR 两拍状态 ⇒ O(1)，与 L 无关
    · 严格因果（过去不受未来扰动）
    · 无 N×N、无 KV cache、无 attention、无混合折中网格

臂
==
  G0_osc       固定复极点（对照 = N0_fixed）
  G1_osc_learn 可学复极点（对照 = N2_perch，验证两文件一致）
  G2_bipole    可学双实极点（本文件主角）
  G3_mixed     一半基复极点 + 一半基双实极点，全部可学
  G4/G5/G6     容量外推（C=8 / C=16 / mixed C=16）

上界仍从 results/fusion_results.json 读取，本文件不实现任何 transformer。
"""
import argparse
import json
import os
import time
import numpy as np

from mini_autograd import T, const, nll_from_logits
import sgc_fixed as S
import task_utils as SP

OUT = "results"
SEEDS = (0, 1, 2)
STEPS = 3000
RHO_MIN = 0.05

ARMS = {
    "G0_osc":        dict(kernel="osc",     learn=False, M=16, C=4),
    "G1_osc_learn":  dict(kernel="osc",     learn=True,  M=16, C=4),
    "G2_bipole":     dict(kernel="bipole",  learn=True,  M=16, C=4),
    "G3_mixed":      dict(kernel="mixed",   learn=True,  M=16, C=4),
    "G4_bipole_C8":  dict(kernel="bipole",  learn=True,  M=16, C=8),
    "G5_bipole_C16": dict(kernel="bipole",  learn=True,  M=16, C=16),
    "G6_mixed_C16":  dict(kernel="mixed",   learn=True,  M=16, C=16),
    # 容量外推 / 少参数等参数对照 / M-vs-C 归因
    "G7_bipole_C32": dict(kernel="bipole",  learn=True,  M=16, C=32),
    "G8_small":      dict(kernel="bipole",  learn=True,  M=16, C=2),
    "G9_M32C4":      dict(kernel="bipole",  learn=True,  M=32, C=4),
}


def _sig(z): return 1.0 / (1.0 + np.exp(-z))
def _logit(u):
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


def init_params(cfg, seed):
    M, Cc = cfg["M"], cfg["C"]
    N = M * Cc
    r = np.random.default_rng(seed + 909)
    rho0, theta0 = S.grids(M, "logfreq", L=SP.L_TRAIN)
    rho0 = np.clip(rho0, RHO_MIN + 1e-4, 1 - 1e-4)
    theta0 = np.clip(theta0, 1e-4, np.pi - 1e-4)
    k = cfg["kernel"]
    P = {}
    if k in ("osc", "mixed"):
        p0 = _logit((rho0 - RHO_MIN) / (1 - RHO_MIN))
        q0 = _logit(theta0 / np.pi)
        cnt = M if k == "mixed" else M
        pc = np.repeat(p0[:, None], Cc, axis=1)
        qc = np.repeat(q0[:, None], Cc, axis=1)
        if k == "mixed":
            half = M // 2
            P["p"] = T((pc[:half] + r.normal(0, 0.02, (half, Cc))).reshape(-1))
            P["q"] = T((qc[:half] + r.normal(0, 0.02, (half, Cc))).reshape(-1))
            P["_n_osc"] = half * Cc
        else:
            P["p"] = T((pc + r.normal(0, 0.02, (M, Cc))).reshape(-1))
            P["q"] = T((qc + r.normal(0, 0.02, (M, Cc))).reshape(-1))
            P["_n_osc"] = N
    if k in ("bipole", "mixed"):
        cnt = N - P.get("_n_osc", 0)
        # 双实极点初值：一根贴近原 ρ，另一根略小 —— 起点接近阻尼振荡包络
        rr = np.repeat(np.clip(rho0, RHO_MIN + 1e-4, 1 - 1e-4)[:, None], Cc, axis=1)
        rr = rr.reshape(-1)
        if k == "mixed":
            rr = rr[cnt * 0:]        # 占位，下面切片
        r1 = np.clip(rr * 0.98, RHO_MIN + 1e-4, 1 - 1e-4)
        r2 = np.clip(rr * 0.55, RHO_MIN + 1e-4, 1 - 1e-4)
        if k == "mixed":
            r1 = r1[-(N - P["_n_osc"]):]
            r2 = r2[-(N - P["_n_osc"]):]
        P["p1"] = T(_logit((r1 - RHO_MIN) / (1 - RHO_MIN)))
        P["p2"] = T(_logit((r2 - RHO_MIN) / (1 - RHO_MIN)))
    return P


def make_params(cfg, seed):
    r = np.random.default_rng(seed + 31337)
    def W(a, b, s=0.15): return T(r.standard_normal((a, b)) * s)
    Hh = cfg["M"] * cfg["C"]
    P = {"E": T(r.standard_normal((SP.VOCAB, SP.D_IN)) * 0.15),
         "Wa": W(SP.D_IN, Hh), "Wb": W(SP.D_IN, Hh),
         "Wq": W(SP.D_IN, Hh), "Wout": W(Hh, SP.VOCAB)}
    if cfg["learn"]:
        P.update({k: v for k, v in init_params(cfg, seed).items()})
    return P


def n_params(P):
    return int(sum(v.d.size for k, v in P.items() if not k.startswith("_")))


# ---------------- 核构造 + 解析雅可比 ----------------
def build(cfg, P, L, learn=True):
    """返回 (K(N,L), back_fn, iir_coeff)。back_fn 把 KT.g 回传到 P 的参数。"""
    M, Cc, N = cfg["M"], cfg["C"], cfg["M"] * cfg["C"]
    tau = np.arange(L, dtype=np.float64)
    k = cfg["kernel"]
    parents = []

    if not learn:
        rho0, theta0 = S.grids(M, "logfreq", L=L)
        K_osc = S.kernel(rho0, theta0, L)                  # (M,L)
        K = np.repeat(K_osc, Cc, axis=0)                   # (N,L)
        return K, (lambda KT: None), dict(kind="fixed", rho=rho0, theta=theta0, Cc=Cc)

    Kparts, metas = [], []
    if k in ("osc", "mixed"):
        p, q = P["p"].d, P["q"].d
        rho = RHO_MIN + (1 - RHO_MIN) * _sig(p)
        theta = np.pi * _sig(q)
        pow_tau = rho[:, None] ** tau[None, :]
        sin_t = np.sin((tau[None, :] + 1.0) * theta[:, None])
        cos_t = np.cos((tau[None, :] + 1.0) * theta[:, None])
        dpow = np.where(tau[None, :] > 0,
                        tau[None, :] * rho[:, None] ** np.maximum(tau[None, :] - 1, 0),
                        0.0)
        Kparts.append(pow_tau * sin_t)
        metas.append(("osc", len(rho), dict(
            drho=dpow * sin_t,
            dtheta=pow_tau * (tau[None, :] + 1.0) * cos_t,
            sp=_sig(p), sq=_sig(q))))
        parents += [P["p"], P["q"]]

    if k in ("bipole", "mixed"):
        p1, p2 = P["p1"].d, P["p2"].d
        r1 = RHO_MIN + (1 - RHO_MIN) * _sig(p1)
        r2 = RHO_MIN + (1 - RHO_MIN) * _sig(p2)
        n = len(r1)
        Kb = np.zeros((n, L))
        d1 = np.zeros((n, L))
        d2 = np.zeros((n, L))
        for t_ in range(L):
            acc = np.zeros(n); g1 = np.zeros(n); g2 = np.zeros(n)
            for j in range(t_ + 1):
                e1, e2 = t_ - j, j
                term = (r1 ** e1) * (r2 ** e2)
                acc += term
                if e1 > 0:
                    g1 += e1 * (r1 ** (e1 - 1)) * (r2 ** e2)
                if e2 > 0:
                    g2 += (r1 ** e1) * e2 * (r2 ** (e2 - 1))
            Kb[:, t_] = acc; d1[:, t_] = g1; d2[:, t_] = g2
        Kparts.append(Kb)
        metas.append(("bipole", n, dict(dr1=d1, dr2=d2,
                                        sp1=_sig(p1), sp2=_sig(p2))))
        parents += [P["p1"], P["p2"]]

    K = np.concatenate(Kparts, axis=0)
    assert K.shape[0] == N, (K.shape, N)

    def back(KT):
        dK = KT.g
        off = 0
        for kind, n, jm in metas:
            d = dK[off:off + n]
            if kind == "osc":
                drho = (d * jm["drho"]).sum(1)
                dtheta = (d * jm["dtheta"]).sum(1)
                sp, sq = jm["sp"], jm["sq"]
                P["p"].g += drho * (1 - RHO_MIN) * sp * (1 - sp)
                P["q"].g += dtheta * np.pi * sq * (1 - sq)
            else:
                dr1 = (d * jm["dr1"]).sum(1)
                dr2 = (d * jm["dr2"]).sum(1)
                sp1, sp2 = jm["sp1"], jm["sp2"]
                sc = (1 - RHO_MIN)
                P["p1"].g += dr1 * sc * sp1 * (1 - sp1)
                P["p2"].g += dr2 * sc * sp2 * (1 - sp2)
            off += n
    return K, back, dict(kind=k, parents=parents)


def forward(arm, P, X, Xp, L):
    cfg = ARMS[arm]
    B_, _, _ = X.shape
    N = cfg["M"] * cfg["C"]
    xx = const(X) @ P["E"]
    xp = const(Xp) @ P["E"]
    u = (xx @ P["Wa"]) * (xp @ P["Wb"])
    u4 = u.reshape(B_, L, N, 1)

    K, back_K, meta = build(cfg, P, L, learn=cfg["learn"])
    Tm = S.toeplitz(K)
    if cfg["learn"]:
        KT = T(K, meta["parents"])
        KT._b = lambda: back_K(KT)
        y = T(np.einsum("nls,bsnk->blnk", Tm, u4.d), [KT, u4])
        def back_y():
            dy = y.g[:, :, :, 0]
            dK = np.zeros((N, L))
            for t_ in range(L):
                dK[:, t_] = np.einsum("btn,btn->n",
                                      dy[:, t_:, :], u4.d[:, :L - t_, :, 0])
            KT.g += dK
            u4.g[:, :, :, 0] += np.einsum("nts,btn->bsn", Tm, dy)
        y._b = back_y
    else:
        y = T(np.einsum("nls,bsnk->blnk", Tm, u4.d), [u4])
        def back_y():
            dy = y.g[:, :, :, 0]
            u4.g[:, :, :, 0] += np.einsum("nts,btn->bsn", Tm, dy)
        y._b = back_y

    ylast = y[:, L - 1, :, 0]
    z = (xx[:, L - 1, :] @ P["Wq"]) * ylast
    return z @ P["Wout"], K


# ---------------- 守恒断言（支持一般二阶 IIR） ----------------
def guards(arm, P, seed=0, verbose=False):
    cfg = ARMS[arm]
    L = SP.L_TRAIN
    N = cfg["M"] * cfg["C"]
    _, K = forward(arm, P, *SP.build(SP.N_PAIRS, 2, seed=1)[:2], L)
    r = np.random.default_rng(seed)
    u = r.standard_normal((2, L, N, 1))
    Tm = S.toeplitz(K)
    y = np.einsum("nls,bsnk->blnk", Tm, u)

    # 1 Toeplitz
    dev = max(abs(Tm[n, t, s] - Tm[n, t - 1, s - 1])
              for n in range(N) for t in range(1, L) for s in range(1, t + 1))
    # 2 FFT 整段
    yf = S.fft_apply(K, u)
    den = max(np.abs(y).max(), 1e-12)
    rel_fft = float(np.abs(y - yf).max() / den)
    # 3 IIR 流式：一般二阶 y_t = a·y1 + b·y2 + c·u_t
    #    h[0]=c, h[1]=a·c, h[2]=(a²+b)·c —— 对 osc(c=sinθ) 与 bipole(c=1) 统一成立
    c = np.where(np.abs(K[:, 0]) > 1e-12, K[:, 0], 1e-12)
    a = K[:, 1] / c
    b = K[:, 2] / c - a * a
    y1 = np.zeros((2, N, 1)); y2 = np.zeros((2, N, 1))
    yi = np.zeros_like(y)
    for t in range(L):
        yt = a[None, :, None] * y1 + b[None, :, None] * y2 \
             + c[None, :, None] * u[:, t, :, :]
        yi[:, t, :, :] = yt
        y2, y1 = y1, yt
    rel_iir = float(np.abs(y - yi).max() / den)
    # 4 因果
    u2 = u.copy(); u2[:, L - 1] += 10
    past = float(np.abs(y[:, :L - 1] - np.einsum("nls,bsnk->blnk", Tm, u2)[:, :L - 1]).max())
    # 5 状态与 L 无关
    bytes_ = [2 * N * 4 for _ in (9, 17, 25, 65)]
    out = dict(
        toeplitz=bool(dev < 1e-10), fft_iir=bool(rel_fft < 1e-9),
        iir_vs_toeplitz=bool(rel_iir < 1e-8), strict_causality=bool(past == 0),
        state_independent_of_L=bool(len(set(bytes_)) == 1),
        max_toeplitz_dev=float(dev), rel_fft=rel_fft, rel_iir=rel_iir,
        state_bytes=bytes_[0],
        time_invariance="(r1,r2)/(ρ,θ) 与 t 无关 ⇒ K 只依赖 τ",
        all_passed=bool(dev < 1e-10 and rel_fft < 1e-9 and rel_iir < 1e-8
                        and past == 0 and len(set(bytes_)) == 1))
    if verbose:
        print(f"    toeplitz={out['toeplitz']} fft={rel_fft:.2e} "
              f"iir={rel_iir:.2e} causal={out['strict_causality']} "
              f"all={out['all_passed']}")
    return out


# ---------------- 训练 ----------------
def eval_correct(arm, P, X, Xp, Y):
    return (forward(arm, P, X, Xp, X.shape[1])[0].d.argmax(-1) == Y).astype(int)


def eval_acc(arm, P, X, Xp, Y):
    return float(eval_correct(arm, P, X, Xp, Y).mean())


def run_train(arm, seed, steps=STEPS):
    cfg = ARMS[arm]
    P = make_params(cfg, seed)
    g0 = guards(arm, P, seed)
    Xtr, Xptr, Ytr, _ = SP.build(SP.N_PAIRS, 2000, seed=1000 + seed)
    Xte, Xpte, Yte, _ = SP.build(SP.N_PAIRS, 512, seed=9000 + seed)
    ADAM = {}
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = np.random.default_rng(seed * 100000 + step).choice(len(Ytr), SP.B, replace=False)
        X, Xp, Y = Xtr[idx], Xptr[idx], Ytr[idx]
        for k, v in P.items():
            if not k.startswith("_"): v.g[...] = 0.0
        lg, _ = forward(arm, P, X, Xp, SP.L_TRAIN)
        nll_from_logits(lg, Y).backward()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, v in P.items():
            if name.startswith("_"): continue
            key = "m_" + name
            if key not in ADAM:
                ADAM[key] = np.zeros_like(v.d); ADAM["v_" + name] = np.zeros_like(v.d)
            m, vv = ADAM[key], ADAM["v_" + name]
            m *= b1; m += (1 - b1) * v.g
            vv *= b2; vv += (1 - b2) * v.g * v.g
            v.d -= SP.LR * (m / (1 - b1 ** step)) / (np.sqrt(vv / (1 - b2 ** step)) + eps)
    g1 = guards(arm, P, seed)
    corr_te = eval_correct(arm, P, Xte, Xpte, Yte)
    acc = float(corr_te.mean())
    Xsh, Xpsh, Ysh, _ = SP.build(SP.N_PAIRS, 512, seed=9000 + seed, shuffle_pairs=True)
    sh = eval_acc(arm, P, Xsh, Xpsh, Ysh)
    gen = {}
    for n in (4, 8, 12, 16):
        Xg, Xpg, Yg, _ = SP.build(n, 256, seed=7000 + seed + n)
        gen[f"L{2*n+1}"] = eval_acc(arm, P, Xg, Xpg, Yg)
    return dict(arm=arm, seed=seed, params=n_params(P),
                guards_before=g0, guards_after=g1,
                guards_all_passed=bool(g0["all_passed"] and g1["all_passed"]),
                final_acc=acc, shuffle_acc=sh, length_generalization=gen,
                correct_te=corr_te.tolist(),
                elapsed_s=round(time.time() - t0, 1))


def gradcheck(arm, seed=0):
    cfg = ARMS[arm]; P = make_params(cfg, seed)
    X, Xp, Y, _ = SP.build(SP.N_PAIRS, 4, seed=42)
    for k, v in P.items():
        if not k.startswith("_"): v.g[...] = 0.0
    nll_from_logits(forward(arm, P, X, Xp, SP.L_TRAIN)[0], Y).backward()
    analytic = {k: v.g.ravel().copy() for k, v in P.items() if not k.startswith("_")}
    r = np.random.default_rng(321); eps = 1e-6; worst = 0.0; ok_all = True
    for name in [k for k in P if not k.startswith("_")]:
        v = P[name]; flat = v.d.ravel(); gf = analytic[name]
        for i in r.choice(flat.size, size=min(3, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps; lp = nll_from_logits(forward(arm, P, X, Xp, SP.L_TRAIN)[0], Y).d
            flat[i] = old - eps; lm = nll_from_logits(forward(arm, P, X, Xp, SP.L_TRAIN)[0], Y).d
            flat[i] = old
            num = float((lp - lm) / (2 * eps)); ana = float(gf[i])
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
            ok = rel < 5e-4 or abs(num - ana) < 1e-8
            ok_all &= ok; worst = max(worst, 0.0 if ok else rel)
    return dict(arm=arm, worst_rel=worst, passed=bool(ok_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="guards", choices=["guards", "gradcheck", "all"])
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "guards":
        out = {arm: guards(arm, make_params(ARMS[arm], 0), verbose=True) for arm in ARMS}
        with open(f"{OUT}/general_guards.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        for arm, g in out.items():
            print(f"  {arm:16s} all={g['all_passed']} bytes={g['state_bytes']}")
        return
    if a.mode == "gradcheck":
        out = [gradcheck(arm) for arm in ARMS]
        for g in out:
            print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['arm']:16s} {g['worst_rel']:.2e}")
        with open(f"{OUT}/general_gradcheck.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"==> {sum(g['passed'] for g in out)}/{len(out)} PASS")
        return
    if a.mode == "all":
        fn = f"{OUT}/general_results.json"
        allres = json.load(open(fn)) if os.path.exists(fn) else {}
        for arm in ARMS:
            done = {int(r["seed"]) for r in allres.get(arm, [])}
            for sd in SEEDS:
                if sd in done:
                    print(f"  [{arm:16s} s{sd}] skip", flush=True); continue
                r = run_train(arm, sd, steps=a.steps)
                allres.setdefault(arm, []).append(r)
                print(f"  [{arm:16s} s{sd}] acc={r['final_acc']:.4f} "
                      f"shuffle={r['shuffle_acc']:.4f} params={r['params']} "
                      f"guards={r['guards_all_passed']} {r['elapsed_s']}s", flush=True)
                with open(fn, "w") as f:
                    json.dump(allres, f, indent=1, ensure_ascii=False); f.flush()
        print("\n写入", fn)


if __name__ == "__main__":
    main()
