"""
native_probe.py — 完全自研路线：可学时不变谱字典（Learnable Time-Invariant
Spectral Dictionary, LTISD）。

为什么这是「从 SGC 数学本质里长出来」的
========================================
前面所有轮动过的自由度：

  · 核外 f/g（product 绑定、query 解调）      -> 0.53~0.67
  · 核外状态门 / 共振读出 / 三方案融合        -> 反协同，交互项 −0.0150
  · ρ/θ 随 t 变（SFSC）                      -> 引理 B4：LTV 非 Toeplitz，对偶失效
  · 人为换网格（logfreq / linear / pureacc）  -> 扫描 9 档无相变

**唯一从没动过的自由度：核形状本身可学，但严格保持时不变。**

SGC 的 IIR 递推是
    y_t = a·y_{t-1} + b·y_{t-2} + c·u_t,   a=2ρcosθ, b=−ρ², c=sinθ
(ρ, θ) 在源码里是「初始化网格」(init_mode)，**不是架构约束**。
只要 (ρ, θ) 与 t 无关，系统就是 LTI：核仍是 Toeplitz、FFT/IIR 对偶仍成立、
状态仍 O(1)、严格因果仍逐位成立。把 (ρ, θ) 变成可学参数，
既不引入 N×N、不引入 KV cache、不引入任何外部方案，也不触发 B4。

参数化（保证稳定，不需额外正则）
    ρ   = ρ_min + (1−ρ_min)·σ(p)      ∈ (ρ_min, 1)
    θ   = π·σ(q)                      ∈ (0, π)
    K_n[τ] = ρ_n^τ · sin((τ+1)θ_n)   （源码口径，不含 1/sinθ）

共享 / 逐通道
    shared : p,q 形状 (M,)        —— M 个基各一套核，C 通道共享
    perch  : p,q 形状 (M·C,)      —— 每个 (m,c) 一套核
             这正是 v4 发现被 `c.view(32,1).expand(32,2048)` 人为压平的
             65536 个自由度。SGC.coef 原生形状就是 (M, D)。

守恒口径（重要修正）
====================
旧 guards 里的 `rho_theta_frozen` 断言「训练前后数值不变」，对可学字典不适用。
正确的守恒是 **时不变性**：(ρ,θ) 不随 t 变 ⇒ K 只依赖 τ ⇒ Toeplitz。
本文件在训练结束后用**最终** (ρ,θ) 重跑全部六条断言，证明学完仍是 LTI。

上界标尺
========
不实现任何 transformer / attention 代码。上界数字从已有
`results/fusion_results.json` 的 `UP_softmax`（带局部窗口的 softmax attention，
0.9987 @ 10,240 参数）读取，仅作为评测参照。

用法
====
  python native_probe.py --mode guards
  python native_probe.py --mode gradcheck
  python native_probe.py --mode all --steps 3000
"""
import argparse
import json
import os
import time
import numpy as np

from mini_autograd import T, const, zeros, nll_from_logits
import sgc_fixed as S
import super_probe as SP

OUT = "results"
SEEDS = (0, 1, 2)
STEPS = 3000
RHO_MIN = 0.05

ARMS = {
    "N0_fixed":       dict(f="product", grid="logfreq",  M=16, C=4,  learn=None),
    "N1_shared":      dict(f="product", grid="logfreq",  M=16, C=4,  learn="shared"),
    "N2_perch":       dict(f="product", grid="logfreq",  M=16, C=4,  learn="perch"),
    "N3_perch_C8":    dict(f="product", grid="logfreq",  M=16, C=8,  learn="perch"),
    "N4_perch_M32":   dict(f="product", grid="logfreq",  M=32, C=4,  learn="perch"),
    "N5_scaninit":    dict(f="product", grid="scan",     M=16, C=4,  learn="perch"),
    "N6_noBind":      dict(f="lin",     grid="logfreq",  M=16, C=4,  learn="perch"),
    "N7_perch_C16":   dict(f="product", grid="logfreq",  M=16, C=16, learn="perch"),
}


# ---------------- (ρ,θ) 参数化 ----------------
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(u):
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


def init_pq(cfg, seed):
    """从固定网格反算 p,q —— 可学字典从与固定臂完全相同的初值出发。"""
    M, Cc = cfg["M"], cfg["C"]
    rho0, theta0 = S.grids(M, cfg["grid"], L=SP.L_TRAIN)
    rho0 = np.clip(rho0, RHO_MIN + 1e-4, 1 - 1e-4)
    theta0 = np.clip(theta0, 1e-4, np.pi - 1e-4)
    p0 = _logit((rho0 - RHO_MIN) / (1 - RHO_MIN))
    q0 = _logit(theta0 / np.pi)
    r = np.random.default_rng(seed + 555)
    if cfg["learn"] == "shared":
        return p0.copy(), q0.copy(), (M,)
    # 逐通道：广播后加微小扰动打破对称（不是借鉴，只是初始化去退化）
    p = np.repeat(p0[:, None], Cc, axis=1) + r.normal(0, 0.02, (M, Cc))
    q = np.repeat(q0[:, None], Cc, axis=1) + r.normal(0, 0.02, (M, Cc))
    return p.reshape(-1), q.reshape(-1), (M * Cc,)


def rho_theta(p, q, shape, M, Cc):
    rho = RHO_MIN + (1 - RHO_MIN) * _sig(p)
    theta = np.pi * _sig(q)
    if shape == (M,):        # shared → 广播到每个通道
        rho = np.repeat(rho[:, None], Cc, axis=1).reshape(-1)
        theta = np.repeat(theta[:, None], Cc, axis=1).reshape(-1)
    return rho, theta


def make_kernel(rho, theta, L):
    tau = np.arange(L, dtype=np.float64)
    return (rho[:, None] ** tau[None, :]) * np.sin((tau[None, :] + 1.0) * theta[:, None])


def make_params(cfg, seed):
    r = np.random.default_rng(seed + 31337)
    def W(a, b, s=0.15):
        return T(r.standard_normal((a, b)) * s)
    Hh = cfg["M"] * cfg["C"]
    P = {"E": T(r.standard_normal((SP.VOCAB, SP.D_IN)) * 0.15)}
    if cfg["f"] == "product":
        P["Wa"] = W(SP.D_IN, Hh)
        P["Wb"] = W(SP.D_IN, Hh)
    else:
        P["Wf"] = W(SP.D_IN, Hh)
    P["Wq"] = W(SP.D_IN, Hh)
    P["Wout"] = W(Hh, SP.VOCAB)
    if cfg["learn"]:
        p0, q0, shape = init_pq(cfg, seed)
        P["p"] = T(p0)
        P["q"] = T(q0)
        P["_shape"] = shape
    return P


def n_params(P):
    return int(sum(v.d.size for k, v in P.items() if not k.startswith("_")))


# ---------------- 前向：含 K 的解析雅可比 ----------------
def forward(arm, P, X, Xp, L):
    cfg = ARMS[arm]
    B_, _, _ = X.shape
    M, Cc = cfg["M"], cfg["C"]
    N = M * Cc
    xx = const(X) @ P["E"]
    xp = const(Xp) @ P["E"]

    if cfg["f"] == "product":
        u = (xx @ P["Wa"]) * (xp @ P["Wb"])
    else:
        u = xx @ P["Wf"]
    u4 = u.reshape(B_, L, N, 1)

    if cfg["learn"]:
        p, q = P["p"].d, P["q"].d
        shape = P["_shape"]
        rho, theta = rho_theta(p, q, shape, M, Cc)
        K = make_kernel(rho, theta, L)
        Tm = S.toeplitz(K)

        # 自定义节点：K 对 (p,q) 的解析雅可比
        KT = T(K, [P["p"], P["q"]])
        tau = np.arange(L, dtype=np.float64)
        pow_tau = rho[:, None] ** tau[None, :]
        dpow = np.where(tau[None, :] > 0,
                        tau[None, :] * rho[:, None] ** np.maximum(tau[None, :] - 1, 0),
                        0.0)
        sin_t = np.sin((tau[None, :] + 1.0) * theta[:, None])
        cos_t = np.cos((tau[None, :] + 1.0) * theta[:, None])
        dK_drho = dpow * sin_t
        dK_dtheta = pow_tau * (tau[None, :] + 1.0) * cos_t

        def back_K():
            dK = KT.g
            drho = (dK * dK_drho).sum(axis=1)
            dtheta = (dK * dK_dtheta).sum(axis=1)
            if shape == (M,) and drho.size == N:    # shared：先按通道求和
                drho = drho.reshape(M, Cc).sum(axis=1)
                dtheta = dtheta.reshape(M, Cc).sum(axis=1)
            sp, sq = _sig(p), _sig(q)
            dp = drho * (1 - RHO_MIN) * sp * (1 - sp)
            dq = dtheta * np.pi * sq * (1 - sq)
            P["p"].g += dp
            P["q"].g += dq
        KT._b = back_K

        y = T(np.einsum("nls,bsnk->blnk", Tm, u4.d), [KT, u4])
        def back_y():
            dy = y.g[:, :, :, 0]                                   # (B,L,N)
            dK = np.zeros((N, L))
            for t_ in range(L):
                dK[:, t_] = np.einsum("btn,btn->n",
                                      dy[:, t_:, :], u4.d[:, :L - t_, :, 0])
            KT.g += dK
            u4.g[:, :, :, 0] += np.einsum("nts,btn->bsn", Tm, dy)
        y._b = back_y
        rho_out, theta_out = rho, theta
    else:
        rho0, theta0 = S.grids(M, cfg["grid"], L=L)
        K = make_kernel(rho0, theta0, L)
        Tm = S.toeplitz(K)
        y = T(np.einsum("nls,bsnk->blnk", Tm, u4.d), [u4]) if False else None
        # 固定臂：把 (M,) 核广播到 (M*C,) 通道
        Kb = np.repeat(K, Cc, axis=0)
        Tmb = S.toeplitz(Kb)
        y = T(np.einsum("nls,bsnk->blnk", Tmb, u4.d), [u4])
        def back_y():
            dy = y.g[:, :, :, 0]
            u4.g[:, :, :, 0] += np.einsum("nts,btn->bsn", Tmb, dy)
        y._b = back_y
        rho_out = np.repeat(rho0, Cc)
        theta_out = np.repeat(theta0, Cc)

    ylast = y[:, L - 1, :, 0]                       # (B,N)
    z = (xx[:, L - 1, :] @ P["Wq"]) * ylast
    return z @ P["Wout"], rho_out, theta_out


# ---------------- 守恒断言 ----------------
def guards(rho, theta, L, seed=0, verbose=False):
    """用最终 (ρ,θ) 重跑六条：证明学完仍是 LTI / Toeplitz / O(1) / 因果。"""
    g = S.guards(rho, theta, L, C=1, rng_seed=seed)
    # 时不变性：核只依赖 τ（由构造保证）；此处显式记录
    g["time_invariance"] = dict(
        passed=True,
        statement="(ρ,θ) 与 t 无关 ⇒ K 只依赖 τ ⇒ Toeplitz；数值可学 ≠ 时变",
        rho_min=float(rho.min()), rho_max=float(rho.max()),
        theta_min=float(theta.min()), theta_max=float(theta.max()))
    if verbose:
        for x in g["guards"]:
            print(f"    [{'PASS' if x['passed'] else 'FAIL'}] {x['name']}")
        print(f"    all_passed={g['all_passed']}  cond={g['cond_gram']:.3e}  "
              f"ρ∈[{rho.min():.4f},{rho.max():.4f}] θ∈[{theta.min():.4f},{theta.max():.4f}]")
    return g


# ---------------- 训练 / 评估 ----------------
def eval_acc(arm, P, X, Xp, Y):
    L = X.shape[1]
    lg, _, _ = forward(arm, P, X, Xp, L)
    return float((lg.d.argmax(-1) == Y).mean())


def run_train(arm, seed, steps=STEPS):
    cfg = ARMS[arm]
    P = make_params(cfg, seed)
    _, rho0, theta0 = forward(arm, P, *SP.build(SP.N_PAIRS, 2, seed=1)[:2],
                              SP.L_TRAIN)
    g0 = guards(rho0, theta0, SP.L_TRAIN, seed)
    Xtr, Xptr, Ytr, _ = SP.build(SP.N_PAIRS, 2000, seed=1000 + seed)
    Xte, Xpte, Yte, _ = SP.build(SP.N_PAIRS, 512, seed=9000 + seed)
    ADAM = {}
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = np.random.default_rng(seed * 100000 + step).choice(len(Ytr), SP.B, replace=False)
        X, Xp, Y = Xtr[idx], Xptr[idx], Ytr[idx]
        for k, v in P.items():
            if not k.startswith("_"):
                v.g[...] = 0.0
        lg, _, _ = forward(arm, P, X, Xp, SP.L_TRAIN)
        loss = nll_from_logits(lg, Y)
        loss.backward()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, v in P.items():
            if name.startswith("_"):
                continue
            key = "m_" + name
            if key not in ADAM:
                ADAM[key] = np.zeros_like(v.d)
                ADAM["v_" + name] = np.zeros_like(v.d)
            m, vv = ADAM[key], ADAM["v_" + name]
            m *= b1; m += (1 - b1) * v.g
            vv *= b2; vv += (1 - b2) * v.g * v.g
            mh = m / (1 - b1 ** step)
            vh = vv / (1 - b2 ** step)
            v.d -= SP.LR * mh / (np.sqrt(vh) + eps)
        if step % 500 == 0 or step == 1:
            curve.append(dict(step=step, loss=float(loss.d),
                              acc=eval_acc(arm, P, Xte, Xpte, Yte)))

    _, rho1, theta1 = forward(arm, P, Xte[:2], Xpte[:2], SP.L_TRAIN)
    g1 = guards(rho1, theta1, SP.L_TRAIN, seed)
    acc = eval_acc(arm, P, Xte, Xpte, Yte)
    Xsh, Xpsh, Ysh, _ = SP.build(SP.N_PAIRS, 512, seed=9000 + seed, shuffle_pairs=True)
    sh = eval_acc(arm, P, Xsh, Xpsh, Ysh)
    gen = {}
    for n in (4, 8, 12, 16):
        Xg, Xpg, Yg, _ = SP.build(n, 256, seed=7000 + seed + n)
        gen[f"L{2*n+1}"] = eval_acc(arm, P, Xg, Xpg, Yg)

    delta = (float(np.abs(rho1 - rho0).max()), float(np.abs(theta1 - theta0).max()))
    return dict(arm=arm, seed=seed, params=n_params(P),
                guards_before=g0, guards_after=g1,
                guards_all_passed=bool(g0["all_passed"] and g1["all_passed"]),
                rho_theta_changed=(delta[0] > 0 or delta[1] > 0),
                max_delta_rho=delta[0], max_delta_theta=delta[1],
                cond_before=g0["cond_gram"], cond_after=g1["cond_gram"],
                final_acc=acc, shuffle_acc=sh, length_generalization=gen,
                curve=curve, elapsed_s=round(time.time() - t0, 1))


def gradcheck(arm, seed=0):
    cfg = ARMS[arm]
    P = make_params(cfg, seed)
    X, Xp, Y, _ = SP.build(SP.N_PAIRS, 4, seed=42)
    for k, v in P.items():
        if not k.startswith("_"):
            v.g[...] = 0.0
    lg, _, _ = forward(arm, P, X, Xp, SP.L_TRAIN)
    nll_from_logits(lg, Y).backward()
    analytic = {k: v.g.ravel().copy() for k, v in P.items()
                if not k.startswith("_")}
    r = np.random.default_rng(321)
    eps = 1e-6
    worst, details = 0.0, []
    names = [k for k in P if not k.startswith("_")]
    for name in names:
        v = P[name]; flat = v.d.ravel(); gf = analytic[name]
        for i in r.choice(flat.size, size=min(3, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = nll_from_logits(forward(arm, P, X, Xp, SP.L_TRAIN)[0], Y).d
            flat[i] = old - eps
            lm = nll_from_logits(forward(arm, P, X, Xp, SP.L_TRAIN)[0], Y).d
            flat[i] = old
            num = float((lp - lm) / (2 * eps)); ana = float(gf[i])
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
            ok = rel < 5e-4 or abs(num - ana) < 1e-8
            worst = max(worst, 0.0 if ok else rel)
            details.append(dict(param=name, numerical=num, analytic=ana,
                                rel=rel, passed=bool(ok)))
    return dict(arm=arm, worst_rel=worst,
                passed=all(d["passed"] for d in details), details=details)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="guards",
                    choices=["guards", "gradcheck", "all"])
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.mode == "guards":
        out = {}
        for arm, cfg in ARMS.items():
            P = make_params(cfg, 0)
            _, rho, theta = forward(arm, P, *SP.build(SP.N_PAIRS, 2, seed=1)[:2],
                                    SP.L_TRAIN)
            out[arm] = guards(rho, theta, SP.L_TRAIN, verbose=True)
            print(f"  {arm:16s} all={out[arm]['all_passed']} "
                  f"state_bytes={out[arm]['state_bytes']}")
        with open(f"{OUT}/native_guards.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        return

    if a.mode == "gradcheck":
        out = [gradcheck(arm) for arm in ARMS]
        for g in out:
            print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['arm']:16s} "
                  f"worst={g['worst_rel']:.2e}")
        with open(f"{OUT}/native_gradcheck.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"==> {sum(g['passed'] for g in out)}/{len(out)} PASS")
        return

    if a.mode == "all":
        fn = f"{OUT}/native_results.json"
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
                      f"guards={r['guards_all_passed']} "
                      f"Δρ={r['max_delta_rho']:.3f} {r['elapsed_s']}s", flush=True)
                with open(fn, "w") as f:
                    json.dump(allres, f, indent=1, ensure_ascii=False)
                    f.flush()
        print("\n写入", fn)
        return


if __name__ == "__main__":
    main()
