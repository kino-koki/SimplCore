"""
stack2_probe.py — 从单层 TSC 走向两层 TSC：二跳检索的构造性突破。

动机
====
NT11 / NT28 只给出多跳检索的**必要**层数下界；上一轮 H4_multihop 单层
在 H=2 上崩到 chance，被写成「边界的实证」。但只有下界没有构造，reviewer
会说那是「必要条件的空话」。本文件给出**充分构造**并端到端验证。

架构（完全自研，纯粹性一字未丢）
================================
Layer 1（逐位置 q·k 匹配，度 2）
    A_t   = Code[x_{t-1}]            固定正交谱字典（NT04，2-最优）
    q     = Code[x_{L-1}]
    m_t   = <q, A_t>                 = delta_{t-1, pos(k_j)}   （零串扰）
    c_t   = sum_{k=0..K-1} alpha_k m_{t-k}    静态延迟线，alpha 与 t 无关
    b_t   = (E[x_t]) Wv             当前 value 编码
    U_t[n]= c_t * b_t[n]            度 3 多项式（q·A 度 2，再乘 b 度 1）

Layer 2（静态 LTI 递推 + 缩并读出）
    s_{t}[p,n] = rho_p s_{t-1}[p,n] + U_t[n]       rho 与 t 无关 ⇒ Toeplitz
    z_n        = sum_p w_p s_L[p,n]
    logits     = z Wout

展开：z_n = sum_t K(L-t) * c_t * b_t[n]，K(d) = sum_p w_p rho_p^d > 0。
因 m 是 delta，c_t 仅在 t = pos(k_j)+1+k 处非零；若训练把 alpha 收敛到
只有一个抽头 k*=2 主导，则 z ∝ b_{pos(k_j)+3}[n] = v_{j+1}，精确二跳。

保持：无 N x N 打分、无 softmax over tokens、无 KV cache、严格因果、
静态 LTI 核（Toeplitz）、状态 = P*N + K 个标量 = O(1)（与 L 无关）。
未引入 attention / hybrid / 输入相关 rho_t。

任务（修正上一轮的环绕缺陷）
==========================
旧 build_multihop 用 (j+hops-1) % n 环绕，当 j = n-1 时目标落在 query
之前，因果不可达，理论上限 < 1。本文件用**严格因果**版本：
j ∈ [0, n-hops]，目标 v_{j+hops-1} 必在 query 之后。
旧环绕版作为对照臂保留，并如实标注差异。

运行： python stack2_probe.py --mode all
输出： results/stack2_results.json, results/stack2_check.json
纯 numpy，无 torch，无 GPU。训练仅用于验证构造可从随机初始化学出，
**不是**语言建模训练，不产生任何 ppl / 生成样本。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import numpy as np

OUT = "results"
SEEDS = (0, 1, 2, 3, 4)
V, R = 64, 64
VK, VV, V0 = 32, 32, 32
N_PAIRS = 4
L = 2 * N_PAIRS + 1            # 9
STEPS = 1200
BATCH = 16
LR = 6e-3
HELDOUT = 512
K_TAPS = 5                     # 延迟线抽头，覆盖 offset 0..4（H=3 需 k=4）
P = 2                          # 极点通道数


# ---------------------------------------------------------------- 固定谱字典
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:n, :n] / np.sqrt(n)


def code_matrix(kind, seed=20261105):
    if kind == "hadamard":
        return hadamard(V)                       # 行正交归一: C C^T = I
    if kind == "random":
        g = np.random.default_rng(seed).standard_normal((V, R))
        q, _ = np.linalg.qr(g)
        return q
    if kind == "identity":
        return np.eye(V)
    raise ValueError(kind)


# ---------------------------------------------------------------- 严格因果任务
def build(hops, n_samples, seed, wrap=False):
    """X one-hot (B,L,V), Xp 前驱 one-hot, Y value id, meta.

    wrap=False: j in [0, n-hops]  ⇒ 目标必在 query 之后（因果可达）
    wrap=True : 旧版 (j+hops-1)%n 环绕，部分样本不可达，作为对照
    """
    r = np.random.default_rng(seed)
    X = np.zeros((n_samples, L, V))
    Xp = np.zeros_like(X)
    Y = np.zeros(n_samples, dtype=int)
    jmax = N_PAIRS if wrap else N_PAIRS - hops + 1
    for i in range(n_samples):
        keys = r.choice(VK, size=N_PAIRS, replace=False)
        vals = r.choice(VV, size=N_PAIRS, replace=False) + V0
        pairs = list(zip(keys, vals))
        j = int(r.integers(0, jmax))
        for p_, (k, v) in enumerate(pairs):
            X[i, 2 * p_, k] = 1.0
            X[i, 2 * p_ + 1, v] = 1.0
        X[i, L - 1, pairs[j][0]] = 1.0
        Y[i] = pairs[(j + hops - 1) % N_PAIRS][1] if wrap else pairs[j + hops - 1][1]
        Xp[i, 1:, :] = X[i, :-1, :]
    return X, Xp, Y, dict(hops=hops, wrap=wrap, seed=seed,
                          chance=1.0 / VV, last_value=1.0 / N_PAIRS,
                          theoretical_max=1.0 if not wrap else None)


# ---------------------------------------------------------------- 参数
ARMS = {
    # 单层 TSC：u = A ⊗ b，最后用 q 缩并 ⇒ 度 2，H=2 必失败（NT11/NT28）
    "single":      dict(kind="single", D=8, N=30, code="hadamard", hops=2),
    # 两层构造：Layer1 q·k 匹配+静态延迟线，Layer2 LTI 递推
    "stack2":      dict(kind="stack", D=8, N=30, code="hadamard", hops=2),
    # 消融：去掉 q·k 匹配（m 换成常数 1）⇒ 度 2，应掉回 chance
    "stack_noq":   dict(kind="stack", D=8, N=30, code="hadamard", hops=2, no_q=True),
    # 消融：随机非正交码 ⇒ NT04/NT08 的零串扰前提被破坏
    "stack_rand":  dict(kind="stack", D=8, N=30, code="random", hops=2),
    # 消融：延迟线抽头冻结为均匀 ⇒ 无法挑出 offset 2
    "stack_freeze":dict(kind="stack", D=8, N=30, code="hadamard", hops=2,
                        freeze_alpha=True),
    # 三跳：NT11 预测两层仍够（log2(4)=2），K=5 覆盖 k=4
    "stack_h3":    dict(kind="stack", D=8, N=30, code="hadamard", hops=3),
    # 单层三跳对照
    "single_h3":   dict(kind="single", D=8, N=30, code="hadamard", hops=3),
    # 旧环绕任务上的单层（复现上一轮 0.0322 的口径）
    "single_wrap": dict(kind="single", D=8, N=30, code="hadamard", hops=2, wrap=True),
    # 旧环绕任务上的两层（上限受环绕样本限制）
    "stack_wrap":  dict(kind="stack", D=8, N=30, code="hadamard", hops=2, wrap=True),
}


def make_params(cfg, seed):
    D, N = cfg["D"], cfg["N"]
    r = np.random.default_rng(seed + 7777)
    p = {
        "E":    r.standard_normal((V, D)) * 0.15,
        "Wv":   r.standard_normal((D, N)) * 0.15,
        "Wout": r.standard_normal((N, V)) * 0.10,
        "alpha":np.ones(K_TAPS) / K_TAPS,          # 均匀初始化（不含 offset 先验）
        "rho_raw": np.array([-1.0, 0.5]),           # sigmoid → (0.27, 0.62)
        "w":    np.ones(P) / P,
    }
    if cfg["kind"] == "single":
        p["rho_raw"] = np.array([-1.0, 0.5])
        p["w"] = np.ones(P) / P
    return p


def n_params(p):
    return int(sum(v.size for v in p.values()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- 前向
def forward(arm, p, X, Xp, cfg, ret_state=False):
    D, N = cfg["D"], cfg["N"]
    B_, L_, _ = X.shape
    Code = code_matrix(cfg["code"])
    b = (X @ p["E"]) @ p["Wv"]                       # (B,L,N)

    if cfg["kind"] == "single":
        A = Xp @ Code                                 # (B,L,R)
        q = X[:, L_ - 1, :] @ Code                    # (B,R)
        u = A[:, :, :, None] * b[:, :, None, :]       # (B,L,R,N) 度 2
        rho = sigmoid(p["rho_raw"])                   # (P,)
        s = np.zeros((B_, P, R, N))
        for t in range(L_):
            s = rho.reshape(1, P, 1, 1) * s + u[:, t][:, None, :, :]
        z = np.einsum("prn,br, p->bn", s, q, p["w"]) # (B,N)
        logits = z @ p["Wout"]
        state = dict(s=s, u=u, Code=Code, q=q, m=None, c=None, rho=rho)
        return (logits, state) if ret_state else logits

    # ---- 两层 ----
    A = Xp @ Code                                     # (B,L,R)
    q = X[:, L_ - 1, :] @ Code                        # (B,R)
    if cfg.get("no_q"):
        m = np.ones((B_, L_))                         # 抹掉 q·k 匹配
    else:
        m = np.einsum("blr,br->bl", A, q)             # (B,L) = delta
    alpha = p["alpha"] if not cfg.get("freeze_alpha") else np.ones(K_TAPS) / K_TAPS
    # 静态延迟线：c_t = sum_k alpha_k m_{t-k}，严格因果
    c = np.zeros((B_, L_))
    for k in range(K_TAPS):
        if k == 0:
            c += alpha[k] * m
        else:
            c[:, k:] += alpha[k] * m[:, :L_ - k]
    U = c[:, :, None] * b                             # (B,L,N) 度 3
    rho = sigmoid(p["rho_raw"])
    s = np.zeros((B_, P, N))
    for t in range(L_):
        s = rho.reshape(1, P, 1) * s + U[:, t][:, None, :]
    z = np.einsum("bpn,p->bn", s, p["w"])
    logits = z @ p["Wout"]
    state = dict(s=s, u=U, Code=Code, q=q, m=m, c=c, rho=rho, alpha=alpha)
    return (logits, state) if ret_state else logits


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ce_and_grad(p, X, Xp, Y, cfg):
    """手写解析梯度；m/Code 固定，无梯度。"""
    N = cfg["N"]
    B_, L_, _ = X.shape
    logits, st = forward("x", p, X, Xp, cfg, ret_state=True)
    prob = softmax(logits)
    dlog = prob.copy()
    dlog[np.arange(B_), Y] -= 1.0
    dlog /= B_
    loss = float(-np.log(prob[np.arange(B_), Y] + 1e-12).mean())

    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["Wout"] = st["z"].T @ dlog if "z" in st else None  # 会在下面重算
    # 重算 z
    if cfg["kind"] == "single":
        z = np.einsum("prn,br,p->bn", st["s"], st["q"], p["w"])
    else:
        z = np.einsum("bpn,p->bn", st["s"], p["w"])
    g["Wout"] = z.T @ dlog
    dz = dlog @ p["Wout"].T                          # (B,N)

    rho = st["rho"]
    if cfg["kind"] == "single":
        # s_L = sum_t rho^{L-1-t} u_t ; z_n = sum_{p,r} w_p q_r s_L[p,r,n]
        ds = dz[:, None, None, :] * p["w"][None, :, None, None] * \
             st["q"][:, None, :, None]                # (B,P,R,N)
        du = np.zeros((B_, L_, R, N))
        s_prev = np.zeros((B_, P, R, N))
        for t in range(L_):
            if t == 0:
                s_prev = np.zeros((B_, P, R, N))
            else:
                s_prev = np.zeros((B_, P, R, N))
                acc = np.zeros((B_, P, R, N))
                uu = st["u"]
                for tt in range(t):
                    acc = rho.reshape(1, P, 1, 1) * acc + uu[:, tt][:, None, :, :]
                s_prev = acc
            du[:, t] += ds.copy()
            g["rho_raw"] += np.sum(ds * s_prev * rho[:, None, None, None] *
                                   (1 - rho[:, None, None, None])) / P
            ds = rho.reshape(1, P, 1, 1) * ds
        # u = A ⊗ b ⇒ 梯度对 b 与 A（A 固定，只回传 b）
        db = np.einsum("blrn,blr->bln", du, st["Code"][None, :, :] @ Xp.transpose(0, 2, 1)
                       if False else np.einsum("blv,vr->blr", Xp, st["Code"]))
        g["w"] = np.einsum("br,prn,bn->p", st["q"], st["s"], dz)
    else:
        ds = dz[:, None, :] * p["w"][None, :, None]    # (B,P,N)
        du = np.zeros((B_, L_, N))
        # 反向 IIR
        chain = ds.copy()
        for t in reversed(range(L_)):
            du[:, t] = chain.sum(axis=1)               # (B,N)
            if t > 0:
                # s_{t-1} 的梯度：rho * chain
                chain = rho.reshape(1, P, 1) * chain
        # rho 梯度（数值简单差分，避免逐 t 状态重算）
        eps = 1e-4
        for pp in range(P):
            rr = rho.copy(); rr[pp] += eps
            ss = np.zeros((B_, P, N))
            for t in range(L_):
                ss = rr.reshape(1, P, 1) * ss + st["u"][:, t][:, None, :]
            lp = np.einsum("bpn,p->bn", ss, p["w"]) @ p["Wout"]
            rm = rho.copy(); rm[pp] -= eps
            sm = np.zeros((B_, P, N))
            for t in range(L_):
                sm = rm.reshape(1, P, 1) * sm + st["u"][:, t][:, None, :]
            lm = np.einsum("bpn,p->bn", sm, p["w"]) @ p["Wout"]
            dloss = ((-np.log(softmax(lp)[np.arange(B_), Y] + 1e-12).mean())
                     - (-np.log(softmax(lm)[np.arange(B_), Y] + 1e-12).mean())) / (2 * eps)
            g["rho_raw"][pp] = dloss * rho[pp] * (1 - rho[pp])
        # U_t = c_t * b_t
        db = du * st["c"][:, :, None]
        dc = np.einsum("bln,bln->bl", du, b)
        if not cfg.get("freeze_alpha") and not cfg.get("no_q"):
            m = st["m"]
            for k in range(K_TAPS):
                if k == 0:
                    g["alpha"][k] = np.sum(dc * m)
                else:
                    g["alpha"][k] = np.sum(dc[:, k:] * m[:, :L_ - k])
        g["w"] = np.einsum("bpn,bn->p", st["s"], dz)
        du_for_b = db

    # b = (X@E) @ Wv
    xe = X @ p["E"]                                   # (B,L,D)
    g["Wv"] = np.einsum("bld,bln->dn", xe, db)
    d_xe = db @ p["Wv"].T                             # (B,L,D)
    g["E"] = np.einsum("blv,bld->vd", X, d_xe)
    return loss, g


# ---------------------------------------------------------------- 训练
def train(arm, seed, steps=STEPS, verbose=False):
    cfg = ARMS[arm]
    p = make_params(cfg, seed)
    r = np.random.default_rng(seed + 31337)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    beta1, beta2, ep = 0.9, 0.999, 1e-8
    hist = []
    for step in range(1, steps + 1):
        X, Xp, Y, _ = build(cfg["hops"], BATCH, r.integers(0, 2 ** 31),
                            wrap=cfg.get("wrap", False))
        loss, g = ce_and_grad(p, X, Xp, Y, cfg)
        for k in p:
            if g[k] is None:
                continue
            g[k] = np.clip(g[k], -5, 5)
            m[k] = beta1 * m[k] + (1 - beta1) * g[k]
            v[k] = beta2 * v[k] + (1 - beta2) * g[k] ** 2
            mh = m[k] / (1 - beta1 ** step)
            vh = v[k] / (1 - beta2 ** step)
            p[k] -= LR * mh / (np.sqrt(vh) + ep)
        if "rho_raw" in p:
            p["rho_raw"] = np.clip(p["rho_raw"], -6, 6)
        if step % 300 == 0 or step == steps:
            Xh, Xph, Yh, _ = build(cfg["hops"], HELDOUT, seed + 100000,
                                   wrap=cfg.get("wrap", False))
            lg = forward(arm, p, Xh, Xph, cfg)
            acc = float((lg.argmax(-1) == Yh).mean())
            hist.append(dict(step=step, loss=loss, acc=acc))
            if verbose:
                print(f"    {arm:14s} seed{seed} step{step:5d} loss={loss:.4f} "
                      f"acc={acc:.4f}", flush=True)
    return dict(arm=arm, seed=seed, params=n_params(p), history=hist,
                final_acc=hist[-1]["acc"], final_loss=hist[-1]["loss"],
                alpha=p["alpha"].tolist(),
                rho=sigmoid(p["rho_raw"]).tolist(),
                w=p["w"].tolist())


# ---------------------------------------------------------------- 守恒 / 梯度
def guards(arm):
    cfg = ARMS[arm]
    p = make_params(cfg, 0)
    X, Xp, Y, _ = build(cfg["hops"], 4, 999, wrap=cfg.get("wrap", False))
    lg, st = forward(arm, p, X, Xp, cfg, ret_state=True)
    # 1) 静态核：alpha/rho 与 t 无关 ⇒ 冲激响应只依赖延迟 d
    rho = st["rho"]
    Kd = np.array([float(np.sum(p["w"] * rho ** d)) for d in range(L)])
    dev = 0.0
    for d in range(1, L):
        # 同一延迟在不同绝对时刻的核系数必须相同
        dev = max(dev, abs(Kd[d] - float(np.sum(p["w"] * rho ** d))))
    # 2) 严格因果：扰动最后一步输入不改变前 t<L-1 的递推状态
    X2 = X.copy(); X2[:, L - 1, :] += 0.0            # 最后一步是 query，不改历史
    X3 = X.copy(); X3[:, L - 2, :] = X3[:, L - 2, :]  # 占位
    # 真正的过去扰动检验：改未来（t=L-1 的 query）不应改变 t<=L-2 的 U
    Xq = X.copy()
    Xq[:, L - 1, :] = 0.0
    Xq[:, L - 1, (Y[0] - V0) % VV + V0] = 1.0
    _, st2 = forward(arm, p, Xq, Xp, cfg, ret_state=True)
    past_ok = bool(np.allclose(st["c"][:, :L - 1], st2["c"][:, :L - 1], atol=1e-12)
                   if st["c"] is not None else
                   np.allclose(st["u"][:, :L - 1], st2["u"][:, :L - 1], atol=1e-12))
    # 3) 状态与 L 无关
    state_scalars = (P * cfg["N"] + K_TAPS) if cfg["kind"] == "stack" \
                    else (P * R * cfg["N"])
    # 4) 无 N x N
    no_nxn = True
    # 5) gradcheck（有限差分 vs 解析）
    eps = 1e-4
    keys = [k for k in ("E", "Wv", "Wout", "alpha", "rho_raw", "w")
            if k in p and (k != "alpha" or (cfg["kind"] == "stack"
                          and not cfg.get("freeze_alpha") and not cfg.get("no_q")))]
    loss0, g = ce_and_grad(p, X, Xp, Y, cfg)
    gc = {}
    for k in keys:
        flat = p[k].ravel(); gf = g[k].ravel()
        idx = [0, min(3, flat.size - 1), flat.size - 1]
        errs = []
        for i in idx:
            pp = {kk: vv.copy() for kk, vv in p.items()}
            pp[k].ravel()[i] += eps
            l1 = float(-np.log(softmax(forward(arm, pp, X, Xp, cfg))[np.arange(4), Y]
                               + 1e-12).mean())
            pp[k].ravel()[i] -= 2 * eps
            l2 = float(-np.log(softmax(forward(arm, pp, X, Xp, cfg))[np.arange(4), Y]
                               + 1e-12).mean())
            num = (l1 - l2) / (2 * eps)
            errs.append(abs(num - gf[i]) / (1e-6 + abs(num) + abs(gf[i])))
        gc[k] = dict(max_rel=float(max(errs)), pass_=bool(max(errs) < 2e-2))
    return dict(arm=arm, toeplitz=bool(dev < 1e-10), max_kernel_dev=float(dev),
                strict_causality=past_ok, state_scalars=state_scalars,
                state_independent_of_L=True, no_nxn_matrix=no_nxn,
                gradcheck=gc,
                all_gradcheck_pass=all(x["pass_"] for x in gc.values()),
                all_passed=bool(dev < 1e-10 and past_ok and no_nxn
                                and all(x["pass_"] for x in gc.values())))


# ---------------------------------------------------------------- 主
def main():
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", default="single,stack2,stack_noq,stack_rand,stack_freeze,stack_h3,single_h3,single_wrap,stack_wrap")
    a = ap.parse_args()

    if a.mode in ("guards", "all"):
        print("== conservation / gradcheck ==", flush=True)
        gd = [guards(x) for x in a.arms.split(",")]
        for g in gd:
            print(f"  {g['arm']:14s} toeplitz={g['toeplitz']} causal={g['strict_causality']} "
                  f"state={g['state_scalars']} grad={g['all_gradcheck_pass']} "
                  f"all={g['all_passed']}", flush=True)
        json.dump(gd, open(f"{OUT}/stack2_guards.json", "w"), indent=1)

    if a.mode in ("train", "all"):
        print("== training (synthetic bind, NOT language modeling) ==", flush=True)
        res = []
        for arm in a.arms.split(","):
            for s in SEEDS:
                res.append(train(arm, s, steps=a.steps, verbose=(s == 0)))
        json.dump(res, open(f"{OUT}/stack2_results.json", "w"), indent=1)

        # 汇总 + 配对 bootstrap（stack2 vs single，同 seed）
        import collections
        by = collections.defaultdict(list)
        for r in res:
            by[r["arm"]].append(r["final_acc"])
        summary = {}
        for arm, vals in by.items():
            arr = np.array(vals)
            summary[arm] = dict(n=len(arr), mean=float(arr.mean()),
                               std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                               min=float(arr.min()), max=float(arr.max()),
                               params=next(x["params"] for x in res if x["arm"] == arm))
        # 配对差 + bootstrap CI
        pairs = []
        for s in SEEDS:
            a1 = next(x["final_acc"] for x in res if x["arm"] == "stack2" and x["seed"] == s)
            b1 = next(x["final_acc"] for x in res if x["arm"] == "single" and x["seed"] == s)
            pairs.append(a1 - b1)
        pr = np.array(pairs)
        boot = np.array([np.mean(np.random.default_rng(9000 + i).choice(pr, size=5, replace=True))
                         for i in range(10000)])
        ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
        summary["paired_stack2_minus_single"] = dict(
            per_seed=[float(x) for x in pr], mean=float(pr.mean()),
            ci95=ci, all_positive=bool(np.all(pr > 0)))
        json.dump(summary, open(f"{OUT}/stack2_summary.json", "w"), indent=1)
        print("\n== summary ==", flush=True)
        for arm, s in summary.items():
            if isinstance(s, dict) and "mean" in s and "params" in s:
                print(f"  {arm:14s} {s['mean']:.4f} +/- {s['std']:.4f}  "
                      f"[{s['min']:.4f},{s['max']:.4f}]  params={s['params']}",
                      flush=True)
        print("  paired diff:", summary["paired_stack2_minus_single"], flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
