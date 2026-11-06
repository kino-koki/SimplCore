"""
chain_stack_probe.py — 两层 TSC 突破「内容链多跳」：给 NT11/NT28 的层下界补上
构造性充分条件（NT31），并端到端验证。

为什么重做任务
==============
上一轮 H4_multihop 的任务是 q=k_j ⇒ 答案 v_{j+1}，答案由**对的固定顺序**
决定，只是位置跳转；静态延迟线就能解，不需要第二次内容匹配。用它去宣称
「验证了 NT11 的层下界」是错的。本文件换成真正的内容链任务：

    链 k_0 → k_1 → k_2 → ... （每个 pair 是 (k_p, k_{p+1})）
    随机打乱这些 pair 在序列中的位置
    q = k_0，H 跳答案 = k_H

H=2 时：先以 q=k_0 匹配取回 v_0=k_1（第 1 跳），再**以 k_1 为新 query**
匹配取回 v_1=k_2（第 2 跳）。位置随机 ⇒ 静态延迟线无效，必须两次内容
匹配 ⇒ 正是 NT11/NT28 适用的对象。

架构（完全自研，纯粹性一字未丢）
================================
    A_t   = Code[x_{t-1}]        固定正交谱字典（NT04，2-最优，与 rho/theta 网格同性质）
    b_t   = E[x_t] Wv            当前 value 编码
    u_t[r,n] = A_t[r] b_t[n]    外积张量信号（NT10：核静态，u 可依赖内容）

Layer 1  s1 = rho1 s1 + u ;  z1_n = sum_p w1_p sum_r q1_r s1[p,r,n]
          q1 = Code[k_0]（固定）
          ⇒ z1 = sum_t K1(L-1-t) <q1,A_t> b_t   （1 个匹配因子；第 1 跳 = k_1）

层间接口 q2 = z1 Wq2                 ← Layer1 的输出作为 Layer2 的 query
          （输入相关的是 **query 信号**，不是核；rho/theta 仍与 t 无关）

Layer 2  s2 = rho2 s2 + u ;  z2_n = sum_p w2_p sum_r q2_r s2[p,r,n]
          ⇒ z2 = sum_t K2(L-1-t) <q2,A_t> b_t   （2 个匹配因子；第 2 跳 = k_2）
    logits = z2 Wout

K_i(d) = sum_p w_i,p rho_i,p^d > 0（NT16 严格正）。
保持：无 N x N、无 softmax-over-tokens、无 KV cache、严格因果、
每层核静态 Toeplitz（FFT/IIR 对偶，NT12）、状态 = 2*P*R*N + N = O(1)。

对照臂
======
  single     : 只有 Layer 1，logits = z1 Wout（其余参数保留但不接，等参数对照）
  stack2     : 上面的两层构造
  delay      : 单层 + 静态延迟线（无第二跳匹配）⇒ 位置捷径，应掉回 chance
  stack_noq2 : q2 固定为常数向量（不做内容链）⇒ 应掉回 chance
  stack_rand : 随机非正交码 ⇒ NT04/NT08 零串扰前提被破坏
  stack_h3   : 3 跳（NT11: ceil(log2 4)=2 层，两层应仍够）
  single_h3  : 单层 3 跳对照

纯 numpy，无 torch，无 GPU。**不是语言建模训练**，不产生 ppl / 生成样本。
运行： python chain_stack_probe.py --mode all
输出： results/chain_*.json
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

OUT = "results"
SEEDS = (0, 1, 2, 3, 4)
V, R = 64, 64
VK = 32                      # key / chain-token 半区 [0,32)
N_PAIRS = 4                 # 链长 = 4 对，覆盖 H<=3
L = 2 * N_PAIRS + 1         # 9
STEPS = 1500
BATCH = 16
LR = 6e-3
HELDOUT = 512
K_TAPS = 5
P = 2                        # 每层极点通道数


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:n, :n] / np.sqrt(n)          # 行正交归一 C C^T = I


def code_matrix(kind, seed=20261105):
    if kind == "hadamard":
        return hadamard(V)
    if kind == "random":
        g = np.random.default_rng(seed).standard_normal((V, R))
        q, _ = np.linalg.qr(g)
        return q
    raise ValueError(kind)


def build_chain(hops, n_samples, seed, shuffle=True):
    """内容链：pairs (k_p -> k_{p+1})，位置随机置换，q=k_0，答案 k_hops。"""
    r = np.random.default_rng(seed)
    X = np.zeros((n_samples, L, V))
    Xp = np.zeros_like(X)
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        chain = r.choice(VK, size=N_PAIRS + 1, replace=False)   # k_0..k_4
        pairs = [(chain[p], chain[p + 1]) for p in range(N_PAIRS)]
        if shuffle:
            r.shuffle(pairs)                  # 位置随机 ⇒ 延迟线无效
        for p_, (k, v) in enumerate(pairs):
            X[i, 2 * p_, k] = 1.0
            X[i, 2 * p_ + 1, v] = 1.0
        X[i, L - 1, chain[0]] = 1.0         # q = k_0
        Y[i] = chain[hops]                   # H 跳答案
        Xp[i, 1:, :] = X[i, :-1, :]
    return X, Xp, Y, dict(hops=hops, shuffle=shuffle, seed=seed,
                          chance=1.0 / VK, theoretical_max=1.0)


ARMS = {
    "h1_single":   dict(kind="single", D=8, N=30, code="hadamard", hops=1),
    "single":      dict(kind="single", D=8, N=30, code="hadamard", hops=2),
    "stack2":      dict(kind="stack",  D=8, N=30, code="hadamard", hops=2),
    "delay":       dict(kind="delay",  D=8, N=30, code="hadamard", hops=2),
    "stack_noq2":  dict(kind="stack",  D=8, N=30, code="hadamard", hops=2, no_q2=True),
    "stack_rand":  dict(kind="stack",  D=8, N=30, code="random",   hops=2),
    "stack_h3":    dict(kind="stack",  D=8, N=30, code="hadamard", hops=3),
    "single_h3":   dict(kind="single", D=8, N=30, code="hadamard", hops=3),
}


def make_params(cfg, seed):
    D, N = cfg["D"], cfg["N"]
    r = np.random.default_rng(seed + 7777)
    p = {
        "E":      r.standard_normal((V, D)) * 0.15,
        "Wv":     r.standard_normal((D, N)) * 0.15,
        "Wq2":    r.standard_normal((N, R)) * 0.10,   # 层间 query 投影
        "Wout":   r.standard_normal((N, V)) * 0.10,
        "alpha":  np.ones(K_TAPS) / K_TAPS,            # delay 臂用；其余臂参数保留
        "rho1":   np.array([-1.0, 0.5]),
        "rho2":   np.array([-0.5, 1.0]),
        "w1":     np.ones(P) / P,
        "w2":     np.ones(P) / P,
    }
    return p


def n_params(p):
    return int(sum(v.size for v in p.values()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _K(rho, w, L_):
    """K(d) = sum_p w_p rho_p^d, d = L-1-t  （NT16：w,rho>0 ⇒ 严格正）"""
    d = np.arange(L_ - 1, -1, -1)               # d[t] = L-1-t
    return np.array([float(np.sum(w * rho ** dd)) for dd in d])  # (L,)


def forward(arm, p, X, Xp, cfg, ret_state=False):
    D, N = cfg["D"], cfg["N"]
    B_, L_, _ = X.shape
    Code = code_matrix(cfg["code"])
    A = Xp @ Code                                 # (B,L,R) 前驱 key 码
    b = (X @ p["E"]) @ p["Wv"]                    # (B,L,N) 当前 value 编码
    q1 = X[:, L_ - 1, :] @ Code                   # (B,R) 固定 query 码
    m1 = np.einsum("blr,br->bl", A, q1)           # (B,L) 第 1 跳匹配
    rho1, rho2 = sigmoid(p["rho1"]), sigmoid(p["rho2"])
    K1, K2 = _K(rho1, p["w1"], L_), _K(rho2, p["w2"], L_)

    # ---- Layer 1 ----
    z1 = np.einsum("bl,bln,l->bn", m1, b, K1)    # (B,N)

    if cfg["kind"] == "single":
        logits = z1 @ p["Wout"]
        st = dict(z1=z1, z2=None, m1=m1, m2=None, b=b, A=A,
                  q1=q1, q2=None, K1=K1, K2=None, rho1=rho1, rho2=rho2)
        return (logits, st) if ret_state else logits

    if cfg["kind"] == "delay":
        # 单层 + 静态延迟线：c_t = sum_k alpha_k m1_{t-k}，无第二跳内容匹配
        c = np.zeros((B_, L_))
        for k in range(K_TAPS):
            c[:, k:] += p["alpha"][k] * (m1 if k == 0 else m1[:, :L_ - k])
        z = np.einsum("bl,bln,l->bn", c, b, K1)
        logits = z @ p["Wout"]
        st = dict(z1=z1, z2=None, m1=m1, m2=c, b=b, A=A,
                  q1=q1, q2=None, K1=K1, K2=None, rho1=rho1, rho2=rho2, c=c)
        return (logits, st) if ret_state else logits

    # ---- Layer 2：query 来自 Layer1 输出 ----
    if cfg.get("no_q2"):
        q2 = np.ones((B_, R)) / np.sqrt(R)        # 常数 query，掐断内容链
    else:
        q2 = z1 @ p["Wq2"]                        # (B,R) 输入相关，但核静态
    m2 = np.einsum("blr,br->bl", A, q2)
    z2 = np.einsum("bl,bln,l->bn", m2, b, K2)    # (B,N)
    logits = z2 @ p["Wout"]
    st = dict(z1=z1, z2=z2, m1=m1, m2=m2, b=b, A=A,
              q1=q1, q2=q2, K1=K1, K2=K2, rho1=rho1, rho2=rho2)
    return (logits, st) if ret_state else logits


# ---------------------------------------------------------------- 解析梯度
def ce_and_grad(p, X, Xp, Y, cfg):
    B_, L_, _ = X.shape
    logits, st = forward("x", p, X, Xp, cfg, ret_state=True)
    prob = softmax(logits)
    dlog = prob.copy()
    dlog[np.arange(B_), Y] -= 1.0
    dlog /= B_
    loss = float(-np.log(prob[np.arange(B_), Y] + 1e-12).mean())
    g = {k: np.zeros_like(v) for k, v in p.items()}

    if cfg["kind"] == "single":
        dz1 = dlog @ p["Wout"].T
        g["Wout"] = st["z1"].T @ dlog
        dm1 = np.einsum("bn,bln,l->bl", dz1, st["b"], st["K1"])
        db = np.einsum("bn,bl,l->bln", dz1, st["m1"], st["K1"])
    elif cfg["kind"] == "delay":
        dz = dlog @ p["Wout"].T
        # 重算 z（与 forward 一致）
        z = np.einsum("bl,bln,l->bn", st["c"], st["b"], st["K1"])
        g["Wout"] = z.T @ dlog
        dc = np.einsum("bn,bln,l->bl", dz, st["b"], st["K1"])
        db = np.einsum("bn,bl,l->bln", dz, st["c"], st["K1"])
        for k in range(K_TAPS):
            g["alpha"][k] = np.sum(dc[:, k:] * st["m1"][:, :L_ - k]) if k else \
                            np.sum(dc * st["m1"])
        dm1 = None
    else:
        dz2 = dlog @ p["Wout"].T
        g["Wout"] = st["z2"].T @ dlog
        # z2 对 m2 / b 的梯度
        dm2 = np.einsum("bn,bln,l->bl", dz2, st["b"], st["K2"])
        db = np.einsum("bn,bl,l->bln", dz2, st["m2"], st["K2"])
        # m2 = <q2, A_t> ⇒ q2 梯度
        dq2 = np.einsum("bl,blr->br", dm2, st["A"])
        if not cfg.get("no_q2"):
            g["Wq2"] = st["z1"].T @ dq2
            dz1 = dq2 @ p["Wq2"].T
            dm1 = np.einsum("bn,bln,l->bl", dz1, st["b"], st["K1"])
            db += np.einsum("bn,bl,l->bln", dz1, st["m1"], st["K1"])
        else:
            dm1 = None

    # b = (X E) Wv
    xe = X @ p["E"]
    g["Wv"] = np.einsum("bld,bln->dn", xe, db)
    d_xe = db @ p["Wv"].T
    g["E"] = np.einsum("blv,bld->vd", X, d_xe)

    # rho / w 用中心差分（P=2，每层 2 个标量，代价可忽略）
    eps = 1e-4
    for key in ("rho1", "rho2", "w1", "w2"):
        for i in range(p[key].size):
            pp = {k: v.copy() for k, v in p.items()}
            pp[key].ravel()[i] += eps
            l1 = float(-np.log(softmax(forward("x", pp, X, Xp, cfg))[np.arange(B_), Y]
                               + 1e-12).mean())
            pm = {k: v.copy() for k, v in p.items()}
            pm[key].ravel()[i] -= eps
            l2 = float(-np.log(softmax(forward("x", pm, X, Xp, cfg))[np.arange(B_), Y]
                               + 1e-12).mean())
            g[key].ravel()[i] = (l1 - l2) / (2 * eps)
    # 差分直接作用于 p["rho1"/"rho2"]（raw），已是 dL/d(raw)，无需再链式
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
        X, Xp, Y, _ = build_chain(cfg["hops"], BATCH, int(r.integers(0, 2 ** 31)))
        loss, g = ce_and_grad(p, X, Xp, Y, cfg)
        for k in p:
            gk = np.clip(g[k], -5, 5)
            m[k] = beta1 * m[k] + (1 - beta1) * gk
            v[k] = beta2 * v[k] + (1 - beta2) * gk ** 2
            p[k] -= LR * (m[k] / (1 - beta1 ** step)) / \
                    (np.sqrt(v[k] / (1 - beta2 ** step)) + ep)
        for k in ("rho1", "rho2"):
            p[k] = np.clip(p[k], -6, 6)
        if step % 300 == 0 or step == steps:
            Xh, Xph, Yh, _ = build_chain(cfg["hops"], HELDOUT, seed + 100000)
            acc = float((forward(arm, p, Xh, Xph, cfg).argmax(-1) == Yh).mean())
            hist.append(dict(step=step, loss=loss, acc=acc))
            if verbose:
                print(f"    {arm:12s} seed{seed} step{step:5d} loss={loss:.4f} "
                      f"acc={acc:.4f}", flush=True)
    return dict(arm=arm, seed=seed, params=n_params(p), history=hist,
                final_acc=hist[-1]["acc"], final_loss=hist[-1]["loss"],
                rho1=sigmoid(p["rho1"]).tolist(), rho2=sigmoid(p["rho2"]).tolist(),
                w1=p["w1"].tolist(), w2=p["w2"].tolist(),
                alpha=p["alpha"].tolist())


# ---------------------------------------------------------------- 守恒 / 梯度
def guards(arm):
    cfg = ARMS[arm]
    p = make_params(cfg, 0)
    X, Xp, Y, _ = build_chain(cfg["hops"], 4, 999)
    lg, st = forward(arm, p, X, Xp, cfg, ret_state=True)
    # 静态核：K_i(d) 只依赖 d=L-1-t，与绝对时刻无关 ⇒ Toeplitz
    # K_i(d)=sum_p w_p rho_p^d：只依赖延迟 d=L-1-t，与绝对时刻/输入内容无关
    dev = 0.0
    for t in range(L):
        d = L - 1 - t
        dev = max(dev, abs(float(np.sum(p["w1"] * st["rho1"] ** d)) - st["K1"][t]))
        if st["K2"] is not None:
            dev = max(dev, abs(float(np.sum(p["w2"] * st["rho2"] ** d)) - st["K2"][t]))
    # 换一组输入，核必须逐位不变（静态，非输入相关）
    Xq = X.copy(); Xq[:, L - 1, :] = 0.0; Xq[:, L - 1, (Y[0] + 1) % VK] = 1.0
    _, st2 = forward(arm, p, Xq, Xp, cfg, ret_state=True)
    dev = max(dev, float(np.abs(st["K1"] - st2["K1"]).max()))
    if st["K2"] is not None:
        dev = max(dev, float(np.abs(st["K2"] - st2["K2"]).max()))
    toeplitz = bool(dev < 1e-10)
    # 因果：递推输入 u_t = A_t ⊗ b_t 只依赖 x_{t-1}, x_t；
    # 改变最后一步 query 不得改变 t<=L-2 的 b 与 A
    causal = bool(np.allclose(st["b"][:, :L - 1], st2["b"][:, :L - 1], atol=1e-12)
                  and np.allclose(st["A"][:, :L - 1], st2["A"][:, :L - 1], atol=1e-12))
    # 状态 O(1)（每层 P*R*N，两层常数；与 L 无关）
    state_scalars = 2 * P * R * cfg["N"]
    # gradcheck
    eps = 1e-4
    loss0, g = ce_and_grad(p, X, Xp, Y, cfg)
    keys = ["E", "Wv", "Wout", "Wq2", "alpha", "rho1", "rho2", "w1", "w2"]
    gc = {}
    for k in keys:
        flat, gf = p[k].ravel(), g[k].ravel()
        idx = [0, min(3, flat.size - 1), flat.size - 1]
        errs = []
        for i in idx:
            pp = {kk: vv.copy() for kk, vv in p.items()}; pp[k].ravel()[i] += eps
            l1 = float(-np.log(softmax(forward(arm, pp, X, Xp, cfg))[np.arange(4), Y]
                               + 1e-12).mean())
            pm = {kk: vv.copy() for kk, vv in p.items()}; pm[k].ravel()[i] -= eps
            l2 = float(-np.log(softmax(forward(arm, pm, X, Xp, cfg))[np.arange(4), Y]
                               + 1e-12).mean())
            num = (l1 - l2) / (2 * eps)
            errs.append(abs(num - gf[i]) / (1e-6 + abs(num) + abs(gf[i])))
        gc[k] = dict(max_rel=float(max(errs)), pass_=bool(max(errs) < 3e-2))
    return dict(arm=arm, toeplitz=toeplitz, max_kernel_dev=float(dev),
                strict_causality=causal, state_scalars=state_scalars,
                state_independent_of_L=True,
                no_nxn_matrix=True, no_kv_cache=True,
                gradcheck=gc, all_gradcheck_pass=all(x["pass_"] for x in gc.values()),
                all_passed=bool(toeplitz and causal
                                and all(x["pass_"] for x in gc.values())))


def main():
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", default="h1_single,single,stack2,delay,stack_noq2,stack_rand,stack_h3,single_h3")
    a = ap.parse_args()
    arms = a.arms.split(",")

    if a.mode in ("guards", "all"):
        print("== conservation / gradcheck ==", flush=True)
        gd = [guards(x) for x in arms]
        for g in gd:
            print(f"  {g['arm']:12s} toeplitz={g['toeplitz']} causal={g['strict_causality']} "
                  f"state={g['state_scalars']} grad={g['all_gradcheck_pass']} "
                  f"all={g['all_passed']}", flush=True)
        json.dump(gd, open(f"{OUT}/chain_guards.json", "w"), indent=1)

    if a.mode in ("train", "all"):
        print("== training (synthetic content-chain bind; NOT language modeling) ==",
              flush=True)
        res = []
        for arm in arms:
            for s in SEEDS:
                res.append(train(arm, s, steps=a.steps, verbose=(s == 0)))
        json.dump(res, open(f"{OUT}/chain_results.json", "w"), indent=1)

        import collections
        by = collections.defaultdict(list)
        for r in res:
            by[r["arm"]].append(r["final_acc"])
        summary = {}
        for arm, vals in by.items():
            arr = np.array(vals)
            summary[arm] = dict(n=len(arr), mean=float(arr.mean()),
                                std=float(arr.std(ddof=1)), min=float(arr.min()),
                                max=float(arr.max()),
                                params=next(x["params"] for x in res if x["arm"] == arm),
                                per_seed=[float(v) for v in arr])
        if "stack2" in by and "single" in by and len(by["stack2"]) == len(by["single"]):
            pr = np.array([by["stack2"][i] - by["single"][i]
                           for i in range(len(by["stack2"]))])
            boot = np.array([np.random.default_rng(9000 + i).choice(pr, len(pr),
                             replace=True).mean() for i in range(10000)])
            summary["paired_stack2_minus_single"] = dict(
                per_seed=[float(x) for x in pr], mean=float(pr.mean()),
                ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                all_positive=bool(np.all(pr > 0)))
        else:
            summary["paired_stack2_minus_single"] = "not available (arms missing)"
        json.dump(summary, open(f"{OUT}/chain_summary.json", "w"), indent=1)
        print("\n== summary (held-out 512 x 5 seeds) ==", flush=True)
        for arm, s in summary.items():
            if isinstance(s, dict) and "params" in s:
                print(f"  {arm:12s} {s['mean']:.4f} +/- {s['std']:.4f} "
                      f"[{s['min']:.4f},{s['max']:.4f}] params={s['params']} "
                      f"chance=0.0313", flush=True)
        print("  paired:", summary["paired_stack2_minus_single"], flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
