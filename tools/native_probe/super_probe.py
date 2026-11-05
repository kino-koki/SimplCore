"""
super_probe.py — 「完全保持 SGC」前提下突破 attention 的判别性前哨。

核心赌注
========
B1 约束的是「静态核 + 线性 read-out」；B4 约束的是「时变递推系数」。
两者都 **不约束核外的逐位置非线性 f / g**。
于是把内容寻址信息编进输入 u_t 的方向/相位，用 query 在读出端解调，
递推核 K(τ) 仍固定、仍 Toeplitz、仍可单次 FFT、状态仍 O(1)。

所有臂共用同一个固定 SGC 核；只有 f（写入编码）、g（读出）、
网格（logfreq / linear / flat）与 M 变化。ρ/θ 全程不更新。

每个臂训练前必须过 sgc_fixed.guards() 六条守恒断言，否则直接淘汰。

任务
====
bind1：序列 [k1 v1 k2 v2 ... kn vn q]，q == 某个 key 的 id，
目标输出对应 value。只在 t=L-1 算 CE。
平凡基线：chance=1/V；last_value 启发式=1/n；shuffle 检验内容 vs 位置。

用法
====
  python super_probe.py --mode guards
  python super_probe.py --mode gradcheck
  python super_probe.py --mode train --arm S3_bind_orth --seed 0
  python super_probe.py --mode all              # 全臂 × 3 seeds + 消融 + 长度泛化
"""
import argparse
import json
import os
import sys
import time
import numpy as np

from mini_autograd import T, cat, const, zeros, nll_from_logits
import sgc_fixed as S

# ---------------- 配置 ----------------
D_IN = 32           # 嵌入维
M_DEF = 16          # 基数量
C = 4               # 每基通道数 -> H = M*C
H = M_DEF * C       # 64
V_KEY = 32          # key 词表
V_VAL = 32          # value 词表
VOCAB = V_KEY + V_VAL
N_PAIRS = 4
L_TRAIN = 2 * N_PAIRS + 1     # 9
B = 16
STEPS = 800
LR = 3e-3
SEEDS = (0, 1, 2)
OUT = "results"


# ---------------- 任务 ----------------
def build(n_pairs, n_samples, seed, shuffle_pairs=False):
    """返回 one-hot 输入、前驱 one-hot（t=0 为零）、target id、元数据。"""
    r = np.random.default_rng(seed)
    L = 2 * n_pairs + 1
    X = np.zeros((n_samples, L, VOCAB))
    Xp = np.zeros((n_samples, L, VOCAB))
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        keys = r.choice(V_KEY, size=n_pairs, replace=False)
        vals = r.choice(V_VAL, size=n_pairs, replace=False) + V_KEY
        pairs = list(zip(keys, vals))
        if shuffle_pairs and n_pairs > 1:
            r.shuffle(pairs)
        j = int(r.integers(0, n_pairs))
        for p, (k, v) in enumerate(pairs):
            X[i, 2 * p, k] = 1.0
            X[i, 2 * p + 1, v] = 1.0
        q = pairs[j][0]
        X[i, L - 1, q] = 1.0
        Y[i] = pairs[j][1]
        Xp[i, 1:, :] = X[i, :-1, :]
    return X, Xp, Y, dict(L=L, n_pairs=n_pairs, shuffle=shuffle_pairs, seed=seed)


def trivial_baselines(Y, n_pairs):
    """返回两个平凡基线的 acc（用于 sanity，不是模型）。"""
    # chance
    chance = 1.0 / V_VAL
    return dict(chance=chance, last_value=1.0 / n_pairs)


# ---------------- 参数 ----------------
ARMS = {
    "S0_static":      dict(f="lin",    g="linear", grid="logfreq", M=M_DEF),
    "S1_demod":       dict(f="lin",    g="demod",  grid="logfreq", M=M_DEF),
    "S2_bind":        dict(f="product",g="demod",  grid="logfreq", M=M_DEF),
    "S3_bind_orth":   dict(f="product",g="demod",  grid="linear",  M=M_DEF),
    "S4_phase_orth":  dict(f="phase",  g="phase",  grid="linear",  M=M_DEF),
    "S5_bind_flat":   dict(f="product",g="demod",  grid="flat",    M=M_DEF),
    "S6_bind_M32":    dict(f="product",g="demod",  grid="linear",  M=32),
    "A_attn_lr":      dict(f=None,      g=None,     grid=None,      M=None),
    "A_softmax":      dict(f="attn",    g="attn",   grid=None,      M=None),
    "A_softmax_win":  dict(f="attn",    g="attnw",  grid=None,      M=None),
    # 第二轮：核外竞争性解调 / 写入门（均不改递推核）
    "S8_softgate":    dict(f="product", g="demod_soft", grid="logfreq", M=M_DEF),
    "S9_writegate":   dict(f="product_gate", g="demod", grid="logfreq", M=M_DEF),
    "S10_both":       dict(f="product_gate", g="demod_soft", grid="logfreq", M=M_DEF),
    # 第三轮：容量消融（全部保持 LTI / Toeplitz / O(1) 状态）
    "S12_C8":         dict(f="product", g="demod", grid="logfreq", M=M_DEF, C=8),
    "S13_C16":        dict(f="product", g="demod", grid="logfreq", M=M_DEF, C=16),
    "S14_pureacc":    dict(f="product", g="demod", grid="pureacc", M=M_DEF),
    "S15_pureacc_M32":dict(f="product", g="demod", grid="pureacc", M=32),
    "S16_pureacc_nd": dict(f="product", g="demod", grid="pureacc_nodecay", M=M_DEF),
}


def make_params(cfg, seed):
    r = np.random.default_rng(seed + 7777)
    P = {"E": T(r.standard_normal((VOCAB, D_IN)) * 0.15)}
    Cc = cfg.get("C", C)
    Hh = cfg["M"] * Cc if cfg["M"] else H
    def W(a, b, s=0.15):
        return T(r.standard_normal((a, b)) * s)
    if cfg["f"] == "lin":
        P["Wf"] = W(D_IN, Hh)
    elif cfg["f"] == "product":
        P["Wa"] = W(D_IN, Hh)
        P["Wb"] = W(D_IN, Hh)
    elif cfg["f"] == "product_gate":
        P["Wa"] = W(D_IN, Hh)
        P["Wb"] = W(D_IN, Hh)
        P["Wg"] = W(D_IN, Hh, s=0.02)
    elif cfg["f"] == "phase":
        P["Wv"] = W(D_IN, Hh)
        P["Wp"] = W(D_IN, Hh)
    if cfg["g"] in ("demod", "phase", "demod_soft"):
        P["Wq"] = W(D_IN, Hh)
    P["Wout"] = W(Hh, VOCAB)
    if cfg["f"] is None:        # 线性注意力（无局部窗口）
        R = 8
        P = dict(E=P["E"],
                 Wk=W(D_IN, R), Wv=W(D_IN, Hh), Wq=W(D_IN, R),
                 Wout=W(Hh, VOCAB))
    elif cfg["f"] == "attn":    # softmax attention 上界
        R = 32
        d2 = 2 * D_IN if cfg["g"] == "attnw" else D_IN
        P = dict(E=P["E"],
                 Wk=W(d2, R), Wv=W(d2, R), Wq=W(d2, R),
                 Wout=W(R, VOCAB))
    return P


def n_params(P):
    return int(sum(v.d.size for v in P.values()))


# ---------------- 前向 ----------------
def forward(arm, P, X, Xp, rho, theta, Tm):
    B_, L, _ = X.shape
    cfg = ARMS[arm]
    xx = const(X) @ P["E"]          # (B,L,d)
    xp = const(Xp) @ P["E"]         # (B,L,d)，t=0 为 0
    if cfg["f"] is None:
        return _attn_lr(P, xx, L)
    if cfg["f"] == "attn":
        return _attn_softmax(cfg["g"], P, xx, xp, L)
    Cc = cfg.get("C", C)
    Hh = cfg["M"] * Cc

    if cfg["f"] == "lin":
        u = xx @ P["Wf"]
    elif cfg["f"] == "product":
        u = (xx @ P["Wa"]) * (xp @ P["Wb"])
    elif cfg["f"] == "product_gate":
        u = (xx @ P["Wa"]) * (xp @ P["Wb"]) * (xx @ P["Wg"]).sigmoid()
    elif cfg["f"] == "phase":
        u = (xx @ P["Wv"]) * (xp @ P["Wp"]).sine()

    u4 = u.reshape(B_, L, cfg["M"], Cc)
    y4 = T(np.einsum("mls,bsmc->blmc", Tm, u4.d), [u4])
    def bw():
        # dL/du4 = Tm^T @ dL/dy4  -> einsum
        u4.g += np.einsum("mls,blmc->bsmc", Tm, y4.g)
    y4._b = bw
    ylast = y4.reshape(B_, L, Hh)[:, L - 1, :]      # (B,H)
    ylast4 = y4[:, L - 1, :, :]                     # (B,M,C)

    if cfg["g"] == "linear":
        z = ylast
    elif cfg["g"] == "demod":
        z = (xx[:, L - 1, :] @ P["Wq"]) * ylast
    elif cfg["g"] == "phase":
        z = (xx[:, L - 1, :] @ P["Wq"]).sine() * ylast
    elif cfg["g"] == "demod_soft":
        # 沿 M 个基做 softmax 竞争：锐化 query 匹配，仍是核外逐位置非线性
        qh = xx[:, L - 1, :] @ P["Wq"]              # (B,H)
        q4 = qh.reshape(B_, cfg["M"], Cc)
        prod = q4 * ylast4
        score = prod.sum(axis=-1)                   # (B,M)
        w = softmax_T(score, axis=-1)               # (B,M)
        z = (w.reshape(B_, cfg["M"], 1) * prod).reshape(B_, Hh)
    else:
        raise ValueError(cfg["g"])

    return z @ P["Wout"]


def softmax_T(t, axis=-1):
    """手写 softmax 节点（mini_autograd 无 max op）。"""
    a = t.d
    e = np.exp(a - a.max(axis=axis, keepdims=True))
    p = e / e.sum(axis=axis, keepdims=True)
    out = T(p, [t])

    def back():
        g = out.g
        t.g += p * (g - (g * p).sum(axis=axis, keepdims=True))
    out._b = back
    return out


def _attn_softmax(kind, P, xx, xp, L):
    """单层 softmax attention 上界。
    attn : 每个位置用 x_s（无局部窗口）→ 预期解不了 (k,v) 分置的任务
    attnw: 用 [x_s ; x_{s-1}] 局部窗口，与 S2_bind 同样具备绑定能力
    """
    B_, L_, d = xx.d.shape
    src = cat([xx, xp], -1) if kind == "attnw" else xx
    k = src @ P["Wk"]
    v = src @ P["Wv"]
    q = src[:, L - 1, :] @ P["Wq"]
    kh = k[:, :L - 1, :]
    vh = v[:, :L - 1, :]
    R = q.d.shape[-1]
    sc = T(np.einsum("bsr,br->bs", kh.d, q.d) / np.sqrt(R), [kh, q])
    def bsc():
        kh.g += np.einsum("bs,br->bsr", sc.g, q.d) / np.sqrt(R)
        q.g += np.einsum("bs,bsr->br", sc.g, kh.d) / np.sqrt(R)
    sc._b = bsc
    e = np.exp(sc.d - sc.d.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)
    Pn = T(p, [sc])
    def bp():
        sc.g += p * (Pn.g - (Pn.g * p).sum(axis=-1, keepdims=True))
    Pn._b = bp
    out = T(np.einsum("bs,bsr->br", p, vh.d), [Pn, vh])
    def bo():
        Pn.g += np.einsum("br,bsr->bs", out.g, vh.d)
        vh.g += np.einsum("br,bs->bsr", out.g, p)
    out._b = bo
    return out @ P["Wout"]


def _attn_lr(P, xx, L):
    """等参数线性注意力对照：φ=1+tanh，S=Σφ(k)v^T，out=φ(q)^T S / den。"""
    B_, L_, d = xx.d.shape
    k = (xx @ P["Wk"]).tanh() + const(1.0)          # (B,L,R)
    v = xx @ P["Wv"]                                 # (B,L,H)
    hist = k[:, :L - 1, :]                           # 不含 query 位置
    hv = v[:, :L - 1, :]
    S_ = T(np.einsum("bsr,bsd->brd", hist.d, hv.d), [hist, hv])
    def bwS():
        hist.g += np.einsum("brd,bsd->bsr", S_.g, hv.d)
        hv.g += np.einsum("brd,bsr->bsd", S_.g, hist.d)
    S_._b = bwS
    q = (xx[:, L - 1, :] @ P["Wq"]).tanh() + const(1.0)   # (B,R)
    num = T(np.einsum("br,brd->bd", q.d, S_.d), [q, S_])
    def bwn():
        q.g += np.einsum("bd,brd->br", num.g, S_.d)
        S_.g += np.einsum("bd,br->brd", num.g, q.d)
    num._b = bwn
    den = T(np.einsum("bsr,br->b", hist.d, q.d), [hist, q])
    def bwd():
        # den_b = Σ_{s,r} hist[b,s,r] q[b,r]
        hist.g += den.g[:, None, None] * q.d[:, None, :]
        q.g += den.g[:, None] * hist.d.sum(axis=1)
    den._b = bwd
    z = num / (den.reshape(B_, 1) + const(1e-6))
    return z @ P["Wout"]


# 需要 T.__truediv__
def _truediv(self, o):
    return _bin_div(self, o)


def _bin_div(x, o):
    y = o if isinstance(o, T) else None
    a, b = (x.d, (o.d if isinstance(o, T) else np.asarray(o, dtype=np.float64)))
    out = T(a / b, [x] + ([y] if y is not None else []))
    def back():
        ga = out.g / b
        gb = -out.g * a / (b * b)
        x.g += _unbroadcast_div(ga, x.d.shape)
        if y is not None:
            y.g += _unbroadcast_div(gb, y.d.shape)
    out._b = back
    return out


def _unbroadcast_div(g, shape):
    from mini_autograd import _unbroadcast
    return _unbroadcast(g, shape)


T.__truediv__ = _truediv


# ---------------- 训练 ----------------
def zero_grad(P):
    for v in P.values():
        v.g[...] = 0.0


def adam_step(P, lr=LR, t=1):
    b1, b2, eps = 0.9, 0.999, 1e-8
    for name, v in P.items():
        key = "m_" + name
        if key not in ADAM:
            ADAM[key] = np.zeros_like(v.d)
            ADAM["v_" + name] = np.zeros_like(v.d)
        m, vv = ADAM[key], ADAM["v_" + name]
        m *= b1
        m += (1 - b1) * v.g
        vv *= b2
        vv += (1 - b2) * v.g * v.g
        mh = m / (1 - b1 ** t)
        vh = vv / (1 - b2 ** t)
        v.d -= lr * mh / (np.sqrt(vh) + eps)


ADAM = {}


def evaluate(arm, P, rho, theta, X, Xp, Y):
    L = X.shape[1]
    K = S.kernel(rho, theta, L) if rho is not None else None
    Tm = S.toeplitz(K) if K is not None else None
    logits = forward(arm, P, X, Xp, rho, theta, Tm)
    pred = logits.d.argmax(-1)
    return float((pred == Y).mean()), logits.d


def run_train(arm, seed, steps=STEPS, verbose=False):
    global ADAM
    ADAM = {}
    cfg = ARMS[arm]
    P = make_params(cfg, seed)
    if cfg["M"] is not None:
        rho, theta = S.grids(cfg["M"], cfg["grid"], L=L_TRAIN,
                             tmax=cfg.get("tmax"))
        ginfo = S.guards(rho, theta, L_TRAIN, C=cfg.get("C", C), rng_seed=seed)
        if not ginfo["all_passed"]:
            return dict(arm=arm, seed=seed, eliminated=True,
                        guards=ginfo, reason="守恒断言未过，不算保持 SGC")
        fp_before = float(np.abs(rho).sum() + np.abs(theta).sum())
    else:
        rho = theta = None
        ginfo = None
        fp_before = None

    Xtr, Xptr, Ytr, _ = build(N_PAIRS, 2000, seed=1000 + seed)
    Xte, Xpte, Yte, _ = build(N_PAIRS, 512, seed=9000 + seed)
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = np.random.default_rng(seed * 100000 + step).choice(len(Ytr), B, replace=False)
        X, Xp, Y = Xtr[idx], Xptr[idx], Ytr[idx]
        K = S.kernel(rho, theta, L_TRAIN) if rho is not None else None
        Tm = S.toeplitz(K) if K is not None else None
        zero_grad(P)
        logits = forward(arm, P, X, Xp, rho, theta, Tm)
        loss = nll_from_logits(logits, Y)
        loss.backward()
        adam_step(P, t=step)
        if step % 200 == 0 or step == 1:
            acc, _ = evaluate(arm, P, rho, theta, Xte, Xpte, Yte)
            curve.append(dict(step=step, loss=float(loss.d), acc=acc))
            if verbose:
                print(f"  [{arm} s{seed}] step {step:4d} loss {loss.d:.4f} acc {acc:.4f}")

    acc_te, _ = evaluate(arm, P, rho, theta, Xte, Xpte, Yte)
    # shuffle 检验：同一组 kv 对打乱顺序（模型未重训）
    Xsh, Xpsh, Ysh, _ = build(N_PAIRS, 512, seed=9000 + seed, shuffle_pairs=True)
    acc_sh, _ = evaluate(arm, P, rho, theta, Xsh, Xpsh, Ysh)

    # 长度泛化（只重建核，ρ/θ 不变）
    gen = {}
    if rho is not None:
        for n in (4, 8, 12, 16):
            Xg, Xpg, Yg, _ = build(n, 256, seed=7000 + seed + n)
            acc_g, _ = evaluate(arm, P, rho, theta, Xg, Xpg, Yg)
            gen[f"L{2*n+1}"] = acc_g

    fp_after = float(np.abs(rho).sum() + np.abs(theta).sum()) if rho is not None else None
    frozen = (fp_before == fp_after) if rho is not None else True

    return dict(arm=arm, seed=seed, params=n_params(P),
                guards=ginfo, rho_theta_frozen=frozen,
                fingerprint_before=fp_before, fingerprint_after=fp_after,
                final_acc=acc_te, shuffle_acc=acc_sh,
                length_generalization=gen,
                curve=curve, elapsed_s=round(time.time() - t0, 1),
                eliminated=False)


# ---------------- gradcheck ----------------
def gradcheck(arm, seed=0):
    cfg = ARMS[arm]
    P = make_params(cfg, seed)
    rho, theta = S.grids(cfg["M"] or M_DEF, cfg["grid"] or "linear", L=L_TRAIN,
                         tmax=cfg.get("tmax"))
    X, Xp, Y, _ = build(N_PAIRS, 4, seed=42)
    K = S.kernel(rho, theta, L_TRAIN) if rho is not None else None
    Tm = S.toeplitz(K) if K is not None else None
    zero_grad(P)
    loss = nll_from_logits(forward(arm, P, X, Xp, rho, theta, Tm), Y)
    loss.backward()
    # 一次性快照：数值差分循环里若再清 .g 会污染后续参数的解析梯度
    analytic = {name: v.g.ravel().copy() for name, v in P.items()}
    r = np.random.default_rng(123)
    worst = 0.0
    details = []
    eps = 1e-6
    for name in list(P)[:4]:
        v = P[name]
        flat = v.d.ravel()
        gflat = analytic[name]
        idxs = r.choice(flat.size, size=min(4, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp = nll_from_logits(forward(arm, P, X, Xp, rho, theta, Tm), Y).d
            flat[i] = old - eps
            lm = nll_from_logits(forward(arm, P, X, Xp, rho, theta, Tm), Y).d
            flat[i] = old
            num = float((lp - lm) / (2 * eps))
            ana = float(gflat[i])
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
            # 标准混合判据：极小梯度处只用相对误差会被 fp64 舍入噪声放大
            ok = (rel < 5e-4) or (abs(num - ana) < 1e-8)
            worst = max(worst, 0.0 if ok else rel)
            details.append(dict(param=name, idx=int(i), numerical=num,
                                analytic=ana, rel=rel, passed=bool(ok)))
    return dict(arm=arm, worst_rel=worst,
                passed=all(d["passed"] for d in details),
                criterion="rel<5e-4 OR abs(num-ana)<1e-8",
                details=details)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="guards",
                    choices=["guards", "gradcheck", "train", "all",
                             "gridscan", "long", "capacity"])
    ap.add_argument("--arm", default="S3_bind_orth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)


    if a.mode == "capacity":
        # 容量扫描：只改 M / C / 网格，递推核仍 LTI、Toeplitz、O(1) 状态
        combos = [
            ("m16c4",   16, 4,  "logfreq"),
            ("m16c8",   16, 8,  "logfreq"),
            ("m16c16",  16, 16, "logfreq"),
            ("m16c32",  16, 32, "logfreq"),
            ("m16c64",  16, 64, "logfreq"),
            ("m32c8",   32, 8,  "logfreq"),
            ("m32c16",  32, 16, "logfreq"),
            ("m64c8",   64, 8,  "logfreq"),
            ("m16c16pa",16, 16, "pureacc"),
            ("m32c16pa",32, 16, "pureacc"),
        ]
        out = {}
        for tag, M_, C_, grid in combos:
            arm = "CAP_" + tag
            ARMS[arm] = dict(f="product", g="demod", grid=grid, M=M_, C=C_)
            rr = [run_train(arm, sd, steps=a.steps) for sd in SEEDS]
            out[tag] = dict(M=M_, C=C_, grid=grid, H=M_ * C_,
                            params=rr[0]["params"],
                            state_bytes=rr[0]["guards"]["state_bytes"],
                            accs=[r["final_acc"] for r in rr],
                            mean=float(np.mean([r["final_acc"] for r in rr])),
                            sd=float(np.std([r["final_acc"] for r in rr])),
                            shuffle_mean=float(np.mean([r["shuffle_acc"] for r in rr])),
                            gen=rr[0]["length_generalization"],
                            guards_ok=all(r["guards"]["all_passed"] for r in rr))
            print(f"  [{tag:9s}] M={M_:2d} C={C_:2d} H={M_*C_:3d} "
                  f"params={out[tag]['params']:6d} acc={out[tag]['mean']:.4f}"
                  f"±{out[tag]['sd']:.4f} shuffle={out[tag]['shuffle_mean']:.4f} "
                  f"guards={out[tag]['guards_ok']}", flush=True)
            with open(f"{OUT}/capacity.json", "w") as fh:
                json.dump(out, fh, indent=1, ensure_ascii=False)
                fh.flush()
        print("\n写入 results/capacity.json")
        return


    if a.mode == "capacity":
        # 容量扫描：只改 M / C / 网格，递推核仍 LTI、Toeplitz、O(1) 状态
        combos = [
            ("m16c4",   16, 4,  "logfreq"),
            ("m16c8",   16, 8,  "logfreq"),
            ("m16c16",  16, 16, "logfreq"),
            ("m16c32",  16, 32, "logfreq"),
            ("m16c64",  16, 64, "logfreq"),
            ("m32c8",   32, 8,  "logfreq"),
            ("m32c16",  32, 16, "logfreq"),
            ("m64c8",   64, 8,  "logfreq"),
            ("m16c16pa",16, 16, "pureacc"),
            ("m32c16pa",32, 16, "pureacc"),
        ]
        out = {}
        for tag, M_, C_, grid in combos:
            arm = "CAP_" + tag
            ARMS[arm] = dict(f="product", g="demod", grid=grid, M=M_, C=C_)
            rr = [run_train(arm, sd, steps=a.steps) for sd in SEEDS]
            out[tag] = dict(M=M_, C=C_, grid=grid, H=M_ * C_,
                            params=rr[0]["params"],
                            state_bytes=rr[0]["guards"]["state_bytes"],
                            accs=[r["final_acc"] for r in rr],
                            mean=float(np.mean([r["final_acc"] for r in rr])),
                            sd=float(np.std([r["final_acc"] for r in rr])),
                            shuffle_mean=float(np.mean([r["shuffle_acc"] for r in rr])),
                            gen=rr[0]["length_generalization"],
                            guards_ok=all(r["guards"]["all_passed"] for r in rr))
            print(f"  [{tag:9s}] M={M_:2d} C={C_:2d} H={M_*C_:3d} "
                  f"params={out[tag]['params']:6d} acc={out[tag]['mean']:.4f}"
                  f"±{out[tag]['sd']:.4f} shuffle={out[tag]['shuffle_mean']:.4f} "
                  f"guards={out[tag]['guards_ok']}", flush=True)
            with open(f"{OUT}/capacity.json", "w") as fh:
                json.dump(out, fh, indent=1, ensure_ascii=False)
                fh.flush()
        print("\n写入 results/capacity.json")
        return

    if a.mode == "gridscan":
        # 相变点扫描：theta 上界从近零扫到正交区间，f/g 固定为 bind+demod
        TMAXS = [("t0", None), ("t001", 0.01), ("t005", 0.05), ("t01", 0.1),
                 ("t03", 0.3), ("t06", 0.6), ("t10", 1.0), ("t20", 2.0),
                 ("t29", 2.9)]
        out = {}
        for tag, tmax in TMAXS:
            arm = "SCAN_" + tag
            ARMS[arm] = dict(f="product", g="demod", grid="scan",
                             M=M_DEF, tmax=tmax)
            rr = [run_train(arm, sd, steps=a.steps) for sd in SEEDS]
            out[tag] = dict(tmax=tmax,
                            theta_max=float(S.grids(M_DEF, "scan", tmax=tmax)[1].max()),
                            accs=[r["final_acc"] for r in rr],
                            shuffles=[r["shuffle_acc"] for r in rr],
                            mean=float(np.mean([r["final_acc"] for r in rr])),
                            sd=float(np.std([r["final_acc"] for r in rr])),
                            gen=rr[0]["length_generalization"],
                            guards_ok=all(r["guards"]["all_passed"] for r in rr))
            print(f"  [{tag:5s}] theta_max={out[tag]['theta_max']:.5f} "
                  f"acc={out[tag]['mean']:.4f}±{out[tag]['sd']:.4f} "
                  f"shuffle={np.mean(out[tag]['shuffles']):.4f} "
                  f"guards={out[tag]['guards_ok']}", flush=True)
            with open(f"{OUT}/gridscan.json", "w") as fh:
                json.dump(out, fh, indent=1, ensure_ascii=False)
                fh.flush()
        with open(f"{OUT}/gridscan.json", "w") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
            fh.flush()
        print("\n写入 results/gridscan.json")
        return

    if a.mode == "long":
        arms = ["S2_bind", "S8_softgate", "S9_writegate", "S10_both"]
        out = {}
        for arm in arms:
            rr = [run_train(arm, sd, steps=a.steps) for sd in SEEDS]
            out[arm] = dict(accs=[r["final_acc"] for r in rr],
                            mean=float(np.mean([r["final_acc"] for r in rr])),
                            sd=float(np.std([r["final_acc"] for r in rr])),
                            shuffles=[r["shuffle_acc"] for r in rr],
                            gen=[r["length_generalization"] for r in rr],
                            curves=[r["curve"] for r in rr],
                            frozen=all(r["rho_theta_frozen"] for r in rr))
            print(f"  [{arm:16s}] {out[arm]['mean']:.4f}±{out[arm]['sd']:.4f}",
                  flush=True)
            json.dump(out, open(f"{OUT}/long_results.json", "w"),
                      indent=1, ensure_ascii=False)
        print("\n写入 results/long_results.json")
        return

    if a.mode == "guards":
        out = {}
        for grid in ("logfreq", "linear", "flat"):
            rho, theta = S.grids(M_DEF, grid, L=L_TRAIN)
            out[grid] = S.guards(rho, theta, L_TRAIN, C=C, verbose=True)
        json.dump(out, open(f"{OUT}/guards.json", "w"), indent=1, ensure_ascii=False)
        print("\n写入 results/guards.json")
        return

    if a.mode == "gradcheck":
        out = [gradcheck(arm) for arm in ARMS]
        for g in out:
            print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['arm']:16s} "
                  f"worst_rel={g['worst_rel']:.2e}")
        json.dump(out, open(f"{OUT}/gradcheck.json", "w"), indent=1, ensure_ascii=False)
        print(f"\n==> {sum(g['passed'] for g in out)}/{len(out)} PASS")
        return

    if a.mode == "train":
        r = run_train(a.arm, a.seed, steps=a.steps, verbose=True)
        fn = f"{OUT}/train_{a.arm}_s{a.seed}.json"
        json.dump(r, open(fn, "w"), indent=1, ensure_ascii=False)
        print(json.dumps({k: v for k, v in r.items() if k not in ("curve", "guards")},
                         indent=1, ensure_ascii=False))
        print("\n写入", fn)
        return

    if a.mode == "all":
        fn = f"{OUT}/all_results.json"
        allres = json.load(open(fn)) if os.path.exists(fn) else {}
        for arm in list(ARMS):
            done = {int(r["seed"]) for r in allres.get(arm, [])
                    if not r.get("eliminated")}
            for sd in SEEDS:
                if sd in done:
                    print(f"  [{arm:16s} s{sd}] skip (已有)", flush=True)
                    continue
                r = run_train(arm, sd, steps=a.steps, verbose=False)
                allres.setdefault(arm, []).append(r)
                tag = "ELIM" if r.get("eliminated") else f"acc={r['final_acc']:.4f}"
                print(f"  [{arm:16s} s{sd}] {tag}  "
                      f"shuffle={r.get('shuffle_acc')} "
                      f"params={r.get('params')}  {r.get('elapsed_s')}s", flush=True)
                json.dump(allres, open(fn, "w"), indent=1, ensure_ascii=False)
        print("\n写入", fn)
        return


if __name__ == "__main__":
    main()
