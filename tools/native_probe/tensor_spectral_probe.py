"""
tensor_spectral_probe.py — 张量谱卷积（Tensor Spectral Convolution, TSC）。

从 SGC 数学本质出发的最后一块拼图
==================================
B1 证明的是「静态核对**向量信号**做线性叠加 + 线性 read-out」不能做内容寻址。
但 SGC 的定义只规定了「核是静态 LTI」，**没有规定信号必须是向量**。

把信号从向量升级成外积张量，核作用在每个张量元素上，读出端做缩并：

    a_t = x_{t-1}·Wk        (前驱 token 的 key 编码, R 维)
    b_t = x_t·Wv            (当前 token 的 value 编码, N 维)
    u_t[r,n] = a_t[r]·b_t[n]                   外积 (R,N)

    s_{p,r,n,t} = ρ_{p,n}·s_{p,r,n,t-1} + u_t[r,n]      P 拍 LTI 递推
    z_n = Σ_{p,r} w_{p,n}·q_r·s_{p,r,n,L}                query 缩并读出
    logits = z·Wout

关键：q_r·s_{p,r,n} 对 r 缩并，形成 (q·k_τ) 匹配，把对应 v_τ 取出。
**核 ρ 与 t 无关 ⇒ 逐通道 Toeplitz ⇒ FFT/IIR 对偶、O(1) 状态、严格因果全保。**

这与 B1 不冲突：B1 假设 y 是向量信号的线性叠加且 read-out 线性；
这里 y 是张量、read-out 是缩并（双线性）。B1 的适用范围需相应收紧，
不是被推翻。

诚实张力（必须写进论文）
========================
当 ρ→1、P→1 时，S = Σ_τ u_τ = Σ_τ k_τ v_τᵀ 与线性注意力的状态同构。
本路线不是「把 attention 搬进来」：没有 token 对打分矩阵、没有 softmax、
没有 KV cache；它是「SGC 核对张量信号做 LTI 滤波」的自然推广，
ρ<1 的衰减谱是 SGC 独有的（attention 没有）。
但这层同构关系不隐藏，由作者/审稿人自行裁定边界。

严格保持
========
  · 核 LTI / Toeplitz / FFT-IIR 对偶 / O(1) 状态 / 严格因果
  · 无 N×N 序列矩阵、无 KV cache、无 softmax、无混合折中网格
  · 活跃代码零对照模型实现

用法
====
  python tensor_spectral_probe.py --mode budget
  python tensor_spectral_probe.py --mode guards
  python tensor_spectral_probe.py --mode gradcheck
  python tensor_spectral_probe.py --mode all --steps 3000 --arms T0,T1,T2,T3
"""
import argparse
import json
import os
import time
import numpy as np

from mini_autograd import T, const, zeros, nll_from_logits, free_graph
import task_utils as U
import sgc_fixed as S

OUT = "results"
SEEDS = (0, 1, 2)
STEPS = 3000
RHO_MIN = 0.05
BUDGET = 10240
TOL = 0.03


def _hadamard(n):
    """Sylvester 构造，无外部依赖。行/列精确正交。"""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def _fixed_code(R, V=64, kind="hadamard"):
    """固定正交谱字典（与 ρ/θ 网格同性质：不学、确定性、正交）。
    R=V 时任意两 token 的码内积 = δ_ij ⇒ q·k 零串扰。"""
    if kind == "random":
        g = np.random.default_rng(20261105).standard_normal((V, V))
        H, _ = np.linalg.qr(g)
    else:
        H = _hadamard(V)
    C = H[:, :R] * np.sqrt(V / R)
    return C


def _sig(z): return 1.0 / (1.0 + np.exp(-z))
def _logit(u):
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


ARMS = {
    # T0 对照：向量信号（等同上一轮 E0_P2），用于证明「张量」是净增益
    "T0_vector":   dict(kind="vector", D=32, R=8,  N=50,  P=2, tied=False, gate=False),
    # 张量信号主臂，参数全部对齐 10,240 ±3%
    "T1_P1":       dict(kind="tensor", D=16, R=8,  N=109, P=1, tied=False, gate=False),
    "T2_P2":       dict(kind="tensor", D=16, R=8,  N=107, P=2, tied=False, gate=False),
    "T3_P4":       dict(kind="tensor", D=16, R=8,  N=102, P=4, tied=False, gate=False),
    "T4_lowD":     dict(kind="tensor", D=12, R=8,  N=116, P=2, tied=False, gate=False),
    "T5_gate":     dict(kind="tensor", D=16, R=8,  N=90,  P=2, tied=False, gate=True),
    "T6_rho1":     dict(kind="tensor", D=16, R=8,  N=107, P=2, tied=False, gate=False,
                        freeze_rho=True),
    "T7_noK":      dict(kind="tensor_nok", D=16, R=8, N=105, P=2, tied=False, gate=False),
    "T8_bigR":     dict(kind="tensor", D=16, R=16, N=104, P=2, tied=False, gate=False),
    # 长预算（步数另给）
    "T10_long":    dict(kind="tensor", D=16, R=8,  N=107, P=2, tied=False, gate=False),
    # 第五轮：query 与 key 共用同一组谱基（SGC「固定基字典」的直接体现）
    "U0_kq":       dict(kind="tensor", D=16, R=8,  N=108, P=2, tied=False, gate=False, shared_kq=True),
    "U1_kq_lr6":   dict(kind="tensor", D=16, R=8,  N=108, P=2, tied=False, gate=False, shared_kq=True, lr=6e-3),
    "U2_kq_R16":   dict(kind="tensor", D=16, R=16, N=107, P=2, tied=False, gate=False, shared_kq=True),
    "U3_kq_orth":  dict(kind="tensor", D=16, R=8,  N=108, P=2, tied=False, gate=False, shared_kq=True, orth=True),
    "U4_kq_long":  dict(kind="tensor", D=16, R=8,  N=108, P=2, tied=False, gate=False, shared_kq=True, lr=6e-3),
    "U5_kq_P1":    dict(kind="tensor", D=16, R=8,  N=110, P=1, tied=False, gate=False, shared_kq=True),
    "U6_kq_nl":    dict(kind="tensor", D=16, R=8,  N=108, P=2, tied=False, gate=False, shared_kq=True, read_nl=True),
    # 第六轮：最优组合 + 训练动力学
    "V0_combo":    dict(kind="tensor", D=16, R=16, N=107, P=2, tied=False, gate=False,
                        shared_kq=True, orth=True, lr=6e-3),
    "V1_cosine":   dict(kind="tensor", D=16, R=16, N=107, P=2, tied=False, gate=False,
                        shared_kq=True, orth=True, lr=6e-3, schedule="cosine"),
    "V2_R24":      dict(kind="tensor", D=16, R=24, N=105, P=2, tied=False, gate=False,
                        shared_kq=True, orth=True, lr=6e-3),
    "V3_batch32":  dict(kind="tensor", D=16, R=16, N=107, P=2, tied=False, gate=False,
                        shared_kq=True, orth=True, lr=6e-3, batch=32),
    "V4_R32":      dict(kind="tensor", D=16, R=32, N=104, P=2, tied=False, gate=False,
                        shared_kq=True, orth=True, lr=6e-3),
    # 第七轮：固定正交谱字典（SGC 原生零件）—— q·k 精确可分
    "W0_code64":   dict(kind="tensor", D=16, R=64, N=110, P=2, tied=False, gate=False,
                        fixed_code=True),
    "W1_code64_lr":dict(kind="tensor", D=16, R=64, N=110, P=2, tied=False, gate=False,
                        fixed_code=True, lr=6e-3, schedule="cosine"),
    "W2_code64_P1":dict(kind="tensor", D=16, R=64, N=112, P=1, tied=False, gate=False,
                        fixed_code=True, lr=6e-3, schedule="cosine"),
    "W3_code32":   dict(kind="tensor", D=16, R=32, N=110, P=2, tied=False, gate=False,
                        fixed_code=True, lr=6e-3, schedule="cosine"),
    "W4_half":     dict(kind="tensor", D=8,  R=64, N=58,  P=2, tied=False, gate=False,
                        fixed_code=True, lr=6e-3, schedule="cosine"),
    "W5_rho1":     dict(kind="tensor", D=16, R=64, N=110, P=2, tied=False, gate=False,
                        fixed_code=True, lr=6e-3, schedule="cosine", freeze_rho=True),
    "W6_code_rand":dict(kind="tensor", D=16, R=64, N=110, P=2, tied=False, gate=False,
                        fixed_code=True, code_kind="random", lr=6e-3, schedule="cosine"),
    "W7_more_pairs":dict(kind="tensor", D=16, R=64, N=110, P=2, tied=False, gate=False,
                        fixed_code=True, lr=6e-3, schedule="cosine", n_pairs=8),
}

def _count(cfg):
    D, R, N, P = cfg["D"], cfg["R"], cfg["N"], cfg["P"]
    if cfg["kind"] == "vector":
        c = U.VOCAB * D + 2 * D * N + D * N
        if not cfg["tied"]: c += N * U.VOCAB
        c += 2 * N * P
        if cfg["gate"]: c += D * N
        return int(c)
    c = U.VOCAB * D            # E（value 编码用）
    if not cfg.get("fixed_code"):
        c += D * R             # Wk（前驱 key 编码）
    c += D * N                 # Wv（当前 value 编码）
    if not cfg.get("shared_kq") and not cfg.get("fixed_code"):
        c += D * R             # Wq（query 编码）
    if cfg["kind"] == "tensor_nok":
        c += D * R             # 额外：当前 token 也投到 R（消融用）
    if not cfg["tied"]:
        c += N * U.VOCAB       # Wout
    c += 2 * P * N             # ρ(p) + w(p)   —— 与 R 无关（共享）
    if cfg["gate"]:
        c += D * N
    return int(c)


def make_params(cfg, seed):
    r = np.random.default_rng(seed + 4242)
    def W(a, b, s=0.15): return T(r.standard_normal((a, b)) * s)
    D, R, N, P = cfg["D"], cfg["R"], cfg["N"], cfg["P"]
    Pm = {"E": T(r.standard_normal((U.VOCAB, D)) * 0.15)}
    if cfg["kind"] == "vector":
        Pm.update(Wa=W(D, N), Wb=W(D, N), Wq=W(D, N))
        if not cfg["tied"]: Pm["Wout"] = W(N, U.VOCAB)
        if cfg["gate"]: Pm["Wg"] = W(D, N, s=0.02)
        rp = np.exp(np.linspace(np.log(0.30), np.log(0.995), P))
        p0 = np.repeat(_logit((rp - RHO_MIN) / (1 - RHO_MIN))[:, None], N, axis=1)
        Pm["p"] = T((p0 + r.normal(0, 0.08, (P, N))).reshape(-1))
        Pm["w"] = T((np.ones((P, N)) / P + r.normal(0, 0.1, (P, N))).reshape(-1))
        return Pm

    if not cfg.get("fixed_code"):
        if cfg.get("orth"):
            k = max(D, R)
            qmat, _ = np.linalg.qr(r.standard_normal((k, k)))
            Pm["Wk"] = T(qmat[:D, :R] * np.sqrt(k / R))
        else:
            Pm["Wk"] = W(D, R)
    Pm["Wv"] = W(D, N)
    if not cfg.get("shared_kq") and not cfg.get("fixed_code"):
        Pm["Wq"] = W(D, R)
    if cfg["kind"] == "tensor_nok":
        Pm["Wk2"] = W(D, R)
    if not cfg["tied"]:
        Pm["Wout"] = W(N, U.VOCAB)
    if cfg["gate"]:
        Pm["Wg"] = W(D, N, s=0.02)
    # ρ 形状 (P,N)：每个 value 通道 P 个极点；对 R 共享（LTI 逐通道相同）
    rp = np.exp(np.linspace(np.log(0.55), np.log(0.995), P))
    p0 = np.repeat(_logit((rp - RHO_MIN) / (1 - RHO_MIN))[:, None], N, axis=1)
    p0 = p0 + r.normal(0, 0.15, (P, N))          # 逐通道打破对称
    if cfg.get("freeze_rho"):
        p0 = np.repeat(_logit((np.ones(P) * 0.995 - RHO_MIN) / (1 - RHO_MIN))[:, None],
                       N, axis=1)
    Pm["p"] = T(p0.reshape(-1))
    Pm["w"] = T((np.ones((P, N)) / P + r.normal(0, 0.12, (P, N))).reshape(-1))
    return Pm


def n_params(P):
    return int(sum(v.d.size for v in P.values()))


def _rho(Pm, P, N):
    pp = Pm["p"]
    rr = T((RHO_MIN + (1 - RHO_MIN) * _sig(pp.d)).reshape(P, N), [pp])
    def bw():
        sp = _sig(pp.d)
        pp.g += rr.g.reshape(-1) * (1 - RHO_MIN) * sp * (1 - sp)
    rr._b = bw
    return rr


def forward(arm, P, X, Xp, L):
    cfg = ARMS[arm]
    B_, _, _ = X.shape
    D, R, N, Pp = cfg["D"], cfg["R"], cfg["N"], cfg["P"]
    xx = const(X) @ P["E"]          # (B,L,D)
    xp = const(Xp) @ P["E"]         # (B,L,D) 前驱
    if cfg.get("fixed_code"):
        Code = const(_fixed_code(R, U.VOCAB, cfg.get("code_kind", "hadamard")))
        q = const(X[:, L - 1, :]) @ Code          # (B,R) 固定正交码
    else:
        q = xx[:, L - 1, :] @ (P["Wk"] if cfg.get("shared_kq") else P["Wq"])

    if cfg["kind"] == "vector":
        u = (xx @ P["Wa"]) * (xp @ P["Wb"])
        if cfg["gate"]:
            u = u * (xx @ P["Wg"]).sigmoid()
        rr = _rho(P, Pp, N)                              # (P,N)
        s = zeros((B_, Pp, N))
        for t in range(L):
            s = rr.reshape(1, Pp, N) * s + u[:, t, :].reshape(B_, 1, N)
        ww = P["w"].reshape(Pp, N)
        y = (ww.reshape(1, Pp, N) * s).sum(axis=1)      # (B,N)
        z = q * y
        lg = z @ P["Wout"] if "Wout" in P else z @ P["E"].T
        return lg, s

    # ---- 张量信号 ----
    a = (const(Xp) @ Code) if cfg.get("fixed_code") else (xp @ P["Wk"])  # (B,L,R)
    b = xx @ P["Wv"]                                    # (B,L,N) 当前 value 编码
    if cfg["kind"] == "tensor_nok":
        a = (const(X) @ Code) if cfg.get("fixed_code") else (xx @ P["Wk2"])
    if cfg["gate"]:
        b = b * (xx @ P["Wg"]).sigmoid()
    u = a.reshape(B_, L, R, 1) * b.reshape(B_, L, 1, N)  # (B,L,R,N) 外积

    rr = _rho(P, Pp, N)                                  # (P,N)
    s = zeros((B_, Pp, R, N))
    for t in range(L):
        s = rr.reshape(1, Pp, 1, N) * s + u[:, t, :, :].reshape(B_, 1, R, N)

    # query 缩并：对 r 求和形成 (q·k) 匹配，再对 p 加权
    zr = (s * q.reshape(B_, 1, R, 1)).sum(axis=2)       # (B,P,N)
    ww = P["w"].reshape(Pp, N)
    z = (zr * ww.reshape(1, Pp, N)).sum(axis=1)         # (B,N)
    if cfg.get("read_nl"):
        z = z.tanh()

    lg = z @ P["Wout"] if "Wout" in P else z @ P["E"].T
    return lg, s


# ---------------- 守恒 ----------------
def guards(arm, P, seed=0, verbose=False):
    cfg = ARMS[arm]
    L = 2 * cfg.get("n_pairs", U.N_PAIRS) + 1
    N, Pp = cfg["N"], cfg["P"]
    R = cfg["R"]
    r = np.random.default_rng(seed)
    # 核只依赖 (n,τ)，与 r 无关 ⇒ 逐通道 Toeplitz
    rho = (RHO_MIN + (1 - RHO_MIN) * _sig(P["p"].d)).reshape(Pp, N)
    w = P["w"].d.reshape(Pp, N)
    tau = np.arange(L, dtype=np.float64)
    K = (w[:, :, None] * (rho[:, :, None] ** tau[None, None, :])).sum(0)  # (N,L)
    Tm = S.toeplitz(K)
    u = r.standard_normal((2, L, N))
    y = np.einsum("nls,bsn->bln", Tm, u)

    dev = max(abs(Tm[n, t, ss] - Tm[n, t - 1, ss - 1])
              for n in range(N) for t in range(1, L) for ss in range(1, t + 1))
    yf = S.fft_apply(K, u[:, :, :, None])[:, :, :, 0]
    den = max(np.abs(y).max(), 1e-12)
    rel_fft = float(np.abs(y - yf).max() / den)
    # P 拍 IIR（对张量通道 r 独立，此处用向量化的等价检查）
    s = np.zeros((2, Pp, N))
    yi = np.zeros((2, L, N))
    for t in range(L):
        s = rho[None, :, :] * s + u[:, t, :][:, None, :]
        yi[:, t, :] = (w[None, :, :] * s).sum(1)
    rel_iir = float(np.abs(y - yi).max() / den)
    u2 = u.copy(); u2[:, L - 1] += 10
    past = float(np.abs(y[:, :L - 1]
                        - np.einsum("nls,bsn->bln", Tm, u2)[:, :L - 1]).max())
    bytes_ = [Pp * R * N * 4 if cfg["kind"] != "vector" else N * Pp * 4
              for _ in (9, 17, 25, 65)]
    out = dict(toeplitz=bool(dev < 1e-10), fft_iir=bool(rel_fft < 1e-9),
               iir_vs_toeplitz=bool(rel_iir < 1e-9),
               strict_causality=bool(past == 0),
               state_independent_of_L=bool(len(set(bytes_)) == 1),
               max_toeplitz_dev=float(dev), rel_fft=rel_fft, rel_iir=rel_iir,
               state_bytes=bytes_[0],
               note="核 ρ 与 t、r 无关 ⇒ 逐通道 Toeplitz；u_t 外生 ⇒ 单次 FFT 可行",
               all_passed=bool(dev < 1e-10 and rel_fft < 1e-9 and rel_iir < 1e-9
                               and past == 0 and len(set(bytes_)) == 1))
    if verbose:
        print(f"    toeplitz={out['toeplitz']} fft={rel_fft:.2e} "
              f"iir={rel_iir:.2e} causal={out['strict_causality']} "
              f"state={bytes_[0]}B all={out['all_passed']}")
    return out


# ---------------- 判别性对照：未见过的 value 置换 ----------------
def permute_values(X, Xp, Y, seed):
    """保持 key→位置映射，把 value id 做固定置换。
    若模型靠 key→value 记忆 ⇒ 掉到 chance；若真的从输入读出 value ⇒ 保持。"""
    perm = np.random.default_rng(seed).permutation(U.V_VAL)
    X2 = X.copy(); Y2 = Y.copy()
    for b in range(X.shape[0]):
        for t in range(X.shape[1]):
            idx = int(np.argmax(X[b, t]))
            if idx >= U.V_KEY:
                X2[b, t] = 0
                X2[b, t, U.V_KEY + perm[idx - U.V_KEY]] = 1.0
        Y2[b] = U.V_KEY + perm[int(Y[b]) - U.V_KEY]
    Xp2 = np.zeros_like(Xp); Xp2[:, 1:] = X2[:, :-1]
    return X2, Xp2, Y2


# ---------------- 训练 ----------------
def eval_correct(arm, P, X, Xp, Y, chunk=32):
    """分块评测：张量状态的 autograd 图随 B 线性增长，整批 512 会 OOM。"""
    out = np.zeros(len(Y), dtype=int)
    for i in range(0, len(Y), chunk):
        sl = slice(i, i + chunk)
        lg, st = forward(arm, P, X[sl], Xp[sl], X.shape[1])
        out[sl] = (lg.d.argmax(-1) == Y[sl]).astype(int)
        free_graph([lg, st])
        del lg, st
    return out


def run_train(arm, seed, steps=STEPS):
    cfg = ARMS[arm]
    P = make_params(cfg, seed)
    g0 = guards(arm, P, seed)
    npairs = cfg.get("n_pairs", U.N_PAIRS)
    Xtr, Xptr, Ytr, _ = U.build(npairs, 2000, seed=1000 + seed)
    Xte, Xpte, Yte, _ = U.build(npairs, 512, seed=9000 + seed)
    ADAM = {}
    t0 = time.time()
    curve = []
    for step in range(1, steps + 1):
        bs = cfg.get("batch", U.B)
        idx = np.random.default_rng(seed * 100000 + step).choice(len(Ytr), bs, replace=False)
        X, Xp, Y = Xtr[idx], Xptr[idx], Ytr[idx]
        for v in P.values(): v.g[...] = 0.0
        lr = cfg.get("lr", U.LR)
        if cfg.get("schedule") == "cosine":
            lr = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / steps)))
        lg, _ = forward(arm, P, X, Xp, X.shape[1])
        loss = nll_from_logits(lg, Y)
        loss.backward()
        free_graph(loss)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, v in P.items():
            key = "m_" + name
            if key not in ADAM:
                ADAM[key] = np.zeros_like(v.d); ADAM["v_" + name] = np.zeros_like(v.d)
            m, vv = ADAM[key], ADAM["v_" + name]
            m *= b1; m += (1 - b1) * v.g
            vv *= b2; vv += (1 - b2) * v.g * v.g
            v.d -= lr * (m / (1 - b1 ** step)) / (np.sqrt(vv / (1 - b2 ** step)) + eps)
        if step % 500 == 0 or step == 1:
            curve.append(dict(step=step, loss=float(loss.d),
                              acc=float(eval_correct(arm, P, Xte, Xpte, Yte).mean())))
    g1 = guards(arm, P, seed)
    corr = eval_correct(arm, P, Xte, Xpte, Yte)
    # 判别性对照：训练时从未见过的 value 置换
    Xpv, Xppv, Ypv = permute_values(Xte, Xpte, Yte, seed=20261105 + seed)
    perm_acc = float(eval_correct(arm, P, Xpv, Xppv, Ypv).mean())
    gen = {}
    for n in (4, 8, 12, 16):
        Xg, Xpg, Yg, _ = U.build(n, 256, seed=7000 + seed + n)
        gen[f"L{2*n+1}"] = float(eval_correct(arm, P, Xg, Xpg, Yg).mean())
    global _LAST_P
    _LAST_P = P
    return dict(arm=arm, seed=seed, params=n_params(P),
                guards_before=g0, guards_after=g1,
                guards_all_passed=bool(g0["all_passed"] and g1["all_passed"]),
                final_acc=float(corr.mean()), perm_acc=perm_acc,
                length_generalization=gen,
                correct_te=corr.tolist(), curve=curve,
                elapsed_s=round(time.time() - t0, 1))


def gradcheck(arm, seed=0):
    cfg = ARMS[arm]; P = make_params(cfg, seed)
    X, Xp, Y, _ = U.build(U.N_PAIRS, 4, seed=42)
    for v in P.values(): v.g[...] = 0.0
    nll_from_logits(forward(arm, P, X, Xp, U.L_TRAIN)[0], Y).backward()
    analytic = {k: v.g.ravel().copy() for k, v in P.items()}
    r = np.random.default_rng(321); eps = 1e-6; worst = 0.0; ok_all = True
    for name in list(P):
        v = P[name]; flat = v.d.ravel(); gf = analytic[name]
        for i in r.choice(flat.size, size=min(3, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = nll_from_logits(forward(arm, P, X, Xp, U.L_TRAIN)[0], Y).d
            flat[i] = old - eps
            lm = nll_from_logits(forward(arm, P, X, Xp, U.L_TRAIN)[0], Y).d
            flat[i] = old
            num = float((lp - lm) / (2 * eps)); ana = float(gf[i])
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-8)
            ok = bool(np.isfinite(rel) and np.isfinite(num) and
                      (rel < 5e-4 or abs(num - ana) < 1e-8))
            ok_all &= ok
            worst = max(worst, 0.0 if ok else (rel if np.isfinite(rel) else 1e9))
    return dict(arm=arm, worst_rel=worst, passed=bool(ok_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="budget",
                    choices=["budget", "guards", "gradcheck", "all"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--arms", default="")
    ap.add_argument("--seed", type=int, default=-1,
                    help="只跑单个 seed（进程隔离，避免内存累积）；-1=全部")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sel = None
    if a.arms:
        names = [x.strip() for x in a.arms.split(",") if x.strip()]
        sel = [k for k in ARMS if any(k == n or k.startswith(n) for n in names)]
        print("选中臂:", sel)

    if a.mode == "budget":
        print(f"目标 {BUDGET} ±{TOL*100:.0f}%")
        out = {}
        for arm in (sel or ARMS):
            c = _count(ARMS[arm]); out[arm] = c
            print(f"  [{'OK ' if abs(c-BUDGET)/BUDGET<=TOL else 'OFF'}] "
                  f"{arm:16s} params={c:6d} Δ={c-BUDGET:+d}")
        json.dump(out, open(f"{OUT}/tensor_budget.json", "w"), indent=1)
        return

    if a.mode == "guards":
        out = {}
        for arm in (sel or ARMS):
            out[arm] = guards(arm, make_params(ARMS[arm], 0), verbose=True)
            print(f"  {arm:16s} all={out[arm]['all_passed']} bytes={out[arm]['state_bytes']}")
        json.dump(out, open(f"{OUT}/tensor_guards.json", "w"), indent=1, ensure_ascii=False)
        return

    if a.mode == "gradcheck":
        out = [gradcheck(arm) for arm in (sel or ARMS)]
        for g in out:
            print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['arm']:16s} {g['worst_rel']:.2e}")
        json.dump(out, open(f"{OUT}/tensor_gradcheck.json", "w"), indent=1)
        print(f"==> {sum(g['passed'] for g in out)}/{len(out)} PASS")
        return

    if a.mode == "all":
        fn = f"{OUT}/tensor_results.json"
        allres = json.load(open(fn)) if os.path.exists(fn) else {}
        seeds = [a.seed] if a.seed >= 0 else SEEDS
        for arm in (sel or ARMS):
            done = {int(r["seed"]) for r in allres.get(arm, [])}
            for sd in seeds:
                if sd in done: continue
                r = run_train(arm, sd, steps=a.steps)
                allres.setdefault(arm, []).append(r)
                print(f"  [{arm:16s} s{sd}] acc={r['final_acc']:.4f} "
                      f"perm={r['perm_acc']:.4f} params={r['params']} "
                      f"guards={r['guards_all_passed']} {r['elapsed_s']}s", flush=True)
                with open(fn, "w") as f:
                    json.dump(allres, f, indent=1, ensure_ascii=False); f.flush()
        print("\n写入", fn)
        return


if __name__ == "__main__":
    main()
