"""SFSC / FD-CAO 双方案判别性原型（纯 NumPy，fp64，手写 BPTT）。

四臂等骨架对照，只换 mixer 核心：
  static   : 静态 SGC 二阶 IIR（LTI）
  attn     : causal self-attention（上界参照）
  sfsc     : 状态反馈谱卷积（ρ_t/θ_t 由状态 EMA 调制 → LTV/双线性）
  fdcao    : 频域内容寻址振荡器（相位共振，每 token 自己相位写入）
  fdcao_p  : 同上，但写入用前一 token 相位（显式 (key,value) 配对）

三个任务：
  bind  : MQAR 精确 key→value 绑定（chance 1/16）
  mod   : 上下文调制（整体 regime 切换规则，非精确跳转）
  osc   : 周期/连续信号频率分类（SGC 主场）

外加数值检验：时变核是否 Toeplitz → FFT 对偶是否成立。
"""
import argparse
import json
import time
import numpy as np

np.seterr(all="ignore")

D = 64
M = 16
V = 64          # 词表
NCLS = 16
B = 32
BETA = 0.9      # EMA 系数


# ---------------------------------------------------------------- 工具
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return x - m - np.log(e.sum(axis=axis, keepdims=True))


def linear_grid(m=16, floor=0.3):
    u = (np.arange(m) + 0.5) / m
    th = floor + (np.pi - 2 * floor) * u
    rho = np.exp(-1.0 / np.exp(np.linspace(0, np.log(8192.0), m)))
    return rho, th


def init_params(kind, seed=0):
    rng = np.random.default_rng(seed)
    rho0, th0 = linear_grid(M)
    P = {}
    P["rho"] = rho0.copy()
    P["theta"] = th0.copy()
    P["coef"] = rng.standard_normal((M, D)) * 0.05
    if kind == "sfsc":
        # 初始化成"接近静态核"：b=logit(初值)，W 小随机 → 训练才学出调制
        P["W_rho"] = rng.standard_normal((M, D)) * 0.01
        P["W_theta"] = rng.standard_normal((M, D)) * 0.01
        P["b_rho"] = logit(np.clip(rho0, 0.01, 0.99))
        P["b_theta"] = logit(np.clip(th0 / np.pi, 0.01, 0.99))
    if kind in ("fdcao", "fdcao_p"):
        P["w_phi"] = rng.standard_normal(D) * 0.05
        P["b_phi"] = np.array(0.0)
        P["W_v"] = rng.standard_normal((D, D)) * (1 / np.sqrt(D))
    if kind in ("attn", "attn_lr"):
        r = 8
        for k in ("Wq", "Wk", "Wv", "Wo"):
            if kind == "attn":
                P[k] = rng.standard_normal((D, D)) * (1 / np.sqrt(D))
            else:
                P[k + "a"] = rng.standard_normal((D, r)) * (1 / np.sqrt(D))
                P[k + "b"] = rng.standard_normal((r, D)) * (1 / np.sqrt(r))
    return P


def _params(P):
    for k, v in P.items():
        yield k, v


def n_params(P):
    return sum(v.size for _, v in _params(P))


# ---------------------------------------------------------------- mixer: static
def fwd_static(u, P):
    """u (B,L,D) -> out (B,L,D)"""
    B, L, _ = u.shape
    rho, th = P["rho"], P["theta"]
    a = 2 * rho * np.cos(th)
    b = -(rho ** 2)
    y1 = np.zeros((B, M, D)); y2 = np.zeros((B, M, D))
    out = np.zeros((B, L, D))
    ys = []
    for t in range(L):
        yn = a[None, :, None] * y1 + b[None, :, None] * y2 + u[:, t, :][:, None, :]
        ys.append((y1.copy(), y2.copy(), yn))
        out[:, t, :] = (P["coef"][None, :, :] * yn).sum(1)
        y2, y1 = y1, yn
    return out, dict(ys=ys, a=a, b=b)


def bwd_static(u, P, ca, d_out):
    B, L, _ = u.shape
    ys, a, b = ca["ys"], ca["a"], ca["b"]
    g = {k: np.zeros_like(v) for k, v in _params(P)}
    d_u = np.zeros_like(u)
    D1 = np.zeros((B, M, D)); D2 = np.zeros((B, M, D))
    for t in reversed(range(L)):
        y1p, y2p, yn = ys[t]
        dyn = P["coef"][None, :, :] * d_out[:, t, :][:, None, :] + D1
        g["coef"] += (d_out[:, t, :][:, None, :] * yn).sum(0)
        d_u[:, t, :] = dyn.sum(1)
        da = (dyn * y1p).sum((0, 2))
        db = (dyn * y2p).sum((0, 2))
        dy1p = a[None, :, None] * dyn + D2
        dy2p = b[None, :, None] * dyn
        # a=2ρcosθ, b=−ρ²
        g["rho"] += da * 2 * np.cos(P["theta"]) + db * (-2 * P["rho"])
        g["theta"] += da * (-2 * P["rho"] * np.sin(P["theta"]))
        D1, D2 = dy1p, dy2p
    return d_u, g


# ---------------------------------------------------------------- mixer: SFSC
def fwd_sfsc(u, P):
    B, L, _ = u.shape
    y1 = np.zeros((B, M, D)); y2 = np.zeros((B, M, D))
    e = np.zeros((B, D))
    out = np.zeros((B, L, D))
    rec = []
    As, Bs, Rs, Ts, Es, Zs = [], [], [], [], [], []
    for t in range(L):
        m_t = y1.mean(1)                            # (B,D) 状态整体趋势
        e = BETA * e + (1 - BETA) * m_t
        z_r = e @ P["W_rho"].T + P["b_rho"][None, :]
        z_t = e @ P["W_theta"].T + P["b_theta"][None, :]
        rho = sigmoid(z_r) * 0.999                 # (B,M)
        th = np.pi * sigmoid(z_t)                  # (B,M)
        a = 2 * rho * np.cos(th)                    # (B,M)
        b = -(rho ** 2)
        yn = a[:, :, None] * y1 + b[:, :, None] * y2 + u[:, t, :][:, None, :]
        rec.append((y1.copy(), y2.copy(), yn, m_t, e.copy(), z_r, z_t, rho, th, a, b))
        out[:, t, :] = (P["coef"][None, :, :] * yn).sum(1)
        y2, y1 = y1, yn
    return out, dict(rec=rec)


def bwd_sfsc(u, P, ca, d_out):
    B, L, _ = u.shape
    rec = ca["rec"]
    g = {k: np.zeros_like(v) for k, v in _params(P)}
    d_u = np.zeros_like(u)
    D1 = np.zeros((B, M, D)); D2 = np.zeros((B, M, D))
    d_e_carry = np.zeros((B, D))
    pending_y1 = np.zeros((B, M, D))               # EMA 分支传给 t−1 的 y1 梯度
    for t in reversed(range(L)):
        y1p, y2p, yn, m_t, e_t, z_r, z_t, rho, th, a, b = rec[t]
        dyn = P["coef"][None, :, :] * d_out[:, t, :][:, None, :] + D1 + pending_y1
        g["coef"] += (d_out[:, t, :][:, None, :] * yn).sum(0)
        d_u[:, t, :] = dyn.sum(1)
        da = (dyn * y1p).sum(2)                    # (B,M)
        db = (dyn * y2p).sum(2)
        dy1p = a[:, :, None] * dyn + D2
        dy2p = b[:, :, None] * dyn
        # ρ_t/θ_t ← a=2ρcosθ, b=−ρ²
        d_rho = da * 2 * np.cos(th) + db * (-2 * rho)
        d_th = da * (-2 * rho * np.sin(th))
        s_r = sigmoid(z_r); s_t = sigmoid(z_t)
        d_zr = d_rho * 0.999 * s_r * (1 - s_r)
        d_zt = d_th * np.pi * s_t * (1 - s_t)
        g["W_rho"] += d_zr.T @ e_t
        g["W_theta"] += d_zt.T @ e_t
        g["b_rho"] += d_zr.sum(0)
        g["b_theta"] += d_zt.sum(0)
        d_e_local = d_zr @ P["W_rho"] + d_zt @ P["W_theta"]     # (B,D)
        d_e = d_e_local + d_e_carry
        d_m = (1 - BETA) * d_e
        # e_t = β e_{t−1} + (1−β) m_t，m_t = mean_M(y1_{t−1})
        pending_y1 = np.broadcast_to((d_m / M)[:, None, :], (B, M, D)).copy()
        d_e_carry = BETA * d_e
        D1, D2 = dy1p, dy2p
    return d_u, g


# ---------------------------------------------------------------- mixer: FD-CAO
def fwd_fdcao(u, P, pair=False):
    B, L, _ = u.shape
    th, rho = P["theta"], P["rho"]
    z = u @ P["w_phi"] + P["b_phi"]
    phi = np.pi * sigmoid(z)                        # (B,L)
    gq = np.cos(th[None, None, :] - phi[:, :, None])       # (B,L,M) 读出相位
    gw = gq.copy()
    if pair:
        # 写入用前一 token 的相位：显式把 (key at t−1, value at t) 绑定
        gw = np.concatenate([np.cos(th[None, None, :] + np.zeros((B, 1, 1))),
                             gq[:, :-1, :]], axis=1)
    w = u @ P["W_v"].T                              # (B,L,D)
    y = np.zeros((B, M, D))
    out = np.zeros((B, L, D))
    ys = []
    for t in range(L):
        y = rho[None, :, None] * y + gw[:, t, :][:, :, None] * w[:, t, :][:, None, :]
        ys.append(y.copy())
        out[:, t, :] = (P["coef"][None, :, :] * gq[:, t, :][:, :, None] * y).sum(1)
    return out, dict(ys=ys, phi=phi, gq=gq, gw=gw, w=w, z=z, s=sigmoid(z))


def bwd_fdcao(u, P, ca, d_out, pair=False):
    """g = cos(θ−φ):  ∂g/∂φ = +sin(θ−φ),  ∂g/∂θ = −sin(θ−φ)."""
    B, L, _ = u.shape
    ys, gq, gw, w, s, phi = ca["ys"], ca["gq"], ca["gw"], ca["w"], ca["s"], ca["phi"]
    th, rho = P["theta"], P["rho"]
    g = {k: np.zeros_like(v) for k, v in _params(P)}
    d_u = np.zeros_like(u)
    Dn = np.zeros((B, M, D))
    # 相位差分项：读出用 φ_t；写入用 φ_t (fdcao) 或 φ_{t−1} (fdcao_p)
    sin_q = np.sin(th[None, None, :] - phi[:, :, None])            # (B,L,M)
    sin_w = sin_q.copy()
    if pair:
        sin_w = np.concatenate([np.sin(th[None, None, :] + np.zeros((B, 1, 1))),
                                sin_q[:, :-1, :]], axis=1)
    for t in reversed(range(L)):
        y = ys[t]
        base = P["coef"][None, :, :] * gq[:, t, :][:, :, None] * d_out[:, t, :][:, None, :]
        dy = base + Dn
        g["coef"] += (d_out[:, t, :][:, None, :] * gq[:, t, :][:, :, None] * y).sum(0)
        d_gq_t = (P["coef"][None, :, :] * d_out[:, t, :][:, None, :] * y).sum(2)
        d_gw_t = (dy * w[:, t, :][:, None, :]).sum(2)
        d_w_t = (dy * gw[:, t, :][:, :, None]).sum(1)
        g["W_v"] += d_w_t.T @ u[:, t, :]
        d_u[:, t, :] += d_w_t @ P["W_v"]
        g["rho"] += (dy * (ys[t - 1] if t > 0 else np.zeros_like(y))).sum((0, 2))
        Dn = rho[None, :, None] * dy
        # θ 梯度（读出相位 + 写入相位）
        g["theta"] += (-d_gq_t * sin_q[:, t, :]).sum(0) + (-d_gw_t * sin_w[:, t, :]).sum(0)
        # φ 梯度（读出来自 φ_t；写入来自 φ_t 或 φ_{t−1}）
        kq = s[:, t] * (1 - s[:, t]) * np.pi
        d_phi_q = (d_gq_t * sin_q[:, t, :]).sum(1) * kq
        d_u[:, t, :] += d_phi_q[:, None] * P["w_phi"][None, :]
        g["w_phi"] += d_phi_q @ u[:, t, :]
        g["b_phi"] += d_phi_q.sum()
        d_phi_w = (d_gw_t * sin_w[:, t, :]).sum(1)
        src = t - 1 if pair and t > 0 else t
        if pair and t == 0:
            continue                      # gw[0]=cos(θ) 与 φ 无关
        kw = s[:, src] * (1 - s[:, src]) * np.pi
        d_z_w = d_phi_w * kw
        d_u[:, src, :] += d_z_w[:, None] * P["w_phi"][None, :]
        g["w_phi"] += d_z_w @ u[:, src, :]
        g["b_phi"] += d_z_w.sum()
    return d_u, g


# ---------------------------------------------------------------- mixer: attention
def proj(x, P, k):
    if k in P:
        return x @ P[k], None
    return x @ P[k + "a"] @ P[k + "b"], (x, P[k + "a"], P[k + "b"])


def proj_bwd(dy, c, k):
    if c is None:
        return dy, {}
    x, A, B = c                       # x (...,Din), A (Din,r), B (r,Dout)
    N = x.size // x.shape[-1]
    X = x.reshape(N, x.shape[-1])
    DY = dy.reshape(N, dy.shape[-1])
    Z = X @ A                          # (N,r)
    dZ = DY @ B.T                     # (N,r)
    dA = X.T @ dZ
    dB = Z.T @ DY
    dX = dZ @ A.T
    return dX.reshape(x.shape), {k + "a": dA, k + "b": dB}


def fwd_attn(u, P):
    B, L, _ = u.shape
    H, DH = 4, D // 4
    cq, ck, cv, co = {}, {}, {}, {}
    qq, cq = proj(u, P, "Wq"); kk, ck = proj(u, P, "Wk")
    vv, cv = proj(u, P, "Wv")
    q = qq.reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    k = kk.reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    v = vv.reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(DH)
    mask = np.triu(np.ones((L, L)), 1) * (-1e9)
    a = softmax(s + mask[None, None], axis=-1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
    out, co = proj(o, P, "Wo")
    return out, dict(q=q, k=k, v=v, a=a, o=o, u=u, cq=cq, ck=ck, cv=cv, co=co)


def bwd_attn(u, P, ca, d_out):
    B, L, _ = u.shape
    H, DH = 4, D // 4
    q, k, v, a, o = ca["q"], ca["k"], ca["v"], ca["a"], ca["o"]
    g = {kk: np.zeros_like(vv) for kk, vv in _params(P)}
    if ca["co"] is None:
        g["Wo"] = o.reshape(-1, D).T @ d_out.reshape(-1, D)
        do = d_out @ P["Wo"].T
    else:
        do, go = proj_bwd(d_out, ca["co"], "Wo")
        for kk, vv in go.items():
            g[kk] = vv
    do_h = do.reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    dv = (a.transpose(0, 1, 3, 2) @ do_h).transpose(0, 2, 1, 3).reshape(B, L, D)
    da = do_h @ v.transpose(0, 1, 3, 2)
    ds = a * (da - (a * da).sum(-1, keepdims=True))
    dq = (ds @ k / np.sqrt(DH)).transpose(0, 2, 1, 3).reshape(B, L, D)
    dk = (ds.transpose(0, 1, 3, 2) @ q / np.sqrt(DH)).transpose(0, 2, 1, 3).reshape(B, L, D)
    d_u = np.zeros_like(ca["u"])
    for nm, dx, c in (("Wq", dq, ca["cq"]), ("Wk", dk, ca["ck"]), ("Wv", dv, ca["cv"])):
        if c is None:
            g[nm] = ca["u"].reshape(-1, D).T @ dx.reshape(-1, D)
            dback = dx @ P[nm].T
        else:
            dback, gg = proj_bwd(dx, c, nm)
            for kk, vv in gg.items():
                g[kk] = vv
        d_u += dback
    return d_u, g


MIXERS = {
    "static": (fwd_static, bwd_static, False),
    "sfsc": (fwd_sfsc, bwd_sfsc, False),
    "fdcao": (fwd_fdcao, bwd_fdcao, False),
    "fdcao_p": (fwd_fdcao, bwd_fdcao, True),
    "attn": (fwd_attn, bwd_attn, False),
    "attn_lr": (fwd_attn, bwd_attn, False),
}


# ---------------------------------------------------------------- 模型（共享骨架）
def init_model(kind, seed=0):
    rng = np.random.default_rng(seed + 777)
    return dict(P=init_params(kind, seed),
                E=rng.standard_normal((V, D)) * 0.05,
                Wo=rng.standard_normal((D, NCLS)) * (1 / np.sqrt(D)),
                bo=np.zeros(NCLS))


def model_fwd(idx, md, kind):
    fwd, _bwd, pair = MIXERS[kind]
    u = md["E"][idx]
    out, ca = fwd(u, md["P"], **({"pair": pair} if kind == "fdcao_p" else {}))
    h = u[:, -1, :] + out[:, -1, :]              # 残差（所有臂一致）
    return h @ md["Wo"] + md["bo"], (ca, h, idx, u)


def model_bwd(dlogits, md, kind, cache):
    ca, h, idx, u = cache
    _fwd, bwd, pair = MIXERS[kind]
    g = {"Wo": h.T @ dlogits, "bo": dlogits.sum(0)}
    dh = dlogits @ md["Wo"].T
    d_out = np.zeros_like(u)
    d_out[:, -1, :] = dh
    d_u_m, gP = bwd(u, md["P"], ca, d_out, **({"pair": pair} if kind == "fdcao_p" else {}))
    d_u = dh[:, None, :] * np.eye(1)[None] if False else np.zeros_like(u)
    d_u[:, -1, :] += dh
    d_u = d_u + d_u_m
    gE = np.zeros_like(md["E"])
    np.add.at(gE, idx.ravel(), d_u.reshape(-1, D))
    return dict(P=gP, E=gE, Wo=g["Wo"], bo=g["bo"])


class AdamW:
    def __init__(self, md, lr=3e-3):
        self.lr = lr
        self.m = {}
        self.v = {}
        for k, v in md.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    self.m[f"P.{kk}"] = np.zeros_like(vv); self.v[f"P.{kk}"] = np.zeros_like(vv)
            else:
                self.m[k] = np.zeros_like(v); self.v[k] = np.zeros_like(v)
        self.t = 0

    def step(self, md, grads):
        self.t += 1
        grads = {f"P.{k}": v for k, v in grads["P"].items()} | {
            k: v for k, v in grads.items() if k != "P"}
        bc1 = 1 - 0.9 ** self.t; bc2 = 1 - 0.95 ** self.t
        for k, v in list(_flat(md)):
            gg = grads.get(k)
            if gg is None: continue
            gg = np.clip(gg, -1, 1)
            self.m[k] = 0.9 * self.m[k] + 0.1 * gg
            self.v[k] = 0.95 * self.v[k] + 0.05 * gg * gg
            v -= self.lr * ((self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + 1e-8) + 0.01 * v)


def _flat(md):
    for k, v in md.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                yield f"P.{kk}", vv
        else:
            yield k, v


# ---------------------------------------------------------------- 任务
def gen_bind(rng, n, L=13):
    """MQAR: k1 v1 ... k4 v4 [filler*] q → 输出 value。chance 1/16"""
    idx = np.zeros((n, L), dtype=np.int32); y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ks = rng.choice(np.arange(0, 16), size=4, replace=False)
        vs = rng.choice(np.arange(16, 32), size=4, replace=False)
        seq = []
        for k, v in zip(ks, vs): seq += [int(k), int(v)]
        qi = int(rng.integers(0, 4))
        seq += [int(rng.integers(32, V)) for _ in range(L - 9)] + [int(ks[qi])]
        idx[i] = seq; y[i] = int(vs[qi]) - 16      # 类别 0..15
    return idx, y


def gen_bind1(rng, n, L=9):
    """单跳内容寻址：每个组合 token 同时编码 (key,value)，query=key，
    输出该 key 对应的 value。单层 attention 一跳可解（真正的上界对照）。
    chance = 1/8。"""
    idx = np.zeros((n, L), dtype=np.int32); y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        keys = rng.choice(4, size=4, replace=False)
        vals = rng.choice(8, size=4, replace=True)
        combos = (32 + keys * 8 + vals).astype(np.int32)
        order = rng.permutation(4)
        fill = rng.choice(np.arange(16, 32), size=4, replace=True)
        q = int(rng.integers(0, 4))
        idx[i] = np.concatenate([combos[order], fill, [q]])
        y[i] = int(vals[int(np.where(keys == q)[0][0])])
    return idx, y


def gen_mod(rng, n, L=25):
    """上下文调制：整体 A/B 占比决定 regime，末尾 query x，
    输出 = (x + c_regime) mod 16。regime 是全局平滑统计，不是精确跳转。"""
    idx = np.zeros((n, L), dtype=np.int32); y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        reg = int(rng.integers(0, 2))
        pa = 0.8 if reg == 0 else 0.2
        body = np.where(rng.random(L - 2) < pa, 0, 1)      # A=0 / B=1
        x = int(rng.integers(0, 16))
        idx[i, :L - 2] = body
        idx[i, L - 2] = 40                                  # 分隔
        idx[i, L - 1] = x
        y[i] = (x + (1 if reg == 0 else -1)) % 16
    return idx, y


def gen_osc(rng, n, L=32):
    """周期信号：频率类别 ∈ {1..8}，序列 = sin(2π f t/16) 量化到词表 + 噪声。
    输出频率类别。静态振荡基天然擅长。"""
    idx = np.zeros((n, L), dtype=np.int32); y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        f = int(rng.integers(1, 9)); y[i] = f - 1
        t = np.arange(L)
        s = np.sin(2 * np.pi * f * t / 16.0)
        q = np.clip(((s + 1) * 0.5 * 20 + rng.normal(0, 0.6, L)).round(), 0, 19).astype(int)
        idx[i] = q + 32
    return idx, y


TASKS = {"bind": gen_bind, "bind1": gen_bind1, "mod": gen_mod, "osc": gen_osc}
STEPS = {"bind": 2500, "bind1": 2500, "mod": 3000, "osc": 1500}


# ---------------------------------------------------------------- 训练 / 评估
def run_one(task, kind, seed, steps=None):
    steps = steps or STEPS[task]
    rng = np.random.default_rng(seed * 1000 + 7)
    md = init_model(kind, seed)
    opt = AdamW(md, lr=3e-3)
    gen = TASKS[task]
    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx, y = gen(rng, B)
        logits, cache = model_fwd(idx, md, kind)
        ls = log_softmax(logits, axis=-1)
        loss = -ls[np.arange(B), y].mean()
        p = np.exp(ls); p[np.arange(B), y] -= 1.0
        grads = model_bwd(p / B, md, kind, cache)
        opt.step(md, grads)
        if step % 250 == 0:
            acc = eval_acc(task, kind, md, seed=12345 + seed)
            hist.append([step, float(loss), float(acc)])
    return dict(acc=float(eval_acc(task, kind, md, seed=12345 + seed)),
                final_train_loss=float(loss), hist=hist,
                params=n_params(md["P"]), seconds=time.time() - t0)


def eval_acc(task, kind, md, seed=999, n=512):
    rng = np.random.default_rng(seed)
    idx, y = TASKS[task](rng, n)
    tot = 0
    for s in range(0, n, B):
        lg, _ = model_fwd(idx[s:s + B], md, kind)
        tot += (lg.argmax(1) == y[s:s + B]).sum()
    return tot / n


# ---------------------------------------------------------------- 梯度检验
def gradcheck(kind, seed=0):
    rng = np.random.default_rng(seed)
    md = init_model(kind, seed)
    idx, _ = gen_bind(rng, 2)
    W = rng.standard_normal((2, NCLS))

    def f():
        lg, _ = model_fwd(idx, md, kind)
        return float((lg * W).sum() / 2)

    dlogits = (W / 2)
    cache = model_fwd(idx, md, kind)[1]
    grads = model_bwd(dlogits, md, kind, cache)
    grads = {f"P.{k}": v for k, v in grads["P"].items()} | {
        k: v for k, v in grads.items() if k != "P"}
    flat = dict(_flat(md))
    out = []; bad = 0
    for name, arr in flat.items():
        gv = grads.get(name)
        if gv is None:
            print("  MISSING", name); bad += 1; continue
        a = arr.reshape(-1); g = gv.reshape(-1)
        sel = rng.choice(a.size, size=min(5, a.size), replace=False)
        for j in sel:
            o = a[j]
            a[j] = o + 1e-6; vp = f()
            a[j] = o - 1e-6; vm = f()
            a[j] = o
            num = (vp - vm) / 2e-6
            rel = abs(num - g[j]) / max(abs(g[j]), 1e-8)
            out.append((name, rel))
            if rel > 5e-4: bad += 1
    worst = max(out, key=lambda x: x[1])
    print(f"  [{kind:8s}] 最坏相对误差 {worst[1]:.2e} ({worst[0]}), 超差 {bad}/{len(out)}")
    return bad == 0


# ---------------------------------------------------------------- 理论检验：时变核是否 Toeplitz
def check_toeplitz(seed=3):
    """SFSC 的 ρ_t/θ_t 随 t 变 → 冲激响应 K(t,s) 是否只依赖 t−s？
    若否，则单一 FFT（Toeplitz 卷积）无法表示该算子 → 双形式对偶失效。"""
    rng = np.random.default_rng(seed)
    L = 12
    md = init_model("sfsc", seed)
    P = md["P"]
    # 随机时变参数（模拟训练后状态反馈产生的 ρ_t）
    rho_t = 0.90 + 0.09 * rng.random(L)
    th_t = 0.4 + 1.8 * rng.random(L)
    K = np.zeros((L, L))
    for s in range(L):
        y1 = np.zeros(M); y2 = np.zeros(M)
        for t in range(s, L):
            a = 2 * rho_t[t] * np.cos(th_t[t]); b = -rho_t[t] ** 2
            yn = a * y1 + b * y2 + (1.0 if t == s else 0.0) * np.sin(th_t[t])
            K[t, s] = yn.sum()
            y2, y1 = y1, yn
    toe_err = 0.0; n = 0
    for t in range(1, L):
        for s in range(t):
            ref = K[t - s, 0]
            toe_err += abs(K[t, s] - ref); n += 1
    # 用"平均核"做 FFT 卷积，与真实时变算子对随机输入的响应比较
    avg = np.array([K[t - s, 0] if t >= s else 0 for t in range(L) for s in range(L)]).reshape(L, L)
    x = rng.standard_normal(L)
    y_true = np.array([K[t, :t + 1] @ x[:t + 1] for t in range(L)])
    y_fft = np.array([avg[t, :t + 1] @ x[:t + 1] for t in range(L)])
    rel = np.linalg.norm(y_true - y_fft) / max(np.linalg.norm(y_true), 1e-12)
    return dict(toeplitz_mean_abs_deviation=float(toe_err / n),
                fft_avg_kernel_rel_error=float(rel),
                is_toeplitz=bool(toe_err / n < 1e-9),
                example_K_row_5=K[5, :6].round(4).tolist(),
                example_K_row_8=K[8, :9].round(4).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gradcheck", "toeplitz", "train", "all"], default="all")
    ap.add_argument("--task", default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="results_sfsc_fdcao.json")
    a = ap.parse_args()

    if a.mode in ("gradcheck", "all"):
        print("=== fp64 梯度检验 ===")
        oks = {k: gradcheck(k) for k in MIXERS}
        print("==>", "PASS" if all(oks.values()) else "FAIL", oks)
        if not all(oks.values()):
            raise SystemExit("梯度检验未过，不进入训练")

    if a.mode in ("toeplitz", "all"):
        print("\n=== 时变核 Toeplitz / FFT 对偶检验 ===")
        r = check_toeplitz()
        print(json.dumps(r, indent=1, ensure_ascii=False))

    if a.mode in ("train", "all"):
        tasks = [a.task] if a.task else list(TASKS)
        arms = [a.arm] if a.arm else list(MIXERS)
        seeds = [a.seed] if a.seed is not None else [0, 1, 2]
        path = a.out
        try:
            res = json.load(open(path))
        except Exception:
            res = {}
        for task in tasks:
            res.setdefault(task, {})
            for arm in arms:
                res[task].setdefault(arm, {})
                for sd in seeds:
                    if str(sd) in res[task][arm]:
                        continue
                    r = run_one(task, arm, sd)
                    res[task][arm][str(sd)] = r
                    json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
                    print(f"[{task}/{arm}/seed{sd}] acc={r['acc']:.4f} "
                          f"params={r['params']} {r['seconds']:.0f}s", flush=True)
        print("\n->", path)


if __name__ == "__main__":
    main()
