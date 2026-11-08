"""纯 NumPy 实现的 SimplCore 原生模型（约 47 万参数）与同参数 Transformer 对照。

手写前向 + 反向，无 torch / 无 autograd tape。
SGC 长程卷积用 Toeplitz 展开实现（与上游 sgc.py 的
  K[tau,d] = sum_m coef[m,d] * sin(theta_m) * rho_m^tau * sin((tau+1)theta_m)
完全同一个静态核；推理侧另有 IIR 流式 step 验证 O(1) 状态与逐位一致）。

参数量（V=512, D=128, L=2 层, FFN=512, M=32, Kloc=4）：
  Embedding 512*128 = 65536
  每层 SGC: Wu/Wg/Wo 3*16384 + coef 4096 + feed 128 + local 512 + 2 norm 256 = 64245
  每层 FFN: W1/W2 2*65536 + norm 128 = 131200
  2 层 = 390890 ; final norm 128 ; head tied
  合计 ≈ 456,554  (TS 臂把 SGC mixer 换成 4-head attention，参数量相当)
"""
import numpy as np

F32 = np.float32
V = 512
D = 128
N_LAYERS = 2
FFN = 512
M = 32
KLOC = 4
H = 4
DH = D // H
EPS = 1e-6


# ---------------------------------------------------------------- 基础算子
def rmsnorm_fwd(x, w):
    r = np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    xh = x / r
    return xh * w, (r, xh)


def rmsnorm_bwd(dy, w, cache):
    r, xh = cache
    dxh = dy * w
    dx = (dxh - (dxh * xh).mean(-1, keepdims=True) * xh) / r
    dw = (dy * xh).sum(axis=(0, 1))
    return dx, dw


def mm_fwd(x, W):
    return x @ W


def mm_bwd(dy, x, W):
    B, L, Di = x.shape
    Do = W.shape[1]
    dx = dy @ W.T
    dW = x.reshape(-1, Di).T @ dy.reshape(-1, Do)
    return dx, dW


def silu_fwd(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return x * s, (x, s)


def silu_bwd(dy, cache):
    x, s = cache
    return dy * (s + x * s * (1.0 - s))


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = e.sum(axis=axis, keepdims=True)
    return x - m - np.log(s)


# ---------------------------------------------------------------- 网格（冻结）
def make_grid(mode="linear", decay_max=8192.0, floor=0.3):
    """与上游 sgc.py::_init_oscillator_grid 同公式。"""
    lam = np.exp(np.linspace(np.log(1.0), np.log(decay_max), M))
    rho = np.exp(-1.0 / lam)
    if mode == "linear":
        u = (np.arange(M) + 0.5) / M
        theta = floor + (np.pi - 2 * floor) * u
    elif mode == "logfreq":
        theta = np.exp(np.linspace(np.log(np.pi / decay_max),
                                   np.log(0.9 * np.pi / 2), M))
    else:
        raise ValueError(mode)
    return rho.astype(F32), theta.astype(F32)


def psi_eff(rho, theta, L):
    """psi[m,tau] = rho_m^tau * sin((tau+1)theta_m)   —— 上游 sgc.py::build_kernel 口径。

    与流式 step() 的一致性推导：step 的递推
        y_t = 2ρcosθ·y_{t-1} − ρ²·y_{t-2} + sinθ·u_t
    其冲激响应 = sinθ · ρ^t sin((t+1)θ)/sinθ = ρ^t sin((t+1)θ)，
    恰好等于本函数。故整段 Toeplitz 与 O(1) 流式 IIR 数值一致。
    （此前版本误乘了 sinθ，导致双形式偏差 0.32，已修正。）
    """
    t = np.arange(L, dtype=np.float64)[None, :]
    p = (rho[:, None].astype(np.float64) ** t) * np.sin((t + 1.0) * theta[:, None].astype(np.float64))
    return p.astype(F32)


# ---------------------------------------------------------------- SGC mixer
def sgc_fwd(x, P, psi):
    """P: dict(Wu,Wg,Wo,coef,feed,wloc,norm,out_scale). 返回 y, cache"""
    B, L, _ = x.shape
    h, c_norm = rmsnorm_fwd(x, P["norm"])
    up = mm_fwd(h, P["Wu"])
    gp = mm_fwd(h, P["Wg"])
    sg = 1.0 / (1.0 + np.exp(-gp))
    u = up * sg
    K = psi.T @ P["coef"]                       # (L,D)

    # 长程因果卷积（Toeplitz）
    lng = np.zeros_like(u)
    for tau in range(L):
        lng[:, tau:, :] += K[tau][None, None, :] * u[:, :L - tau, :]
    # 局部 depthwise 因果卷积
    loc = np.zeros_like(u)
    for k in range(KLOC):
        if k == 0:
            loc += P["wloc"][:, k][None, None, :] * u
        else:
            loc[:, k:, :] += P["wloc"][:, k][None, None, :] * u[:, :L - k, :]
    inst = P["feed"][None, None, :] * u
    y = lng + loc + inst
    out = mm_fwd(y, P["Wo"])
    return x + P["out_scale"] * out, (c_norm, h, up, gp, sg, u, K, lng, loc, inst, y, x)


def sgc_bwd(dO, P, psi, cache):
    (c_norm, h, up, gp, sg, u, K, lng, loc, inst, y, x) = cache
    B, L, _ = u.shape
    d_y, dWo = mm_bwd(dO * P["out_scale"], y, P["Wo"])
    dP = {"Wo": dWo}
    # 三路
    dK = np.zeros((L, D), dtype=F32)
    du_lng = np.zeros_like(u)
    for tau in range(L):
        seg = d_y[:, tau:, :]
        dK[tau] = (seg * u[:, :L - tau, :]).sum(axis=(0, 1))
        du_lng[:, :L - tau, :] += K[tau][None, None, :] * seg
    dcoef = psi @ dK                          # (M,D)
    du_loc = np.zeros_like(u)
    dwloc = np.zeros_like(P["wloc"])
    for k in range(KLOC):
        seg = d_y[:, k:, :] if k > 0 else d_y
        uu = u[:, :L - k, :] if k > 0 else u
        dwloc[:, k] = (seg * uu).sum(axis=(0, 1))
        add = P["wloc"][:, k][None, None, :] * seg
        if k > 0:
            du_loc[:, :L - k, :] += add
        else:
            du_loc += add
    du_inst = P["feed"][None, None, :] * d_y
    dfeed = (d_y * u).sum(axis=(0, 1))
    du = du_lng + du_loc + du_inst
    # gate
    d_up = du * sg
    d_sg = du * up
    d_gp = d_sg * sg * (1.0 - sg)
    d_h, dWu = mm_bwd(d_up, h, P["Wu"])
    d_h2, dWg = mm_bwd(d_gp, h, P["Wg"])
    d_h = d_h + d_h2
    d_x, dnorm = rmsnorm_bwd(d_h, P["norm"], c_norm)
    dP.update(Wu=dWu, Wg=dWg, coef=dcoef, feed=dfeed, wloc=dwloc, norm=dnorm)
    return dO + d_x, dP                        # 残差分支


# ---------------------------------------------------------------- TS attention
def attn_fwd(x, P):
    B, L, _ = x.shape
    h, c_norm = rmsnorm_fwd(x, P["norm"])
    q = mm_fwd(h, P["Wq"]).reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    k = mm_fwd(h, P["Wk"]).reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    v = mm_fwd(h, P["Wv"]).reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(F32(DH))
    mask = np.triu(np.ones((L, L), dtype=F32), 1) * (-1e9)
    a = softmax(s + mask[None, None, :, :], axis=-1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
    out = mm_fwd(o, P["Wo"])
    return x + out, (c_norm, h, q, k, v, a, o, x)


def attn_bwd(dO, P, cache):
    (c_norm, h, q, k, v, a, o, x) = cache
    B, L, _ = h.shape
    d_o, dWo = mm_bwd(dO, o, P["Wo"])
    do_h = d_o.reshape(B, L, H, DH).transpose(0, 2, 1, 3)
    dv_h = a.transpose(0, 1, 3, 2) @ do_h
    da = do_h @ v.transpose(0, 1, 3, 2)
    ds = a * (da - (a * da).sum(-1, keepdims=True))
    dq_h = ds @ k / np.sqrt(F32(DH))
    dk_h = ds.transpose(0, 1, 3, 2) @ q / np.sqrt(F32(DH))
    dq = dq_h.transpose(0, 2, 1, 3).reshape(B, L, D)
    dk = dk_h.transpose(0, 2, 1, 3).reshape(B, L, D)
    dv = dv_h.transpose(0, 2, 1, 3).reshape(B, L, D)
    d_h, dWq = mm_bwd(dq, h, P["Wq"])
    d_h2, dWk = mm_bwd(dk, h, P["Wk"])
    d_h3, dWv = mm_bwd(dv, h, P["Wv"])
    d_x, dnorm = rmsnorm_bwd(d_h + d_h2 + d_h3, P["norm"], c_norm)
    return dO + d_x, dict(Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo, norm=dnorm)


# ---------------------------------------------------------------- FFN
def ffn_fwd(x, P):
    h, c_norm = rmsnorm_fwd(x, P["norm"])
    a, c_silu = silu_fwd(mm_fwd(h, P["W1"]))
    out = mm_fwd(a, P["W2"])
    return x + out, (c_norm, h, a, c_silu, x)


def ffn_bwd(dO, P, cache):
    (c_norm, h, a, c_silu, x) = cache
    d_a, dW2 = mm_bwd(dO, a, P["W2"])
    d_pre = silu_bwd(d_a, c_silu)
    d_h, dW1 = mm_bwd(d_pre, h, P["W1"])
    d_x, dnorm = rmsnorm_bwd(d_h, P["norm"], c_norm)
    return dO + d_x, dict(W1=dW1, W2=dW2, norm=dnorm)


# ---------------------------------------------------------------- 模型
def init_params(kind, seed=0, grid_mode="linear"):
    rng = np.random.default_rng(seed)
    s = lambda *sh: (rng.standard_normal(sh) * (1 / np.sqrt(sh[0]))).astype(F32)
    P = {"E": (rng.standard_normal((V, D)) * 0.02).astype(F32),
         "fn": np.ones(D, dtype=F32)}
    if kind == "ts":
        # 可学习位置嵌入：纯 causal attention 本身位置无关，不给位置信息会
        # 构成弱化基线（SGC 的阻尼振荡核天然编码位置）。MAXL=512。
        P["pos"] = (rng.standard_normal((512, D)) * 0.01).astype(F32)
    rho, theta = make_grid(grid_mode)
    P["rho"], P["theta"] = rho, theta
    for l in range(N_LAYERS):
        if kind == "sgc":
            P[f"l{l}"] = dict(
                Wu=s(D, D), Wg=s(D, D), Wo=s(D, D),
                coef=(rng.standard_normal((M, D)) * 0.02).astype(F32),
                feed=np.ones(D, dtype=F32),
                wloc=(rng.standard_normal((D, KLOC)) * 0.05).astype(F32),
                norm=np.ones(D, dtype=F32), out_scale=F32(1.0))
            P[f"l{l}"]["_type"] = "sgc"
        else:
            P[f"l{l}"] = dict(Wq=s(D, D), Wk=s(D, D), Wv=s(D, D), Wo=s(D, D),
                              norm=np.ones(D, dtype=F32))
            P[f"l{l}"]["_type"] = "attn"
        P[f"f{l}"] = dict(W1=s(D, FFN), W2=s(FFN, D), norm=np.ones(D, dtype=F32))
    return P


def param_count(P):
    return sum(v.size for k, v in _iter_params(P))


def _iter_params(P):
    for k, v in P.items():
        if k in ("rho", "theta"):
            continue                       # 冻结网格
        if isinstance(v, dict):
            for kk, vv in v.items():
                if kk == "_type" or kk == "out_scale":
                    continue
                yield f"{k}.{kk}", vv
        else:
            yield k, v


def forward(idx, P, psi=None):
    """idx (B,L) int32 -> logits (B,L,V), caches"""
    B, L = idx.shape
    if psi is None and P["l0"]["_type"] == "sgc":
        psi = psi_eff(P["rho"], P["theta"], L)
    x = P["E"][idx]                            # (B,L,D)
    if "pos" in P:
        x = x + P["pos"][:L][None, :, :]
    caches = []
    for l in range(N_LAYERS):
        lm = P[f"l{l}"]
        if lm["_type"] == "sgc":
            x, c = sgc_fwd(x, lm, psi)
        else:
            x, c = attn_fwd(x, lm)
        ff = P[f"f{l}"]
        x, cf = ffn_fwd(x, ff)
        caches.append((lm["_type"], c, cf, l))
    h, c_fn = rmsnorm_fwd(x, P["fn"])
    logits = h @ P["E"].T                     # tied head
    return logits, (caches, c_fn, h, idx, psi)


def backward(dlogits, P, cache):
    """返回与 _iter_params 同键的梯度 dict"""
    caches, c_fn, h, idx, psi = cache
    B, L = idx.shape
    # tied head: dE += dlogits^T @ h ; dh = dlogits @ E
    dh = dlogits @ P["E"]
    dE = dlogits.reshape(-1, V).T @ h.reshape(-1, D)
    d_x, dfn = rmsnorm_bwd(dh, P["fn"], c_fn)
    grads = {"E": dE, "fn": dfn}
    for l in reversed(range(N_LAYERS)):
        typ, c, cf, _ = caches[l]
        lm, ff = P[f"l{l}"], P[f"f{l}"]
        d_x, gf = ffn_bwd(d_x, ff, cf)
        for kk, vv in gf.items():
            grads[f"f{l}.{kk}"] = vv
        if typ == "sgc":
            d_x, gm = sgc_bwd(d_x, lm, psi, c)
        else:
            d_x, gm = attn_bwd(d_x, lm, c)
        for kk, vv in gm.items():
            if kk == "out_scale":
                continue
            grads[f"l{l}.{kk}"] = vv
    # embedding 的 lookup 梯度 + 位置嵌入梯度
    np.add.at(grads["E"], idx.ravel(), d_x.reshape(-1, D))
    if "pos" in P:
        gpos = np.zeros_like(P["pos"])
        gpos[:L] += d_x.sum(axis=0)
        grads["pos"] = gpos
    # 上面 d_x 是残差流回传到 embedding 的梯度；tied head 已单独累加
    return grads


def ce_loss(logits, tgt):
    """label CE，(B,L) -> (loss, dlogits)"""
    B, L, Vo = logits.shape
    ls = log_softmax(logits, axis=-1)
    n = B * L
    loss = -ls.reshape(-1, Vo)[np.arange(n), tgt.ravel()].mean()
    p = np.exp(ls)
    p.reshape(-1, Vo)[np.arange(n), tgt.ravel()] -= 1.0
    return float(loss), (p / n).astype(F32)


def kd_loss(s_logits, t_logits, T=5.0):
    """每位置 KL(teacher||student)，返回 (loss, d z_s)"""
    B, L, Vo = s_logits.shape
    n = B * L
    pt = softmax(t_logits / T, axis=-1)
    ls_s = log_softmax(s_logits / T, axis=-1)
    kl = (pt * (np.log(pt + 1e-9) - ls_s)).sum(-1).mean()
    p_s = np.exp(ls_s)
    dz = (T * (p_s - pt) / n).astype(F32)     # T²·KL 的梯度
    return float(T * T * kl), dz


# ---------------------------------------------------------------- 流式 IIR（SGC）
def init_stream_state(P, B):
    st = [np.zeros((B, M, D), dtype=F32) for _ in range(2 * N_LAYERS)]
    # 每层独立的局部 conv 历史（此前所有层共用同一个 buf，是双形式偏差的主因）
    st += [np.zeros((B, KLOC - 1, D), dtype=F32) for _ in range(N_LAYERS)]
    return st


def stream_step(tok, P, state, psi=None):
    """单 token 推进，O(1) 状态。返回 logits (B,V)。
    只支持 sgc 模型；与 forward 整段 Toeplitz 形式应数值一致。"""
    B = tok.shape[0]
    x = P["E"][tok]
    for l in range(N_LAYERS):
        lm, ff = P[f"l{l}"], P[f"f{l}"]
        # mixer（逐 token RMSNorm 是逐位置独立，合法）
        r = np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
        h = x / r * lm["norm"]
        u = (h @ lm["Wu"]) * (1 / (1 + np.exp(-(h @ lm["Wg"]))))
        rho, th = P["rho"], P["theta"]
        a1 = (2 * rho * np.cos(th)).astype(F32)
        a2 = (-(rho ** 2)).astype(F32)
        c = np.sin(th).astype(F32)
        y1, y2 = state[2 * l], state[2 * l + 1]
        um = u[:, None, :]                       # (B,1,D)
        yn = a1[None, :, None] * y1 + a2[None, :, None] * y2 + c[None, :, None] * um
        state[2 * l], state[2 * l + 1] = yn, y1
        lng = (yn * lm["coef"][None, :, :]).sum(1)     # (B,D)
        # 局部 conv：每层独立的历史缓冲
        bi = 2 * N_LAYERS + l
        buf = state[bi]                         # (B,KLOC-1,D)
        loc = lm["wloc"][:, 0][None, :] * u
        for k in range(1, KLOC):
            loc += lm["wloc"][:, k][None, :] * buf[:, k - 1, :]
        state[bi] = np.concatenate([u[:, None, :], buf[:, :-1, :]], axis=1).astype(F32)
        inst = lm["feed"][None, :] * u
        y = lng + loc + inst
        x = x + lm["out_scale"] * (y @ lm["Wo"])
        # FFN（逐位置）
        r2 = np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
        h2 = x / r2 * ff["norm"]
        pre = h2 @ ff["W1"]
        a = pre / (1 + np.exp(-pre))
        x = x + (a @ ff["W2"])
    r = np.sqrt((x * x).mean(-1, keepdims=True) + EPS)
    hf = x / r * P["fn"]
    return hf @ P["E"].T


# ---------------------------------------------------------------- AdamW
class AdamW:
    def __init__(self, P, lr=3e-4, wd=0.01, b1=0.9, b2=0.95):
        self.lr, self.wd, self.b1, self.b2 = lr, wd, b1, b2
        self.m = {k: np.zeros_like(v) for k, v in _iter_params(P)}
        self.v = {k: np.zeros_like(v) for k, v in _iter_params(P)}
        self.t = 0

    def step(self, P, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k, v in _iter_params(P):
            g = grads.get(k)
            if g is None:
                continue
            g = np.clip(g, -1.0, 1.0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / bc1
            vh = self.v[k] / bc2
            v -= lr * (mh / (np.sqrt(vh) + 1e-8) + self.wd * v)

    def state(self):
        return dict(t=self.t, m=self.m, v=self.v)

    def load(self, s):
        self.t, self.m, self.v = s["t"], s["m"], s["v"]
