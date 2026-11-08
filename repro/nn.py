"""
极简自动微分 + 两个模型（纯 numpy，白盒实现）

模型 A：SGC（论文的 SimplCore mixer）—— 谱全局卷积 + 深度卷积 + 内容门控
模型 B：Transformer 基线（causal self-attention + RoPE-free sin 位置编码 + FFN）

两者参数量刻意做得接近（论文 §5.5 声称 SGC 用 72% 参数对照）。
"""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
import sgc_core as S

DT = np.float32          # 全局计算 dtype（gradcheck 时切 float64）
F32 = DT


# ------------------------------------------------------------------ autograd
class T:
    __slots__ = ("v", "g", "kids", "bwd")

    def __init__(self, v, kids=(), bwd=None):
        self.v = v if isinstance(v, np.ndarray) else np.asarray(v, dtype=F32)
        self.g = np.zeros_like(self.v)
        self.kids = kids
        self.bwd = bwd

    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            n, exp = stack.pop()
            if exp:
                topo.append(n)
                continue
            if id(n) in seen:
                continue
            seen.add(id(n))
            stack.append((n, True))
            for k in n.kids:
                stack.append((k, False))
        self.g = np.ones_like(self.v)
        for n in reversed(topo):      # 拓扑序：父节点先于子节点
            if n.bwd is not None:
                n.bwd(n.g)


def _sum_to(g, shape):
    if g.shape == shape:
        return g
    n = g.ndim - len(shape)
    if n > 0:
        g = g.sum(axis=tuple(range(n)))
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


def _acc(x, g):
    x.g += _sum_to(g, x.v.shape)


def add(a, b):
    y = T(a.v + b.v, (a, b))
    y.bwd = lambda g: (_acc(a, g), _acc(b, g))
    return y


def mul(a, b):
    y = T(a.v * b.v, (a, b))
    y.bwd = lambda g: (_acc(a, g * b.v), _acc(b, g * a.v))
    return y


def matmul(a, b):
    """支持 batched 左操作数 (...,D) @ (D,V)。"""
    y = T(a.v @ b.v, (a, b))
    def bwd(g):
        _acc(a, g @ b.v.T)
        if a.v.ndim == 2:
            _acc(b, a.v.T @ g)
        else:
            ax = list(range(a.v.ndim - 1))
            _acc(b, np.tensordot(a.v, g, axes=(ax, list(range(g.ndim - 1)))))
    y.bwd = bwd
    return y


def silu(x):
    s = 1.0 / (1.0 + np.exp(-x.v))
    y = T(x.v * s, (x,))
    y.bwd = lambda g: _acc(x, g * (s + x.v * s * (1 - s)))
    return y


def gelu(x):
    """精确 GELU: y = x * Phi(x), Phi(x)=0.5(1+erf(x/sqrt2)); dy/dx = Phi + x*phi(x)。"""
    from scipy.special import erf
    z = x.v / np.sqrt(2.0)
    e = 0.5 * (1.0 + erf(z))                       # Phi(x)
    de = np.exp(-0.5 * x.v ** 2) / np.sqrt(2.0 * np.pi)   # phi(x)
    y = T(x.v * e, (x,))
    y.bwd = lambda g: _acc(x, g * (e + x.v * de))
    return y


def rmsnorm(x, w):
    rms = np.sqrt((x.v ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    xn = x.v / rms
    y = T(xn * w.v, (x, w))
    def bwd(g):
        gx = g * w.v
        _acc(w, (g * xn).sum(axis=tuple(range(g.ndim - 1))))
        n = x.v.shape[-1]
        gg = (gx - (gx * xn).mean(axis=-1, keepdims=True) * xn) / rms
        _acc(x, gg)
    y.bwd = bwd
    return y


def gather(E, idx):
    """E: (V,D) 参数；idx: (B,L) int -> (B,L,D)"""
    y = T(E.v[idx], (E,))
    def bwd(g):
        np.add.at(E.g, idx, g)
    y.bwd = bwd
    return y


def last(x):
    """取序列最后一个位置 (B,L,V) -> (B,V)（保留梯度）。"""
    y = T(np.ascontiguousarray(x.v[:, -1, :]), (x,))
    y.bwd = lambda g: _acc_at_last(x, g)
    return y


def _acc_at_last(x, g):
    x.g[:, -1, :] += g


def flatten2d(x):
    """(B,L,V) -> (B*L, V)（保留梯度）。"""
    B, L, V = x.v.shape
    y = T(x.v.reshape(B * L, V), (x,))
    y.bwd = lambda g: _acc(x, g.reshape(B, L, V))
    return y


def tied_head(hf, E):
    """logits = hf @ E.T；梯度回传 E（绑定词表，否则 head 梯度丢失）。"""
    y = T(hf.v @ E.v.T, (hf, E))
    def bwd(g):
        _acc(hf, g @ E.v)
        _acc(E, np.tensordot(g, hf.v, axes=([0, 1], [0, 1])))
    y.bwd = bwd
    return y


def slice_tail(x, k):
    """取前 L-k 个时间步 (B,L,V) -> (B,L-k,V)（保留梯度，LM 用）。"""
    y = T(np.ascontiguousarray(x.v[:, :-k, :]), (x,))
    def bwd(g):
        x.g[:, :-k, :] += g
    y.bwd = bwd
    return y


def sum_all(x):
    """标量 reduce（保留梯度），用于 gradcheck 的随机投影。"""
    y = T(np.array(x.v.sum(), dtype=x.v.dtype), (x,))
    y.bwd = lambda g: _acc(x, np.full_like(x.v, g))
    return y


def proj_scalar(x, W):
    """<x, W> 的 autograd 版本（gradcheck 用）。"""
    return sum_all(mul(x, T(W)))


def cross_entropy(logits, targets):
    """logits (N,V) float32, targets (N,) int；返回标量 loss T"""
    z = logits.v - logits.v.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    n = logits.v.shape[0]
    lp = (z - np.log(np.exp(z).sum(axis=-1, keepdims=True)))
    loss = -lp[np.arange(n), targets].mean()
    y = T(np.array(loss, dtype=F32), (logits,))
    def bwd(g):
        gp = p.copy()
        gp[np.arange(n), targets] -= 1.0
        _acc(logits, (gp / n) * g)
    y.bwd = bwd
    return y


# ------------------------------------------------- SGC 的谱混合层（含反向）
def corr_causal(dY, h):
    """(...,L) 与 (M,L) 的反因果相关：out[t] = sum_{tau>=t} dY[tau] h[tau-t]。"""
    L = dY.shape[-1]
    Lp = next_fast_len(2 * L)
    rev = np.ascontiguousarray(dY[..., ::-1])
    c = S.causal_conv_fft(rev, h)              # (M,...,L)
    return c[..., ::-1]


def sgc_mix(u, H, c, grid=None):
    """u (B,L,D) -> y (B,L,D)：y[t,d] = sum_m c[m,d] * (h_m * u)[t,d]"""
    B, L, D = u.v.shape
    M = H.shape[0]
    Lp = next_fast_len(2 * L)
    # 布局 (B,M,D,L)：FFT 轴放最后一维，避免 scipy 内部转置
    Uf = rfft(np.moveaxis(u.v, 1, -1), n=Lp, axis=-1)     # (B,D,F)
    Hf = rfft(H, n=Lp, axis=-1)                            # (M,F)
    Yf = Uf[:, None, :, :] * Hf[None, :, None, :]          # (B,M,D,F)
    Y = irfft(Yf, n=Lp, axis=-1)[..., :L]                  # (B,M,D,L)
    out = np.einsum('bmdl,md->bld', Y, c.v, optimize=True)
    y = T(np.ascontiguousarray(out, dtype=F32), (u, c))
    def bwd(g):                                            # g (B,L,D)
        dY = np.einsum('bld,md->bmdl', g, c.v, optimize=True)
        _acc(c, np.einsum('bmdl,bld->md', Y, g, optimize=True))
        dYf = rfft(np.moveaxis(np.ascontiguousarray(dY), -1, -1), n=Lp, axis=-1)  # (B,M,D,F)
        dUf = (dYf * np.conj(Hf)[None, :, None, :]).sum(axis=1)                   # (B,D,F)
        _acc(u, np.moveaxis(irfft(dUf, n=Lp, axis=-1)[..., :L], -1, 1))
    y.bwd = bwd
    return y


def dwconv(x, w):
    """因果深度卷积：x (B,L,D)，w (D,K)。

    翻核写法：wr[j] = w[:, K-1-j]，out[t] = sum_j wr[j] * x[t-j]（零填充）。
    前向/反向同一套切片，避免负索引环绕。
    """
    B, L, D = x.v.shape
    K = w.v.shape[1]
    xp = np.zeros((B, L + K - 1, D), dtype=x.v.dtype)
    xp[:, K - 1:, :] = x.v                              # xp[j] = x[j-(K-1)]
    # out[t] = sum_k w[:,k] * x[t-(K-1-k)] = sum_k w[:,k] * xp[t+k]
    out = np.zeros((B, L, D), dtype=x.v.dtype)
    for k in range(K):
        out += xp[:, k:k + L, :] * w.v[None, None, :, k]
    y = T(out, (x, w))
    def bwd(g):
        gext = np.zeros((B, L + K - 1, D), dtype=g.dtype)
        gext[:, :L, :] = g                              # gext[t] = g[t]
        dx = np.zeros((B, L, D), dtype=g.dtype)
        dwv = np.zeros_like(w.v)
        for k in range(K):
            # dx[s] = sum_k w[:,k] * g[s+K-1-k]
            dx += gext[:, K - 1 - k:K - 1 - k + L, :] * w.v[None, None, :, k]
            dwv[:, k] = (g * xp[:, k:k + L, :]).sum(axis=(0, 1))
        _acc(x, dx)
        _acc(w, dwv)
    y.bwd = bwd
    return y


# -------------------------------------------------------------------- 模型
def param(*shape, rng, scale=None):
    scale = 1.0 / np.sqrt(shape[-1]) if scale is None else scale
    return T((rng.standard_normal(shape) * scale).astype(F32))


class SGCNet:
    def __init__(self, V, D=64, M=32, n_layers=4, K=4, ctx=2048, rng=None, grid=None):
        rng = rng or np.random.default_rng(0)
        self.V, self.D, self.M, self.nl, self.K = V, D, M, n_layers, K
        self.grid = grid or S.make_grid(M=M, rho_max=0.999, rho_min=0.9)
        self.hfun = None          # 若设定 hfun(L)->(M,L)，则用它代替 impulse_response
        self.hfun = None          # 若设定 hfun(L)->(M,L)，则用它代替 impulse_response
        self.E = param(V, D, rng=rng, scale=0.08)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(dict(
                w1=param(D, rng=rng), Wu=param(D, D, rng=rng), Wg=param(D, D, rng=rng),
                Wo=param(D, D, rng=rng), c=param(M, D, rng=rng), wdw=param(D, K, rng=rng),
                w2=param(D, rng=rng), Wup=param(4 * D, D, rng=rng), Wdn=param(D, 4 * D, rng=rng)))
        self.wf = param(D, rng=rng)

    def params(self):
        ps = [self.E, self.wf]
        for l in self.layers:
            ps += list(l.values())
        return ps

    def n_params(self):
        return int(sum(p.v.size for p in self.params()))

    def forward(self, idx, L=None):
        L = idx.shape[1]
        H = (S.impulse_response(self.grid, L) if self.hfun is None else self.hfun(L))
        h = gather(self.E, idx)                                     # (B,L,D)
        for l in self.layers:
            xn = rmsnorm(h, l["w1"])
            u = matmul(xn, l["Wu"])
            y = sgc_mix(u, H, l["c"])
            y = dwconv(y, l["wdw"])
            g = silu(matmul(xn, l["Wg"]))
            h = add(h, matmul(mul(y, g), l["Wo"]))
            xn2 = rmsnorm(h, l["w2"])
            h = add(h, matmul(gelu(matmul(xn2, l["Wdn"])), l["Wup"]))
        hf = rmsnorm(h, self.wf)
        return tied_head(hf, self.E)

    def kernels(self):
        return None


class TSNet:
    """标准 causal Transformer（对照组）。"""

    def __init__(self, V, D=64, n_heads=4, n_layers=4, ctx=2048, rng=None):
        rng = rng or np.random.default_rng(0)
        self.V, self.D, self.H, self.nl, self.ctx = V, D, n_heads, n_layers, ctx
        self.dh = D // n_heads
        self.E = param(V, D, rng=rng, scale=0.08)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(dict(
                w1=param(D, rng=rng),
                Wq=param(D, D, rng=rng), Wk=param(D, D, rng=rng),
                Wv=param(D, D, rng=rng), Wo=param(D, D, rng=rng),
                w2=param(D, rng=rng), Wup=param(4 * D, D, rng=rng), Wdn=param(D, 4 * D, rng=rng)))
        self.wf = param(D, rng=rng)

    def params(self):
        ps = [self.E, self.wf]
        for l in self.layers:
            ps += list(l.values())
        return ps

    def n_params(self):
        return int(sum(p.v.size for p in self.params()))

    def _pos(self, L):
        p = np.arange(L)[:, None]
        i = np.arange(self.D)[None, :]
        ang = p / (10000 ** (2 * (i // 2) / self.D))
        pe = np.where(i % 2 == 0, np.sin(ang), np.cos(ang))
        return pe.astype(F32)[None, :, :]

    def forward(self, idx, L=None):
        B, L = idx.shape
        h = add(gather(self.E, idx), T(self._pos(L)))
        dh = self.dh
        mask = np.triu(np.full((L, L), -np.inf, dtype=F32), 1)
        for l in self.layers:
            xn = rmsnorm(h, l["w1"])
            q = matmul(xn, l["Wq"]).v.reshape(B, L, self.H, dh).transpose(0, 2, 1, 3)
            k = matmul(xn, l["Wk"]).v.reshape(B, L, self.H, dh).transpose(0, 2, 1, 3)
            v = matmul(xn, l["Wv"]).v.reshape(B, L, self.H, dh).transpose(0, 2, 1, 3)
            s = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh) + mask
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=-1, keepdims=True)
            o = (p @ v).transpose(0, 2, 1, 3).reshape(B, L, self.D)
            # 注意力分支的反向（手算，避免为 mask/softmax 再写 autograd 节点）
            lq = l["Wq"]; lk = l["Wk"]; lv = l["Wv"]; xn_ = xn
            def attn_bwd_factory(q, k, v, p, xn_, lq, lk, lv, B, L, H, dh, o):
                def bwd_cache(go):
                    go4 = go.reshape(B, L, H, dh).transpose(0, 2, 1, 3)
                    gv = p.transpose(0, 1, 3, 2) @ go4
                    gp = go4 @ v.transpose(0, 1, 3, 2)
                    gs = p * (gp - (gp * p).sum(axis=-1, keepdims=True))
                    gs = gs / np.sqrt(dh)
                    gq = gs @ k
                    gk = gs.transpose(0, 1, 3, 2) @ q
                    gqkv = np.concatenate([gq.transpose(0, 2, 1, 3).reshape(B, L, H * dh),
                                           gk.transpose(0, 2, 1, 3).reshape(B, L, H * dh),
                                           gv.transpose(0, 2, 1, 3).reshape(B, L, H * dh)], axis=-1)
                    return gqkv
                return bwd_cache
            ot = T(o.astype(F32), (xn_, lq, lk, lv))
            cache = attn_bwd_factory(q, k, v, p, xn_, lq, lk, lv, B, L, self.H, dh, o)
            def bwd(g, cache=cache, xn_=xn_, lq=lq, lk=lk, lv=lv, B=B, L=L, dh=dh, D=self.D):
                gqkv = cache(g)
                for W, sl in ((lq, slice(0, D)), (lk, slice(D, 2 * D)), (lv, slice(2 * D, 3 * D))):
                    _acc(W, np.einsum('bld,blc->dc', xn_.v, gqkv[:, :, sl], optimize=True))
                _acc(xn_, gqkv[:, :, :D] @ lq.v.T + gqkv[:, :, D:2 * D] @ lk.v.T
                     + gqkv[:, :, 2 * D:] @ lv.v.T)
            ot.bwd = bwd
            h = add(h, matmul(ot, l["Wo"]))
            xn2 = rmsnorm(h, l["w2"])
            h = add(h, matmul(gelu(matmul(xn2, l["Wdn"])), l["Wup"]))
        hf = rmsnorm(h, self.wf)
        return tied_head(hf, self.E)


# ------------------------------------------------------------------ 优化器
class AdamW:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.98, wd=0.01, clip=1.0):
        self.p = params
        self.lr, self.b1, self.b2, self.wd, self.clip = lr, b1, b2, wd, clip
        self.m = [np.zeros_like(p.v) for p in params]
        self.v = [np.zeros_like(p.v) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        norms = []
        for i, p in enumerate(self.p):
            g = p.g
            norms.append(float(np.linalg.norm(g.ravel())))
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.v = (p.v - self.lr * (mh / (np.sqrt(vh) + 1e-8) + self.wd * p.v)).astype(F32)
            p.g[:] = 0.0
        return float(np.linalg.norm(np.array(norms)))


# =====================================================================
# selective SGC：把 LTI 二阶 IIR 变成输入相关（LTV），换取内容寻址能力
#
#   y_t = alpha_t ⊙ (a1·y_{t-1} + a2·y_{t-2}) + beta_t ⊙ u_t
#   alpha_t = sigmoid(u_t @ Wa)   ∈ (0,1)^D   遗忘门（alpha→0 = 立刻清空状态）
#   beta_t  = sigmoid(u_t @ Wb)   ∈ (0,1)^D   写入门（控制是否写入当前 token）
#
# 代价：破坏卷积结构 → 训练不能再走 FFT，必须 O(L·D) 循环 scan（即论文的 srf 档
#       O(D·L/chunk)）。收益：模型可以按内容决定"记住 / 丢弃"，这正是 key→value
#       绑定所必需的最小机制。状态仍是 (y_{t-1}, y_{t-2}) 两拍 → 仍是 O(1)。
# =====================================================================
def selective_mix(u, c, Wa, Wb, grid, ba=None, bb=None):
    B, L, D = u.v.shape
    M = grid["M"]
    a = __import__("sgc_core").iir_coeffs(grid, np.float64)      # (M,2) fp64 求稳
    a1 = a[:, 0][None, :, None]                                   # (1,M,1)
    a2 = a[:, 1][None, :, None]
    Av = u.v
    # 遗忘门 bias 必须初始化为正（alpha≈1）--> 初始退化为 LTI，梯度不沿时间消失
    al = 1.0 / (1.0 + np.exp(-(Av @ Wa.v + (0.0 if ba is None else ba.v))))
    be = 1.0 / (1.0 + np.exp(-(Av @ Wb.v + (0.0 if bb is None else bb.v))))
    Y = np.empty((L, B, M, D), dtype=Av.dtype)
    S = np.empty((L, B, M, D), dtype=Av.dtype)
    y1 = np.zeros((B, M, D), dtype=Av.dtype)
    y2 = np.zeros((B, M, D), dtype=Av.dtype)
    for t in range(L):
        s = a1 * y1 + a2 * y2
        yt = al[:, t, :][:, None, :] * s + be[:, t, :][:, None, :] * Av[:, t, :][:, None, :]
        S[t] = s
        Y[t] = yt
        y2, y1 = y1, yt
    out = np.einsum('lbmd,md->bld', Y, c.v, optimize=True)
    kids = (u, c, Wa, Wb) + ((ba, bb) if ba is not None else ())
    y = T(np.ascontiguousarray(out, dtype=F32), kids)
    def bwd(g):
        _acc(c, np.einsum('lbmd,bld->md', Y, g, optimize=True))
        dYacc = np.einsum('bld,md->lbmd', g, c.v, optimize=True)
        du = np.zeros((B, L, D), dtype=Av.dtype)
        dWa = np.zeros_like(Wa.v)
        dWb = np.zeros_like(Wb.v)
        dba = np.zeros(D, dtype=Av.dtype)
        dbb = np.zeros(D, dtype=Av.dtype)
        for t in range(L - 1, -1, -1):
            dy = dYacc[t]
            dS = al[:, t, :][:, None, :] * dy
            # 门控梯度（sigmoid 局部导数）
            dal = (S[t] * dy).sum(axis=1)                          # (B,D)
            dbe = (Av[:, t, :][:, None, :] * dy).sum(axis=1)
            gal = dal * al[:, t, :] * (1.0 - al[:, t, :])
            gbe = dbe * be[:, t, :] * (1.0 - be[:, t, :])
            dWa += Av[:, t, :].T @ gal
            dWb += Av[:, t, :].T @ gbe
            dba = dba + gal.sum(axis=0) if ba is not None else None
            dbb = dbb + gbe.sum(axis=0) if bb is not None else None
            du[:, t, :] += (be[:, t, :][:, None, :] * dy).sum(axis=1)
            du[:, t, :] += gal @ Wa.v.T + gbe @ Wb.v.T
            if t - 1 >= 0:
                dYacc[t - 1] += a1[0] * dS
            if t - 2 >= 0:
                dYacc[t - 2] += a2[0] * dS
        _acc(u, du)
        _acc(Wa, dWa)
        _acc(Wb, dWb)
        if ba is not None:
            _acc(ba, dba)
        if bb is not None:
            _acc(bb, dbb)
    y.bwd = bwd
    return y


class SelSGCNet(SGCNet):
    """SGCNet 的 selective 变体：mixer 换成输入相关的二阶 IIR。"""

    def __init__(self, V, D=48, M=32, n_layers=2, K=4, rng=None, grid=None):
        super().__init__(V=V, D=D, M=M, n_layers=n_layers, K=K, rng=rng, grid=grid)
        self.grid = grid or __import__("sgc_core").make_grid(M=M, rho_max=0.999, rho_min=0.9)
        for l in self.layers:
            l["Wa"] = param(D, D, rng=rng)
            l["Wb"] = param(D, D, rng=rng)
            l["ba"] = T(np.full(D, 4.0, dtype=F32))     # 遗忘门正偏置
            l["bb"] = T(np.zeros(D, dtype=F32))

    def forward(self, idx, L=None):
        L = idx.shape[1]
        h = gather(self.E, idx)
        for l in self.layers:
            xn = rmsnorm(h, l["w1"])
            u = matmul(xn, l["Wu"])
            y = selective_mix(u, l["c"], l["Wa"], l["Wb"], self.grid, l["ba"], l["bb"])
            y = dwconv(y, l["wdw"])
            g = silu(matmul(xn, l["Wg"]))
            h = add(h, matmul(mul(y, g), l["Wo"]))
            xn2 = rmsnorm(h, l["w2"])
            h = add(h, matmul(gelu(matmul(xn2, l["Wdn"])), l["Wup"]))
        return tied_head(rmsnorm(h, self.wf), self.E)
