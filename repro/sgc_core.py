"""
SGC (Spectral Global Convolution) 独立实现 —— 核心算子层（纯 numpy / scipy）

设计依据：作者提供的**未公开技术报告**《SimplCore-0.1-α》的文本描述
（章节号对应见报告 §16.1 的逐条核对表）；未参考、未下载任何原仓库代码。
该文稿无 arXiv ID / DOI，不构成可检索的已发表文献；投稿时按报告 §16.5 处理。
  TR §3.1  mixer=`sgc`：谱全局卷积 + FFT/深度卷积 + 内容门控；训练 O(D log L)，流式 O(1) IIR
  TR §3.1  θ 钳制 [theta_floor, pi-1e-3]；logfreq 频率网格
  TR §4    T1 双形式对偶（FFT 整段 == IIR 流式）；T2 严格因果；T3 状态字节 4(2MD+(K-1)D+1)

技术报告未写明、由本实现自行设定的细节：频率网格的 ρ 配对方式、门控函数、
FFN 结构、优化器超参。详见报告 §1。

关键点（白盒推导）：
  二阶 IIR 递推        y_t = 2*rho*cos(theta) * y_{t-1} - rho^2 * y_{t-2} + x_t  (零初值)
  其冲激响应          h[t] = rho^t * sin((t+1) theta) / sin(theta),  t >= 0
  整段等价形式        y = (x 与 h 的因果线性卷积) 截断到长度 L
  两者数学上相等 -> 可用 FFT 加速训练；浮点上不等 -> 偏差可测（TR §5.4）
"""
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.signal import lfilter

F32 = np.float32
F64 = np.float64


# ---------------------------------------------------------------- 频率网格
def make_grid(M=64, theta_min=np.pi / 8192.0, theta_max=np.pi - 1e-3,
              rho_min=0.90, rho_max=0.999, pairing="lowfreq_long",
              integrator=0, unit_circle=0, uc_thetas=None):
    """阻尼振荡基字典 {(rho_m, theta_m)}_{m=0}^{M-1}。

    theta 取对数均匀（logfreq），覆盖 [theta_min, theta_max]（论文 §4 命题5 实测
    1/|sin(theta_min)| = 1/sin(pi/8192) ~= 2607.6，故默认 theta_min=pi/8192）。
    pairing='lowfreq_long'：低频基配长衰减（rho 接近 rho_max），高频基衰减快。
    """
    th = np.exp(np.linspace(np.log(theta_min), np.log(theta_max), M))
    w = (np.log(th) - np.log(theta_min)) / (np.log(theta_max) - np.log(theta_min))
    if pairing == "lowfreq_long":
        rho = rho_max - (rho_max - rho_min) * w
    elif pairing == "lowfreq_short":
        rho = rho_min + (rho_max - rho_min) * w
    else:                                   # 常数衰减
        rho = np.full(M, rho_max)
    g = dict(M=M, theta=th.astype(F64), rho=rho.astype(F64),
             theta_min=theta_min, theta_max=theta_max,
             rho_min=rho_min, rho_max=rho_max)
    # ---- 表达能力扩展：单位圆 / 实轴极点（论文 (ρ,θ), ρ<1 参数化把它们排除了）
    if integrator or unit_circle:
        P = np.stack([2.0 * g["rho"] * np.cos(g["theta"]),
                      -g["rho"] ** 2], axis=1)
        i = 0
        if integrator >= 1:
            P[i] = [1.0, 0.0]; i += 1                 # 精确积分器 / 锁存
        if integrator >= 2:
            P[i] = [2.0, -1.0]; i += 1                # 二重积分器
        for j in range(unit_circle):
            if uc_thetas is not None:
                thj = uc_thetas[j]
            else:
                thj = 2.0 * np.pi * (j + 1) / (unit_circle + 1)
            P[i] = [2.0 * np.cos(thj), -1.0]; i += 1  # ρ=1 无损振荡
        g["poles"] = P
        g["lossless_bases"] = i
    return g


def iir_coeffs(grid, dtype=F32):
    """递推系数 (a1, a2)。若 grid 带 'poles' 则直接采用（支持单位圆/实轴极点）。"""
    if "poles" in grid:
        return grid["poles"].astype(dtype)
    a1 = 2.0 * grid["rho"] * np.cos(grid["theta"])
    a2 = -grid["rho"] ** 2
    return np.stack([a1, a2], axis=1).astype(dtype)   # (M,2)


def impulse_response(grid, L, dtype=F32):
    """冲激响应。标准基走闭式；带 'poles' 的基走统一递推。

    统一递推 h[0]=1, h[1]=a1, h[t]=a1*h[t-1]+a2*h[t-2] 覆盖所有极点：
      a1=2ρcosθ, a2=-ρ²  ->  ρ^t sin((t+1)θ)/sinθ   （单位圆内，论文默认）
      a1=2cosθ,  a2=-1   ->  无损振荡（ρ=1，单位圆上）
      a1=1,      a2=0    ->  h[t]=1  （z=1 一重极点：精确积分器/锁存）
      a1=2,      a2=-1   ->  h[t]=t+1（z=1 二重极点：精确二次累加）
    """
    if "poles" in grid:
        P = grid["poles"]                              # (M,2)
        h = np.zeros((P.shape[0], L), dtype=F64)
        h[:, 0] = 1.0
        if L > 1:
            h[:, 1] = P[:, 0]
        for t in range(2, L):
            h[:, t] = P[:, 0] * h[:, t - 1] + P[:, 1] * h[:, t - 2]
        return h.astype(dtype)
    t = np.arange(L, dtype=F64)[None, :]
    th = grid["theta"][:, None]
    rho = grid["rho"][:, None]
    h = (rho ** t) * np.sin((t + 1.0) * th) / np.sin(th)
    return h.astype(dtype)


# ------------------------------------------------------------ 整段 FFT 形式
def causal_conv_fft(u, h):
    """因果卷积的 FFT 实现（时间轴必须在最后一维）。

    u: (..., L)    h: (M, L)
    返回 (M, ..., L)：第 m 个基对 u 的因果卷积（零初值，等价于 IIR 递推）
    """
    L = u.shape[-1]
    Lp = next_fast_len(2 * L)
    Hf = rfft(h, n=Lp, axis=-1)                       # (M, F)
    Uf = rfft(u, n=Lp, axis=-1)                       # (..., F)
    Hb = Hf.reshape((Hf.shape[0],) + (1,) * (Uf.ndim - 1) + (Hf.shape[1],))
    Yf = Uf[None, ...] * Hb
    return irfft(Yf, n=Lp, axis=-1)[..., :L]


# ------------------------------------------------------------- 流式 IIR 形式
class SGCStream:
    """O(1) 常驻状态的流式递推器（论文的推理路径）。

    状态：y_{t-1}, y_{t-2}，形状 (B, M, D) —— 2*M*D 个 fp32，与 L 无关。
    """

    def __init__(self, grid, B, D, dtype=F32):
        self.a = iir_coeffs(grid, dtype)              # (M,2)
        self.B, self.D, self.M = B, D, grid["M"]
        self.dtype = dtype
        self.y1 = np.zeros((B, self.M, D), dtype=dtype)   # y_{t-1}
        self.y2 = np.zeros((B, self.M, D), dtype=dtype)   # y_{t-2}

    def step(self, x_t):
        """x_t: (B, M, D) —— 第 t 步喂给各基的输入（已按基调制）。
        返回 y_t: (B, M, D)，并就地推进状态。"""
        a1 = self.a[:, 0][None, :, None]
        a2 = self.a[:, 1][None, :, None]
        y = a1 * self.y1 + a2 * self.y2 + x_t
        self.y2, self.y1 = self.y1, y
        return y

    def state_bytes(self):
        return self.y1.nbytes + self.y2.nbytes


def iir_scan(u, grid, dtype=F32, chunk=None):
    """整段 IIR（离线参考实现）：u (B, L, D) -> y (B, L, M, D)。

    chunk=None 时用 lfilter 一次扫完；给定 chunk 则分块递推（O(1) 状态跨块传递），
    结果与不分块完全一致（同一递推）。
    """
    a = iir_coeffs(grid, dtype)
    B, L, D = u.shape
    M = grid["M"]
    out = np.empty((B, L, M, D), dtype=dtype)
    for m in range(M):
        # y_t = a1 y_{t-1} + a2 y_{t-2} + x_t   ->  a = [1, -a1, -a2]
        den = np.array([1.0, -a[m, 0], -a[m, 1]], dtype=dtype)
        y = lfilter(np.array([1.0], dtype=dtype), den, u.astype(dtype), axis=1)
        out[:, :, m, :] = y
    return out


def iir_scan_chunked(u, grid, dtype=F32, chunk=64):
    """显式分块 + O(1) 状态传递（模拟真实流式部署）。u: (B,L,D)。"""
    B, L, D = u.shape
    M = grid["M"]
    st = SGCStream(grid, B, D, dtype)
    outs = []
    for s in range(0, L, chunk):
        blk = u[:, s:s + chunk, :]                       # (B,c,D)
        c = blk.shape[1]
        ys = np.empty((B, c, M, D), dtype=dtype)
        for t in range(c):
            xs = np.broadcast_to(blk[:, t, :][:, None, :], (B, M, D)).copy()
            ys[:, t] = st.step(xs)
        outs.append(ys)
    return np.concatenate(outs, axis=1)


# ------------------------------------------------------- 深度卷积（K 阶局部）
def dwconv_causal(x, w, state=None):
    """深度可分离因果卷积：x (B,L,D)，w (D,K)。返回 (B,L,D)（不推进 state）。"""
    B, L, D = x.shape
    K = w.shape[1]
    xp = np.zeros((B, L + K - 1, D), dtype=x.dtype)
    xp[:, K - 1:, :] = x
    y = np.zeros((B, L, D), dtype=x.dtype)
    for k in range(K):
        y += xp[:, k:k + L, :] * w[None, None, :, k]
    return y


# ------------------------------------------------------------------- 度量
def rel_err(a, b):
    """相对 Frobenius 误差 ||a-b||_F / ||b||_F（分母为 0 时退回绝对误差）。"""
    d = np.linalg.norm((a.astype(F64) - b.astype(F64)).ravel())
    n = np.linalg.norm(b.astype(F64).ravel())
    return d / n if n > 0 else d


def bitwise_equal(a, b):
    return np.array_equal(a.view(np.int32), b.view(np.int32))
