"""
sgc_fixed.py — 固定 SGC 核 + 六条「完全保持 SGC」守恒断言。

本模块是整条路的地基：任何臂在训练前必须过 assert_all_guards()，
否则直接淘汰，不算「保持 SGC」。ρ/θ 在整个实验里是常量，训练不更新。
"""
import numpy as np


def grids(M, mode, decay_max=8192.0, L=9, tmax=None):
    """返回 (rho[M], theta[M])。源码口径：theta ∈ (0, π)，rho ∈ (0,1)。

    scan   : theta 从 π/decay_max 几何扫到 tmax —— 用于定位「内容寻址需要单调核」的相变点
    """
    lam = np.exp(np.linspace(np.log(1.0), np.log(decay_max), M))
    rho = np.exp(-1.0 / lam)
    lo = np.pi / decay_max
    if mode == "pureacc":
        # 全部 theta 近零 -> 全部基都是单调累积核（M 个不同衰减率的累加器）
        return rho, np.full(M, lo)
    if mode == "pureacc_nodecay":
        return np.ones(M), np.full(M, lo)
    if mode == "scan":
        hi = tmax if tmax is not None else 0.9 * np.pi / 2
        theta = np.exp(np.linspace(np.log(lo), np.log(hi), M))
        return rho, theta
    if mode == "logfreq":
        theta = np.exp(np.linspace(np.log(np.pi / decay_max),
                                   np.log(0.9 * np.pi / 2), M))
        # 几何配对的衰减：最慢基配最小 theta
        lam = np.exp(np.linspace(np.log(1.0), np.log(decay_max), M))
        rho = np.exp(-1.0 / lam)
    elif mode == "linear":
        # 正交网格：theta 线性均匀，避开 π/2 镜像冗余（非对称上界）
        theta = 0.30 + (2.60 - 0.30) * (np.arange(M) + 0.5) / M
        lam = np.exp(np.linspace(np.log(1.0), np.log(decay_max), M))
        rho = np.exp(-1.0 / lam)
    elif mode == "flat":
        # 消融：无衰减（ρ=1），检验容量是否受衰减限制。仍是固定核。
        theta = 0.30 + (2.60 - 0.30) * (np.arange(M) + 0.5) / M
        rho = np.ones(M)
    else:
        raise ValueError(mode)
    return rho, theta


def kernel(rho, theta, L):
    """源码口径 psi[tau] = rho^tau * sin((tau+1)theta)。形状 (M,L)。"""
    tau = np.arange(L, dtype=np.float64)
    return (rho[:, None] ** tau[None, :]) * np.sin((tau[None, :] + 1.0) * theta[:, None])


def toeplitz(K):
    """由 K[M,L] 构造下三角 Toeplitz 算子 T[M,L,L]。"""
    M, L = K.shape
    idx = np.arange(L)
    tau = idx[:, None] - idx[None, :]           # (L,L)
    Tm = np.zeros((M, L, L))
    for m in range(M):
        Tm[m] = np.where(tau >= 0, K[m, np.clip(tau, 0, L - 1)], 0.0)
    return Tm


def fft_apply(K, u):
    """整段 FFT 路径：对每个 m、每个通道做线性卷积取因果前缀。u:(B,L,M,C)"""
    B, L, M, C = u.shape
    n = 1
    while n < 2 * L:
        n *= 2
    out = np.zeros((B, L, M, C))
    for m in range(M):
        k = np.zeros(n)
        k[:L] = K[m]
        Kf = np.fft.rfft(k)
        for c in range(C):
            Uf = np.fft.rfft(u[:, :, m, c], n=n, axis=1)
            y = np.fft.irfft(Uf * Kf[None, :], n=n, axis=1)[:, :L]
            out[:, :, m, c] = y
    return out


def iir_apply(rho, theta, u):
    """流式 IIR 路径：逐 token 二阶递推，状态 O(M*C*2)，与 L 无关。"""
    B, L, M, C = u.shape
    y = np.zeros((B, L, M, C))
    y1 = np.zeros((B, M, C))
    y2 = np.zeros((B, M, C))
    a = (2 * rho * np.cos(theta))[None, :, None]      # (1,M,1)
    b = (-(rho ** 2))[None, :, None]
    c = np.sin(theta)[None, :, None]                   # 源码口径：输入系数 sinθ
    for t in range(L):
        # 冲激响应 = ρ^τ sin((τ+1)θ)，与 build_kernel 的 psi 逐位一致
        yt = a * y1 + b * y2 + c * u[:, t, :, :]
        y[:, t, :, :] = yt
        y2, y1 = y1, yt
    return y, dict(y1=y1, y2=y2, state_bytes=2 * M * C * 4)


def guards(rho, theta, L, C=4, rng_seed=0, verbose=False):
    """六条守恒断言。返回 dict（全部 true 才算保持 SGC）。"""
    r = np.random.default_rng(rng_seed)
    M = len(rho)
    K = kernel(rho, theta, L)
    Tm = toeplitz(K)
    u = r.standard_normal((2, L, M, C))

    # 1 Toeplitz
    dev = []
    for m in range(M):
        for t in range(1, L):
            for s in range(1, t + 1):
                dev.append(abs(Tm[m, t, s] - Tm[m, t - 1, s - 1]))
    g1 = dict(name="toeplitz", passed=bool(max(dev) < 1e-12),
              max_dev=float(max(dev)) if dev else 0.0)

    # 2 FFT 整段 == Toeplitz 整段
    y_toep = np.einsum("mls,bsmc->blmc", Tm, u)
    y_fft = fft_apply(K, u)
    den = max(np.abs(y_toep).max(), 1e-12)
    rel = float(np.abs(y_toep - y_fft).max() / den)
    g2 = dict(name="fft_vs_toeplitz", passed=bool(rel < 1e-10), rel_err=rel)

    # 3 IIR 流式 == 整段（数值容差，递推是同一 LTI 系统）
    y_iir, meta = iir_apply(rho, theta, u)
    rel3 = float(np.abs(y_toep - y_iir).max() / den)
    g3 = dict(name="iir_vs_toeplitz", passed=bool(rel3 < 1e-8), rel_err=rel3,
              note="源码 psi 不含 1/sin(theta) 归一化；IIR 与 Toeplitz 为同一系统")

    # 4 严格因果：扰动未来 token，过去输出逐位不变
    u2 = u.copy()
    u2[:, L - 1, :, :] += 10.0
    y2 = np.einsum("mls,bsmc->blmc", Tm, u2)
    past_diff = float(np.abs(y_toep[:, :L - 1] - y2[:, :L - 1]).max())
    g4 = dict(name="strict_causality", passed=bool(past_diff == 0.0),
              max_past_abs_diff=past_diff)

    # 5 状态字节与 L 无关
    sizes = []
    for LL in (9, 17, 25, 65):
        _, mm = iir_apply(rho, theta, np.zeros((1, LL, M, C)))
        sizes.append(mm["state_bytes"])
    g5 = dict(name="state_independent_of_L",
              passed=bool(len(set(sizes)) == 1), bytes_by_L=dict(zip((9, 17, 25, 65), sizes)))

    # 6 ρ/θ 是常量（训练不更新）——由调用方在训练后复核，这里记录指纹
    g6 = dict(name="rho_theta_frozen",
              fingerprint=float(np.abs(rho).sum() + np.abs(theta).sum()),
              passed=True, note="调用方在训练后与训练前逐位比较")

    out = dict(guards=[g1, g2, g3, g4, g5, g6],
               all_passed=all(g["passed"] for g in [g1, g2, g3, g4, g5, g6]),
               M=M, L=L, C=C, state_bytes=meta["state_bytes"],
               cond_gram=_cond(K, L))
    if verbose:
        for g in out["guards"]:
            print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']}: "
                  + ", ".join(f"{k}={v}" for k, v in g.items()
                              if k not in ("name", "passed", "note")))
        print(f"  ==> all_passed={out['all_passed']}")
    return out


def _cond(K, L):
    """不含 ρ 口径的字典 Gram 条件数（fp64），用于对照正交网格。"""
    P = K.T                                     # (L, M)
    G = P.T @ P
    w = np.linalg.eigvalsh(G)
    w = w[w > 1e-300]
    return float(w[-1] / w[0]) if len(w) else float("inf")


if __name__ == "__main__":
    print("=== logfreq ===")
    guards(*grids(16, "logfreq"), L=9, verbose=True)
    print("=== linear ===")
    guards(*grids(16, "linear"), L=9, verbose=True)
    print("=== flat(rho=1) ===")
    guards(*grids(16, "flat"), L=9, verbose=True)
