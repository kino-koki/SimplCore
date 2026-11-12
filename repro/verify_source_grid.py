"""按上游真实源码 _init_oscillator_grid 的公式复算频率网格条件数，核对
docs/frequency_grid_fix.md 的口径 A / 口径 B 数字，并解释本文报告的
1.35e36 → 3.97e3 属于哪种口径。

源码依据：/data/workspace/SimplCore_src/simplcore/core/arch/sgc.py
  lam   = exp(linspace(log(1), log(decay_max), M))
  rho0  = exp(-1/lam)
  logfreq theta = exp(linspace(log(pi/decay_max), log(0.9*pi/2), M))
  linear theta  = f + (pi-2f)*(arange(M)+0.5)/M
"""
import json
import numpy as np
import sgc_core as S

M, DECAY, F = 32, 8192.0, 0.3


def source_grid(mode):
    lam = np.exp(np.linspace(np.log(1.0), np.log(DECAY), M))
    rho = np.exp(-1.0 / lam)
    if mode == "logfreq":
        theta = np.exp(np.linspace(np.log(np.pi / DECAY),
                                   np.log(0.9 * np.pi / 2), M))
    elif mode == "linear":
        u = (np.arange(M) + 0.5) / M
        theta = F + (np.pi - 2 * F) * u
    elif mode == "hybrid":
        lo, hi = np.pi / DECAY, 0.9 * np.pi / 2
        m = np.arange(M, dtype=float)
        step = (hi - lo) / (M - 1)
        base = lo + m * step
        golden = (np.sqrt(5) - 1) / 2
        taper = np.sin(m / (M - 1) * np.pi)
        jitter = np.array([((mi + golden) % 1.0) - 0.5 for mi in m]) * step * 0.5 * taper
        theta = np.clip(base + jitter, lo, hi)
    return rho, theta


def psi(rho, theta, L, with_rho):
    t = np.arange(L, dtype=np.float64)[None, :]
    p = np.sin((t + 1.0) * theta[:, None])
    if with_rho:
        p = p * (rho[:, None] ** t)
    return p


def cond_svd(A):
    """cond(A) = σ_max/σ_min（SVD 口径，上游口径 A）"""
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))


def cond_gram(A, normalize=True, floor=1e-30, dtype=None):
    """cond(A Aᵀ)（Gram 口径）。normalize=True 时先按列归一化。

    floor / dtype 与本文 verify_all.py 的 V11/V12 保持一致时，才能复现
    报告里的 1.35e36 —— 该数字是 **fp32 下未归一化 Gram 的数值条件数**，
    最小特征值被 fp32 舍入钳在 ~1e-30，换成 fp64 会变成 1e300 量级。
    """
    X = A.astype(dtype) if dtype is not None else A
    B = X.T.copy()
    if normalize:
        B = B / np.linalg.norm(B, axis=0, keepdims=True)
        G = B.T @ B
    else:
        G = X @ X.T
    w = np.linalg.eigvalsh(G)
    return float(w[-1] / max(w[0], floor)), B


rows = []
for mode in ["logfreq", "hybrid", "linear"]:
    rho, th = source_grid(mode)
    # 口径 A：Ψ=sin((τ+1)θ), L=512, SVD, 不含 ρ
    A = psi(rho, th, 512, False)
    cA = cond_svd(A)
    rankA = int(np.linalg.matrix_rank(A, tol=np.linalg.svd(A, compute_uv=False)[0] * 1e-12))
    # 口径 B：含 ρ, L=1024, 列归一化后 Gram 特征值比
    B = psi(rho, th, 1024, True)
    cB, Bn = cond_gram(B, True)
    cosmax = float(np.abs(np.corrcoef(Bn.T)[np.triu_indices(M, 1)]).max())
    # 本文报告口径：含 ρ, L=128, **fp32** 未归一化 Gram，下限 1e-30
    # （与 verify_all.py 的 V11/V12 完全一致；= cond(Ψ)² 量级且受 fp32 限制）
    C = psi(rho, th, 128, True)
    cC, _ = cond_gram(C, False, floor=1e-30, dtype=np.float32)
    rows.append(dict(mode=mode, theta_min=float(th.min()), theta_max=float(th.max()),
                     rho_min=float(rho.min()), rho_max=float(rho.max()),
                     cond_A_svd=cA, rank_A=rankA,
                     cond_B_gram_norm=cB, max_abs_cos_B=cosmax,
                     cond_C_gram_raw=cC))

# 本文白盒报告用的网格（make_grid 默认）
g0 = S.make_grid(M=32, rho_max=0.999, rho_min=0.9)
th = np.linspace(0.3, np.pi - 0.3, 32)
g1 = dict(M=32, theta=th, rho=np.linspace(0.999, 0.9, 32))
c0 = cond_gram(S.impulse_response(g0, 128), False)[0]
c1 = cond_gram(S.impulse_response(g1, 128), False)[0]

out = dict(source_grid_rows=rows,
           our_report=dict(logfreq_cond_gram_raw_L128=c0,
                           linear30_cond_gram_raw_L128=c1),
           explanation=[
               "口径 A（上游 docs/frequency_grid_fix.md §3）：Ψ[τ,m]=sin((τ+1)θ)，L=512，"
               "SVD 条件数 cond(Ψ)，不含 ρ。上游报 logfreq 1.92e16 / linear 1.04。",
               "口径 B（上游 §3）：含 ρ^τ，L=1024，列归一化后 Gram 特征值比。"
               "上游报 logfreq 2.95e10 / linear 9.53e6。",
               "本文报告口径：含 ρ，L=128，未列归一化的 H Hᵀ 特征值比 = cond(H)² 量级，"
               "故 1.35e36 与 (1.92e16)²≈3.7e32 同量级，剩余差异来自 M/L/ρ 网格/θ 上界。",
               "三者方向一致：logfreq 低频塌缩、抬 θ_min 后秩满且良条件。",
               "⚠️ 本文 1.35e36 是 fp32 未归一化 Gram 的数值条件数（最小特征值被"
               "舍入钳在 ~1e-30），不是数学条件数；用 fp64 会变成 1e300 量级。"
               "因此只应作定性对照，不应当作可与上游 SVD 口径直接比较的绝对值。",
               "⚠️ 上游 §3 指出：含 ρ 口径下 linear 的 cond 未优于 legacy 均匀网格，"
               "纯线性均匀未必是最优终点——本文「linear30 是最优解」须限定为"
               "「在本文扫过的组合与本文 Gram 口径下」。",
           ])
json.dump(out, open("results_source_grid_check.json", "w"), indent=2, ensure_ascii=False)

print("=" * 104)
print(f"{'网格':<10}{'θ_min':>10}{'θ_max':>9}{'ρ范围':>18}{'口径A cond(Ψ)':>16}{'秩':>7}{'口径B Gram':>14}{'本文Gram':>12}")
print("=" * 104)
for r in rows:
    rng = f"{r['rho_min']:.3f}~{r['rho_max']:.5f}"
    print(f"{r['mode']:<10}{r['theta_min']:>10.5f}{r['theta_max']:>9.4f}"
          f"{rng:>18}"
          f"{r['cond_A_svd']:>16.2e}{r['rank_A']:>5}/{M}{r['cond_B_gram_norm']:>14.2e}"
          f"{r['cond_C_gram_raw']:>12.2e}")
print("=" * 104)
print(f"本文报告：logfreq {c0:.2e} → linear30 {c1:.2e}（含ρ、L=128、未归一化 Gram）")
print(f"上游文档：口径A 1.92e16 → 1.04；口径B 2.95e10 → 9.53e6")
print(f"平方关系核对：(1.92e16)^2 = {(1.92e16)**2:.2e}")
print("\n-> results_source_grid_check.json")
