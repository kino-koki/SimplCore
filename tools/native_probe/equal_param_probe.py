"""
equal_param_probe.py — 同参数追平标尺的判别性实验。

数学线（完全自研，从 SGC 的 ρ^τ 一路推广）
==========================================
SGC 核 h[τ]=ρ^τ sin((τ+1)θ)  ⟺  一对复共轭极点
  → 一般二阶（两个实极点，上一轮 G2）
  → **P 个一阶极点的部分分式分解**：

      h_n[τ] = Σ_{k=1}^{P} w_{n,k} · r_{n,k}^τ,   r∈(ρ_min,1)

  任意稳定 LTI 系统（分母可分解）都能写成这个形式。P≥L 时
  Vandermonde 可逆 ⇒ 字典能表示**任意**长度-L 的 LTI 核。
  这是「静态核族」的完整上界，没有比它更一般的静态核了。

递推实现（P 拍状态，O(1) 与 L 无关，严格因果）
      s_{k,t} = r_k·s_{k,t-1} + u_t
      y_t     = Σ_k w_k·s_{k,t}
  u_t 是**外生**输入（无状态反馈）⇒ 整段 Toeplitz / 单次 FFT 路径完全可行；
  这里用 scan 只是为了显式拿到 s_{k,L} 供读出端使用。

读出端（核外，不破坏 LTI）
      β_{b,k} = 1 + tanh(q_b · Wsel)          query 条件化的极点权重
      eff_{b,n,k} = w_{n,k} · β_{b,k}
      y_b = Σ_k eff ⊙ s_{k,L}
      z   = (q·Wq) ⊙ y
      logits = z·Wout   或   z·Eᵀ（weight tying，与输入嵌入共享）

  这是 B1 缝隙里的「非线性 read-out」：B1 只排除线性 read-out。
  与历史 S8_softgate 的区别：这里 β 是**线性组合权重**（初始≈1，退化为 w），
  不是 softmax 锐化竞争，梯度不炸。

严格保持
========
  · 核 LTI / Toeplitz / FFT-IIR 对偶 / O(1) 状态 / 严格因果
  · 无 N×N、无 KV cache、无 attention、无 hybrid、无对照模型实现
  · 活跃代码零混合折中网格、零对照实现

参数预算
========
  目标 10,240（历史标尺的参数量），允许 ±3%。每个臂在脚本启动时打印精确参数数。

用法
====
  python equal_param_probe.py --mode budget
  python equal_param_probe.py --mode guards
  python equal_param_probe.py --mode gradcheck
  python equal_param_probe.py --mode all --steps 3000
"""
import argparse
import json
import os
import time
import numpy as np

from mini_autograd import T, const, zeros, nll_from_logits
import task_utils as U
import sgc_fixed as S

OUT = "results"
SEEDS = (0, 1, 2)
STEPS = 3000
RHO_MIN = 0.05
BUDGET = 10240
TOL = 0.03

# ---------------------------------------------------------------
# 配置：每个臂 (D, N, P, tied, gate, qcpr, mlp, layers)
#   tied   : 输出投影 = Eᵀ（要求 N==D），省 N*VOCAB 参数
#   gate   : 核外写入门 sigmoid(x·Wg)
#   qcpr   : query 条件化极点读出权重
#   mlp    : 读出端隐层宽度（0 表示线性）
#   layers : LTI 层串联数；>1 时中间有逐位置非线性 ⇒ 端到端非 LTI（单独标注）
# ---------------------------------------------------------------
ARMS = {
    # 第一轮：P 扫描（其余固定），预算对齐
    "E0_P2":      dict(D=32, N=50, P=2,  tied=False, gate=False, qcpr=False, mlp=0, layers=1),
    "E1_P4":      dict(D=32, N=48, P=4,  tied=False, gate=False, qcpr=False, mlp=0, layers=1),
    "E2_P8":      dict(D=32, N=46, P=8,  tied=False, gate=False, qcpr=False, mlp=0, layers=1),
    "E3_P16":     dict(D=32, N=42, P=16, tied=False, gate=False, qcpr=False, mlp=0, layers=1),
    "E4_P32":     dict(D=32, N=36, P=32, tied=False, gate=False, qcpr=False, mlp=0, layers=1),
    "E5_P64":     dict(D=32, N=28, P=64, tied=False, gate=False, qcpr=False, mlp=0, layers=1),
    # 第二轮：weight tying + query 条件化极点读出
    "E6_tied_P64":   dict(D=32, N=32, P=64, tied=True,  gate=True,  qcpr=False, mlp=0, layers=1),
    "E7_qcpr_P16":   dict(D=32, N=40, P=16, tied=False, gate=False, qcpr=True,  mlp=0, layers=1),
    "E8_qcpr_P32":   dict(D=32, N=32, P=32, tied=False, gate=False, qcpr=True,  mlp=0, layers=1),
    "E9_qcpr_tied":  dict(D=32, N=32, P=52, tied=True,  gate=False, qcpr=True,  mlp=0, layers=1),
    "E10_qcpr_gate": dict(D=32, N=32, P=44, tied=True,  gate=True,  qcpr=True,  mlp=0, layers=1),
    # 第三轮：读出端容量 / 高层数 / 高 P
    "E11_mlp":       dict(D=32, N=32, P=24, tied=True,  gate=False, qcpr=True,  mlp=32, layers=1),
    "E12_twolayer":  dict(D=24, N=24, P=58, tied=True,  gate=False, qcpr=True,  mlp=0, layers=2),
    "E13_highP":     dict(D=24, N=24, P=98, tied=True,  gate=False, qcpr=True,  mlp=0, layers=1),
    # 第四轮：参数重分配 —— 低 D、高 N（上一轮已证 N 是主杠杆）
    "F0_D16_N72_P8":   dict(D=16, N=72,  P=8,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F1_D16_N64_P16":  dict(D=16, N=64,  P=16, tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F2_D8_N93_P8":    dict(D=8,  N=93,  P=8,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F3_D8_N101_P4":   dict(D=8,  N=101, P=4,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F4_D4_N118_P4":   dict(D=4,  N=118, P=4,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F5_D12_N81_P8":   dict(D=12, N=81,  P=8,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F6_D16_N71_qcpr": dict(D=16, N=71,  P=8,  tied=False, gate=False, qcpr=True,  mlp=0, layers=1, init="spread"),
    "F7_D8_N105_P2":   dict(D=8,  N=105, P=2,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F8_D16_N79_P2":   dict(D=16, N=79,  P=2,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F9_D8_N81_P16":   dict(D=8,  N=81,  P=16, tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    "F10_D4_N108_P8":  dict(D=4,  N=108, P=8,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
    # 复现对照：与上一轮 G2 同配置（不卡预算，验证新脚本一致性）
    "R0_g2_repro":     dict(D=32, N=64,  P=2,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="bipole"),
    # 初始化消融：同配置只换 init
    "A0_init_legacy":  dict(D=32, N=50,  P=2,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="legacy"),
    "A1_init_spread":  dict(D=32, N=50,  P=2,  tied=False, gate=False, qcpr=False, mlp=0, layers=1, init="spread"),
}

def _count(cfg):
    D, N, P = cfg["D"], cfg["N"], cfg["P"]
    c = U.VOCAB * D                       # E
    c += 2 * D * N                       # Wa, Wb
    c += D * N                           # Wq
    if not cfg["tied"]:
        c += N * U.VOCAB                 # Wout
    c += 2 * N * P                       # r(p) + w
    if cfg["gate"]:
        c += D * N
    if cfg["qcpr"]:
        c += D * P
    if cfg["mlp"]:
        c += N * cfg["mlp"] + cfg["mlp"] * U.VOCAB
    if cfg["layers"] > 1:
        c += (cfg["layers"] - 1) * (2 * N * P)          # 额外层的 r,w
    return int(c)


# ---------------- 参数 ----------------
def _sig(z): return 1.0 / (1.0 + np.exp(-z))
def _logit(u):
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


def make_params(cfg, seed):
    r = np.random.default_rng(seed + 31337)
    def W(a, b, s=0.15): return T(r.standard_normal((a, b)) * s)
    D, N, P = cfg["D"], cfg["N"], cfg["P"]
    Pm = {"E": T(r.standard_normal((U.VOCAB, D)) * 0.15),
          "Wa": W(D, N), "Wb": W(D, N), "Wq": W(D, N)}
    if not cfg["tied"]:
        Pm["Wout"] = W(N, U.VOCAB)
    if cfg["gate"]:
        Pm["Wg"] = W(D, N, s=0.02)
    if cfg["qcpr"]:
        Pm["Wsel"] = W(D, P, s=0.02)
    if cfg["mlp"]:
        Pm["W1"] = W(N, cfg["mlp"], s=0.08)
        Pm["W2"] = W(cfg["mlp"], U.VOCAB, s=0.08)
    init = cfg.get("init", "spread")
    rho0 = np.exp(np.linspace(np.log(0.30), np.log(0.995), P))   # P 个衰减基
    for li in range(cfg["layers"]):
        tag = "" if li == 0 else f"_{li}"
        if init == "bipole":
            # 复现 general_probe G2：逐通道 (r1,r2)，w 由部分分式给出
            M = 16; Cc = max(1, N // M)
            grid = np.exp(np.linspace(np.log(np.pi / 8192.0),
                                      np.log(0.9 * np.pi / 2), M))
            rr = np.repeat(grid[:, None], Cc, axis=1).reshape(-1)[:N]
            rr = np.clip(rr, 0.05 + 1e-4, 1 - 1e-4)
            r1 = np.clip(rr * 0.98, 0.05 + 1e-4, 1 - 1e-4)
            r2 = np.clip(rr * 0.55, 0.05 + 1e-4, 1 - 1e-4)
            rp = np.stack([r1, r2], axis=1)
            den = np.where(np.abs(r1 - r2) > 1e-3, r1 - r2, 1e-3)
            wp = np.stack([r1 / den, -r2 / den], axis=1)
        elif init == "legacy":
            rp = np.repeat(rho0[None, :], N, axis=0)
            wp = np.ones((N, P)) / P + r.normal(0, 0.01, (N, P))
        else:   # spread：P 个基跨通道共享初值，w 逐通道随机 ⇒ 每个通道一套可学核
            rp = np.repeat(rho0[None, :], N, axis=0)
            wp = np.ones((N, P)) / P + r.normal(0, 0.3 / np.sqrt(P), (N, P))
        p0 = _logit((np.clip(rp, RHO_MIN + 1e-4, 1 - 1e-4) - RHO_MIN) / (1 - RHO_MIN))
        if init == "spread":
            p0 = p0 + r.normal(0, 0.08, (N, P))       # 打破通道对称
        Pm["p" + tag] = T(p0.reshape(-1))
        Pm["w" + tag] = T(wp.reshape(-1))
    return Pm


def n_params(P):
    return int(sum(v.d.size for v in P.values()))


def _rho(p, N, P):
    return (RHO_MIN + (1 - RHO_MIN) * _sig(p)).reshape(N, P)


# ---------------- 前向 ----------------
def _scan(u, pp, ww, N, P):
    """P 拍一阶极点递推。u:(B,L,N) T；返回 s:(B,N,P) T（最终状态）。"""
    B_, L, _ = u.d.shape
    rr = T((RHO_MIN + (1 - RHO_MIN) * _sig(pp.d)).reshape(N, P), [pp])
    def bwr():
        sp = _sig(pp.d)                       # (N*P,)
        pp.g += rr.g.reshape(-1) * (1 - RHO_MIN) * sp * (1 - sp)
    rr._b = bwr
    s = zeros((B_, N, P))
    for t in range(L):
        s = rr[None, :, :] * s + u[:, t, :].reshape(B_, N, 1)
    return s, rr


def forward(arm, P, X, Xp, L):
    cfg = ARMS[arm]
    B_, _, _ = X.shape
    D, N, Pp = cfg["D"], cfg["N"], cfg["P"]
    xx = const(X) @ P["E"]
    xp = const(Xp) @ P["E"]
    u = (xx @ P["Wa"]) * (xp @ P["Wb"])              # product 绑定编码
    if cfg["gate"]:
        u = u * (xx @ P["Wg"]).sigmoid()

    q = xx[:, L - 1, :]                                # (B,D)
    qh = q @ P["Wq"]                                   # (B,N)

    s = None
    for li in range(cfg["layers"]):
        tag = "" if li == 0 else f"_{li}"
        s, _ = _scan(u, P["p" + tag], P["w" + tag], N, Pp)
        # w 必须留在 autograd 图里：reshape 成 T 节点再广播（不能走 numpy→const）
        wwT = P["w" + tag].reshape(N, Pp)
        eff = wwT.reshape(1, N, Pp) * const(np.ones((B_, 1, 1)))
        if cfg["qcpr"]:
            beta = (q @ P["Wsel"]).tanh() + const(1.0)     # (B,P) 初始≈1
            eff = eff * beta.reshape(B_, 1, Pp)
        y = (eff * s).sum(axis=-1)                         # (B,N)
        if li < cfg["layers"] - 1:
            # 中间逐位置非线性（核外）；此后端到端不再是 LTI，单独标注
            u = u * y.tanh().reshape(B_, 1, N)
    z = qh * y
    if cfg["mlp"]:
        logits = (z @ P["W1"]).tanh() @ P["W2"]
    elif cfg["tied"]:
        logits = z @ P["E"].T
    else:
        logits = z @ P["Wout"]
    return logits, s


# ---------------- 守恒断言 ----------------
def guards(arm, P, seed=0, verbose=False):
    cfg = ARMS[arm]
    L = U.L_TRAIN
    N, Pp = cfg["N"], cfg["P"]
    r = np.random.default_rng(seed)
    u = r.standard_normal((2, L, N))
    ww = P["w"].d.reshape(N, Pp)
    rr = _rho(P["p"].d, N, Pp)
    tau = np.arange(L, dtype=np.float64)
    K = (ww[:, :, None] * (rr[:, :, None] ** tau[None, None, :])).sum(axis=1)  # (N,L)
    Kb = K
    Tm = S.toeplitz(Kb)
    y_toep = np.einsum("nls,bsn->bln", Tm, u)

    # 1 Toeplitz
    dev = max(abs(Tm[n, t, ss] - Tm[n, t - 1, ss - 1])
              for n in range(N) for t in range(1, L) for ss in range(1, t + 1))
    # 2 FFT 整段
    yf = S.fft_apply(Kb, u[:, :, :, None])[:, :, :, 0]
    den = max(np.abs(y_toep).max(), 1e-12)
    rel_fft = float(np.abs(y_toep - yf).max() / den)
    # 3 P 拍 IIR
    s = np.zeros((2, N, Pp))
    yi = np.zeros((2, L, N))
    for t in range(L):
        s = rr[None, :, :] * s + u[:, t, :, None]
        yi[:, t, :] = (ww[None, :, :] * s).sum(-1)
    rel_iir = float(np.abs(y_toep - yi).max() / den)
    # 4 严格因果
    u2 = u.copy(); u2[:, L - 1] += 10
    past = float(np.abs(y_toep[:, :L - 1]
                        - np.einsum("nls,bsn->bln", Tm, u2)[:, :L - 1]).max())
    # 5 状态与 L 无关
    bytes_ = [N * Pp * 4 * cfg["layers"] for _ in (9, 17, 25, 65)]
    out = dict(toeplitz=bool(dev < 1e-10), fft_iir=bool(rel_fft < 1e-9),
               iir_vs_toeplitz=bool(rel_iir < 1e-9),
               strict_causality=bool(past == 0),
               state_independent_of_L=bool(len(set(bytes_)) == 1),
               max_toeplitz_dev=float(dev), rel_fft=rel_fft, rel_iir=rel_iir,
               state_bytes=bytes_[0],
               end_to_end_LTI=bool(cfg["layers"] == 1),
               note=("两层含中间逐位置非线性 ⇒ 核级 LTI 但端到端非 LTI"
                     if cfg["layers"] > 1 else
                     "u_t 外生 ⇒ 整段 Toeplitz / 单次 FFT 路径完全可行；scan 仅用于取 s_{k,L}"),
               all_passed=bool(dev < 1e-10 and rel_fft < 1e-9 and rel_iir < 1e-9
                               and past == 0 and len(set(bytes_)) == 1))
    if verbose:
        print(f"    toeplitz={out['toeplitz']} fft={rel_fft:.2e} "
              f"iir={rel_iir:.2e} causal={out['strict_causality']} "
              f"e2e_LTI={out['end_to_end_LTI']} all={out['all_passed']}")
    return out


# ---------------- 训练 ----------------
def eval_correct(arm, P, X, Xp, Y):
    return (forward(arm, P, X, Xp, X.shape[1])[0].d.argmax(-1) == Y).astype(int)


def run_train(arm, seed, steps=STEPS):
    cfg = ARMS[arm]
    P = make_params(cfg, seed)
    g0 = guards(arm, P, seed)
    Xtr, Xptr, Ytr, _ = U.build(U.N_PAIRS, 2000, seed=1000 + seed)
    Xte, Xpte, Yte, _ = U.build(U.N_PAIRS, 512, seed=9000 + seed)
    ADAM = {}
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = np.random.default_rng(seed * 100000 + step).choice(len(Ytr), U.B, replace=False)
        X, Xp, Y = Xtr[idx], Xptr[idx], Ytr[idx]
        for v in P.values():
            v.g[...] = 0.0
        lg, _ = forward(arm, P, X, Xp, U.L_TRAIN)
        nll_from_logits(lg, Y).backward()
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, v in P.items():
            key = "m_" + name
            if key not in ADAM:
                ADAM[key] = np.zeros_like(v.d); ADAM["v_" + name] = np.zeros_like(v.d)
            m, vv = ADAM[key], ADAM["v_" + name]
            m *= b1; m += (1 - b1) * v.g
            vv *= b2; vv += (1 - b2) * v.g * v.g
            v.d -= U.LR * (m / (1 - b1 ** step)) / (np.sqrt(vv / (1 - b2 ** step)) + eps)
    g1 = guards(arm, P, seed)
    corr = eval_correct(arm, P, Xte, Xpte, Yte)
    Xsh, Xpsh, Ysh, _ = U.build(U.N_PAIRS, 512, seed=9000 + seed, shuffle_pairs=True)
    sh = float(eval_correct(arm, P, Xsh, Xpsh, Ysh).mean())
    gen = {}
    for n in (4, 8, 12, 16):
        Xg, Xpg, Yg, _ = U.build(n, 256, seed=7000 + seed + n)
        gen[f"L{2*n+1}"] = float(eval_correct(arm, P, Xg, Xpg, Yg).mean())
    return dict(arm=arm, seed=seed, params=n_params(P),
                guards_before=g0, guards_after=g1,
                guards_all_passed=bool(g0["all_passed"] and g1["all_passed"]),
                final_acc=float(corr.mean()), shuffle_acc=sh,
                length_generalization=gen, correct_te=corr.tolist(),
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
    ap.add_argument("--arms", default="",
                    help="逗号分隔的臂名前缀/精确名；空=全部")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sel = None
    if a.arms:
        names = [x.strip() for x in a.arms.split(",") if x.strip()]
        sel = [k for k in ARMS if any(k == n or k.startswith(n) for n in names)]
        print("选中臂:", sel)

    if a.mode == "budget":
        print(f"目标预算 {BUDGET} ±{TOL*100:.0f}% = [{BUDGET*(1-TOL):.0f}, {BUDGET*(1+TOL):.0f}]")
        out = {}
        for arm in (sel or ARMS):
            c = _count(ARMS[arm]); out[arm] = c
            flag = "OK " if abs(c - BUDGET) / BUDGET <= TOL else "OFF"
            print(f"  [{flag}] {arm:16s} D={cfg['D']:2d} N={cfg['N']:2d} "
                  f"P={cfg['P']:3d} tied={int(cfg['tied'])} qcpr={int(cfg['qcpr'])} "
                  f"layers={cfg['layers']}  params={c:6d}  Δ={c-BUDGET:+d}")
        json.dump(out, open(f"{OUT}/budget.json", "w"), indent=1)
        return

    if a.mode == "guards":
        out = {}
        for arm in (sel or ARMS):
            out[arm] = guards(arm, make_params(ARMS[arm], 0), verbose=True)
            print(f"  {arm:16s} all={out[arm]['all_passed']} "
                  f"e2e_LTI={out[arm]['end_to_end_LTI']} bytes={out[arm]['state_bytes']}")
        json.dump(out, open(f"{OUT}/equal_guards.json", "w"), indent=1, ensure_ascii=False)
        return

    if a.mode == "gradcheck":
        out = [gradcheck(arm) for arm in (sel or ARMS)]
        for g in out:
            print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['arm']:16s} {g['worst_rel']:.2e}")
        json.dump(out, open(f"{OUT}/equal_gradcheck.json", "w"), indent=1)
        print(f"==> {sum(g['passed'] for g in out)}/{len(out)} PASS")
        return

    if a.mode == "all":
        fn = f"{OUT}/equal_results.json"
        allres = json.load(open(fn)) if os.path.exists(fn) else {}
        for arm in (sel or ARMS):
            done = {int(r["seed"]) for r in allres.get(arm, [])}
            for sd in SEEDS:
                if sd in done:
                    continue
                r = run_train(arm, sd, steps=a.steps)
                allres.setdefault(arm, []).append(r)
                print(f"  [{arm:16s} s{sd}] acc={r['final_acc']:.4f} "
                      f"shuffle={r['shuffle_acc']:.4f} params={r['params']} "
                      f"guards={r['guards_all_passed']} {r['elapsed_s']}s", flush=True)
                with open(fn, "w") as f:
                    json.dump(allres, f, indent=1, ensure_ascii=False); f.flush()
        print("\n写入", fn)
        return


if __name__ == "__main__":
    main()
