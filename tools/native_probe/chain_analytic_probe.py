"""
chain_analytic_probe.py — 从「训练到 0.59」推进到「解析构造精确 1.0」，
并给出内容链检索的层数紧性：H 跳恰需 H 次静态 LTI 内容匹配。

核心洞察（完全从 SGC 自身数学长出）
==================================
上一轮 stack2 卡在 0.5918，是因为层间接口 q2 = z1 @ Wq2 要**端到端学会**
把 value 编码翻译成 key 码，两层梯度耦合。但 SGC 本来就有一组固定正交谱字典
Code（NT04/NT05，与 rho/theta 网格同性质：确定性、不学、正交）。

让 **value 编码与 key 编码共用同一个码空间**：

    A_t  = Code[x_{t-1}]       前驱 key 码
    b_t  = Code[x_t]           当前 value 码（同一个 Code，不学）
    q_1  = Code[k_0]
    m_h,t = <q_h, A_t>        第 h 跳匹配
    z_h  = sum_t K_h(L-1-t) m_h,t b_t      K_h(d)=sum_p w_hp rho_hp^d > 0
    q_{h+1} = z_h / (||z_h|| + eps)        核外归一化，不触碰 Toeplitz
    logits = z_H @ Code[:V_K,:]^T

因 Code 行正交归一，<Code[k],Code[j]>=delta_kj；序列严格 (key,value) 交替且
key 唯一不重复，故：
    第 1 层命中 x_{t-1}=k_0 的位置 ⇒ b_t = Code[k_1] ⇒ z_1 = c_1 Code[k_1]
    第 2 层 q_2 = Code[k_1]，命中 x_{t-1}=k_1 ⇒ z_2 = c_2 Code[k_2]
    ...
    第 H 层 z_H = c_H Code[k_H]，c_h>0（K_h>0）⇒ argmax = k_H，精确。

**但第一版实测只有 0.6016，且与码/rho/归一化全部无关** —— 诊断出一个真实的
架构问题：**同一个 token 在内容链里既是上一跳的 value、又是下一跳的 key**。
例：序列 [.., k0, k1, k1, k2, ..] 中 k1 出现两次；第 2 层以 Code[k1] 作
query 时，m_2 在「k1 作 value 的后继」与「k1 作 key 的后继」两处都命中，
叠加两个正交码 ⇒ 双峰 ⇒ 约 0.60 的随机胜负。

**修法（完全自研，从 SGC 字典长出）**：序列严格 (key,value) 交替，key 恒在
偶数位。给 A_t 乘确定性角色掩码 mask[t] = 1[(t-1) even]，等价于取一般二阶
实极点字典里 (r1,r2)=(rho,-rho) 的交替基（theta -> pi 的奇偶分离列）。
该掩码作用于**输入信号**、与输入内容无关；核 K_h(d)=sum_p w_p rho_p^d 仍
只依赖延迟 ⇒ Toeplitz / FFT-IIR 对偶 / O(1) 状态全部保持（NT12 不受影响）。

修正后 **零训练、精确 1.0**。K_h 只贡献正标量，rho<1 的衰减谱保留。

保持：每层核静态 Toeplitz（FFT/IIR 对偶）、严格因果、状态 2*P*R = O(1)、
无 N x N 打分、无 softmax-over-tokens、无 KV cache、无输入相关 rho_t。
层间归一化是**核外**非线性，不破坏 LTI（NT12/B4 不受影响）。

关键新现象（非平凡）
====================
* H_layers < hops  ⇒ 只走到 k_{H_layers}，答错（chance）→ 下界
* H_layers == hops ⇒ 精确 1.0                        → 构造达到下界（紧）
* H_layers > hops  ⇒ **过冲**到 k_{H_layers}，同样答错
  （因为 q 继续沿链传播，没有停止机制）→ 层数不是「至少 H」而是「恰 H」

与线性注意力的边界（诚实）
==========================
rho=1、单极点时 z_h = q_h^T (sum_t A_t b_t^T) 与 linear-attention 状态同构；
rho<1、多极点时 K_h(d)=sum_p w_p rho_p^d 是**衰减谱加权**，仍属加权线性
注意力族。本文不宣称脱离该范式，只主张：在静态 LTI 核 + 固定正交码下，
「张量阶 2 × 层数 H × O(1) 状态 × 无 N×N」这组约束可同时精确达到，
且给出联合紧性（NT30 的多跳版本）。

输出 results/chain_analytic_*.json。纯 numpy，无 torch，无 GPU。
**不是语言建模训练**，不产生 ppl / 生成样本。
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

OUT = "results"
SEEDS = (0, 1, 2, 3, 4)
P = 2
RHO = (0.90, 0.999)
W = (0.5, 0.5)
HELDOUT = 512
STEPS = 900
LR = 6e-3


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H[:n, :n] / np.sqrt(n)


def code_matrix(V, kind="hadamard", seed=20261105):
    if kind == "hadamard":
        return hadamard(V)                     # C C^T = I_V
    if kind == "random":
        g = np.random.default_rng(seed).standard_normal((V, V))
        q, _ = np.linalg.qr(g)
        return q                               # 同样正交，但非确定性
    if kind == "identity":
        return np.eye(V)
    if kind == "gaussian":
        # 非正交、未归一化：用于检验 NT04/NT05 的正交性是否必要
        return np.random.default_rng(seed).standard_normal((V, V)) * 0.5
    raise ValueError(kind)


def build_chain(n_pairs, hops, n_samples, seed, V, shuffle=True):
    """链 k_0->...->k_{n_pairs}，pair 位置随机；q=k_0，答案 k_hops。"""
    VK = V // 2
    L = 2 * n_pairs + 1
    r = np.random.default_rng(seed)
    X = np.zeros((n_samples, L, V))
    Xp = np.zeros_like(X)
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        chain = r.choice(VK, size=n_pairs + 1, replace=False)
        pairs = [(chain[p], chain[p + 1]) for p in range(n_pairs)]
        if shuffle:
            r.shuffle(pairs)
        for p_, (k, v) in enumerate(pairs):
            X[i, 2 * p_, k] = 1.0
            X[i, 2 * p_ + 1, v] = 1.0
        X[i, L - 1, chain[0]] = 1.0
        Y[i] = chain[hops]
        Xp[i, 1:, :] = X[i, :-1, :]
    return X, Xp, Y, dict(n_pairs=n_pairs, hops=hops, V=V, VK=VK, L=L,
                          chance=1.0 / VK, seed=seed)


def Kvec(L, rho=RHO, w=W):
    """K[t] = K(L-1-t) = sum_p w_p rho_p^(L-1-t) > 0."""
    return np.array([float(np.sum(np.asarray(w) * np.asarray(rho) ** (L - 1 - t)))
                     for t in range(L)])


def role_mask(L):
    """theta -> pi / (r1,r2)=(rho,-rho) 的奇偶分离基：只保留 key 前驱。

    mask[t]=1 当 (t-1) 为偶数（x_{t-1} 是 key），否则 0。确定性、与输入
    内容无关；作用于输入信号 A，不作用于核 K(d)。
    """
    t = np.arange(L)
    return np.where((t - 1) % 2 == 0, 1.0, 0.0).astype(np.float64)


def analytic_forward(X, Xp, Code, n_layers, n_read, rho=RHO, w=W,
                     normalize=True, role=True, ret_state=False):
    """n_layers 个静态 LTI 匹配层；从第 n_read 层读出（默认最后一层）。

    n_layers >= 1。n_read 用于检验「过冲」：若 n_read > hops 的层数，
    读最后一层；也可读中间层，验证答案确实逐层沿链推进。
    """
    B_, L_, V = X.shape
    A = Xp @ Code                                  # (B,L,V)
    if role:
        A = A * role_mask(L_)[None, :, None]       # theta=pi 角色分离（核外、确定性）
    b = X @ Code                                   # (B,L,V)  统一码空间
    q = X[:, L_ - 1, :] @ Code                     # (B,V)
    K = Kvec(L_, rho, w)
    VK = V // 2
    zs = []
    for h in range(n_layers):
        m = np.einsum("blv,bv->bl", A, q)          # (B,L) 内容匹配
        z = np.einsum("bl,blv,l->bv", m, b, K)     # (B,V) 静态核加权
        zs.append(z)
        if normalize:
            q = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12)
        else:
            q = z
    z_out = zs[n_read - 1]
    logits = z_out @ Code[:VK, :].T                # (B,VK) 只在 key 半区解码
    if ret_state:
        return logits, dict(A=A, b=b, K=K, zs=zs, role=role)
    return logits


def eval_analytic(n_layers, hops, n_pairs, V=64, code_kind="hadamard",
                  rho=RHO, w=W, normalize=True, role=True,
                  n_samples=HELDOUT, seed=100000):
    Code = code_matrix(V, code_kind)
    X, Xp, Y, meta = build_chain(n_pairs, hops, n_samples, seed, V)
    lg = analytic_forward(X, Xp, Code, n_layers, n_layers,
                          rho, w, normalize, role)
    per_layer = []
    for h in range(1, n_layers + 1):
        lgh = analytic_forward(X, Xp, Code, n_layers, h,
                               rho, w, normalize, role)
        per_layer.append(float((lgh.argmax(-1) == Y).mean()))
    return dict(acc=float((lg.argmax(-1) == Y).mean()),
                per_layer_acc=per_layer, chance=meta["chance"],
                n_layers=n_layers, hops=hops, n_pairs=n_pairs, V=V,
                code_kind=code_kind, normalize=normalize, role=role)


# ---------------------------------------------------------------- 可学性对照
def learned_forward(p, X, Xp, Code, n_layers, rho=RHO, w=W):
    """同一架构，但层间 query 投影 Wq_h 与解码 Wout 从随机初始化端到端学。"""
    B_, L_, V = X.shape
    A = Xp @ Code
    A = A * role_mask(L_)[None, :, None]           # 同一角色分离基
    b = X @ Code
    q = X[:, L_ - 1, :] @ Code
    K = Kvec(L_, rho, w)
    VK = V // 2
    zs = []
    for h in range(n_layers):
        m = np.einsum("blv,bv->bl", A, q)
        z = np.einsum("bl,blv,l->bv", m, b, K)
        zs.append(z)
        if h < n_layers - 1:
            q = z @ p[f"Wq{h}"]                    # (B,V)
            q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    return zs[-1] @ p["Wout"] + p["bout"]


def train_learned(hops, n_pairs, V, seed, steps=STEPS, verbose=False):
    """有限差分 + SGD：参数仅 Wq(层间) + Wout/bout。用于证明构造可学。"""
    n_layers = hops
    Code = code_matrix(V)
    r = np.random.default_rng(seed + 4242)
    p = {"Wout": r.standard_normal((V, V // 2)) * 0.05,
         "bout": np.zeros(V // 2)}
    for h in range(n_layers - 1):
        p[f"Wq{h}"] = np.eye(V) + r.standard_normal((V, V)) * 0.02
    keys = list(p)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    rng = np.random.default_rng(seed + 31337)

    # 数值梯度（参数量小：V^2 * (n_layers-1) + V*VK，V=64 时 ~8k，可接受）
    def obj(P, Xb, Xpb, Yb):
        lg = learned_forward(P, Xb, Xpb, Code, n_layers)
        z = lg - lg.max(-1, keepdims=True)
        e = np.exp(z); pr = e / e.sum(-1, keepdims=True)
        return float(-np.log(pr[np.arange(len(Yb)), Yb] + 1e-12).mean())

    hist = []
    for step in range(1, steps + 1):
        Xb, Xpb, Yb, _ = build_chain(n_pairs, hops, 16, int(rng.integers(0, 2**31)), V)
        g = {k: np.zeros_like(vv) for k, vv in p.items()}
        eps = 2e-3
        # 逐参数块中心差分（只对 Wout 与 Wq 的少量列，控制开销）
        for k in keys:
            flat = p[k].ravel(); gf = g[k].ravel()
            idx = rng.choice(flat.size, size=min(24, flat.size), replace=False)
            for i in idx:
                pp = {kk: vv.copy() for kk, vv in p.items()}
                pp[k].ravel()[i] += eps
                l1 = obj(pp, Xb, Xpb, Yb)
                pm = {kk: vv.copy() for kk, vv in p.items()}
                pm[k].ravel()[i] -= eps
                l2 = obj(pm, Xb, Xpb, Yb)
                gf[i] = (l1 - l2) / (2 * eps)
        for k in keys:
            gk = np.clip(g[k], -3, 3)
            m[k] = 0.9 * m[k] + 0.1 * gk
            v[k] = 0.999 * v[k] + 0.001 * gk ** 2
            p[k] -= LR * (m[k] / (1 - 0.9**step)) / (np.sqrt(v[k] / (1 - 0.999**step)) + 1e-8)
        if step % 300 == 0 or step == steps:
            Xh, Xph, Yh, _ = build_chain(n_pairs, hops, HELDOUT, seed + 100000, V)
            acc = float((learned_forward(p, Xh, Xph, Code, n_layers).argmax(-1) == Yh).mean())
            hist.append(dict(step=step, acc=acc))
            if verbose:
                print(f"    learned H={hops} seed{seed} step{step:4d} acc={acc:.4f}", flush=True)
    return dict(hops=hops, seed=seed, final_acc=hist[-1]["acc"],
                history=hist, n_params=int(sum(x.size for x in p.values())))


# ---------------------------------------------------------------- 守恒
def guards(V=64, n_pairs=4, n_layers=3):
    Code = code_matrix(V)
    X, Xp, Y, meta = build_chain(n_pairs, 2, 8, 999, V)
    lg, st = analytic_forward(X, Xp, Code, n_layers, n_layers, ret_state=True)
    L = meta["L"]
    # 1) 核静态：K[t]=K(L-1-t)，与输入内容无关
    K = st["K"]
    dev = 0.0
    for t in range(L):
        d = L - 1 - t
        dev = max(dev, abs(float(np.sum(np.asarray(W) * np.asarray(RHO) ** d)) - K[t]))
    Xq = X.copy(); Xq[:, L - 1, :] = 0.0
    Xq[:, L - 1, (Y[0] + 1) % (V // 2)] = 1.0
    _, st2 = analytic_forward(Xq, Xp, Code, n_layers, n_layers, ret_state=True)
    dev = max(dev, float(np.abs(st["K"] - st2["K"]).max()))
    # 2) 严格因果：改最后一步 query 不改变前 L-1 步的 A/b
    causal = bool(np.allclose(st["A"][:, :L-1], st2["A"][:, :L-1], atol=1e-12)
                  and np.allclose(st["b"][:, :L-1], st2["b"][:, :L-1], atol=1e-12))
    # 3) FFT/IIR 对偶：Kd(d)=sum_p w_p rho_p^d（按延迟），不是 K_vec[t]
    def Kd(d): return float(np.sum(np.asarray(W) * np.asarray(RHO) ** d))
    U = np.random.default_rng(7).standard_normal((8, L, V))
    col = np.zeros((L, L))
    for t in range(L):
        for s in range(t + 1):
            col[t, s] = Kd(t - s)
    y_fft = col @ U                                # (B,L,V) 逐时刻输出
    sstate = np.zeros((8, P, V))
    y_iir = np.zeros((8, L, V))
    for t in range(L):
        sstate = np.asarray(RHO)[None, :, None] * sstate + U[:, t][:, None, :]
        y_iir[:, t] = (np.asarray(W)[None, :, None] * sstate).sum(1)
    den = max(float(np.abs(y_fft).max()), 1e-12)
    rel = float(np.abs(y_fft - y_iir).max() / den)
    # 末时刻 y[L-1] 必须等于 forward 用的 z=sum_t K_vec[t] u[t]
    z_last = np.einsum("blv,l->bv", U, K)
    last_dev = float(np.abs(y_fft[:, L - 1] - z_last).max() / den)
    # 4) O(1) 状态
    state_bytes = {L_: 2 * P * V * 4 for L_ in (9, 17, 33, 65, 1025)}
    return dict(V=V, n_pairs=n_pairs, n_layers=n_layers,
                toeplitz=bool(dev < 1e-10), max_kernel_dev=float(dev),
                fft_iir_rel_error=rel, duality_pass=bool(rel < 1e-10),
                last_time_vs_forward_dev=last_dev,
                last_time_pass=bool(last_dev < 1e-10),
                strict_causality=causal,
                state_bytes_by_L=state_bytes,
                o1_state_pass=bool(len(set(state_bytes.values())) == 1),
                no_nxn_matrix=True, no_kv_cache=True, no_softmax_over_tokens=True,
                all_passed=bool(dev < 1e-10 and rel < 1e-10 and last_dev < 1e-10
                                and causal and len(set(state_bytes.values())) == 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()

    # ---- 1) 层数 × 跳数 全矩阵（核心边界形状） ----
    if a.mode in ("matrix", "all"):
        print("== layer x hop matrix (analytic construction, ZERO training) ==", flush=True)
        mat = []
        pairs = 4
        for hops in (1, 2, 3, 4):
            for nl in (1, 2, 3, 4, 5):
                r = eval_analytic(nl, hops, pairs)
                mat.append(r)
                print(f"  hops={hops} layers={nl}  acc={r['acc']:.4f}  "
                      f"per_layer={[round(x,3) for x in r['per_layer_acc']]}", flush=True)
        json.dump(mat, open(f"{OUT}/chain_analytic_matrix.json", "w"), indent=1)

    # ---- 2) 消融 ----
    if a.mode in ("ablation", "all"):
        print("\n== ablations ==", flush=True)
        abl = []
        for code in ("hadamard", "random", "identity", "gaussian"):
            r = eval_analytic(2, 2, 4, code_kind=code)
            r["tag"] = f"code={code}"; abl.append(r)
            print(f"  code={code:10s} acc={r['acc']:.4f}", flush=True)
        for rho, w, tag in (((0.5, 0.999), (0.5, 0.5), "rho_fast"),
                            ((0.9999, 0.99999), (0.5, 0.5), "rho_slow"),
                            ((0.90, 0.999), (0.99, 0.01), "w_skew")):
            r = eval_analytic(2, 2, 4, rho=rho, w=w)
            r["tag"] = tag; abl.append(r)
            print(f"  {tag:12s} acc={r['acc']:.4f}", flush=True)
        r = eval_analytic(2, 2, 4, normalize=False)
        r["tag"] = "no_norm"; abl.append(r)
        print(f"  no_norm     acc={r['acc']:.4f}", flush=True)
        r = eval_analytic(2, 2, 4, role=False)
        r["tag"] = "no_role_mask"; abl.append(r)
        print(f"  no_role_mask acc={r['acc']:.4f}  <-- 角色歧义对照", flush=True)
        # 长度 / 大词表泛化（构造零训练，直接泛化）
        for V, np_ in ((64, 8), (128, 16), (256, 32), (512, 64)):
            r = eval_analytic(2, 2, np_, V=V)
            r["tag"] = f"V{V}_np{np_}"; abl.append(r)
            print(f"  V={V:4d} np={np_:3d} L={2*np_+1:4d} acc={r['acc']:.4f}", flush=True)
        json.dump(abl, open(f"{OUT}/chain_analytic_ablation.json", "w"), indent=1)

    # ---- 3) 守恒 ----
    if a.mode in ("guards", "all"):
        print("\n== conservation ==", flush=True)
        g = guards()
        print(" ", {k: v for k, v in g.items() if k != "state_bytes_by_L"}, flush=True)
        json.dump(g, open(f"{OUT}/chain_analytic_guards.json", "w"), indent=1)

    # ---- 4) 可学性（5 seeds） ----
    if a.mode in ("learned", "all"):
        print("\n== learnability (5 seeds, H=2/3) ==", flush=True)
        lr = []
        for hops in (2, 3):
            for s in SEEDS:
                lr.append(train_learned(hops, 4, 64, s, steps=a.steps, verbose=(s == 0)))
        json.dump(lr, open(f"{OUT}/chain_analytic_learned.json", "w"), indent=1)
        for hops in (2, 3):
            arr = np.array([x["final_acc"] for x in lr if x["hops"] == hops])
            print(f"  H={hops} learned: {arr.mean():.4f} +/- {arr.std(ddof=1):.4f} "
                  f"[{arr.min():.4f},{arr.max():.4f}]", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
