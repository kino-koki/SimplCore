"""B1 严格证明的数值注脚：静态核「可加分离」下界（纯 NumPy，秒级）。

命题 B1（见 docs/EXPRESSIVITY_BOUNDARIES.md）：
    静态核输出 y_L = sum_{tau>0} K(tau) u_{L-tau} + K(0) E(q)
    历史摘要与 query 可加分离，query 只经由一项线性项进入。
    任何固定线性 readout 都保持该分离性，故无法实现内容寻址。

任务（刻意取**最有利**的设置，使下界更强）：
    序列 [k_1 v_1 ... k_n v_n ... q]，key 用正交 one-hot，
    value 用共享的随机单位向量，query == 第 j 个 key。
    目标输出 v_j。j 在 {0..n-1} 间变化。
    在这些最干净的条件下静态仍失败 -> 更真实、更嘈杂的设置只会更差。

四类对照（全部在**同一套共享嵌入**上 train / held-out 分离评估）：
  (1) static     : 任意固定线性映射作用于拼接的整段 u
                   （比静态核 K(tau) 更宽松 = 乐观上界）
  (2) naive_bilin: mean_s(q ⊙ u_s)，有乘积交互但抹平位置、无 key/value 分离
  (3) matched    : F = Σ_p (q·k_p) v_p —— attention / 线性注意力 / FD-CAO
                   的最小交互形式（key/value 分离 + 成对匹配加权）
  (4) 解析玩具实例：显式写出「左边与 value 无关 / 右边随 value 任意」的矛盾。

不训练模型、不依赖 torch、不写入任何外部数据。
"""
import json
import os
import numpy as np


def make_embed(V, n_pairs, d, seed):
    """共享嵌入：value = 随机单位向量；key = 正交 one-hot（前 n_pairs 个坐标）。"""
    r = np.random.default_rng(seed)
    val = r.standard_normal((V, d))
    val /= np.linalg.norm(val, axis=1, keepdims=True)
    key = np.zeros((n_pairs, d))
    for p in range(n_pairs):
        key[p, p] = 1.0
    return val, key


def build_task(V, n_pairs, d, n_samples, val, key, seed):
    r = np.random.default_rng(seed)
    L = 2 * n_pairs + 1
    U = np.zeros((n_samples, L, d))
    Y = np.zeros((n_samples, d))
    K = np.zeros((n_samples, n_pairs, d))
    Vv = np.zeros((n_samples, n_pairs, d))
    for i in range(n_samples):
        vals = r.choice(V, size=n_pairs, replace=False)
        j = int(r.integers(0, n_pairs))
        for p in range(n_pairs):
            U[i, 2 * p, :] = key[p]
            U[i, 2 * p + 1, :] = val[vals[p]]
            K[i, p, :] = key[p]
            Vv[i, p, :] = val[vals[p]]
        U[i, L - 1, :] = key[j]          # query == 第 j 个 key（正交 one-hot）
        Y[i] = val[vals[j]]
    return U, Y, K, Vv


def ridge(X, Y, lam=1e-6):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


def l2(pred, Y):
    return float(np.sqrt(((pred - Y) ** 2).sum(1)).mean())


def static(U, Y, lam=1e-6):
    n, L, d = U.shape
    A = ridge(U.reshape(n, L * d), Y, lam)
    return A, lambda Uu: Uu.reshape(Uu.shape[0], L * d) @ A


def naive_bilin(U, Y, lam=1e-6):
    n, L, d = U.shape
    q = U[:, -1, :]
    F = (q[:, None, :] * U[:, :-1, :]).mean(1)
    W = ridge(F, Y, lam)
    return lambda Uu: ((Uu[:, -1, :][:, None, :] * Uu[:, :-1, :]).mean(1)) @ W


def matched(U, Y, K, Vv, lam=1e-6):
    n, L, d = U.shape
    q = U[:, -1, :]
    scores = (q[:, None, :] * K).sum(2)
    F = (scores[:, :, None] * Vv).sum(1)
    W = ridge(F, Y, lam)
    def fn(Uu, Kk, Vvv):
        s = (Uu[:, -1, :][:, None, :] * Kk).sum(2)
        return ((s[:, :, None] * Vvv).sum(1)) @ W
    return fn


def analytic_toy():
    """d=1 极小实例：q1->j1, q2->j2, j1≠j2。
    静态输出 = A + c0·E(q)，A 与 q 无关。相减：
        c0·(E(q1)−E(q2)) = E(v_j1)−E(v_j2)
    左边与 value 赋值无关，右边随 value 任意变化 → 矛盾。"""
    c0, Eq1, Eq2 = 0.7, 0.9, -0.4
    lhs = c0 * (Eq1 - Eq2)
    rhses = [v1 - v2 for v1, v2 in
             [(0.1, 0.8), (-0.6, 0.3), (0.5, -0.9), (0.2, 0.2)]]
    return dict(lhs_fixed=lhs, rhs_possible=rhses,
                max_abs_gap=float(max(abs(lhs - r) for r in rhses)),
                contradiction=bool(max(abs(lhs - r) for r in rhses) > 1e-12))


def main():
    out = {
        "note": "B1 additive-separation numerical corollary; shared embeddings; "
                "train/held-out disjoint; no training, no torch, no external data",
        "setting": "key = orthogonal one-hot, value = random unit vectors, "
                   "query = j-th key. Deliberately the MOST favorable setting, "
                   "so a static failure here is a stronger lower bound.",
        "static_baseline": "arbitrary FIXED linear map on the concatenated u "
                           "(strictly more permissive than a static K(tau)) -> "
                           "optimistic upper bound on any static-kernel model",
    }

    rows = []
    for V in (8, 16, 32, 64):
        for d in (16, 32, 64):
            val, key = make_embed(V, 4, d, seed=4242 + V * 100 + d)
            Utr, Ytr, Ktr, Vtr = build_task(V, 4, d, 256, val, key, seed=1000 + V * 10 + d)
            Ute, Yte, Kte, Vte = build_task(V, 4, d, 128, val, key, seed=9000 + V * 10 + d)
            _, fs = static(Utr, Ytr)
            fn = naive_bilin(Utr, Ytr)
            fm = matched(Utr, Ytr, Ktr, Vtr)
            rows.append(dict(
                V=V, d=d,
                static_train_l2=round(l2(fs(Utr), Ytr), 6),
                static_heldout_l2=round(l2(fs(Ute), Yte), 6),
                naive_bilinear_heldout_l2=round(l2(fn(Ute), Yte), 6),
                matched_bilinear_heldout_l2=round(l2(fm(Ute, Kte, Vte), Yte), 6)))

    out["scaling"] = rows
    out["analytic_toy"] = analytic_toy()
    out["static_stays_positive"] = all(r["static_heldout_l2"] > 0.5 for r in rows)
    out["naive_bilinear_also_fails"] = all(r["naive_bilinear_heldout_l2"] > 0.5 for r in rows)
    out["matched_bilinear_collapses"] = all(r["matched_bilinear_heldout_l2"] < 0.05 for r in rows)
    out["interpretation"] = (
        "静态线性读出（连比 K(tau) 更宽松的版本）在最有利设置下 held-out 误差恒 >0.5；"
        "仅加乘积交互但抹平位置同样失败；只有 key/value 分离 + (q·k_p) 匹配加权"
        "（attention / 线性注意力 / FD-CAO 的最小形式）把误差压到 ~0。"
        "与 B1 推论一致：打破可加分离需要 query 与 key 的成对匹配，"
        "不是任意非线性、任意门控或加长记忆。")

    os.makedirs("results", exist_ok=True)
    with open("results/b1_corollary.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
