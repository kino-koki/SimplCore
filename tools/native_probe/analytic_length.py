"""
analytic_length.py — 不建 autograd 图的解析式评测。

用于验证固定正交谱字典在**训练长度之外**的外推能力。
与 forward() 数学完全一致，但不保留中间节点，因此可跑到 L=257+。

用法: python analytic_length.py
"""
import numpy as np
import tensor_spectral_probe as T
import task_utils as U


def analytic_eval(arm, P, X, Xp, Y, chunk=16):
    cfg = T.ARMS[arm]
    B_tot = len(Y); L = X.shape[1]
    R, N, Pp = cfg["R"], cfg["N"], cfg["P"]
    Code = T._fixed_code(R, U.VOCAB, cfg.get("code_kind", "hadamard"))
    rho = (T.RHO_MIN + (1 - T.RHO_MIN) * T._sig(P["p"].d)).reshape(Pp, N)
    w = P["w"].d.reshape(Pp, N)
    E = P["E"].d; Wv = P["Wv"].d; Wout = P["Wout"].d
    out = np.zeros(B_tot, dtype=int)
    for i in range(0, B_tot, chunk):
        sl = slice(i, i + chunk)
        xb, xpb = X[sl], Xp[sl]
        B = len(xb)
        a = xpb @ Code                                  # (B,L,R)
        b = (xb @ E) @ Wv                                # (B,L,N)
        q = xb[:, L - 1, :] @ Code                       # (B,R)
        s = np.zeros((B, Pp, R, N))
        for t in range(L):
            u_t = a[:, t, :, None] * b[:, t, None, :]    # (B,R,N)
            s = rho[None, :, None, :] * s + u_t[:, None, :, :]
        zr = (s * q[:, None, :, None]).sum(axis=2)       # (B,P,N)
        z = (zr * w[None, :, :]).sum(axis=1)             # (B,N)
        logits = z @ Wout
        out[sl] = (logits.argmax(-1) == Y[sl]).astype(int)
    return out


def build_long(n_pairs, n_samples, seed):
    """允许 key 重复，但保证 query 指向的 key 在序列中唯一 ⇒ 答案仍唯一。"""
    r = np.random.default_rng(seed)
    L = 2 * n_pairs + 1
    X = np.zeros((n_samples, L, U.VOCAB)); Xp = np.zeros_like(X)
    Y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        # 其余位置只用 0..V_KEY-2，留 V_KEY-1 作为唯一 query key
        keys = r.choice(U.V_KEY - 1, size=n_pairs, replace=True)
        vals = r.choice(U.V_VAL, size=n_pairs, replace=True) + U.V_KEY
        j = int(r.integers(0, n_pairs))
        q = U.V_KEY - 1
        keys[j] = q                      # 保证该 key 全序列仅出现一次
        for t, (k, v) in enumerate(zip(keys, vals)):
            X[i, 2 * t, k] = 1.0
            X[i, 2 * t + 1, v] = 1.0
        X[i, L - 1, q] = 1.0
        Y[i] = vals[j]
        Xp[i, 1:] = X[i, :-1]
    return X, Xp, Y


def main():
    arm = "W7_more_pairs"
    print("训练 W7 (n_pairs=8, L=17) ...", flush=True)
    r = T.run_train(arm, 0, steps=1500)
    P = T._LAST_P
    print(f"train acc={r['final_acc']:.4f} perm={r['perm_acc']:.4f}\n", flush=True)

    # 一致性校验：L=9 上解析式 vs forward
    X, Xp, Y, _ = U.build(4, 64, seed=42)
    c1 = T.eval_correct(arm, P, X, Xp, Y)
    c2 = analytic_eval(arm, P, X, Xp, Y)
    print(f"一致性校验 (L=9): forward={c1.mean():.4f}  analytic={c2.mean():.4f}  "
          f"逐样本相同={np.array_equal(c1,c2)}\n", flush=True)

    print("=== 训练长度之外的外推（模型在 L=17 训练，参数不更新）===", flush=True)
    res = {}
    for n in (4, 8, 12, 16, 20, 32):
        L = 2 * n + 1
        Xg, Xpg, Yg, _ = U.build(n, 256, seed=7000 + n)
        c = analytic_eval(arm, P, Xg, Xpg, Yg)
        res[f"L{L}"] = float(c.mean())
        print(f"  L={L:4d}  acc={c.mean():.4f}", flush=True)
    print("  --- 以下允许 key 重复（query 指向唯一 key）---", flush=True)
    for n in (64, 128, 256, 512):
        L = 2 * n + 1
        Xg, Xpg, Yg = build_long(n, 128, seed=7000 + n)
        c = analytic_eval(arm, P, Xg, Xpg, Yg, chunk=4)
        res[f"L{L}"] = float(c.mean())
        print(f"  L={L:4d}  acc={c.mean():.4f}", flush=True)
    import json
    json.dump(dict(train_acc=r["final_acc"], perm_acc=r["perm_acc"],
                   forward_analytic_match=bool(np.array_equal(c1, c2)),
                   length_extrapolation=res),
              open("results/analytic_length.json", "w"), indent=1)
    print("\n写入 results/analytic_length.json")


if __name__ == "__main__":
    main()
