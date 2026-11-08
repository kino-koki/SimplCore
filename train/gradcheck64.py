"""fp64 严格梯度检验：把 core 全局切到 float64，用 1e-6 中心差分。"""
import numpy as np
import core as C

C.F32 = np.float64           # 全局 fp64（含 mask dtype / astype）
B, L, SEED = 2, 7, 0
EPS = 1e-6
rng = np.random.default_rng(SEED)


def gradcheck(kind):
    P = C.init_params(kind, seed=SEED)
    idx = rng.integers(0, C.V, size=(B, L)).astype(np.int32)
    psi = C.psi_eff(P["rho"], P["theta"], L) if kind == "sgc" else None
    W = rng.standard_normal((B, L, C.V))

    def f():
        logits, _ = C.forward(idx, P, psi)
        return float((logits * W).sum() / (B * L))

    dlogits = (W / (B * L)).astype(C.F32)
    cache = C.forward(idx, P, psi)[1]
    g = C.backward(dlogits, P, cache)

    out = []
    for name, arr in C._iter_params(P):
        gv = g.get(name)
        if gv is None:
            print("  MISSING GRAD", name); continue
        assert gv.shape == arr.shape, (name, gv.shape, arr.shape)
        flat, gf = arr.reshape(-1), gv.reshape(-1)
        sel = rng.choice(flat.size, size=min(8, flat.size), replace=False)
        num, ana = [], []
        for j in sel:
            o = flat[j]
            flat[j] = o + EPS; vp = f()
            flat[j] = o - EPS; vm = f()
            flat[j] = o
            num.append((vp - vm) / (2 * EPS)); ana.append(float(gf[j]))
        num, ana = np.array(num), np.array(ana)
        rel = np.abs(num - ana) / np.maximum(np.abs(ana), 1e-9)
        out.append((name, float(rel.max()), float(np.abs(num - ana).max()),
                    int((rel > 1e-4).sum()), len(sel)))
    worst = max(out, key=lambda x: x[1])
    print(f"\n[{kind}] fp64 | 参数组 {len(out)} | 最坏相对 {worst[1]:.3e} ({worst[0]})")
    for n, r, a, bad, tot in sorted(out, key=lambda x: -x[1])[:5]:
        print(f"   rel={r:.2e} abs={a:.2e} 超差 {bad}/{tot}  {n}")
    return all(bad == 0 for _, _, _, bad, _ in out)


if __name__ == "__main__":
    ok1 = gradcheck("ts")
    ok2 = gradcheck("sgc")
    print("\n==>", "GRADCHECK PASS (fp64)" if (ok1 and ok2) else "GRADCHECK FAIL",
          "| ts", ok1, "sgc", ok2)
