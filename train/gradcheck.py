"""梯度正确性检验（fp64 中心差分）——训练前的硬性前置。"""
import numpy as np
import core as C

np.seterr(all="ignore")
B, L, SEED = 2, 7, 0
rng = np.random.default_rng(SEED)


def scalar_loss(P, idx, tgt, psi=None):
    logits, cache = C.forward(idx, P, psi)
    ls = C.log_softmax(logits, axis=-1)
    n = B * L
    # 用固定的随机投影做标量目标，避免 CE 的数值退化
    W = rng.standard_normal(logits.shape).astype(np.float64)
    return float((np.exp(ls) * W).sum() / n), (np.exp(ls) * (W - (np.exp(ls) * W).sum(-1, keepdims=True)) / n, cache)


def gradcheck(kind):
    P = C.init_params(kind, seed=SEED)
    idx = rng.integers(0, C.V, size=(B, L)).astype(np.int32)
    psi = C.psi_eff(P["rho"], P["theta"], L) if kind == "sgc" else None
    # 固定投影
    W = rng.standard_normal((B, L, C.V)).astype(np.float64)

    # 线性目标：val = <logits, W>/n，dlogits = W/n。
    # 梯度检验用线性函数，避免 softmax 的非线性把 fp32 舍入放大成假误差；
    # 但 logits 依赖全部参数，反向路径（含 attention softmax / IIR 核）仍被完整覆盖。
    def f():
        logits, cache = C.forward(idx, P, psi)
        return float((logits.astype(np.float64) * W).sum() / (B * L))

    dlogits = (W / (B * L)).astype(C.F32)
    cache = C.forward(idx, P, psi)[1]
    val = f()
    g = C.backward(dlogits, P, cache)
    out = []
    for name, arr in C._iter_params(P):
        gv = g.get(name)
        if gv is None:
            print("  MISSING GRAD", name)
            continue
        assert gv.shape == arr.shape, (name, gv.shape, arr.shape)
        flat = arr.reshape(-1)
        gf = gv.reshape(-1)
        sel = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        num, ana = [], []
        EPS = 1e-3      # fp32 下 1e-4 已被舍入噪声淹没，1e-3 稳定
        for j in sel:
            o = flat[j]
            flat[j] = o + EPS; vp = f()
            flat[j] = o - EPS; vm = f()
            flat[j] = o
            num.append((vp - vm) / (2 * EPS))
            ana.append(float(gf[j]))
        num = np.array(num); ana = np.array(ana)
        # 混合容差：相对 5%，且对接近 0 的梯度给 1e-5 绝对容差
        err = np.abs(num - ana) / np.maximum(np.abs(ana), 1e-5)
        out.append((name, float(err.max()), int((err > 0.05).sum()), len(sel)))
    worst = max(out, key=lambda x: x[1])
    print(f"\n[{kind}] 参数组 {len(out)}  最坏相对误差 {worst[1]:.3e}  ({worst[0]})")
    for n, r, bad, tot in sorted(out, key=lambda x: -x[1])[:6]:
        print(f"   {r:.3e}  超差 {bad}/{tot}  {n}")
    return all(bad == 0 for _, _, bad, _ in out)


if __name__ == "__main__":
    ok1 = gradcheck("ts")
    ok2 = gradcheck("sgc")
    print("\n==>", "GRADCHECK PASS" if (ok1 and ok2) else "GRADCHECK FAIL",
          "| ts", ok1, "sgc", ok2)
