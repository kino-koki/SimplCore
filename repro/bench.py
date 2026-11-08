"""训练吞吐 vs 序列长度：验证 SGC 的 O(D log L) 与 TS 的 O(L²) 缩放主张。"""
import json, time
import numpy as np
import nn as N

LENS = [128, 256, 512, 1024]
B = 2
REPEAT = 3


def bench(model, V, L, repeat=REPEAT):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, V, size=(B, L))
    tgt = rng.integers(0, V, size=(B,))
    ts = []
    for _ in range(repeat):
        t0 = time.time()
        lg = model.forward(idx)
        loss = N.cross_entropy(N.last(lg), tgt)
        loss.backward()
        for p in model.params():
            p.g[:] = 0
        ts.append(time.time() - t0)
    return float(np.median(ts))


def main():
    out = {"B": B, "repeat": REPEAT, "lens": LENS,
           "sgc": dict(D=48, M=32, K=4, n_layers=2),
           "ts": dict(D=56, n_heads=4, n_layers=2), "rows": []}
    V = 64
    for L in LENS:
        row = dict(L=L)
        for kind in ["sgc", "ts"]:
            rng = np.random.default_rng(0)
            m = (N.SGCNet(V=V, rng=rng, D=48, M=32, K=4, n_layers=2) if kind == "sgc"
                 else N.TSNet(V=V, rng=rng, D=56, n_heads=4, n_layers=2))
            row[kind] = bench(m, V, L)
        out["rows"].append(row)
        print(f"L={L:>5}  SGC={row['sgc']:.3f}s  TS={row['ts']:.3f}s  "
              f"TS/SGC={row['ts']/row['sgc']:.2f}x", flush=True)
    # 拟合 log-log 斜率（后半段）
    ls = np.log(np.array(LENS, dtype=float))
    for kind in ["sgc", "ts"]:
        ts_ = np.log(np.array([r[kind] for r in out["rows"]]))
        k, b = np.polyfit(ls, ts_, 1)
        out[kind + "_slope"] = float(k)
        print(f"{kind} log-log 斜率 = {k:.2f}", flush=True)
    json.dump(out, open("results_throughput.json", "w"), indent=2)
    # L=2048 单独补测（B=1，避免内存抖动影响计时）
    try:
        L = 2048
        rng = np.random.default_rng(0)
        idx = rng.integers(0, V, size=(1, L)); tgt = rng.integers(0, V, size=(1,))
        row = {"L": L, "B": 1}
        for kind in ["sgc", "ts"]:
            m = (N.SGCNet(V=V, rng=np.random.default_rng(0), D=48, M=32, K=4, n_layers=2)
                 if kind == "sgc" else
                 N.TSNet(V=V, rng=np.random.default_rng(0), D=56, n_heads=4, n_layers=2))
            t0 = time.time()
            lg = m.forward(idx); l = N.cross_entropy(N.last(lg), tgt); l.backward()
            row[kind] = time.time() - t0
        out["rows"].append(row)
        print(f"L=2048(B=1) SGC={row['sgc']:.3f}s TS={row['ts']:.3f}s "
              f"ratio={row['ts']/row['sgc']:.2f}x", flush=True)
    except Exception as e:
        print("L=2048 skipped:", e)
    json.dump(out, open("results_throughput.json", "w"), indent=2)
    ls = np.log(np.array([r["L"] for r in out["rows"]], dtype=float))
    for kind in ["sgc", "ts"]:
        ts_ = np.log(np.array([r[kind] for r in out["rows"]]))
        k, b = np.polyfit(ls, ts_, 1)
        print(f"{kind} log-log slope (含L=2048) = {k:.2f}")


if __name__ == "__main__":
    main()
