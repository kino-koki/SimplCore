"""增量训练：python run_chunk.py <task> <seed> <kind> <total_steps> <chunk>
模型+优化器状态落盘 npz，可跨多次调用续跑（规避单次调用超时）。
"""
import sys, os, json, time
import numpy as np
import nn as N
import exp_probe as P

SGC_CFG = dict(D=48, M=32, K=4, n_layers=2)
TS_CFG = dict(D=56, n_heads=4, n_layers=2)
DB = "results_all.json"
CKPT = "ckpt"


def spec_of(name):
    return P.TASKS[name](np.random.default_rng(abs(hash(name)) % 10000))


def build(name, kind, seed):
    spec = spec_of(name)
    rng = np.random.default_rng(1000 + seed)
    m = (N.SGCNet(V=spec.V, rng=rng, **SGC_CFG) if kind == "sgc"
         else N.TSNet(V=spec.V, rng=rng, **TS_CFG))
    return spec, m


def path(name, seed, kind):
    os.makedirs(CKPT, exist_ok=True)
    return f"{CKPT}/{name}_{seed}_{kind}.npz"


def save(name, seed, kind, model, opt, done):
    d = dict(done=np.array(done))
    for i, p in enumerate(model.params()):
        d[f"p{i}"] = p.v
        d[f"m{i}"] = opt.m[i]
        d[f"v{i}"] = opt.v[i]
    p = path(name, seed, kind)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=CKPT, suffix=".npz")
    os.close(fd)
    np.savez(tmp, **d)                 # 原子写：避免中断写出损坏的 npz
    os.replace(tmp, p)


def load(model, opt, name, seed, kind):
    f = path(name, seed, kind)
    if not os.path.exists(f):
        return 0
    try:
        d = np.load(f)
    except Exception:
        os.remove(f); return 0         # 损坏的断点直接丢弃重来
    for i, p in enumerate(model.params()):
        p.v = d[f"p{i}"]
        opt.m[i] = d[f"m{i}"]
        opt.v[i] = d[f"v{i}"]
    return int(d["done"])


def evaluate(model, spec, n_eval=512, bs=16):
    Xte, Yte = spec.gen(n_eval, np.random.default_rng(12345))
    if spec.is_lm:
        nll, ntok = 0.0, 0
        for s in range(0, Xte.shape[0], bs):
            xb = Xte[s:s + bs]
            lg = model.forward(xb).v[:, :-1, :]
            z = lg - lg.max(axis=-1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
            b, l = np.arange(xb.shape[0]), np.arange(xb.shape[1] - 1)
            nll += -lp[b[:, None], l[None, :], xb[:, 1:]].sum()
            ntok += xb.shape[0] * (xb.shape[1] - 1)
        return dict(ppl=float(np.exp(nll / ntok))), None
    preds = []
    for s in range(0, Xte.shape[0], bs):
        pr = model.forward(Xte[s:s + bs]).v[:, -1, :spec.ncls]
        preds.append(pr.argmax(axis=-1))
    preds = np.concatenate(preds)
    correct = (preds == Yte).astype(np.float64)
    return dict(acc=float(correct.mean())), correct


if __name__ == "__main__":
    name, seed, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    total, chunk = int(sys.argv[4]), int(sys.argv[5])
    spec, model = build(name, kind, seed)
    opt = N.AdamW(model.params(), lr=3e-3, wd=0.0)
    done = load(model, opt, name, seed, kind)
    if done >= total and len(sys.argv) < 7:
        print(f"{name}|{seed}|{kind} already {done}/{total}")
        sys.exit(0)
    rng = np.random.default_rng(seed * 7 + 1)
    opt.t = done
    t0 = time.time()
    n = min(chunk, total - done)
    last = 0.0
    for st in range(n):
        xb, yb = spec.gen(16, rng)
        lg = model.forward(xb)
        if spec.is_lm:
            loss = N.cross_entropy(N.flatten2d(N.slice_tail(lg, 1)), xb[:, 1:].ravel())
        else:
            loss = N.cross_entropy(N.last(lg), yb)
        loss.backward()
        opt.step()
        last = float(loss.v)
    done += n
    save(name, seed, kind, model, opt, done)
    out, correct = evaluate(model, spec)
    db = json.load(open(DB)) if os.path.exists(DB) else {}
    key = f"{name}|{seed}"
    row = db.get(key, dict(seed=seed, chance=spec.chance, V=int(spec.V),
                           L=int(spec.L), steps=total))
    for k, v in out.items():
        row[kind + "_" + k] = v
    row[kind + "_params"] = model.n_params()
    row[kind + "_loss"] = last
    row[kind + "_done"] = done
    row["steps"] = done
    if correct is not None:
        row[kind + "_correct"] = correct.tolist()
    if "sgc_params" in row and "ts_params" in row:
        row["param_ratio"] = row["sgc_params"] / row["ts_params"]
    db[key] = row
    json.dump(db, open(DB, "w"), indent=2)
    print(f"{name}|{seed}|{kind} {done}/{total} "
          + " ".join(f"{k}={v:.4f}" for k, v in out.items())
          + f" loss={last:.3f} ({time.time()-t0:.0f}s)", flush=True)
