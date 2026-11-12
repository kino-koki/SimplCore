"""表达能力实验：精确状态跟踪的长度泛化
python expr_chunk.py <model> <total> <chunk> [seed]
model: sgc_pole (ρ=1 积分器+无损振荡) | sgc_default (论文默认 ρ<1) | ts
训练长度 L=64，评估长度 L ∈ {64,128,256,512}，k 分布与 L 解耦（公平）
"""
import sys, os, json, time, tempfile
import numpy as np
import nn as N, sgc_core as S

KIND, TOTAL, CHUNK = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
V, KMAX, LTRAIN = 8, 24, 64   # V=8 既是词表也是分类数；输入只用 0/1/2
LENS = [64, 128, 256, 512]


def gen(n, rng, L):
    """Σx = k 固定分布（与 L 无关），k 个 1 随机散布；末尾加 query token。"""
    k = rng.integers(0, KMAX, size=n)
    X = np.zeros((n, L), dtype=np.int64)
    for i in range(n):
        if k[i] > 0:
            X[i, rng.choice(L, size=int(k[i]), replace=False)] = 1
    seq = np.concatenate([X, np.full((n, 1), 2)], axis=1)
    return seq, (k % V).astype(np.int64)


def build():
    rng = np.random.default_rng(1000 + SEED)
    if KIND == "ts":
        return N.TSNet(V=8, D=56, n_heads=4, n_layers=2, rng=rng), None
    grid = (S.make_grid(M=32, rho_max=0.999, rho_min=0.9,
                        integrator=2, unit_circle=3,
                        uc_thetas=[2*np.pi/8, 2*np.pi/4, 2*np.pi*3/8])
            if KIND == "sgc_pole" else
            S.make_grid(M=32, rho_max=0.999, rho_min=0.9))
    return N.SGCNet(V=8, D=48, M=32, K=4, n_layers=2, rng=rng, grid=grid), grid


model, _ = build()
opt = N.AdamW(model.params(), lr=3e-3, wd=0.0)
rngd = np.random.default_rng(SEED * 7 + 1)
os.makedirs("ckpt", exist_ok=True)
CK = f"ckpt/expr_{KIND}_{SEED}.npz"
done = 0
if os.path.exists(CK):
    d0 = np.load(CK)
    for i, p in enumerate(model.params()):
        p.v = d0[f"p{i}"]; opt.m[i] = d0[f"m{i}"]; opt.v[i] = d0[f"v{i}"]
    done = int(d0["done"]); opt.t = done
    for _ in range(done): gen(16, rngd, LTRAIN)
t0 = time.time(); n = min(CHUNK, TOTAL - done)
last = None
for st in range(n):
    xb, yb = gen(16, rngd, LTRAIN)
    lg = model.forward(xb)
    loss = N.cross_entropy(N.last(lg), yb)
    loss.backward(); opt.step()
    last = float(loss.v)
done += n
fd, tmp = tempfile.mkstemp(dir="ckpt", suffix=".npz"); os.close(fd)
np.savez(tmp, done=np.array(done),
         **{k: v for i, p in enumerate(model.params())
            for k, v in [(f"p{i}", p.v), (f"m{i}", opt.m[i]), (f"v{i}", opt.v[i])]})
os.replace(tmp, CK)

accs = {}
for L in LENS:
    Xte, Yte = gen(256, np.random.default_rng(999), L)
    preds = []
    for s in range(0, 256, 16):
        preds.append(model.forward(Xte[s:s+16]).v[:, -1, :V].argmax(axis=-1))
    accs[str(L)] = float((np.concatenate(preds) == Yte).mean())

db = "results_expressivity.json"
try: d = json.load(open(db))
except Exception: d = {}
d[KIND] = dict(done=done, seed=SEED, params=model.n_params(),
               loss=last, accs=accs)
json.dump(d, open(db, "w"), indent=2)
print(f"[{KIND} {done}/{TOTAL}步] " + " ".join(f"L{L}={accs[str(L)]:.3f}" for L in LENS)
      + f"  loss={last:.3f} ({time.time()-t0:.0f}s)", flush=True)
