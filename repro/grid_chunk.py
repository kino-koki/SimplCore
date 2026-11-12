"""网格扫描（可续跑块）：python grid_chunk.py <task> <grid> <total> <chunk> [seed]"""
import sys, os, json, time, tempfile
import numpy as np
import nn as N, sgc_core as S, exp_probe as P

task, gname, total, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
spec = P.TASKS[task](np.random.default_rng(abs(hash(task)) % 10000))
M = 32
if gname == "default":
    grid = S.make_grid(M=M, rho_max=0.999, rho_min=0.9)
elif gname.startswith("linear"):
    lo = float(gname.replace("linear", "")) / 100
    grid = dict(M=M, theta=np.linspace(lo, np.pi - lo, M),
                rho=np.linspace(0.999, 0.9, M))
m = N.SGCNet(V=spec.V, D=48, M=M, K=4, n_layers=2,
             rng=np.random.default_rng(1000 + seed), grid=grid)
opt = N.AdamW(m.params(), lr=3e-3, wd=0.0)
rngd = np.random.default_rng(seed * 7 + 1)
os.makedirs("ckpt", exist_ok=True)
CK = f"ckpt/gc_{task}_{gname}_{seed}.npz"
done = 0
if os.path.exists(CK):
    d0 = np.load(CK)
    for i, p in enumerate(m.params()):
        p.v = d0[f"p{i}"]; opt.m[i] = d0[f"m{i}"]; opt.v[i] = d0[f"v{i}"]
    done = int(d0["done"]); opt.t = done
    for _ in range(done): spec.gen(16, rngd)
t0 = time.time(); n = min(chunk, total - done)
last = None
for st in range(n):
    xb, yb = spec.gen(16, rngd)
    lg = m.forward(xb); loss = N.cross_entropy(N.last(lg), yb)
    loss.backward(); opt.step()
    last = float(loss.v)
done += n
fd, tmp = tempfile.mkstemp(dir="ckpt", suffix=".npz"); os.close(fd)
np.savez(tmp, done=np.array(done),
         **{k: v for i, p in enumerate(m.params())
            for k, v in [(f"p{i}", p.v), (f"m{i}", opt.m[i]), (f"v{i}", opt.v[i])]})
os.replace(tmp, CK)
Xte, Yte = spec.gen(512, np.random.default_rng(12345))
preds = []
for s in range(0, 512, 16):
    preds.append(m.forward(Xte[s:s+16]).v[:, -1, :8].argmax(axis=-1))
correct = (np.concatenate(preds) == Yte).astype(float)
acc = float(correct.mean())
db = "results_grid_chunk.json"
try: d = json.load(open(db))
except Exception: d = {}
d[f"{task}|{gname}|{done}|{seed}"] = dict(task=task, grid=gname, done=done, seed=seed,
                                          acc=acc, loss=last, correct=correct.tolist())
json.dump(d, open(db, "w"), indent=2)
print(f"[{task} {gname:9s} {done}/{total}步] acc={acc:.4f} loss={last:.3f} "
      f"({time.time()-t0:.0f}s)", flush=True)
