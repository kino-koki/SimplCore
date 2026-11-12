"""最优解搜索：频率网格 × 任务
python sweep_grid.py <task> <gridname> <steps> <seed>
gridname: default | linear03 | linear01 | rand
"""
import sys, json, time
import numpy as np
import nn as N, sgc_core as S, exp_probe as P

task, gname, steps, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
spec = P.TASKS[task](np.random.default_rng(abs(hash(task)) % 10000))
M = 32
if gname == "default":
    grid = S.make_grid(M=M, rho_max=0.999, rho_min=0.9)
elif gname.startswith("linear"):
    lo = float(gname.replace("linear", "")) / 100
    th = np.linspace(lo, np.pi - lo, M)
    grid = dict(M=M, theta=th, rho=np.linspace(0.999, 0.9, M))
elif gname == "rand":
    th = np.sort(np.random.default_rng(7).uniform(0.05, np.pi - 0.05, M))
    grid = dict(M=M, theta=th, rho=np.linspace(0.999, 0.9, M))
m = N.SGCNet(V=spec.V, D=48, M=M, K=4, n_layers=2,
             rng=np.random.default_rng(1000 + seed), grid=grid)
opt = N.AdamW(m.params(), lr=3e-3, wd=0.0)
rngd = np.random.default_rng(seed * 7 + 1)
# 断点续跑
import os, tempfile
CK = f"ckpt/grid_{task}_{gname}_{seed}.npz"; os.makedirs("ckpt", exist_ok=True)
done = 0
if os.path.exists(CK):
    d0 = np.load(CK)
    for i, p in enumerate(m.params()):
        p.v = d0[f"p{i}"]; opt.m[i] = d0[f"m{i}"]; opt.v[i] = d0[f"v{i}"]
    done = int(d0["done"]); opt.t = done
    for _ in range(done): spec.gen(16, rngd)
t0 = time.time()
for st in range(done, steps):
    xb, yb = spec.gen(16, rngd)
    lg = m.forward(xb); loss = N.cross_entropy(N.last(lg), yb)
    loss.backward(); opt.step()
fd, tmp = tempfile.mkstemp(dir="ckpt", suffix=".npz"); os.close(fd)
np.savez(tmp, done=np.array(steps),
         **{k: v for i, p in enumerate(m.params())
            for k, v in [(f"p{i}", p.v), (f"m{i}", opt.m[i]), (f"v{i}", opt.v[i])]})
os.replace(tmp, CK)
fd, tmp = tempfile.mkstemp(dir="ckpt", suffix=".npz"); os.close(fd)
np.savez(tmp, done=np.array(steps),
         **{k: v for i, p in enumerate(m.params())
            for k, v in [(f"p{i}", p.v), (f"m{i}", opt.m[i]), (f"v{i}", opt.v[i])]})
os.replace(tmp, CK)
Xte, Yte = spec.gen(512, np.random.default_rng(12345))
preds = []
for s in range(0, 512, 16):
    preds.append(m.forward(Xte[s:s+16]).v[:, -1, :8].argmax(axis=-1))
preds = np.concatenate(preds)
correct = (preds == Yte).astype(float)
acc = float(correct.mean())
H = S.impulse_response(grid, spec.L)
G = H @ H.T; w = np.linalg.eigvalsh(G); cnd = w[-1] / max(w[0], 1e-30)
print(f"[{task} grid={gname:9s} {steps}步 seed={seed}] acc={acc:.4f} "
      f"cond={cnd:.2e} params={m.n_params()} ({time.time()-t0:.0f}s)", flush=True)
db = "results_grid_sweep.json"
try: d = json.load(open(db))
except Exception: d = {}
d[f"{task}|{gname}|{steps}|{seed}"] = dict(task=task, grid=gname, steps=steps, seed=seed,
                                           acc=acc, cond=float(cnd), correct=correct.tolist())
json.dump(d, open(db, "w"), indent=2)
