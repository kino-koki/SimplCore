"""selective SGC 增量训练：python sel_chunk.py <task> <seed> <total> <chunk>"""
import sys, os, json, time
import numpy as np
import nn as N, sgc_core as S, exp_probe as P

task, seed, total, chunk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
spec = P.TASKS[task](np.random.default_rng(abs(hash(task)) % 10000))
grid = S.make_grid(M=32, rho_max=0.999, rho_min=0.9)
CK = f"ckpt/sel_{task}_{seed}.npz"; os.makedirs("ckpt", exist_ok=True)
m = N.SelSGCNet(V=spec.V, D=48, M=32, K=4, n_layers=2,
                rng=np.random.default_rng(1000 + seed), grid=grid)
opt = N.AdamW(m.params(), lr=3e-3, wd=0.0)
done = 0
if os.path.exists(CK):
    d = np.load(CK)
    for i, p in enumerate(m.params()):
        p.v = d[f"p{i}"]; opt.m[i] = d[f"m{i}"]; opt.v[i] = d[f"v{i}"]
    done = int(d["done"]); opt.t = done
rngd = np.random.default_rng(seed * 7 + 1)
for _ in range(done):                      # 对齐数据流（严格复现用）
    spec.gen(16, rngd)
t0 = time.time(); n = min(chunk, total - done)
for st in range(n):
    xb, yb = spec.gen(16, rngd)
    lg = m.forward(xb); loss = N.cross_entropy(N.last(lg), yb)
    loss.backward(); opt.step()
done += n
np.savez(CK, done=np.array(done),
         **{k: v for i, p in enumerate(m.params())
            for k, v in [(f"p{i}", p.v), (f"m{i}", opt.m[i]), (f"v{i}", opt.v[i])]})
Xte, Yte = spec.gen(512, np.random.default_rng(12345))
preds = []
for s in range(0, 512, 16):
    preds.append(m.forward(Xte[s:s+16]).v[:, -1, :8].argmax(axis=-1))
acc = float((np.concatenate(preds) == Yte).mean())
db = "results_selective.json"
try: d = json.load(open(db))
except Exception: d = {}
d[f"{task}|{total}|{seed}|sel"] = dict(steps=done, acc=acc, params=m.n_params(),
                                       loss=float(loss.v))
json.dump(d, open(db, "w"), indent=2)
print(f"sel {task} {done}/{total} acc={acc:.4f} params={m.n_params()} "
      f"loss={float(loss.v):.3f} ({time.time()-t0:.0f}s)", flush=True)
