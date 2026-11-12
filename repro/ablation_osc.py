"""消融：把 SGC 的阻尼振荡基换成随机基，检验"振荡归纳偏置"是否是因果因素
python ablation_osc.py <task> <steps> <mode:osc|rand|perm>
"""
import sys, json, time
import numpy as np
import nn as N, sgc_core as S, exp_probe as P

task, steps, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
spec = P.TASKS[task](np.random.default_rng(abs(hash(task)) % 10000))
grid = S.make_grid(M=32, rho_max=0.999, rho_min=0.9)
base = S.impulse_response(grid, spec.L)
rng = np.random.default_rng(999)
if mode == "osc":
    hfun = None
elif mode == "rand":
    R = rng.standard_normal((32, spec.L)).astype(np.float32)
    R /= np.abs(R).max(axis=1, keepdims=True)
    hfun = lambda L: R[:, :L]
elif mode == "perm":                      # 保留幅度谱，打乱时间结构（非振荡）
    Pm = rng.permutation(spec.L)
    hfun = lambda L: np.ascontiguousarray(base[:, Pm[:L]])
m = N.SGCNet(V=spec.V, D=48, M=32, K=4, n_layers=2,
             rng=np.random.default_rng(1000), grid=grid)
m.hfun = hfun
opt = N.AdamW(m.params(), lr=3e-3, wd=0.0)
rngd = np.random.default_rng(1)
t0 = time.time()
for st in range(steps):
    xb, yb = spec.gen(16, rngd)
    lg = m.forward(xb); loss = N.cross_entropy(N.last(lg), yb)
    loss.backward(); opt.step()
Xte, Yte = spec.gen(512, np.random.default_rng(12345))
preds = []
for s in range(0, 512, 16):
    preds.append(m.forward(Xte[s:s+16]).v[:, -1, :8].argmax(axis=-1))
acc = float((np.concatenate(preds) == Yte).mean())
print(f"[{task} {steps}步] basis={mode:5s} acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
db = "results_osc_ablation.json"
try: d = json.load(open(db))
except Exception: d = {}
d[f"{task}|{steps}|{mode}"] = dict(mode=mode, steps=steps, acc=acc)
json.dump(d, open(db, "w"), indent=2)
