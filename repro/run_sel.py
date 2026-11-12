"""selective SGC vs SGC vs Transformer on MQAR
python run_sel.py <steps> <seed> [task]
"""
import sys, json, time
import numpy as np
import nn as N, sgc_core as S, exp_probe as P

STEPS = int(sys.argv[1]); SEED = int(sys.argv[2])
NAME = sys.argv[3] if len(sys.argv) > 3 else "mqar"
spec = P.TASKS[NAME](np.random.default_rng(abs(hash(NAME)) % 10000))
grid = S.make_grid(M=32, rho_max=0.999, rho_min=0.9)
res = {}
for kind in ["sel", "sgc"]:
    rng = np.random.default_rng(1000 + SEED)
    m = (N.SelSGCNet(V=spec.V, D=48, M=32, K=4, n_layers=2, rng=rng, grid=grid)
         if kind == "sel" else N.SGCNet(V=spec.V, D=48, M=32, K=4, n_layers=2, rng=rng, grid=grid))
    opt = N.AdamW(m.params(), lr=3e-3, wd=0.0)
    rngd = np.random.default_rng(SEED * 7 + 1)
    t0 = time.time()
    for st in range(STEPS):
        xb, yb = spec.gen(16, rngd)
        lg = m.forward(xb)
        loss = N.cross_entropy(N.last(lg), yb)
        loss.backward()
        opt.step()
    Xte, Yte = spec.gen(512, np.random.default_rng(12345))
    preds = []
    for s in range(0, 512, 16):
        preds.append(m.forward(Xte[s:s+16]).v[:, -1, :8].argmax(axis=-1))
    acc = float((np.concatenate(preds) == Yte).mean())
    res[kind] = acc
    print(f"[{NAME} seed={SEED} {STEPS}步] {kind.upper():4s} acc={acc:.4f} "
          f"params={m.n_params()} loss={float(loss.v):.3f} ({time.time()-t0:.0f}s)", flush=True)
db = "results_selective.json"
try: d = json.load(open(db))
except Exception: d = {}
d[f"{NAME}|{STEPS}|{SEED}"] = dict(steps=STEPS, seed=SEED, **res)
json.dump(d, open(db, "w"), indent=2)
