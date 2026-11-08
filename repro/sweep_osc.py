"""振荡轴最优解搜索：收敛曲线 + 超参扫描
python sweep_osc.py curve <task> <kind> <steps>      每 100 步评估一次
python sweep_osc.py grid  <task> <steps>             M/rho_max 扫描
"""
import sys, json, time
import numpy as np
import nn as N, sgc_core as S, exp_probe as P

def evaluate(m, spec, n=512, bs=16):
    Xte, Yte = spec.gen(n, np.random.default_rng(12345))
    preds = []
    for s in range(0, n, bs):
        preds.append(m.forward(Xte[s:s+bs]).v[:, -1, :8].argmax(axis=-1))
    return float((np.concatenate(preds) == Yte).mean())

mode = sys.argv[1]
DB = "results_osc_sweep.json"
try: db = json.load(open(DB))
except Exception: db = {}

if mode == "curve":
    task, kind, steps = sys.argv[2], sys.argv[3], int(sys.argv[4])
    spec = P.TASKS[task](np.random.default_rng(abs(hash(task)) % 10000))
    grid = S.make_grid(M=32, rho_max=0.999, rho_min=0.9)
    rng = np.random.default_rng(1000)
    m = (N.SGCNet(V=spec.V, D=48, M=32, K=4, n_layers=2, rng=rng, grid=grid) if kind == "sgc"
         else N.TSNet(V=spec.V, D=56, n_heads=4, n_layers=2, rng=rng))
    opt = N.AdamW(m.params(), lr=3e-3, wd=0.0)
    rngd = np.random.default_rng(1)
    curve = []
    t0 = time.time()
    for st in range(steps):
        xb, yb = spec.gen(16, rngd)
        lg = m.forward(xb); loss = N.cross_entropy(N.last(lg), yb)
        loss.backward(); opt.step()
        if (st + 1) % 100 == 0 and (st+1) >= 100:
            curve.append([st + 1, evaluate(m, spec)])
            print(f"  {kind} {task} step={st+1} acc={curve[-1][1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    db[f"curve|{task}|{kind}"] = curve
    json.dump(db, open(DB, "w"), indent=2)

elif mode == "grid":
    task, steps = sys.argv[2], int(sys.argv[3])
    spec = P.TASKS[task](np.random.default_rng(abs(hash(task)) % 10000))
    for M, rho in [(8, 0.999), (32, 0.999), (64, 0.999), (32, 0.9999), (32, 0.99)]:
        grid = S.make_grid(M=M, rho_max=rho, rho_min=0.9)
        m = N.SGCNet(V=spec.V, D=48, M=M, K=4, n_layers=2,
                     rng=np.random.default_rng(1000), grid=grid)
        opt = N.AdamW(m.params(), lr=3e-3, wd=0.0)
        rngd = np.random.default_rng(1)
        t0 = time.time()
        for st in range(steps):
            xb, yb = spec.gen(16, rngd)
            lg = m.forward(xb); loss = N.cross_entropy(N.last(lg), yb)
            loss.backward(); opt.step()
        acc = evaluate(m, spec)
        db[f"grid|{task}|M{M}|rho{rho}"] = dict(M=M, rho=rho, steps=steps, acc=acc,
                                                params=m.n_params())
        print(f"  M={M:>3} rho={rho:<7} {steps}步 acc={acc:.4f} params={m.n_params()} "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(db, open(DB, "w"), indent=2)
