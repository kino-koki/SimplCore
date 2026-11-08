"""批量跑：python run_batch.py <steps> <seed> <task1> [task2 ...]
结果累积写入 results_all.json（已跑过的 (task,seed) 会跳过）。
"""
import sys, os, json, time
import numpy as np
import nn as N
import exp_probe as P

SGC_CFG = dict(D=48, M=32, K=4, n_layers=2)
TS_CFG = dict(D=56, n_heads=4, n_layers=2)
NSAMP = 512
DB = "results_all.json"


def load():
    return json.load(open(DB)) if os.path.exists(DB) else {}


def save(db):
    json.dump(db, open(DB, "w"), indent=2)


def run(name, seed, steps, kinds=("sgc","ts")):
    spec = P.TASKS[name](np.random.default_rng(abs(hash(name)) % 10000))
    row = dict(seed=seed, chance=spec.chance, V=int(spec.V),
               L=int(spec.L), steps=steps)
    for kind in kinds:
        rng = np.random.default_rng(1000 + seed)
        model = (N.SGCNet(V=spec.V, rng=rng, **SGC_CFG) if kind == "sgc"
                 else N.TSNet(V=spec.V, rng=rng, **TS_CFG))
        t0 = time.time()
        r = P.train_eval(model, spec, steps=steps, bs=16,
                         rng=np.random.default_rng(seed * 7 + 1))
        row[kind] = float(r["ppl"] if "ppl" in r else r["acc"])
        row[kind + "_params"] = model.n_params()
        row[kind + "_sec"] = round(time.time() - t0, 1)
        row[kind + "_loss"] = r["hist"][-1][1]
        row[kind + "_correct"] = r.get("correct")
    if "sgc" in row and "ts" in row and "sgc_params" in row and "ts_params" in row:
        row["param_ratio"] = row["sgc_params"] / row["ts_params"]
    return row


if __name__ == "__main__":
    steps = int(sys.argv[1])
    seed = int(sys.argv[2])
    kinds = sys.argv[3].split(",")          # "sgc" / "ts" / "sgc,ts"
    tasks = sys.argv[4:]
    db = load()
    t0 = time.time()
    for name in tasks:
        key = f"{name}|{seed}"
        row = db.get(key)
        need = [k for k in kinds if row is None or k not in row]
        if not need:
            print(f"skip {key} ({','.join(kinds)} done)", flush=True)
            continue
        r2 = run(name, seed, steps, need)
        row = dict(row or {}); row.update(r2)
        # 补齐每个 kind 的公共字段
        row.setdefault("seed", seed); row["steps"] = steps
        db[key] = row
        save(db)
        print(f"[{name} seed={seed} {steps}步] SGC={row['sgc']:.4f} TS={row['ts']:.4f} "
              f"chance={row['chance']} ratio={row['param_ratio']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("total %.0fs" % (time.time() - t0))
