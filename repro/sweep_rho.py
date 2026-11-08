"""判别性实验：MQAR 差距的根因是【感受野】还是【绑定能力】？

做法：只改 SGC 的最长记忆基 rho_max（记忆窗口 tau = -1/ln(rho)），
      其它全部不变，看 MQAR 是否随之改善。
  - 若感受野是瓶颈：rho 0.999(tau~1k) -> 0.9999(tau~10k) 应当明显改善
  - 若绑定能力是瓶颈：改 rho 无效（序列只有 65 tokens，远短于 tau）

python sweep_rho.py <rho_max> <steps> <seed>
"""
import sys, json, time
import numpy as np
import nn as N
import sgc_core as S
import exp_probe as P

STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 700
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
RHO = float(sys.argv[1])

spec = P.TASKS["mqar"](np.random.default_rng(abs(hash("mqar")) % 10000))
grid = S.make_grid(M=32, rho_min=0.9, rho_max=RHO, pairing="lowfreq_long")
model = N.SGCNet(V=spec.V, D=48, M=32, K=4, n_layers=2,
                 rng=np.random.default_rng(1000 + SEED), grid=grid)
opt = N.AdamW(model.params(), lr=3e-3, wd=0.0)
rng = np.random.default_rng(SEED * 7 + 1)
t0 = time.time()
for st in range(STEPS):
    xb, yb = spec.gen(16, rng)
    lg = model.forward(xb)
    loss = N.cross_entropy(N.last(lg), yb)
    loss.backward()
    opt.step()
# 评估
Xte, Yte = spec.gen(512, np.random.default_rng(12345))
preds = []
for s in range(0, 512, 16):
    preds.append(model.forward(Xte[s:s + 16]).v[:, -1, :8].argmax(axis=-1))
preds = np.concatenate(preds)
acc = float((preds == Yte).mean())
tau = -1.0 / np.log(RHO)
print(f"rho_max={RHO}  tau={tau:.0f} tokens  steps={STEPS}  seed={SEED}  "
      f"MQAR acc={acc:.4f} (chance=0.125)  loss={float(loss.v):.3f}  "
      f"({time.time()-t0:.0f}s)", flush=True)
db = "results_rho.json"
try:
    d = json.load(open(db))
except Exception:
    d = {}
d[f"rho{RHO}|{STEPS}|{SEED}"] = dict(rho=RHO, tau=tau, steps=STEPS, seed=SEED,
                                     acc=acc, loss=float(loss.v))
json.dump(d, open(db, "w"), indent=2)
