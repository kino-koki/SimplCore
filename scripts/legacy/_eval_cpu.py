
import os
import sys, time, torch as rt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplcore.mix as mix
from simplcore.mix.core import SCFunctional
import simplcore.tensor as torch

cfg = mix.mix_150m_text()
model = mix.SimplCoreMix(cfg).train()
opt = mix.SCAdamW(model.parameters(), lr=3e-4)
toks = torch.randint(0, cfg.vocab_size, (4, 128), dtype=torch.long)

for nt in (rt.get_num_threads(), 32, 96):
    rt.set_num_threads(nt)
    # warmup
    l = model(toks)
    l = SCFunctional.cross_entropy(l[:, :-1].reshape(-1, cfg.vocab_size), toks[:, 1:].reshape(-1))
    l.backward()
    opt.step()
    t0 = time.time()
    for _ in range(3):
        l = model(toks)
        l = SCFunctional.cross_entropy(l[:, :-1].reshape(-1, cfg.vocab_size), toks[:, 1:].reshape(-1))
        l.backward()
        opt.step()
    dt = (time.time()-t0)/3
    tokens = 4*128
    print(f"threads={nt:3d}: {dt:6.2f}s/step ({tokens/dt:7.0f} tokens/s)")
