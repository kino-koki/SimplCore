
import os
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplcore.mix as mix
from simplcore.mix.core import SCFunctional
import simplcore.tensor as torch
import torch as rt
print("torch threads:", rt.get_num_threads())
cfg = mix.mix_150m_text()
model = mix.SimplCoreMix(cfg).train()
opt = mix.SCAdamW(model.parameters(), lr=3e-4)
toks = torch.randint(0, cfg.vocab_size, (4, 128), dtype=torch.long)
l = model(toks); l = SCFunctional.cross_entropy(l[:, :-1].reshape(-1, cfg.vocab_size), toks[:, 1:].reshape(-1)); l.backward(); opt.step()  # warmup
t0 = time.time()
l = model(toks); l = SCFunctional.cross_entropy(l[:, :-1].reshape(-1, cfg.vocab_size), toks[:, 1:].reshape(-1)); l.backward(); opt.step()
dt = time.time() - t0
print(f"CPU 150m: {dt:.2f}s/step (512 tokens) = {512/dt:.0f} tokens/s")
