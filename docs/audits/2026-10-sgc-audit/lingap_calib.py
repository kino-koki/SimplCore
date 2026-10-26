# -*- coding: utf-8 -*-
import os
import sys, io, time, torch, torch.nn.functional as F, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
from sgc_v3 import build_model, count_params
from _train_lingap import make_batch, last_recall, V, VOCAB
torch.set_num_threads(4)

def run(mixer, G, steps, dim=96, layers=2, bs=32, feat=64, seed=7, chunk=64, tag=""):
    torch.manual_seed(seed)
    m = build_model(VOCAB, dim, layers, 4, 2*128+1+1024, mixer, ca_feat=feat, ca_chunk=chunk).train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    rng = random.Random(seed)
    t0 = time.perf_counter(); losses = []
    for step in range(1, steps+1):
        opt.zero_grad()
        inp, tgt, tv, L = make_batch(rng, G, bs)
        logits = m(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward(); opt.step()
        if step % max(1, steps//8) == 0 or step == 1:
            with torch.no_grad():
                m.eval(); i2, t2, v2, _ = make_batch(rng, G, bs)
                r = last_recall(m(i2).detach(), v2); m.train()
            losses.append((step, round(loss.item(),3), r))
    dt = time.perf_counter()-t0
    print(f"  {mixer:5s} G={G:4d} steps={steps} wall={dt:.1f}s  (seqlen={G+3})")
    print("     step/loss/recall:", losses)
    print(f"     random-baseline loss = ln(V) = {torch.log(torch.tensor(float(V))).item():.3f}, chance={1/V:.4f}")
    return dt

print("=== calibration: short runs to see whether the task is learnable at all at G=64 / G=512 ===")
run("hm", 64, 60)
run("ca", 64, 60)
run("attn", 64, 60)
print()
run("attn", 512, 20)
