# -*- coding: utf-8 -*-
import os
import sys, io, time, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
torch.set_num_threads(4); torch.manual_seed(0)
from sgc_v3 import build_model, count_params
V, D, NL, H, SL = 1084, 176, 4, 4, 256
print("=== build_model param counts at report config (V=1084,D=176,L=4,H=4,seq=256,rank=8) ===")
print(f"{'mixer':8s} {'params':>10s}   (reference values seen in results/*.json)")
ref = {"attn":2221476, "sgc":1987492, "sgcv3":2236712, "ca":2142624}
for mx in ["attn","sgc","sgcv3","ca","cagate"]:
    m = build_model(V, D, NL, H, SL, mx, rank=8, ca_feat=32, ca_chunk=128)
    r = ref.get(mx)
    print(f"{mx:8s} {count_params(m):10d}   json={r}  diff={count_params(m)-(r or 0):+d}")
for sl in (256, 512):
    m = build_model(V, D, NL, H, sl, "hm", ca_feat=32, ca_chunk=max(64,sl))
    print(f"{'hm':8s} {count_params(m):10d}   (seq_len={sl}, buffer=8, slots=32)")

print("\n=== fwd+bwd wall time, one batch (B=16, seq=256, D=176, 4 layers) ===")
import torch.nn.functional as F
for mx in ["attn","sgc","sgcv3","ca","cagate"]:
    torch.manual_seed(0)
    m = build_model(V, D, NL, H, SL, mx, rank=8, ca_feat=32, ca_chunk=256)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, V, (16, SL))
    for i in range(2):
        opt.zero_grad(); l = m(x); F.cross_entropy(l.reshape(-1,V), x.reshape(-1)).backward(); opt.step()
    t0 = time.perf_counter()
    for i in range(5):
        opt.zero_grad(); l = m(x); F.cross_entropy(l.reshape(-1,V), x.reshape(-1)).backward(); opt.step()
    print(f"  {mx:8s} {(time.perf_counter()-t0)/5*1000:8.0f} ms/step")
