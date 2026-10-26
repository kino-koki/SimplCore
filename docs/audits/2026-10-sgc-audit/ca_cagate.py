# -*- coding: utf-8 -*-
import os
import sys, io, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
torch.set_num_threads(4); torch.manual_seed(0)
from sgc_v3 import ContentAdaptiveMixer, GatedAssocMixer
D, L, R = 32, 64, 8

def first_changed(m, x):
    x2 = x.clone(); x2[0, L-1] += 8.0
    with torch.no_grad(): y1, y2 = m(x), m(x2)
    d = (y1-y2).abs().amax(dim=(0,2))          # (L,)
    return d

for name, m in [("ca", ContentAdaptiveMixer(D, feat=R, chunk=16)),
                ("cagate", GatedAssocMixer(D, slots=R, chunk=16))]:
    m.eval(); x = torch.randn(1, L, D)
    d = first_changed(m, x)
    ch = (d > 1e-6).nonzero().flatten().tolist()
    print(f"{name}: n_changed={len(ch)} changed={ch}")
    print("   |d| shape", tuple(d.shape), "max", float(d.max()))

print("\n=== ca: is the output at t<63 invariant to ALL changes of x[63]? and does it even vary with t? ===")
ca = ContentAdaptiveMixer(D, feat=R, chunk=16).eval()
x = torch.randn(1, L, D)
with torch.no_grad(): y = ca(x)
print("  y norm per position (first 8):", [round(float(y[0,t].norm()),3) for t in range(8)])
x3 = x.clone(); x3[0, L-1] = torch.randn(D) * 50
with torch.no_grad(): y3 = ca(x3)
dd = (y[0,:63] - y3[0,:63]).abs().max()
print(f"  max|y[0:63] change| when x[63] replaced by 50x noise = {float(dd):.3e}  (0 => ca truly causal here)")

print("\n=== cagate: manual recurrence check at t=0 and t=1 (state should be zero-initialised before t=0) ===")
cg = GatedAssocMixer(D, slots=R, chunk=16).eval()
print("  NOTE: GatedAssocMixer.forward_stream initialises M with torch.randn (line 190) => output depends on RNG at call time")
with torch.no_grad():
    yA = cg(x)
    yB = cg(x)
print("  two identical calls, same input: max|yA-yB| =", float((yA-yB).abs().max()), "(!=0 => non-deterministic)")
