# -*- coding: utf-8 -*-
import os
import sys, io, torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
torch.manual_seed(0)
from sgc_v3 import GatedAssocMixer
D, L, R = 32, 64, 8
for trial in range(3):
    cg = GatedAssocMixer(D, slots=R, chunk=16).eval()
    x = torch.randn(1, L, D)
    with torch.no_grad():
        cg(x)                          # warm up RNG identical for both
        y1 = cg(x)
    x2 = x.clone(); x2[0, L-1] = x2[0, L-1] + 8.0
    with torch.no_grad():
        cg2 = GatedAssocMixer(D, slots=R, chunk=16).eval()
        # same RNG state as cg had? cg used 1 draw for M. replicate:
        torch.manual_seed(0)
        cg3 = GatedAssocMixer(D, slots=R, chunk=16).eval()
        cg(x) # advance
        y2 = None
    d = None
    cg4 = GatedAssocMixer(D, slots=R, chunk=16).eval()
    torch.manual_seed(123)
    with torch.no_grad():
        a = cg4(x)
        torch.manual_seed(123)
        cg5 = GatedAssocMixer(D, slots=R, chunk=16).eval()
        b = cg5(x2)
    diff = (a-b).abs().amax(dim=(0,2))
    ch = (diff > 1e-5).nonzero().flatten().tolist()
    print(f"trial {trial}: changed={ch}")
    for t in ch[:8]:
        print(f"    t={t:2d} |dy|_max={float(diff[t]):.6f}")
