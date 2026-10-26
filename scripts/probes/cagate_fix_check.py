# -*- coding: utf-8 -*-
"""Verify GatedAssocMixer is now deterministic AND causal."""
import sys, os, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verify_sgc")))
from sgc_v3 import GatedAssocMixer

torch.manual_seed(0)
D, L = 32, 64
m = GatedAssocMixer(D, slots=16, beta=0.1, chunk=16).eval()
x = torch.randn(2, L, D)

with torch.no_grad():
    a = m(x); b = m(x)
print("determinism: max|m(x) - m(x)| = %.3e  -> %s"
      % ((a - b).abs().max().item(), "DETERMINISTIC" if torch.equal(a, b) else "NON-DETERMINISTIC"))

split = 48
with torch.no_grad():
    y1 = m(x)
    x2 = x.clone(); x2[:, split:, :] = torch.randn(2, L - split, D) * 10
    y2 = m(x2)
d = (y1[:, :split] - y2[:, :split]).abs().max().item()
print("causality  : max|delta at t<%d| = %.3e  -> %s"
      % (split, d, "CAUSAL" if d < 1e-5 else "LEAKS FUTURE"))
