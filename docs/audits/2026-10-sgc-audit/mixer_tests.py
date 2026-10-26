# -*- coding: utf-8 -*-
import os
import sys, os, io, torch, torch.nn.functional as F
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
torch.set_num_threads(4)
torch.manual_seed(0)
from sgc_pytorch import SpectralGlobalConvV2, AttentionMixer, FFN
from sgc_v3 import ContentRoutedSGC, ContentAdaptiveMixer, GatedAssocMixer, SGC_HM, build_model, count_params
from sgc_gated import GatedSGC

D, L, R = 32, 64, 8
def pc(m): return sum(p.numel() for p in m.parameters())

mixers = {
 "attn":   (AttentionMixer(D, 4, L), 4*2*D*D + 2*D + 1),
 "sgc":    (SpectralGlobalConvV2(D, 3, L, R), 2*D*D + L*R + R*D + 2*D + 1),
 "sgcv3":  (ContentRoutedSGC(D, L, R), 3*D*D + L*R + R*D + 4*D + 1),
 "ca":     (ContentAdaptiveMixer(D, feat=R, chunk=16), 5*D*D + 2*D*R + 2*D),
 "cagate": (GatedAssocMixer(D, slots=R, chunk=16), 5*D*D + 2*D),
 "hm":     (SGC_HM(D, buffer=8, slots=R, chunk=16), 5*D*D + 3*D + 1),
 "gated":  (GatedSGC(D, L, R, use_gate=True), 2*D*D + L*R + R*D + 2*D + 1 + D*D + D),
}
print(f"{'mixer':8s} {'actual':>9s} {'formula':>9s} {'match':>6s}")
for k,(m,f) in mixers.items():
    print(f"{k:8s} {pc(m):9d} {f:9d} {str(pc(m)==f):>6s}")

# ---- causality probe: change x[t0], see if output changes at positions < t0 ----
print("\n=== causality probe (D=32, L=64, x[0] modified; report positions whose output changed) ===")
for name, m in mixers.items():
    m.eval()
    x = torch.randn(1, L, D)
    x2 = x.clone(); x2[0, 0] = x2[0, 0] + 5.0
    with torch.no_grad():
        y1, y2 = m(x), m(x2)
    d = (y1 - y2).abs().amax(dim=(0, 2))          # (L,)
    changed = (d > 1e-6).nonzero().flatten().tolist()
    print(f"  {name:8s} n_changed={len(changed):3d}  first={changed[:8]}  last={changed[-1] if changed else None}")

# ---- FFT causal conv vs explicit causal sum ----
print("\n=== FFT _causal_conv vs explicit y[t]=sum_{s<=t}K[t-s]u[s] (implementation actually computes y[t]=sum_{s=0}^{L-1}K[t-s]u[s]) ===")
m = SpectralGlobalConvV2(D, 3, L, R)
with torch.no_grad():
    m.low.normal_(0, 0.5); m.up.normal_(0, 0.5)
u = torch.randn(2, L, D)
Kern = m.low[:L, :] @ m.up
y_fft = m._causal_conv(u)
# explicit "kernel taps at every offset" (what the code computes):
y_full = torch.zeros_like(u)
for t in range(L):
    acc = torch.zeros(u.shape[0], D)
    for s in range(L):
        acc = acc + Kern[(t - s) % L] * u[:, s]
    y_full[:, t] = acc
# truly causal:
y_causal = torch.zeros_like(u)
for t in range(L):
    acc = torch.zeros(u.shape[0], D)
    for s in range(t + 1):
        acc = acc + Kern[t - s] * u[:, s]
    y_causal[:, t] = acc
print("  max|fft - full-kernel|   =", float((y_fft - y_full).abs().max()))
print("  max|fft - truly-causal|  =", float((y_fft - y_causal).abs().max()))
print("  scale: max|y|=", float(y_fft.abs().max()))

# ---- does position t depend on future u? zero out future inputs and compare ----
u_fut = u.clone(); u_fut[:, 1:, :] = 0.0
with torch.no_grad():
    y_fut = m._causal_conv(u_fut)
print("  max|y_fft(u)[0] - y_fft(u with all future zeroed)[0]| =", float((y_fft[:,0,:] - y_fut[:,0,:]).abs().max()))

# ---- CA: forward_stream(chunked) vs monolithic ----
print("\n=== ContentAdaptiveMixer: full forward vs per-chunk streaming equivalence ===")
for chunk in (8, 16, 64):
    mm = ContentAdaptiveMixer(D, feat=R, chunk=chunk).eval()
    x = torch.randn(2, L, D)
    with torch.no_grad():
        a = mm(x)
        st = None; outs = []
        for s in range(0, L, chunk):
            o, st = mm.forward_stream(x[:, s:s+chunk], st); outs.append(o)
        b = torch.cat(outs, 1)
    print(f"  chunk={chunk:3d} max|full-stream| = {float((a-b).abs().max()):.3e}")

# ---- CA: does chunk size change the result? (masking must not depend on chunk) ----
print("\n=== ContentAdaptiveMixer: is output invariant to internal chunk size? ===")
xx = torch.randn(2, L, D)
prev = None
for chunk in (8, 16, 32, 64):
    mm = ContentAdaptiveMixer(D, feat=R, chunk=chunk).eval()
    with torch.no_grad(): o = mm(xx)
    if prev is not None:
        print(f"  chunk={chunk:3d} vs prev: maxdiff={float((o-prev).abs().max()):.3e}")
    prev = o

# ---- HM: sequence longer than max_len used at construction ----
print("\n=== SGC_HM with L longer than construction max_len (window truncation?) ===")
from sgc_v3 import build_model
MAXLEN = 2*128 + 1 + 1024
big = build_model(64, D, 1, 4, MAXLEN, "hm", ca_feat=64, ca_chunk=64).eval()
print("  model built with max_len =", MAXLEN)
for Lq in (600, 700, 769, 800):
    ids = torch.randint(0, 64, (1, Lq))
    try:
        with torch.no_grad(): out = big(ids)
        print(f"  L={Lq:5d} OK  out={tuple(out.shape)}")
    except Exception as e:
        print(f"  L={Lq:5d} FAIL {type(e).__name__}: {e}")
