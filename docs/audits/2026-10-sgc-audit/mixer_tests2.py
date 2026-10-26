# -*- coding: utf-8 -*-
import os
import sys, os, io, torch, torch.nn.functional as F
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
torch.set_num_threads(4); torch.manual_seed(0)
from sgc_pytorch import SpectralGlobalConvV2, AttentionMixer
from sgc_v3 import ContentRoutedSGC, ContentAdaptiveMixer, GatedAssocMixer, SGC_HM, build_model
from sgc_gated import GatedSGC
D, L, R = 32, 64, 8
def pc(m): return sum(p.numel() for p in m.parameters())
ms = {
 "attn":   AttentionMixer(D, 4, L),
 "sgc":    SpectralGlobalConvV2(D, 3, L, R),
 "sgcv3":  ContentRoutedSGC(D, L, R),
 "ca":     ContentAdaptiveMixer(D, feat=R, chunk=16),
 "cagate": GatedAssocMixer(D, slots=R, chunk=16),
 "hm":     SGC_HM(D, buffer=8, slots=R, chunk=16),
 "gated":  GatedSGC(D, L, R, use_gate=True),
}
form = {
 "attn":   2*D*D + 2*D + 1,
 "sgc":    2*D*D + L*R + R*D + 2*D + 1,
 "sgcv3":  3*D*D + L*R + R*D + 4*D + 1,
 "ca":     5*D*D + 2*D*R + 2*D,
 "cagate": 5*D*D + 2*D,
 "hm":     5*D*D + 3*D + 1,
 "gated":  2*D*D + L*R + R*D + 2*D + 1 + D*D + D,
}
print(f"{'mixer':8s} {'actual':>8s} {'formula':>8s} {'diff':>7s}  named-params")
for k, m in ms.items():
    parts = ",".join(f"{n}:{p.numel()}" for n, p in m.named_parameters())
    print(f"{k:8s} {pc(m):8d} {form[k]:8d} {pc(m)-form[k]:7d}  {parts}")

print("\n=== causality: perturb x[0]; which output positions change? ===")
for name, m in ms.items():
    m.eval()
    x = torch.randn(1, L, D); x2 = x.clone(); x2[0,0] += 5.0
    with torch.no_grad(): y1, y2 = m(x), m(x2)
    d = (y1-y2).abs().amax(dim=(0,2))
    ch = (d > 1e-6).nonzero().flatten().tolist()
    print(f"  {name:8s} n_changed={len(ch):3d} first={ch[:6]} last={ch[-1] if ch else None}")

print("\n=== SGC FFT conv: does y[t] depend on u[s] for s>t ? ===")
m = SpectralGlobalConvV2(D, 3, L, R)
with torch.no_grad(): m.low.normal_(0,0.5); m.up.normal_(0,0.5)
u = torch.randn(2, L, D)
with torch.no_grad(): y = m._causal_conv(u)
Kern = m.low[:L,:] @ m.up
# explicit non-causal (code) sum over all s
y_code = torch.zeros_like(u)
for t in range(L):
    acc = torch.zeros(2, D)
    for s in range(L): acc = acc + Kern[(t - s) % L] * u[:, s]
    y_code[:, t] = acc
y_causal = torch.zeros_like(u)
for t in range(L):
    acc = torch.zeros(2, D)
    for s in range(t+1): acc = acc + Kern[t - s] * u[:, s]
    y_causal[:, t] = acc
print("  max|fft - code-sum|  =", float((y-y_code).abs().max()))
print("  max|fft - causal|    =", float((y-y_causal).abs().max()), " (0 => actually causal)")
uf = u.clone(); uf[:, 1:, :] = 0
with torch.no_grad(): yf = m._causal_conv(uf)
print("  pos0: |fft(u) - fft(future-zeroed)| =", float((y[:,0]-yf[:,0]).abs().max()))

print("\n=== CA: chunk invariance + streaming equivalence ===")
xx = torch.randn(2, L, D); prev=None
for chunk in (8,16,32,64):
    mm = ContentAdaptiveMixer(D, feat=R, chunk=chunk).eval()
    with torch.no_grad():
        o = mm(xx); st=None; outs=[]
        for s in range(0, L, chunk):
            oo, st = mm.forward_stream(xx[:, s:s+chunk], st); outs.append(oo)
        streamed = torch.cat(outs,1)
    line = f"  chunk={chunk:3d} |full-stream|={float((o-streamed).abs().max()):.2e}"
    if prev is not None: line += f"  vs prev chunk: {float((o-prev).abs().max()):.3e}"
    print(line); prev = o

print("\n=== HM: input longer than construction max_len ===")
MAXLEN = 2*128 + 1 + 1024
big = build_model(64, D, 1, 4, MAXLEN, "hm", ca_feat=64, ca_chunk=64).eval()
for Lq in (600, 769, 800):
    ids = torch.randint(0, 64, (1, Lq))
    try:
        with torch.no_grad(): out = big(ids)
        print(f"  L={Lq:5d} OK shape={tuple(out.shape)}")
    except Exception as e:
        print(f"  L={Lq:5d} FAIL {type(e).__name__}: {e}")
