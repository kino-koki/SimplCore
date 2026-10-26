# -*- coding: utf-8 -*-
import os, sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "verify_sgc")))
from sgc_pytorch import SpectralGlobalConvV2, AttentionMixer
torch.manual_seed(0)

def explicit_causal(x, Kern, L):
    B = x.shape[0]
    y = torch.zeros_like(x)
    K = Kern.shape[0]
    for t in range(L):
        lo = max(0, t - K + 1)
        for s in range(lo, t + 1):
            y[:, t, :] += x[:, s, :] * Kern[t - s].unsqueeze(0)
    return y

print("2) L > max_len: what does it actually compute?")
D, R = 16, 4
m = SpectralGlobalConvV2(dim=D, max_len=16, rank=R)
with torch.no_grad():
    m.low.normal_(0, 0.5); m.up.normal_(0, 0.5)
x = torch.randn(1, 48, D)
with torch.no_grad():
    y = m._causal_conv(x)
print("   ran fine, out shape =", tuple(y.shape))
full = m.low[:48, :] @ m.up          # (16,16) because low has only 16 rows
print("   effective kernel taps =", full.shape[0], " requested L =", 48)
ref = explicit_causal(x, full, L=48)
print("   max|model - conv(truncated 16-tap kernel)| = %.3e" % (y - ref).abs().max().item())
print("   -> model is a FIXED-WINDOW conv of width max_len, NOT a global conv over L.")
print("      It does not error or warn: max_len silently caps the receptive field.")
print("      Also y[:, :L] slices a length-2L buffer, so masking is implicit but undocumented.")

print()
print("4) rank bottleneck (random init, as at end of training)")
m = SpectralGlobalConvV2(dim=64, max_len=64, rank=8)
with torch.no_grad():
    m.low.normal_(0, 0.5); m.up.normal_(0, 0.5)
kern = m.low[:64, :] @ m.up
print("   K shape (L,D) =", tuple(kern.shape), " rank(K^T) =", torch.linalg.matrix_rank(kern.T).item())
for r in (1, 2, 4, 8, 16, 32, 64):
    mm = SpectralGlobalConvV2(dim=64, max_len=64, rank=r)
    with torch.no_grad():
        mm.low.normal_(0, 0.5); mm.up.normal_(0, 0.5)
    k = mm.low[:64, :] @ mm.up
    print(f"   rank param {r:3d} -> rank(K^T) = {torch.linalg.matrix_rank(k.T).item():3d}, params = {sum(p.numel() for p in mm.parameters()):6d}")

print()
print("6) kernels are NOT causal-checked for in_proj/out_proj params; test trainability of low rows>0")
m = SpectralGlobalConvV2(dim=D, max_len=32, rank=R)
x = torch.randn(2, 32, D, requires_grad=True)
loss = (m(x) ** 2).mean(); loss.backward()
print("   low.grad nonzero rows:", int((m.low.grad.abs().sum(-1) > 0).sum()), "/", m.low.shape[0])
print("   up.grad norm = %.4e   out_scale.grad = %.4e" % (m.up.grad.norm().item(), m.out_scale.grad.norm().item()))

print()
print("7) FFT conv cost vs attention at A100-relevant sizes (CPU, relative)")
for L in (512, 1024, 2048):
    dm = 256
    sgc = SpectralGlobalConvV2(dim=dm, max_len=L, rank=8).eval()
    att = AttentionMixer(dim=dm, n_heads=4, max_len=L).eval()
    xx = torch.randn(1, L, dm)
    import time
    with torch.no_grad():
        for _ in range(2): sgc(xx); att(xx)
        t0 = time.perf_counter()
        for _ in range(5): sgc(xx)
        ts = (time.perf_counter() - t0) / 5
        t0 = time.perf_counter()
        for _ in range(5): att(xx)
        ta = (time.perf_counter() - t0) / 5
    print(f"   L={L:5d}  sgc={ts*1000:8.2f} ms  attn={ta*1000:8.2f} ms  speedup={ta/ts:5.2f}x")
