# -*- coding: utf-8 -*-
"""Validate the SGC position-path design: damped-oscillation basis
   (a) closed form vs 2nd-order IIR recurrence
   (b) FFT causal conv (parallel/training form) vs IIR streaming form
   (c) constant streaming state size, independent of L
"""
import torch, math
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)

# ---------- (a) basis: closed form vs recurrence ----------
M = 5
L = 64
rho = torch.tensor([0.5, 0.7, 0.85, 0.93, 0.98])
theta = torch.tensor([0.3, 0.8, 1.4, 2.1, 2.9])

def closed_form(rho, theta, L):
    tau = torch.arange(L, dtype=torch.float32)
    r = rho.unsqueeze(-1) ** tau                 # (M,L)
    ang = theta.unsqueeze(-1) * (tau + 1)        # (M,L)
    return r * torch.sin(ang) / torch.sin(theta).unsqueeze(-1)

def recurrence(rho, theta, L):
    a = 2 * rho * torch.cos(theta)
    b = -(rho ** 2)
    h = torch.zeros(M, L)
    y1 = torch.zeros(M); y2 = torch.zeros(M)
    for t in range(L):
        y = a * y1 + b * y2
        y = y + (1.0 if t == 0 else 0.0)          # unit impulse input
        h[:, t] = y
        y2, y1 = y1, y
    return h

h_cf = closed_form(rho, theta, L)
h_rc = recurrence(rho, theta, L)
print("(a) closed form vs IIR recurrence, max abs err = %.3e" % (h_cf - h_rc).abs().max().item())
print("    h_m[0] should be 1.0 ->", h_cf[:, 0])

# ---------- (b) FFT conv vs IIR streaming, same impulse response ----------
B, D = 2, 8
x = torch.randn(B, L, D)
C = torch.randn(M, D) * 0.3                   # basis -> channel coefficients
w = torch.randn(D) * 0.2 + 1.0                # direct feedthrough

# parallel / FFT form
K = (h_cf.t() @ C)                            # (L, D) kernel bank
N = 2 * L
Xp = F.pad(x, (0, 0, 0, N - L))
Kp = F.pad(K, (0, 0, 0, L))
cX = torch.fft.rfft(Xp, n=N, dim=1)
cK = torch.fft.rfft(Kp.unsqueeze(0), n=N, dim=1)
y_fft = torch.fft.irfft(cX * cK, n=N, dim=1)[:, :L, :] + w * x

# streaming / IIR form
a = 2 * rho * torch.cos(theta); b = -(rho ** 2)
y1 = torch.zeros(B, M, D); y2 = torch.zeros(B, M, D)
y_iir = torch.zeros(B, L, D)
for t in range(L):
    yy = a.view(1, M, 1) * y1 + b.view(1, M, 1) * y2 + x[:, t].unsqueeze(1)
    y2, y1 = y1, yy
    y_iir[:, t] = torch.einsum('bmd,md->bd', yy, C) + w * x[:, t]

err = (y_fft - y_iir).abs().max().item()
print("(b) FFT(parallel) vs IIR(streaming), max abs err = %.3e  -> %s" % (err, "MATCH" if err < 1e-4 else "MISMATCH"))

# causality check
x2 = x.clone(); x2[:, L//2:, :] = torch.randn(B, L - L//2, D) * 50
with torch.no_grad():
    Xp2 = F.pad(x2, (0, 0, 0, N - L))
    y2f = torch.fft.irfft(torch.fft.rfft(Xp2, n=N, dim=1) * cK, n=N, dim=1)[:, :L, :] + w * x2
d = (y_fft[:, :L//2] - y2f[:, :L//2]).abs().max().item()
print("(c) causal? perturbing future changes past by %.3e -> %s" % (d, "CAUSAL" if d < 1e-4 else "LEAKS"))

# ---------- (d) streaming state size independent of L ----------
print("(d) streaming state per layer = 2*M*D floats = %d  (M=%d, D=%d); independent of L" % (2*M*D, M, D))
for LL in (256, 4096, 65536):
    print("     L=%-6d  IIR state bytes (fp32, B=1) = %d" % (LL, 2*M*D*4))
