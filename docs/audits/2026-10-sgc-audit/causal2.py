# -*- coding: utf-8 -*-
import os
import sys, io, torch, torch.nn.functional as F
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "verify_sgc")
torch.set_num_threads(4); torch.manual_seed(0)
from sgc_pytorch import SpectralGlobalConvV2, AttentionMixer
from sgc_v3 import ContentRoutedSGC, ContentAdaptiveMixer, GatedAssocMixer, SGC_HM, build_model
D, L, R = 32, 64, 8
ms = {"attn": AttentionMixer(D,4,L), "sgc": SpectralGlobalConvV2(D,3,L,R),
      "sgcv3": ContentRoutedSGC(D,L,R), "ca": ContentAdaptiveMixer(D,feat=R,chunk=16),
      "cagate": GatedAssocMixer(D,slots=R,chunk=16), "hm": SGC_HM(D,buffer=8,slots=R,chunk=16)}
print("=== CORRECT causality test: perturb FINAL position x[L-1]; do outputs at t<L-1 change? ===")
for name, m in ms.items():
    m.eval()
    x = torch.randn(1, L, D); x2 = x.clone(); x2[0, L-1] += 8.0
    with torch.no_grad(): y1, y2 = m(x), m(x2)
    d = (y1-y2).abs().amax(dim=(0,2))
    ch = (d > 1e-6).nonzero().flatten().tolist()
    print(f"  {name:7s} changed positions: {ch if len(ch)<12 else str(ch[:12])+f' ... ({len(ch)} total)'}  -> causal={ch==[L-1]}")

print("\n=== positional-embedding dependence (permutation test on pos_emb) ===")
lm = build_model(50, D, 2, 4, L, "sgc")
ids = torch.randint(0, 50, (1, L))
p0 = lm.pos_emb.detach().clone()
with torch.no_grad(): a = lm(ids)
lm.pos_emb.data.normal_(0, 5.0)          # scramble positional embedding
with torch.no_grad(): b = lm(ids)
print("  pos_emb perturbed by N(0,5): max logit change =", float((a-b).abs().max()),
      " (0 => pos_emb is never read)")
print("  pos_emb requires_grad:", p0.requires_grad, " shape:", tuple(p0.shape))
