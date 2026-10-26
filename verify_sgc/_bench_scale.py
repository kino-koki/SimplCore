# -*- coding: utf-8 -*-
"""扫块大小 C 与序列长度 L: 找 CA 反超 attention 的交叉点"""
import torch, time
from sgc_v3 import build_model

torch.manual_seed(0)
D, r, B = 176, 32, 8
vocab = 1084

def bench(m, L, iters=12):
    tok = torch.randint(0, vocab, (B, L))
    for _ in range(2):
        logits = m(tok); logits.sum().backward()
    t0 = time.perf_counter()
    for _ in range(iters):
        logits = m(tok); logits.sum().backward()
    return (time.perf_counter() - t0) / iters

attn = build_model(vocab, D, 4, 4, 2048, 'attn', dropout=0.05)
print("== block size sweep @ L=256 (bwd) ==")
for C in (16, 32, 64, 128, 256):
    ca = build_model(vocab, D, 4, 4, 256, 'ca', ca_feat=r, ca_chunk=C, dropout=0.05)
    t = bench(ca, 256)
    print(f"  C={C:4d}: {t*1000:6.0f} ms")
print("== length scaling CA(C=64) vs attn ==")
ca64 = build_model(vocab, D, 4, 4, 2048, 'ca', ca_feat=r, ca_chunk=64, dropout=0.05)
for L in (256, 512, 1024, 2048):
    t_ca = bench(ca64, L)
    t_at = bench(attn, L)
    print(f"  L={L:5d}: CA={t_ca*1000:6.0f}ms  Attn={t_at*1000:6.0f}ms  比值(CA/Attn)={t_ca/t_at:.3f}")