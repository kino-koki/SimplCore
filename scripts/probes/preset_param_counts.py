# -*- coding: utf-8 -*-
"""Analytic parameter accounting for the SGC presets (no model instantiation)."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from simplcore.sgc.config import SGC_PRESETS, SGCConfig

def count(cfg: SGCConfig):
    d, L, V = cfg.dim, cfg.num_layers, cfg.vocab_size
    h = cfg.hidden
    emb = V * d
    per = 0
    if cfg.mixer == "attn":
        per += 4 * d * d                      # qkv + out_proj
    else:
        per += 2 * d * d                      # in_proj + out_proj
        if cfg.content_gate and cfg.mixer != "sgc-nogate":
            per += d * d                      # gate_proj
        if cfg.out_gate:
            per += d * d                      # out_gate_proj
        per += cfg.num_bases * d              # oscillator coefficients
        per += cfg.num_bases * 2              # raw_rho, raw_theta
        per += d                              # feed
        k = 1 if cfg.mixer == "sgc-nolocal" else cfg.local_kernel
        per += d * k                          # depthwise local conv
        per += 1                              # out_scale
    per += 3 * d * h                          # SwiGLU gate/up/down
    per += 2 * d                              # norm weights (mixer + ffn)
    total = emb + L * per
    return total, total - (0 if cfg.tie_embeddings else emb), per

print(f"{'preset':12s} {'params':>13s} {'non-emb':>13s} {'per-layer':>11s}  {'bf16 wts':>9s}  mixer")
for name, cfg in SGC_PRESETS.items():
    tot, nonemb, per = count(cfg)
    print(f"{name:12s} {tot:13,d} {nonemb:13,d} {per:11,d}  {tot*2/1e9:8.2f}G  {cfg.mixer}")
