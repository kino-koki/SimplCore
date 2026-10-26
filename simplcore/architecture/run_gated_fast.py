# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd.
# -*- coding: utf-8 -*-
"""门控消融快速验证 (400步)."""
from __future__ import annotations
import time
import numpy as np
import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
from .spectral_global_conv import SpectralGlobalConvV2
from .gated_sgc import GatedSGCV2


def make_batch(batch_size, seq_len, vocab_size, n_copy, rng):
    k = n_copy
    tokens = rng.randint(0, vocab_size, size=(batch_size, seq_len))
    y_tokens = np.zeros_like(tokens)
    y_tokens[:, k:] = tokens[:, :-k]
    x = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
    y = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
    for b in range(batch_size):
        x[b, np.arange(seq_len), tokens[b]] = 1.0
        y[b, np.arange(seq_len), y_tokens[b]] = 1.0
    return x, y


def build_attn(dim, n_heads=4):
    class CausalSelfAttn(nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.dim, self.n_heads, self.head_dim = dim, n_heads, dim // n_heads
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)
        def forward(self, x):
            B, L, D = x.shape
            q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(2, 3)) / np.sqrt(self.head_dim)
            mask = torch.zeros((L, L), dtype=torch.float32)
            for i in range(L):
                for j in range(i + 1, L):
                    mask.data[i, j] = float('-inf')
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, L, D)
            return self.out_proj(out)
    return CausalSelfAttn(dim, n_heads)


def train_eval(model, x, y, steps, lr=0.01):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    t0 = time.time()
    last_loss = None
    for _ in range(steps):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        last_loss = float(loss.data)
        loss.backward()
        opt.step()
        opt.zero_grad()
    elapsed = time.time() - t0
    pred_np = pred.data
    y_np = y.data
    acc = float((np.argmax(pred_np, axis=-1) == np.argmax(y_np, axis=-1)).mean() * 100)
    return {"loss": last_loss, "acc": acc, "time": elapsed}


def main(steps=400):
    torch.manual_seed(42)
    rng = np.random.RandomState(42)
    L, D, vocab, k, batch = 48, 64, 64, 6, 32
    print("=" * 60)
    print(f"门控消融快速验证 ({steps} 步)")
    print("=" * 60)
    x_np, y_np = make_batch(batch, L, vocab, k, rng)
    x = torch.tensor(x_np, requires_grad=False)
    y = torch.tensor(y_np, requires_grad=False)

    attn = build_attn(D, 4)
    r1 = train_eval(attn, x, y, steps)
    print(f"[1/3] Attention:      loss={r1['loss']:.4f} acc={r1['acc']:.2f}% time={r1['time']:.1f}s")

    sgc = SpectralGlobalConvV2(dim=D, max_len=L, rank=8)
    r2 = train_eval(sgc, x, y, steps)
    print(f"[2/3] SGC-V2(原始):   loss={r2['loss']:.4f} acc={r2['acc']:.2f}% time={r2['time']:.1f}s")

    gated = GatedSGCV2(dim=D, max_len=L, rank=8)
    r3 = train_eval(gated, x, y, steps)
    print(f"[3/3] Gated-SGC-V2:   loss={r3['loss']:.4f} acc={r3['acc']:.2f}% time={r3['time']:.1f}s")

    print()
    print(f"Attention acc: {r1['acc']:.2f}%")
    print(f"SGC-V2    acc: {r2['acc']:.2f}%  (vs Attn {r2['acc']-r1['acc']:+.2f}pp)")
    print(f"Gated-SGC acc: {r3['acc']:.2f}%  (vs Attn {r3['acc']-r1['acc']:+.2f}pp, vs SGC {r3['acc']-r2['acc']:+.2f}pp)")
    print()
    if r3["acc"] > r1["acc"]:
        print("✅ 修复版门控超越 Attention!")
    elif r3["acc"] > r2["acc"]:
        print("⚠️ 门控提升了 SGC-V2 但未超越 Attention")
    else:
        print("❌ 门控未提升")
    print("=" * 60)

if __name__ == "__main__":
    main(steps=400)
