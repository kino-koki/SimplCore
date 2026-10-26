# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""门控消融实验: SGC-v2 vs GatedSGCV3 vs Attention (快速验证模式).

注意: 完整 4000 步实验在 CPU 上需数分钟, 请在本机运行:
    python -m simplcore.architecture.gated_ablation

本脚本默认 steps=400, batch=16 (30秒内可跑完) 用于快速验证门控有效性。
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

from .spectral_global_conv import SpectralGlobalConvV2
from .gated_sgc_v3 import GatedSGCV3


def make_selective_copy_batch(
    batch_size: int, seq_len: int, vocab_size: int, n_copy: int,
    rng: np.random.RandomState,
) -> tuple:
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


def build_attention_model(dim: int, n_heads: int = 4) -> nn.Module:
    class CausalSelfAttn(nn.Module):
        def __init__(self, dim: int, n_heads: int):
            super().__init__()
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def train_and_eval(model, x, y, steps, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    final_loss = None
    for _ in range(steps):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = float(loss.data)
    elapsed = time.time() - start
    pred_np = pred.data
    y_np = y.data
    pred_idx = np.argmax(pred_np, axis=-1)
    y_idx = np.argmax(y_np, axis=-1)
    acc = np.mean(pred_idx == y_idx) * 100.0
    return {"acc": acc, "loss": final_loss, "time": elapsed}


if __name__ == "__main__":
    torch.manual_seed(42)
    rng = np.random.RandomState(42)

    L, D, vocab = 48, 64, 64
    k = 6
    batch = 16
    steps = 400

    print("=" * 70)
    print("门控消融实验 (快速400步): SGC-v2 vs GatedSGCV3 vs Attention")
    print("=" * 70)
    print(f"配置: L={L}, D={D}, vocab={vocab}, k={k}, batch={batch}, steps={steps}")
    print()

    x_np, y_np = make_selective_copy_batch(batch, L, vocab, k, rng)
    x = torch.tensor(x_np, requires_grad=False)
    y = torch.tensor(y_np, requires_grad=False)

    results = {}
    models = {
        "SGC-v2": lambda: SpectralGlobalConvV2(dim=D, max_len=L, rank=8),
        "GatedSGCV3": lambda: GatedSGCV3(dim=D, max_len=L, rank=8, gate_bias=-1.0, dropout_rate=0.1),
        "Attention": lambda: build_attention_model(D, n_heads=4),
    }

    for name, factory in models.items():
        model = factory()
        print(f"正在训练 {name} ...")
        r = train_and_eval(model, x, y, steps, lr=0.001)
        results[name] = r
        print(f"  acc={r['acc']:.2f}%  loss={r['loss']:.4f}  耗时={r['time']:.1f}s")
        print()

    print("=" * 70)
    print("汇总对比:")
    print(f"{'模型':<14} {'acc(%)':>8} {'loss':>8} {'耗时(s)':>8}")
    print("-" * 42)
    for name, r in results.items():
        print(f"{name:<14} {r['acc']:>8.2f} {r['loss']:>8.4f} {r['time']:>8.1f}")
    print()

    sgc = results["SGC-v2"]
    gated = results["GatedSGCV3"]
    attn = results["Attention"]
    print(f"SGC-v2 vs Attention: accΔ={sgc['acc']-attn['acc']:+.2f}pp, lossΔ={sgc['loss']-attn['loss']:+.4f}")
    print(f"GatedSGCV3 vs Attention: accΔ={gated['acc']-attn['acc']:+.2f}pp, lossΔ={gated['loss']-attn['loss']:+.4f}")
    print(f"GatedSGCV3 vs SGC-v2: accΔ={gated['acc']-sgc['acc']:+.2f}pp, lossΔ={gated['loss']-sgc['loss']:+.4f}")
    print("=" * 70)
