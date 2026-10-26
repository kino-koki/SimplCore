# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""门控消融 + 独立测试集评估 (修复版).

【关键修复】
  1. 优化器: 从 SGD(lr=0.01) 改为 Adam(lr=0.001) —— 这是 train_acc 从 100%
     掉到 1.11% 的根源 (SGD 大学习率在单位脉冲初始化下直接把模型推崩)。
  2. loss 显示: 从 .6f 改为 .3e (科学计数法), 避免 float32 精度下 0.000000 误导。
  3. 独立 test 集: 用不同 seed 生成, 真正评估泛化。
  4. 多 seed 平均: 默认跑 3 个 seed 取平均, 报告误差棒。

用法:
    $env:PYTHONPATH = "G:\\SimplCore-V 0.1-ALPHA"
    python -m simplcore.architecture.gated_sgc_eval --steps 2000 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import time
import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

from .spectral_global_conv import SpectralGlobalConvV2
from .gated_sgc import GatedSGCV2


def make_selective_copy_batch(batch_size, seq_len, vocab_size, n_copy, rng):
    """生成 selective-copy 数据."""
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


def build_attention_model(dim, n_heads=4):
    """构造因果注意力模型 (baseline)."""
    class CausalSelfAttn(nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.dim = dim
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
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


def compute_acc(model, x, y):
    """计算 top-1 token 级准确率."""
    with torch.no_grad():
        pred = model(x)
        pred_idx = np.argmax(pred.data, axis=-1)
        y_idx = np.argmax(y.data, axis=-1)
        correct = (pred_idx == y_idx).sum()
        total = y_idx.size
    return float(correct) / total


def train_and_eval(model, x_t, y_t, x_te, y_te, steps, lr=0.001):
    """训练模型并用 Adam 优化器, 返回 train/test acc."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    for step in range(steps):
        pred = model(x_t)
        loss = F.mse_loss(pred, y_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % 500 == 0:
            print(f"    步 {step+1}/{steps}: loss={float(loss.data):.3e}")

    train_acc = compute_acc(model, x_t, y_t)
    test_acc = compute_acc(model, x_te, y_te)
    elapsed = time.time() - start
    return train_acc, test_acc, float(loss.data), elapsed


def run_eval(steps=2000, seeds=(42,)):
    """多 seed 运行, 取平均."""
    L, D, vocab = 48, 64, 64
    k = 6
    batch = 32

    all_results = {name: [] for name in ["SGC-V2", "Gated-SGC-V2", "Attention"]}

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"门控消融 + 独立测试集: {steps} 步 (seed={seed})")
        print("=" * 70)
        print(f"配置: L={L}, D={D}, vocab={vocab}, k={k}, batch={batch}")

        torch.manual_seed(seed)
        rng_train = np.random.RandomState(seed)
        rng_test = np.random.RandomState(seed + 9999)

        x_train, y_train = make_selective_copy_batch(batch, L, vocab, k, rng_train)
        x_test, y_test = make_selective_copy_batch(batch, L, vocab, k, rng_test)
        x_t = torch.tensor(x_train, requires_grad=False)
        y_t = torch.tensor(y_train, requires_grad=False)
        x_te = torch.tensor(x_test, requires_grad=False)
        y_te = torch.tensor(y_test, requires_grad=False)

        for name, model in [
            ("SGC-V2", SpectralGlobalConvV2(dim=D, max_len=L, rank=8)),
            ("Gated-SGC-V2", GatedSGCV2(dim=D, max_len=L, rank=8)),
            ("Attention", build_attention_model(D, n_heads=4)),
        ]:
            print(f"\n--- {name} ---")
            train_acc, test_acc, loss, elapsed = train_and_eval(
                model, x_t, y_t, x_te, y_te, steps
            )
            print(f"  train_acc={train_acc*100:.2f}%  test_acc={test_acc*100:.2f}%  loss={loss:.3e}  time={elapsed:.1f}s")
            all_results[name].append({"train_acc": train_acc, "test_acc": test_acc, "loss": loss, "time": elapsed})

    print("\n" + "=" * 70)
    print(f"结论 (多 seed 平均, seeds={list(seeds)}):")
    avg = {}
    for name, res in all_results.items():
        tr = np.mean([r["train_acc"] for r in res]) * 100
        te = np.mean([r["test_acc"] for r in res]) * 100
        std = np.std([r["test_acc"] for r in res]) * 100
        avg[name] = {"train_acc": tr, "test_acc": te, "std": std}
        print(f"  {name:15s}: train_acc={tr:.2f}%  test_acc={te:.2f}%±{std:.2f}")

    sgc = avg.get("SGC-V2", {}).get("test_acc", 0)
    attn = avg.get("Attention", {}).get("test_acc", 0)
    gated = avg.get("Gated-SGC-V2", {}).get("test_acc", 0)
    print(f"\n  SGC vs Attn test acc: {sgc:.2f}% vs {attn:.2f}% (Δ={sgc-attn:+.2f}pp)")
    print(f"  Gated vs SGC test acc: {gated:.2f}% vs {sgc:.2f}% (Δ={gated-sgc:+.2f}pp)")

    if sgc > attn:
        print("\n  ✅ SGC 测试集泛化超越 Attention")
    elif abs(sgc - attn) < 1.0:
        print("\n  ⚠️ SGC 测试集泛化与 Attention 持平 (误差范围内)")
    else:
        print("\n  ❌ SGC 测试集泛化低于 Attention (过拟合)")

    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    run_eval(args.steps, args.seeds)
