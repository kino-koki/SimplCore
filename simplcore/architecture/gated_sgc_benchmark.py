# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""门控消融实验: 对比 SGC-V2 / Gated-SGC-V2 / Attention.

目的: 验证门控是否能缓解 SGC-V2 的过拟合, 并在长步数下达到或超越 Attention.
配置: selective-copy 探针, L=48, D=64, k=6, batch=32.

【修复1】加入独立 test 集 (与原训练集不同 seed 生成), 评估泛化能力.
【修复2】--seed 参数现在真正生效 (原脚本内部写死 seed=42, 导致改 --seed 无效).
【修复3】loss 用科学计数法显示 (避免 float32 精度下 0.000000 误导).

用法:
    python -m simplcore.architecture.gated_sgc_benchmark --steps 2000 --seed 42
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

from .spectral_global_conv import SpectralGlobalConvV2
from .gated_sgc import GatedSGCV2


# ----------------------------------------------------------------------
# selective-copy 数据生成
# ----------------------------------------------------------------------
def make_selective_copy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    n_copy: int,
    rng: np.random.RandomState,
) -> tuple:
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


# ----------------------------------------------------------------------
# 注意力模型 (baseline)
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# 准确率计算
# ----------------------------------------------------------------------
def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算 token 级准确率 (argmax 匹配)."""
    pred_idx = np.argmax(pred.data, axis=-1)  # (B, L)
    target_idx = np.argmax(target.data, axis=-1)  # (B, L)
    return float(np.mean(pred_idx == target_idx))


# ----------------------------------------------------------------------
# 训练, 并在独立 test 集上评估
# ----------------------------------------------------------------------
def train_long(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    steps: int,
    lr: float = 0.001,
) -> Dict:
    """训练模型 `steps` 步, 返回最终 train/test loss 和 test acc."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    final_train_loss = None
    final_test_loss = None
    final_test_acc = None

    start = time.time()
    for step in range(steps):
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        final_train_loss = float(loss.data)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 500 == 0:
            test_pred = model(x_test)
            test_loss = float(F.mse_loss(test_pred, y_test).data)
            test_acc = compute_accuracy(test_pred, y_test)
            print(f"    步 {step+1}/{steps}: train={final_train_loss:.3e}  test={test_loss:.3e}  test_acc={test_acc*100:.2f}%")

    test_pred = model(x_test)
    final_test_loss = float(F.mse_loss(test_pred, y_test).data)
    final_test_acc = compute_accuracy(test_pred, y_test)
    elapsed = time.time() - start

    return {
        "train_loss": final_train_loss,
        "test_loss": final_test_loss,
        "test_acc": final_test_acc,
        "time": elapsed,
    }


# ----------------------------------------------------------------------
# 主验证
# ----------------------------------------------------------------------
def run_benchmark(seed: int = 42, steps: int = 4000) -> Dict:
    """执行门控消融实验, 带独立 test 集."""
    # 使用传入的 seed (修复: 原脚本内部写死, 导致 --seed 无效)
    torch.manual_seed(seed)
    rng_train = np.random.RandomState(seed)
    rng_test = np.random.RandomState(seed + 1000)  # 独立 test seed

    L, D, vocab = 48, 64, 64
    k = 6
    batch = 32

    print("=" * 70)
    print(f"门控消融实验: {steps} 步 (seed={seed}, 独立 test 集)")
    print("=" * 70)
    print(f"配置: L={L}, D={D}, vocab={vocab}, k={k}, batch={batch}")
    print()

    x_train_np, y_train_np = make_selective_copy_batch(batch, L, vocab, k, rng_train)
    x_train = torch.tensor(x_train_np, requires_grad=False)
    y_train = torch.tensor(y_train_np, requires_grad=False)

    x_test_np, y_test_np = make_selective_copy_batch(batch, L, vocab, k, rng_test)
    x_test = torch.tensor(x_test_np, requires_grad=False)
    y_test = torch.tensor(y_test_np, requires_grad=False)

    results = {}

    # 1) SGC-V2 (无门控)
    print("\n--- SGC-V2 (无门控) ---")
    sgc = SpectralGlobalConvV2(dim=D, max_len=L, rank=8)
    r_sgc = train_long(sgc, x_train, y_train, x_test, y_test, steps, lr=0.001)
    print(f"  train={r_sgc['train_loss']:.3e}  test={r_sgc['test_loss']:.3e}  test_acc={r_sgc['test_acc']*100:.2f}%  time={r_sgc['time']:.1f}s")
    results["sgc_v2"] = r_sgc

    # 2) Gated-SGC-V2 (有门控)
    print("\n--- Gated-SGC-V2 (有门控, 修复版) ---")
    gsgc = GatedSGCV2(dim=D, max_len=L, rank=8)
    r_gsgc = train_long(gsgc, x_train, y_train, x_test, y_test, steps, lr=0.001)
    print(f"  train={r_gsgc['train_loss']:.3e}  test={r_gsgc['test_loss']:.3e}  test_acc={r_gsgc['test_acc']*100:.2f}%  time={r_gsgc['time']:.1f}s")
    results["gated_sgc_v2"] = r_gsgc

    # 3) Attention (基线)
    print("\n--- Attention (基线) ---")
    attn = build_attention_model(D, n_heads=4)
    r_attn = train_long(attn, x_train, y_train, x_test, y_test, steps, lr=0.001)
    print(f"  train={r_attn['train_loss']:.3e}  test={r_attn['test_loss']:.3e}  test_acc={r_attn['test_acc']*100:.2f}%  time={r_attn['time']:.1f}s")
    results["attention"] = r_attn

    # 汇总
    print("\n" + "=" * 70)
    print(f"结论 ({steps} 步, 独立 test 集):")
    print(f"  SGC-V2      : test_acc={r_sgc['test_acc']*100:.2f}%  test_loss={r_sgc['test_loss']:.3e}")
    print(f"  Gated-SGC-V2: test_acc={r_gsgc['test_acc']*100:.2f}%  test_loss={r_gsgc['test_loss']:.3e}")
    print(f"  Attention   : test_acc={r_attn['test_acc']*100:.2f}%  test_loss={r_attn['test_loss']:.3e}")
    print(f"\n  Gated vs SGC   : Δacc={ (r_gsgc['test_acc']-r_sgc['test_acc'])*100:+.2f}pp  Δloss={r_gsgc['test_loss']-r_sgc['test_loss']:+.3e}")
    print(f"  Gated vs Attn : Δacc={ (r_gsgc['test_acc']-r_attn['test_acc'])*100:+.2f}pp  Δloss={r_gsgc['test_loss']-r_attn['test_loss']:+.3e}")
    print(f"  SGC   vs Attn : Δacc={ (r_sgc['test_acc']-r_attn['test_acc'])*100:+.2f}pp  Δloss={r_sgc['test_loss']-r_attn['test_loss']:+.3e}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="门控消融实验")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()
    run_benchmark(seed=args.seed, steps=args.steps)
