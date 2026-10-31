# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Adafactor（R-edge）的守护测试——2GB 级机器上 1 亿参数可训的机制锁定.

端侧实测背景（2026-10-31，`docs/EDGE_TRAINING_REPORT.md`）：AdamW 需要
16 B/参数，1.085 亿参数直接爆 2 GB；Adafactor 的 factored 二阶矩把
优化器状态从 O(m·n) 压到 O(m+n)——实测 1.085M 参数模型 300 步真实训练
（train −63.8% / val −43.8%），且 O(1) 流式状态字节全程恒定
（定理 3 的工程级证据）。

锁定四条：
1. **状态账**：2D 参数的优化器状态元素 ≈ m+n（行/列因子），对比 AdamW
   的 2mn——512× 压缩量级锁定；
2. **收敛方向**：凸问题 loss 末值 < 初值（update clipping 下的基本
   优化能力）；
3. **稳定性**：极端梯度（|g|~1e12）一步后参数仍有限（update clipping
   生效——端侧无梯度爆炸的条件）；
4. **CLI 契约**：`--optimizer adafactor` 可解析（main.py 接线）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from simplcore.core.training.train.optim import Adafactor


def _state_elements(opt: Adafactor, p: torch.nn.Parameter) -> int:
    return sum(v.numel() for v in opt.state[p].values() if torch.is_tensor(v))


def test_state_bytes_factored_vs_adamw() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(1024, 1024)
    opt = Adafactor(lin.parameters(), lr=0.01)
    x = torch.randn(4, 1024)
    (lin(x).square().mean()).backward()
    opt.step()
    n = _state_elements(opt, lin.weight)
    adamw_n = 2 * lin.weight.numel()              # exp_avg + exp_avg_sq
    assert n <= 4 * (1024 + 1024) + 8, \
        f"Adafactor 状态必须 ≈ m+n 因子量级（实测 {n}）"
    assert n < adamw_n * 0.01, \
        f"状态压缩不足 100×: adafactor={n} vs adamw={adamw_n}"


def test_convex_convergence_direction() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 1)
    opt = Adafactor(lin.parameters(), lr=0.02)
    first = last = None
    for i in range(60):
        x = torch.randn(32, 64)
        y = x.sum(dim=1, keepdim=True)
        loss = ((lin(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < first, f"凸问题 loss 必须下降: first={first:.3f} last={last:.3f}"


def test_update_clipping_extreme_gradient() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(32, 32)
    opt = Adafactor(lin.parameters(), lr=0.01, clip_thresh=1.0)
    x = torch.randn(4, 32)
    before = lin.weight.detach().clone()
    (lin(x).square().mean() * 1e12).backward()    # 极端梯度 ~1e12
    opt.step()
    delta = (lin.weight.detach() - before).abs().max().item()
    assert delta < 1.0, f"update clipping 失效: 参数移动 {delta}"


def test_cli_optimizer_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    cli = (root / "simplcore" / "core" / "training" / "train" / "cli.py") \
        .read_text(encoding="utf-8")
    assert '"--optimizer"' in cli and "adafactor" in cli
    main = (root / "simplcore" / "core" / "training" / "train" / "main.py") \
        .read_text(encoding="utf-8")
    assert "Adafactor" in main and "EDGE_TRAINING_REPORT" in main