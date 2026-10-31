# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC 训练栈可复现性守护测试.

守护 ``simplcore/core/train.py`` 的可复现性契约:
  * 同 seed + 同数据 + 同超参 两次训练必须逐位一致 (bit-exact);
  * 检查点保存/恢复必须保持 RNG 状态, 续训结果与无中断训练一致;
  * 不同 seed 两次训练必须产生不同轨迹 (证明 seed 真的生效)。

预算: 全模块 CPU 上 < 60 s (小模型, 短步数)。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import get_preset                      # noqa: E402
from simplcore.core.model import SimplCoreModel                        # noqa: E402
from simplcore.core.train import (                                 # noqa: E402
    build_param_groups,
    save_checkpoint,
    load_checkpoint,
    cosine_with_warmup,
)


# ---------------------------------------------------------------------------
# 训练辅助
# ---------------------------------------------------------------------------
def _make_model_cfg():
    cfg = get_preset("debug")
    cfg.vocab_size = 64
    return cfg


def _train_step(model, optim, cfg, device="cpu"):
    """一步训练 (固定随机 token 数据), 返回 loss。"""
    x = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    optim.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    return float(loss.detach())


def _train_run(seed: int, n_steps: int) -> dict:
    """跑 n_steps 步 AdamW 训练 (随机 token 数据), 返回轨迹 + 权重。"""
    torch.manual_seed(seed)
    cfg = _make_model_cfg()
    model = SimplCoreModel(cfg)
    optim = torch.optim.AdamW(
        build_param_groups(model, weight_decay=0.1),
        lr=3e-4, betas=(0.9, 0.95),
    )
    losses = []
    for _ in range(n_steps):
        losses.append(_train_step(model, optim, cfg))
    return {
        "losses": losses,
        "param_sum": float(sum(p.sum() for p in model.parameters())),
    }


# ---------------------------------------------------------------------------
# 1. 同 seed bit-exact
# ---------------------------------------------------------------------------
def test_same_seed_two_runs_bit_exact():
    """同 seed 两次独立训练必须逐位一致 (param sum + 逐步 loss)。"""
    r1 = _train_run(seed=42, n_steps=20)
    r2 = _train_run(seed=42, n_steps=20)
    assert r1["param_sum"] == r2["param_sum"], "same seed gave different trajectories"
    assert len(r1["losses"]) == len(r2["losses"])
    for i, (a, b) in enumerate(zip(r1["losses"], r2["losses"])):
        assert a == b, f"step {i}: loss {a} != {b}"


# ---------------------------------------------------------------------------
# 2. 不同 seed 不同轨迹
# ---------------------------------------------------------------------------
def test_different_seed_gives_different_trajectory():
    """不同 seed 两次训练必须产生不同轨迹 (证明 seed 真的生效)。"""
    r1 = _train_run(seed=42, n_steps=20)
    r2 = _train_run(seed=123, n_steps=20)
    assert r1["param_sum"] != r2["param_sum"], "different seeds gave identical trajectories"


# ---------------------------------------------------------------------------
# 3. 检查点 RNG 恢复 (核心)
# ---------------------------------------------------------------------------
def test_checkpoint_saves_and_restores_rng():
    """保存/恢复检查点后, 续训 5 步的轨迹必须等于无中断训练的第 15-20 步。

    设计: 无中断跑 20 步 (seed=42), 记录全 loss。
    中断版: 同 seed 跑到 15 步, 保存检查点 (含 RNG 状态),
           新建模型/优化器, 恢复, 续训 5 步。
    因为中断版在保存前消耗了与无中断版前 15 步完全相同的 RNG 流
    (同 seed, 同数据采样序列), 所以 restore_rng=True 恢复的 RNG 状态
    与无中断版第 15 步结束时的 RNG 状态一致, 续训 5 步的数据采样
    与无中断版第 15-20 步完全相同 → 轨迹逐位一致。
    """
    cfg = _make_model_cfg()
    total = 20
    interrupt_at = 15

    # ---- 无中断: 跑 20 步, 记录全 loss ----
    torch.manual_seed(42)
    model = SimplCoreModel(cfg)
    optim = torch.optim.AdamW(build_param_groups(model, 0.1), lr=3e-4)
    losses_full = [_train_step(model, optim, cfg) for _ in range(total)]

    # ---- 中断: 跑到 15 步保存, 恢复, 续训 5 步 ----
    torch.manual_seed(42)
    model2 = SimplCoreModel(cfg)
    optim2 = torch.optim.AdamW(build_param_groups(model2, 0.1), lr=3e-4)
    for _ in range(interrupt_at):
        _train_step(model2, optim2, cfg)

    ckpt_path = os.path.join(tempfile.gettempdir(), f"sgc_ckpt_{os.getpid()}.pt")
    save_checkpoint(ckpt_path, model2, optim2, step=interrupt_at,
                    tokens=interrupt_at * 64, cfg=cfg,
                    best_val=float("inf"), elapsed=0.0)

    model3 = SimplCoreModel(cfg)
    optim3 = torch.optim.AdamW(build_param_groups(model3, 0.1), lr=3e-4)
    load_checkpoint(ckpt_path, model3, optim3, restore_rng=True)

    losses_resumed = [_train_step(model3, optim3, cfg)
                      for _ in range(total - interrupt_at)]
    os.unlink(ckpt_path)

    assert losses_resumed == losses_full[interrupt_at:], (
        f"resume mismatch:\n  resumed : {losses_resumed}\n  expected: {losses_full[interrupt_at:]}"
    )


# ---------------------------------------------------------------------------
# 4. 学习率调度确定性
# ---------------------------------------------------------------------------
def test_cosine_with_warmup_is_deterministic():
    """warmup + cosine 调度器输出与 seed 无关 (纯函数)。"""
    a = [cosine_with_warmup(s, 100, 20, 3e-4, 3e-5) for s in range(100)]
    b = [cosine_with_warmup(s, 100, 20, 3e-4, 3e-5) for s in range(100)]
    assert a == b
    assert a[0] < a[19]        # warmup 单调升
    assert a[50] > a[99]       # cosine 单调降


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
