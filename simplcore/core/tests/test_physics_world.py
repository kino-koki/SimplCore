# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""物理世界模型（实验功能，opt-in）的守护测试.

覆盖六条语义：
1. **opt-in 纪律**：config 默认 False；**全部预设（含 tuan-2b）不置 True**；
   训练主路径（training/）源码零引用——tuan-2b 训练完全不经过本模块；
2. 抽取器：位置/上升/下降事件的数值、符号、span（「第几个字」可溯源）；
3. 约束：连续性（平滑=1，跳变=0）、运动学、位移闭合的数值口径；
4. 信号分离：相容文本低信号、矛盾文本高信号（机制冒烟）；
5. bias_fn 钩子：未触发返回 None（零侵入）；触发时 EOS 正偏置且位置正确；
6. 预测头：形状与梯度可达；纯净性由 purity 全仓扫描覆盖（无注意力）。

诚实边界：抽取器是结构化占位（阿拉伯数字）；约束是手工先验——
测试锁**机制**，不锁「理解物理」。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from simplcore.core.model.config import SIMPLCORE_PRESETS, SimplCoreConfig
from simplcore.core.model.model import SimplCoreModel
from simplcore.core.model.physics_world import (PhysicsConstraints,
                                                PhysicsEvent,
                                                PhysicsPredictor,
                                                PhysicsWorldModel,
                                                RegexQuantityExtractor)

ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. opt-in 纪律：默认关、预设全关、训练路径零引用
# ---------------------------------------------------------------------------
def test_opt_in_default_off_and_presets_never_enable() -> None:
    assert SimplCoreConfig().use_physics_world_model is False
    # 全部预设（含 tuan-2b）不置 True ——「tuan-2b 训练时完全不使用」
    for name, preset in SIMPLCORE_PRESETS.items():
        assert preset.use_physics_world_model is False, \
            f"preset {name} 不得默认开启实验性物理世界模型"
    # 显式开启才可用
    assert SimplCoreConfig(use_physics_world_model=True) \
        .use_physics_world_model is True


def test_training_path_never_references_physics_world() -> None:
    """训练主路径零引用：tuan-2b 训练管线不经过本模块（除非显式接线）。"""
    training = ROOT / "simplcore" / "core" / "training"
    hits = [p for p in training.rglob("*.py")
            if "physics_world" in p.read_text(encoding="utf-8")]
    assert hits == [], f"训练路径不得引用物理世界模型: {hits}"


# ---------------------------------------------------------------------------
# 2. 抽取器：数值/符号/span
# ---------------------------------------------------------------------------
def test_extractor_positions_and_deltas_with_span() -> None:
    ex = RegexQuantityExtractor()
    text = "他在 2 米处停住。之后上升 1.5 米。接着下降 0.5 米。"
    events = ex.extract(text)
    kinds = [(e.kind, e.quantity) for e in events]
    assert ("position", 2.0) in kinds
    assert ("delta", 1.5) in kinds
    assert ("delta", -0.5) in kinds
    # span 可溯源：切回原文必须还原数字本身（「第几个字」级）
    for e in events:
        assert text[e.span[0]:e.span[1]] in {"2", "1.5", "0.5"}


# ---------------------------------------------------------------------------
# 3. 约束数值口径
# ---------------------------------------------------------------------------
def test_constraints_numerics() -> None:
    # 连续性：平滑 → 1；跳变 → 0
    assert PhysicsConstraints.continuity([0.0, 1.0, 2.0]) == 1.0
    assert PhysicsConstraints.continuity([0.0, 1.0, 100.0], max_jump=10.0) == 0.0
    # 运动学：匀速 → 1；突然 20 m/s → 0
    assert PhysicsConstraints.kinematics([1.0, 1.0, 1.0], [1, 1, 1]) == 1.0
    assert PhysicsConstraints.kinematics([1.0, 1.0, 21.0], [1, 1, 1]) == 0.0
    # 位移闭合：0 +3 −3 → 0 精确闭合
    assert PhysicsConstraints.net_displacement_consistent(0.0, [3.0, -3.0], 0.0) == 1.0
    assert PhysicsConstraints.net_displacement_consistent(0.0, [3.0, -3.0], 5.0) < 1.0
    # 单点/空轨迹：无信号 = 相容
    assert PhysicsConstraints.continuity([1.0]) == 1.0


# ---------------------------------------------------------------------------
# 4. 信号分离：相容 vs 矛盾
# ---------------------------------------------------------------------------
def test_signal_separates_consistent_from_contradictory() -> None:
    pwm = PhysicsWorldModel(dim=16)
    good = "他在 2 米处停住。之后位于 3 米高度。随后在 4 米处。"
    bad = "他在 2 米处停住。之后位于 3 米高度。突然位于 30 米处。"
    s_good = pwm.hallucination_signal(good)
    s_bad = pwm.hallucination_signal(bad)
    assert s_good < 0.2, f"相容文本应低信号: {s_good}"
    assert s_bad > 0.5, f"跳变文本应高信号: {s_bad}"
    assert s_bad > s_good
    # 无物理量 → 无信号（但审计里如实标注）
    empty = pwm.hallucination_signal("今天天气不错。")
    assert empty == 0.0
    audit = pwm.audit("他在 2 米处。")
    assert audit["note"] and audit["events"][0]["kind"] == "position"


# ---------------------------------------------------------------------------
# 5. bias_fn 生成钩子（零侵入）
# ---------------------------------------------------------------------------
def test_bias_fn_hook_contract() -> None:
    pwm = PhysicsWorldModel(dim=16)
    good = "他在 2 米处停住。之后位于 3 米高度。随后在 4 米处。"
    bad = "他在 2 米处停住。之后位于 3 米高度。突然位于 30 米处。"
    V = 50
    eos = 3
    logits = torch.randn(1, V)
    # 未触发：None（逐位零侵入）
    assert pwm.bias_fn(good, threshold=0.5, eos_token_id=eos)(logits) is None
    # 触发：只有 EOS 位置有正偏置，其余为零
    bias = pwm.bias_fn(bad, threshold=0.5, eos_token_id=eos)(logits)
    assert bias is not None
    assert float(bias[..., eos]) > 0.0
    mask = torch.zeros(V, dtype=torch.bool)
    mask[eos] = True
    assert float(bias[..., ~mask].abs().max()) == 0.0
    # 与 generate 钩子签名可拼（smoke：不实际采样整个序列）
    model = SimplCoreModel(SimplCoreConfig(vocab_size=V, dim=16, num_layers=1,
                                           num_bases=4, ffn_ratio=2.0,
                                           decay_max=256.0, local_kernel=2))
    model.eval()
    out = model.generate(torch.randint(0, V, (1, 4)), max_new_tokens=2,
                         logit_bias_fn=pwm.bias_fn(bad, threshold=0.5,
                                                   eos_token_id=eos))
    assert out.shape == (1, 6)


# ---------------------------------------------------------------------------
# 6. 预测头：形状与梯度
# ---------------------------------------------------------------------------
def test_predictor_shape_and_grad() -> None:
    p = PhysicsPredictor(dim=32, n_quantities=3)
    h = torch.randn(2, 32, requires_grad=True)
    out = p(h)
    assert out.shape == (2, 3)
    out.sum().backward()
    assert p.head.weight.grad is not None
    assert h.grad is not None
    # 未训练头的输出无意义——模块 docstring 已声明；这里只锁"可训练"
    assert any(q.requires_grad for q in p.parameters())