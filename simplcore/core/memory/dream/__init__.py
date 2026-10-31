# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Dreaming 模式 —— 架构层的模型自进化反思循环。

**Dreaming 与 Agent 的区别**：Agent 的反思是**推理期**的（本次回答里多想一步），
Dreaming 的反思是**离线**的：模型在"睡眠"阶段回放自己的经验、反思出洞见、
把洞见变回训练信号、巩固自己，然后才回到线上。

本模块提供的是**循环骨架与契约**，不是某个具体反思策略：

    在线 (serving)                    离线 (dreaming)
    ─────────────                    ────────────────
    推理 → 轨迹入流                   HARVEST    采样高价值经验
         ↓                           REFLECT    模型自评 → 洞见
    CMT 压缩记忆库                    SYNTHESIZE 洞见 → 训练信号
    (检索 → 聚合 → 注入)              CONSOLIDATE 巩固（记忆 或 权重）
         ↑                           VERIFY     held-out 评测
    权重 / 记忆库 ←── COMMIT         ROLLBACK   退化则回滚
                                  （或只回滚，不提交）

四种巩固模式
------------
* ``memory``  —— 只更新 CMT 压缩记忆库。**零遗忘风险**（基座权重冻结），
  是 CMT 的默认模式。
* ``weights`` —— 自蒸馏写回权重。真正的"自进化"，但会引入遗忘，
  因此**必须**过 no-regression gate 才提交。
* ``both``    —— 先写记忆再试写权重，权重部分受 gate 保护。
* ``none``    —— 只反思不巩固（用于观察模式 / 消融）。

硬性约束
--------
1. **no-regression gate**：任何写回权重的巩固，若在 held-out 集上相对基线退化
   超过容差，一律回滚（权重快照 + 记忆快照）。
2. **可恢复**：每轮结束保存 checkpoint（阶段、指标、游标），可中断续跑。
3. **有预算**：每轮的轨迹数、洞见数、轮数上限都是硬约束。
4. **可观测**：每轮产出 :class:`ReflectionReport`，含各阶段耗时与产出计数。

诚实边界
--------
本模块**不保证**自进化会提升能力。默认的反思器做的是**文本层可归因性检查**
（:mod:`simplcore.core.grounding`），不是语义批判：

* 它能发现"这句话在证据里找不到出处"（含数字篡改、否定翻转、跨文档拼接）；
* 它**不能**发现逐字引用但语境误用（反讽、条件句、引述他人观点）；
* 它对语义等价改写偏保守（会把正确改写判成无依据）。

因此 ``synthesized == 0`` 是一个**有意义的结果**：它意味着本轮所有发现都没有
可引用的原文，此时正确行为是补检索，而不是拿模型自评去训练自己。
在没有带引用的独立评测集上实测出提升之前，**不得宣称"自进化有效"**。
"""
from __future__ import annotations

from .buffer import ExperienceBuffer
from .evaluation import EvalScores, GateVerdict, paired_bootstrap_ci, regression_gate
from .hooks import (default_consolidator, default_reflector, default_synthesizer,
                    default_verifier)
from .loop import ReflectionConfig, ReflectionLoop
from .types import ReflectionReport, ReflectionStage, Insight, Trajectory, TrainExample

__all__ = [
    "ReflectionConfig",
    "ReflectionLoop",
    "ReflectionReport",
    "ReflectionStage",
    "EvalScores",
    "ExperienceBuffer",
    "GateVerdict",
    "Insight",
    "TrainExample",
    "Trajectory",
    "default_consolidator",
    "default_reflector",
    "default_synthesizer",
    "default_verifier",
    "paired_bootstrap_ci",
    "regression_gate",
]
