# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""Dreaming 数据类型：阶段枚举、轨迹、洞见、训练样本、报告。"""
from __future__ import annotations



from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple





# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
class ReflectionStage(str, Enum):
    """Dreaming 循环的阶段。"""
    IDLE = "idle"
    HARVEST = "harvest"
    REFLECT = "reflect"
    SYNTHESIZE = "synthesize"
    CONSOLIDATE = "consolidate"
    VERIFY = "verify"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    FAILED = "failed"


@dataclass
class Trajectory:
    """一条经验轨迹（在线推理时落盘）。"""
    prompt: str
    response: str
    confidence: float = 1.0
    grounded: bool = True                 # 是否有外部证据支撑（对应 L4 拒答门）
    score: float = 0.0                    # 事后评分（越高越有价值）
    tokens: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    """一条反思洞见。

    ``repair`` 是**证据原文**（不是模型自己写的修订）：只有存在可引用的
    外部片段时才有值。合成器只接受带 ``repair`` 的洞见，这样"自进化"产生的
    训练信号永远可以追回一句人类可核对的原文，而不是模型的自我加冕。
    """
    text: str
    source: Optional[Trajectory] = None
    weight: float = 1.0
    kind: str = "flagged"                 # flagged | unsupported | partially_supported
    repair: Optional[str] = None          # 外部证据原文（训练目标候选）
    evidence: Optional[str] = None        # 被引用的证据片段
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainExample:
    """由洞见合成出的训练样本。"""
    prompt: str
    target: str
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionReport:
    """一轮 Dreaming 的结果。"""
    round_index: int
    stage: ReflectionStage
    harvested: int = 0
    insights: int = 0
    synthesized: int = 0
    #: 巩固**阶段**是否顺利完成（不含"是否真的写入了东西"，看下面两个字段）
    consolidated: bool = False
    #: 本轮是否真的改了模型权重
    wrote_weights: bool = False
    #: 本轮往记忆库写了多少条
    wrote_memory: int = 0
    rolled_back: bool = False
    baseline_score: Optional[float] = None
    new_score: Optional[float] = None
    score_delta: Optional[float] = None
    # 多指标 + 置信下界（no-regression gate 升级）
    metrics_base: Dict[str, float] = field(default_factory=dict)
    metrics_new: Dict[str, float] = field(default_factory=dict)
    ci_lower: Dict[str, float] = field(default_factory=dict)
    failed_metrics: List[str] = field(default_factory=list)
    #: held-out 泛化记账（「越用越好」的判据；None=调用方未提供 heldout_eval）
    heldout_base: Optional[float] = None
    heldout_new: Optional[float] = None
    heldout_delta: Optional[float] = None
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_index,
            "stage": self.stage.value,
            "harvested": self.harvested,
            "insights": self.insights,
            "synthesized": self.synthesized,
            "consolidated": self.consolidated,
            "wrote_weights": self.wrote_weights,
            "wrote_memory": self.wrote_memory,
            "rolled_back": self.rolled_back,
            "baseline_score": self.baseline_score,
            "new_score": self.new_score,
            "score_delta": self.score_delta,
            "metrics_base": self.metrics_base,
            "metrics_new": self.metrics_new,
            "ci_lower": self.ci_lower,
            "failed_metrics": list(self.failed_metrics),
            "heldout_base": self.heldout_base,
            "heldout_new": self.heldout_new,
            "heldout_delta": self.heldout_delta,
            "seconds": round(self.seconds, 3),
            "notes": list(self.notes),
        }

# 0.1-α 命名降调前的旧公共名（pickle / 外部脚本兼容）
DreamStage = ReflectionStage
DreamReport = ReflectionReport
