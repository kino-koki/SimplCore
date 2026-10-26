"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

curriculum_learning.py - 课程学习模块
课程学习调度、数据难度递增（短→长，简单→复杂）、
任务难度递增、课程进度追踪、自动难度评估
"""

import os
import sys
import math
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import torch
import numpy as np

logger = logging.getLogger("青小团.CurriculumLearning")


class DifficultyMetric(Enum):
    SEQUENCE_LENGTH = "sequence_length"
    PERPLEXITY = "perplexity"
    VOCABULARY_SIZE = "vocabulary_size"
    SYNTACTIC_COMPLEXITY = "syntactic_complexity"
    COMPOSITE = "composite"


class CurriculumStage(Enum):
    SHORT_SIMPLE = "short_simple"
    MEDIUM = "medium"
    LONG_COMPLEX = "long_complex"
    FULL = "full"


@dataclass
class CurriculumConfig:
    enabled: bool = True
    total_steps: int = 100000
    stages: List[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage.SHORT_SIMPLE,
        CurriculumStage.MEDIUM,
        CurriculumStage.LONG_COMPLEX,
        CurriculumStage.FULL,
    ])
    stage_durations: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    difficulty_metric: DifficultyMetric = DifficultyMetric.SEQUENCE_LENGTH
    min_seq_length: int = 1024
    max_seq_length: int = 524288
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    automatic_difficulty: bool = True
    difficulty_assessment_window: int = 100
    performance_threshold: float = 0.8
    progress_check_interval: int = 500
    warmup_steps: int = 1000


@dataclass
class CurriculumStats:
    current_stage: str = "short_simple"
    current_difficulty: float = 0.0
    progress: float = 0.0
    stage_progress: float = 0.0
    avg_seq_length: float = 0.0
    avg_loss: float = float("inf")
    stage_changes: int = 0
    samples_processed: int = 0
    in_warmup: bool = True


class QingxiaotuanCurriculumLearning:
    """
    青小团课程学习调度器
    支持数据难度递增（短→长，简单→复杂）、任务难度递增、
    课程进度追踪、自动难度评估

    针对70B MoE模型的长序列训练，从短序列逐步过渡到512K超长序列
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
    ):
        self.config = config or CurriculumConfig()
        self.stats = CurriculumStats()
        self._step_counter = 0
        self._loss_history: deque = deque(maxlen=self.config.difficulty_assessment_window)
        self._stage_boundaries: List[int] = []
        self._current_stage_idx = 0
        self._difficulty_scores: List[float] = []

        self._compute_stage_boundaries()
        logger.info(f"课程学习初始化完成: {len(self.config.stages)} 阶段, "
                    f"总步数={self.config.total_steps}")

    def _compute_stage_boundaries(self):
        total = self.config.total_steps - self.config.warmup_steps
        boundaries = [self.config.warmup_steps]
        cumulative = 0
        for i, duration in enumerate(self.config.stage_durations):
            cumulative += duration
            boundary = self.config.warmup_steps + int(total * cumulative)
            boundaries.append(boundary)
        self._stage_boundaries = boundaries

    def step(self, loss: Optional[float] = None) -> Dict[str, Any]:
        self._step_counter += 1
        self.stats.samples_processed += 1

        if self._step_counter <= self.config.warmup_steps:
            self.stats.in_warmup = True
            self.stats.current_difficulty = self.config.min_difficulty
            return {"difficulty": self.config.min_difficulty, "stage": "warmup", "progress": 0.0}

        self.stats.in_warmup = False

        if loss is not None:
            self._loss_history.append(loss)
            self.stats.avg_loss = 0.9 * self.stats.avg_loss + 0.1 * loss

        old_stage = self._current_stage_idx
        self._current_stage_idx = self._get_current_stage()
        current_stage = self.config.stages[self._current_stage_idx]
        self.stats.current_stage = current_stage.value

        if self._current_stage_idx != old_stage:
            self.stats.stage_changes += 1
            logger.info(f"课程阶段切换: Stage {old_stage} -> Stage {self._current_stage_idx} "
                        f"({current_stage.value})")

        difficulty = self._compute_difficulty()
        self.stats.current_difficulty = difficulty
        self.stats.progress = self._step_counter / max(1, self.config.total_steps)
        self.stats.stage_progress = self._get_stage_progress()

        auto_advance = False
        if self.config.automatic_difficulty and self._should_advance():
            auto_advance = True

        return {
            "difficulty": difficulty,
            "stage": current_stage.value,
            "stage_idx": self._current_stage_idx,
            "progress": self.stats.progress,
            "stage_progress": self.stats.stage_progress,
            "auto_advance": auto_advance,
        }

    def _get_current_stage(self) -> int:
        for i, boundary in enumerate(self._stage_boundaries):
            if self._step_counter <= boundary:
                return min(i, len(self.config.stages) - 1)
        return len(self.config.stages) - 1

    def _get_stage_progress(self) -> float:
        if self._current_stage_idx >= len(self._stage_boundaries) - 1:
            return 1.0
        stage_start = self._stage_boundaries[self._current_stage_idx]
        stage_end = self._stage_boundaries[self._current_stage_idx + 1]
        duration = stage_end - stage_start
        if duration <= 0:
            return 1.0
        return min(1.0, (self._step_counter - stage_start) / duration)

    def _compute_difficulty(self) -> float:
        stage_progress = self._get_stage_progress()
        stage_idx = self._current_stage_idx
        num_stages = len(self.config.stages)
        base_difficulty = (stage_idx + stage_progress) / max(1, num_stages)
        difficulty = self.config.min_difficulty + base_difficulty * (
            self.config.max_difficulty - self.config.min_difficulty
        )
        return min(self.config.max_difficulty, max(self.config.min_difficulty, difficulty))

    def _should_advance(self) -> bool:
        if self._step_counter % self.config.progress_check_interval != 0:
            return False
        if len(self._loss_history) < self.config.difficulty_assessment_window // 2:
            return False
        recent_losses = list(self._loss_history)[-min(50, len(self._loss_history)):]
        if len(recent_losses) < 10:
            return False
        loss_trend = recent_losses[-1] < np.mean(recent_losses) * self.config.performance_threshold
        return loss_trend

    def get_seq_length(self, base_length: int = 1024) -> int:
        difficulty = self.stats.current_difficulty
        if self.stats.in_warmup:
            return base_length
        length_range = self.config.max_seq_length - self.config.min_seq_length
        target_length = int(self.config.min_seq_length + length_range * difficulty)
        return min(target_length, self.config.max_seq_length)

    def get_sample_weight(self, original_weight: float = 1.0) -> float:
        difficulty = self.stats.current_difficulty
        return original_weight * (0.5 + 0.5 * difficulty)

    def should_use_sample(self, sample_length: int) -> bool:
        if self.stats.in_warmup:
            return sample_length <= self.config.min_seq_length * 2
        max_allowed = int(self.config.min_seq_length + (self.config.max_seq_length - self.config.min_seq_length) * self.stats.current_difficulty)
        return sample_length <= max_allowed

    def assess_difficulty(self, sample: Dict[str, Any]) -> float:
        if self.config.difficulty_metric == DifficultyMetric.SEQUENCE_LENGTH:
            seq_len = sample.get("input_ids", sample.get("tokens"))
            if torch.is_tensor(seq_len):
                seq_len = seq_len.shape[0]
            elif isinstance(seq_len, (list, tuple)):
                seq_len = len(seq_len)
            else:
                seq_len = 0
            return min(1.0, seq_len / self.config.max_seq_length)

        elif self.config.difficulty_metric == DifficultyMetric.PERPLEXITY:
            if "perplexity" in sample:
                return min(1.0, sample["perplexity"] / 100.0)
            return 0.5

        elif self.config.difficulty_metric == DifficultyMetric.COMPOSITE:
            scores = []
            seq_len = sample.get("input_ids", sample.get("tokens"))
            if torch.is_tensor(seq_len):
                scores.append(min(1.0, seq_len.shape[0] / self.config.max_seq_length))
            elif isinstance(seq_len, (list, tuple)):
                scores.append(min(1.0, len(seq_len) / self.config.max_seq_length))
            if "perplexity" in sample:
                scores.append(min(1.0, sample["perplexity"] / 100.0))
            if "vocab_size" in sample:
                scores.append(min(1.0, sample["vocab_size"] / 100000))
            return np.mean(scores) if scores else 0.5

        return 0.5

    def get_curriculum_stats(self) -> CurriculumStats:
        return self.stats

    def get_stage_info(self) -> Dict[str, Any]:
        return {
            "current_stage": self.stats.current_stage,
            "stage_idx": self._current_stage_idx,
            "total_stages": len(self.config.stages),
            "stage_boundaries": self._stage_boundaries,
            "stage_durations": self.config.stage_durations,
        }

    def reset(self):
        self._step_counter = 0
        self._current_stage_idx = 0
        self._loss_history.clear()
        self.stats = CurriculumStats()
        logger.info("课程学习已重置")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "total_steps": self.config.total_steps,
                "min_seq_length": self.config.min_seq_length,
                "max_seq_length": self.config.max_seq_length,
                "difficulty_metric": self.config.difficulty_metric.value,
            },
            "stats": {
                "current_stage": self.stats.current_stage,
                "current_difficulty": self.stats.current_difficulty,
                "progress": self.stats.progress,
                "stage_changes": self.stats.stage_changes,
                "samples_processed": self.stats.samples_processed,
            },
            "_step_counter": self._step_counter,
            "_current_stage_idx": self._current_stage_idx,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._step_counter = state_dict.get("_step_counter", 0)
        self._current_stage_idx = state_dict.get("_current_stage_idx", 0)

        stats_data = state_dict.get("stats", {})
        if stats_data:
            self.stats.current_stage = stats_data.get("current_stage", "short_simple")
            self.stats.current_difficulty = stats_data.get("current_difficulty", 0.0)
            self.stats.progress = stats_data.get("progress", 0.0)
            self.stats.stage_changes = stats_data.get("stage_changes", 0)

        logger.info(f"课程学习状态已恢复: stage={self.stats.current_stage}, "
                    f"difficulty={self.stats.current_difficulty:.4f}")

    def print_curriculum_summary(self):
        stage_info = self.get_stage_info()
        summary = [
            "=" * 60,
            "青小团课程学习 - 课程调度摘要",
            "=" * 60,
            f"课程学习: {'启用' if self.config.enabled else '禁用'}",
            f"难度指标: {self.config.difficulty_metric.value}",
            f"自动难度: {'启用' if self.config.automatic_difficulty else '禁用'}",
            f"总步数: {self.config.total_steps}",
            f"预热步数: {self.config.warmup_steps}",
            f"---",
            f"当前阶段: {self.stats.current_stage} ({self._current_stage_idx + 1}/{len(self.config.stages)})",
            f"当前难度: {self.stats.current_difficulty:.4f}",
            f"课程进度: {self.stats.progress:.4f}",
            f"阶段进度: {self.stats.stage_progress:.4f}",
            f"阶段切换次数: {self.stats.stage_changes}",
            f"处理样本数: {self.stats.samples_processed}",
            f"---",
            f"序列长度范围: {self.config.min_seq_length} ~ {self.config.max_seq_length}",
            f"难度范围: [{self.config.min_difficulty}, {self.config.max_difficulty}]",
            f"性能阈值: {self.config.performance_threshold}",
            f"评估窗口: {self.config.difficulty_assessment_window}",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
