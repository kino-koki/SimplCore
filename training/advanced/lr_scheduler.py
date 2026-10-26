"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

lr_scheduler.py - 学习率调度模块
余弦退火、线性预热、阶段性学习率、循环学习率、
自适应学习率、Warmup-Stable-Decay三阶段、学习率重置/恢复
"""

import os
import sys
import math
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import torch
import numpy as np

logger = logging.getLogger("青小团.LRScheduler")


class SchedulerType(Enum):
    COSINE = "cosine"
    LINEAR_WARMUP = "linear_warmup"
    STAGED = "staged"
    CYCLIC = "cyclic"
    ADAPTIVE = "adaptive"
    WARMUP_STABLE_DECAY = "warmup_stable_decay"


@dataclass
class SchedulerConfig:
    scheduler_type: SchedulerType = SchedulerType.WARMUP_STABLE_DECAY
    warmup_steps: int = 2000
    stable_steps: int = 50000
    decay_steps: int = 48000
    total_steps: int = 100000
    min_lr: float = 3e-5
    max_lr: float = 3e-4
    initial_lr: float = 0.0

    cosine_cycles: float = 1.0

    staged_boundaries: List[int] = field(default_factory=lambda: [30000, 60000, 80000])
    staged_factors: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.1])

    cyclic_period: int = 10000
    cyclic_scale_down: float = 0.5

    adaptive_patience: int = 5
    adaptive_cooldown: int = 10
    adaptive_factor: float = 0.5
    adaptive_threshold: float = 1e-4

    warmup_type: str = "linear"


@dataclass
class LRSchedulerStats:
    current_lr: float = 0.0
    initial_lr: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    warmup_steps: int = 0
    lr_changes: int = 0
    min_lr_reached: bool = False
    phase: str = "warmup"
    lr_decreases: int = 0


class QingxiaotuanLRScheduler:
    """
    青小团学习率调度器
    支持余弦退火、线性预热、阶段性学习率、循环学习率、
    自适应学习率、Warmup-Stable-Decay三阶段调度

    针对70B MoE模型的训练特性，提供灵活的学习率调度策略
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Optional[SchedulerConfig] = None,
    ):
        self.optimizer = optimizer
        self.config = config or SchedulerConfig()
        self.stats = LRSchedulerStats()
        self.stats.initial_lr = self.config.max_lr
        self.stats.total_steps = self.config.total_steps
        self.stats.warmup_steps = self.config.warmup_steps

        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step_count = 0
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._cooldown_counter = 0
        self._last_lr = self.config.max_lr

        self._set_initial_lr()

    def _set_initial_lr(self):
        lr = self.config.initial_lr if self.config.initial_lr > 0 else self.config.min_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self, current_loss: Optional[float] = None) -> float:
        self._step_count += 1
        self.stats.current_step = self._step_count

        lr = self._compute_lr(current_loss)
        self._apply_lr(lr)
        self.stats.current_lr = lr

        return lr

    def _compute_lr(self, current_loss: Optional[float] = None) -> float:
        t = self._step_count

        if t <= self.config.warmup_steps:
            self.stats.phase = "warmup"
            return self._warmup_lr(t)

        if self.config.scheduler_type == SchedulerType.COSINE:
            self.stats.phase = "cosine"
            return self._cosine_lr(t)

        elif self.config.scheduler_type == SchedulerType.STAGED:
            self.stats.phase = "staged"
            return self._staged_lr(t)

        elif self.config.scheduler_type == SchedulerType.CYCLIC:
            self.stats.phase = "cyclic"
            return self._cyclic_lr(t)

        elif self.config.scheduler_type == SchedulerType.ADAPTIVE:
            self.stats.phase = "adaptive"
            return self._adaptive_lr(t, current_loss)

        elif self.config.scheduler_type == SchedulerType.WARMUP_STABLE_DECAY:
            return self._warmup_stable_decay_lr(t)

        return self._cosine_lr(t)

    def _warmup_lr(self, step: int) -> float:
        if self.config.warmup_type == "linear":
            factor = step / max(1, self.config.warmup_steps)
        elif self.config.warmup_type == "exponential":
            factor = 1.0 - math.exp(-5.0 * step / max(1, self.config.warmup_steps))
        elif self.config.warmup_type == "sqrt":
            factor = math.sqrt(step / max(1, self.config.warmup_steps))
        else:
            factor = step / max(1, self.config.warmup_steps)

        return self.config.min_lr + (self.config.max_lr - self.config.min_lr) * factor

    def _cosine_lr(self, step: int) -> float:
        decay_steps = max(1, self.config.total_steps - self.config.warmup_steps)
        progress = (step - self.config.warmup_steps) / decay_steps
        progress = min(progress, 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress * self.config.cosine_cycles))
        lr = self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine
        return lr

    def _staged_lr(self, step: int) -> float:
        boundaries = self.config.staged_boundaries
        factors = self.config.staged_factors

        for i, boundary in enumerate(boundaries):
            if step <= boundary:
                factor = factors[i] if i < len(factors) else factors[-1]
                return self.config.max_lr * factor

        return self.config.max_lr * factors[-1]

    def _cyclic_lr(self, step: int) -> float:
        effective_step = step - self.config.warmup_steps
        if effective_step <= 0:
            return self.config.max_lr

        period = self.config.cyclic_period
        cycle_pos = (effective_step % period) / period
        cosine = 0.5 * (1.0 + math.cos(math.pi * cycle_pos * 2))

        cycle_number = effective_step // period
        amplitude = self.config.cyclic_scale_down ** cycle_number
        lr_min = self.config.min_lr * (1.0 - amplitude) + self.config.min_lr * amplitude

        lr = lr_min + (self.config.max_lr - lr_min) * cosine * amplitude
        return max(lr, self.config.min_lr * 0.1)

    def _adaptive_lr(self, step: int, current_loss: Optional[float]) -> float:
        if current_loss is None:
            return self._cosine_lr(step)

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return self._last_lr

        if current_loss < self._best_loss * (1.0 - self.config.adaptive_threshold):
            self._best_loss = current_loss
            self._patience_counter = 0
            return self._last_lr
        else:
            self._patience_counter += 1
            if self._patience_counter >= self.config.adaptive_patience:
                new_lr = self._last_lr * self.config.adaptive_factor
                new_lr = max(new_lr, self.config.min_lr)
                self._last_lr = new_lr
                self._patience_counter = 0
                self._cooldown_counter = self.config.adaptive_cooldown
                self.stats.lr_changes += 1
                self.stats.lr_decreases += 1
                logger.info(f"自适应学习率降低至 {new_lr:.6e}")
                return new_lr

        return self._last_lr

    def _warmup_stable_decay_lr(self, step: int) -> float:
        if step <= self.config.warmup_steps:
            self.stats.phase = "warmup"
            return self._warmup_lr(step)

        warmup_end = self.config.warmup_steps
        stable_end = warmup_end + self.config.stable_steps
        total = warmup_end + self.config.stable_steps + self.config.decay_steps

        if step <= stable_end:
            self.stats.phase = "stable"
            return self.config.max_lr

        self.stats.phase = "decay"
        decay_progress = (step - stable_end) / max(1, self.config.decay_steps)
        decay_progress = min(decay_progress, 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        lr = self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine_decay
        return lr

    def _apply_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        return self.stats.current_lr

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def reset(self):
        self._step_count = 0
        self._best_loss = float("inf")
        self._patience_counter = 0
        self._cooldown_counter = 0
        self.stats = LRSchedulerStats()
        self.stats.initial_lr = self.config.max_lr
        self.stats.total_steps = self.config.total_steps
        self.stats.warmup_steps = self.config.warmup_steps
        self._set_initial_lr()
        logger.info("学习率调度器已重置")

    def restore_lr(self, lr: float):
        self._last_lr = lr
        self._apply_lr(lr)
        logger.info(f"学习率已恢复至 {lr:.6e}")

    def set_phase(self, total_steps: Optional[int] = None,
                  warmup_steps: Optional[int] = None,
                  min_lr: Optional[float] = None,
                  max_lr: Optional[float] = None):
        if total_steps is not None:
            self.config.total_steps = total_steps
            self.stats.total_steps = total_steps
        if warmup_steps is not None:
            self.config.warmup_steps = warmup_steps
            self.stats.warmup_steps = warmup_steps
        if min_lr is not None:
            self.config.min_lr = min_lr
        if max_lr is not None:
            self.config.max_lr = max_lr
        logger.info(f"学习率参数已更新: total={self.config.total_steps}, "
                    f"warmup={self.config.warmup_steps}, "
                    f"lr=[{self.config.min_lr:.6e}, {self.config.max_lr:.6e}]")

    def get_scheduler_type(self) -> SchedulerType:
        return self.config.scheduler_type

    def get_current_phase(self) -> str:
        return self.stats.phase

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "scheduler_type": self.config.scheduler_type.value,
                "warmup_steps": self.config.warmup_steps,
                "stable_steps": self.config.stable_steps,
                "decay_steps": self.config.decay_steps,
                "total_steps": self.config.total_steps,
                "min_lr": self.config.min_lr,
                "max_lr": self.config.max_lr,
                "cosine_cycles": self.config.cosine_cycles,
            },
            "stats": {
                "current_lr": self.stats.current_lr,
                "current_step": self.stats.current_step,
                "phase": self.stats.phase,
                "lr_changes": self.stats.lr_changes,
            },
            "_step_count": self._step_count,
            "_best_loss": self._best_loss,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        config_data = state_dict.get("config", {})
        if config_data:
            self.config.scheduler_type = SchedulerType(config_data.get("scheduler_type", "warmup_stable_decay"))
            self.config.warmup_steps = config_data.get("warmup_steps", self.config.warmup_steps)
            self.config.stable_steps = config_data.get("stable_steps", self.config.stable_steps)
            self.config.decay_steps = config_data.get("decay_steps", self.config.decay_steps)
            self.config.total_steps = config_data.get("total_steps", self.config.total_steps)
            self.config.min_lr = config_data.get("min_lr", self.config.min_lr)
            self.config.max_lr = config_data.get("max_lr", self.config.max_lr)

        self._step_count = state_dict.get("_step_count", 0)
        self._best_loss = state_dict.get("_best_loss", float("inf"))
        self._last_lr = state_dict.get("_last_lr", self.config.max_lr)

        stats_data = state_dict.get("stats", {})
        if stats_data:
            self.stats.current_lr = stats_data.get("current_lr", 0.0)
            self.stats.current_step = stats_data.get("current_step", 0)
            self.stats.phase = stats_data.get("phase", "warmup")

        logger.info(f"学习率调度器状态已恢复: phase={self.stats.phase}, lr={self.stats.current_lr:.6e}")

    def print_schedule(self, num_points: int = 20):
        steps = np.linspace(0, self.config.total_steps, num_points).astype(int)
        lrs = []

        original_step = self._step_count
        for s in steps:
            self._step_count = s
            lr = self._compute_lr(None)
            lrs.append(lr)
        self._step_count = original_step

        summary = [
            "=" * 60,
            "青小团学习率调度 - 调度计划",
            "=" * 60,
            f"调度类型: {self.config.scheduler_type.value}",
            f"总步数: {self.config.total_steps}",
            f"预热步数: {self.config.warmup_steps}",
            f"稳定步数: {self.config.stable_steps}",
            f"衰减步数: {self.config.decay_steps}",
            f"最大LR: {self.config.max_lr:.6e}",
            f"最小LR: {self.config.min_lr:.6e}",
            f"当前步数: {self._step_count}",
            f"当前阶段: {self.stats.phase}",
            f"当前LR: {self.stats.current_lr:.6e}",
            f"---",
        ]
        for i, (s, lr) in enumerate(zip(steps, lrs)):
            bar_length = min(40, int(lr / self.config.max_lr * 40))
            bar = "█" * bar_length + "░" * (40 - bar_length)
            summary.append(f"Step {s:7d} | {bar} | {lr:.6e}")

        summary.append("=" * 60)
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
