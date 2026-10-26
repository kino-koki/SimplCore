"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

mixed_precision_trainer.py - 混合精度训练模块
支持FP16/BF16/FP32动态切换、自动loss scaling、梯度裁剪与累积、
精度监控、逐层精度分配、MoE层特殊精度处理
"""

import os
import sys
import math
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.MixedPrecisionTrainer")


class PrecisionMode(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass
class PrecisionConfig:
    mode: PrecisionMode = PrecisionMode.BF16
    initial_scale: float = 2.0 ** 16
    scale_factor: float = 2.0
    growth_interval: int = 2000
    backoff_factor: float = 0.5
    max_scale: float = 2.0 ** 24
    min_scale: float = 1.0
    hysteresis: int = 2
    overflow_count_limit: int = 4


@dataclass
class LayerPrecisionSpec:
    layer_name_pattern: str = ""
    precision: PrecisionMode = PrecisionMode.FP32


@dataclass
class PrecisionStats:
    total_steps: int = 0
    overflow_steps: int = 0
    current_scale: float = 1.0
    loss_scale: float = 1.0
    overflow_rate: float = 0.0
    avg_gradient_norm: float = 0.0
    fp32_layer_count: int = 0
    fp16_layer_count: int = 0
    bf16_layer_count: int = 0
    last_overflow_step: int = -1
    consecutive_overflow: int = 0
    scale_change_count: int = 0


class QingxiaotuanMixedPrecisionTrainer:
    """
    青小团混合精度训练器
    支持FP16/BF16/FP32动态切换、自动loss scaling、梯度裁剪与累积、
    精度监控、逐层精度分配、MoE层特殊精度处理

    针对70B MoE模型优化，确保专家层和路由层在适当精度下稳定训练
    """

    def __init__(
        self,
        model: nn.Module,
        precision_config: Optional[PrecisionConfig] = None,
        layer_precision_specs: Optional[List[LayerPrecisionSpec]] = None,
        gradient_clip_value: float = 1.0,
        gradient_clip_norm: float = 1.0,
        gradient_accumulation_steps: int = 12,
        enable_autocast: bool = True,
        sync_each_layer: bool = False,
        moe_fp32_experts: bool = True,
        moe_fp32_gate: bool = True,
    ):
        self.model = model
        self.config = precision_config or PrecisionConfig()
        self.layer_precision_specs = layer_precision_specs or []
        self.gradient_clip_value = gradient_clip_value
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_autocast = enable_autocast
        self.sync_each_layer = sync_each_layer
        self.moe_fp32_experts = moe_fp32_experts
        self.moe_fp32_gate = moe_fp32_gate

        self.loss_scale = self.config.initial_scale
        self.growth_counter = 0
        self.overflow_count = 0
        self.accumulation_counter = 0
        self.optimizer = None
        self.stats = PrecisionStats()
        self.stats.loss_scale = self.loss_scale
        self.stats.current_scale = self.loss_scale

        self._layer_precision_map: Dict[str, PrecisionMode] = {}
        self._allocated_precision_layers()
        self._apply_moe_precision()

    def _allocated_precision_layers(self):
        for name, module in self.model.named_modules():
            assigned = False
            for spec in self.layer_precision_specs:
                if spec.layer_name_pattern in name:
                    self._layer_precision_map[name] = spec.precision
                    assigned = True
                    break
            if not assigned:
                if self.moe_fp32_experts and ("experts" in name.lower() or "expert" in name.lower()):
                    self._layer_precision_map[name] = PrecisionMode.FP32
                elif self.moe_fp32_gate and ("gate" in name.lower() or "router" in name.lower()):
                    self._layer_precision_map[name] = PrecisionMode.FP32
                else:
                    self._layer_precision_map[name] = self.config.mode

        for prec in PrecisionMode:
            count = sum(1 for p in self._layer_precision_map.values() if p == prec)
            if prec == PrecisionMode.FP32:
                self.stats.fp32_layer_count = count
            elif prec == PrecisionMode.FP16:
                self.stats.fp16_layer_count = count
            elif prec == PrecisionMode.BF16:
                self.stats.bf16_layer_count = count

        logger.info(
            f"逐层精度分配完成: FP32={self.stats.fp32_layer_count}, "
            f"FP16={self.stats.fp16_layer_count}, BF16={self.stats.bf16_layer_count}"
        )

    def _apply_moe_precision(self):
        for name, module in self.model.named_modules():
            precision = self._layer_precision_map.get(name)
            if precision is None:
                continue
            if precision == PrecisionMode.FP32:
                for param in module.parameters():
                    if param.data.dtype != torch.float32:
                        param.data = param.data.float()
            elif precision == PrecisionMode.FP16:
                for param in module.parameters():
                    if param.data.dtype != torch.float16:
                        param.data = param.data.half()
            elif precision == PrecisionMode.BF16:
                for param in module.parameters():
                    if param.data.dtype != torch.bfloat16:
                        param.data = param.data.bfloat16()

        logger.info("MoE层精度处理完成: 专家层和路由层已配置为FP32")

    def _get_autocast_dtype(self) -> torch.dtype:
        if self.config.mode == PrecisionMode.FP16:
            return torch.float16
        elif self.config.mode == PrecisionMode.BF16:
            return torch.bfloat16
        else:
            return torch.float32

    def _get_layer_autocast_dtype(self, layer_name: str) -> torch.dtype:
        precision = self._layer_precision_map.get(layer_name, self.config.mode)
        if precision == PrecisionMode.FP16:
            return torch.float16
        elif precision == PrecisionMode.BF16:
            return torch.bfloat16
        else:
            return torch.float32

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def get_learning_rate(self) -> float:
        if self.optimizer is None:
            return 0.0
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    def set_precision_mode(self, mode: PrecisionMode):
        old_mode = self.config.mode
        self.config.mode = mode
        logger.info(f"精度模式切换: {old_mode.value} -> {mode.value}")
        self._allocated_precision_layers()
        self._apply_moe_precision()

    def set_layer_precision(self, layer_name_pattern: str, precision: PrecisionMode):
        for name in self._layer_precision_map:
            if layer_name_pattern in name:
                self._layer_precision_map[name] = precision
                module = dict(self.model.named_modules()).get(name)
                if module is not None:
                    if precision == PrecisionMode.FP32:
                        for param in module.parameters():
                            param.data = param.data.float()
                    elif precision == PrecisionMode.FP16:
                        for param in module.parameters():
                            param.data = param.data.half()
                    elif precision == PrecisionMode.BF16:
                        for param in module.parameters():
                            param.data = param.data.bfloat16()
                logger.info(f"层精度已更新: {name} -> {precision.value}")

    def train_step(
        self,
        loss_fn: Callable,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行单个训练步骤（含前向、反向、精度管理）
        返回包含loss、梯度范数、溢出状态等信息的字典
        """
        self.model.train()
        precision_dtype = self._get_autocast_dtype()

        if self.enable_autocast and self.config.mode != PrecisionMode.FP32:
            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
                loss = loss_fn(*args, **kwargs)
        else:
            loss = loss_fn(*args, **kwargs)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=next(self.model.parameters()).device)

        scaled_loss = loss * self.loss_scale
        scaled_loss.backward()

        self.growth_counter += 1
        step_info = {
            "loss": loss.detach().item(),
            "loss_scale": self.loss_scale,
            "gradient_norm": 0.0,
            "overflow": False,
            "clipped": False,
            "accumulation_step": self.accumulation_counter,
        }

        self.accumulation_counter += 1

        if self.accumulation_counter >= self.gradient_accumulation_steps:
            grad_norm = self._unscale_and_clip()
            step_info["gradient_norm"] = grad_norm
            step_info["clipped"] = True

            overflow = self._check_overflow()
            step_info["overflow"] = overflow

            if not overflow:
                if self.optimizer is not None:
                    self.optimizer.step()
                self._update_loss_scale(success=True)
                self.accumulation_counter = 0
                self._zero_grad()
            else:
                self._update_loss_scale(success=False)
                self._zero_grad()
                logger.warning(f"溢出检测，loss_scale={self.loss_scale}，跳过此步")

        self.stats.total_steps += 1
        if step_info["overflow"]:
            self.stats.overflow_steps += 1
        self.stats.overflow_rate = self.stats.overflow_steps / max(1, self.stats.total_steps)
        self.stats.loss_scale = self.loss_scale
        self.stats.current_scale = self.loss_scale

        return step_info

    def _unscale_and_clip(self) -> float:
        if self.loss_scale == 1.0 and self.config.mode == PrecisionMode.FP32:
            total_norm = 0.0
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm)

            if self.gradient_clip_norm > 0 and total_norm > self.gradient_clip_norm:
                clip_coef = self.gradient_clip_norm / (total_norm + 1e-6)
                for p in parameters:
                    p.grad.data.mul_(clip_coef)
            return total_norm

        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        inv_scale = 1.0 / self.loss_scale
        for p in parameters:
            p.grad.data.mul_(inv_scale)

        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm)
        self.stats.avg_gradient_norm = 0.9 * self.stats.avg_gradient_norm + 0.1 * total_norm

        if self.gradient_clip_norm > 0 and total_norm > self.gradient_clip_norm:
            clip_coef = self.gradient_clip_norm / (total_norm + 1e-6)
            for p in parameters:
                p.grad.data.mul_(clip_coef)

        if self.gradient_clip_value > 0:
            for p in parameters:
                p.grad.data.clamp_(-self.gradient_clip_value, self.gradient_clip_value)

        return total_norm

    def _check_overflow(self) -> bool:
        for p in self.model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    return True
                if (p.grad.data.abs() > 1e8).any():
                    return True
        return False

    def _update_loss_scale(self, success: bool):
        if self.config.mode == PrecisionMode.FP32:
            return

        if success:
            self.overflow_count = max(0, self.overflow_count - 1)
            self.growth_counter += 1
            if self.growth_counter >= self.config.growth_interval:
                self.loss_scale = min(self.loss_scale * self.config.scale_factor, self.config.max_scale)
                self.growth_counter = 0
                self.stats.scale_change_count += 1
        else:
            self.overflow_count += 1
            self.stats.consecutive_overflow += 1
            self.stats.last_overflow_step = self.stats.total_steps
            if self.overflow_count >= self.config.hysteresis:
                self.loss_scale = max(self.loss_scale * self.config.backoff_factor, self.config.min_scale)
                self.overflow_count = 0
                self.stats.scale_change_count += 1
                logger.info(f"loss_scale降低至 {self.loss_scale}")

    def _zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None

    def apply_gradient_accumulation(self, loss: torch.Tensor) -> torch.Tensor:
        scaled = loss / self.gradient_accumulation_steps
        return scaled

    def get_precision_stats(self) -> PrecisionStats:
        return self.stats

    def get_layer_precision_map(self) -> Dict[str, str]:
        return {k: v.value for k, v in self._layer_precision_map.items()}

    def get_moe_layers(self) -> List[Tuple[str, nn.Module]]:
        moe_layers = []
        for name, module in self.model.named_modules():
            if "expert" in name.lower() or "moe" in name.lower() or "gate" in name.lower():
                moe_layers.append((name, module))
        return moe_layers

    def convert_moe_to_fp32(self):
        for name, module in self.get_moe_layers():
            for param in module.parameters():
                if param.data.dtype != torch.float32:
                    param.data = param.data.float()
            self._layer_precision_map[name] = PrecisionMode.FP32
        logger.info("所有MoE层已转换为FP32")

    def convert_moe_to_bf16(self):
        for name, module in self.get_moe_layers():
            for param in module.parameters():
                if param.data.dtype != torch.bfloat16:
                    param.data = param.data.bfloat16()
            self._layer_precision_map[name] = PrecisionMode.BF16
        logger.info("所有MoE层已转换为BF16")

    def get_autocast_context(self, layer_name: Optional[str] = None):
        if not self.enable_autocast:
            return torch.no_grad()
        dtype = self._get_layer_autocast_dtype(layer_name) if layer_name else self._get_autocast_dtype()
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    def compute_moe_aux_loss(self, router_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        if router_logits.dtype != torch.float32:
            router_logits = router_logits.float()
        routing_weights = torch.softmax(router_logits, dim=-1)
        expert_mask = torch.nn.functional.one_hot(expert_indices, num_classes=router_logits.size(-1)).float()
        tokens_per_expert = expert_mask.mean(dim=0)
        routing_probs = routing_weights.mean(dim=0)
        aux_loss = (tokens_per_expert * routing_probs).sum() * router_logits.size(-1)
        return aux_loss

    def state_dict(self) -> Dict[str, Any]:
        return {
            "loss_scale": self.loss_scale,
            "growth_counter": self.growth_counter,
            "overflow_count": self.overflow_count,
            "accumulation_counter": self.accumulation_counter,
            "config": {
                "mode": self.config.mode.value,
                "initial_scale": self.config.initial_scale,
                "scale_factor": self.config.scale_factor,
                "growth_interval": self.config.growth_interval,
                "backoff_factor": self.config.backoff_factor,
                "max_scale": self.config.max_scale,
                "min_scale": self.config.min_scale,
                "hysteresis": self.config.hysteresis,
            },
            "stats": {
                "total_steps": self.stats.total_steps,
                "overflow_steps": self.stats.overflow_steps,
                "overflow_rate": self.stats.overflow_rate,
                "avg_gradient_norm": self.stats.avg_gradient_norm,
                "scale_change_count": self.stats.scale_change_count,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.loss_scale = state_dict.get("loss_scale", self.config.initial_scale)
        self.growth_counter = state_dict.get("growth_counter", 0)
        self.overflow_count = state_dict.get("overflow_count", 0)
        self.accumulation_counter = state_dict.get("accumulation_counter", 0)

        config_data = state_dict.get("config", {})
        if config_data:
            self.config.mode = PrecisionMode(config_data.get("mode", "bf16"))
            self.config.initial_scale = config_data.get("initial_scale", self.config.initial_scale)
            self.config.scale_factor = config_data.get("scale_factor", self.config.scale_factor)

        stats_data = state_dict.get("stats", {})
        if stats_data:
            self.stats.total_steps = stats_data.get("total_steps", 0)
            self.stats.overflow_steps = stats_data.get("overflow_steps", 0)
            self.stats.overflow_rate = stats_data.get("overflow_rate", 0.0)
            self.stats.avg_gradient_norm = stats_data.get("avg_gradient_norm", 0.0)

        logger.info(f"混合精度训练器状态已恢复，loss_scale={self.loss_scale}")

    def detect_optimal_precision(self, sample_batch: Dict[str, torch.Tensor]) -> PrecisionMode:
        self.model.eval()
        with torch.no_grad():
            try:
                if self.config.mode == PrecisionMode.FP16:
                    test_dtype = torch.float16
                elif self.config.mode == PrecisionMode.BF16:
                    test_dtype = torch.bfloat16
                else:
                    return PrecisionMode.FP32

                with torch.amp.autocast(device_type="cuda", dtype=test_dtype):
                    output = self.model(**sample_batch)

                has_nan_or_inf = False
                if isinstance(output, torch.Tensor):
                    has_nan_or_inf = torch.isnan(output).any() or torch.isinf(output).any()
                elif hasattr(output, "loss") and output.loss is not None:
                    has_nan_or_inf = torch.isnan(output.loss).any() or torch.isinf(output.loss).any()

                if has_nan_or_inf:
                    logger.warning(f"精度模式 {self.config.mode.value} 不稳定，建议切换为FP32")
                    return PrecisionMode.FP32
                return self.config.mode
            except Exception as e:
                logger.error(f"精度检测异常: {e}")
                return PrecisionMode.FP32

    def print_precision_summary(self):
        summary = [
            "=" * 60,
            "青小团混合精度训练 - 精度配置摘要",
            "=" * 60,
            f"全局精度模式: {self.config.mode.value}",
            f"Loss Scale: {self.loss_scale:.2e}",
            f"梯度累积步数: {self.gradient_accumulation_steps}",
            f"梯度裁剪范数: {self.gradient_clip_norm}",
            f"梯度裁剪值: {self.gradient_clip_value}",
            f"MoE专家层精度: {'FP32' if self.moe_fp32_experts else '自动'}",
            f"MoE路由层精度: {'FP32' if self.moe_fp32_gate else '自动'}",
            f"---",
            f"FP32层数: {self.stats.fp32_layer_count}",
            f"FP16层数: {self.stats.fp16_layer_count}",
            f"BF16层数: {self.stats.bf16_layer_count}",
            f"总步数: {self.stats.total_steps}",
            f"溢出率: {self.stats.overflow_rate:.4f}",
            f"平均梯度范数: {self.stats.avg_gradient_norm:.4f}",
            f"Scale调整次数: {self.stats.scale_change_count}",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
