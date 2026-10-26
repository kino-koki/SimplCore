"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

regularization.py - 正则化模块
Dropout（注意力/MLP）、LayerDrop、Stochastic Depth、
Label Smoothing、MoE正则化（负载均衡loss增强）、权重衰减调度
"""

import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("青小团.Regularizer")


class DropoutType(Enum):
    STANDARD = "standard"
    STRUCTURED = "structured"
    VARIATIONAL = "variational"


class MoERegularizationType(Enum):
    LOAD_BALANCING = "load_balancing"
    Z_LOSS = "z_loss"
    IMPORTANCE = "importance"
    ALL = "all"


@dataclass
class RegularizationConfig:
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    embedding_dropout: float = 0.0
    dropout_type: DropoutType = DropoutType.STANDARD

    layerdrop_rate: float = 0.0
    layerdrop_decay: float = 0.0

    stochastic_depth_rate: float = 0.0
    stochastic_depth_decay: bool = True

    label_smoothing: float = 0.0

    moe_load_balancing_coef: float = 0.01
    moe_z_loss_coef: float = 0.001
    moe_importance_coef: float = 0.01
    moe_reg_type: MoERegularizationType = MoERegularizationType.LOAD_BALANCING

    weight_decay: float = 0.1
    weight_decay_schedule: bool = False
    weight_decay_final: float = 0.0


@dataclass
class RegularizationStats:
    layerdrop_active_count: int = 0
    stochastic_depth_active_count: int = 0
    moe_load_balancing_loss: float = 0.0
    moe_z_loss: float = 0.0
    total_reg_loss: float = 0.0
    current_label_smoothing: float = 0.0


class QingxiaotuanRegularizer:
    """
    青小团正则化器
    支持Dropout（注意力/MLP）、LayerDrop、Stochastic Depth、
    Label Smoothing、MoE正则化（负载均衡/ Z-loss）、权重衰减调度

    针对70B MoE模型，通过负载均衡loss增强确保专家利用率平衡
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[RegularizationConfig] = None,
    ):
        self.model = model
        self.config = config or RegularizationConfig()
        self.stats = RegularizationStats()
        self._step_counter = 0
        self._layerdrop_schedule: List[float] = []
        self._stochastic_depth_schedule: List[float] = []
        self._dropout_modules: List[nn.Dropout] = []

        self._apply_dropout()

    def _apply_dropout(self):
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                self._dropout_modules.append(module)

    def get_attention_dropout(self) -> float:
        return self.config.attention_dropout

    def get_mlp_dropout(self) -> float:
        return self.config.mlp_dropout

    def set_dropout(self, attention_dropout: Optional[float] = None,
                    mlp_dropout: Optional[float] = None):
        if attention_dropout is not None:
            self.config.attention_dropout = attention_dropout
        if mlp_dropout is not None:
            self.config.mlp_dropout = mlp_dropout
        logger.info(f"Dropout已更新: attention={self.config.attention_dropout}, "
                    f"mlp={self.config.mlp_dropout}")

    def apply_layerdrop(self, layer_idx: int, total_layers: int) -> bool:
        if self.config.layerdrop_rate <= 0:
            return True
        if self.config.layerdrop_decay > 0:
            decay = 1.0 - self.config.layerdrop_decay * (layer_idx / max(1, total_layers))
            keep_prob = 1.0 - self.config.layerdrop_rate * decay
        else:
            keep_prob = 1.0 - self.config.layerdrop_rate
        if torch.rand(1).item() < keep_prob:
            return True
        self.stats.layerdrop_active_count += 1
        return False

    def apply_stochastic_depth(self, layer_idx: int, total_layers: int, x: torch.Tensor) -> torch.Tensor:
        if self.config.stochastic_depth_rate <= 0:
            return x
        if self.config.stochastic_depth_decay:
            survival_prob = 1.0 - self.config.stochastic_depth_rate * (layer_idx / max(1, total_layers))
        else:
            survival_prob = 1.0 - self.config.stochastic_depth_rate
        if not self.training:
            return x * survival_prob
        mask = torch.rand(1, device=x.device).item() < survival_prob
        if mask:
            return x / max(survival_prob, 1e-6)
        self.stats.stochastic_depth_active_count += 1
        return torch.zeros_like(x)

    @property
    def training(self) -> bool:
        return self.model.training

    def compute_label_smoothing_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        smoothing = self.config.label_smoothing
        if smoothing <= 0:
            return torch.tensor(0.0, device=logits.device)

        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, labels, reduction="mean")

        smooth_loss = -log_probs.mean(dim=-1).mean()
        loss = (1.0 - smoothing) * nll_loss + smoothing * smooth_loss
        self.stats.current_label_smoothing = smoothing
        return loss

    def compute_moe_regularization(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int,
    ) -> Dict[str, torch.Tensor]:
        reg_losses = {}
        total_loss = torch.tensor(0.0, device=router_logits.device)

        if self.config.moe_reg_type in (MoERegularizationType.LOAD_BALANCING, MoERegularizationType.ALL):
            lb_loss = self._compute_load_balancing_loss(router_logits, expert_indices, num_experts)
            reg_losses["load_balancing"] = lb_loss * self.config.moe_load_balancing_coef
            total_loss = total_loss + reg_losses["load_balancing"]
            self.stats.moe_load_balancing_loss = lb_loss.item()

        if self.config.moe_reg_type in (MoERegularizationType.Z_LOSS, MoERegularizationType.ALL):
            z_loss = self._compute_z_loss(router_logits)
            reg_losses["z_loss"] = z_loss * self.config.moe_z_loss_coef
            total_loss = total_loss + reg_losses["z_loss"]
            self.stats.moe_z_loss = z_loss.item()

        if self.config.moe_reg_type in (MoERegularizationType.IMPORTANCE, MoERegularizationType.ALL):
            imp_loss = self._compute_importance_loss(router_logits, num_experts)
            reg_losses["importance"] = imp_loss * self.config.moe_importance_coef
            total_loss = total_loss + reg_losses["importance"]

        reg_losses["total"] = total_loss
        self.stats.total_reg_loss = total_loss.item()
        return reg_losses

    def _compute_load_balancing_loss(
        self, router_logits: torch.Tensor, expert_indices: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_mask = F.one_hot(expert_indices, num_classes=num_experts).float()
        tokens_per_expert = expert_mask.mean(dim=0)
        routing_probs = routing_weights.mean(dim=0)
        aux_loss = (tokens_per_expert * routing_probs).sum() * num_experts
        return aux_loss

    def _compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        return (router_logits ** 2).mean()

    def _compute_importance_loss(self, router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
        routing_weights = F.softmax(router_logits, dim=-1)
        importance = routing_weights.sum(dim=0)
        importance_loss = importance.var() / (importance.mean() + 1e-6)
        return importance_loss

    def get_weight_decay(self, current_step: int, total_steps: int) -> float:
        if not self.config.weight_decay_schedule:
            return self.config.weight_decay
        progress = current_step / max(1, total_steps)
        decay = self.config.weight_decay + (self.config.weight_decay_final - self.config.weight_decay) * progress
        return max(decay, self.config.weight_decay_final)

    def update_regularization(self, step: int, total_steps: int):
        self._step_counter = step
        decay_factor = 1.0 - (step / max(1, total_steps))

        if self.config.layerdrop_rate > 0:
            current_layerdrop = self.config.layerdrop_rate * (0.5 + 0.5 * decay_factor)
            self.config.layerdrop_rate = current_layerdrop

        if self.config.stochastic_depth_rate > 0:
            current_sd = self.config.stochastic_depth_rate * (0.5 + 0.5 * decay_factor)
            self.config.stochastic_depth_rate = current_sd

        if step % 100 == 0:
            logger.debug(f"正则化更新: step={step}, "
                        f"layerdrop={self.config.layerdrop_rate:.4f}, "
                        f"stochastic_depth={self.config.stochastic_depth_rate:.4f}")

    def get_regularization_stats(self) -> RegularizationStats:
        return self.stats

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "attention_dropout": self.config.attention_dropout,
                "mlp_dropout": self.config.mlp_dropout,
                "label_smoothing": self.config.label_smoothing,
                "layerdrop_rate": self.config.layerdrop_rate,
                "stochastic_depth_rate": self.config.stochastic_depth_rate,
                "moe_load_balancing_coef": self.config.moe_load_balancing_coef,
                "moe_z_loss_coef": self.config.moe_z_loss_coef,
                "weight_decay": self.config.weight_decay,
            },
            "stats": {
                "moe_load_balancing_loss": self.stats.moe_load_balancing_loss,
                "moe_z_loss": self.stats.moe_z_loss,
                "total_reg_loss": self.stats.total_reg_loss,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        config_data = state_dict.get("config", {})
        if config_data:
            self.config.attention_dropout = config_data.get("attention_dropout", 0.0)
            self.config.mlp_dropout = config_data.get("mlp_dropout", 0.0)
            self.config.label_smoothing = config_data.get("label_smoothing", 0.0)
            self.config.layerdrop_rate = config_data.get("layerdrop_rate", 0.0)
            self.config.stochastic_depth_rate = config_data.get("stochastic_depth_rate", 0.0)
            self.config.moe_load_balancing_coef = config_data.get("moe_load_balancing_coef", 0.01)
            self.config.weight_decay = config_data.get("weight_decay", 0.1)

        logger.info("正则化器状态已恢复")

    def print_regularization_summary(self):
        summary = [
            "=" * 60,
            "青小团正则化 - 正则化信息摘要",
            "=" * 60,
            f"Attention Dropout: {self.config.attention_dropout}",
            f"MLP Dropout: {self.config.mlp_dropout}",
            f"Label Smoothing: {self.config.label_smoothing}",
            f"LayerDrop: {self.config.layerdrop_rate:.4f} (decay={self.config.layerdrop_decay})",
            f"Stochastic Depth: {self.config.stochastic_depth_rate:.4f}",
            f"---",
            f"MoE负载均衡系数: {self.config.moe_load_balancing_coef}",
            f"MoE Z-loss系数: {self.config.moe_z_loss_coef}",
            f"MoE重要性系数: {self.config.moe_importance_coef}",
            f"权重衰减: {self.config.weight_decay}",
            f"---",
            f"LayerDrop激活次数: {self.stats.layerdrop_active_count}",
            f"Stochastic Depth激活次数: {self.stats.stochastic_depth_active_count}",
            f"MoE负载均衡loss: {self.stats.moe_load_balancing_loss:.6f}",
            f"MoE Z-loss: {self.stats.moe_z_loss:.6f}",
            f"正则化总loss: {self.stats.total_reg_loss:.6f}",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
