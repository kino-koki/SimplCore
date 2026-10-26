"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

initialization.py - 参数初始化模块
多种初始化方法（正态/均匀/xavier/kaiming）、DeepNet初始化、
MoE初始化（路由和专家）、Embedding初始化、Head初始化、初始化验证
"""

import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

logger = logging.getLogger("青小团.Initializer")


class InitMethod(Enum):
    NORMAL = "normal"
    UNIFORM = "uniform"
    XAVIER_NORMAL = "xavier_normal"
    XAVIER_UNIFORM = "xavier_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    DEEPNET = "deepnet"
    MOE_ROUTER = "moe_router"
    MOE_EXPERT = "moe_expert"
    EMBEDDING = "embedding"
    HEAD = "head"


@dataclass
class InitConfig:
    default_method: InitMethod = InitMethod.NORMAL
    mean: float = 0.0
    std: float = 0.02
    uniform_bound: float = 0.02
    xavier_gain: float = 1.0
    kaiming_gain: float = 1.0
    deepnet_alpha: float = 0.1
    deepnet_beta: float = 1.0
    num_layers: int = 80
    moe_router_std: float = 0.01
    moe_expert_std: float = 0.02
    embedding_std: float = 0.01
    head_std: float = 0.01
    use_deepnet: bool = True
    init_checkpoint_enabled: bool = True


@dataclass
class InitStats:
    initialized_modules: int = 0
    initialized_params: int = 0
    mean_value: float = 0.0
    std_value: float = 0.0
    max_value: float = 0.0
    min_value: float = 0.0
    nan_count: int = 0
    inf_count: int = 0


class QingxiaotuanInitializer:
    """
    青小团参数初始化器
    支持正态/均匀/Xavier/Kaiming/DeepNet初始化、
    MoE路由和专家初始化、Embedding/Head初始化、初始化验证

    针对70B MoE深层模型采用DeepNet初始化策略确保训练稳定
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[InitConfig] = None,
    ):
        self.model = model
        self.config = config or InitConfig()
        self.stats = InitStats()
        self._init_methods: Dict[str, InitMethod] = {}

    def initialize_all(self):
        self._assign_init_methods()
        self._apply_initialization()
        self._apply_deepnet_rescaling()
        self._validate_initialization()
        logger.info(f"参数初始化完成: {self.stats.initialized_modules} 模块, "
                    f"{self.stats.initialized_params:,} 参数")

    def _assign_init_methods(self):
        for name, module in self.model.named_modules():
            if any(k in name.lower() for k in ["router", "gate", "moe_gate"]):
                self._init_methods[name] = InitMethod.MOE_ROUTER
            elif any(k in name.lower() for k in ["expert", "moe_expert", "ffn_expert"]):
                self._init_methods[name] = InitMethod.MOE_EXPERT
            elif "embed" in name.lower() or "embedding" in name.lower():
                self._init_methods[name] = InitMethod.EMBEDDING
            elif "head" in name.lower() or "lm_head" in name.lower() or "output_projection" in name.lower():
                self._init_methods[name] = InitMethod.HEAD

    def _apply_initialization(self):
        for name, module in self.model.named_modules():
            method = self._init_methods.get(name, self.config.default_method)
            self._init_module(module, method, name)

    def _init_module(self, module: nn.Module, method: InitMethod, name: str):
        for param_name, param in module.named_parameters(recurse=False):
            if "weight" in param_name:
                self._init_weight(param, method, name + "." + param_name)
            elif "bias" in param_name:
                if method != InitMethod.MOE_ROUTER:
                    init.zeros_(param)
                self.stats.initialized_params += param.numel()

    def _init_weight(self, weight: torch.Tensor, method: InitMethod, full_name: str):
        dim = weight.dim()

        if method == InitMethod.MOE_ROUTER:
            init.normal_(weight, mean=0.0, std=self.config.moe_router_std)
            if dim >= 2:
                with torch.no_grad():
                    weight.data = weight.data / (weight.data.norm(dim=-1, keepdim=True) + 1e-8)

        elif method == InitMethod.MOE_EXPERT:
            if self.config.use_deepnet:
                self._deepnet_init(weight, self.config.num_layers)
            else:
                init.normal_(weight, mean=self.config.mean, std=self.config.moe_expert_std)

        elif method == InitMethod.EMBEDDING:
            init.normal_(weight, mean=0.0, std=self.config.embedding_std)

        elif method == InitMethod.HEAD:
            init.normal_(weight, mean=0.0, std=self.config.head_std)

        elif method == InitMethod.DEEPNET:
            self._deepnet_init(weight, self.config.num_layers)

        elif method == InitMethod.XAVIER_NORMAL:
            init.xavier_normal_(weight, gain=self.config.xavier_gain)

        elif method == InitMethod.XAVIER_UNIFORM:
            init.xavier_uniform_(weight, gain=self.config.xavier_gain)

        elif method == InitMethod.KAIMING_NORMAL:
            init.kaiming_normal_(weight, a=0, mode="fan_in", nonlinearity="relu")

        elif method == InitMethod.KAIMING_UNIFORM:
            init.kaiming_uniform_(weight, a=0, mode="fan_in", nonlinearity="relu")

        elif method == InitMethod.UNIFORM:
            init.uniform_(weight, -self.config.uniform_bound, self.config.uniform_bound)

        else:
            init.normal_(weight, mean=self.config.mean, std=self.config.std)

        self.stats.initialized_modules += 1
        self.stats.initialized_params += weight.numel()

    def _deepnet_init(self, weight: torch.Tensor, num_layers: int):
        if weight.dim() < 2:
            init.normal_(weight, mean=0.0, std=self.config.std)
            return

        fan_in = weight.shape[1] if weight.dim() >= 2 else 1
        alpha = self.config.deepnet_alpha
        beta = self.config.deepnet_beta

        std = beta * alpha / math.sqrt(2 * num_layers * fan_in)
        init.normal_(weight, mean=0.0, std=std)

    def _apply_deepnet_rescaling(self):
        if not self.config.use_deepnet:
            return

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                for param_name, param in module.named_parameters(recurse=False):
                    if "weight" in param_name:
                        with torch.no_grad():
                            param.data.mul_(self.config.deepnet_alpha)

        logger.info(f"DeepNet初始化缩放已应用: alpha={self.config.deepnet_alpha}")

    def _validate_initialization(self):
        all_params = []
        for p in self.model.parameters():
            all_params.append(p.data.flatten())
        if not all_params:
            return
        all_params = torch.cat(all_params)

        self.stats.mean_value = all_params.mean().item()
        self.stats.std_value = all_params.std().item()
        self.stats.max_value = all_params.max().item()
        self.stats.min_value = all_params.min().item()
        self.stats.nan_count = torch.isnan(all_params).sum().item()
        self.stats.inf_count = torch.isinf(all_params).sum().item()

        if self.stats.nan_count > 0 or self.stats.inf_count > 0:
            logger.error(f"初始化检测到 NaN={self.stats.nan_count}, Inf={self.stats.inf_count}")
            if self.config.init_checkpoint_enabled:
                logger.warning("初始化存在异常值，建议检查初始化参数")

    def initialize_module(self, module_name: str, method: InitMethod):
        for name, module in self.model.named_modules():
            if name == module_name:
                self._init_module(module, method, name)
                logger.info(f"模块 {module_name} 已使用 {method.value} 重新初始化")
                return
        logger.warning(f"未找到模块: {module_name}")

    def initialize_layer(self, layer_idx: int, method: InitMethod):
        for name, module in self.model.named_modules():
            if f".{layer_idx}." in name or f"layers.{layer_idx}" in name:
                self._init_module(module, method, name)
        logger.info(f"第 {layer_idx} 层已使用 {method.value} 重新初始化")

    def get_init_stats(self) -> InitStats:
        return self.stats

    def set_init_method(self, method: InitMethod):
        self.config.default_method = method
        logger.info(f"默认初始化方法已设置为: {method.value}")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "default_method": self.config.default_method.value,
                "mean": self.config.mean,
                "std": self.config.std,
                "use_deepnet": self.config.use_deepnet,
                "deepnet_alpha": self.config.deepnet_alpha,
                "num_layers": self.config.num_layers,
            },
            "stats": {
                "initialized_modules": self.stats.initialized_modules,
                "initialized_params": self.stats.initialized_params,
                "mean_value": self.stats.mean_value,
                "std_value": self.stats.std_value,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        config_data = state_dict.get("config", {})
        if config_data:
            self.config.default_method = InitMethod(config_data.get("default_method", "normal"))
            self.config.use_deepnet = config_data.get("use_deepnet", True)
            self.config.deepnet_alpha = config_data.get("deepnet_alpha", 0.1)

        stats_data = state_dict.get("stats", {})
        if stats_data:
            self.stats.initialized_modules = stats_data.get("initialized_modules", 0)
            self.stats.initialized_params = stats_data.get("initialized_params", 0)

        logger.info("初始化器状态已恢复")

    def print_init_summary(self):
        summary = [
            "=" * 60,
            "青小团参数初始化 - 初始化信息摘要",
            "=" * 60,
            f"默认方法: {self.config.default_method.value}",
            f"DeepNet初始化: {'启用' if self.config.use_deepnet else '禁用'}",
            f"DeepNet alpha: {self.config.deepnet_alpha}",
            f"初始化标准差: {self.config.std}",
            f"---",
            f"初始化模块数: {self.stats.initialized_modules}",
            f"初始化参数数: {self.stats.initialized_params:,}",
            f"参数均值: {self.stats.mean_value:.6e}",
            f"参数标准差: {self.stats.std_value:.6e}",
            f"参数最大值: {self.stats.max_value:.6e}",
            f"参数最小值: {self.stats.min_value:.6e}",
            f"NaN数量: {self.stats.nan_count}",
            f"Inf数量: {self.stats.inf_count}",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
