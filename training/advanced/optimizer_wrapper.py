"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

optimizer_wrapper.py - 优化器封装模块
包装多种优化器（AdamW, Lion, Adafactor, SophiaG）、
参数分组（不同层不同学习率）、MoE参数特殊处理、
Embedding层特殊处理、Head层特殊处理、权重衰减分组
"""

import os
import sys
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.OptimizerWrapper")


class OptimizerType(Enum):
    ADAMW = "adamw"
    LION = "lion"
    ADAFACTOR = "adafactor"
    SOPHIAG = "sophiag"


@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    moe_lr_scale: float = 0.5
    moe_weight_decay: float = 0.05

    embedding_lr_scale: float = 1.0
    embedding_weight_decay: float = 0.0

    head_lr_scale: float = 1.5
    head_weight_decay: float = 0.0

    layerwise_lr_decay: float = 1.0
    num_hidden_layers: int = 80

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    lion_weight_decay: float = 0.0

    adafactor_decay_rate: float = -0.8
    adafactor_beta1: float = 0.0

    sophiag_rho: float = 0.03
    sophiag_beta: float = 0.5


class QingxiaotuanOptimizerWrapper:
    """
    青小团优化器封装器
    支持 AdamW / Lion / Adafactor / SophiaG 等多种优化器、
    参数分组（不同层不同学习率）、MoE/Embedding/Head特殊处理、
    权重衰减分组

    针对70B MoE模型，实现精细化的参数分组和优化策略
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OptimizerConfig] = None,
    ):
        self.model = model
        self.config = config or OptimizerConfig()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.param_groups_info: Dict[str, Any] = {}
        self._moe_params: List[Dict[str, Any]] = []
        self._embedding_params: List[Dict[str, Any]] = []
        self._head_params: List[Dict[str, Any]] = []
        self._normal_params: List[Dict[str, Any]] = []
        self._weight_decay_params: List[Dict[str, Any]] = []
        self._no_weight_decay_params: List[Dict[str, Any]] = []

        self._build_param_groups()
        self._create_optimizer()

    def _build_param_groups(self):
        self._classify_parameters()

        num_layers = self.config.num_hidden_layers
        params_with_lr = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            lr_scale = 1.0
            wd = self.config.weight_decay

            if any(k in name.lower() for k in ["expert", "moe", "gate", "router"]):
                lr_scale = self.config.moe_lr_scale
                wd = self.config.moe_weight_decay
                self._moe_params.append({"params": param, "lr_scale": lr_scale, "wd": wd})
                continue

            if "embed" in name.lower() or "embedding" in name.lower():
                lr_scale = self.config.embedding_lr_scale
                wd = self.config.embedding_weight_decay
                self._embedding_params.append({"params": param, "lr_scale": lr_scale, "wd": wd})
                continue

            if "head" in name.lower() or "lm_head" in name.lower() or "output_projection" in name.lower():
                lr_scale = self.config.head_lr_scale
                wd = self.config.head_weight_decay
                self._head_params.append({"params": param, "lr_scale": lr_scale, "wd": wd})
                continue

            if self.config.layerwise_lr_decay < 1.0:
                layer_idx = self._extract_layer_idx(name)
                if layer_idx >= 0:
                    lr_scale = self.config.layerwise_lr_decay ** (num_layers - 1 - layer_idx)

            self._normal_params.append({"params": param, "lr_scale": lr_scale, "wd": wd})

        logger.info(f"参数分组完成: MoE={len(self._moe_params)}, "
                    f"Embedding={len(self._embedding_params)}, "
                    f"Head={len(self._head_params)}, "
                    f"Normal={len(self._normal_params)}")

    def _classify_parameters(self):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2 and "weight" in name:
                self._weight_decay_params.append(name)
            else:
                self._no_weight_decay_params.append(name)

    def _extract_layer_idx(self, name: str) -> int:
        import re
        match = re.search(r'\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        match = re.search(r'layers\.(\d+)', name)
        if match:
            return int(match.group(1))
        return -1

    def _create_optimizer(self):
        param_groups = []

        def add_group(params_list, lr_scale, wd, group_name):
            if not params_list:
                return
            params = [p["params"] for p in params_list]
            effective_lr = self.config.learning_rate * lr_scale
            group = {
                "params": params,
                "lr": effective_lr,
                "weight_decay": wd,
                "name": group_name,
            }
            param_groups.append(group)
            self.param_groups_info[group_name] = {
                "lr": effective_lr,
                "weight_decay": wd,
                "num_params": sum(p.numel() for p in params),
            }

        add_group(self._moe_params, self.config.moe_lr_scale, self.config.moe_weight_decay, "moe")
        add_group(self._embedding_params, self.config.embedding_lr_scale, self.config.embedding_weight_decay, "embedding")
        add_group(self._head_params, self.config.head_lr_scale, self.config.head_weight_decay, "head")
        add_group(self._normal_params, 1.0, self.config.weight_decay, "normal")

        if self.config.optimizer_type == OptimizerType.ADAMW:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == OptimizerType.LION:
            self.optimizer = self._create_lion_optimizer(param_groups)
        elif self.config.optimizer_type == OptimizerType.ADAFACTOR:
            self.optimizer = self._create_adafactor_optimizer(param_groups)
        elif self.config.optimizer_type == OptimizerType.SOPHIAG:
            self.optimizer = self._create_sophiag_optimizer(param_groups)
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )

        logger.info(f"优化器创建完成: {self.config.optimizer_type.value}, "
                    f"param_groups={len(param_groups)}, "
                    f"lr={self.config.learning_rate:.6e}")

    def _create_lion_optimizer(self, param_groups: List) -> torch.optim.Optimizer:
        class Lion(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)

            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                for group in self.param_groups:
                    beta1, beta2 = group["betas"]
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad.data
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p.data)
                        exp_avg = state["exp_avg"]
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        update = exp_avg.sign()
                        if group["weight_decay"] > 0:
                            p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                        p.data.add_(update, alpha=-group["lr"])
                return loss

        return Lion(param_groups, lr=self.config.learning_rate,
                    betas=(self.config.lion_beta1, self.config.lion_beta2),
                    weight_decay=self.config.lion_weight_decay)

    def _create_adafactor_optimizer(self, param_groups: List) -> torch.optim.Optimizer:
        class Adafactor(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-3, decay_rate=-0.8, beta1=0.0, eps=1e-8):
                defaults = dict(lr=lr, decay_rate=decay_rate, beta1=beta1, eps=eps)
                super().__init__(params, defaults)

            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad.data
                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p.data)
                            state["exp_avg_sq"] = torch.zeros_like(p.data)
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        state["step"] += 1
                        beta1 = group["beta1"]
                        decay_rate = group["decay_rate"]
                        eps = group["eps"]
                        lr = group["lr"]
                        if beta1 > 0:
                            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        else:
                            exp_avg = grad
                        exp_avg_sq.mul_(decay_rate).addcmul_(grad, grad, value=1.0 - decay_rate)
                        denom = exp_avg_sq.sqrt().add_(eps)
                        if group["weight_decay"] > 0:
                            p.data.mul_(1.0 - lr * group["weight_decay"])
                        p.data.addcdiv_(exp_avg, denom, value=-lr)
                return loss

        return Adafactor(param_groups, lr=self.config.learning_rate,
                         decay_rate=self.config.adafactor_decay_rate,
                         beta1=self.config.adafactor_beta1,
                         eps=self.config.epsilon)

    def _create_sophiag_optimizer(self, param_groups: List) -> torch.optim.Optimizer:
        class SophiaG(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), rho=0.03, beta=0.5, eps=1e-8):
                defaults = dict(lr=lr, betas=betas, rho=rho, beta=beta, eps=eps)
                super().__init__(params, defaults)

            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                for group in self.param_groups:
                    beta1, beta2 = group["betas"]
                    rho = group["rho"]
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad.data
                        state = self.state[p]
                        if len(state) == 0:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p.data)
                            state["exp_avg_sq"] = torch.zeros_like(p.data)
                            state["hessian"] = torch.zeros_like(p.data)
                        exp_avg = state["exp_avg"]
                        exp_avg_sq = state["exp_avg_sq"]
                        hessian = state["hessian"]
                        state["step"] += 1
                        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        if state["step"] % 10 == 0:
                            hessian.mul_(0.5).add_(grad.abs(), alpha=0.5)
                        denom = (exp_avg_sq + group["eps"]).sqrt()
                        update = exp_avg / denom
                        if group["weight_decay"] > 0:
                            p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                        p.data.add_(update, alpha=-group["lr"])
                return loss

        return SophiaG(param_groups, lr=self.config.learning_rate,
                       betas=(self.config.sophiag_rho, self.config.sophiag_beta),
                       rho=self.config.sophiag_rho,
                       beta=self.config.sophiag_beta,
                       eps=self.config.epsilon)

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_optimizer(self) -> torch.optim.Optimizer:
        return self.optimizer

    def get_param_groups(self) -> List[Dict]:
        if self.optimizer is None:
            return []
        return self.optimizer.param_groups

    def get_learning_rate(self) -> float:
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]

    def set_learning_rate(self, lr: float):
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    def set_param_group_lr(self, group_name: str, lr: float):
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group.get("name") == group_name:
                group["lr"] = lr
                logger.info(f"参数组 '{group_name}' 学习率已设置为 {lr:.6e}")

    def get_param_group_info(self) -> Dict[str, Dict[str, Any]]:
        return self.param_groups_info

    def state_dict(self) -> Dict[str, Any]:
        if self.optimizer is None:
            return {}
        return {
            "optimizer_type": self.config.optimizer_type.value,
            "optimizer": self.optimizer.state_dict(),
            "param_groups_info": self.param_groups_info,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.optimizer is None or "optimizer" not in state_dict:
            return
        self.optimizer.load_state_dict(state_dict["optimizer"])
        info = state_dict.get("param_groups_info", {})
        if info:
            self.param_groups_info = info
        logger.info("优化器状态已恢复")

    def switch_optimizer(self, optimizer_type: OptimizerType):
        if optimizer_type == self.config.optimizer_type:
            return
        old_state = self.state_dict() if self.optimizer else {}
        self.config.optimizer_type = optimizer_type
        self._create_optimizer()
        if old_state and "optimizer" in old_state:
            try:
                self.optimizer.load_state_dict(old_state["optimizer"])
            except Exception as e:
                logger.warning(f"优化器状态迁移失败: {e}")
        logger.info(f"优化器已切换为: {optimizer_type.value}")

    def log_param_groups(self):
        info = [
            "=" * 60,
            "青小团优化器 - 参数组信息",
            "=" * 60,
        ]
        for name, group_info in self.param_groups_info.items():
            info.append(f"  {name}: lr={group_info['lr']:.6e}, "
                        f"wd={group_info['weight_decay']:.4f}, "
                        f"params={group_info['num_params']:,}")
        info.append("=" * 60)
        for line in info:
            logger.info(line)
        return "\n".join(info)
