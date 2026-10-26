# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Optimizer — 青小团自研优化器
========================================
提供与 ``torch.optim`` 等价的核心优化器，包括 AdamW、SGD、Adam。

所有优化器操作 ``Tensor.grad``（``numpy.ndarray``），
直接更新 ``Tensor.data``，不依赖 PyTorch。

使用示例
--------
    from simplcore.tensor import Tensor, Linear, AdamW

    model = Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    x = Tensor([[0.1] * 10])
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""
from __future__ import annotations

import math
import numpy as np
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

from .tensor import Tensor


# ============================================================
# Optimizer 基类
# ============================================================
class Optimizer:
    """所有优化器的基类。

    Parameters
    ----------
    params : iterable of Tensor
        可迭代的参数列表。
    defaults : dict
        默认超参数。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        defaults: dict,
    ) -> None:
        self.param_groups: List[dict] = []
        self.defaults: dict = dict(defaults)
        self.state: dict = {}

        # 将参数组织为参数组
        params_list = list(params)
        if len(params_list) == 0:
            raise ValueError("优化器收到空参数列表")
        self.param_groups.append({"params": params_list, **self.defaults})

    # ------------------------------------------------------------
    # 参数管理
    # ------------------------------------------------------------
    def add_param_group(self, param_group: dict) -> None:
        """添加新的参数组。"""
        if "params" not in param_group:
            raise ValueError("param_group 必须包含 'params' 键")
        for key, val in self.defaults.items():
            param_group.setdefault(key, val)
        self.param_groups.append(param_group)

    @property
    def param_groups_list(self) -> List[Tensor]:
        """展开所有参数为扁平列表。"""
        result: List[Tensor] = []
        for group in self.param_groups:
            result.extend(group["params"])
        return result

    # ------------------------------------------------------------
    # 梯度清零
    # ------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零所有参数的梯度。"""
        for group in self.param_groups:
            for p in group["params"]:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad = np.zeros_like(p.data)

    # ------------------------------------------------------------
    # 梯度裁剪
    # ------------------------------------------------------------
    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> float:
        """梯度范数裁剪。"""
        total_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if norm_type == float("inf"):
                    total_norm = max(total_norm, float(np.max(np.abs(p.grad))))
                else:
                    total_norm += float(np.sum(p.grad ** norm_type))
        total_norm = total_norm ** (1.0 / norm_type)

        if max_norm > 0 and total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.grad = p.grad * clip_coef

        return total_norm

    def clip_grad_value_(self, clip_value: float) -> None:
        """梯度值裁剪。"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad = np.clip(p.grad, -clip_value, clip_value)

    # ------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------
    def _param_key(self, group_idx: int, param_idx: int) -> str:
        """稳定的状态键：基于 (组索引, 组内位置)，而非 id(p)。

        这样跨进程续训时即使 Python 对象 id 改变也不会丢失动量。
        """
        return f"g{group_idx}.p{param_idx}"

    def state_dict(self) -> dict:
        """序列化优化器状态（按稳定键存，支持续训）。"""
        state_dict: dict = {
            "param_groups": [],
            "state": {},
        }
        for i, group in enumerate(self.param_groups):
            g = {k: v for k, v in group.items() if k != "params"}
            state_dict["param_groups"].append(g)
        for key, val in self.state.items():
            if isinstance(val, np.ndarray):
                state_dict["state"][key] = val.copy()
            elif isinstance(val, dict):
                state_dict["state"][key] = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in val.items()
                }
            else:
                state_dict["state"][key] = val
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """加载优化器状态（按稳定键还原）。"""
        self.state = {}
        for key, val in state_dict.get("state", {}).items():
            self.state[key] = val

    # ------------------------------------------------------------
    # step（子类实现）
    # ------------------------------------------------------------
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """执行一步参数更新。"""
        raise NotImplementedError("子类必须实现 step()")


# ============================================================
# AdamW 优化器
# ============================================================
class AdamW(Optimizer):
    """AdamW 优化器（解耦权重衰减）。

    Parameters
    ----------
    params : iterable of Tensor
    lr : float
        学习率（默认 1e-3）。
    betas : Tuple[float, float]
        一阶/二阶动量衰减系数（默认 (0.9, 0.999)）。
    eps : float
        分母稳定项（默认 1e-8）。
    weight_decay : float
        权重衰减系数（默认 0.01）。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr 必须为正数，得到 {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 必须在 [0, 1) 内，得到 {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 必须在 [0, 1) 内，得到 {betas[1]}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for gi, group in enumerate(self.param_groups):
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for pi, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.dtype != p.data.dtype:
                    grad = grad.astype(p.data.dtype)

                # 状态键（稳定，基于组/位置，支持续训）
                key = self._param_key(gi, pi)
                if key not in self.state:
                    self.state[key] = {
                        "step": 0,
                        "m": np.zeros_like(p.data),  # 一阶动量
                        "v": np.zeros_like(p.data),  # 二阶动量
                    }
                state = self.state[key]
                m = state["m"]
                v = state["v"]
                state["step"] += 1
                t = state["step"]

                # 解耦权重衰减（先衰减，再更新动量）
                if wd > 0:
                    p.data = p.data * (1.0 - lr * wd)

                # 一阶动量
                m[...] = beta1 * m + (1.0 - beta1) * grad
                # 二阶动量
                v[...] = beta2 * v + (1.0 - beta2) * (grad * grad)

                # 偏差校正
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                # 参数更新
                p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)

        return loss


# ============================================================
# Adam 优化器（带耦合权重衰减）
# ============================================================
class Adam(Optimizer):
    """Adam 优化器（L2 正则化耦合在梯度中）。

    Parameters
    ----------
    params : iterable of Tensor
    lr : float
    betas : Tuple[float, float]
    eps : float
    weight_decay : float
        L2 正则化系数。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for gi, group in enumerate(self.param_groups):
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for pi, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.astype(p.data.dtype)

                # 耦合权重衰减（加到梯度上）
                if wd > 0:
                    grad = grad + wd * p.data

                key = self._param_key(gi, pi)
                if key not in self.state:
                    self.state[key] = {
                        "step": 0,
                        "m": np.zeros_like(p.data),
                        "v": np.zeros_like(p.data),
                    }
                state = self.state[key]
                m = state["m"]
                v = state["v"]
                state["step"] += 1
                t = state["step"]

                m[...] = beta1 * m + (1.0 - beta1) * grad
                v[...] = beta2 * v + (1.0 - beta2) * (grad * grad)

                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                p.data = p.data - lr * m_hat / (np.sqrt(v_hat) + eps)

        return loss


# ============================================================
# SGD 优化器
# ============================================================
class SGD(Optimizer):
    """随机梯度下降（带动量）。

    Parameters
    ----------
    params : iterable of Tensor
    lr : float
        学习率。
    momentum : float
        动量系数（默认 0.0）。
    dampening : float
        动量阻尼（默认 0.0）。
    weight_decay : float
        权重衰减（默认 0.0）。
    nesterov : bool
        是否使用 Nesterov 动量。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov 动量需要正的 momentum 且 dampening=0")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()

        for gi, group in enumerate(self.param_groups):
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]

            for pi, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad.astype(p.data.dtype)

                if wd > 0:
                    grad = grad + wd * p.data

                if momentum > 0:
                    key = self._param_key(gi, pi)
                    if key not in self.state:
                        self.state[key] = {"momentum_buffer": grad.copy()}
                    buf = self.state[key]["momentum_buffer"]
                    buf[...] = momentum * buf + (1.0 - dampening) * grad
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf

                p.data = p.data - lr * grad

        return loss


# ============================================================
# 学习率调度器（简易版）
# ============================================================
class _LRScheduler:
    """学习率调度器基类。"""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step()

    def get_lr(self) -> List[float]:
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr


class CosineAnnealingLR(_LRScheduler):
    """余弦退火学习率调度。"""

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class LinearLR(_LRScheduler):
    """线性学习率调度。"""

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
    ) -> None:
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        if self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        factor = (
            (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
            + self.start_factor
        )
        return [base_lr * factor for base_lr in self.base_lrs]


class LambdaLR(_LRScheduler):
    """Lambda 学习率调度。"""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float],
    ) -> None:
        self.lr_lambda = lr_lambda
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        return [base_lr * self.lr_lambda(self.last_epoch) for base_lr in self.base_lrs]


class SequentialLR(_LRScheduler):
    """顺序组合多个调度器（兼容 torch.optim.lr_scheduler.SequentialLR）。

    在 milestones 指定的 epoch 处切换到下一个调度器。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: List[_LRScheduler],
        milestones: List[int],
    ) -> None:
        self.schedulers = schedulers
        self.milestones = milestones
        super().__init__(optimizer)

    def _active_idx(self) -> int:
        idx = 0
        for m in self.milestones:
            if self.last_epoch >= m:
                idx += 1
        return min(idx, len(self.schedulers) - 1)

    def get_lr(self) -> List[float]:
        return self.schedulers[self._active_idx()].get_lr()

    def step(self, epoch: Optional[int] = None) -> None:
        # 同步推进所有子调度器
        super().step(epoch)
        for s in self.schedulers:
            if hasattr(s, "last_epoch"):
                s.last_epoch = self.last_epoch
