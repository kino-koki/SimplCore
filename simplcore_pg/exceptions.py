# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 生产级异常体系与张量健康度工具。

异常层级：
    SimplCoreError (基类)
      ├── ConfigError
      ├── ModelError
      ├── MemoryStateError
      ├── RolloutError
      ├── GradientCheckpointError
      ├── ModalityError
      ├── StackingError
      └── LossGuardError

另提供：
    * :func:`check_tensor_health` —— 前向/反向中的 NaN/Inf 守卫；
    * :func:`handle_gradient_clipping` —— 带健康检查的 L2 梯度裁剪；
    * :func:`format_exception` —— 统一可读格式化。
"""
from __future__ import annotations

from typing import Optional

import torch


class SimplCoreError(Exception):
    """SimplCore 架构基础异常，捕获此异常可兜底所有架构层错误。"""

    def __init__(self, message: str = "", details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict = dict(details) if details else {}

    def __str__(self) -> str:
        if self.details:
            detail = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail})"
        return self.message


class ConfigError(SimplCoreError):
    """配置错误：参数非法、缺失或类型错误。"""


class ModelError(SimplCoreError):
    """模型运行时错误：维度不匹配、NaN/Inf、非法生成参数等。"""


class MemoryStateError(SimplCoreError):
    """DMC 记忆状态错误：未初始化、形状不匹配、溢出等。"""


class RolloutError(SimplCoreError):
    """潜空间推演错误：步数非法、状态发散、超最大步数等。"""


class GradientCheckpointError(SimplCoreError):
    """梯度检查点错误：段数不匹配、NaN/Inf、爆炸/消失。"""


class ModalityError(SimplCoreError):
    """多模态输入错误：无模态输入、尺寸不匹配、类型不支持等。"""


class StackingError(SimplCoreError):
    """参数堆叠错误：参数量不匹配、层数超限、不兼容。"""


class LossGuardError(SimplCoreError):
    """损失守卫阈值违规：超 max_loss、NaN/Inf、异常波动。"""


def check_tensor_health(
    tensor: torch.Tensor,
    *,
    name: str = "tensor",
    allow_inf: bool = False,
) -> None:
    """检查张量健康度，异常时按类型抛出。

    Args:
        tensor: 待检查张量。
        name: 张量名称 (用于错误详情)。
        allow_inf: 是否允许 Inf (默认不允许)。

    Raises:
        ModelError: 含 NaN 或 (不允许的) Inf。
    """
    if not torch.is_tensor(tensor):
        raise ModelError(f"{name} 不是 torch.Tensor。", details={"type": type(tensor).__name__})
    if torch.isnan(tensor).any():
        n = int(torch.isnan(tensor).sum().item())
        raise ModelError(f"{name} 含 NaN。", details={"nan_count": n, "shape": tuple(tensor.shape)})
    if not allow_inf and torch.isinf(tensor).any():
        n = int(torch.isinf(tensor).sum().item())
        raise ModelError(f"{name} 含 Inf。", details={"inf_count": n, "shape": tuple(tensor.shape)})


def handle_gradient_clipping(
    grad: Optional[torch.Tensor],
    max_norm: float = 1.0,
) -> Optional[torch.Tensor]:
    """带健康检查的 L2 梯度裁剪。

    Args:
        grad: 梯度张量 (可为 None)。
        max_norm: 最大梯度范数。

    Returns:
        裁剪后的梯度；输入为 None 时返回 None。

    Raises:
        GradientCheckpointError: 梯度含 NaN/Inf 或 max_norm 非法。
    """
    if grad is None:
        return None
    if max_norm <= 0 or not torch.isfinite(torch.tensor(max_norm)):
        raise GradientCheckpointError(f"max_norm 必须为正有限值，得到 {max_norm}。")

    if torch.isnan(grad).any():
        raise GradientCheckpointError("梯度包含 NaN，无法裁剪。", details={"shape": tuple(grad.shape)})
    if torch.isinf(grad).any():
        raise GradientCheckpointError("梯度包含 Inf，无法裁剪。", details={"shape": tuple(grad.shape)})

    total_norm = grad.norm(p=2).item()
    if total_norm > max_norm and total_norm > 0:
        grad = grad * (max_norm / (total_norm + 1e-6))
    return grad


def format_exception(exc: SimplCoreError) -> str:
    """格式化 SimplCore 异常为可读多行字符串。"""
    lines = [f"[{exc.__class__.__name__}] {exc.message}"]
    for key, value in exc.details.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)
