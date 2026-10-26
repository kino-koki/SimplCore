# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor NN Utils — 梯度裁剪工具（自研，零框架依赖）
兼容 torch.nn.utils.clip_grad_norm_ / clip_grad_value_
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Union

import numpy as np

from ..tensor import Tensor


def _collect_params(parameters) -> list:
    """收集参数中的 Tensor（兼容 Parameter / Tensor / Module.parameters()）。"""
    params: list = []
    for p in parameters:
        if isinstance(p, Tensor):
            params.append(p)
        elif hasattr(p, "grad") and isinstance(getattr(p, "grad", None), Tensor):
            params.append(p)
        elif hasattr(p, "parameters"):
            # 递归 Module
            params.extend(_collect_params(p.parameters()))
    return params


def clip_grad_norm_(
    parameters: Iterable[object],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """按范数裁剪梯度，兼容 torch.nn.utils.clip_grad_norm_。

    Args:
        parameters: 参数迭代器（Parameter / Module / Tensor）
        max_norm:   梯度总范数上限
        norm_type:  范数类型（默认 2.0，支持 1.0 / inf）

    Returns:
        裁剪前的梯度总范数（float）
    """
    params = [p for p in _collect_params(parameters) if p.grad is not None]
    if len(params) == 0:
        return 0.0

    if norm_type == math.inf:
        total_norm = max(np.asarray(p.grad.data).abs().max() for p in params)
        total_norm = float(total_norm)
    else:
        total_norm = 0.0
        for p in params:
            g = np.asarray(p.grad.data).astype(np.float64)
            total_norm += float(np.sum(np.abs(g) ** norm_type))
        total_norm = float(total_norm ** (1.0 / norm_type))

    if total_norm > max_norm > 0:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in params:
            g = np.asarray(p.grad.data)
            p.grad = Tensor(g * clip_coef, requires_grad=False)

    return float(total_norm)


def clip_grad_value_(
    parameters: Iterable[object],
    clip_value: float,
) -> None:
    """按值裁剪梯度到 [-clip_value, clip_value]，兼容 torch.nn.utils.clip_grad_value_。

    Args:
        parameters: 参数迭代器
        clip_value: 梯度绝对值上限
    """
    params = [p for p in _collect_params(parameters) if p.grad is not None]
    for p in params:
        data = p.grad.data
        p.grad.data = data.clip(-clip_value, clip_value)


__all__ = ["clip_grad_norm_", "clip_grad_value_"]
