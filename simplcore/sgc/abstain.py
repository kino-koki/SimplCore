# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""L4 拒答门 (abstention gate) —— 四层闸门的输出侧闸门。

四层闸门总览（见 ``docs/ARCHITECTURE_PIVOT.md`` §4）::

    L1 选择门   selective    mixer 输入    按内容选择参与位置/通道
    L2 证据门   evidence     mixer 谱基    前缀能量不足时压制长程路径
    L3 不确定门 uncertainty   mixer 输出    长短路径分歧大时回退到局部证据
    L4 拒答门   abstention   输出头        没有外部证据支撑时拒绝作答

L1~L3 作用在 mixer 内部（``simplcore/sgc/mixer.py``），L4 作用在输出侧：
从最终隐状态预测"该位置的回答是否被证据支撑"，低于阈值即拒答。

设计要点
--------
* **不引入 softmax 注意力**：L4 只是一个 ``Linear(dim -> 1)`` 的标量头。
* **默认不拒答**：偏置初始化为 -2.0（sigmoid≈0.12），未训练时几乎不会触发拒答，
  避免未训练模型行为突变。
* **监督契约**：``grounded`` 标签必须来自**外部证据**（引文、检索命中、
  工具返回值、人工标注），不能由模型自身生成 —— 自监督的"拒答标签"会退化成
  另一个自信度估计，四层闸门就白做了。
* **诚实边界**：四层闸门是**降低幻觉倾向的结构性约束**，不构成"消除幻觉"的保证。
  必须用带引用的评测集实测后才能宣称有效。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstentionHead(nn.Module):
    """Predicts whether the answer at each position is supported by evidence.

    Args:
        dim: hidden width.
        bias_init: initial bias of the logit. -2.0 gives sigmoid(-2) ~= 0.12,
            i.e. an untrained model almost never abstains.
    """

    def __init__(self, dim: int, bias_init: float = -2.0) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.full((1,), float(bias_init)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Args: hidden (B, L, D). Returns: logit (B, L)."""
        return (self.proj(hidden) * torch.exp(self.logit_scale) + self.bias).squeeze(-1)

    def extra_repr(self) -> str:
        return f"dim={self.proj.in_features}, bias={float(self.bias):.2f}"


def abstention_loss(
    grounded_logit: torch.Tensor,
    grounded: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy for the grounded/abstain decision.

    Args:
        grounded_logit: (B, L) raw logits from :class:`AbstentionHead`.
        grounded: (B, L) float targets in [0, 1]. 1 = the answer at that
            position **is** supported by external evidence, 0 = it is not.
            Must come from external evidence, never from the model itself.
        weight: optional per-element loss weight.
        pos_weight: optional positive-class weight (for class imbalance).

    Returns:
        Scalar loss.
    """
    target = grounded.to(grounded_logit.dtype)
    if target.shape != grounded_logit.shape:
        raise ValueError(
            f"grounded shape {tuple(target.shape)} != logit shape "
            f"{tuple(grounded_logit.shape)}"
        )
    return F.binary_cross_entropy_with_logits(
        grounded_logit, target, weight=weight, pos_weight=pos_weight
    )


def grounded_probability(grounded_logit: torch.Tensor) -> torch.Tensor:
    """Sigmoid of the grounded logit: probability that the answer is supported."""
    return torch.sigmoid(grounded_logit)


def should_abstain(grounded_logit: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Boolean mask of positions that should be refused.

    A position abstains when the probability that it is grounded falls below
    ``threshold``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    return grounded_probability(grounded_logit) < threshold


def calibrate_temperature(logit: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling for confidence calibration (Guo et al., 2017).

    ``temperature < 1`` sharpens, ``> 1`` softens. Used at inference to
    correct over/under-confidence of the grounded head before thresholding.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logit / temperature
