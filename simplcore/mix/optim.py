# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 优化器抽象层。"""
from __future__ import annotations

from typing import Iterable

import simplcore.tensor as torch


class SCOptimizer:
    """SimplCore Mix 优化器基类。

    当前委托隐藏训练后端，后续可切换到自研 autograd 或 Java 训练后端。
    """

    backend_name = "simplcore_native"


class SCAdamW(torch.optim.AdamW, SCOptimizer):
    """SimplCore Mix AdamW 优化器。"""

    def __init__(self, params: Iterable, lr: float = 1e-3, **kwargs):
        super().__init__(params, lr=lr, **kwargs)


__all__ = ["SCOptimizer", "SCAdamW"]
