# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor NN — 神经网络模块兼容层（自研 NumPy 引擎，零第三方框架依赖）
=================================================================================
所有类均由 ``simplcore.tensor.module`` 原生实现，提供与 PyTorch ``nn`` 兼容的接口。
"""
from __future__ import annotations

from ..module import (
    Parameter,
    Module,
    ModuleList,
    Sequential,
    Linear,
    Embedding,
    Dropout,
    RMSNorm,
    ReLU,
    Sigmoid,
    LayerNorm,
    Functional,
    init,
)

import numpy as np
from ..tensor import Tensor

from . import utils
from . import parallel

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        from ..tensor import tanh
        return tanh(x)

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from ..tensor import gelu
        return gelu(x)

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from ..tensor import silu
        return silu(x)

class Identity(Module):
    def forward(self, x):
        return x

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)

class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        from ..tensor import softmax
        return softmax(x, dim=self.dim)

class LogSoftmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        from ..tensor import log_softmax
        return log_softmax(x, dim=self.dim)

class MSELoss(Module):
    def forward(self, input: Tensor, target) -> Tensor:
        from ..tensor import mse_loss
        return mse_loss(input, target)

class CrossEntropyLoss(Module):
    def forward(self, input: Tensor, target) -> Tensor:
        from ..tensor import cross_entropy
        return cross_entropy(input, target)

class BCELoss(Module):
    def forward(self, input: Tensor, target) -> Tensor:
        from ..tensor import binary_cross_entropy
        return binary_cross_entropy(input, target)

class L1Loss(Module):
    def forward(self, input: Tensor, target) -> Tensor:
        from ..tensor import l1_loss
        return l1_loss(input, target)


__all__ = [
    "Parameter", "Module", "ModuleList", "Sequential",
    "Linear", "Embedding", "Dropout", "RMSNorm",
    "ReLU", "Sigmoid", "Tanh", "GELU", "SiLU",
    "Identity", "Flatten", "Softmax", "LogSoftmax",
    "LayerNorm", "MSELoss", "CrossEntropyLoss",
    "BCELoss", "L1Loss", "Functional", "init",
]
