# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 自研核心抽象层。

本模块定义 SimplCore Mix 对外使用的自研命名接口 (SCModule/SCLinear/...)。
底层后端为青小团自研张量引擎 ``simplcore_native``（纯 NumPy 自动微分实现），
**不依赖任何第三方深度学习框架**（PyTorch / TensorFlow 已被彻底移除）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# ── 从自研张量引擎导入，替代 torch ──────────────────────────
import simplcore.tensor as torch
from simplcore.tensor import nn as _hidden_nn
from simplcore.tensor.nn import functional as _hidden_functional


@dataclass(frozen=True)
class SCBackend:
    """SimplCore Mix 后端描述。"""

    name: str
    family: str
    training_capable: bool = True

    @classmethod
    def default(cls) -> "SCBackend":
        # 青小团自研张量引擎（纯原生，零第三方框架依赖）。
        return cls(name="simplcore_native", family="SimplCore")


class SCModule(_hidden_nn.Module):
    """SimplCore Mix 模块基类。

    公开模型统一继承该类；底层后端为自研张量引擎。
    """

    simplcore_backend = SCBackend.default()

    def sc_backend_name(self) -> str:
        return self.simplcore_backend.name


class SCParameter(_hidden_nn.Parameter):
    """SimplCore Mix 参数类型。"""


class SCLinear(_hidden_nn.Linear):
    """SimplCore Mix 线性层。"""

    simplcore_backend = SCBackend.default()


class SCEmbedding(_hidden_nn.Embedding):
    """SimplCore Mix 嵌入层。"""

    simplcore_backend = SCBackend.default()


class SCDropout(_hidden_nn.Dropout):
    """SimplCore Mix Dropout 层。"""

    simplcore_backend = SCBackend.default()


class SCModuleList(_hidden_nn.ModuleList):
    """SimplCore Mix 模块列表。"""

    simplcore_backend = SCBackend.default()


class SCSequential(_hidden_nn.Sequential):
    """SimplCore Mix 顺序容器。"""

    simplcore_backend = SCBackend.default()


class SCReLU(_hidden_nn.ReLU):
    """SimplCore Mix ReLU 层。"""

    simplcore_backend = SCBackend.default()


class SCSigmoid(_hidden_nn.Sigmoid):
    """SimplCore Mix Sigmoid 层。"""

    simplcore_backend = SCBackend.default()


class SCRMSNorm(SCModule):
    """SimplCore Mix RMSNorm。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = SCParameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SCFunctional:
    """SimplCore Mix 函数式算子命名空间。"""

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return _hidden_functional.silu(x)

    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        return _hidden_functional.relu(x)

    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return _hidden_functional.softmax(x, dim=dim)

    @staticmethod
    def normalize(x: torch.Tensor, p: float = 2.0, dim: int = -1) -> torch.Tensor:
        return _hidden_functional.normalize(x, p=p, dim=dim)

    @staticmethod
    def dropout(x: torch.Tensor, p: float = 0.0, training: bool = False) -> torch.Tensor:
        return _hidden_functional.dropout(x, p=p, training=training)

    @staticmethod
    def cross_entropy(input: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return _hidden_functional.cross_entropy(input, target, **kwargs)


__all__ = [
    "SCBackend",
    "SCDropout",
    "SCEmbedding",
    "SCFunctional",
    "SCLinear",
    "SCModule",
    "SCModuleList",
    "SCParameter",
    "SCReLU",
    "SCRMSNorm",
    "SCSequential",
    "SCSigmoid",
]
