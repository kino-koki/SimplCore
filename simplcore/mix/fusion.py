# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 深度融合组件。

这些模块承接 Java SimplCore 中的 CIN、IB、MRD、PUM 语义，并以
SimplCore Mix 的 ``SCModule`` 形式接入循环推理主干。

SGI (StructuredGatedIntervention) 是本包的核心创新组件 —
在表征空间实现结构化因果推理，包含分组有向注意力、门控父节点替换、
上下文残差减除三大机制。详见 cin.py。
"""
from __future__ import annotations

import simplcore.tensor as torch

from .core import SCFunctional, SCLinear, SCModule, SCParameter
from .cin import StructuredGatedIntervention


class SimplCoreGatedIntervention(SCModule):
    """CIN 因果干预网络 (轻量版, 向后兼容)。

    新项目请使用 StructuredGatedIntervention (深度结构化因果干预)。
    """
    def __init__(self, dim: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.confounder = SCLinear(dim, dim, bias=False)
        self.intervention = SCLinear(dim, dim, bias=False)
        self.gate = SCLinear(dim, 1, bias=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        source = context if context is not None else x
        confounder = torch.tanh(self.confounder(source))
        correction = torch.tanh(self.intervention(x - confounder))
        gate = torch.sigmoid(self.gate(x))
        return x + gate * correction * self.scale


class GatedCompression(SCModule):
    """IB 门控压缩。

    通过窄通道压缩再展开，过滤循环状态中的噪声并保留主表征。
    """

    def __init__(self, dim: int, bottleneck_dim: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.down = SCLinear(dim, bottleneck_dim, bias=False)
        self.up = SCLinear(bottleneck_dim, dim, bias=False)
        self.gate = SCLinear(dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = SCFunctional.silu(self.down(x))
        restored = self.up(compressed)
        gate = torch.sigmoid(self.gate(x))
        return x + gate * restored * self.scale


class LowRankProjection(SCModule):
    """MRD 低秩特征投影。

    将当前 token 表征投影到低维元表示，再蒸馏回主隐藏空间。
    """

    def __init__(self, dim: int, meta_dim: int, scale: float = 0.05):
        super().__init__()
        self.scale = scale
        self.to_meta = SCLinear(dim, meta_dim, bias=False)
        self.from_meta = SCLinear(meta_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        meta = SCFunctional.silu(self.to_meta(x))
        distilled = self.from_meta(meta)
        return x + distilled * self.scale


class ResidualAdaptationModule(SCModule):
    """PUM 残差适应模块。

    提供稳定的轻量可塑性通道，和 RecurrentBlock 里已有 SPM 形成互补。
    """

    def __init__(self, dim: int, scale: float = 0.02):
        super().__init__()
        self.scale = scale
        self.adapt_matrix = SCParameter(torch.eye(dim) * 0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        plastic = torch.matmul(x, self.adapt_matrix.T)
        return x + plastic * self.scale


__all__ = [
    "SimplCoreGatedIntervention",  # 轻量版 (向后兼容)
    "StructuredGatedIntervention",        # SGI 结构化门控干预 (推荐)
    "GatedCompression",
    "LowRankProjection",
    "ResidualAdaptationModule",
]
