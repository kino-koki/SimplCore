# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 里程碑规格。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass(frozen=True)
class SCMilestoneSpec:
    """T0 目标门槛。

    该类不声称当前已经达到 T0，而是把从当前 T3+ 研究架构迈向 T0 所需
    的工程与实验门槛写成可测试契约。
    """

    current_tier: str = "T3+"
    target_tier: str = "T0"
    required_gates: List[str] = field(
        default_factory=lambda: [
            "150M完整训练",
            "7B参数堆叠校准",
            "中文Benchmark评测",
            "消融实验",
            "Java26高性能热路径",
            "Python3.11生态训练编排",
            "长上下文稳定性验证",
            "推理一致性与权重导出验证",
        ]
    )

    @classmethod
    def default(cls) -> "SCMilestoneSpec":
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["SCMilestoneSpec"]
