# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 运行时画像。

该模块把 Python 3.11 生态侧与 Java26 高性能侧写成明确、可测试的
运行时契约，供训练、导出和 Java 互操作使用。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SCRuntimeProfile:
    """SimplCore Mix 跨运行时目标。"""

    python_target: str = "3.11"
    java_target: str = "26"
    target_tier: str = "T0"
    ecosystem_mode: str = "python_ecosystem"
    performance_mode: str = "java26_hotpath"
    backends: List[str] = field(
        default_factory=lambda: [
            "simplcore_native",
            "simplcore_java26",
        ]
    )
    notes: Dict[str, str] = field(
        default_factory=lambda: {
            "python": "用于数据生态、实验脚本、训练编排和原型验证",
            "java26": "用于高性能推理、权重装载、热路径计算和服务化部署",
            "tier": "目标剑指T0，当前通过训练、评测和堆叠逐步逼近",
        }
    )

    @classmethod
    def default(cls) -> "SCRuntimeProfile":
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


__all__ = ["SCRuntimeProfile"]
