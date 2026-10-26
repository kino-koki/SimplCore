# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""硬件优化层 (Hardware Optimization Layer).

本模块提供针对 AMD MI250 GPU 及多卡互联场景的深度优化能力, 包括:
  - mi250_optimizer: 针对 CDNA 2 架构 (MI250) 的算子与内存布局优化
  - memory_manager:  多级显存管理 (L1/L2/HBM), 内存池化与碎片控制
  - communication:   GPU 间通信优化 (Infinity Fabric, all_reduce 等)

所有组件均设计为可在无 AMD GPU 的 CPU 环境下导入并基本运行,
通过条件检测自动降级, 保证开发与调试的便利性。
"""

from SimplCore.hardware.mi250_optimizer import (
    MI250Optimizer,
    MI250Specs,
)
from SimplCore.hardware.memory_manager import (
    SimplCoreMemoryManager,
    MemoryStats,
    MemoryPool,
)
from SimplCore.hardware.communication import (
    GPUCommunicator,
    CommunicationStats,
)

__all__ = [
    "MI250Optimizer",
    "MI250Specs",
    "SimplCoreMemoryManager",
    "MemoryStats",
    "MemoryPool",
    "GPUCommunicator",
    "CommunicationStats",
]


