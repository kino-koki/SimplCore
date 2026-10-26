# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore 训练模块 (Training Module).

提供面向 72B 大模型的分布式训练框架:
  - TrainingConfig:  统一训练配置数据类
  - SimplCoreTrainer: 训练循环、评估、Checkpoint、故障恢复与稳定性监控
"""

from SimplCore.training.trainer import (
    SimplCoreTrainer,
    TrainingConfig,
)

__all__ = [
    "TrainingConfig",
    "SimplCoreTrainer",
]


