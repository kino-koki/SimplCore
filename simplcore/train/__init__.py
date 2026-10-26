# -*- coding: utf-8 -*-
"""SimplCore · 训练模块.

增强版综合训练系统，支持多模块联合训练和自适应调度.

核心组件:
1. EnhancedTrainer - 增强版综合训练器
2. MultiTaskLoss - 多任务联合损失
3. EnhancedSyntheticDataGenerator - 增强版数据生成器
4. AdaptiveTrainingScheduler - 自适应训练调度器
"""

from SimplCore.train.enhanced_trainer import (
    MultiTaskLoss,
    EnhancedSyntheticDataGenerator,
    EnhancedTrainer,
    AdaptiveTrainingScheduler,
)

__all__ = [
    "MultiTaskLoss",
    "EnhancedSyntheticDataGenerator",
    "EnhancedTrainer",
    "AdaptiveTrainingScheduler",
]

