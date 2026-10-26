"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

training/advanced/ - 高级训练模块
包含混合精度训练、梯度优化、通信优化、内存优化、学习率调度、
优化器封装、数据加载、初始化、正则化、课程学习、日志配置等工具
"""

from .mixed_precision_trainer import QingxiaotuanMixedPrecisionTrainer
from .gradient_optimizer import QingxiaotuanGradientOptimizer
from .communication_optimizer import QingxiaotuanCommunicationOptimizer
from .memory_optimizer import QingxiaotuanMemoryOptimizer
from .lr_scheduler import QingxiaotuanLRScheduler
from .optimizer_wrapper import QingxiaotuanOptimizerWrapper
from .data_loader import QingxiaotuanDataLoader
from .initialization import QingxiaotuanInitializer
from .regularization import QingxiaotuanRegularizer
from .curriculum_learning import QingxiaotuanCurriculumLearning
from .logging_config import QingxiaotuanLoggingConfig

__all__ = [
    "QingxiaotuanMixedPrecisionTrainer",
    "QingxiaotuanGradientOptimizer",
    "QingxiaotuanCommunicationOptimizer",
    "QingxiaotuanMemoryOptimizer",
    "QingxiaotuanLRScheduler",
    "QingxiaotuanOptimizerWrapper",
    "QingxiaotuanDataLoader",
    "QingxiaotuanInitializer",
    "QingxiaotuanRegularizer",
    "QingxiaotuanCurriculumLearning",
    "QingxiaotuanLoggingConfig",
]

__version__ = "1.0.0"
__author__ = "青奈科技"
__model__ = "青小团 70B MoE"
