# -*- coding: utf-8 -*-
"""SimplCore · 评估模块.

图灵测试评估框架，提供全面的AI系统评估能力.

核心组件:
1. TuringTestEvaluator - 图灵测试评估器
2. EnhancedHumanSimilarityEvaluator - 增强版人类相似度评估器
3. TuringScore - 图灵测试评分
4. EvaluationMetric - 评估指标枚举
"""

from SimplCore.evaluation.turing_test_eval import (
    EvaluationMetric,
    TuringScore,
    HumanSimilarityEvaluator,
    TuringTestEvaluator,
)

from SimplCore.evaluation.enhanced_evaluator import (
    TrainableEvaluator,
    EnhancedHumanSimilarityEvaluator,
)

__all__ = [
    "EvaluationMetric",
    "TuringScore",
    "HumanSimilarityEvaluator",
    "TuringTestEvaluator",
    "TrainableEvaluator",
    "EnhancedHumanSimilarityEvaluator",
]

