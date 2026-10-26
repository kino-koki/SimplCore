#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""运行图灵测试基准测试"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from SimplCore.engine.tensor import manual_seed, randn
from SimplCore.cognition.human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode
from SimplCore.evaluation.turing_test_eval import TuringTestBenchmark

manual_seed(42)
dim = 64

print("=== 图灵测试基准测试 - HumanLikeCognitiveReasoner ===")

hlcr = HumanLikeCognitiveReasoner(
    dim=dim,
    mode=HumanLikeMode.CONVERSATION,
    enable_nlu=True,
    enable_disambiguation=True,
    enable_dialogue_tracking=True,
    enable_long_range_memory=True,
    enable_emotional_intelligence=True,
    enable_response_variety=True,
    enable_reasoning=True,
)

def model_wrapper(x):
    return hlcr(x)

benchmark = TuringTestBenchmark(dim)
result = benchmark.run_benchmark(model_wrapper)

print("\n=== 基准测试结果汇总 ===")
print(f"总体平均得分: {result['overall_average_percentage']:.1f}%")
print(f"基准测试状态: {result['status']}")
print(f"评估器状态: {result['evaluator_status']}")

report = benchmark.evaluator.generate_report()
print("\n详细指标平均:")
for metric, avg in report['metric_averages'].items():
    print(f"  {metric}: {avg:.2f}/10")

print(f"\n目标: 100% | 当前: {result['overall_average_percentage']:.1f}%")
print(f"差距: {100 - result['overall_average_percentage']:.1f}%")

