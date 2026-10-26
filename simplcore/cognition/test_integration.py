#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
综合集成测试 - 验证所有认知模块的端到端梯度传播和功能
"""
import numpy as np

from ..engine.tensor import SimpTensor, randn, zeros
from ..engine.nn import Module, Linear, LayerNorm

from .tree_of_thought import TreeOfThought
from .reflexion import ReflexionModule
from .associative_reasoner import AssociativeReasoner
from .emotional_engine import EmotionalDecisionEngine
from .knowledge_iteration import KnowledgeSelfIteration
from .nonlinear_thinking import NonLinearThinkingChain, ParallelThinkingChain
from .metacognition import MetaCognition, StrategyOptimizer
from .analogical_reasoner import AnalogicalReasoner
from .causal_reasoner import CausalReasoner, CounterfactualReasoner
from .cognitive_reasoner import CognitiveReasoner, ReasoningMode


def test_gradient_flow(module: Module, x: SimpTensor, name: str) -> bool:
    """测试梯度是否正常传播"""
    output = module(x)
    if isinstance(output, tuple):
        output = output[0]
    
    loss = output.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in module.parameters())
    print(f"  梯度传播: {has_grad}")
    
    return has_grad


def test_all_modules(dim: int = 64):
    """测试所有认知模块"""
    print(f"=== 综合集成测试 (维度={dim}) ===\n")
    
    x = randn(dim)
    context = randn(dim)
    
    modules = [
        ("NonLinearThinkingChain", NonLinearThinkingChain(dim)),
        ("ParallelThinkingChain", ParallelThinkingChain(dim)),
        ("MetaCognition", MetaCognition(dim)),
        ("StrategyOptimizer", StrategyOptimizer(dim)),
        ("AnalogicalReasoner", AnalogicalReasoner(dim)),
        ("CausalReasoner", CausalReasoner(dim)),
        ("CounterfactualReasoner", CounterfactualReasoner(dim)),
        ("TreeOfThought", TreeOfThought(dim)),
        ("ReflexionModule", ReflexionModule(dim)),
        ("AssociativeReasoner", AssociativeReasoner(dim)),
        ("EmotionalDecisionEngine", EmotionalDecisionEngine(dim)),
        ("KnowledgeSelfIteration", KnowledgeSelfIteration(dim)),
    ]
    
    passed = []
    failed = []
    
    for name, module in modules:
        print(f"测试 {name}:")
        try:
            if hasattr(module, 'add_knowledge'):
                module.add_knowledge("test", randn(dim))
            
            if name == "AnalogicalReasoner":
                output = module(x, context)
            else:
                output = module(x)
            
            if isinstance(output, tuple):
                output = output[0]
            
            loss = output.sum()
            loss.backward()
            
            has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in module.parameters())
            print(f"  梯度传播: {has_grad}")
            
            success = has_grad
            if success:
                passed.append(name)
            else:
                failed.append(name)
            print(f"  {name} {'PASSED' if success else 'FAILED'}\n")
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(name)
            print(f"  {name} FAILED\n")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {len(passed)}/{len(modules)}")
    print(f"失败: {len(failed)}/{len(modules)}")
    if passed:
        print(f"通过列表: {passed}")
    if failed:
        print(f"失败列表: {failed}")
    
    return len(failed) == 0


def test_cognitive_reasoner_all_modes(dim: int = 64):
    """测试CognitiveReasoner的所有推理模式"""
    print(f"\n=== 测试 CognitiveReasoner 所有推理模式 ===")
    
    cr = CognitiveReasoner(dim)
    x = randn(dim)
    
    modes = list(ReasoningMode)
    passed = []
    failed = []
    
    for mode in modes:
        print(f"\n推理模式: {mode.value}")
        try:
            cr.reasoning_mode = mode
            output, info = cr(x)
            
            loss = output.sum()
            loss.backward()
            
            has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in cr.parameters())
            
            print(f"  输入: {x.shape}, 输出: {output.shape}")
            print(f"  置信度: {info.get('confidence', 'N/A'):.3f}" if isinstance(info.get('confidence'), (float, np.floating)) else f"  置信度: {info.get('confidence', 'N/A')}")
            print(f"  梯度传播: {has_grad}")
            
            if has_grad:
                passed.append(mode.value)
            else:
                failed.append(mode.value)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(mode.value)
    
    print(f"\n=== 推理模式测试结果 ===")
    print(f"通过: {len(passed)}/{len(modes)}")
    print(f"失败: {len(failed)}/{len(modes)}")
    print(f"通过模式: {passed}")
    if failed:
        print(f"失败模式: {failed}")
    
    return len(failed) == 0


def test_end_to_end(dim: int = 64):
    """端到端测试 - 模拟完整认知推理流程"""
    print(f"\n=== 端到端认知推理流程测试 ===")
    
    cr = CognitiveReasoner(dim)
    
    cr.add_knowledge("数学知识", randn(dim), {"category": "math"})
    cr.add_knowledge("科学知识", randn(dim), {"category": "science"})
    cr.add_knowledge("历史知识", randn(dim), {"category": "history"})
    
    test_cases = [
        {"name": "简单推理", "mode": ReasoningMode.FAST, "knowledge_mode": "enhance"},
        {"name": "深度推理", "mode": ReasoningMode.DEEP, "knowledge_mode": "multi_hop"},
        {"name": "非线性推理", "mode": ReasoningMode.NONLINEAR, "knowledge_mode": "enhance"},
        {"name": "元认知推理", "mode": ReasoningMode.META, "knowledge_mode": "enhance"},
        {"name": "类比推理", "mode": ReasoningMode.ANALOGICAL, "knowledge_mode": "enhance"},
        {"name": "因果推理", "mode": ReasoningMode.CAUSAL, "knowledge_mode": "enhance"},
    ]
    
    all_passed = True
    
    for tc in test_cases:
        print(f"\n{tc['name']}:")
        try:
            cr.reasoning_mode = tc["mode"]
            x = randn(dim)
            output, info = cr(x, knowledge_mode=tc["knowledge_mode"])
            
            loss = output.sum()
            loss.backward()
            
            has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in cr.parameters())
            
            print(f"  模式: {tc['mode'].value}")
            print(f"  知识模式: {tc['knowledge_mode']}")
            print(f"  置信度: {info.get('confidence', 'N/A')}")
            print(f"  梯度传播: {has_grad}")
            
            if not has_grad:
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            all_passed = False
    
    status = "PASSED" if all_passed else "FAILED"
    print(f"\n端到端测试: {status}")
    return all_passed


if __name__ == "__main__":
    dim = 64
    
    result1 = test_all_modules(dim)
    result2 = test_cognitive_reasoner_all_modes(dim)
    result3 = test_end_to_end(dim)
    
    print(f"\n{'='*60}")
    print(f"综合测试结果: {'ALL PASSED' if all([result1, result2, result3]) else 'SOME FAILED'}")
    print(f"{'='*60}")
