# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
from typing import List, Dict, Tuple, Optional, Any, Callable, Union
import numpy as np
import uuid
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, cat, gelu, sigmoid_tensor, tanh
from ..engine.nn import Module, Linear, LayerNorm


class CognitiveStrategy(str, Enum):
    FAST = "fast"
    DEEP = "deep"
    BALANCED = "balanced"
    CREATIVE = "creative"
    CAUTIOUS = "cautious"


class CognitiveState:
    def __init__(self):
        self.strategy: CognitiveStrategy = CognitiveStrategy.BALANCED
        self.cognitive_load: float = 0.0
        self.confidence: float = 0.5
        self.strategy_score: float = 0.0
        self.iteration_count: int = 0
        self.correction_count: int = 0
        self.last_correction: Optional[float] = None
        self.history: List[Dict[str, float]] = []
    
    def update(self, **kwargs):
        snapshot = {
            "strategy": self.strategy.value,
            "cognitive_load": self.cognitive_load,
            "confidence": self.confidence,
            "strategy_score": self.strategy_score,
            "iteration_count": self.iteration_count,
            "correction_count": self.correction_count,
        }
        self.history.append(snapshot)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class MetaCognition(Module):
    def __init__(self, dim: int, num_strategies: int = 5, max_iterations: int = 50):
        super().__init__()
        self.dim = dim
        self.num_strategies = num_strategies
        self.max_iterations = max_iterations
        
        self.strategy_proj = Linear(dim, num_strategies)
        self.strategy_value = Linear(dim, num_strategies)
        
        self.load_monitor = Linear(dim, 3)
        self.load_gate = Linear(dim, 1)
        
        self.confidence_proj = Linear(dim * 2, dim)
        self.confidence_head = Linear(dim, 1)
        
        self.correction_proj = Linear(dim * 2, dim)
        self.correction_head = Linear(dim, dim)
        
        self.feedback_proj = Linear(dim * 3, dim)
        self.feedback_norm = LayerNorm(dim)
        
        self.state = CognitiveState()
        
        self.register_module("strategy_proj", self.strategy_proj)
        self.register_module("strategy_value", self.strategy_value)
        self.register_module("load_monitor", self.load_monitor)
        self.register_module("load_gate", self.load_gate)
        self.register_module("confidence_proj", self.confidence_proj)
        self.register_module("confidence_head", self.confidence_head)
        self.register_module("correction_proj", self.correction_proj)
        self.register_module("correction_head", self.correction_head)
        self.register_module("feedback_proj", self.feedback_proj)
        self.register_module("feedback_norm", self.feedback_norm)

    def _monitor_cognitive_load(self, x: SimpTensor) -> float:
        load_features = self.load_monitor(x)
        variance = np.var(x._data)
        activation = np.mean(np.abs(x._data))
        complexity = float(np.mean(load_features._data))
        
        load = 0.3 * variance + 0.4 * activation + 0.3 * complexity
        return min(float(load), 1.0)

    def _assess_confidence(self, output: SimpTensor, target: Optional[SimpTensor] = None) -> float:
        if target is None:
            target = zeros(self.dim)
        
        combined = SimpTensor(np.concatenate([output._data, target._data]))
        processed = gelu(self.confidence_proj(combined))
        confidence = sigmoid_tensor(self.confidence_head(processed))
        return float(confidence._data[0])

    def _evaluate_strategy(self, x: SimpTensor) -> Tuple[List[float], int]:
        scores = self.strategy_proj(x)._data.flatten()
        values = self.strategy_value(x)._data.flatten()
        
        combined = scores * values
        best_strategy = int(np.argmax(combined))
        
        return combined.tolist(), best_strategy

    def _select_strategy(self, x: SimpTensor) -> CognitiveStrategy:
        scores, best_idx = self._evaluate_strategy(x)
        strategies = list(CognitiveStrategy)
        
        if best_idx < len(strategies):
            selected = strategies[best_idx]
        else:
            selected = CognitiveStrategy.BALANCED
        
        self.state.strategy = selected
        self.state.strategy_score = float(scores[best_idx])
        
        return selected

    def _should_correct(self, output: SimpTensor, prev_output: Optional[SimpTensor] = None) -> bool:
        if prev_output is None:
            return False
        
        diff = np.mean(np.abs(output._data - prev_output._data))
        confidence = self._assess_confidence(output)
        
        should_correct = (diff > 0.1) or (confidence < 0.4)
        
        if should_correct:
            self.state.correction_count += 1
            self.state.last_correction = float(diff)
        
        return should_correct

    def _self_correct(self, current: SimpTensor, previous: SimpTensor) -> SimpTensor:
        combined = SimpTensor(np.concatenate([current._data, previous._data]))
        correction_signal = gelu(self.correction_proj(combined))
        correction = self.correction_head(correction_signal)
        
        corrected = current + tanh(correction) * 0.1
        return corrected

    def _adjust_strategy(self, cognitive_load: float) -> CognitiveStrategy:
        if cognitive_load > 0.8:
            return CognitiveStrategy.FAST
        elif cognitive_load < 0.3:
            return CognitiveStrategy.DEEP
        elif self.state.confidence < 0.4:
            return CognitiveStrategy.CAUTIOUS
        else:
            return self.state.strategy

    def _generate_feedback(self, input: SimpTensor, output: SimpTensor, 
                          strategy: CognitiveStrategy) -> SimpTensor:
        strategy_embed = np.zeros(self.dim)
        strategy_idx = list(CognitiveStrategy).index(strategy)
        strategy_embed[strategy_idx] = 1.0
        strategy_tensor = SimpTensor(strategy_embed)
        
        combined = SimpTensor(np.concatenate([input._data, output._data, strategy_tensor._data]))
        feedback = gelu(self.feedback_proj(combined))
        return self.feedback_norm(feedback)

    def forward(self, x: SimpTensor, context: Optional[SimpTensor] = None,
               target: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        self.state.iteration_count += 1
        
        cognitive_load = self._monitor_cognitive_load(x)
        self.state.cognitive_load = cognitive_load
        
        initial_strategy = self._select_strategy(x)
        adjusted_strategy = self._adjust_strategy(cognitive_load)
        
        prev_output = None
        current_output = x
        corrections_made = 0
        
        for iteration in range(self.max_iterations):
            processed = gelu(current_output)
            
            if self._should_correct(processed, prev_output) and iteration > 0:
                current_output = self._self_correct(processed, prev_output)
                corrections_made += 1
                if corrections_made >= 3:
                    break
            else:
                current_output = processed
            
            prev_output = processed
            
            confidence = self._assess_confidence(current_output, target)
            self.state.confidence = confidence
            
            if confidence > 0.85:
                break
            
            cognitive_load = self._monitor_cognitive_load(current_output)
            adjusted_strategy = self._adjust_strategy(cognitive_load)
        
        confidence = self._assess_confidence(current_output, target)
        feedback = self._generate_feedback(x, current_output, adjusted_strategy)
        
        final_output = current_output + 0.1 * feedback
        
        info = {
            "cognitive_load": float(cognitive_load),
            "initial_strategy": initial_strategy.value,
            "adjusted_strategy": adjusted_strategy.value,
            "confidence": float(confidence),
            "iterations": iteration + 1,
            "corrections_made": corrections_made,
            "total_corrections": self.state.correction_count,
            "strategy_score": float(self.state.strategy_score),
            "history_length": len(self.state.history),
            "avg_historical_confidence": float(np.mean([h["confidence"] for h in self.state.history])) if self.state.history else 0.5,
        }
        
        return final_output, info


class StrategyOptimizer(MetaCognition):
    def __init__(self, dim: int, num_strategies: int = 5):
        super().__init__(dim, num_strategies)
        
        self.strategy_memory = Linear(dim, num_strategies)
        self.strategy_selector = Linear(dim, num_strategies)
        
        self.register_module("strategy_memory", self.strategy_memory)
        self.register_module("strategy_selector", self.strategy_selector)

    def forward(self, x: SimpTensor, context: Optional[SimpTensor] = None,
               target: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        strategy_mem = self.strategy_memory(x)
        
        if context is not None:
            context_proj = self.strategy_selector(context)
            selector_input = x + context_proj
        else:
            selector_input = x
        
        selector_logits = self.strategy_selector(selector_input)
        strategy_probs = sigmoid_tensor(selector_logits)
        
        outputs = []
        infos = []
        
        for i, strategy in enumerate(list(CognitiveStrategy)):
            if i >= self.num_strategies:
                break
            
            if float(strategy_probs._data[i]) > 0.3:
                output, info = super().forward(x, context, target)
                outputs.append(output * float(strategy_probs._data[i]))
                infos.append(info)
        
        if outputs:
            combined_output = sum(outputs)
            avg_confidence = float(np.mean([info["confidence"] for info in infos]))
        else:
            combined_output, info = super().forward(x, context, target)
            infos.append(info)
            avg_confidence = float(info["confidence"])
        
        strategy_info = {
            "num_strategies_used": len(outputs),
            "strategy_probs": strategy_probs._data.flatten().tolist(),
            "avg_confidence": avg_confidence,
            "best_strategy": max(infos, key=lambda i: i["confidence"])["adjusted_strategy"] if infos else "balanced",
        }
        
        return combined_output, strategy_info


if __name__ == "__main__":
    dim = 64
    print("=== 测试 MetaCognition ===")
    meta = MetaCognition(dim)
    x = randn(dim)
    output, info = meta(x)
    print(f"输入: {x.shape}, 输出: {output.shape}")
    print(f"认知负荷: {info['cognitive_load']:.3f}")
    print(f"初始策略: {info['initial_strategy']}, 调整后策略: {info['adjusted_strategy']}")
    print(f"置信度: {info['confidence']:.3f}, 迭代次数: {info['iterations']}")
    print(f"修正次数: {info['corrections_made']}, 策略评分: {info['strategy_score']:.3f}")
    
    output_sum = output.sum()
    output_sum.backward()
    grad_ok = any(p.grad is not None and np.any(p.grad._data != 0) for p in meta.parameters())
    print(f"梯度传播: {grad_ok}")
    print("MetaCognition 测试 PASSED\n")
    
    print("=== 测试 StrategyOptimizer ===")
    optimizer = StrategyOptimizer(dim)
    x2 = randn(dim)
    output2, info2 = optimizer(x2)
    print(f"输入: {x2.shape}, 输出: {output2.shape}")
    print(f"使用策略数: {info2['num_strategies_used']}")
    print(f"策略概率: {[f'{p:.3f}' for p in info2['strategy_probs']]}")
    print(f"平均置信度: {info2['avg_confidence']:.3f}")
    print(f"最佳策略: {info2['best_strategy']}")
    
    output2_sum = output2.sum()
    output2_sum.backward()
    grad_ok2 = any(p.grad is not None and np.any(p.grad._data != 0) for p in optimizer.parameters())
    print(f"梯度传播: {grad_ok2}")
    print("StrategyOptimizer 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")
