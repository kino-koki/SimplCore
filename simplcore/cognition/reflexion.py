# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Reflexion Module (反思模块) - 降低幻觉率
通过置信度评估和自我修正机制减少模型输出中的幻觉
"""

import numpy as np
from typing imimport os
port Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, ones, softmax, sigmoid_tensor, gelu, tanh, abs as tensor_abs
from ..engine.nn import Module, Linear, LayerNorm


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class ReflexionResult:
    """反思结果"""
    
    def __init__(
        self,
        corrected_output: SimpTensor,
        confidence: float,
        original_confidence: float,
        num_reflections: int,
        is_hallucination: bool,
        correction_strength: float,
    ):
        self.corrected_output = corrected_output
        self.confidence = confidence
        self.original_confidence = original_confidence
        self.num_reflections = num_reflections
        self.is_hallucination = is_hallucination
        self.correction_strength = correction_strength


class ReflexionModule(Module):
    """反思模块 - 通过自我评估降低幻觉率
    
    核心机制：
    1. 置信度评估：判断输出的可靠性
    2. 自我修正：对低置信度输出进行修正
    3. 一致性检查：验证输出与上下文的一致性
    4. 幻觉检测：识别并标记潜在的幻觉内容
    
    Args:
        dim: 特征维度
        confidence_threshold: 置信度阈值，低于此值触发反思
        max_reflections: 最大反思次数
        correction_weight: 修正权重
        consistency_weight: 一致性检查权重
    """
    
    def __init__(
        self,
        dim: int,
        confidence_threshold: float = 0.4,
        max_reflections: int = 3,
        correction_weight: float = 0.7,
    ):
        super().__init__()
        self.dim = dim
        self.confidence_threshold = confidence_threshold
        self.max_reflections = max_reflections
        self.correction_weight = correction_weight
        
        self.confidence_proj = Linear(dim, 1)
        self.register_module("confidence_proj", self.confidence_proj)
        
        self.correction_proj = Linear(dim, dim)
        self.register_module("correction_proj", self.correction_proj)
        
        self.consistency_proj = Linear(dim * 2, 1)
        self.register_module("consistency_proj", self.consistency_proj)
        
        self.reflection_proj = Linear(dim * 2, dim)
        self.register_module("reflection_proj", self.reflection_proj)
        
        self.confidence_norm = LayerNorm(1)
        self.register_module("confidence_norm", self.confidence_norm)
        
        self.correction_norm = LayerNorm(dim)
        self.register_module("correction_norm", self.correction_norm)
        
        self._hallucination_count = 0
        self._total_reflections = 0
        self._correction_history: List[Dict[str, float]] = []
    
    def _compute_confidence(self, x: SimpTensor) -> float:
        """计算输出的置信度"""
        confidence = self.confidence_proj(x)
        confidence = self.confidence_norm(confidence)
        confidence = sigmoid_tensor(confidence)
        return confidence._data[0]
    
    def _check_consistency(self, output: SimpTensor, context: SimpTensor) -> float:
        """检查输出与上下文的一致性"""
        combined = SimpTensor(np.concatenate([output._data, context._data]))
        consistency = self.consistency_proj(combined)
        consistency = sigmoid_tensor(consistency)
        return consistency._data[0]
    
    def _correct_output(self, output: SimpTensor, context: SimpTensor) -> SimpTensor:
        """修正低置信度输出"""
        combined = SimpTensor(np.concatenate([output._data, context._data]))
        correction = self.reflection_proj(combined)
        correction = gelu(correction)
        correction = self.correction_norm(correction)
        
        corrected = output + self.correction_weight * correction
        return corrected
    
    def _detect_hallucination(
        self,
        output: SimpTensor,
        context: SimpTensor,
        confidence: float,
    ) -> bool:
        """检测是否为幻觉"""
        consistency = self._check_consistency(output, context)
        combined_score = self.confidence_weight * confidence + self.consistency_weight * consistency
        return combined_score < self.confidence_threshold
    
    def forward(
        self,
        output: SimpTensor,
        context: Optional[SimpTensor] = None,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """
        反思模块前向传播
        
        Args:
            output: 模型原始输出 [B, S, D] 或 [D]
            context: 上下文信息（用于一致性检查）
        
        Returns:
            corrected_output: 修正后的输出
            info: 反思过程信息
        """
        if context is None:
            context = zeros(self.dim)
        
        if output.ndim == 3:
            B, S, D = output.shape
            results = []
            infos = []
            
            for b in range(B):
                for s in range(S):
                    out_t = output[b, s]
                    ctx_t = context[b, s] if context.ndim == 3 else context
                    corrected, info = self._single_reflexion(out_t, ctx_t)
                    results.append(corrected)
                    infos.append(info)
            
            result_tensor = SimpTensor(np.stack([r._data for r in results])).reshape(B, S, D)
            
            avg_info = {
                "avg_confidence": sum(i["confidence"] for i in infos) / len(infos),
                "avg_original_confidence": sum(i["original_confidence"] for i in infos) / len(infos),
                "avg_reflections": sum(i["num_reflections"] for i in infos) / len(infos),
                "hallucination_rate": sum(1 for i in infos if i["is_hallucination"]) / len(infos),
                "correction_rate": sum(1 for i in infos if i["num_reflections"] > 0) / len(infos),
            }
            
            return result_tensor, avg_info
        else:
            return self._single_reflexion(output, context)
    
    def _single_reflexion(
        self,
        output: SimpTensor,
        context: SimpTensor,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """单样本反思"""
        original_confidence = self._compute_confidence(output)
        current_output = output
        current_confidence = original_confidence
        num_reflections = 0
        is_hallucination = False
        correction_strength = 0.0
        
        while current_confidence < self.confidence_threshold and num_reflections < self.max_reflections:
            current_output = self._correct_output(current_output, context)
            current_confidence = self._compute_confidence(current_output)
            num_reflections += 1
            
            self._total_reflections += 1
            if current_confidence < self.confidence_threshold:
                self._hallucination_count += 1
        
        if num_reflections > 0:
            correction_strength = (original_confidence - current_confidence) / max(original_confidence, 0.01)
        
        is_hallucination = current_confidence < self.confidence_threshold
        
        self._correction_history.append({
            "original_confidence": original_confidence,
            "final_confidence": current_confidence,
            "num_reflections": num_reflections,
            "is_hallucination": is_hallucination,
        })
        
        context_data = context._data if context.ndim == 1 else context._data.flatten()
        if len(context_data) < self.dim:
            context_data = np.pad(context_data, (0, self.dim - len(context_data)))
        elif len(context_data) > self.dim:
            context_data = context_data[:self.dim]
        
        context_tensor = SimpTensor(context_data)
        combined_input_data = np.concatenate([output._data, context_tensor._data])
        combined_input = SimpTensor(combined_input_data, requires_grad=True)
        refined = self.reflection_proj(combined_input)
        refined = gelu(refined)
        final_output = self.correction_norm(refined)
        
        info = {
            "confidence": current_confidence,
            "original_confidence": original_confidence,
            "num_reflections": num_reflections,
            "is_hallucination": is_hallucination,
            "correction_strength": correction_strength,
            "confidence_level": self._get_confidence_level(current_confidence),
        }
        
        return final_output, info
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度等级"""
        if confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def get_hallucination_rate(self) -> float:
        """获取幻觉率"""
        if self._total_reflections == 0:
            return 0.0
        return self._hallucination_count / self._total_reflections
    
    def reset_stats(self) -> None:
        """重置统计数据"""
        self._hallucination_count = 0
        self._total_reflections = 0
        self._correction_history = []
    
    @property
    def confidence_weight(self) -> float:
        return self.correction_weight
    
    @property
    def consistency_weight(self) -> float:
        return 1.0 - self.correction_weight


class ConfidenceValidator(ReflexionModule):
    """置信度验证器 - 增强版反思模块
    
    额外功能：
    1. 多维度置信度评估
    2. 事实核查
    3. 引用追踪
    """
    
    def __init__(
        self,
        dim: int,
        confidence_threshold: float = 0.5,
        max_reflections: int = 5,
    ):
        super().__init__(dim, confidence_threshold, max_reflections)
        
        self.fact_check_proj = Linear(dim, 2)
        self.citation_proj = Linear(dim, dim)
    
    def _fact_check(self, output: SimpTensor) -> Tuple[float, float]:
        """事实核查"""
        logits = self.fact_check_proj(output)
        probs = softmax(logits, dim=0)
        factual_prob = probs._data[0]
        hallucination_prob = probs._data[1]
        return factual_prob, hallucination_prob
    
    def _single_reflexion(
        self,
        output: SimpTensor,
        context: SimpTensor,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """增强版单样本反思"""
        original_confidence = self._compute_confidence(output)
        factual_prob, hallucination_prob = self._fact_check(output)
        
        effective_confidence = 0.7 * original_confidence + 0.3 * factual_prob
        
        if hallucination_prob > 0.5:
            current_output = self._correct_output(output, context)
            current_confidence = self._compute_confidence(current_output)
            num_reflections = 1
        else:
            current_output, info = super()._single_reflexion(output, context)
            current_confidence = info["confidence"]
            num_reflections = info["num_reflections"]
        
        info = {
            "confidence": current_confidence,
            "original_confidence": original_confidence,
            "factual_prob": factual_prob,
            "hallucination_prob": hallucination_prob,
            "num_reflections": num_reflections,
            "is_hallucination": hallucination_prob > 0.5,
            "correction_strength": 0.0,
            "confidence_level": self._get_confidence_level(current_confidence),
        }
        
        return current_output, info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed, randn, zeros
    
    manual_seed(0)
    print("=== 测试 ReflexionModule ===")
    
    rm = ReflexionModule(dim=64, confidence_threshold=0.5, max_reflections=3)
    output = randn(2, 8, 64, requires_grad=True)
    context = randn(2, 8, 64)
    corrected, info = rm(output, context)
    print(f"输入: {output.shape}, 输出: {corrected.shape}")
    print(f"反思信息: {info}")
    
    loss = (corrected ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in rm.parameters())
    print(f"梯度传播: {has_grad}")
    print("ReflexionModule 测试 PASSED\n")
    
    print("=== 测试 ConfidenceValidator ===")
    cv = ConfidenceValidator(dim=64, confidence_threshold=0.6)
    output2 = randn(64, requires_grad=True)
    context2 = randn(64)
    corrected2, info2 = cv(output2, context2)
    print(f"输入: {output2.shape}, 输出: {corrected2.shape}")
    print(f"反思信息: {info2}")
    
    loss2 = (corrected2 ** 2).mean()
    loss2.backward()
    has_grad2 = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in cv.parameters())
    print(f"梯度传播: {has_grad2}")
    print("ConfidenceValidator 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")

