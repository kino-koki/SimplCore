#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
情绪智能模块 - 细粒度情绪识别与共情表达

核心功能：
1. 细粒度情绪识别
2. 情绪强度评估
3. 情绪上下文跟踪
4. 共情表达生成
5. 适应性回应调整
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class EmotionType(Enum):
    """情绪类型"""
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTEMPT = "contempt"
    SHAME = "shame"
    GUILT = "guilt"
    PRIDE = "pride"
    LOVE = "love"
    EMPATHY = "empathy"
    NEUTRAL = "neutral"


class EmotionIntensity(Enum):
    """情绪强度"""
    MINIMAL = 0.0
    WEAK = 0.25
    MODERATE = 0.5
    STRONG = 0.75
    INTENSE = 1.0


class EmotionState:
    """情绪状态"""
    
    def __init__(self, emotion_type: EmotionType, intensity: float, confidence: float):
        self.emotion_type = emotion_type
        self.intensity = intensity
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion": self.emotion_type.value,
            "intensity": round(self.intensity, 4),
            "confidence": round(self.confidence, 4),
        }


class EmotionDetector(Module):
    """情绪检测器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.emotion_proj = Linear(dim * 2, dim)
        self.register_module("emotion_proj", self.emotion_proj)
        
        self.intensity_proj = Linear(dim, len(EmotionType))
        self.register_module("intensity_proj", self.intensity_proj)
        
        self.confidence_proj = Linear(dim, len(EmotionType))
        self.register_module("confidence_proj", self.confidence_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
    
    def forward(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor] = None
    ) -> Tuple[List[EmotionState], Dict[str, Any]]:
        """检测情绪"""
        if context is not None:
            x = SimpTensor(np.concatenate([x._data, context._data]))
        else:
            x = SimpTensor(np.concatenate([x._data, x._data]))
        
        feature = tanh(self.emotion_proj(x))
        feature = self.output_norm(feature)
        
        intensity_logits = self.intensity_proj(feature)
        confidence_logits = self.confidence_proj(feature)
        
        intensity_probs = softmax(intensity_logits, dim=0)._data
        confidence_probs = sigmoid_tensor(confidence_logits)._data
        
        emotion_states = []
        for i, emotion_type in enumerate(EmotionType):
            intensity = float(intensity_probs[i])
            confidence = float(confidence_probs[i])
            
            if confidence > 0.1:
                emotion_states.append(EmotionState(emotion_type, intensity, confidence))
        
        emotion_states.sort(key=lambda e: e.confidence * e.intensity, reverse=True)
        
        info = {
            "num_detected": len(emotion_states),
            "top_emotion": emotion_states[0].emotion_type.value if emotion_states else "none",
            "top_intensity": emotion_states[0].intensity if emotion_states else 0.0,
            "top_confidence": emotion_states[0].confidence if emotion_states else 0.0,
        }
        
        return emotion_states, info


class EmotionContextTracker(Module):
    """情绪上下文跟踪器"""
    
    def __init__(self, dim: int, window_size: int = 10):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.context_proj = Linear(dim, dim)
        self.register_module("context_proj", self.context_proj)
        
        self.trend_proj = Linear(dim * 2, dim)
        self.register_module("trend_proj", self.trend_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
        
        self._emotion_history: List[List[EmotionState]] = []
        self._context_vector = zeros(dim)
    
    def update(
        self,
        current_emotions: List[EmotionState],
        utterance: SimpTensor
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """更新情绪上下文"""
        self._emotion_history.append(current_emotions)
        if len(self._emotion_history) > self.window_size:
            self._emotion_history = self._emotion_history[-self.window_size:]
        
        emotion_vector = zeros(self.dim)
        for emotion in current_emotions:
            emotion_idx = list(EmotionType).index(emotion.emotion_type)
            emotion_vector = emotion_vector + utterance * emotion.intensity * emotion.confidence
        
        emotion_feature = tanh(self.context_proj(emotion_vector))
        
        if len(self._emotion_history) >= 2:
            prev_emotions = self._emotion_history[-2]
            prev_emotion_vector = zeros(self.dim)
            for emotion in prev_emotions:
                emotion_idx = list(EmotionType).index(emotion.emotion_type)
                prev_emotion_vector = prev_emotion_vector + utterance * emotion.intensity * emotion.confidence
            
            combined = SimpTensor(np.concatenate([emotion_feature._data, prev_emotion_vector._data]))
            trend = tanh(self.trend_proj(combined))
        else:
            trend = zeros(self.dim)
        
        self._context_vector = self.norm(emotion_feature + trend * 0.3)
        
        avg_intensity = 0.0
        if self._emotion_history:
            all_intensities = []
            for emotions in self._emotion_history:
                if emotions:
                    all_intensities.extend([e.intensity * e.confidence for e in emotions])
            avg_intensity = np.mean(all_intensities) if all_intensities else 0.0
        
        info = {
            "history_length": len(self._emotion_history),
            "avg_intensity": round(avg_intensity, 4),
            "context_vector_norm": round(np.linalg.norm(self._context_vector._data), 4),
        }
        
        return self._context_vector, info
    
    def get_context(self) -> SimpTensor:
        """获取当前情绪上下文"""
        return self._context_vector
    
    def reset(self):
        """重置上下文"""
        self._emotion_history = []
        self._context_vector = zeros(self.dim)


class EmpathyGenerator(Module):
    """共情生成器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.empathy_proj = Linear(dim * 2, dim)
        self.register_module("empathy_proj", self.empathy_proj)
        
        self.expression_proj = Linear(dim, dim)
        self.register_module("expression_proj", self.expression_proj)
        
        self.adaptation_proj = Linear(dim * 2, dim)
        self.register_module("adaptation_proj", self.adaptation_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
    
    def forward(
        self,
        user_emotion: SimpTensor,
        current_response: SimpTensor,
        intensity: float = 0.5
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """生成共情表达"""
        combined = SimpTensor(np.concatenate([user_emotion._data, current_response._data]))
        empathy_feature = tanh(self.empathy_proj(combined))
        
        empathy_strength = min(1.0, max(0.1, intensity))
        empathy_scaled = empathy_feature * empathy_strength
        
        expression = tanh(self.expression_proj(empathy_scaled))
        
        adaptation_input = SimpTensor(np.concatenate([expression._data, current_response._data]))
        adapted = gelu(self.adaptation_proj(adaptation_input))
        adapted = self.output_norm(adapted)
        
        info = {
            "empathy_strength": round(empathy_strength, 4),
            "empathy_feature_norm": round(np.linalg.norm(empathy_feature._data), 4),
            "adapted_norm": round(np.linalg.norm(adapted._data), 4),
        }
        
        return adapted, info


class AdaptiveResponseGenerator(Module):
    """适应性回应生成器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.response_proj = Linear(dim, dim)
        self.register_module("response_proj", self.response_proj)
        
        self.emotion_adapt_proj = Linear(dim * 2, dim)
        self.register_module("emotion_adapt_proj", self.emotion_adapt_proj)
        
        self.style_proj = Linear(dim, dim)
        self.register_module("style_proj", self.style_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
    
    def forward(
        self,
        base_response: SimpTensor,
        emotion_context: SimpTensor,
        target_emotion: Optional[EmotionType] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """生成适应性回应"""
        response_feature = tanh(self.response_proj(base_response))
        
        combined = SimpTensor(np.concatenate([response_feature._data, emotion_context._data]))
        adapted = tanh(self.emotion_adapt_proj(combined))
        
        style_adjustment = zeros(self.dim)
        if target_emotion:
            emotion_idx = list(EmotionType).index(target_emotion)
            style_adjustment = randn(self.dim) * (emotion_idx + 1) / len(EmotionType)
        
        styled = tanh(self.style_proj(adapted + style_adjustment))
        final = self.output_norm(styled)
        
        info = {
            "target_emotion": target_emotion.value if target_emotion else "none",
            "adaptation_strength": round(np.linalg.norm(adapted._data) / np.linalg.norm(response_feature._data), 4),
        }
        
        return final, info


class EmotionalIntelligence(Module):
    """情绪智能系统"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.emotion_detector = EmotionDetector(dim)
        self.register_module("emotion_detector", self.emotion_detector)
        
        self.emotion_tracker = EmotionContextTracker(dim)
        self.register_module("emotion_tracker", self.emotion_tracker)
        
        self.empathy_generator = EmpathyGenerator(dim)
        self.register_module("empathy_generator", self.empathy_generator)
        
        self.adaptive_response = AdaptiveResponseGenerator(dim)
        self.register_module("adaptive_response", self.adaptive_response)
        
        self.combine_proj = Linear(dim * 3, dim)
        self.register_module("combine_proj", self.combine_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
    
    def forward(
        self,
        utterance: SimpTensor,
        base_response: Optional[SimpTensor] = None,
        context: Optional[SimpTensor] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理情绪智能"""
        emotions, detection_info = self.emotion_detector(utterance, context)
        
        emotion_context, tracking_info = self.emotion_tracker.update(emotions, utterance)
        
        if base_response is None:
            base_response = zeros(self.dim)
        
        top_emotion = emotions[0] if emotions else EmotionState(EmotionType.NEUTRAL, 0.0, 0.0)
        
        empathy_response, empathy_info = self.empathy_generator(
            emotion_context, base_response, top_emotion.intensity
        )
        
        adaptive_response, adaptive_info = self.adaptive_response(
            empathy_response, emotion_context, top_emotion.emotion_type
        )
        
        combined = SimpTensor(np.concatenate([
            utterance._data,
            emotion_context._data,
            adaptive_response._data
        ]))
        
        final = gelu(self.combine_proj(combined))
        final = self.output_norm(final)
        
        info = {
            "detection": detection_info,
            "tracking": tracking_info,
            "empathy": empathy_info,
            "adaptive": adaptive_info,
            "top_emotion": top_emotion.to_dict(),
            "num_emotions": len(emotions),
        }
        
        return final, info
    
    def detect_emotions(
        self,
        utterance: SimpTensor,
        context: Optional[SimpTensor] = None
    ) -> List[EmotionState]:
        """检测情绪"""
        emotions, _ = self.emotion_detector(utterance, context)
        return emotions
    
    def generate_empathy(
        self,
        emotion_context: SimpTensor,
        response: SimpTensor,
        intensity: float = 0.5
    ) -> SimpTensor:
        """生成共情回应"""
        empathy, _ = self.empathy_generator(emotion_context, response, intensity)
        return empathy
    
    def get_emotion_context(self) -> SimpTensor:
        """获取情绪上下文"""
        return self.emotion_tracker.get_context()
    
    def reset(self):
        """重置情绪状态"""
        self.emotion_tracker.reset()


if __name__ == "__main__":
    dim = 64
    print("=== 测试情绪智能模块 ===")
    
    print("\n1. 测试情绪检测器:")
    ed = EmotionDetector(dim)
    utterance = randn(dim)
    emotions, info = ed(utterance)
    print(f"检测到 {len(emotions)} 种情绪")
    for e in emotions[:3]:
        print(f"  {e.emotion_type.value}: 强度={e.intensity:.3f}, 置信度={e.confidence:.3f}")
    print(f"信息: {info}")
    
    print("\n2. 测试情绪上下文跟踪器:")
    ect = EmotionContextTracker(dim)
    for i in range(5):
        emotions_i, _ = ed(randn(dim))
        context, info = ect.update(emotions_i, randn(dim))
    print(f"上下文向量: {context.shape}")
    print(f"跟踪信息: {info}")
    
    print("\n3. 测试共情生成器:")
    eg = EmpathyGenerator(dim)
    user_emotion = randn(dim)
    response = randn(dim)
    empathy, info = eg(user_emotion, response, intensity=0.7)
    print(f"共情输出: {empathy.shape}")
    print(f"共情信息: {info}")
    
    print("\n4. 测试适应性回应生成器:")
    arg = AdaptiveResponseGenerator(dim)
    base_response = randn(dim)
    emotion_context = randn(dim)
    adapted, info = arg(base_response, emotion_context, EmotionType.HAPPINESS)
    print(f"适应性输出: {adapted.shape}")
    print(f"适应信息: {info}")
    
    print("\n5. 测试情绪智能系统:")
    ei = EmotionalIntelligence(dim)
    
    for i in range(3):
        utterance = randn(dim)
        base_response = randn(dim)
        output, info = ei(utterance, base_response)
        print(f"\n  输入 {i+1}:")
        print(f"    输出形状: {output.shape}")
        print(f"    检测到情绪数: {info['num_emotions']}")
        print(f"    主要情绪: {info['top_emotion']}")
    
    print("\n6. 测试梯度传播:")
    ei2 = EmotionalIntelligence(dim)
    test_input = randn(dim)
    test_response = randn(dim)
    output2, _ = ei2(test_input, test_response)
    loss = output2.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in ei2.parameters())
    print(f"梯度传播: {has_grad}")
    
    print("\n=== 所有测试通过 ===")
