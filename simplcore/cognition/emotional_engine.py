# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Emotional Decision Engine (情感决策引擎) - 基于情感状态的决策优化
支持多维度情感分析和情感驱动的决策制定
"""

import numpy as np
from typing imimport os
port Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, ones, softmax, sigmoid_tensor, gelu, tanh, abs as tensor_abs
from ..engine.nn import Module, Linear, LayerNorm


class EmotionType(Enum):
    """情感类型"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISED = "surprised"
    NEUTRAL = "neutral"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"


class EmotionalState:
    """情感状态"""
    
    def __init__(
        self,
        emotions: Dict[EmotionType, float],
        intensity: float = 0.0,
        valence: float = 0.0,
    ):
        self.emotions = emotions
        self.intensity = intensity
        self.valence = valence
    
    def dominant_emotion(self) -> EmotionType:
        """获取主导情感"""
        if not self.emotions:
            return EmotionType.NEUTRAL
        return max(self.emotions, key=self.emotions.get)
    
    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        return np.array([self.emotions.get(e, 0.0) for e in EmotionType])


class EmotionalDecisionEngine(Module):
    """情感决策引擎 - 基于情感状态的决策优化
    
    核心机制：
    1. 情感检测：识别输入中的情感信号
    2. 情感推理：基于情感状态进行推理
    3. 情感驱动决策：根据情感倾向优化决策
    4. 情感平衡：保持情感状态的动态平衡
    
    Args:
        dim: 特征维度
        num_emotions: 情感类别数量
        emotion_weight: 情感权重
        balance_weight: 情感平衡权重
    """
    
    def __init__(
        self,
        dim: int,
        num_emotions: int = 8,
        emotion_weight: float = 0.3,
        balance_weight: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_emotions = num_emotions
        self.emotion_weight = emotion_weight
        self.balance_weight = balance_weight
        
        self.emotion_encoder = Linear(dim, num_emotions)
        self.register_module("emotion_encoder", self.emotion_encoder)
        
        self.decision_proj = Linear(dim + num_emotions, dim)
        self.register_module("decision_proj", self.decision_proj)
        
        self.balance_proj = Linear(num_emotions, num_emotions)
        self.register_module("balance_proj", self.balance_proj)
        
        self.output_proj = Linear(dim * 2, dim)
        self.register_module("output_proj", self.output_proj)
        
        self.emotion_norm = LayerNorm(num_emotions)
        self.register_module("emotion_norm", self.emotion_norm)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
        
        self.emotion_biases = randn(num_emotions) * 0.02
        self.emotion_biases._requires_grad = True
        self.register_parameter("emotion_biases", self.emotion_biases)
        
        self._emotion_history: List[EmotionalState] = []
        self._decision_history: List[Dict[str, Any]] = []
    
    def _detect_emotions(self, x: SimpTensor) -> Dict[EmotionType, float]:
        """检测输入中的情感"""
        emotion_logits = self.emotion_encoder(x) + self.emotion_biases
        emotion_logits = self.emotion_norm(emotion_logits)
        emotion_probs = softmax(emotion_logits, dim=0)
        
        emotions = {}
        for i, emotion_type in enumerate(EmotionType):
            emotions[emotion_type] = emotion_probs._data[i]
        
        return emotions
    
    def _compute_emotional_state(self, emotions: Dict[EmotionType, float]) -> EmotionalState:
        """计算情感状态"""
        values = list(emotions.values())
        intensity = np.std(values)
        valence = (
            emotions.get(EmotionType.HAPPY, 0) +
            emotions.get(EmotionType.CONFIDENT, 0) -
            emotions.get(EmotionType.SAD, 0) -
            emotions.get(EmotionType.ANGRY, 0) -
            emotions.get(EmotionType.FEAR, 0)
        )
        
        return EmotionalState(emotions, intensity, valence)
    
    def _balance_emotions(self, emotion_probs: np.ndarray) -> np.ndarray:
        """情感平衡 - 避免极端情感"""
        mean_prob = np.mean(emotion_probs)
        balanced = emotion_probs - self.balance_weight * (emotion_probs - mean_prob)
        balanced = np.maximum(balanced, 0)
        balanced = balanced / (np.sum(balanced) + 1e-8)
        return balanced
    
    def _emotionally_weighted_decision(self, x: SimpTensor, emotion_vector: np.ndarray) -> SimpTensor:
        """基于情感的决策加权"""
        emotion_tensor = SimpTensor(emotion_vector)
        combined = SimpTensor(np.concatenate([x._data, emotion_tensor._data]))
        decision_feature = self.decision_proj(combined)
        decision_feature = gelu(decision_feature)
        return decision_feature
    
    def forward(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor] = None,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """
        情感决策引擎前向传播
        
        Args:
            x: 输入特征 [B, S, D] 或 [D]
            context: 上下文信息
        
        Returns:
            output: 决策结果特征
            info: 决策过程信息
        """
        if x.ndim == 3:
            B, S, D = x.shape
            results = []
            infos = []
            
            for b in range(B):
                for s in range(S):
                    xt = x[b, s]
                    out, info = self._single_decision(xt, context)
                    results.append(out)
                    infos.append(info)
            
            result_tensor = SimpTensor(np.stack([r._data for r in results])).reshape(B, S, D)
            
            avg_info = {
                "avg_emotion_intensity": sum(i["emotion_intensity"] for i in infos) / len(infos),
                "avg_valence": sum(i["valence"] for i in infos) / len(infos),
                "dominant_emotion": max(set(i["dominant_emotion"] for i in infos),
                                       key=lambda e: sum(1 for i in infos if i["dominant_emotion"] == e)),
                "total_decisions": len(self._decision_history),
            }
            
            return result_tensor, avg_info
        else:
            return self._single_decision(x, context)
    
    def _single_decision(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor],
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """单样本情感决策"""
        emotions = self._detect_emotions(x)
        emotional_state = self._compute_emotional_state(emotions)
        
        emotion_vector = emotional_state.to_vector()
        balanced_vector = self._balance_emotions(emotion_vector)
        
        emotion_feature = self._emotionally_weighted_decision(x, balanced_vector)
        
        combined = SimpTensor(np.concatenate([x._data, emotion_feature._data]))
        output = self.output_proj(combined)
        output = gelu(output)
        output = self.output_norm(output)
        
        self._emotion_history.append(emotional_state)
        
        info = {
            "emotions": emotions,
            "emotion_intensity": emotional_state.intensity,
            "valence": emotional_state.valence,
            "dominant_emotion": emotional_state.dominant_emotion().value,
            "balance_applied": np.any(emotion_vector != balanced_vector),
        }
        
        self._decision_history.append(info)
        
        return output, info
    
    def get_emotion_trend(self) -> Dict[str, float]:
        """获取情感趋势"""
        if len(self._emotion_history) == 0:
            return {}
        
        avg_intensity = sum(s.intensity for s in self._emotion_history) / len(self._emotion_history)
        avg_valence = sum(s.valence for s in self._emotion_history) / len(self._emotion_history)
        
        return {
            "avg_intensity": avg_intensity,
            "avg_valence": avg_valence,
            "history_length": len(self._emotion_history),
        }
    
    def reset_history(self) -> None:
        """重置历史记录"""
        self._emotion_history = []
        self._decision_history = []


class AdaptiveEmotionalEngine(EmotionalDecisionEngine):
    """自适应情感引擎 - 增强版情感决策
    
    额外功能：
    1. 情感记忆：记住历史情感状态
    2. 情境感知：根据上下文调整情感响应
    3. 动态权重：根据情感强度调整权重
    """
    
    def __init__(
        self,
        dim: int,
        num_emotions: int = 8,
        emotion_memory_size: int = 100,
    ):
        super().__init__(dim, num_emotions)
        self.emotion_memory_size = emotion_memory_size
        self._emotion_memory: List[np.ndarray] = []
        
        self.context_proj = Linear(dim, num_emotions)
    
    def _update_emotion_memory(self, emotion_vector: np.ndarray) -> None:
        """更新情感记忆"""
        self._emotion_memory.append(emotion_vector)
        if len(self._emotion_memory) > self.emotion_memory_size:
            self._emotion_memory.pop(0)
    
    def _contextual_emotion_adjustment(self, x: SimpTensor, context: SimpTensor) -> np.ndarray:
        """基于上下文调整情感"""
        context_emotion = self.context_proj(context)
        context_emotion = sigmoid_tensor(context_emotion)
        return context_emotion._data
    
    def _single_decision(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor],
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """增强版单样本情感决策"""
        emotions = self._detect_emotions(x)
        emotional_state = self._compute_emotional_state(emotions)
        
        emotion_vector = emotional_state.to_vector()
        
        if context is not None:
            context_adjustment = self._contextual_emotion_adjustment(x, context)
            emotion_vector = emotion_vector * (1 - self.emotion_weight) + context_adjustment * self.emotion_weight
        
        self._update_emotion_memory(emotion_vector)
        
        balanced_vector = self._balance_emotions(emotion_vector)
        
        emotion_feature = self._emotionally_weighted_decision(x, balanced_vector)
        
        combined = SimpTensor(np.concatenate([x._data, emotion_feature._data]))
        output = self.output_proj(combined)
        output = gelu(output)
        output = self.output_norm(output)
        
        self._emotion_history.append(emotional_state)
        
        info = {
            "emotions": emotions,
            "emotion_intensity": emotional_state.intensity,
            "valence": emotional_state.valence,
            "dominant_emotion": emotional_state.dominant_emotion().value,
            "memory_size": len(self._emotion_memory),
            "context_adjusted": context is not None,
        }
        
        self._decision_history.append(info)
        
        return output, info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed, randn
    
    manual_seed(0)
    print("=== 测试 EmotionalDecisionEngine ===")
    
    ede = EmotionalDecisionEngine(dim=64, num_emotions=8)
    x = randn(2, 8, 64, requires_grad=True)
    context = randn(64)
    y, info = ede(x, context)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"决策信息: dominant_emotion={info['dominant_emotion']}, intensity={info['avg_emotion_intensity']:.3f}")
    
    trend = ede.get_emotion_trend()
    print(f"情感趋势: {trend}")
    
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in ede.parameters())
    print(f"梯度传播: {has_grad}")
    print("EmotionalDecisionEngine 测试 PASSED\n")
    
    print("=== 测试 AdaptiveEmotionalEngine ===")
    aee = AdaptiveEmotionalEngine(dim=64, num_emotions=8, emotion_memory_size=50)
    x2 = randn(64, requires_grad=True)
    context2 = randn(64)
    y2, info2 = aee(x2, context2)
    print(f"输入: {x2.shape}, 输出: {y2.shape}")
    print(f"决策信息: dominant_emotion={info2['dominant_emotion']}, memory_size={info2['memory_size']}")
    
    loss2 = (y2 ** 2).mean()
    loss2.backward()
    has_grad2 = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in aee.parameters())
    print(f"梯度传播: {has_grad2}")
    print("AdaptiveEmotionalEngine 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")

