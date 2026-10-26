# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""情感引擎 - PyTorch版本，整合到SimplCore AGI系统."""

from __future__ import annotations

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
from typing import Dict, List, Optional, Any
from enum import Enum


class EmotionType(Enum):
    """情感类型."""
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


class EmotionState:
    """情感状态."""
    
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


class EmotionDetector(nn.Module):
    """情感检测器 - 从文本/潜状态中检测情感."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.emotion_proj = nn.Linear(input_dim * 2, hidden_dim)
        self.intensity_proj = nn.Linear(hidden_dim, len(EmotionType))
        self.confidence_proj = nn.Linear(hidden_dim, len(EmotionType))
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """检测情感.
        
        Args:
            x: (B, L, H) 或 (B, H) 输入张量.
            context: 上下文张量.
            
        Returns:
            dict: 包含检测到的情感状态和信息.
        """
        if x.dim() == 3:
            x = x.mean(dim=1)
        
        if context is not None:
            if context.dim() == 3:
                context = context.mean(dim=1)
            x_combined = torch.cat([x, context], dim=-1)
        else:
            x_combined = torch.cat([x, x], dim=-1)
        
        feature = torch.tanh(self.emotion_proj(x_combined))
        feature = self.output_norm(feature)
        
        intensity_logits = self.intensity_proj(feature)
        confidence_logits = self.confidence_proj(feature)
        
        intensity_probs = F.softmax(intensity_logits, dim=-1)
        confidence_probs = torch.sigmoid(confidence_logits)
        
        emotion_states = []
        for i, emotion_type in enumerate(EmotionType):
            intensity = float(intensity_probs[0, i].item())
            confidence = float(confidence_probs[0, i].item())
            if confidence > 0.1:
                emotion_states.append(EmotionState(emotion_type, intensity, confidence))
        
        emotion_states.sort(key=lambda e: e.confidence * e.intensity, reverse=True)
        
        info = {
            "num_detected": len(emotion_states),
            "top_emotion": emotion_states[0].emotion_type.value if emotion_states else "none",
            "top_intensity": emotion_states[0].intensity if emotion_states else 0.0,
            "top_confidence": emotion_states[0].confidence if emotion_states else 0.0,
        }
        
        return {"emotions": emotion_states, "info": info}


class EmotionContextTracker(nn.Module):
    """情感上下文跟踪器 - 维护情感历史."""
    
    def __init__(self, dim: int, window_size: int = 10):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        
        self.context_proj = nn.Linear(dim, dim)
        self.trend_proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        
        self._emotion_history: List[List[EmotionState]] = []
        self._context_vector = torch.zeros(dim)
    
    def update(
        self,
        current_emotions: List[EmotionState],
        utterance: torch.Tensor
    ) -> Dict[str, Any]:
        """更新情感上下文."""
        if utterance.dim() == 3:
            utterance = utterance.mean(dim=1)
        
        self._emotion_history.append(current_emotions)
        if len(self._emotion_history) > self.window_size:
            self._emotion_history = self._emotion_history[-self.window_size:]
        
        emotion_vector = torch.zeros(self.dim, device=utterance.device)
        for emotion in current_emotions:
            emotion_idx = list(EmotionType).index(emotion.emotion_type)
            emotion_vector = emotion_vector + utterance * emotion.intensity * emotion.confidence
        
        emotion_feature = torch.tanh(self.context_proj(emotion_vector))
        
        if len(self._emotion_history) >= 2:
            prev_emotions = self._emotion_history[-2]
            prev_emotion_vector = torch.zeros(self.dim, device=utterance.device)
            for emotion in prev_emotions:
                emotion_idx = list(EmotionType).index(emotion.emotion_type)
                prev_emotion_vector = prev_emotion_vector + utterance * emotion.intensity * emotion.confidence
            
            combined = torch.cat([emotion_feature, prev_emotion_vector], dim=-1)
            trend = torch.tanh(self.trend_proj(combined))
        else:
            trend = torch.zeros(self.dim, device=utterance.device)
        
        self._context_vector = self.norm(emotion_feature + trend * 0.3)
        
        avg_intensity = 0.0
        if self._emotion_history:
            all_intensities = []
            for emotions in self._emotion_history:
                if emotions:
                    all_intensities.extend([e.intensity * e.confidence for e in emotions])
            avg_intensity = sum(all_intensities) / len(all_intensities) if all_intensities else 0.0
        
        info = {
            "history_length": len(self._emotion_history),
            "avg_intensity": round(avg_intensity, 4),
            "context_vector_norm": round(float(self._context_vector.norm().item()), 4),
        }
        
        return {"context": self._context_vector, "info": info}
    
    def get_context(self) -> torch.Tensor:
        """获取当前情感上下文."""
        return self._context_vector
    
    def reset(self):
        """重置上下文."""
        self._emotion_history = []
        self._context_vector = torch.zeros(self.dim)


class EmpathyGenerator(nn.Module):
    """共情生成器 - 生成共情回应."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.empathy_proj = nn.Linear(dim * 2, dim)
        self.expression_proj = nn.Linear(dim, dim)
        self.adaptation_proj = nn.Linear(dim * 2, dim)
        self.output_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        user_emotion: torch.Tensor,
        current_response: torch.Tensor,
        intensity: float = 0.5
    ) -> Dict[str, Any]:
        """生成共情表达."""
        if user_emotion.dim() == 3:
            user_emotion = user_emotion.mean(dim=1)
        if current_response.dim() == 3:
            current_response = current_response.mean(dim=1)
        
        combined = torch.cat([user_emotion, current_response], dim=-1)
        empathy_feature = torch.tanh(self.empathy_proj(combined))
        
        empathy_strength = min(1.0, max(0.1, intensity))
        empathy_scaled = empathy_feature * empathy_strength
        
        expression = torch.tanh(self.expression_proj(empathy_scaled))
        
        adaptation_input = torch.cat([expression, current_response], dim=-1)
        adapted = F.gelu(self.adaptation_proj(adaptation_input))
        adapted = self.output_norm(adapted)
        
        info = {
            "empathy_strength": round(empathy_strength, 4),
            "empathy_feature_norm": round(float(empathy_feature.norm().item()), 4),
            "adapted_norm": round(float(adapted.norm().item()), 4),
        }
        
        return {"output": adapted, "info": info}


class EmotionEngine(nn.Module):
    """情感引擎 - 整合所有情感能力."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.emotion_detector = EmotionDetector(dim)
        self.emotion_tracker = EmotionContextTracker(dim)
        self.empathy_generator = EmpathyGenerator(dim)
        
        self.combine_proj = nn.Linear(dim * 3, dim)
        self.output_norm = nn.LayerNorm(dim)
    
    def detect_emotions(
        self,
        utterance: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> List[EmotionState]:
        """检测情感."""
        result = self.emotion_detector(utterance, context)
        return result["emotions"]
    
    def track_emotions(
        self,
        emotions: List[EmotionState],
        utterance: torch.Tensor
    ) -> Dict[str, Any]:
        """跟踪情感上下文."""
        return self.emotion_tracker.update(emotions, utterance)
    
    def generate_empathy(
        self,
        emotion_context: torch.Tensor,
        response: torch.Tensor,
        intensity: float = 0.5
    ) -> Dict[str, Any]:
        """生成共情回应."""
        return self.empathy_generator(emotion_context, response, intensity)
    
    def forward(
        self,
        utterance: torch.Tensor,
        base_response: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """处理情感智能."""
        if base_response is None:
            base_response = torch.zeros(1, self.dim, device=utterance.device)
        
        detection_result = self.emotion_detector(utterance, context)
        emotions = detection_result["emotions"]
        
        tracking_result = self.emotion_tracker.update(emotions, utterance)
        emotion_context = tracking_result["context"]
        
        top_emotion = emotions[0] if emotions else EmotionState(EmotionType.NEUTRAL, 0.0, 0.0)
        
        empathy_result = self.empathy_generator(
            emotion_context, base_response, top_emotion.intensity
        )
        
        if utterance.dim() == 3:
            utterance_avg = utterance.mean(dim=1)
        else:
            utterance_avg = utterance
        
        combined = torch.cat([
            utterance_avg,
            emotion_context,
            empathy_result["output"]
        ], dim=-1)
        
        final = F.gelu(self.combine_proj(combined))
        final = self.output_norm(final)
        
        info = {
            "detection": detection_result["info"],
            "tracking": tracking_result["info"],
            "empathy": empathy_result["info"],
            "top_emotion": top_emotion.to_dict(),
            "num_emotions": len(emotions),
        }
        
        return {"output": final, "info": info}
    
    def get_emotion_context(self) -> torch.Tensor:
        """获取情感上下文."""
        return self.emotion_tracker.get_context()
    
    def reset(self):
        """重置情感状态."""
        self.emotion_tracker.reset()


# ----------------------------------------------------------------------
# 情感回应模板
# ----------------------------------------------------------------------
EMOTION_RESPONSES = {
    EmotionType.HAPPINESS: [
        "我为你感到高兴！",
        "太棒了！",
        "祝贺你！",
        "听到这个消息真让人开心！",
        "愿这份快乐持续下去！",
    ],
    EmotionType.SADNESS: [
        "我很抱歉听到这个消息。",
        "你一定很难过。",
        "需要我陪伴你吗？",
        "一切都会好起来的。",
        "我在这里支持你。",
    ],
    EmotionType.ANGER: [
        "我理解你的愤怒。",
        "深呼吸，冷静下来。",
        "让我们一起解决问题。",
        "愤怒只会伤害自己。",
        "我愿意倾听你的不满。",
    ],
    EmotionType.FEAR: [
        "不要害怕，我在这儿。",
        "我们一起面对。",
        "恐惧是正常的，但不要让它控制你。",
        "一切都会好起来的。",
        "相信自己的能力。",
    ],
    EmotionType.SURPRISE: [
        "真的吗？太令人惊讶了！",
        "哇，没想到！",
        "这真是个意外！",
        "太神奇了！",
        "我也很惊讶！",
    ],
    EmotionType.LOVE: [
        "爱能战胜一切。",
        "感受到爱的力量了。",
        "爱是最美的情感。",
        "珍惜这份爱。",
        "愿爱永远伴随你。",
    ],
    EmotionType.EMPATHY: [
        "我能感受到你的感受。",
        "我理解你的处境。",
        "你不是一个人。",
        "我与你同在。",
        "共情是连接人心的桥梁。",
    ],
    EmotionType.NEUTRAL: [
        "我理解你的想法。",
        "继续说，我在听。",
        "有意思。",
        "我明白了。",
        "请继续分享。",
    ],
}


def generate_emotional_response(emotion_state: EmotionState) -> str:
    """根据情感状态生成回应文本."""
    responses = EMOTION_RESPONSES.get(emotion_state.emotion_type, EMOTION_RESPONSES[EmotionType.NEUTRAL])
    idx = min(int(emotion_state.intensity * len(responses)), len(responses) - 1)
    return responses[idx]


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dim = 128
    print("=== 情感引擎测试 ===")
    
    engine = EmotionEngine(dim)
    
    print("\n1. 测试情感检测:")
    utterance = torch.randn(1, dim)
    result = engine.detect_emotions(utterance)
    print(f"检测到 {len(result)} 种情感")
    for e in result[:3]:
        print(f"  {e.emotion_type.value}: 强度={e.intensity:.3f}, 置信度={e.confidence:.3f}")
    
    print("\n2. 测试情感跟踪:")
    for i in range(3):
        emotions = engine.detect_emotions(torch.randn(1, dim))
        track_result = engine.track_emotions(emotions, torch.randn(1, dim))
    print(f"上下文向量: {track_result['context'].shape}")
    print(f"跟踪信息: {track_result['info']}")
    
    print("\n3. 测试共情生成:")
    user_emotion = torch.randn(1, dim)
    response = torch.randn(1, dim)
    empathy_result = engine.generate_empathy(user_emotion, response, intensity=0.7)
    print(f"共情输出: {empathy_result['output'].shape}")
    
    print("\n4. 测试完整流程:")
    utterance = torch.randn(1, dim)
    base_response = torch.randn(1, dim)
    result = engine(utterance, base_response)
    print(f"输出: {result['output'].shape}")
    print(f"情感信息: {result['info']}")
    
    print("\n5. 测试情感回应生成:")
    emotion = EmotionState(EmotionType.HAPPINESS, intensity=0.8, confidence=0.9)
    print(f"情感: {emotion.emotion_type.value}, 回应: {generate_emotional_response(emotion)}")
    
    print("\n=== 情感引擎测试完成 ===")


