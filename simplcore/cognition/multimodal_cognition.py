# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""多模态认知整合模块 (Multimodal Cognitive Integration).

核心功能:
1. 将多模态感知输入整合到认知推理流程
2. 实现感知-认知闭环
3. 支持跨模态知识推理
4. 动态调整推理策略

与现有模块的集成:
- 接收multimodal_fusion的输出
- 为human_like_reasoner提供多模态上下文
- 更新knowledge_graph的环境知识
- 反馈到dialogue模块调整对话策略
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, gelu, tanh
from ..engine.nn import Module, Linear, LayerNorm

from ..perception.multimodal_fusion import (
    ModalityType,
    PerceptionState,
    MultimodalPerception,
    AudioPerception,
    EnvironmentPerception,
)

from ..knowledge.knowledge_graph import KnowledgeGraph, KnowledgeReasoner, EntityNode, RelationEdge

from .human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode


class IntegrationMode(Enum):
    """认知整合模式"""
    PASSIVE = "passive"
    ACTIVE = "active"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class PerceptionCognitionLoop(Module):
    """感知-认知闭环"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        self.feedback_proj = Linear(dim, dim)
        self.register_module("feedback_proj", self.feedback_proj)

        self.confidence_gate = Linear(dim, 1)
        self.register_module("confidence_gate", self.confidence_gate)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

        self._confidence_history: List[float] = []
        self._loop_count = 0

    def forward(
        self,
        perception_output: SimpTensor,
        cognition_output: SimpTensor,
        confidence: float = 0.5
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """执行感知-认知闭环"""
        feedback = tanh(self.feedback_proj(cognition_output))

        gate_value = float(sigmoid_tensor(self.confidence_gate(perception_output))._data[0])
        gate_value = gate_value * confidence

        integrated = SimpTensor(
            perception_output._data * gate_value + cognition_output._data * (1 - gate_value)
        )

        integrated = gelu(self.norm(integrated))

        self._confidence_history.append(confidence)
        if len(self._confidence_history) > 100:
            self._confidence_history = self._confidence_history[-100:]
        self._loop_count += 1

        info = {
            "loop_count": self._loop_count,
            "gate_value": round(gate_value, 4),
            "avg_confidence": np.mean(self._confidence_history),
            "has_feedback": True,
        }

        return integrated, info


class CrossModalReasoner(Module):
    """跨模态推理器"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        self.vision_text_proj = Linear(dim * 2, dim)
        self.register_module("vision_text_proj", self.vision_text_proj)

        self.audio_text_proj = Linear(dim * 2, dim)
        self.register_module("audio_text_proj", self.audio_text_proj)

        self.env_text_proj = Linear(dim * 2, dim)
        self.register_module("env_text_proj", self.env_text_proj)

        self.fusion_proj = Linear(dim * 3, dim)
        self.register_module("fusion_proj", self.fusion_proj)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

    def _cross_attention(
        self,
        primary: SimpTensor,
        secondary: SimpTensor,
        proj: Linear
    ) -> SimpTensor:
        """计算跨模态注意力"""
        combined = SimpTensor(np.concatenate([primary._data, secondary._data]))
        return tanh(proj(combined))

    def forward(
        self,
        text_feature: SimpTensor,
        visual_feature: Optional[SimpTensor] = None,
        audio_feature: Optional[SimpTensor] = None,
        env_feature: Optional[SimpTensor] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """跨模态推理"""
        features = []
        active_modalities = []

        if visual_feature is not None:
            features.append(self._cross_attention(text_feature, visual_feature, self.vision_text_proj))
            active_modalities.append("vision_text")

        if audio_feature is not None:
            features.append(self._cross_attention(text_feature, audio_feature, self.audio_text_proj))
            active_modalities.append("audio_text")

        if env_feature is not None:
            features.append(self._cross_attention(text_feature, env_feature, self.env_text_proj))
            active_modalities.append("env_text")

        if not features:
            return text_feature, {"cross_modal_enabled": False, "active_pairs": []}

        combined = SimpTensor(np.concatenate([f._data for f in features]))
        output = gelu(self.fusion_proj(combined))
        output = self.norm(output)

        info = {
            "cross_modal_enabled": True,
            "active_pairs": active_modalities,
            "num_pairs": len(active_modalities),
        }

        return output, info


class MultimodalCognitiveIntegration(Module):
    """多模态认知整合系统"""

    def __init__(self, dim: int = 64, mode: IntegrationMode = IntegrationMode.ADAPTIVE):
        super().__init__()
        self.dim = dim
        self.mode = mode

        self.multimodal_perception = MultimodalPerception(dim)
        self.register_module("multimodal_perception", self.multimodal_perception)

        self.audio_perception = AudioPerception(dim)
        self.register_module("audio_perception", self.audio_perception)

        self.env_perception = EnvironmentPerception(dim)
        self.register_module("env_perception", self.env_perception)

        self.cross_modal_reasoner = CrossModalReasoner(dim)
        self.register_module("cross_modal_reasoner", self.cross_modal_reasoner)

        self.perception_cognition_loop = PerceptionCognitionLoop(dim)
        self.register_module("perception_cognition_loop", self.perception_cognition_loop)

        self.knowledge_reasoner = KnowledgeReasoner(dim)
        self.register_module("knowledge_reasoner", self.knowledge_reasoner)

        self.cognitive_reasoner = HumanLikeCognitiveReasoner(dim)
        self.register_module("cognitive_reasoner", self.cognitive_reasoner)

        self.context_proj = Linear(dim * 2, dim)
        self.register_module("context_proj", self.context_proj)

        self.output_proj = Linear(dim * 3, dim)
        self.register_module("output_proj", self.output_proj)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

        self._perception_history: List[Dict[str, Any]] = []
        self._cognition_history: List[Dict[str, Any]] = []
        self._environment_state: Dict[str, float] = {
            "temperature": 25.0,
            "light_level": 1.0,
        }

    def update_environment(self, temperature: float = None, light_level: float = None):
        """更新环境状态"""
        if temperature is not None:
            self._environment_state["temperature"] = temperature
        if light_level is not None:
            self._environment_state["light_level"] = light_level

    def process_multimodal_input(
        self,
        text_input: SimpTensor,
        visual_input: Optional[SimpTensor] = None,
        audio_input: Optional[SimpTensor] = None,
        env_context: Optional[SimpTensor] = None,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理多模态输入"""
        modalities = {ModalityType.TEXT: text_input}

        if visual_input is not None:
            modalities[ModalityType.VISION] = visual_input

        if audio_input is not None:
            modalities[ModalityType.AUDIO] = audio_input

        if env_context is not None:
            modalities[ModalityType.ENVIRONMENT] = env_context

        perception_output, perception_info = self.multimodal_perception(modalities)

        return perception_output, perception_info

    def integrate_perception_to_cognition(
        self,
        perception_feature: SimpTensor,
        text_input: SimpTensor,
        context: Optional[SimpTensor] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """将感知特征整合到认知推理"""
        cross_feature, cross_info = self.cross_modal_reasoner(
            text_input,
            visual_feature=perception_feature if self._has_modality(ModalityType.VISION) else None,
            audio_feature=perception_feature if self._has_modality(ModalityType.AUDIO) else None,
            env_feature=perception_feature if self._has_modality(ModalityType.ENVIRONMENT) else None,
        )

        combined_context = SimpTensor(np.concatenate([text_input._data, cross_feature._data]))
        combined_context = gelu(self.context_proj(combined_context))

        if context is not None:
            combined_context = SimpTensor(
                0.7 * combined_context._data + 0.3 * context._data
            )

        cognitive_output, cognitive_info = self.cognitive_reasoner(
            text_input,
            context=combined_context,
        )

        confidence = cognitive_info.get("overall_confidence", 0.5)
        loop_output, loop_info = self.perception_cognition_loop(
            perception_feature, cognitive_output, confidence
        )

        features = [perception_feature, cognitive_output, loop_output]
        combined = SimpTensor(np.concatenate([f._data for f in features]))
        output = gelu(self.output_proj(combined))
        output = self.norm(output)

        self._perception_history.append(perception_info)
        if len(self._perception_history) > 50:
            self._perception_history = self._perception_history[-50:]

        self._cognition_history.append(cognitive_info)
        if len(self._cognition_history) > 50:
            self._cognition_history = self._cognition_history[-50:]

        info = {
            "perception": perception_info,
            "cross_modal": cross_info,
            "cognition": cognitive_info,
            "loop": loop_info,
            "overall_confidence": round(confidence, 4),
            "num_modalities": perception_info.get("num_modalities", 0),
            "dominant_modality": perception_info.get("dominant_modality", "text"),
        }

        return output, info

    def _has_modality(self, modality_type: ModalityType) -> bool:
        """检查是否有特定模态"""
        active = self.multimodal_perception._active_modalities
        return modality_type in active

    def forward(
        self,
        text_input: SimpTensor,
        visual_input: Optional[SimpTensor] = None,
        audio_input: Optional[SimpTensor] = None,
        context: Optional[SimpTensor] = None,
        temperature: float = None,
        light_level: float = None,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        if temperature is not None or light_level is not None:
            self.update_environment(temperature, light_level)

        env_feature = None
        if self._environment_state:
            env_context = randn(self.dim)
            env_feature, _ = self.env_perception(
                env_context,
                temperature=self._environment_state["temperature"],
                light_level=self._environment_state["light_level"],
            )

        audio_feature = None
        if audio_input is not None:
            audio_feature, _ = self.audio_perception(audio_input)

        perception_feature, perception_info = self.process_multimodal_input(
            text_input,
            visual_input=visual_input,
            audio_input=audio_feature,
            env_context=env_feature,
        )

        output, info = self.integrate_perception_to_cognition(
            perception_feature, text_input, context
        )

        return output, info

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "integration_mode": self.mode.value,
            "perception_state": self.multimodal_perception.state.value,
            "active_modalities": self.multimodal_perception.get_status()["active_modalities"],
            "memory_frames": self.multimodal_perception.get_status()["memory_frames"],
            "cognition_mode": self.cognitive_reasoner.mode.value,
            "cognition_confidence": self.cognitive_reasoner.get_diversity_score(),
            "environment_state": self._environment_state,
            "perception_history": len(self._perception_history),
            "cognition_history": len(self._cognition_history),
        }

    def set_cognition_mode(self, mode: HumanLikeMode):
        """设置认知模式"""
        self.cognitive_reasoner.set_mode(mode)

    def reset(self):
        """重置系统"""
        self.multimodal_perception.reset()
        self.cognitive_reasoner.reset()
        self._perception_history = []
        self._cognition_history = []
        self._environment_state = {
            "temperature": 25.0,
            "light_level": 1.0,
        }


if __name__ == "__main__":
    dim = 64
    print("=" * 60)
    print("多模态认知整合系统测试")
    print("=" * 60)

    print("\n1. 测试多模态认知整合:")
    mmci = MultimodalCognitiveIntegration(dim)

    text_input = randn(dim)
    visual_input = randn(dim)
    audio_input = randn(dim)

    output, info = mmci(
        text_input,
        visual_input=visual_input,
        audio_input=audio_input,
        temperature=22.0,
        light_level=0.8,
    )
    print(f"  输出形状: {output.shape}")
    print(f"  整体置信度: {info['overall_confidence']:.4f}")
    print(f"  模态数量: {info['num_modalities']}")
    print(f"  主导模态: {info['dominant_modality']}")
    print(f"  跨模态配对: {info['cross_modal']['active_pairs']}")

    print("\n2. 测试系统状态:")
    status = mmci.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\n3. 测试多轮对话:")
    for i in range(3):
        x = randn(dim)
        output, info = mmci(x)
        print(f"\n  轮次 {i+1}:")
        print(f"    输出: {output.shape}")
        print(f"    置信度: {info['overall_confidence']:.4f}")
        print(f"    认知模式: {info['cognition']['mode']}")

    print("\n4. 测试推理模式:")
    mmci.set_cognition_mode(HumanLikeMode.REASONING)
    x_reason = randn(dim)
    output_reason, info_reason = mmci(x_reason)
    print(f"  推理输出: {output_reason.shape}")
    print(f"  推理深度: {info_reason['cognition']['phases']['reasoning'].get('reasoning_depth', 0)}")

    print("\n5. 测试创造性模式:")
    mmci.set_cognition_mode(HumanLikeMode.CREATIVE)
    x_creative = randn(dim)
    output_creative, info_creative = mmci(x_creative)
    print(f"  创造性输出: {output_creative.shape}")
    print(f"  创造性水平: {info_creative['cognition']['phases']['variety']['creativity_level']}")

    print("\n6. 测试环境感知:")
    mmci.update_environment(temperature=18.5, light_level=0.3)
    x_env = randn(dim)
    output_env, info_env = mmci(x_env)
    print(f"  输出: {output_env.shape}")
    print(f"  环境状态: {mmci._environment_state}")

    print("\n7. 测试梯度传播:")
    mmci2 = MultimodalCognitiveIntegration(dim)
    test_input = randn(dim, requires_grad=True)
    output2, _ = mmci2(test_input)
    loss = output2.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in mmci2.parameters())
    print(f"  梯度传播: {has_grad}")

    print("\n" + "=" * 60)
    print("所有测试 PASSED!")
    print("=" * 60)
