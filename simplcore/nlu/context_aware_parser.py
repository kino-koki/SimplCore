# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""语境感知语义解析模块 (Context-Aware Semantic Parsing).

核心功能:
1. 语境记忆机制 - 维护对话语境的长期记忆
2. 语义连贯性检测 - 检测当前输入与语境的连贯性
3. 话题追踪 - 追踪对话话题的变化
4. 语境自适应加权 - 根据语境相关度动态调整语义特征
5. 多轮对话理解 - 理解跨越多轮的复杂对话

与现有模块的集成:
- 扩展SemanticParser支持语境感知
- 为dialogue模块提供语义上下文
- 为cognition模块提供语境特征
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, gelu, tanh
from ..engine.nn import Module, Linear, LayerNorm

from .semantic_parser import SemanticParser, SemanticFrame, IntentType, SemanticRole


class CoherenceLevel(Enum):
    """连贯性等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DISCONNECTED = "disconnected"


class TopicState(Enum):
    """话题状态"""
    SAME = "same"
    SHIFTING = "shifting"
    NEW = "new"
    UNKNOWN = "unknown"


class ContextMemory:
    """语境记忆"""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.frames: List[SemanticFrame] = []
        self.intents: List[IntentType] = []
        self.vectors: List[SimpTensor] = []

    def add_frame(self, frame: SemanticFrame):
        """添加语义框架"""
        self.frames.append(frame)
        if frame.intent:
            self.intents.append(frame.intent)
        if frame.semantic_vector:
            self.vectors.append(frame.semantic_vector)

        if len(self.frames) > self.max_size:
            self.frames = self.frames[-self.max_size:]
            self.intents = self.intents[-self.max_size:]
            self.vectors = self.vectors[-self.max_size:]

    def get_recent(self, num: int = 5) -> List[SemanticFrame]:
        """获取最近的语境"""
        return self.frames[-num:]

    def get_context_vector(self) -> SimpTensor:
        """获取语境向量"""
        if not self.vectors:
            return zeros(64)
        stacked = np.stack([v._data for v in self.vectors], axis=0)
        return SimpTensor(np.mean(stacked, axis=0))

    def get_intent_sequence(self) -> List[IntentType]:
        """获取意图序列"""
        return self.intents

    def clear(self):
        """清空记忆"""
        self.frames = []
        self.intents = []
        self.vectors = []


class CoherenceDetector(Module):
    """语义连贯性检测器"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        self.similarity_proj = Linear(dim * 2, dim)
        self.register_module("similarity_proj", self.similarity_proj)

        self.coherence_head = Linear(dim, 4)
        self.register_module("coherence_head", self.coherence_head)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

    def forward(
        self,
        current: SimpTensor,
        context_vector: SimpTensor
    ) -> Tuple[CoherenceLevel, float, Dict[str, Any]]:
        """检测连贯性"""
        combined = SimpTensor(np.concatenate([current._data, context_vector._data]))
        feature = tanh(self.similarity_proj(combined))
        feature = self.norm(feature)

        logits = self.coherence_head(feature)
        probs = softmax(logits, dim=0)

        coherence_idx = int(np.argmax(probs._data))
        coherence_level = list(CoherenceLevel)[coherence_idx]
        confidence = float(probs._data[coherence_idx])

        cosine_sim = float(np.dot(current._data, context_vector._data) / (
            np.linalg.norm(current._data) * np.linalg.norm(context_vector._data) + 1e-8
        ))

        threshold = 0.3
        if cosine_sim > 0.7:
            coherence_level = CoherenceLevel.HIGH
        elif cosine_sim > 0.4:
            coherence_level = CoherenceLevel.MEDIUM
        elif cosine_sim > 0.1:
            coherence_level = CoherenceLevel.LOW
        else:
            coherence_level = CoherenceLevel.DISCONNECTED

        info = {
            "coherence_level": coherence_level.value,
            "confidence": confidence,
            "cosine_similarity": cosine_sim,
            "probabilities": {c.value: float(probs._data[i]) for i, c in enumerate(CoherenceLevel)},
        }

        return coherence_level, confidence, info


class TopicTracker(Module):
    """话题追踪器"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        self.topic_proj = Linear(dim, dim)
        self.register_module("topic_proj", self.topic_proj)

        self.state_head = Linear(dim * 2, 4)
        self.register_module("state_head", self.state_head)

        self.similarity_proj = Linear(dim * 2, 1)
        self.register_module("similarity_proj", self.similarity_proj)

        self._topic_history: List[SimpTensor] = []
        self._topic_labels: List[str] = []

    def forward(
        self,
        current: SimpTensor,
        context_history: List[SimpTensor]
    ) -> Tuple[TopicState, float, Dict[str, Any]]:
        """追踪话题"""
        current_topic = tanh(self.topic_proj(current))

        if not context_history:
            self._topic_history.append(current_topic)
            return TopicState.NEW, 1.0, {"topic_change": True, "topic_count": 1}

        recent_topics = context_history[-5:]

        similarities = []
        for ctx in recent_topics:
            ctx_topic = tanh(self.topic_proj(ctx))
            combined = SimpTensor(np.concatenate([current_topic._data, ctx_topic._data]))
            sim = float(sigmoid_tensor(self.similarity_proj(combined))._data[0])
            similarities.append(sim)

        avg_similarity = np.mean(similarities)

        if avg_similarity > 0.7:
            topic_state = TopicState.SAME
        elif avg_similarity > 0.4:
            topic_state = TopicState.SHIFTING
        else:
            topic_state = TopicState.NEW

        context_topic = SimpTensor(np.mean([t._data for t in recent_topics], axis=0))
        combined = SimpTensor(np.concatenate([current_topic._data, context_topic._data]))
        logits = self.state_head(combined)
        probs = softmax(logits, dim=0)

        state_idx = int(np.argmax(probs._data))
        topic_state = list(TopicState)[state_idx]
        confidence = float(probs._data[state_idx])

        self._topic_history.append(current_topic)
        if len(self._topic_history) > 100:
            self._topic_history = self._topic_history[-100:]

        info = {
            "topic_state": topic_state.value,
            "confidence": confidence,
            "average_similarity": avg_similarity,
            "topic_change": topic_state != TopicState.SAME,
            "topic_count": len(set(self._topic_labels)) if self._topic_labels else len(self._topic_history),
            "history_length": len(self._topic_history),
        }

        return topic_state, confidence, info


class ContextAdaptiveWeighting(Module):
    """语境自适应加权"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        self.weight_proj = Linear(dim * 2, dim)
        self.register_module("weight_proj", self.weight_proj)

        self.gate_proj = Linear(dim, 1)
        self.register_module("gate_proj", self.gate_proj)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

    def forward(
        self,
        semantic_feature: SimpTensor,
        context_vector: SimpTensor,
        coherence_level: CoherenceLevel
    ) -> Tuple[SimpTensor, float, Dict[str, Any]]:
        """自适应加权语义特征"""
        combined = SimpTensor(np.concatenate([semantic_feature._data, context_vector._data]))
        weight_feature = tanh(self.weight_proj(combined))
        weight_feature = self.norm(weight_feature)

        gate_value = float(sigmoid_tensor(self.gate_proj(weight_feature))._data[0])

        coherence_weights = {
            CoherenceLevel.HIGH: 0.8,
            CoherenceLevel.MEDIUM: 0.5,
            CoherenceLevel.LOW: 0.2,
            CoherenceLevel.DISCONNECTED: 0.05,
        }
        coherence_weight = coherence_weights.get(coherence_level, 0.5)

        final_weight = gate_value * coherence_weight

        weighted_feature = semantic_feature * final_weight + context_vector * (1 - final_weight)

        info = {
            "gate_value": round(gate_value, 4),
            "coherence_weight": coherence_weight,
            "final_weight": round(final_weight, 4),
            "context_influence": round(1 - final_weight, 4),
        }

        return weighted_feature, final_weight, info


class ContextAwareSemanticParser(Module):
    """语境感知语义解析器"""

    def __init__(self, dim: int = 64, max_context_size: int = 50):
        super().__init__()
        self.dim = dim

        self.semantic_parser = SemanticParser(dim)
        self.register_module("semantic_parser", self.semantic_parser)

        self.context_memory = ContextMemory(max_context_size)

        self.coherence_detector = CoherenceDetector(dim)
        self.register_module("coherence_detector", self.coherence_detector)

        self.topic_tracker = TopicTracker(dim)
        self.register_module("topic_tracker", self.topic_tracker)

        self.context_adaptive_weighting = ContextAdaptiveWeighting(dim)
        self.register_module("context_adaptive_weighting", self.context_adaptive_weighting)

        self.output_proj = Linear(dim * 2, dim)
        self.register_module("output_proj", self.output_proj)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

        self._parse_count = 0

    def parse(
        self,
        x: SimpTensor,
        external_context: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """语境感知语义解析"""
        all_context = list(self.context_memory.vectors)
        if external_context:
            all_context.extend(external_context)

        frame, parser_info = self.semantic_parser.parse(x, all_context)

        self.context_memory.add_frame(frame)

        context_vector = self.context_memory.get_context_vector()

        coherence_level, coherence_confidence, coherence_info = self.coherence_detector(
            frame.semantic_vector, context_vector
        )

        topic_state, topic_confidence, topic_info = self.topic_tracker(
            frame.semantic_vector, all_context
        )

        weighted_feature, weight_value, weighting_info = self.context_adaptive_weighting(
            frame.semantic_vector, context_vector, coherence_level
        )

        combined = SimpTensor(np.concatenate([frame.semantic_vector._data, weighted_feature._data]))
        output = gelu(self.output_proj(combined))
        output = self.norm(output)

        self._parse_count += 1

        info = {
            "intent": parser_info["intent"],
            "intent_confidence": parser_info["intent_confidence"],
            "coherence": {
                "level": coherence_level.value,
                "confidence": coherence_confidence,
                "cosine_similarity": coherence_info["cosine_similarity"],
            },
            "topic": {
                "state": topic_state.value,
                "confidence": topic_confidence,
                "is_topic_change": topic_info["topic_change"],
                "topic_count": topic_info["topic_count"],
            },
            "weighting": weighting_info,
            "context_size": len(all_context),
            "parse_count": self._parse_count,
            "semantic_vector_shape": output.shape,
        }

        return output, info

    def forward(
        self,
        x: SimpTensor,
        external_context: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        return self.parse(x, external_context)

    def get_context_stats(self) -> Dict[str, Any]:
        """获取语境统计"""
        return {
            "frames_count": len(self.context_memory.frames),
            "intents_count": len(self.context_memory.intents),
            "vectors_count": len(self.context_memory.vectors),
            "parse_count": self._parse_count,
        }

    def get_coherence_trend(self, recent_n: int = 10) -> List[float]:
        """获取连贯性趋势"""
        if not self.context_memory.vectors:
            return []

        recent = self.context_memory.vectors[-recent_n:]
        trends = []
        for i in range(1, len(recent)):
            sim = float(np.dot(recent[i]._data, recent[i-1]._data) / (
                np.linalg.norm(recent[i]._data) * np.linalg.norm(recent[i-1]._data) + 1e-8
            ))
            trends.append(sim)
        return trends

    def reset(self):
        """重置"""
        self.context_memory.clear()
        self._parse_count = 0


class MultiTurnComprehension(Module):
    """多轮对话理解器"""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

        self.context_parser = ContextAwareSemanticParser(dim)
        self.register_module("context_parser", self.context_parser)

        self.turn_proj = Linear(dim, dim)
        self.register_module("turn_proj", self.turn_proj)

        self.comprehension_proj = Linear(dim * 3, dim)
        self.register_module("comprehension_proj", self.comprehension_proj)

        self.summary_proj = Linear(dim * 5, dim)
        self.register_module("summary_proj", self.summary_proj)

        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)

        self._turn_history: List[SimpTensor] = []

    def process_turn(
        self,
        x: SimpTensor,
        context: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理单轮对话"""
        parsed, info = self.context_parser.parse(x, context)

        turn_feature = tanh(self.turn_proj(parsed))

        self._turn_history.append(turn_feature)
        if len(self._turn_history) > 20:
            self._turn_history = self._turn_history[-20:]

        return parsed, info

    def get_multi_turn_summary(self) -> Tuple[SimpTensor, Dict[str, Any]]:
        """获取多轮对话摘要"""
        if not self._turn_history:
            return zeros(self.dim), {"turns": 0, "summary_available": False}

        if len(self._turn_history) <= 5:
            combined = SimpTensor(np.concatenate([t._data for t in self._turn_history]))
        else:
            sampled = self._turn_history[::2]
            combined = SimpTensor(np.concatenate([t._data for t in sampled]))

        if combined.shape[0] > self.dim * 5:
            combined = SimpTensor(combined._data[:self.dim * 5])

        summary = gelu(self.summary_proj(combined))
        summary = self.norm(summary)

        info = {
            "turns": len(self._turn_history),
            "summary_available": True,
            "summary_shape": summary.shape,
        }

        return summary, info

    def forward(
        self,
        x: SimpTensor,
        context: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        parsed, info = self.process_turn(x, context)

        summary, summary_info = self.get_multi_turn_summary()

        comprehension_input = SimpTensor(np.concatenate([parsed._data, summary._data]))
        if len(self._turn_history) > 1:
            prev_turn = self._turn_history[-2]
            comprehension_input = SimpTensor(np.concatenate([
                parsed._data,
                summary._data,
                prev_turn._data
            ]))

        comprehension = gelu(self.comprehension_proj(comprehension_input))
        comprehension = self.norm(comprehension)

        info["multi_turn"] = summary_info
        info["total_turns"] = len(self._turn_history)

        return comprehension, info

    def reset(self):
        """重置"""
        self.context_parser.reset()
        self._turn_history = []


if __name__ == "__main__":
    dim = 64
    print("=" * 60)
    print("语境感知语义解析模块测试")
    print("=" * 60)

    print("\n1. 测试语境记忆:")
    context_memory = ContextMemory(max_size=10)
    for i in range(5):
        frame = SemanticFrame()
        frame.semantic_vector = randn(dim)
        frame.intent = IntentType.STATEMENT
        context_memory.add_frame(frame)
    print(f"  框架数量: {len(context_memory.frames)}")
    print(f"  语境向量: {context_memory.get_context_vector().shape}")

    print("\n2. 测试连贯性检测器:")
    coherence_detector = CoherenceDetector(dim)
    current = randn(dim)
    context_vector = randn(dim)
    level, confidence, info = coherence_detector(current, context_vector)
    print(f"  连贯性等级: {level.value}, 置信度: {confidence:.3f}")
    print(f"  余弦相似度: {info['cosine_similarity']:.3f}")

    print("\n3. 测试话题追踪器:")
    topic_tracker = TopicTracker(dim)
    context_history = [randn(dim) for _ in range(3)]
    state, topic_confidence, topic_info = topic_tracker(randn(dim), context_history)
    print(f"  话题状态: {state.value}, 置信度: {topic_confidence:.3f}")
    print(f"  话题变化: {topic_info['topic_change']}")

    print("\n4. 测试语境自适应加权:")
    weighting = ContextAdaptiveWeighting(dim)
    semantic_feature = randn(dim)
    weighted, weight, weighting_info = weighting(semantic_feature, context_vector, CoherenceLevel.HIGH)
    print(f"  加权输出: {weighted.shape}")
    print(f"  权重信息: {weighting_info}")

    print("\n5. 测试语境感知语义解析器:")
    parser = ContextAwareSemanticParser(dim)

    for i in range(5):
        x = randn(dim)
        output, info = parser.parse(x)
        print(f"\n  轮次 {i+1}:")
        print(f"    意图: {info['intent']}, 置信度: {info['intent_confidence']:.3f}")
        print(f"    连贯性: {info['coherence']['level']}, 相似度: {info['coherence']['cosine_similarity']:.3f}")
        print(f"    话题: {info['topic']['state']}, 是否变化: {info['topic']['is_topic_change']}")

    stats = parser.get_context_stats()
    print(f"\n  语境统计: {stats}")

    print("\n6. 测试多轮对话理解器:")
    multi_turn = MultiTurnComprehension(dim)

    for i in range(5):
        x = randn(dim)
        output, info = multi_turn(x)
        print(f"\n  轮次 {i+1}:")
        print(f"    输出: {output.shape}")
        print(f"    意图: {info['intent']}")
        print(f"    总轮数: {info['total_turns']}")
        print(f"    话题变化: {info['topic']['is_topic_change']}")

    summary, summary_info = multi_turn.get_multi_turn_summary()
    print(f"\n  多轮摘要: {summary.shape}, 轮数: {summary_info['turns']}")

    print("\n7. 测试梯度传播:")
    parser2 = ContextAwareSemanticParser(dim)
    test_input = randn(dim, requires_grad=True)
    output2, _ = parser2(test_input)
    loss = output2.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in parser2.parameters())
    print(f"  梯度传播: {has_grad}")

    print("\n" + "=" * 60)
    print("所有测试 PASSED!")
    print("=" * 60)
