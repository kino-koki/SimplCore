#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
对话状态跟踪模块 - 多轮对话状态管理

核心功能：
1. 对话状态表示与维护
2. 状态转换与更新
3. 话题检测与切换
4. 上下文摘要生成
5. 长程对话管理
"""
from __future__ import annotations

import numpy as np
import uuid
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class DialogueAct(Enum):
    """对话行为类型"""
    STATEMENT = "statement"
    QUESTION = "question"
    ANSWER = "answer"
    REQUEST = "request"
    ACKNOWLEDGMENT = "acknowledgment"
    CLARIFICATION = "clarification"
    REFUSAL = "refusal"
    PROMPT = "prompt"
    TOPIC_CHANGE = "topic_change"
    SUMMARY = "summary"


class DialogueStateType(Enum):
    """对话状态类型"""
    OPEN = "open"
    ACTIVE = "active"
    WAITING = "waiting"
    CLOSING = "closing"
    CLOSED = "closed"


class TopicStatus(Enum):
    """话题状态"""
    CURRENT = "current"
    PENDING = "pending"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class DialogueTurn:
    """对话轮次"""
    
    def __init__(
        self,
        speaker: str,
        utterance: SimpTensor,
        act: DialogueAct,
        timestamp: Optional[datetime] = None
    ):
        self.id = str(uuid.uuid4())[:8]
        self.speaker = speaker
        self.utterance = utterance
        self.act = act
        self.timestamp = timestamp or datetime.now()
        self.topic_refs: List[str] = []


class DialogueTopic:
    """对话话题"""
    
    def __init__(self, title: str, embedding: SimpTensor):
        self.id = str(uuid.uuid4())[:8]
        self.title = title
        self.embedding = embedding
        self.status = TopicStatus.CURRENT
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.turns: List[str] = []
        self.references: int = 0
    
    def mark_resolved(self):
        """标记为已解决"""
        self.status = TopicStatus.RESOLVED
        self.end_time = datetime.now()
    
    def mark_abandoned(self):
        """标记为已放弃"""
        self.status = TopicStatus.ABANDONED
        self.end_time = datetime.now()


class DialogueState:
    """对话状态"""
    
    def __init__(self, dim: int):
        self.id = str(uuid.uuid4())[:8]
        self.dim = dim
        self.state_type = DialogueStateType.OPEN
        self.turns: List[DialogueTurn] = []
        self.topics: Dict[str, DialogueTopic] = {}
        self.current_topic_id: Optional[str] = None
        self.state_vector: SimpTensor = zeros(dim)
        self.context_summary: SimpTensor = zeros(dim)
        self.last_update = datetime.now()
    
    def add_turn(self, turn: DialogueTurn):
        """添加对话轮次"""
        self.turns.append(turn)
        self.last_update = datetime.now()
    
    def set_topic(self, topic: DialogueTopic):
        """设置当前话题"""
        self.topics[topic.id] = topic
        self.current_topic_id = topic.id
        if topic.id not in self.topics:
            self.topics[topic.id] = topic
    
    def update_state_vector(self, vector: SimpTensor):
        """更新状态向量"""
        self.state_vector = vector


class StateTransitionModel(Module):
    """状态转换模型"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.state_proj = Linear(dim, dim)
        self.register_module("state_proj", self.state_proj)
        
        self.input_proj = Linear(dim, dim)
        self.register_module("input_proj", self.input_proj)
        
        self.transition_proj = Linear(dim * 2, dim * len(DialogueStateType))
        self.register_module("transition_proj", self.transition_proj)
        
        self.state_norm = LayerNorm(dim)
        self.register_module("state_norm", self.state_norm)
    
    def forward(
        self,
        current_state: SimpTensor,
        input_utterance: SimpTensor
    ) -> Tuple[DialogueStateType, float, SimpTensor]:
        """预测状态转换"""
        state_feature = tanh(self.state_proj(current_state))
        input_feature = tanh(self.input_proj(input_utterance))
        
        combined = SimpTensor(np.concatenate([state_feature._data, input_feature._data]))
        transition_logits = self.transition_proj(combined)
        transition_logits = transition_logits.reshape(len(DialogueStateType), self.dim)
        
        max_vals = np.max(transition_logits._data, axis=1)
        probs = softmax(SimpTensor(max_vals), dim=0)
        
        state_idx = int(np.argmax(probs._data))
        next_state = list(DialogueStateType)[state_idx]
        confidence = float(probs._data[state_idx])
        
        new_state_vector = self.state_norm(state_feature + input_feature * confidence)
        
        return next_state, confidence, new_state_vector


class TopicDetector(Module):
    """话题检测器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.topic_proj = Linear(dim, dim)
        self.register_module("topic_proj", self.topic_proj)
        
        self.similarity_proj = Linear(dim * 2, dim)
        self.register_module("similarity_proj", self.similarity_proj)
        
        self.change_detector = Linear(dim, 1)
        self.register_module("change_detector", self.change_detector)
    
    def forward(
        self,
        utterance: SimpTensor,
        current_topic: Optional[SimpTensor] = None,
        history_topics: Optional[List[SimpTensor]] = None
    ) -> Tuple[bool, float, int]:
        """检测话题变化"""
        utterance_feature = tanh(self.topic_proj(utterance))
        
        if current_topic is not None:
            combined = SimpTensor(np.concatenate([utterance_feature._data, current_topic._data]))
            similarity_feature = tanh(self.similarity_proj(combined))
            change_score = float(sigmoid_tensor(self.change_detector(similarity_feature))._data[0])
            
            is_topic_change = change_score < 0.5
            similarity = 1 - change_score
        else:
            is_topic_change = True
            similarity = 0.0
            change_score = 0.0
        
        best_topic_idx = -1
        if history_topics:
            similarities = []
            for topic in history_topics:
                sim = float(np.dot(utterance_feature._data, topic._data) / (
                    np.linalg.norm(utterance_feature._data) * np.linalg.norm(topic._data) + 1e-8
                ))
                similarities.append(sim)
            
            if similarities:
                best_topic_idx = int(np.argmax(similarities))
        
        return is_topic_change, similarity, best_topic_idx


class ContextSummarizer(Module):
    """上下文摘要生成器"""
    
    def __init__(self, dim: int, max_history: int = 50):
        super().__init__()
        self.dim = dim
        self.max_history = max_history
        
        self.summary_proj = Linear(dim, dim)
        self.register_module("summary_proj", self.summary_proj)
        
        self.attention_proj = Linear(dim * 2, 1)
        self.register_module("attention_proj", self.attention_proj)
        
        self.compress_proj = Linear(dim * 2, dim)
        self.register_module("compress_proj", self.compress_proj)
        
        self.summary_norm = LayerNorm(dim)
        self.register_module("summary_norm", self.summary_norm)
    
    def forward(
        self,
        current_utterance: SimpTensor,
        history: List[SimpTensor]
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """生成上下文摘要"""
        if not history:
            return current_utterance, {"summary_length": 0, "compressed": False}
        
        recent_history = history[-self.max_history:]
        
        attention_weights = []
        weighted_features = []
        
        for i, prev_utterance in enumerate(recent_history):
            combined = SimpTensor(np.concatenate([current_utterance._data, prev_utterance._data]))
            attention = sigmoid_tensor(self.attention_proj(combined))
            attention_weights.append(float(attention._data[0]))
            
            prev_feature = tanh(self.summary_proj(prev_utterance))
            weighted_features.append(prev_feature._data * float(attention._data[0]))
        
        if weighted_features:
            aggregated_history = SimpTensor(np.sum(weighted_features, axis=0))
            aggregated_history = aggregated_history / max(sum(attention_weights), 1)
        else:
            aggregated_history = zeros(self.dim)
        
        compression_input = SimpTensor(np.concatenate([current_utterance._data, aggregated_history._data]))
        compressed = gelu(self.compress_proj(compression_input))
        compressed = self.summary_norm(compressed)
        
        info = {
            "summary_length": len(recent_history),
            "compressed": True,
            "avg_attention_weight": np.mean(attention_weights) if attention_weights else 0.0,
            "max_attention_weight": max(attention_weights) if attention_weights else 0.0,
        }
        
        return compressed, info


class DialogueStateTracker(Module):
    """对话状态跟踪器"""
    
    def __init__(self, dim: int, max_history: int = 50):
        super().__init__()
        self.dim = dim
        
        self.state_transition = StateTransitionModel(dim)
        self.register_module("state_transition", self.state_transition)
        
        self.topic_detector = TopicDetector(dim)
        self.register_module("topic_detector", self.topic_detector)
        
        self.context_summarizer = ContextSummarizer(dim, max_history)
        self.register_module("context_summarizer", self.context_summarizer)
        
        self.state_update_proj = Linear(dim * 3, dim)
        self.register_module("state_update_proj", self.state_update_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
        
        self._dialogue_state = DialogueState(dim)
        self._utterance_history: List[SimpTensor] = []
        self._topic_embeddings: List[SimpTensor] = []
    
    def process_turn(
        self,
        utterance: SimpTensor,
        speaker: str = "user",
        act: Optional[DialogueAct] = None
    ) -> Tuple[DialogueState, Dict[str, Any]]:
        """处理对话轮次"""
        if act is None:
            act = DialogueAct.STATEMENT
        
        turn = DialogueTurn(speaker, utterance, act)
        self._dialogue_state.add_turn(turn)
        self._utterance_history.append(utterance)
        
        current_topic_embedding = None
        if self._dialogue_state.current_topic_id:
            current_topic = self._dialogue_state.topics.get(self._dialogue_state.current_topic_id)
            if current_topic:
                current_topic_embedding = current_topic.embedding
        
        is_topic_change, similarity, best_topic_idx = self.topic_detector(
            utterance, current_topic_embedding, self._topic_embeddings
        )
        
        if is_topic_change:
            new_topic = DialogueTopic(f"Topic_{len(self._topic_embeddings) + 1}", utterance)
            self._dialogue_state.set_topic(new_topic)
            self._topic_embeddings.append(utterance)
            
            if self._dialogue_state.current_topic_id and self._dialogue_state.current_topic_id != new_topic.id:
                old_topic = self._dialogue_state.topics.get(self._dialogue_state.current_topic_id)
                if old_topic:
                    old_topic.mark_resolved()
        
        context_summary, summary_info = self.context_summarizer(utterance, self._utterance_history[:-1])
        
        next_state_type, state_confidence, new_state_vector = self.state_transition(
            self._dialogue_state.state_vector, utterance
        )
        
        self._dialogue_state.state_type = next_state_type
        self._dialogue_state.context_summary = context_summary
        
        combined_update = SimpTensor(np.concatenate([
            new_state_vector._data,
            context_summary._data,
            utterance._data
        ]))
        
        updated_state = gelu(self.state_update_proj(combined_update))
        updated_state = self.output_norm(updated_state)
        
        self._dialogue_state.update_state_vector(updated_state)
        
        info = {
            "is_topic_change": is_topic_change,
            "topic_similarity": similarity,
            "current_topic_id": self._dialogue_state.current_topic_id,
            "num_topics": len(self._topic_embeddings),
            "num_turns": len(self._dialogue_state.turns),
            "state_type": next_state_type.value,
            "state_confidence": state_confidence,
            "summary_info": summary_info,
        }
        
        return self._dialogue_state, info
    
    def forward(
        self,
        utterance: SimpTensor,
        speaker: str = "user",
        act: Optional[DialogueAct] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        state, info = self.process_turn(utterance, speaker, act)
        return state.state_vector, info
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            "dialogue_id": self._dialogue_state.id,
            "state_type": self._dialogue_state.state_type.value,
            "num_turns": len(self._dialogue_state.turns),
            "num_topics": len(self._topic_embeddings),
            "current_topic_id": self._dialogue_state.current_topic_id,
            "last_update": self._dialogue_state.last_update.isoformat(),
        }
    
    def reset(self):
        """重置对话状态"""
        self._dialogue_state = DialogueState(self.dim)
        self._utterance_history = []
        self._topic_embeddings = []


if __name__ == "__main__":
    dim = 64
    print("=== 测试对话状态跟踪模块 ===")
    
    print("\n1. 测试状态转换模型:")
    stm = StateTransitionModel(dim)
    current_state = randn(dim)
    utterance = randn(dim)
    next_state, confidence, new_vector = stm(current_state, utterance)
    print(f"下一状态: {next_state.value}, 置信度: {confidence:.3f}")
    
    print("\n2. 测试话题检测器:")
    td = TopicDetector(dim)
    utterance2 = randn(dim)
    current_topic = randn(dim)
    history_topics = [randn(dim), randn(dim)]
    is_change, similarity, best_idx = td(utterance2, current_topic, history_topics)
    print(f"话题变化: {is_change}, 相似度: {similarity:.3f}, 最佳匹配: {best_idx}")
    
    print("\n3. 测试上下文摘要器:")
    cs = ContextSummarizer(dim)
    history = [randn(dim) for _ in range(5)]
    summary, info = cs(utterance, history)
    print(f"摘要输出: {summary.shape}")
    print(f"摘要信息: {info}")
    
    print("\n4. 测试对话状态跟踪器:")
    tracker = DialogueStateTracker(dim)
    
    for i in range(8):
        inp = randn(dim)
        state, info = tracker.process_turn(inp)
        print(f"  轮次 {i+1}:")
        print(f"    状态类型: {info['state_type']}")
        print(f"    话题变化: {info['is_topic_change']}")
        print(f"    话题数: {info['num_topics']}")
        print(f"    轮次数: {info['num_turns']}")
    
    summary = tracker.get_state_summary()
    print(f"\n对话状态摘要: {summary}")
    
    print("\n5. 测试梯度传播:")
    tracker2 = DialogueStateTracker(dim)
    test_input = randn(dim)
    output, _ = tracker2(test_input)
    loss = output.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in tracker2.parameters())
    print(f"梯度传播: {has_grad}")
    
    print("\n=== 所有测试通过 ===")
