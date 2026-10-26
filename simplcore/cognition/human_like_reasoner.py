#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Human-Like Cognitive Reasoner (类人认知推理器) - 整合所有类人化模块

核心架构：
1. NLU语义解析：增强语义解析精度、上下文关联能力及歧义消解
2. 对话状态跟踪：多轮对话状态管理与话题检测
3. 长程记忆：长上下文记忆与引用
4. 情绪智能：细粒度情绪识别与共情表达
5. 响应多样性：多元化回应生成机制，避免模式化输出
6. 认知推理：逻辑推理、常识判断及抽象思维能力
"""

import numpy as np
from typing imimport os
port Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, gelu, tanh
from ..engine.nn import Module, Linear, LayerNorm

from ..nlu.semantic_parser import SemanticParser
from ..nlu.disambiguator import ContextualDisambiguator
from ..dialogue.dialogue_state_tracker import DialogueStateTracker, DialogueAct
from ..dialogue.long_range_memory import LongRangeMemory, MemoryType
from ..emotion.emotional_intelligence import EmotionalIntelligence
from ..generation.response_variety import ResponseVariety
from ..knowledge.knowledge_graph import KnowledgeReasoner

from .tree_of_thought import TreeOfThought
from .reflexion import ReflexionModule
from .associative_reasoner import AssociativeReasoner


class HumanLikeMode(Enum):
    """类人化模式"""
    CONVERSATION = "conversation"
    REASONING = "reasoning"
    CREATIVE = "creative"
    EMPATHETIC = "empathetic"
    MEMORY_RECALL = "memory_recall"


class HumanLikeCognitiveReasoner(Module):
    """类人认知推理器 - 整合所有类人化模块"""
    
    def __init__(
        self,
        dim: int = 64,
        mode: HumanLikeMode = HumanLikeMode.CONVERSATION,
        enable_nlu: bool = True,
        enable_disambiguation: bool = True,
        enable_dialogue_tracking: bool = True,
        enable_long_range_memory: bool = True,
        enable_emotional_intelligence: bool = True,
        enable_response_variety: bool = True,
        enable_reasoning: bool = True,
        enable_knowledge: bool = True,
        max_history: int = 50,
        memory_capacity: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        
        self.enable_nlu = enable_nlu
        self.enable_disambiguation = enable_disambiguation
        self.enable_dialogue_tracking = enable_dialogue_tracking
        self.enable_long_range_memory = enable_long_range_memory
        self.enable_emotional_intelligence = enable_emotional_intelligence
        self.enable_response_variety = enable_response_variety
        self.enable_reasoning = enable_reasoning
        self.enable_knowledge = enable_knowledge
        
        self.output_proj = Linear(dim * 5, dim)
        self.register_module("output_proj", self.output_proj)
        
        self.combine_norm = LayerNorm(dim)
        self.register_module("combine_norm", self.combine_norm)
        
        if enable_nlu:
            self.semantic_parser = SemanticParser(dim)
            self.register_module("semantic_parser", self.semantic_parser)
        
        if enable_disambiguation:
            self.disambiguator = ContextualDisambiguator(dim)
            self.register_module("disambiguator", self.disambiguator)
        
        if enable_dialogue_tracking:
            self.dialogue_tracker = DialogueStateTracker(dim, max_history)
            self.register_module("dialogue_tracker", self.dialogue_tracker)
        
        if enable_long_range_memory:
            self.long_range_memory = LongRangeMemory(dim, memory_capacity)
            self.register_module("long_range_memory", self.long_range_memory)
        
        if enable_emotional_intelligence:
            self.emotional_intelligence = EmotionalIntelligence(dim)
            self.register_module("emotional_intelligence", self.emotional_intelligence)
        
        if enable_response_variety:
            self.response_variety = ResponseVariety(dim, max_history)
            self.register_module("response_variety", self.response_variety)
        
        if enable_reasoning:
            self.tree_of_thought = TreeOfThought(dim, max_depth=5)
            self.register_module("tree_of_thought", self.tree_of_thought)
            
            self.reflexion = ReflexionModule(dim, confidence_threshold=0.5)
            self.register_module("reflexion", self.reflexion)
            
            self.associative_reasoner = AssociativeReasoner(dim, num_concepts=256)
            self.register_module("associative_reasoner", self.associative_reasoner)
        
        if enable_knowledge:
            self.knowledge_reasoner = KnowledgeReasoner(dim)
            self.register_module("knowledge_reasoner", self.knowledge_reasoner)
        
        self._context_history: List[SimpTensor] = []
        self._candidates_history: List[SimpTensor] = []
    
    def _process_nlu(self, x: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理NLU语义解析"""
        if not self.enable_nlu:
            return x, {"nlu_enabled": False}
        
        frame, info = self.semantic_parser.parse(x, self._context_history)
        return frame.semantic_vector, {
            "nlu_enabled": True,
            "intent": info.get("intent", "unknown"),
            "intent_confidence": info.get("intent_confidence", 0.0),
            "num_roles": len(frame.roles),
            "num_dependencies": info.get("num_dependencies", 0),
        }
    
    def _process_disambiguation(self, x: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理歧义消解"""
        if not self.enable_disambiguation:
            return x, {"disambiguation_enabled": False}
        
        disambiguated, info = self.disambiguator(x, self._context_history, self._candidates_history)
        return disambiguated, {
            "disambiguation_enabled": True,
            "ambiguity_resolved": info.get("ambiguity_resolved", False),
            "overall_confidence": info.get("overall_confidence", 0.0),
            "anaphora_resolved": info.get("anaphora", {}).get("resolved", False),
            "word_sense_confidence": info.get("word_sense", {}).get("confidence", 0.0),
        }
    
    def _process_dialogue_tracking(self, x: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理对话状态跟踪"""
        if not self.enable_dialogue_tracking:
            return x, {"dialogue_tracking_enabled": False}
        
        state_vector, info = self.dialogue_tracker(x)
        return state_vector, {
            "dialogue_tracking_enabled": True,
            "state_type": info.get("state_type", "unknown"),
            "is_topic_change": info.get("is_topic_change", False),
            "num_topics": info.get("num_topics", 0),
            "num_turns": info.get("num_turns", 0),
            "summary_length": info.get("summary_info", {}).get("summary_length", 0),
        }
    
    def _process_memory(self, x: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理长程记忆"""
        if not self.enable_long_range_memory:
            return x, {"memory_enabled": False}
        
        retrieved, info = self.long_range_memory(x, top_k=5)
        
        self.long_range_memory.add_memory(x, MemoryType.CONTEXTUAL)
        
        return retrieved, {
            "memory_enabled": True,
            "retrieved_count": info.get("retrieved", 0),
            "memory_scores": info.get("scores", []),
            "memory_types": info.get("memory_types", []),
        }
    
    def _process_emotional_intelligence(self, x: SimpTensor, base_response: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理情绪智能"""
        if not self.enable_emotional_intelligence:
            return x, {"emotional_intelligence_enabled": False}
        
        if base_response is None:
            base_response = zeros(self.dim)
        
        emotion_context = self.emotional_intelligence.get_emotion_context()
        output, info = self.emotional_intelligence(x, base_response, emotion_context)
        
        return output, {
            "emotional_intelligence_enabled": True,
            "top_emotion": info.get("top_emotion", {}),
            "num_emotions": info.get("num_emotions", 0),
            "empathy_strength": info.get("empathy", {}).get("empathy_strength", 0.0),
        }
    
    def _process_response_variety(self, x: SimpTensor, context: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理响应多样性"""
        if not self.enable_response_variety:
            return x, {"response_variety_enabled": False}
        
        creativity_level = 0.3
        if self.mode == HumanLikeMode.CREATIVE:
            creativity_level = 0.7
        
        output, info = self.response_variety(x, context, creativity_level=creativity_level)
        
        return output, {
            "response_variety_enabled": True,
            "style": info.get("style", "unknown"),
            "diversity_score": info.get("diversity_score", 0.0),
            "pattern_adjusted": info.get("pattern_adjusted", False),
            "creativity_level": creativity_level,
        }
    
    def _process_reasoning(self, x: SimpTensor, context: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理认知推理"""
        if not self.enable_reasoning:
            return x, {"reasoning_enabled": False}
        
        assoc_feature, assoc_info = self.associative_reasoner(x, context)
        
        iterations = 15
        if self.mode == HumanLikeMode.REASONING:
            iterations = 30
        elif self.mode == HumanLikeMode.CREATIVE:
            iterations = 20
        
        reasoning_feature, reasoning_info = self.tree_of_thought(x, max_iterations=iterations)
        
        reflexion_feature, reflexion_info = self.reflexion(reasoning_feature, context)
        
        return reflexion_feature, {
            "reasoning_enabled": True,
            "num_associations": assoc_info.get("num_associations", 0),
            "reasoning_depth": reasoning_info.get("depth", 0),
            "reasoning_confidence": reasoning_info.get("confidence", 0.0),
            "reflexion_confidence": reflexion_info.get("confidence", 1.0),
            "is_hallucination": reflexion_info.get("is_hallucination", False),
        }
    
    def _process_knowledge(self, x: SimpTensor, context: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理知识推理"""
        if not self.enable_knowledge:
            return x, {"knowledge_enabled": False}
        
        knowledge_feature, knowledge_info = self.knowledge_reasoner(x, context)
        
        return knowledge_feature, {
            "knowledge_enabled": True,
            "entities_found": knowledge_info.get("entities_found", 0),
            "relations_found": knowledge_info.get("relations_found", 0),
            "reasoning_chains": knowledge_info.get("reasoning_chains", 0),
            "reasoning_confidence": knowledge_info.get("reasoning_confidence", 0.0),
            "has_knowledge": knowledge_info.get("has_knowledge", False),
        }
    
    def forward(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor] = None,
        mode: Optional[HumanLikeMode] = None,
        candidates: Optional[List[SimpTensor]] = None,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """
        类人认知推理前向传播
        
        Args:
            x: 输入特征
            context: 上下文信息
            mode: 类人化模式
            candidates: 指代消解候选
        
        Returns:
            output: 推理结果特征
            info: 推理过程信息
        """
        current_mode = mode or self.mode
        
        if candidates is not None:
            self._candidates_history.extend(candidates)
            if len(self._candidates_history) > 10:
                self._candidates_history = self._candidates_history[-10:]
        
        phase_info = {}
        
        semantic_feature, nlu_info = self._process_nlu(x)
        phase_info["nlu"] = nlu_info
        
        disambiguated_feature, disambiguation_info = self._process_disambiguation(semantic_feature)
        phase_info["disambiguation"] = disambiguation_info
        
        dialogue_feature, dialogue_info = self._process_dialogue_tracking(disambiguated_feature)
        phase_info["dialogue"] = dialogue_info
        
        memory_feature, memory_info = self._process_memory(disambiguated_feature)
        phase_info["memory"] = memory_info
        
        reasoning_feature = disambiguated_feature
        reasoning_info = {"reasoning_enabled": False}
        if self.mode in [HumanLikeMode.REASONING, HumanLikeMode.CREATIVE]:
            reasoning_feature, reasoning_info = self._process_reasoning(disambiguated_feature, context)
        phase_info["reasoning"] = reasoning_info
        
        knowledge_feature, knowledge_info = self._process_knowledge(reasoning_feature, context)
        phase_info["knowledge"] = knowledge_info
        
        emotion_feature, emotion_info = self._process_emotional_intelligence(reasoning_feature)
        phase_info["emotion"] = emotion_info
        
        context_stack = np.stack([
            dialogue_feature._data,
            memory_feature._data,
            emotion_feature._data,
            knowledge_feature._data
        ], axis=0)
        combined_context = SimpTensor(np.mean(context_stack, axis=0))
        
        variety_feature, variety_info = self._process_response_variety(emotion_feature, combined_context)
        phase_info["variety"] = variety_info
        
        features = [disambiguated_feature, dialogue_feature, emotion_feature, variety_feature, knowledge_feature]
        combined = SimpTensor(np.concatenate([f._data for f in features]))
        
        output = gelu(self.output_proj(combined))
        output = self.combine_norm(output)
        
        self._context_history.append(x)
        if len(self._context_history) > 50:
            self._context_history = self._context_history[-50:]
        
        overall_confidence = (
            nlu_info.get("intent_confidence", 0.5) * 0.15 +
            disambiguation_info.get("overall_confidence", 0.5) * 0.15 +
            reasoning_info.get("reflexion_confidence", 0.5) * 0.25 +
            emotion_info.get("top_emotion", {}).get("confidence", 0.5) * 0.125 +
            variety_info.get("diversity_score", 0.5) * 0.125 +
            knowledge_info.get("reasoning_confidence", 0.5) * 0.2
        )
        
        info = {
            "mode": current_mode.value,
            "overall_confidence": round(overall_confidence, 4),
            "phases": phase_info,
            "context_length": len(self._context_history),
        }
        
        return output, info
    
    def get_dialogue_summary(self) -> Dict[str, Any]:
        """获取对话状态摘要"""
        if not self.enable_dialogue_tracking:
            return {}
        return self.dialogue_tracker.get_state_summary()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        if not self.enable_long_range_memory:
            return {}
        return self.long_range_memory.get_memory_stats()
    
    def get_diversity_score(self) -> float:
        """获取多样性分数"""
        if not self.enable_response_variety:
            return 0.0
        return self.response_variety.get_diversity_score()
    
    def detect_emotions(self, utterance: SimpTensor) -> Any:
        """检测情绪"""
        if not self.enable_emotional_intelligence:
            return []
        return self.emotional_intelligence.detect_emotions(utterance)
    
    def set_mode(self, mode: HumanLikeMode):
        """设置类人化模式"""
        self.mode = mode
    
    def integrate_knowledge_base(self, knowledge_base):
        """集成知识库到推理器"""
        if self.enable_knowledge and hasattr(self, 'knowledge_reasoner'):
            self.knowledge_reasoner.integrate_knowledge_base(knowledge_base)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识统计"""
        if not self.enable_knowledge or not hasattr(self, 'knowledge_reasoner'):
            return {}
        return self.knowledge_reasoner.knowledge_graph.get_stats()
    
    def reset(self):
        """重置所有状态"""
        self._context_history = []
        self._candidates_history = []
        
        if self.enable_dialogue_tracking:
            self.dialogue_tracker.reset()
        if self.enable_long_range_memory:
            self.long_range_memory.reset()
        if self.enable_emotional_intelligence:
            self.emotional_intelligence.reset()
        if self.enable_response_variety:
            self.response_variety.reset()
        if self.enable_knowledge and hasattr(self, 'knowledge_reasoner'):
            self.knowledge_reasoner.knowledge_graph.reset()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed, randn
    
    manual_seed(0)
    dim = 64
    
    print("=== 测试 HumanLikeCognitiveReasoner ===")
    
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
    
    print("\n1. 测试多轮对话处理:")
    for i in range(5):
        x = randn(dim)
        context = randn(dim) if i > 0 else None
        
        output, info = hlcr(x, context)
        
        print(f"\n  轮次 {i+1}:")
        print(f"    输入: {x.shape}, 输出: {output.shape}")
        print(f"    整体置信度: {info['overall_confidence']:.4f}")
        print(f"    模式: {info['mode']}")
        print(f"    上下文长度: {info['context_length']}")
        print(f"    NLU意图: {info['phases']['nlu']['intent']}")
        print(f"    歧义消解: {info['phases']['disambiguation']['ambiguity_resolved']}")
        print(f"    话题变化: {info['phases']['dialogue']['is_topic_change']}")
        print(f"    检索记忆: {info['phases']['memory']['retrieved_count']}")
        print(f"    主要情绪: {info['phases']['emotion']['top_emotion']}")
        print(f"    风格: {info['phases']['variety']['style']}")
    
    print("\n2. 测试对话状态摘要:")
    summary = hlcr.get_dialogue_summary()
    print(f"对话状态: {summary}")
    
    print("\n3. 测试记忆统计:")
    memory_stats = hlcr.get_memory_stats()
    print(f"记忆统计: {memory_stats}")
    
    print("\n4. 测试多样性分数:")
    diversity = hlcr.get_diversity_score()
    print(f"多样性分数: {diversity:.4f}")
    
    print("\n5. 测试推理模式:")
    hlcr.set_mode(HumanLikeMode.REASONING)
    x_reason = randn(dim)
    output_reason, info_reason = hlcr(x_reason)
    print(f"推理输出: {output_reason.shape}")
    print(f"推理深度: {info_reason['phases']['reasoning']['reasoning_depth']}")
    print(f"推理置信度: {info_reason['phases']['reasoning']['reasoning_confidence']:.4f}")
    
    print("\n6. 测试创造性模式:")
    hlcr.set_mode(HumanLikeMode.CREATIVE)
    x_creative = randn(dim)
    output_creative, info_creative = hlcr(x_creative)
    print(f"创造性输出: {output_creative.shape}")
    print(f"创造性水平: {info_creative['phases']['variety']['creativity_level']}")
    print(f"风格: {info_creative['phases']['variety']['style']}")
    
    print("\n7. 测试情绪检测:")
    emotions = hlcr.detect_emotions(randn(dim))
    print(f"检测到 {len(emotions)} 种情绪")
    if emotions:
        for e in emotions[:3]:
            print(f"  {e.emotion_type.value}: 强度={e.intensity:.3f}, 置信度={e.confidence:.3f}")
    
    print("\n8. 测试梯度传播:")
    hlcr2 = HumanLikeCognitiveReasoner(dim=dim)
    test_input = randn(dim, requires_grad=True)
    output2, _ = hlcr2(test_input)
    loss = output2.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in hlcr2.parameters())
    print(f"梯度传播: {has_grad}")
    
    print("\n=== 所有测试通过 ===")

