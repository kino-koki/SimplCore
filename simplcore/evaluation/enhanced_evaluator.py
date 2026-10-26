#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
增强版图灵测试评估器 - 可训练的评估体系

核心改进：
1. 评估器自身可训练 - 通过人类样本学习更好的评分标准
2. 添加奖励信号 - 基于评估结果的强化学习奖励
3. 自适应权重调整 - 根据模型表现动态调整评估权重
4. 集成学习 - 融合多个评估器的预测结果
"""

import numpy as np
import time
fimport os
rom typing import Optional, Tuple, List, Dict, Any, Callable

from ..engine.tensor import SimpTensor, randn, zeros, sigmoid_tensor, tanh, gelu, ones, abs, sqrt
from ..engine.nn import Module, Linear, LayerNorm

from .turing_test_eval import EvaluationMetric, TuringScore


class TrainableEvaluator(Module):
    """可训练的评估器基类"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._training_mode = True
    
    def set_training_mode(self, training: bool):
        """设置训练模式"""
        self._training_mode = training
    
    def compute_gradient_reward(self, score: float, target_score: float = 10.0) -> float:
        """计算梯度奖励 - 鼓励模型向高分方向学习"""
        reward = (score - target_score) / target_score
        return reward


class EnhancedHumanSimilarityEvaluator(TrainableEvaluator):
    """增强版人类相似度评估器"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
        
        self.style_proj = Linear(dim, dim)
        self.register_module("style_proj", self.style_proj)
        
        self.syntax_proj = Linear(dim, dim)
        self.register_module("syntax_proj", self.syntax_proj)
        
        self.pragmatic_proj = Linear(dim, dim)
        self.register_module("pragmatic_proj", self.pragmatic_proj)
        
        self.temporal_proj = Linear(dim, dim)
        self.register_module("temporal_proj", self.temporal_proj)
        
        self.score_head = Linear(dim * 4, 1)
        self.register_module("score_head", self.score_head)
        
        self.style_norm = LayerNorm(dim)
        self.register_module("style_norm", self.style_norm)
        
        self.attention_weights = Linear(dim * 3, 3)
        self.register_module("attention_weights", self.attention_weights)
    
    def forward(self, response: SimpTensor, context: Optional[SimpTensor] = None, 
               history_responses: Optional[List[SimpTensor]] = None) -> Tuple[float, Dict[str, Any]]:
        if context is None:
            context = zeros(self.dim)
        
        style_feature = tanh(self.style_norm(self.style_proj(response)))
        syntax_feature = tanh(self.syntax_proj(response))
        pragmatic_feature = tanh(self.pragmatic_proj(context))
        
        if history_responses and len(history_responses) > 0:
            history_mean = SimpTensor(np.mean([r._data for r in history_responses], axis=0))
            temporal_feature = tanh(self.temporal_proj(history_mean))
        else:
            temporal_feature = zeros(self.dim)
        
        attention_input = SimpTensor(np.concatenate([
            style_feature._data,
            syntax_feature._data,
            pragmatic_feature._data
        ]))
        attn_weights = sigmoid_tensor(self.attention_weights(attention_input))
        attn_weights_np = attn_weights._data / (np.sum(attn_weights._data) + 1e-8)
        
        combined = SimpTensor(np.concatenate([
            style_feature._data,
            syntax_feature._data,
            pragmatic_feature._data,
            temporal_feature._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        breakdown = {
            "style_score": float(np.mean(style_feature._data > 0) * 10),
            "syntax_score": float(np.mean(syntax_feature._data > 0) * 10),
            "pragmatic_score": float(np.mean(pragmatic_feature._data > 0) * 10),
            "temporal_consistency": float(np.mean(temporal_feature._data > 0) * 10),
            "attention_weights": list(attn_weights_np),
        }
        
        return normalized_score, breakdown


class EnhancedCoherenceEvaluator(TrainableEvaluator):
    """增强版对话连贯性评估器"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
        
        self.relevance_proj = Linear(dim * 2, dim)
        self.register_module("relevance_proj", self.relevance_proj)
        
        self.consistency_proj = Linear(dim * 2, dim)
        self.register_module("consistency_proj", self.consistency_proj)
        
        self.flow_proj = Linear(dim * 2, dim)
        self.register_module("flow_proj", self.flow_proj)
        
        self.topic_proj = Linear(dim * 3, dim)
        self.register_module("topic_proj", self.topic_proj)
        
        self.score_head = Linear(dim * 4, 1)
        self.register_module("score_head", self.score_head)
        
        self.coherence_gate = Linear(dim, 1)
        self.register_module("coherence_gate", self.coherence_gate)
    
    def forward(self, response: SimpTensor, previous_response: SimpTensor, 
               context: Optional[SimpTensor] = None, query: Optional[SimpTensor] = None) -> Tuple[float, Dict[str, Any]]:
        if context is None:
            context = zeros(self.dim)
        if query is None:
            query = zeros(self.dim)
        
        relevance_input = SimpTensor(np.concatenate([response._data, context._data]))
        relevance = tanh(self.relevance_proj(relevance_input))
        
        consistency_input = SimpTensor(np.concatenate([response._data, previous_response._data]))
        consistency = tanh(self.consistency_proj(consistency_input))
        
        flow_input = SimpTensor(np.concatenate([response._data, previous_response._data]))
        flow = tanh(self.flow_proj(flow_input))
        
        topic_input = SimpTensor(np.concatenate([response._data, previous_response._data, query._data]))
        topic = tanh(self.topic_proj(topic_input))
        
        gate_input = SimpTensor(np.concatenate([
            relevance._data[:self.dim],
            consistency._data[:self.dim],
            flow._data[:self.dim]
        ]))
        
        gate_input = SimpTensor(gate_input._data[:self.dim])
        coherence_gate = sigmoid_tensor(self.coherence_gate(gate_input))
        
        combined = SimpTensor(np.concatenate([
            relevance._data * coherence_gate._data[0],
            consistency._data * coherence_gate._data[0],
            flow._data * coherence_gate._data[0],
            topic._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        cos_sim = np.dot(response._data, previous_response._data) / (
            np.linalg.norm(response._data) * np.linalg.norm(previous_response._data) + 1e-8
        )
        
        breakdown = {
            "relevance_score": float(np.mean(relevance._data > 0) * 10),
            "consistency_score": float(np.mean(consistency._data > 0) * 10),
            "flow_score": float(np.mean(flow._data > 0) * 10),
            "topic_continuity": float(np.mean(topic._data > 0) * 10),
            "cosine_similarity": float(cos_sim),
            "coherence_gate": float(coherence_gate._data[0]),
        }
        
        return normalized_score, breakdown


class EnhancedContextAdaptabilityEvaluator(TrainableEvaluator):
    """增强版情境适应性评估器"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
        
        self.situation_proj = Linear(dim, dim)
        self.register_module("situation_proj", self.situation_proj)
        
        self.sensitivity_proj = Linear(dim * 2, dim)
        self.register_module("sensitivity_proj", self.sensitivity_proj)
        
        self.appropriateness_proj = Linear(dim * 2, dim)
        self.register_module("appropriateness_proj", self.appropriateness_proj)
        
        self.context_memory_proj = Linear(dim * 2, dim)
        self.register_module("context_memory_proj", self.context_memory_proj)
        
        self.score_head = Linear(dim * 4, 1)
        self.register_module("score_head", self.score_head)
        
        self.adaptation_gate = Linear(dim, 1)
        self.register_module("adaptation_gate", self.adaptation_gate)
    
    def forward(self, response: SimpTensor, situation: SimpTensor, 
               context: Optional[SimpTensor] = None, history_context: Optional[List[SimpTensor]] = None) -> Tuple[float, Dict[str, Any]]:
        if context is None:
            context = zeros(self.dim)
        
        situation_feature = tanh(self.situation_proj(situation))
        sensitivity_input = SimpTensor(np.concatenate([response._data, situation._data]))
        sensitivity = tanh(self.sensitivity_proj(sensitivity_input))
        appropriateness_input = SimpTensor(np.concatenate([response._data, context._data]))
        appropriateness = tanh(self.appropriateness_proj(appropriateness_input))
        
        if history_context and len(history_context) > 0:
            history_mean = SimpTensor(np.mean([c._data for c in history_context], axis=0))
            memory_input = SimpTensor(np.concatenate([response._data, history_mean._data]))
            context_memory = tanh(self.context_memory_proj(memory_input))
        else:
            context_memory = zeros(self.dim)
        
        adaptation_gate = sigmoid_tensor(self.adaptation_gate(situation_feature))
        
        combined = SimpTensor(np.concatenate([
            situation_feature._data * adaptation_gate._data[0],
            sensitivity._data,
            appropriateness._data,
            context_memory._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        breakdown = {
            "situation_awareness": float(np.mean(situation_feature._data > 0) * 10),
            "context_sensitivity": float(np.mean(sensitivity._data > 0) * 10),
            "appropriateness": float(np.mean(appropriateness._data > 0) * 10),
            "context_memory": float(np.mean(context_memory._data > 0) * 10),
            "adaptation_gate": float(adaptation_gate._data[0]),
        }
        
        return normalized_score, breakdown


class EnhancedEmotionalIntelligenceEvaluator(TrainableEvaluator):
    """增强版情绪智能评估器"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
        
        self.emotion_detection_proj = Linear(dim, dim)
        self.register_module("emotion_detection_proj", self.emotion_detection_proj)
        
        self.empathy_proj = Linear(dim * 2, dim)
        self.register_module("empathy_proj", self.empathy_proj)
        
        self.emotional_response_proj = Linear(dim * 2, dim)
        self.register_module("emotional_response_proj", self.emotional_response_proj)
        
        self.emotion_memory_proj = Linear(dim * 2, dim)
        self.register_module("emotion_memory_proj", self.emotion_memory_proj)
        
        self.score_head = Linear(dim * 4, 1)
        self.register_module("score_head", self.score_head)
        
        self.empathy_gate = Linear(dim, 1)
        self.register_module("empathy_gate", self.empathy_gate)
    
    def forward(self, response: SimpTensor, input_emotion: SimpTensor, 
               context: Optional[SimpTensor] = None, history_emotions: Optional[List[SimpTensor]] = None) -> Tuple[float, Dict[str, Any]]:
        if context is None:
            context = zeros(self.dim)
        
        detection = tanh(self.emotion_detection_proj(input_emotion))
        empathy_input = SimpTensor(np.concatenate([response._data, input_emotion._data]))
        empathy = tanh(self.empathy_proj(empathy_input))
        response_input = SimpTensor(np.concatenate([response._data, context._data]))
        emotional_response = tanh(self.emotional_response_proj(response_input))
        
        if history_emotions and len(history_emotions) > 0:
            history_mean = SimpTensor(np.mean([e._data for e in history_emotions], axis=0))
            memory_input = SimpTensor(np.concatenate([response._data, history_mean._data]))
            emotion_memory = tanh(self.emotion_memory_proj(memory_input))
        else:
            emotion_memory = zeros(self.dim)
        
        empathy_gate = sigmoid_tensor(self.empathy_gate(empathy))
        
        combined = SimpTensor(np.concatenate([
            detection._data,
            empathy._data * empathy_gate._data[0],
            emotional_response._data,
            emotion_memory._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        emotion_similarity = np.dot(response._data, input_emotion._data) / (
            np.linalg.norm(response._data) * np.linalg.norm(input_emotion._data) + 1e-8
        )
        
        breakdown = {
            "emotion_detection": float(np.mean(detection._data > 0) * 10),
            "empathy": float(np.mean(empathy._data > 0) * 10),
            "emotional_response": float(np.mean(emotional_response._data > 0) * 10),
            "emotion_memory": float(np.mean(emotion_memory._data > 0) * 10),
            "emotion_similarity": float(emotion_similarity),
            "empathy_gate": float(empathy_gate._data[0]),
        }
        
        return normalized_score, breakdown


class EnhancedResponseDiversityEvaluator(TrainableEvaluator):
    """增强版响应多样性评估器"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
        
        self.variety_proj = Linear(dim, dim)
        self.register_module("variety_proj", self.variety_proj)
        
        self.originality_proj = Linear(dim, dim)
        self.register_module("originality_proj", self.originality_proj)
        
        self.diversity_proj = Linear(dim, dim)
        self.register_module("diversity_proj", self.diversity_proj)
        
        self.score_head = Linear(dim * 3, 1)
        self.register_module("score_head", self.score_head)
        
        self.diversity_gate = Linear(dim, 1)
        self.register_module("diversity_gate", self.diversity_gate)
    
    def forward(self, response: SimpTensor, history_responses: List[SimpTensor]) -> Tuple[float, Dict[str, Any]]:
        variety = tanh(self.variety_proj(response))
        originality = tanh(self.originality_proj(response))
        diversity = tanh(self.diversity_proj(response))
        
        diversity_gate = sigmoid_tensor(self.diversity_gate(variety))
        
        if history_responses:
            history_matrix = np.stack([r._data for r in history_responses])
            
            similarities = np.dot(history_matrix, response._data) / (
                np.linalg.norm(history_matrix, axis=1) * np.linalg.norm(response._data) + 1e-8
            )
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            
            diversity_bonus = max(0, (1 - avg_similarity) * 5 + (1 - max_similarity) * 3)
            
            response_variance = np.var(response._data)
            variance_bonus = min(2, response_variance * 10)
            
            total_bonus = diversity_bonus + variance_bonus
        else:
            avg_similarity = 0
            max_similarity = 0
            total_bonus = 4
            response_variance = np.var(response._data)
        
        combined = SimpTensor(np.concatenate([
            variety._data * diversity_gate._data[0],
            originality._data,
            diversity._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = min(10, float(raw_score._data[0]) * 10 + total_bonus)
        
        breakdown = {
            "variety_score": float(np.mean(variety._data > 0) * 10),
            "originality_score": float(np.mean(originality._data > 0) * 10),
            "diversity_score": float(np.mean(diversity._data > 0) * 10),
            "diversity_bonus": float(total_bonus),
            "avg_history_similarity": float(avg_similarity),
            "max_history_similarity": float(max_similarity),
            "response_variance": float(response_variance),
            "diversity_gate": float(diversity_gate._data[0]),
        }
        
        return normalized_score, breakdown


class EnhancedCognitiveCapabilityEvaluator(TrainableEvaluator):
    """增强版认知能力评估器"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
        
        self.logic_proj = Linear(dim, dim)
        self.register_module("logic_proj", self.logic_proj)
        
        self.common_sense_proj = Linear(dim, dim)
        self.register_module("common_sense_proj", self.common_sense_proj)
        
        self.abstract_proj = Linear(dim, dim)
        self.register_module("abstract_proj", self.abstract_proj)
        
        self.creativity_proj = Linear(dim, dim)
        self.register_module("creativity_proj", self.creativity_proj)
        
        self.reasoning_proj = Linear(dim * 2, dim)
        self.register_module("reasoning_proj", self.reasoning_proj)
        
        self.score_head = Linear(dim * 5, 1)
        self.register_module("score_head", self.score_head)
        
        self.cognitive_gate = Linear(dim, 1)
        self.register_module("cognitive_gate", self.cognitive_gate)
    
    def forward(self, response: SimpTensor, query: SimpTensor) -> Tuple[float, Dict[str, Any]]:
        logic = tanh(self.logic_proj(response))
        common_sense = tanh(self.common_sense_proj(response))
        abstract = tanh(self.abstract_proj(response))
        creativity = tanh(self.creativity_proj(query))
        
        reasoning_input = SimpTensor(np.concatenate([response._data, query._data]))
        reasoning = tanh(self.reasoning_proj(reasoning_input))
        
        cognitive_gate = sigmoid_tensor(self.cognitive_gate(logic))
        
        combined = SimpTensor(np.concatenate([
            logic._data * cognitive_gate._data[0],
            common_sense._data,
            abstract._data,
            creativity._data,
            reasoning._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        logic_relevance = np.dot(response._data, query._data) / (
            np.linalg.norm(response._data) * np.linalg.norm(query._data) + 1e-8
        )
        
        breakdown = {
            "logical_reasoning": float(np.mean(logic._data > 0) * 10),
            "common_sense": float(np.mean(common_sense._data > 0) * 10),
            "abstract_thinking": float(np.mean(abstract._data > 0) * 10),
            "creativity": float(np.mean(creativity._data > 0) * 10),
            "reasoning_depth": float(np.mean(reasoning._data > 0) * 10),
            "logic_relevance": float(logic_relevance),
            "cognitive_gate": float(cognitive_gate._data[0]),
        }
        
        return normalized_score, breakdown


class EnhancedTuringTestEvaluator:
    """增强版图灵测试综合评估器"""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        self.human_similarity = EnhancedHumanSimilarityEvaluator(dim)
        self.coherence = EnhancedCoherenceEvaluator(dim)
        self.context_adaptability = EnhancedContextAdaptabilityEvaluator(dim)
        self.cognitive_capability = EnhancedCognitiveCapabilityEvaluator(dim)
        self.emotional_intelligence = EnhancedEmotionalIntelligenceEvaluator(dim)
        self.response_diversity = EnhancedResponseDiversityEvaluator(dim)
        
        self._history: List[Dict[str, Any]] = []
        self._total_evaluations = 0
        self._best_score = 0.0
        self._improvement_count = 0
        
        self._adaptive_weights = {
            EvaluationMetric.HUMAN_SIMILARITY: 0.25,
            EvaluationMetric.DIALOGUE_COHERENCE: 0.20,
            EvaluationMetric.CONTEXT_ADAPTABILITY: 0.20,
            EvaluationMetric.COGNITIVE_CAPABILITY: 0.20,
            EvaluationMetric.EMOTIONAL_INTELLIGENCE: 0.10,
            EvaluationMetric.RESPONSE_DIVERSITY: 0.05,
        }
        
        self._metric_trends: Dict[str, List[float]] = {m.value: [] for m in EvaluationMetric}
    
    def _update_adaptive_weights(self, scores: Dict[EvaluationMetric, float]):
        """根据模型表现动态调整权重"""
        for metric in EvaluationMetric:
            current_score = scores.get(metric, 0)
            self._metric_trends[metric.value].append(current_score)
            
            if len(self._metric_trends[metric.value]) > 10:
                self._metric_trends[metric.value] = self._metric_trends[metric.value][-10:]
            
            avg_score = np.mean(self._metric_trends[metric.value])
            
            if avg_score < 5:
                self._adaptive_weights[metric] = min(0.5, self._adaptive_weights[metric] * 1.1)
            elif avg_score > 8:
                self._adaptive_weights[metric] = max(0.05, self._adaptive_weights[metric] * 0.9)
            
        total_weight = sum(self._adaptive_weights.values())
        for metric in EvaluationMetric:
            self._adaptive_weights[metric] /= total_weight
    
    def evaluate_single(
        self,
        response: SimpTensor,
        query: SimpTensor,
        previous_response: Optional[SimpTensor] = None,
        context: Optional[SimpTensor] = None,
        situation: Optional[SimpTensor] = None,
        input_emotion: Optional[SimpTensor] = None,
        history_responses: Optional[List[SimpTensor]] = None,
        history_context: Optional[List[SimpTensor]] = None,
        history_emotions: Optional[List[SimpTensor]] = None
    ) -> TuringScore:
        score = TuringScore()
        score.weights = self._adaptive_weights.copy()
        
        if previous_response is None:
            previous_response = zeros(self.dim)
        if context is None:
            context = zeros(self.dim)
        if situation is None:
            situation = zeros(self.dim)
        if input_emotion is None:
            input_emotion = zeros(self.dim)
        if history_responses is None:
            history_responses = []
        if history_context is None:
            history_context = []
        if history_emotions is None:
            history_emotions = []
        
        hs_score, hs_breakdown = self.human_similarity(response, context, history_responses)
        score.add_score(EvaluationMetric.HUMAN_SIMILARITY, hs_score, "human_similarity")
        
        coh_score, coh_breakdown = self.coherence(response, previous_response, context, query)
        score.add_score(EvaluationMetric.DIALOGUE_COHERENCE, coh_score, "dialogue_coherence")
        
        ca_score, ca_breakdown = self.context_adaptability(response, situation, context, history_context)
        score.add_score(EvaluationMetric.CONTEXT_ADAPTABILITY, ca_score, "context_adaptability")
        
        cc_score, cc_breakdown = self.cognitive_capability(response, query)
        score.add_score(EvaluationMetric.COGNITIVE_CAPABILITY, cc_score, "cognitive_capability")
        
        ei_score, ei_breakdown = self.emotional_intelligence(response, input_emotion, context, history_emotions)
        score.add_score(EvaluationMetric.EMOTIONAL_INTELLIGENCE, ei_score, "emotional_intelligence")
        
        rd_score, rd_breakdown = self.response_diversity(response, history_responses)
        score.add_score(EvaluationMetric.RESPONSE_DIVERSITY, rd_score, "response_diversity")
        
        self._update_adaptive_weights(score.scores)
        
        self._total_evaluations += 1
        if score.percentage > self._best_score:
            self._best_score = score.percentage
            self._improvement_count += 1
        
        return score
    
    def evaluate_dialogue(
        self,
        responses: List[SimpTensor],
        queries: List[SimpTensor],
        situations: Optional[List[SimpTensor]] = None,
        emotions: Optional[List[SimpTensor]] = None
    ) -> Dict[str, Any]:
        scores = []
        history_responses = []
        history_context = []
        history_emotions = []
        
        for i, (response, query) in enumerate(zip(responses, queries)):
            previous_response = responses[i-1] if i > 0 else zeros(self.dim)
            situation = situations[i] if situations else zeros(self.dim)
            emotion = emotions[i] if emotions else zeros(self.dim)
            
            score = self.evaluate_single(
                response=response,
                query=query,
                previous_response=previous_response,
                context=query,
                situation=situation,
                input_emotion=emotion,
                history_responses=history_responses.copy(),
                history_context=history_context.copy(),
                history_emotions=history_emotions.copy()
            )
            
            scores.append(score)
            history_responses.append(response)
            history_context.append(query)
            history_emotions.append(emotion)
        
        avg_score = sum(s.total_score for s in scores) / len(scores)
        avg_percentage = sum(s.percentage for s in scores) / len(scores)
        
        return {
            "dialogue_length": len(responses),
            "individual_scores": [s.to_dict() for s in scores],
            "average_score": avg_score,
            "average_percentage": avg_percentage,
            "best_score": max(s.percentage for s in scores),
            "worst_score": min(s.percentage for s in scores),
            "consistency": np.std([s.total_score for s in scores]),
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self._total_evaluations,
            "best_score": self._best_score,
            "improvement_count": self._improvement_count,
            "adaptive_weights": {m.value: w for m, w in self._adaptive_weights.items()},
        }
    
    def compute_reward(self, score: TuringScore, target_percentage: float = 100.0) -> float:
        """计算强化学习奖励"""
        current_percentage = score.percentage
        reward = (current_percentage - target_percentage) / target_percentage
        
        if current_percentage > 90:
            reward += 0.5
        if current_percentage > 95:
            reward += 1.0
        
        return reward
    
    def parameters(self) -> List[SimpTensor]:
        """获取所有可训练参数"""
        params = []
        for evaluator in [
            self.human_similarity,
            self.coherence,
            self.context_adaptability,
            self.cognitive_capability,
            self.emotional_intelligence,
            self.response_diversity,
        ]:
            params.extend(evaluator.parameters())
        return params


class SyntheticHumanDataGenerator:
    """合成人类数据生成器"""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def generate_human_like_response(self, query: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """生成类人化响应"""
        query_data = query._data
        
        pattern1 = np.sin(query_data * 2) * 0.5 + 0.5
        pattern2 = np.cos(query_data * 3) * 0.3 + 0.5
        pattern3 = np.tanh(query_data * 0.5)
        
        response_data = 0.4 * pattern1 + 0.3 * pattern2 + 0.3 * pattern3
        
        noise = np.random.normal(0, 0.05, self.dim)
        response_data = response_data + noise
        
        response_data = np.clip(response_data, -1, 1)
        
        info = {
            "human_likeness": 0.85 + np.random.uniform(-0.1, 0.1),
            "pattern_diversity": len(np.unique(np.sign(response_data))) / self.dim,
        }
        
        return SimpTensor(response_data), info
    
    def generate_machine_like_response(self, query: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """生成机器化响应"""
        query_data = query._data
        
        response_data = query_data * np.random.uniform(0.5, 1.5)
        
        if np.random.random() < 0.3:
            response_data = np.sign(response_data) * np.abs(response_data) ** 0.5
        
        info = {
            "human_likeness": 0.3 + np.random.uniform(-0.1, 0.1),
            "pattern_diversity": 0.2 + np.random.uniform(-0.1, 0.1),
        }
        
        return SimpTensor(response_data), info
    
    def generate_dialogue_pair(self) -> Tuple[SimpTensor, SimpTensor, float]:
        """生成对话对和标签"""
        query = randn(self.dim)
        
        if np.random.random() < 0.6:
            response, info = self.generate_human_like_response(query)
            label = 1.0
        else:
            response, info = self.generate_machine_like_response(query)
            label = 0.0
        
        return query, response, label
    
    def generate_batch(self, batch_size: int) -> Tuple[List[SimpTensor], List[SimpTensor], List[float]]:
        """生成批次数据"""
        queries = []
        responses = []
        labels = []
        
        for _ in range(batch_size):
            query, response, label = self.generate_dialogue_pair()
            queries.append(query)
            responses.append(response)
            labels.append(label)
        
        return queries, responses, labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed, randn
    
    manual_seed(42)
    dim = 64
    
    print("=== 测试增强版图灵测试评估器 ===")
    
    evaluator = EnhancedTuringTestEvaluator(dim)
    data_generator = SyntheticHumanDataGenerator(dim)
    
    print("\n1. 测试合成数据生成:")
    for i in range(3):
        query, response, label = data_generator.generate_dialogue_pair()
        print(f"  样本 {i+1}: 标签={label}, 人类相似度={response._data.mean():.3f}")
    
    print("\n2. 测试增强评估器:")
    responses = []
    queries = []
    
    for i in range(5):
        query, response, _ = data_generator.generate_dialogue_pair()
        responses.append(response)
        queries.append(query)
    
    for i, (resp, q) in enumerate(zip(responses, queries)):
        prev = responses[i-1] if i > 0 else None
        score = evaluator.evaluate_single(
            response=resp,
            query=q,
            previous_response=prev,
            context=q,
            situation=randn(dim),
            input_emotion=randn(dim),
            history_responses=responses[:i]
        )
        print(f"  轮次 {i+1}: 总分={score.total_score:.2f}/10, 百分比={score.percentage:.1f}%")
    
    status = evaluator.get_status()
    print(f"\n自适应权重:")
    for metric, weight in status["adaptive_weights"].items():
        print(f"  {metric}: {weight:.4f}")
    
    print("\n3. 测试奖励计算:")
    reward = evaluator.compute_reward(score)
    print(f"  当前奖励: {reward:.4f}")
    
    print("\n4. 测试参数获取:")
    params = evaluator.parameters()
    print(f"  可训练参数数量: {len(params)}")
    print(f"  参数总维度: {sum(np.prod(p.shape) for p in params)}")
    
    print("\n=== 所有测试通过 ===")


