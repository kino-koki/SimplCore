#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
图灵测试评估框架 - 改良版图灵测试指标体系

评估维度：
1. 人类相似度评分 (1-10分制)
2. 对话连贯性评估
3. 情境适应性测试
4. 认知能力基准测试

终极目标：在图灵测试预测评估体系中稳定获得满分（100%）评级
"""
from __future__ import annotations

import numpy as np
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Callable
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class EvaluationMetric(Enum):
    """评估指标"""
    HUMAN_SIMILARITY = "human_similarity"
    DIALOGUE_COHERENCE = "dialogue_coherence"
    CONTEXT_ADAPTABILITY = "context_adaptability"
    COGNITIVE_CAPABILITY = "cognitive_capability"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    RESPONSE_DIVERSITY = "response_diversity"


class TuringScore:
    """图灵测试评分"""
    
    def __init__(self):
        self.scores: Dict[EvaluationMetric, float] = {}
        self.weights: Dict[EvaluationMetric, float] = {
            EvaluationMetric.HUMAN_SIMILARITY: 0.25,
            EvaluationMetric.DIALOGUE_COHERENCE: 0.20,
            EvaluationMetric.CONTEXT_ADAPTABILITY: 0.20,
            EvaluationMetric.COGNITIVE_CAPABILITY: 0.20,
            EvaluationMetric.EMOTIONAL_INTELLIGENCE: 0.10,
            EvaluationMetric.RESPONSE_DIVERSITY: 0.05,
        }
        self.max_score: int = 10
        self.raw_scores: Dict[str, float] = {}
    
    @property
    def total_score(self) -> float:
        """计算总分 (0-10分制)"""
        weighted_sum = sum(
            self.scores.get(metric, 0) * self.weights.get(metric, 0)
            for metric in EvaluationMetric
        )
        return weighted_sum
    
    @property
    def percentage(self) -> float:
        """计算百分比 (0-100%)"""
        return (self.total_score / self.max_score) * 100
    
    def add_score(self, metric: EvaluationMetric, score: float, raw_name: str = None):
        """添加评分"""
        clipped = max(0, min(self.max_score, score))
        self.scores[metric] = clipped
        if raw_name:
            self.raw_scores[raw_name] = clipped
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "scores": {m.value: self.scores.get(m, 0) for m in EvaluationMetric},
            "weights": {m.value: self.weights.get(m, 0) for m in EvaluationMetric},
            "total_score": self.total_score,
            "percentage": self.percentage,
            "raw_scores": self.raw_scores,
            "max_score": self.max_score,
        }


class HumanSimilarityEvaluator(Module):
    """人类相似度评估器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.style_proj = Linear(dim, dim)
        self.register_module("style_proj", self.style_proj)
        
        self.syntax_proj = Linear(dim, dim)
        self.register_module("syntax_proj", self.syntax_proj)
        
        self.pragmatic_proj = Linear(dim, dim)
        self.register_module("pragmatic_proj", self.pragmatic_proj)
        
        self.score_head = Linear(dim * 3, 1)
        self.register_module("score_head", self.score_head)
        
        self.style_norm = LayerNorm(dim)
        self.register_module("style_norm", self.style_norm)
    
    def forward(self, response: SimpTensor, context: Optional[SimpTensor] = None) -> Tuple[float, Dict[str, Any]]:
        """评估人类相似度"""
        if context is None:
            context = zeros(self.dim)
        
        style_feature = tanh(self.style_norm(self.style_proj(response)))
        syntax_feature = tanh(self.syntax_proj(response))
        pragmatic_feature = tanh(self.pragmatic_proj(context))
        
        combined = SimpTensor(np.concatenate([
            style_feature._data,
            syntax_feature._data,
            pragmatic_feature._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        breakdown = {
            "style_score": float(np.mean(style_feature._data > 0) * 10),
            "syntax_score": float(np.mean(syntax_feature._data > 0) * 10),
            "pragmatic_score": float(np.mean(pragmatic_feature._data > 0) * 10),
        }
        
        return normalized_score, breakdown


class CoherenceEvaluator(Module):
    """对话连贯性评估器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.relevance_proj = Linear(dim * 2, dim)
        self.register_module("relevance_proj", self.relevance_proj)
        
        self.consistency_proj = Linear(dim * 2, dim)
        self.register_module("consistency_proj", self.consistency_proj)
        
        self.flow_proj = Linear(dim * 2, dim)
        self.register_module("flow_proj", self.flow_proj)
        
        self.score_head = Linear(dim * 3, 1)
        self.register_module("score_head", self.score_head)
    
    def forward(
        self,
        response: SimpTensor,
        previous_response: SimpTensor,
        context: Optional[SimpTensor] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """评估对话连贯性"""
        if context is None:
            context = zeros(self.dim)
        
        relevance_input = SimpTensor(np.concatenate([response._data, context._data]))
        relevance = tanh(self.relevance_proj(relevance_input))
        
        consistency_input = SimpTensor(np.concatenate([response._data, previous_response._data]))
        consistency = tanh(self.consistency_proj(consistency_input))
        
        flow_input = SimpTensor(np.concatenate([response._data, previous_response._data]))
        flow = tanh(self.flow_proj(flow_input))
        
        combined = SimpTensor(np.concatenate([
            relevance._data,
            consistency._data,
            flow._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        breakdown = {
            "relevance_score": float(np.mean(relevance._data > 0) * 10),
            "consistency_score": float(np.mean(consistency._data > 0) * 10),
            "flow_score": float(np.mean(flow._data > 0) * 10),
        }
        
        return normalized_score, breakdown


class ContextAdaptabilityEvaluator(Module):
    """情境适应性评估器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.situation_proj = Linear(dim, dim)
        self.register_module("situation_proj", self.situation_proj)
        
        self.sensitivity_proj = Linear(dim * 2, dim)
        self.register_module("sensitivity_proj", self.sensitivity_proj)
        
        self.appropriateness_proj = Linear(dim * 2, dim)
        self.register_module("appropriateness_proj", self.appropriateness_proj)
        
        self.score_head = Linear(dim * 3, 1)
        self.register_module("score_head", self.score_head)
    
    def forward(
        self,
        response: SimpTensor,
        situation: SimpTensor,
        context: Optional[SimpTensor] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """评估情境适应性"""
        if context is None:
            context = zeros(self.dim)
        
        situation_feature = tanh(self.situation_proj(situation))
        sensitivity_input = SimpTensor(np.concatenate([response._data, situation._data]))
        sensitivity = tanh(self.sensitivity_proj(sensitivity_input))
        appropriateness_input = SimpTensor(np.concatenate([response._data, context._data]))
        appropriateness = tanh(self.appropriateness_proj(appropriateness_input))
        
        combined = SimpTensor(np.concatenate([
            situation_feature._data,
            sensitivity._data,
            appropriateness._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        breakdown = {
            "situation_awareness": float(np.mean(situation_feature._data > 0) * 10),
            "context_sensitivity": float(np.mean(sensitivity._data > 0) * 10),
            "appropriateness": float(np.mean(appropriateness._data > 0) * 10),
        }
        
        return normalized_score, breakdown


class CognitiveCapabilityEvaluator(Module):
    """认知能力基准评估器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.logic_proj = Linear(dim, dim)
        self.register_module("logic_proj", self.logic_proj)
        
        self.common_sense_proj = Linear(dim, dim)
        self.register_module("common_sense_proj", self.common_sense_proj)
        
        self.abstract_proj = Linear(dim, dim)
        self.register_module("abstract_proj", self.abstract_proj)
        
        self.creativity_proj = Linear(dim, dim)
        self.register_module("creativity_proj", self.creativity_proj)
        
        self.score_head = Linear(dim * 4, 1)
        self.register_module("score_head", self.score_head)
    
    def forward(self, response: SimpTensor, query: SimpTensor) -> Tuple[float, Dict[str, Any]]:
        """评估认知能力"""
        logic = tanh(self.logic_proj(response))
        common_sense = tanh(self.common_sense_proj(response))
        abstract = tanh(self.abstract_proj(response))
        creativity = tanh(self.creativity_proj(query))
        
        combined = SimpTensor(np.concatenate([
            logic._data,
            common_sense._data,
            abstract._data,
            creativity._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        breakdown = {
            "logical_reasoning": float(np.mean(logic._data > 0) * 10),
            "common_sense": float(np.mean(common_sense._data > 0) * 10),
            "abstract_thinking": float(np.mean(abstract._data > 0) * 10),
            "creativity": float(np.mean(creativity._data > 0) * 10),
        }
        
        return normalized_score, breakdown


class EmotionalIntelligenceEvaluator(Module):
    """情绪智能评估器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.emotion_detection_proj = Linear(dim, dim)
        self.register_module("emotion_detection_proj", self.emotion_detection_proj)
        
        self.empathy_proj = Linear(dim * 2, dim)
        self.register_module("empathy_proj", self.empathy_proj)
        
        self.emotional_response_proj = Linear(dim * 2, dim)
        self.register_module("emotional_response_proj", self.emotional_response_proj)
        
        self.score_head = Linear(dim * 3, 1)
        self.register_module("score_head", self.score_head)
    
    def forward(
        self,
        response: SimpTensor,
        input_emotion: SimpTensor,
        context: Optional[SimpTensor] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """评估情绪智能"""
        if context is None:
            context = zeros(self.dim)
        
        detection = tanh(self.emotion_detection_proj(input_emotion))
        empathy_input = SimpTensor(np.concatenate([response._data, input_emotion._data]))
        empathy = tanh(self.empathy_proj(empathy_input))
        response_input = SimpTensor(np.concatenate([response._data, context._data]))
        emotional_response = tanh(self.emotional_response_proj(response_input))
        
        combined = SimpTensor(np.concatenate([
            detection._data,
            empathy._data,
            emotional_response._data
        ]))
        
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = float(raw_score._data[0]) * 10
        
        breakdown = {
            "emotion_detection": float(np.mean(detection._data > 0) * 10),
            "empathy": float(np.mean(empathy._data > 0) * 10),
            "emotional_response": float(np.mean(emotional_response._data > 0) * 10),
        }
        
        return normalized_score, breakdown


class ResponseDiversityEvaluator(Module):
    """响应多样性评估器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.variety_proj = Linear(dim, dim)
        self.register_module("variety_proj", self.variety_proj)
        
        self.originality_proj = Linear(dim, dim)
        self.register_module("originality_proj", self.originality_proj)
        
        self.score_head = Linear(dim * 2, 1)
        self.register_module("score_head", self.score_head)
    
    def forward(
        self,
        response: SimpTensor,
        history_responses: List[SimpTensor]
    ) -> Tuple[float, Dict[str, Any]]:
        """评估响应多样性"""
        variety = tanh(self.variety_proj(response))
        originality = tanh(self.originality_proj(response))
        
        if history_responses:
            history_matrix = np.stack([r._data for r in history_responses])
            similarities = np.dot(history_matrix, response._data) / (
                np.linalg.norm(history_matrix, axis=1) * np.linalg.norm(response._data) + 1e-8
            )
            avg_similarity = np.mean(similarities)
            diversity_bonus = max(0, (1 - avg_similarity) * 5)
        else:
            diversity_bonus = 3
        
        combined = SimpTensor(np.concatenate([variety._data, originality._data]))
        raw_score = sigmoid_tensor(self.score_head(combined))
        normalized_score = min(10, float(raw_score._data[0]) * 10 + diversity_bonus)
        
        breakdown = {
            "variety_score": float(np.mean(variety._data > 0) * 10),
            "originality_score": float(np.mean(originality._data > 0) * 10),
            "diversity_bonus": diversity_bonus,
            "history_similarity": avg_similarity if history_responses else 0,
        }
        
        return normalized_score, breakdown


class TuringTestEvaluator:
    """图灵测试综合评估器"""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        
        self.human_similarity = HumanSimilarityEvaluator(dim)
        self.coherence = CoherenceEvaluator(dim)
        self.context_adaptability = ContextAdaptabilityEvaluator(dim)
        self.cognitive_capability = CognitiveCapabilityEvaluator(dim)
        self.emotional_intelligence = EmotionalIntelligenceEvaluator(dim)
        self.response_diversity = ResponseDiversityEvaluator(dim)
        
        self._history: List[Dict[str, Any]] = []
        self._total_evaluations = 0
        self._best_score = 0.0
    
    def evaluate_single(
        self,
        response: SimpTensor,
        query: SimpTensor,
        previous_response: Optional[SimpTensor] = None,
        context: Optional[SimpTensor] = None,
        situation: Optional[SimpTensor] = None,
        input_emotion: Optional[SimpTensor] = None,
        history_responses: Optional[List[SimpTensor]] = None
    ) -> TuringScore:
        """单次评估"""
        score = TuringScore()
        
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
        
        hs_score, hs_breakdown = self.human_similarity(response, context)
        score.add_score(EvaluationMetric.HUMAN_SIMILARITY, hs_score, "human_similarity")
        
        coh_score, coh_breakdown = self.coherence(response, previous_response, context)
        score.add_score(EvaluationMetric.DIALOGUE_COHERENCE, coh_score, "dialogue_coherence")
        
        ca_score, ca_breakdown = self.context_adaptability(response, situation, context)
        score.add_score(EvaluationMetric.CONTEXT_ADAPTABILITY, ca_score, "context_adaptability")
        
        cc_score, cc_breakdown = self.cognitive_capability(response, query)
        score.add_score(EvaluationMetric.COGNITIVE_CAPABILITY, cc_score, "cognitive_capability")
        
        ei_score, ei_breakdown = self.emotional_intelligence(response, input_emotion, context)
        score.add_score(EvaluationMetric.EMOTIONAL_INTELLIGENCE, ei_score, "emotional_intelligence")
        
        rd_score, rd_breakdown = self.response_diversity(response, history_responses)
        score.add_score(EvaluationMetric.RESPONSE_DIVERSITY, rd_score, "response_diversity")
        
        self._total_evaluations += 1
        if score.percentage > self._best_score:
            self._best_score = score.percentage
        
        eval_record = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_id": str(uuid.uuid4())[:8],
            "score": score.to_dict(),
            "breakdown": {
                "human_similarity": hs_breakdown,
                "dialogue_coherence": coh_breakdown,
                "context_adaptability": ca_breakdown,
                "cognitive_capability": cc_breakdown,
                "emotional_intelligence": ei_breakdown,
                "response_diversity": rd_breakdown,
            },
        }
        self._history.append(eval_record)
        
        return score
    
    def evaluate_dialogue(
        self,
        responses: List[SimpTensor],
        queries: List[SimpTensor],
        situations: Optional[List[SimpTensor]] = None,
        emotions: Optional[List[SimpTensor]] = None
    ) -> Dict[str, Any]:
        """对话序列评估"""
        scores = []
        history_responses = []
        
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
                history_responses=history_responses.copy()
            )
            
            scores.append(score)
            history_responses.append(response)
        
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
    
    def generate_report(self) -> Dict[str, Any]:
        """生成评估报告"""
        if not self._history:
            return {"error": "No evaluations recorded"}
        
        recent_evaluations = self._history[-10:]
        avg_total = sum(h["score"]["total_score"] for h in recent_evaluations) / len(recent_evaluations)
        avg_percentage = sum(h["score"]["percentage"] for h in recent_evaluations) / len(recent_evaluations)
        
        metric_avgs = {}
        for metric in EvaluationMetric:
            metric_scores = [h["score"]["scores"].get(metric.value, 0) for h in recent_evaluations]
            metric_avgs[metric.value] = sum(metric_scores) / len(metric_scores)
        
        return {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "total_evaluations": self._total_evaluations,
            "best_score": self._best_score,
            "recent_average_score": avg_total,
            "recent_average_percentage": avg_percentage,
            "metric_averages": metric_avgs,
            "evaluation_history": [h["score"] for h in recent_evaluations],
            "status": self._determine_status(avg_percentage),
        }
    
    def _determine_status(self, percentage: float) -> str:
        """确定评估状态"""
        if percentage >= 95:
            return "EXCELLENT"
        elif percentage >= 80:
            return "GOOD"
        elif percentage >= 60:
            return "ACCEPTABLE"
        elif percentage >= 40:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        report = self.generate_report()
        return {
            "total_evaluations": self._total_evaluations,
            "best_score": self._best_score,
            "current_status": report.get("status", "UNKNOWN"),
            "recent_average": report.get("recent_average_percentage", 0),
        }


class TuringTestBenchmark:
    """图灵测试基准测试套件"""
    
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.evaluator = TuringTestEvaluator(dim)
        
        self.benchmark_scenarios = [
            self._create_scenario("日常对话", "问候与闲聊", ["你好", "今天天气怎么样", "最近忙什么"]),
            self._create_scenario("问题解决", "逻辑推理", ["2+2等于多少", "解释什么是人工智能", "如何学习编程"]),
            self._create_scenario("情感支持", "共情与安慰", ["我今天心情不太好", "工作压力很大", "感觉很孤单"]),
            self._create_scenario("创意生成", "创造性思考", ["写一首关于春天的诗", "设计一个新的产品", "编一个有趣的故事"]),
            self._create_scenario("知识问答", "专业知识", ["什么是量子力学", "解释相对论", "介绍深度学习"]),
        ]
    
    def _create_scenario(self, name: str, category: str, prompts: List[str]) -> Dict[str, Any]:
        """创建测试场景"""
        return {
            "name": name,
            "category": category,
            "prompts": prompts,
            "query_tensors": [randn(self.dim) for _ in prompts],
            "situation_tensors": [randn(self.dim) for _ in prompts],
            "emotion_tensors": [randn(self.dim) for _ in prompts],
        }
    
    def run_benchmark(
        self,
        model: Callable[[SimpTensor], Tuple[SimpTensor, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """运行基准测试"""
        results = []
        
        for scenario in self.benchmark_scenarios:
            print(f"\n运行场景: {scenario['name']} ({scenario['category']})")
            
            responses = []
            for i, query_tensor in enumerate(scenario["query_tensors"]):
                response, info = model(query_tensor)
                responses.append(response)
            
            dialogue_result = self.evaluator.evaluate_dialogue(
                responses=responses,
                queries=scenario["query_tensors"],
                situations=scenario["situation_tensors"],
                emotions=scenario["emotion_tensors"]
            )
            
            results.append({
                "scenario": scenario["name"],
                "category": scenario["category"],
                **dialogue_result,
            })
            
            print(f"  平均得分: {dialogue_result['average_score']:.2f}/10")
            print(f"  平均百分比: {dialogue_result['average_percentage']:.1f}%")
        
        overall_avg = sum(r["average_percentage"] for r in results) / len(results)
        
        return {
            "benchmark_results": results,
            "overall_average_percentage": overall_avg,
            "status": self._determine_benchmark_status(overall_avg),
            "evaluator_status": self.evaluator.get_status(),
        }
    
    def _determine_benchmark_status(self, percentage: float) -> str:
        """确定基准测试状态"""
        if percentage >= 95:
            return "TURING_TEST_PASSED"
        elif percentage >= 80:
            return "STRONG_PERFORMANCE"
        elif percentage >= 60:
            return "MODERATE_PERFORMANCE"
        elif percentage >= 40:
            return "WEAK_PERFORMANCE"
        else:
            return "FAILURE"


if __name__ == "__main__":
    dim = 64
    print("=== 测试图灵测试评估框架 ===")
    
    print("\n1. 测试单个评估器:")
    hs = HumanSimilarityEvaluator(dim)
    response = randn(dim)
    context = randn(dim)
    score, breakdown = hs(response, context)
    print(f"人类相似度评分: {score:.2f}/10")
    print(f"细分: {breakdown}")
    
    coh = CoherenceEvaluator(dim)
    prev_response = randn(dim)
    score, breakdown = coh(response, prev_response, context)
    print(f"对话连贯性评分: {score:.2f}/10")
    print(f"细分: {breakdown}")
    
    ca = ContextAdaptabilityEvaluator(dim)
    situation = randn(dim)
    score, breakdown = ca(response, situation, context)
    print(f"情境适应性评分: {score:.2f}/10")
    print(f"细分: {breakdown}")
    
    cc = CognitiveCapabilityEvaluator(dim)
    query = randn(dim)
    score, breakdown = cc(response, query)
    print(f"认知能力评分: {score:.2f}/10")
    print(f"细分: {breakdown}")
    
    ei = EmotionalIntelligenceEvaluator(dim)
    emotion = randn(dim)
    score, breakdown = ei(response, emotion, context)
    print(f"情绪智能评分: {score:.2f}/10")
    print(f"细分: {breakdown}")
    
    rd = ResponseDiversityEvaluator(dim)
    history = [randn(dim), randn(dim)]
    score, breakdown = rd(response, history)
    print(f"响应多样性评分: {score:.2f}/10")
    print(f"细分: {breakdown}")
    
    print("\n2. 测试综合评估器:")
    evaluator = TuringTestEvaluator(dim)
    
    responses = [randn(dim) for _ in range(5)]
    queries = [randn(dim) for _ in range(5)]
    
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
    
    report = evaluator.generate_report()
    print(f"\n评估报告:")
    print(f"  总评估次数: {report['total_evaluations']}")
    print(f"  最佳得分: {report['best_score']:.1f}%")
    print(f"  最近平均: {report['recent_average_percentage']:.1f}%")
    print(f"  当前状态: {report['status']}")
    
    print("\n3. 测试基准测试套件:")
    benchmark = TuringTestBenchmark(dim)
    
    def dummy_model(x):
        return randn(dim), {"confidence": 0.7}
    
    result = benchmark.run_benchmark(dummy_model)
    print(f"\n基准测试结果:")
    print(f"  总体平均: {result['overall_average_percentage']:.1f}%")
    print(f"  状态: {result['status']}")
    
    print("\n=== 所有测试通过 ===")
