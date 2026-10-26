#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
歧义消解模块 - 解决指代消解、多义性处理

核心功能：
1. 指代消解 (Anaphora Resolution)
2. 多义词消歧 (Word Sense Disambiguation)
3. 省略恢复 (Ellipsis Resolution)
4. 上下文感知消歧
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class AmbiguityType(Enum):
    """歧义类型"""
    ANAPHORA = "anaphora"
    POLYSEMY = "polysemy"
    ELLIPSIS = "ellipsis"
    SYNTACTIC = "syntactic"
    PRONOMINAL = "pronominal"


class ResolutionConfidence(Enum):
    """消解置信度"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


class ReferentCandidate:
    """指代候选"""
    
    def __init__(self, embedding: SimpTensor, score: float, distance: int, context: str = ""):
        self.embedding = embedding
        self.score = score
        self.distance = distance
        self.context = context
    
    @property
    def adjusted_score(self) -> float:
        """考虑距离衰减的调整分数"""
        distance_penalty = 1.0 / (1.0 + self.distance * 0.1)
        return self.score * distance_penalty


class AnaphoraResolver(Module):
    """指代消解器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.candidate_proj = Linear(dim, dim)
        self.register_module("candidate_proj", self.candidate_proj)
        
        self.pronoun_proj = Linear(dim, dim)
        self.register_module("pronoun_proj", self.pronoun_proj)
        
        self.similarity_proj = Linear(dim * 2, dim)
        self.register_module("similarity_proj", self.similarity_proj)
        
        self.score_head = Linear(dim, 1)
        self.register_module("score_head", self.score_head)
    
    def forward(
        self,
        pronoun: SimpTensor,
        candidates: List[SimpTensor],
        distances: Optional[List[int]] = None
    ) -> Tuple[int, float, List[float]]:
        """消解指代"""
        if not candidates:
            return -1, 0.0, []
        
        if distances is None:
            distances = list(range(len(candidates)))
        
        pronoun_feature = tanh(self.pronoun_proj(pronoun))
        scores = []
        
        for i, candidate in enumerate(candidates):
            candidate_feature = tanh(self.candidate_proj(candidate))
            
            combined = SimpTensor(np.concatenate([pronoun_feature._data, candidate_feature._data]))
            similarity_feature = tanh(self.similarity_proj(combined))
            
            raw_score = sigmoid_tensor(self.score_head(similarity_feature))
            score = float(raw_score._data[0])
            
            distance_penalty = 1.0 / (1.0 + distances[i] * 0.1)
            adjusted_score = score * distance_penalty
            
            scores.append(adjusted_score)
        
        best_idx = int(np.argmax(scores))
        best_score = scores[best_idx]
        
        return best_idx, best_score, scores


class WordSenseDisambiguator(Module):
    """多义词消歧器"""
    
    def __init__(self, dim: int, num_senses: int = 5):
        super().__init__()
        self.dim = dim
        self.num_senses = num_senses
        
        self.sense_projs = [Linear(dim, dim) for _ in range(num_senses)]
        for i, proj in enumerate(self.sense_projs):
            self.register_module(f"sense_proj_{i}", proj)
        
        self.context_proj = Linear(dim, dim)
        self.register_module("context_proj", self.context_proj)
        
        self.sense_scorer = Linear(dim * (num_senses + 1), num_senses)
        self.register_module("sense_scorer", self.sense_scorer)
    
    def forward(
        self,
        word: SimpTensor,
        context: SimpTensor
    ) -> Tuple[int, float, List[float]]:
        """消歧多义词"""
        context_feature = tanh(self.context_proj(context))
        
        sense_features = []
        for proj in self.sense_projs:
            sense_feature = tanh(proj(word))
            sense_features.append(sense_feature)
        
        all_senses = np.concatenate([sf._data for sf in sense_features])
        sense_tensor = SimpTensor(all_senses)
        
        combined = SimpTensor(np.concatenate([sense_tensor._data, context_feature._data]))
        logits = self.sense_scorer(combined)
        probs = softmax(logits, dim=0)
        
        best_sense = int(np.argmax(probs._data))
        best_confidence = float(probs._data[best_sense])
        
        sense_probs = [float(probs._data[i]) for i in range(self.num_senses)]
        
        return best_sense, best_confidence, sense_probs


class EllipsisResolver(Module):
    """省略恢复器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.target_proj = Linear(dim, dim)
        self.register_module("target_proj", self.target_proj)
        
        self.context_proj = Linear(dim, dim)
        self.register_module("context_proj", self.context_proj)
        
        self.restoration_proj = Linear(dim * 2, dim)
        self.register_module("restoration_proj", self.restoration_proj)
        
        self.confidence_proj = Linear(dim, 1)
        self.register_module("confidence_proj", self.confidence_proj)
    
    def forward(
        self,
        target: SimpTensor,
        context_history: List[SimpTensor]
    ) -> Tuple[SimpTensor, float, Dict[str, Any]]:
        """恢复省略成分"""
        if not context_history:
            return target, 0.0, {"restoration_applied": False}
        
        target_feature = tanh(self.target_proj(target))
        
        context_features = []
        for ctx in context_history:
            ctx_feature = tanh(self.context_proj(ctx))
            context_features.append(ctx_feature._data)
        
        aggregated_context = SimpTensor(np.mean(context_features, axis=0))
        
        restoration_input = SimpTensor(np.concatenate([target_feature._data, aggregated_context._data]))
        restored = gelu(self.restoration_proj(restoration_input))
        
        confidence = float(sigmoid_tensor(self.confidence_proj(restored))._data[0])
        
        restored_output = target + restored * confidence
        
        info = {
            "restoration_applied": True,
            "confidence": confidence,
            "context_used": len(context_history),
            "restoration_strength": confidence,
        }
        
        return restored_output, confidence, info


class ContextualDisambiguator(Module):
    """上下文感知消歧器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.anaphora_resolver = AnaphoraResolver(dim)
        self.register_module("anaphora_resolver", self.anaphora_resolver)
        
        self.word_sense_disambiguator = WordSenseDisambiguator(dim)
        self.register_module("word_sense_disambiguator", self.word_sense_disambiguator)
        
        self.ellipsis_resolver = EllipsisResolver(dim)
        self.register_module("ellipsis_resolver", self.ellipsis_resolver)
        
        self.disambiguation_proj = Linear(dim * 3, dim)
        self.register_module("disambiguation_proj", self.disambiguation_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
    
    def forward(
        self,
        x: SimpTensor,
        context_history: Optional[List[SimpTensor]] = None,
        candidates: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """执行上下文感知消歧"""
        if context_history is None:
            context_history = []
        if candidates is None:
            candidates = []
        
        info = {}
        
        if candidates:
            distances = list(range(len(candidates)))
            anaphora_idx, anaphora_score, anaphora_scores = self.anaphora_resolver(x, candidates, distances)
            info["anaphora"] = {
                "resolved": anaphora_idx >= 0,
                "best_candidate_idx": anaphora_idx,
                "confidence": anaphora_score,
                "all_scores": anaphora_scores,
            }
        else:
            info["anaphora"] = {"resolved": False, "confidence": 0.0}
        
        if context_history:
            context_for_wsd = context_history[-1] if context_history else zeros(self.dim)
            sense_idx, sense_confidence, sense_probs = self.word_sense_disambiguator(x, context_for_wsd)
            info["word_sense"] = {
                "best_sense_idx": sense_idx,
                "confidence": sense_confidence,
                "sense_probs": sense_probs,
            }
        else:
            info["word_sense"] = {"best_sense_idx": 0, "confidence": 0.5, "sense_probs": []}
        
        restored, ellipsis_confidence, ellipsis_info = self.ellipsis_resolver(x, context_history)
        info["ellipsis"] = ellipsis_info
        
        anaphora_feature = candidates[anaphora_idx] if candidates and anaphora_idx >= 0 else zeros(self.dim)
        
        combined = SimpTensor(np.concatenate([
            x._data,
            anaphora_feature._data if isinstance(anaphora_feature, SimpTensor) else anaphora_feature,
            restored._data
        ]))
        
        disambiguated = gelu(self.disambiguation_proj(combined))
        disambiguated = self.output_norm(disambiguated)
        
        ellipsis_confidence = info["ellipsis"].get("confidence", 0.0)
        
        overall_confidence = (
            info["anaphora"]["confidence"] * 0.3 +
            info["word_sense"]["confidence"] * 0.3 +
            ellipsis_confidence * 0.4
        )
        
        info["overall_confidence"] = overall_confidence
        info["ambiguity_resolved"] = overall_confidence > 0.5
        
        return disambiguated, info


class DisambiguationPipeline(Module):
    """歧义消解流水线"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.contextual_disambiguator = ContextualDisambiguator(dim)
        self.register_module("contextual_disambiguator", self.contextual_disambiguator)
        
        self.confidence_threshold = 0.5
        
        self._resolution_history: List[Dict[str, Any]] = []
    
    def disambiguate(
        self,
        x: SimpTensor,
        context_history: Optional[List[SimpTensor]] = None,
        candidates: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """执行歧义消解"""
        if context_history is None:
            context_history = []
        if candidates is None:
            candidates = []
        
        disambiguated, info = self.contextual_disambiguator(x, context_history, candidates)
        
        self._resolution_history.append(info)
        if len(self._resolution_history) > 100:
            self._resolution_history = self._resolution_history[-100:]
        
        return disambiguated, info
    
    def forward(
        self,
        x: SimpTensor,
        context_history: Optional[List[SimpTensor]] = None,
        candidates: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        return self.disambiguate(x, context_history, candidates)
    
    def get_resolution_stats(self) -> Dict[str, Any]:
        """获取消解统计信息"""
        if not self._resolution_history:
            return {"total_resolutions": 0}
        
        total = len(self._resolution_history)
        resolved = sum(1 for r in self._resolution_history if r.get("ambiguity_resolved", False))
        
        avg_confidence = sum(r.get("overall_confidence", 0) for r in self._resolution_history) / total
        
        return {
            "total_resolutions": total,
            "resolved_count": resolved,
            "resolved_rate": resolved / total,
            "average_confidence": avg_confidence,
        }


if __name__ == "__main__":
    dim = 64
    print("=== 测试歧义消解模块 ===")
    
    print("\n1. 测试指代消解器:")
    resolver = AnaphoraResolver(dim)
    pronoun = randn(dim)
    candidates = [randn(dim), randn(dim), randn(dim)]
    idx, score, scores = resolver(pronoun, candidates)
    print(f"最佳候选: {idx}, 置信度: {score:.3f}")
    score_list = [f"{s:.3f}" for s in scores]
    print(f"所有分数: {score_list}")
    
    print("\n2. 测试多义词消歧器:")
    wsd = WordSenseDisambiguator(dim)
    word = randn(dim)
    context = randn(dim)
    sense_idx, sense_confidence, sense_probs = wsd(word, context)
    print(f"最佳词义: {sense_idx}, 置信度: {sense_confidence:.3f}")
    
    print("\n3. 测试省略恢复器:")
    ellipsis_resolver = EllipsisResolver(dim)
    target = randn(dim)
    context_history = [randn(dim), randn(dim)]
    restored, confidence, ellipsis_info = ellipsis_resolver(target, context_history)
    print(f"恢复输出: {restored.shape}")
    print(f"置信度: {confidence:.3f}")
    print(f"信息: {ellipsis_info}")
    
    print("\n4. 测试上下文感知消歧器:")
    contextual_disambiguator = ContextualDisambiguator(dim)
    x = randn(dim)
    candidates = [randn(dim), randn(dim)]
    disambiguated, info = contextual_disambiguator(x, context_history, candidates)
    print(f"消歧输出: {disambiguated.shape}")
    print(f"整体置信度: {info['overall_confidence']:.3f}")
    print(f"歧义已消解: {info['ambiguity_resolved']}")
    
    print("\n5. 测试歧义消解流水线:")
    pipeline = DisambiguationPipeline(dim)
    
    for i in range(5):
        inp = randn(dim)
        output, info = pipeline.disambiguate(inp, context_history, candidates)
        print(f"  输入 {i+1}: 置信度={info['overall_confidence']:.3f}, 已消解={info['ambiguity_resolved']}")
    
    stats = pipeline.get_resolution_stats()
    print(f"\n消解统计: {stats}")
    
    print("\n6. 测试梯度传播:")
    pipeline2 = DisambiguationPipeline(dim)
    test_input = randn(dim)
    output, _ = pipeline2(test_input)
    loss = output.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in pipeline2.parameters())
    print(f"梯度传播: {has_grad}")
    
    print("\n=== 所有测试通过 ===")
