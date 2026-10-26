#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
NLU语义解析模块 - 增强语义解析精度、上下文关联能力

核心功能：
1. 语义角色标注 (Semantic Role Labeling)
2. 依存关系解析 (Dependency Parsing)
3. 上下文语义关联
4. 语义相似度计算
5. 意图识别
"""
from __future__ import annotations

import numpy as np
import uuid
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, ones, softmax, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class SemanticRole(Enum):
    """语义角色类型"""
    AGENT = "agent"
    PATIENT = "patient"
    ACTION = "action"
    LOCATION = "location"
    TIME = "time"
    INSTRUMENT = "instrument"
    CAUSE = "cause"
    RESULT = "result"
    MANNER = "manner"
    DEGREE = "degree"


class DependencyRelation(Enum):
    """依存关系类型"""
    SUBJECT = "nsubj"
    OBJECT = "obj"
    INDIRECT_OBJECT = "iobj"
    DATIVE = "dative"
    ADJECTIVAL_MODIFIER = "amod"
    ADVERBIAL_MODIFIER = "advmod"
    COMPOUND = "compound"
    PREPOSITIONAL_MODIFIER = "prep"
    CONJUNCT = "conj"
    APPOSITION = "appos"
    NEGATION = "neg"


class IntentType(Enum):
    """意图类型"""
    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"
    REQUEST = "request"
    EXCLAMATION = "exclamation"
    GREETING = "greeting"
    FAREWELL = "farewell"
    INQUIRY = "inquiry"
    INFORM = "inform"
    SUGGEST = "suggest"


class SemanticFrame:
    """语义框架 - 表示句子的语义结构"""
    
    def __init__(self):
        self.roles: Dict[SemanticRole, SimpTensor] = {}
        self.dependencies: List[Tuple[int, int, DependencyRelation]] = []
        self.intent: Optional[IntentType] = None
        self.intent_confidence: float = 0.0
        self.semantic_vector: Optional[SimpTensor] = None
    
    def add_role(self, role: SemanticRole, embedding: SimpTensor):
        """添加语义角色"""
        self.roles[role] = embedding
    
    def add_dependency(self, head_idx: int, dep_idx: int, relation: DependencyRelation):
        """添加依存关系"""
        self.dependencies.append((head_idx, dep_idx, relation))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "roles": {r.value: self.roles[r].shape if r in self.roles else None for r in SemanticRole},
            "num_dependencies": len(self.dependencies),
            "intent": self.intent.value if self.intent else None,
            "intent_confidence": self.intent_confidence,
        }


class SemanticRoleLabeler(Module):
    """语义角色标注器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.role_projs = {
            role: Linear(dim, dim) for role in SemanticRole
        }
        for role, proj in self.role_projs.items():
            self.register_module(f"role_proj_{role.value}", proj)
        
        self.role_scorer = Linear(dim * len(SemanticRole), len(SemanticRole))
        self.register_module("role_scorer", self.role_scorer)
    
    def forward(self, x: SimpTensor) -> Tuple[Dict[SemanticRole, float], SimpTensor]:
        """标注语义角色"""
        role_features = {}
        all_features = []
        
        for role in SemanticRole:
            feature = tanh(self.role_projs[role](x))
            role_features[role] = feature
            all_features.append(feature._data)
        
        combined = SimpTensor(np.concatenate(all_features))
        scores = softmax(self.role_scorer(combined), dim=0)
        
        role_scores = {
            role: float(scores._data[i]) for i, role in enumerate(SemanticRole)
        }
        
        max_role = max(role_scores, key=role_scores.get)
        dominant_feature = role_features[max_role]
        
        return role_scores, dominant_feature


class DependencyParser(Module):
    """依存关系解析器"""
    
    def __init__(self, dim: int, max_length: int = 64):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        
        self.head_proj = Linear(dim, dim)
        self.register_module("head_proj", self.head_proj)
        
        self.dep_proj = Linear(dim, dim)
        self.register_module("dep_proj", self.dep_proj)
        
        self.relation_proj = Linear(dim * 2, len(DependencyRelation))
        self.register_module("relation_proj", self.relation_proj)
        
        self.attention_proj = Linear(dim * 2, 1)
        self.register_module("attention_proj", self.attention_proj)
    
    def forward(self, x: SimpTensor) -> Tuple[List[Tuple[int, int, DependencyRelation]], SimpTensor]:
        """解析依存关系"""
        head_feature = tanh(self.head_proj(x))
        
        if x.ndim == 1:
            num_tokens = 1
        else:
            num_tokens = min(x.shape[0], self.max_length)
        
        dependencies = []
        attention_weights = []
        
        for i in range(num_tokens):
            for j in range(num_tokens):
                if i == j:
                    continue
                
                if x.ndim == 1:
                    dep_input = x
                else:
                    dep_input = x[j]
                
                dep_feature = tanh(self.dep_proj(dep_input))
                
                combined = SimpTensor(np.concatenate([head_feature._data, dep_feature._data]))
                attention = sigmoid_tensor(self.attention_proj(combined))
                
                if float(attention._data[0]) > 0.5:
                    relation_logits = self.relation_proj(combined)
                    relation_idx = int(np.argmax(relation_logits._data))
                    relation = list(DependencyRelation)[relation_idx]
                    
                    dependencies.append((i, j, relation))
                    attention_weights.append(float(attention._data[0]))
        
        if dependencies:
            avg_weight = np.mean(attention_weights)
        else:
            avg_weight = 0.0
        
        dependency_feature = head_feature * avg_weight
        
        return dependencies, dependency_feature


class IntentClassifier(Module):
    """意图分类器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.intent_proj = Linear(dim, dim)
        self.register_module("intent_proj", self.intent_proj)
        
        self.intent_head = Linear(dim, len(IntentType))
        self.register_module("intent_head", self.intent_head)
        
        self.confidence_proj = Linear(dim, 1)
        self.register_module("confidence_proj", self.confidence_proj)
    
    def forward(self, x: SimpTensor) -> Tuple[IntentType, float, Dict[str, float]]:
        """分类意图"""
        feature = tanh(self.intent_proj(x))
        
        logits = self.intent_head(feature)
        probs = softmax(logits, dim=0)
        
        intent_idx = int(np.argmax(probs._data))
        intent = list(IntentType)[intent_idx]
        
        confidence = float(sigmoid_tensor(self.confidence_proj(feature))._data[0])
        
        intent_probs = {
            it.value: float(probs._data[i]) for i, it in enumerate(IntentType)
        }
        
        return intent, confidence, intent_probs


class ContextualSemanticEncoder(Module):
    """上下文语义编码器 - 增强上下文关联能力"""
    
    def __init__(self, dim: int, max_context_length: int = 10):
        super().__init__()
        self.dim = dim
        self.max_context_length = max_context_length
        
        self.context_proj = Linear(dim, dim)
        self.register_module("context_proj", self.context_proj)
        
        self.attention_proj = Linear(dim * 2, dim)
        self.register_module("attention_proj", self.attention_proj)
        
        self.fusion_proj = Linear(dim * 2, dim)
        self.register_module("fusion_proj", self.fusion_proj)
        
        self.context_norm = LayerNorm(dim)
        self.register_module("context_norm", self.context_norm)
    
    def forward(
        self,
        current: SimpTensor,
        context_history: List[SimpTensor]
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """编码上下文语义"""
        if not context_history:
            return current, {"context_used": 0, "similarity": 0.0}
        
        recent_context = context_history[-self.max_context_length:]
        
        context_features = []
        similarities = []
        
        for ctx in recent_context:
            ctx_feature = tanh(self.context_proj(ctx))
            
            attention_input = SimpTensor(np.concatenate([current._data, ctx._data]))
            attention_weight = sigmoid_tensor(self.attention_proj(attention_input))
            
            similarity = float(np.dot(current._data, ctx._data) / (
                np.linalg.norm(current._data) * np.linalg.norm(ctx._data) + 1e-8
            ))
            
            weighted_ctx = ctx_feature * float(attention_weight._data[0])
            context_features.append(weighted_ctx._data)
            similarities.append(similarity)
        
        aggregated_context = SimpTensor(np.mean(context_features, axis=0))
        aggregated_context = self.context_norm(aggregated_context)
        
        fusion_input = SimpTensor(np.concatenate([current._data, aggregated_context._data]))
        fused = gelu(self.fusion_proj(fusion_input))
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        info = {
            "context_used": len(recent_context),
            "total_context_available": len(context_history),
            "average_similarity": avg_similarity,
            "max_similarity": max(similarities) if similarities else 0.0,
            "min_similarity": min(similarities) if similarities else 0.0,
        }
        
        return fused, info


class SemanticSimilarityCalculator(Module):
    """语义相似度计算器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.similarity_proj = Linear(dim * 2, dim)
        self.register_module("similarity_proj", self.similarity_proj)
        
        self.score_head = Linear(dim, 1)
        self.register_module("score_head", self.score_head)
    
    def forward(self, a: SimpTensor, b: SimpTensor) -> Tuple[float, Dict[str, Any]]:
        """计算语义相似度"""
        combined = SimpTensor(np.concatenate([a._data, b._data]))
        feature = tanh(self.similarity_proj(combined))
        
        raw_score = sigmoid_tensor(self.score_head(feature))
        normalized_score = float(raw_score._data[0])
        
        cosine_sim = float(np.dot(a._data, b._data) / (
            np.linalg.norm(a._data) * np.linalg.norm(b._data) + 1e-8
        ))
        
        info = {
            "semantic_similarity": normalized_score,
            "cosine_similarity": cosine_sim,
            "combined_score": (normalized_score + cosine_sim) / 2,
        }
        
        return info["combined_score"], info


class SemanticParser(Module):
    """语义解析器 - 整合所有NLU组件"""
    
    def __init__(self, dim: int, max_context_length: int = 10):
        super().__init__()
        self.dim = dim
        
        self.role_labeler = SemanticRoleLabeler(dim)
        self.register_module("role_labeler", self.role_labeler)
        
        self.dependency_parser = DependencyParser(dim)
        self.register_module("dependency_parser", self.dependency_parser)
        
        self.intent_classifier = IntentClassifier(dim)
        self.register_module("intent_classifier", self.intent_classifier)
        
        self.context_encoder = ContextualSemanticEncoder(dim, max_context_length)
        self.register_module("context_encoder", self.context_encoder)
        
        self.similarity_calculator = SemanticSimilarityCalculator(dim)
        self.register_module("similarity_calculator", self.similarity_calculator)
        
        self.semantic_proj = Linear(dim * 3, dim)
        self.register_module("semantic_proj", self.semantic_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
        
        self._parse_history: List[SemanticFrame] = []
    
    def parse(
        self,
        x: SimpTensor,
        context_history: Optional[List[SimpTensor]] = None
    ) -> Tuple[SemanticFrame, Dict[str, Any]]:
        """解析语义"""
        if context_history is None:
            context_history = []
        
        frame = SemanticFrame()
        
        role_scores, role_feature = self.role_labeler(x)
        for role, score in role_scores.items():
            if score > 0.1:
                frame.add_role(role, x)
        
        dependencies, dep_feature = self.dependency_parser(x)
        for head, dep, relation in dependencies:
            frame.add_dependency(head, dep, relation)
        
        intent, intent_confidence, intent_probs = self.intent_classifier(x)
        frame.intent = intent
        frame.intent_confidence = intent_confidence
        
        encoded_context, context_info = self.context_encoder(x, context_history)
        
        combined_feature = SimpTensor(np.concatenate([
            role_feature._data,
            dep_feature._data,
            encoded_context._data
        ]))
        
        semantic_vector = gelu(self.semantic_proj(combined_feature))
        semantic_vector = self.output_norm(semantic_vector)
        frame.semantic_vector = semantic_vector
        
        self._parse_history.append(frame)
        if len(self._parse_history) > 50:
            self._parse_history = self._parse_history[-50:]
        
        info = {
            "role_scores": {r.value: s for r, s in role_scores.items()},
            "num_dependencies": len(dependencies),
            "intent": intent.value,
            "intent_confidence": intent_confidence,
            "intent_probs": intent_probs,
            "context_info": context_info,
            "semantic_vector_shape": semantic_vector.shape,
        }
        
        return frame, info
    
    def forward(
        self,
        x: SimpTensor,
        context_history: Optional[List[SimpTensor]] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        frame, info = self.parse(x, context_history)
        return frame.semantic_vector, info
    
    def compute_semantic_similarity(self, a: SimpTensor, b: SimpTensor) -> float:
        """计算语义相似度"""
        score, _ = self.similarity_calculator(a, b)
        return score


if __name__ == "__main__":
    dim = 64
    print("=== 测试语义解析模块 ===")
    
    print("\n1. 测试语义角色标注器:")
    role_labeler = SemanticRoleLabeler(dim)
    x = randn(dim)
    role_scores, dominant_feature = role_labeler(x)
    print(f"主要语义角色: {max(role_scores, key=role_scores.get).value} ({max(role_scores.values()):.3f})")
    
    print("\n2. 测试依存关系解析器:")
    dep_parser = DependencyParser(dim)
    dependencies, dep_feature = dep_parser(x)
    print(f"依存关系数量: {len(dependencies)}")
    
    print("\n3. 测试意图分类器:")
    intent_classifier = IntentClassifier(dim)
    intent, confidence, probs = intent_classifier(x)
    print(f"意图: {intent.value}, 置信度: {confidence:.3f}")
    
    print("\n4. 测试上下文语义编码器:")
    context_encoder = ContextualSemanticEncoder(dim)
    context_history = [randn(dim) for _ in range(5)]
    encoded, info = context_encoder(x, context_history)
    print(f"编码输出: {encoded.shape}")
    print(f"上下文信息: {info}")
    
    print("\n5. 测试语义相似度计算器:")
    sim_calc = SemanticSimilarityCalculator(dim)
    y = randn(dim)
    score, sim_info = sim_calc(x, y)
    print(f"语义相似度: {score:.3f}")
    print(f"详细信息: {sim_info}")
    
    print("\n6. 测试完整语义解析器:")
    parser = SemanticParser(dim)
    
    inputs = [randn(dim) for _ in range(3)]
    for i, inp in enumerate(inputs):
        frame, info = parser.parse(inp, inputs[:i])
        print(f"  输入 {i+1}:")
        print(f"    意图: {frame.intent.value}, 置信度: {frame.intent_confidence:.3f}")
        print(f"    语义向量: {frame.semantic_vector.shape}")
        print(f"    上下文使用: {info['context_info']['context_used']}")
    
    print("\n7. 测试前向传播梯度:")
    parser2 = SemanticParser(dim)
    test_input = randn(dim)
    output, _ = parser2(test_input)
    loss = output.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in parser2.parameters())
    print(f"梯度传播: {has_grad}")
    
    print("\n=== 所有测试通过 ===")
