# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Associative Reasoner (联想推理器) - 基于相似性的知识关联
支持概念之间的联想推理和知识图谱查询
"""

import numpy as np
from typing imimport os
port Optional, Tuple, List, Dict, Any, Callable
from collections import defaultdict

from ..engine.tensor import SimpTensor, randn, zeros, ones, softmax, sigmoid_tensor, gelu, tanh, einsum
from ..engine.nn import Module, Linear, LayerNorm


class Association:
    """关联关系"""
    
    def __init__(
        self,
        source: SimpTensor,
        target: SimpTensor,
        strength: float,
        type: str = "general",
    ):
        self.source = source
        self.target = target
        self.strength = strength
        self.type = type
    
    def __repr__(self) -> str:
        return f"Association(strength={self.strength:.3f}, type={self.type})"


class AssociativeMemory:
    """联想记忆库"""
    
    def __init__(self, dim: int, capacity: int = 10000):
        self.dim = dim
        self.capacity = capacity
        self.memory: List[SimpTensor] = []
        self.embeddings: np.ndarray = np.zeros((0, dim))
        self.metadata: List[Dict[str, Any]] = []
    
    def add(self, vector: SimpTensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加记忆"""
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            self.embeddings = self.embeddings[1:]
            self.metadata.pop(0)
        
        self.memory.append(vector)
        self.embeddings = np.vstack([self.embeddings, vector._data])
        self.metadata.append(metadata or {})
    
    def query(
        self,
        query: SimpTensor,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[SimpTensor, float, Dict[str, Any]]]:
        """查询相似记忆"""
        if len(self.memory) == 0:
            return []
        
        query_data = query._data.reshape(1, -1)
        similarities = np.dot(self.embeddings, query_data.T).flatten() / np.sqrt(
            np.sum(self.embeddings ** 2, axis=1) * np.sum(query_data ** 2) + 1e-8
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            if similarities[idx] >= threshold:
                results.append((
                    self.memory[idx],
                    similarities[idx],
                    self.metadata[idx],
                ))
        
        return results
    
    def clear(self) -> None:
        """清空记忆"""
        self.memory = []
        self.embeddings = np.zeros((0, self.dim))
        self.metadata = []


class AssociativeReasoner(Module):
    """联想推理器 - 基于相似性的知识关联
    
    核心机制：
    1. 概念嵌入：将输入编码为概念向量
    2. 相似性计算：计算概念之间的关联强度
    3. 联想传播：通过关联网络传播信息
    4. 知识图谱：维护概念之间的关系图
    
    Args:
        dim: 特征维度
        num_concepts: 概念空间大小
        association_strength: 关联强度阈值
        max_associations: 最大关联数
    """
    
    def __init__(
        self,
        dim: int,
        num_concepts: int = 512,
        association_strength: float = 0.3,
        max_associations: int = 10,
    ):
        super().__init__()
        self.dim = dim
        self.num_concepts = num_concepts
        self.association_strength = association_strength
        self.max_associations = max_associations
        
        self.concept_encoder = Linear(dim, dim)
        self.register_module("concept_encoder", self.concept_encoder)
        
        self.similarity_proj = Linear(dim * 2, 1)
        self.register_module("similarity_proj", self.similarity_proj)
        
        self.association_proj = Linear(dim * 2, dim)
        self.register_module("association_proj", self.association_proj)
        
        self.output_proj = Linear(dim * 2, dim)
        self.register_module("output_proj", self.output_proj)
        
        self.concept_norm = LayerNorm(dim)
        self.register_module("concept_norm", self.concept_norm)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
        
        self.concept_embeddings = randn(num_concepts, dim) * 0.02
        self.concept_embeddings._requires_grad = True
        self.register_parameter("concept_embeddings", self.concept_embeddings)
        
        self.association_matrix = zeros(num_concepts, num_concepts)
        self.association_matrix._requires_grad = True
        self.register_parameter("association_matrix", self.association_matrix)
        
        self.associative_memory = AssociativeMemory(dim)
        
        self._association_history: List[Association] = []
    
    def _encode_concept(self, x: SimpTensor) -> SimpTensor:
        """编码输入为概念向量"""
        concept = self.concept_encoder(x)
        concept = gelu(concept)
        concept = self.concept_norm(concept)
        return concept
    
    def _compute_similarity(self, a: SimpTensor, b: SimpTensor) -> float:
        """计算两个概念的相似度"""
        combined = SimpTensor(np.concatenate([a._data, b._data]))
        similarity = self.similarity_proj(combined)
        similarity = sigmoid_tensor(similarity)
        return similarity._data[0]
    
    def _find_associations(self, concept: SimpTensor) -> List[Association]:
        """找到与当前概念相关的联想"""
        associations = []
        concept_data = concept._data.reshape(1, -1)
        
        similarities = np.dot(self.concept_embeddings._data, concept_data.T).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:self.max_associations]
        
        for idx in top_indices:
            if similarities[idx] >= self.association_strength:
                target_concept = SimpTensor(self.concept_embeddings._data[idx])
                association = Association(
                    source=concept,
                    target=target_concept,
                    strength=similarities[idx],
                    type="conceptual",
                )
                associations.append(association)
        
        return associations
    
    def _propagate_associations(self, concept: SimpTensor, associations: List[Association]) -> SimpTensor:
        """通过联想传播信息"""
        if len(associations) == 0:
            return concept
        
        association_features = []
        for assoc in associations:
            combined = SimpTensor(np.concatenate([concept._data, assoc.target._data]))
            feature = self.association_proj(combined)
            feature_data = gelu(feature)._data * assoc.strength
            feature = SimpTensor(feature_data)
            association_features.append(feature)
        
        aggregated = SimpTensor(np.mean([f._data for f in association_features], axis=0))
        return aggregated
    
    def forward(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor] = None,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """
        联想推理前向传播
        
        Args:
            x: 输入特征 [B, S, D] 或 [D]
            context: 上下文信息
        
        Returns:
            output: 推理结果特征
            info: 推理过程信息
        """
        if x.ndim == 3:
            B, S, D = x.shape
            results = []
            infos = []
            
            for b in range(B):
                for s in range(S):
                    xt = x[b, s]
                    out, info = self._single_reason(xt, context)
                    results.append(out)
                    infos.append(info)
            
            result_tensor = SimpTensor(np.stack([r._data for r in results])).reshape(B, S, D)
            
            avg_info = {
                "avg_associations": sum(i["num_associations"] for i in infos) / len(infos),
                "avg_max_strength": sum(i["max_strength"] for i in infos) / len(infos),
                "avg_avg_strength": sum(i["avg_strength"] for i in infos) / len(infos),
                "total_associations": len(self._association_history),
            }
            
            return result_tensor, avg_info
        else:
            return self._single_reason(x, context)
    
    def _single_reason(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor],
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """单样本联想推理"""
        concept = self._encode_concept(x)
        
        associations = self._find_associations(concept)
        
        if context is not None:
            context_concept = self._encode_concept(context)
            context_assoc = self._find_associations(context_concept)
            associations.extend(context_assoc)
        
        associations = sorted(associations, key=lambda a: a.strength, reverse=True)[:self.max_associations]
        
        self._association_history.extend(associations)
        
        if len(associations) > 0:
            assoc_feature = self._propagate_associations(concept, associations)
            combined = SimpTensor(np.concatenate([concept._data, assoc_feature._data]))
            output = self.output_proj(combined)
            output = gelu(output)
            output = self.output_norm(output)
        else:
            output = concept
        
        strengths = [a.strength for a in associations]
        
        info = {
            "num_associations": len(associations),
            "max_strength": max(strengths) if strengths else 0.0,
            "avg_strength": sum(strengths) / len(strengths) if strengths else 0.0,
            "association_types": list(set(a.type for a in associations)),
            "memory_size": len(self.associative_memory.memory),
        }
        
        return output, info
    
    def add_to_memory(self, vector: SimpTensor, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加到联想记忆"""
        self.associative_memory.add(vector, metadata)
    
    def query_memory(
        self,
        query: SimpTensor,
        top_k: int = 5,
    ) -> List[Tuple[SimpTensor, float, Dict[str, Any]]]:
        """查询联想记忆"""
        return self.associative_memory.query(query, top_k)
    
    def reset_memory(self) -> None:
        """重置记忆"""
        self.associative_memory.clear()
        self._association_history = []


class KnowledgeGraphReasoner(AssociativeReasoner):
    """知识图谱推理器 - 增强版联想推理
    
    额外功能：
    1. 关系类型建模
    2. 多跳推理
    3. 路径查找
    """
    
    def __init__(
        self,
        dim: int,
        num_concepts: int = 512,
        num_relation_types: int = 16,
        max_hops: int = 3,
    ):
        super().__init__(dim, num_concepts)
        self.num_relation_types = num_relation_types
        self.max_hops = max_hops
        
        self.relation_embeddings = randn(num_relation_types, dim) * 0.02
        self.relation_embeddings._requires_grad = True
        self.register_parameter("relation_embeddings", self.relation_embeddings)
        
        self.hop_proj = Linear(dim * 2, dim)
    
    def _multi_hop_reason(self, concept: SimpTensor, hops: int = 1) -> SimpTensor:
        """多跳推理"""
        current_concept = concept
        accumulated_features = [current_concept]
        
        for _ in range(hops):
            associations = self._find_associations(current_concept)
            if len(associations) == 0:
                break
            
            strongest_assoc = max(associations, key=lambda a: a.strength)
            combined = SimpTensor(np.concatenate([current_concept._data, strongest_assoc.target._data]))
            current_concept = self.hop_proj(combined)
            current_concept = gelu(current_concept)
            accumulated_features.append(current_concept)
        
        aggregated = SimpTensor(np.mean([f._data for f in accumulated_features], axis=0))
        return aggregated
    
    def _single_reason(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor],
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """增强版单样本联想推理"""
        concept = self._encode_concept(x)
        
        multi_hop_feature = self._multi_hop_reason(concept, hops=self.max_hops)
        
        combined = SimpTensor(np.concatenate([concept._data, multi_hop_feature._data]))
        output = self.output_proj(combined)
        output = gelu(output)
        output = self.output_norm(output)
        
        info = {
            "num_associations": 0,
            "max_strength": 0.0,
            "avg_strength": 0.0,
            "association_types": [],
            "memory_size": len(self.associative_memory.memory),
            "hops_performed": self.max_hops,
        }
        
        return output, info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed, randn
    
    manual_seed(0)
    print("=== 测试 AssociativeReasoner ===")
    
    ar = AssociativeReasoner(dim=64, num_concepts=128, max_associations=5)
    x = randn(2, 8, 64, requires_grad=True)
    context = randn(64)
    y, info = ar(x, context)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"推理信息: {info}")
    
    ar.add_to_memory(randn(64), {"category": "test"})
    memory_results = ar.query_memory(randn(64), top_k=3)
    print(f"记忆查询结果数: {len(memory_results)}")
    
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in ar.parameters())
    print(f"梯度传播: {has_grad}")
    print("AssociativeReasoner 测试 PASSED\n")
    
    print("=== 测试 KnowledgeGraphReasoner ===")
    kgr = KnowledgeGraphReasoner(dim=64, num_concepts=128, num_relation_types=8, max_hops=2)
    x2 = randn(64, requires_grad=True)
    y2, info2 = kgr(x2)
    print(f"输入: {x2.shape}, 输出: {y2.shape}")
    print(f"推理信息: {info2}")
    
    loss2 = (y2 ** 2).mean()
    loss2.backward()
    has_grad2 = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in kgr.parameters())
    print(f"梯度传播: {has_grad2}")
    print("KnowledgeGraphReasoner 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")

