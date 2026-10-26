#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长程记忆模块 - 长上下文记忆与引用

核心功能：
1. 记忆存储与检索
2. 记忆索引与相似度匹配
3. 记忆更新与衰减机制
4. 记忆引用与关联
5. 长程上下文管理
"""
from __future__ import annotations

import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class MemoryType(Enum):
    """记忆类型"""
    FACTUAL = "factual"
    CONTEXTUAL = "contextual"
    EMOTIONAL = "emotional"
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"


class MemoryStatus(Enum):
    """记忆状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    RECALLED = "recalled"
    FORGOTTEN = "forgotten"


class MemoryEntry:
    """记忆条目"""
    
    def __init__(
        self,
        content: SimpTensor,
        memory_type: MemoryType = MemoryType.CONTEXTUAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())[:8]
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 1
        self.status = MemoryStatus.ACTIVE
        self.relevance_score = 1.0
    
    def access(self):
        """访问记忆"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def update_relevance(self, score: float):
        """更新相关性分数"""
        self.relevance_score = score


class MemoryRetrievalResult:
    """记忆检索结果"""
    
    def __init__(
        self,
        entries: List[MemoryEntry],
        scores: List[float],
        indices: List[int]
    ):
        self.entries = entries
        self.scores = scores
        self.indices = indices
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Tuple[MemoryEntry, float]:
        return self.entries[idx], self.scores[idx]


class MemoryIndex(Module):
    """记忆索引器"""
    
    def __init__(self, dim: int, max_entries: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_entries = max_entries
        
        self.index_proj = Linear(dim, dim)
        self.register_module("index_proj", self.index_proj)
        
        self.similarity_proj = Linear(dim * 2, dim)
        self.register_module("similarity_proj", self.similarity_proj)
        
        self.score_proj = Linear(dim, 1)
        self.register_module("score_proj", self.score_proj)
    
    def build_index(self, entries: List[SimpTensor]) -> np.ndarray:
        """构建索引"""
        indexed = []
        for entry in entries:
            indexed.append(tanh(self.index_proj(entry))._data)
        return np.array(indexed)
    
    def search(
        self,
        query: SimpTensor,
        index: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[int], List[float]]:
        """搜索索引"""
        query_feature = tanh(self.index_proj(query))._data
        
        scores = []
        for i in range(index.shape[0]):
            entry_feature = index[i]
            combined = SimpTensor(np.concatenate([query_feature, entry_feature]))
            sim_feature = tanh(self.similarity_proj(combined))
            score = float(sigmoid_tensor(self.score_proj(sim_feature))._data[0])
            scores.append(score)
        
        scores_np = np.array(scores)
        top_indices = np.argsort(scores_np)[-top_k:][::-1]
        top_scores = scores_np[top_indices].tolist()
        
        return top_indices.tolist(), top_scores


class MemoryDecay(Module):
    """记忆衰减模型"""
    
    def __init__(self, dim: int, decay_rate: float = 0.01):
        super().__init__()
        self.dim = dim
        self.decay_rate = decay_rate
        
        self.decay_proj = Linear(dim, dim)
        self.register_module("decay_proj", self.decay_proj)
        
        self.time_proj = Linear(1, dim)
        self.register_module("time_proj", self.time_proj)
    
    def calculate_decay(
        self,
        memory: SimpTensor,
        time_since_created: float,
        access_count: int
    ) -> Tuple[SimpTensor, float]:
        """计算衰减"""
        time_feature = tanh(self.time_proj(SimpTensor(np.array([time_since_created]))))
        
        decay_factor = np.exp(-self.decay_rate * time_since_created)
        usage_factor = min(1.0, access_count / 10)
        
        combined_factor = decay_factor * (0.5 + 0.5 * usage_factor)
        
        decayed = memory * combined_factor
        decayed = tanh(self.decay_proj(decayed))
        
        return decayed, combined_factor


class MemoryConsolidation(Module):
    """记忆巩固模型"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.consolidate_proj = Linear(dim * 2, dim)
        self.register_module("consolidate_proj", self.consolidate_proj)
        
        self.merge_proj = Linear(dim * 3, dim)
        self.register_module("merge_proj", self.merge_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
    
    def consolidate(
        self,
        new_memory: SimpTensor,
        existing_memories: List[SimpTensor],
        similarity_threshold: float = 0.7
    ) -> Tuple[SimpTensor, bool, Dict[str, Any]]:
        """巩固记忆"""
        info = {"merged": False, "merged_count": 0, "confidence": 0.0}
        
        if not existing_memories:
            return new_memory, False, info
        
        similarities = []
        for mem in existing_memories:
            sim = float(np.dot(new_memory._data, mem._data) / (
                np.linalg.norm(new_memory._data) * np.linalg.norm(mem._data) + 1e-8
            ))
            similarities.append(sim)
        
        similar_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        
        if similar_indices:
            similar_memories = [existing_memories[i] for i in similar_indices]
            merged = np.mean([mem._data for mem in similar_memories], axis=0)
            
            combined = SimpTensor(np.concatenate([new_memory._data, merged]))
            consolidated = tanh(self.consolidate_proj(combined))
            
            final_input = SimpTensor(np.concatenate([new_memory._data, merged, consolidated._data]))
            final = gelu(self.merge_proj(final_input))
            final = self.norm(final)
            
            info["merged"] = True
            info["merged_count"] = len(similar_indices)
            info["confidence"] = np.mean([similarities[i] for i in similar_indices])
            
            return final, True, info
        
        return new_memory, False, info


class LongRangeMemory(Module):
    """长程记忆系统"""
    
    def __init__(self, dim: int, max_entries: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_entries = max_entries
        
        self.memory_index = MemoryIndex(dim, max_entries)
        self.register_module("memory_index", self.memory_index)
        
        self.memory_decay = MemoryDecay(dim)
        self.register_module("memory_decay", self.memory_decay)
        
        self.memory_consolidation = MemoryConsolidation(dim)
        self.register_module("memory_consolidation", self.memory_consolidation)
        
        self.retrieval_proj = Linear(dim * 2, dim)
        self.register_module("retrieval_proj", self.retrieval_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
        
        self._entries: List[MemoryEntry] = []
        self._index: Optional[np.ndarray] = None
        self._needs_reindex = True
    
    def add_memory(
        self,
        content: SimpTensor,
        memory_type: MemoryType = MemoryType.CONTEXTUAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """添加记忆"""
        if len(self._entries) >= self.max_entries:
            self._prune_memories()
        
        existing_contents = [e.content for e in self._entries if e.memory_type == memory_type]
        consolidated, merged, info = self.memory_consolidation.consolidate(content, existing_contents)
        
        entry = MemoryEntry(consolidated, memory_type, metadata)
        if merged:
            entry.metadata["merged"] = True
            entry.metadata["merged_count"] = info["merged_count"]
        
        self._entries.append(entry)
        self._needs_reindex = True
        
        return entry
    
    def _prune_memories(self):
        """修剪记忆"""
        if not self._entries:
            return
        
        if self.max_entries <= 0:
            self._entries = []
            self._needs_reindex = True
            return
        
        if len(self._entries) <= self.max_entries:
            entries_to_remove = len(self._entries) - (self.max_entries // 2)
            if entries_to_remove <= 0:
                return
        else:
            entries_to_remove = len(self._entries) - self.max_entries
        
        if entries_to_remove <= 0:
            return
        
        entries_to_remove = min(entries_to_remove, len(self._entries) - 1)
        
        if entries_to_remove <= 0:
            return
        
        scores = []
        for entry in self._entries:
            time_since = (datetime.now() - entry.created_at).total_seconds() / 3600
            _, decay_factor = self.memory_decay.calculate_decay(
                entry.content, time_since, entry.access_count
            )
            score = entry.relevance_score * decay_factor
            scores.append(score)
        
        sorted_indices = sorted(range(len(self._entries)), key=lambda i: scores[i])
        to_remove = sorted_indices[:entries_to_remove]
        
        for idx in sorted(to_remove, reverse=True):
            if 0 <= idx < len(self._entries):
                del self._entries[idx]
        
        self._needs_reindex = True
    
    def retrieve(
        self,
        query: SimpTensor,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None
    ) -> MemoryRetrievalResult:
        """检索记忆"""
        filtered_entries = self._entries
        if memory_type:
            filtered_entries = [e for e in self._entries if e.memory_type == memory_type]
        
        if not filtered_entries:
            return MemoryRetrievalResult([], [], [])
        
        contents = [e.content for e in filtered_entries]
        
        if self._needs_reindex or self._index is None:
            self._index = self.memory_index.build_index(contents)
            self._needs_reindex = False
        
        indices, scores = self.memory_index.search(query, self._index, top_k)
        
        retrieved_entries = []
        retrieved_scores = []
        retrieved_indices = []
        
        for idx, score in zip(indices, scores):
            if idx < len(filtered_entries):
                entry = filtered_entries[idx]
                entry.access()
                
                time_since = (datetime.now() - entry.created_at).total_seconds() / 3600
                decayed_content, decay_factor = self.memory_decay.calculate_decay(
                    entry.content, time_since, entry.access_count
                )
                
                adjusted_score = score * decay_factor * entry.relevance_score
                
                entry.update_relevance(adjusted_score)
                
                retrieved_entries.append(entry)
                retrieved_scores.append(adjusted_score)
                retrieved_indices.append(idx)
        
        return MemoryRetrievalResult(retrieved_entries, retrieved_scores, retrieved_indices)
    
    def forward(
        self,
        query: SimpTensor,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        result = self.retrieve(query, top_k, memory_type)
        
        if not result:
            return query, {"retrieved": 0, "scores": [], "memory_types": []}
        
        retrieved_features = []
        for entry, score in result:
            weighted = entry.content * score
            retrieved_features.append(weighted._data)
        
        aggregated = SimpTensor(np.mean(retrieved_features, axis=0))
        
        combined = SimpTensor(np.concatenate([query._data, aggregated._data]))
        output = gelu(self.retrieval_proj(combined))
        output = self.output_norm(output)
        
        info = {
            "retrieved": len(result),
            "scores": [round(s, 4) for s in result.scores],
            "memory_types": [e.memory_type.value for e in result.entries],
            "access_counts": [e.access_count for e in result.entries],
        }
        
        return output, info
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        type_counts = {}
        for entry in self._entries:
            type_counts[entry.memory_type.value] = type_counts.get(entry.memory_type.value, 0) + 1
        
        avg_relevance = np.mean([e.relevance_score for e in self._entries]) if self._entries else 0
        avg_access_count = np.mean([e.access_count for e in self._entries]) if self._entries else 0
        
        return {
            "total_entries": len(self._entries),
            "by_type": type_counts,
            "avg_relevance": round(avg_relevance, 4),
            "avg_access_count": round(avg_access_count, 2),
            "needs_reindex": self._needs_reindex,
        }
    
    def reset(self):
        """重置记忆"""
        self._entries = []
        self._index = None
        self._needs_reindex = True


if __name__ == "__main__":
    dim = 64
    print("=== 测试长程记忆模块 ===")
    
    print("\n1. 测试记忆索引器:")
    mi = MemoryIndex(dim)
    entries = [randn(dim) for _ in range(10)]
    index = mi.build_index(entries)
    print(f"索引形状: {index.shape}")
    
    query = randn(dim)
    indices, scores = mi.search(query, index, top_k=3)
    print(f"搜索结果: 索引={indices}, 分数={[round(s, 4) for s in scores]}")
    
    print("\n2. 测试记忆衰减模型:")
    md = MemoryDecay(dim)
    memory = randn(dim)
    decayed, factor = md.calculate_decay(memory, 24.0, 5)
    print(f"衰减因子: {factor:.4f}, 输出形状: {decayed.shape}")
    
    print("\n3. 测试记忆巩固模型:")
    mc = MemoryConsolidation(dim)
    new_mem = randn(dim)
    existing = [new_mem * 0.9 + randn(dim) * 0.1 for _ in range(3)]
    consolidated, merged, info = mc.consolidate(new_mem, existing)
    print(f"已合并: {merged}, 信息: {info}")
    
    print("\n4. 测试长程记忆系统:")
    lrm = LongRangeMemory(dim, max_entries=50)
    
    for i in range(15):
        content = randn(dim)
        mem_type = MemoryType.CONTEXTUAL if i % 2 == 0 else MemoryType.FACTUAL
        lrm.add_memory(content, mem_type)
    
    stats = lrm.get_memory_stats()
    print(f"记忆统计: {stats}")
    
    for i in range(3):
        query = randn(dim)
        output, info = lrm(query, top_k=3)
        print(f"\n  查询 {i+1}:")
        print(f"    输出形状: {output.shape}")
        print(f"    检索到: {info['retrieved']} 条")
        print(f"    分数: {info['scores']}")
        print(f"    类型: {info['memory_types']}")
    
    print("\n5. 测试梯度传播:")
    lrm2 = LongRangeMemory(dim)
    test_input = randn(dim)
    lrm2.add_memory(test_input)
    output2, _ = lrm2(test_input)
    loss = output2.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in lrm2.parameters())
    print(f"梯度传播: {has_grad}")
    
    print("\n=== 所有测试通过 ===")