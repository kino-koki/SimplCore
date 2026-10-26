# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Knowledge Self-Iteration (知识自迭代) - 每15分钟后台自动更新
实现知识的持续进化和自我完善机制

核心改进：
1. 从模型输出自动学习新知识
2. 智能置信度评估（结合使用频率、质量、时间衰减）
3. 多跳知识推理
4. 与认知推理器深度集成
5. 知识图谱构建与关系维护
"""
from __future__ import annotations

import numpy as np
import threading
import time
import json
import os
import uuid
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict

from ..engine.tensor import SimpTensor, randn, zeros, ones, softmax, sigmoid_tensor, gelu, tanh, relu
from ..engine.nn import Module, Linear, LayerNorm


class KnowledgeStatus(Enum):
    NEW = "new"
    ACTIVE = "active"
    STALE = "stale"
    OUTDATED = "outdated"
    CONFLICTED = "conflicted"


class KnowledgeItem:
    def __init__(
        self,
        id: str,
        embedding: SimpTensor,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        usage_count: int = 0,
        last_accessed: Optional[datetime] = None,
        status: KnowledgeStatus = KnowledgeStatus.ACTIVE,
        creation_time: Optional[datetime] = None,
        quality_score: float = 0.5,
        related_ids: Optional[List[str]] = None,
    ):
        self.id = id
        self.embedding = embedding
        self.content = content
        self.metadata = metadata or {}
        self.confidence = confidence
        self.usage_count = usage_count
        self.last_accessed = last_accessed or datetime.now()
        self.status = status
        self.creation_time = creation_time or datetime.now()
        self.quality_score = quality_score
        self.related_ids = related_ids or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "embedding": self.embedding._data.tolist(),
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "last_accessed": self.last_accessed.isoformat(),
            "status": self.status.value,
            "creation_time": self.creation_time.isoformat(),
            "quality_score": self.quality_score,
            "related_ids": self.related_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeItem":
        return cls(
            id=data["id"],
            embedding=SimpTensor(np.array(data["embedding"])),
            content=data["content"],
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0),
            usage_count=data.get("usage_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            status=KnowledgeStatus(data.get("status", "active")),
            creation_time=datetime.fromisoformat(data.get("creation_time", datetime.now().isoformat())),
            quality_score=data.get("quality_score", 0.5),
            related_ids=data.get("related_ids", []),
        )


class KnowledgeBase:
    def __init__(self, dim: int, capacity: int = 10000):
        self.dim = dim
        self.capacity = capacity
        self.knowledge: Dict[str, KnowledgeItem] = {}
        self.embeddings: np.ndarray = np.zeros((0, dim))
        self.id_to_index: Dict[str, int] = {}
        self._lock = threading.RLock()

    def add(self, item: KnowledgeItem) -> None:
        with self._lock:
            if item.id in self.knowledge:
                self.remove(item.id)

            if len(self.knowledge) >= self.capacity:
                self._evict_lowest_priority()

            idx = len(self.knowledge)
            self.knowledge[item.id] = item
            self.embeddings = np.vstack([self.embeddings, item.embedding._data])
            self.id_to_index[item.id] = idx

    def _evict_lowest_priority(self) -> None:
        with self._lock:
            lowest_priority = float("inf")
            lowest_id = None

            for id, item in self.knowledge.items():
                priority = self._calculate_priority(item)
                if priority < lowest_priority:
                    lowest_priority = priority
                    lowest_id = id

            if lowest_id:
                self.remove(lowest_id)

    def _calculate_priority(self, item: KnowledgeItem) -> float:
        age_days = (datetime.now() - item.creation_time).days
        recency_score = 1.0 / (1.0 + age_days * 0.1)
        usage_score = min(item.usage_count * 0.1, 1.0)
        confidence_score = item.confidence

        return recency_score * 0.3 + usage_score * 0.4 + confidence_score * 0.3

    def get(self, id: str) -> Optional[KnowledgeItem]:
        with self._lock:
            if id in self.knowledge:
                self.knowledge[id].last_accessed = datetime.now()
                self.knowledge[id].usage_count += 1
                return self.knowledge[id]
            return None

    def remove(self, id: str) -> None:
        with self._lock:
            if id in self.knowledge:
                idx = self.id_to_index[id]
                del self.knowledge[id]
                del self.id_to_index[id]

                mask = np.ones(len(self.embeddings), dtype=bool)
                mask[idx] = False
                self.embeddings = self.embeddings[mask]

                for k, v in self.id_to_index.items():
                    if v > idx:
                        self.id_to_index[k] = v - 1

    def query(
        self,
        query: SimpTensor,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[KnowledgeItem, float]]:
        with self._lock:
            if len(self.knowledge) == 0:
                return []

            query_data = query._data.reshape(1, -1)
            similarities = np.dot(self.embeddings, query_data.T).flatten() / np.sqrt(
                np.sum(self.embeddings ** 2, axis=1) * np.sum(query_data ** 2) + 1e-8
            )

            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = []

            for idx in top_indices:
                if similarities[idx] >= threshold:
                    for id, i in self.id_to_index.items():
                        if i == idx:
                            item = self.knowledge[id]
                            item.last_accessed = datetime.now()
                            item.usage_count += 1
                            results.append((item, similarities[idx]))
                            break

            return results

    def query_by_relation(self, source_id: str, relation_type: str = None) -> List[Tuple[KnowledgeItem, float]]:
        with self._lock:
            if source_id not in self.knowledge:
                return []

            source_item = self.knowledge[source_id]
            results = []

            for related_id in source_item.related_ids:
                if related_id in self.knowledge:
                    related_item = self.knowledge[related_id]
                    similarity = float(np.dot(source_item.embedding._data, related_item.embedding._data) / (
                        np.linalg.norm(source_item.embedding._data) * np.linalg.norm(related_item.embedding._data) + 1e-8
                    ))
                    results.append((related_item, similarity))

            return sorted(results, key=lambda x: x[1], reverse=True)

    def update_confidence(self, id: str, confidence: float) -> None:
        with self._lock:
            if id in self.knowledge:
                self.knowledge[id].confidence = confidence
                if confidence < 0.3:
                    self.knowledge[id].status = KnowledgeStatus.STALE
                elif confidence < 0.1:
                    self.knowledge[id].status = KnowledgeStatus.OUTDATED

    def update_quality(self, id: str, quality_score: float) -> None:
        with self._lock:
            if id in self.knowledge:
                self.knowledge[id].quality_score = quality_score
                alpha = 0.7
                self.knowledge[id].confidence = alpha * self.knowledge[id].confidence + (1 - alpha) * quality_score

    def add_relation(self, id1: str, id2: str) -> None:
        with self._lock:
            if id1 in self.knowledge and id2 in self.knowledge:
                if id2 not in self.knowledge[id1].related_ids:
                    self.knowledge[id1].related_ids.append(id2)
                if id1 not in self.knowledge[id2].related_ids:
                    self.knowledge[id2].related_ids.append(id1)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self.knowledge)
            active = sum(1 for k in self.knowledge if self.knowledge[k].status == KnowledgeStatus.ACTIVE)
            avg_confidence = sum(self.knowledge[k].confidence for k in self.knowledge) / max(total, 1)
            avg_usage = sum(self.knowledge[k].usage_count for k in self.knowledge) / max(total, 1)
            avg_quality = sum(self.knowledge[k].quality_score for k in self.knowledge) / max(total, 1)

            return {
                "total": total,
                "active": active,
                "avg_confidence": avg_confidence,
                "avg_usage": avg_usage,
                "avg_quality": avg_quality,
            }

    def save(self, path: str) -> None:
        with self._lock:
            data = {
                "dim": self.dim,
                "capacity": self.capacity,
                "knowledge": [item.to_dict() for item in self.knowledge.values()],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return

        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.dim = data.get("dim", self.dim)
            self.capacity = data.get("capacity", self.capacity)
            self.knowledge = {}
            self.embeddings = np.zeros((0, self.dim))
            self.id_to_index = {}

            for item_data in data.get("knowledge", []):
                item = KnowledgeItem.from_dict(item_data)
                self.add(item)


class KnowledgeSelfIteration(Module):
    """知识自迭代机制 - 增强版

    核心机制：
    1. 知识嵌入：将新知识编码为向量
    2. 知识融合：将新知识与现有知识融合
    3. 知识评估：智能评估知识质量和置信度（结合使用频率、质量评分、时间衰减）
    4. 知识更新：定期更新和清理知识库
    5. 后台线程：每15分钟自动执行更新
    6. 自动学习：从高置信度模型输出中提取新知识
    7. 多跳推理：通过知识图谱进行多跳推理
    """

    def __init__(
        self,
        dim: int,
        update_interval: int = 15,
        knowledge_capacity: int = 10000,
        auto_learn_threshold: float = 0.75,
    ):
        super().__init__()
        self.dim = dim
        self.update_interval = update_interval * 60
        self.knowledge_capacity = knowledge_capacity
        self.auto_learn_threshold = auto_learn_threshold

        self.embedding_proj = Linear(dim, dim)
        self.register_module("embedding_proj", self.embedding_proj)

        self.fusion_proj = Linear(dim * 2, dim)
        self.register_module("fusion_proj", self.fusion_proj)

        self.quality_proj = Linear(dim * 3, 1)
        self.register_module("quality_proj", self.quality_proj)

        self.relation_proj = Linear(dim * 2, 1)
        self.register_module("relation_proj", self.relation_proj)

        self.multi_hop_proj = Linear(dim * 4, dim)
        self.register_module("multi_hop_proj", self.multi_hop_proj)

        self.embedding_norm = LayerNorm(dim)
        self.register_module("embedding_norm", self.embedding_norm)

        self.knowledge_base = KnowledgeBase(dim, knowledge_capacity)

        self._update_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_update = datetime.now()
        self._update_count = 0

        self._pending_updates: List[Dict[str, Any]] = []
        self._update_lock = threading.RLock()

        self._learning_history: List[Dict[str, Any]] = []
        self._total_learned = 0

    def _embed_knowledge(self, x: SimpTensor) -> SimpTensor:
        embedding = self.embedding_proj(x)
        embedding = gelu(embedding)
        embedding = self.embedding_norm(embedding)
        norm = np.sqrt(np.sum(embedding._data ** 2) + 1e-8)
        embedding._data = embedding._data / norm
        return embedding

    def _assess_quality(self, embedding: SimpTensor, context: Optional[SimpTensor] = None) -> float:
        if context is None:
            context = zeros(self.dim)

        emb_data = embedding._data.flatten()
        ctx_data = context._data.flatten()
        
        if len(emb_data) < self.dim:
            emb_data = np.pad(emb_data, (0, self.dim - len(emb_data)))
        elif len(emb_data) > self.dim:
            emb_data = emb_data[:self.dim]
            
        if len(ctx_data) < self.dim:
            ctx_data = np.pad(ctx_data, (0, self.dim - len(ctx_data)))
        elif len(ctx_data) > self.dim:
            ctx_data = ctx_data[:self.dim]
        
        combined_data = np.concatenate([emb_data, ctx_data, emb_data * ctx_data])
        combined = SimpTensor(combined_data, requires_grad=True)
        quality = self.quality_proj(combined)
        quality = sigmoid_tensor(quality)
        return quality._data[0]

    def _fuse_knowledge(self, existing: SimpTensor, new: SimpTensor) -> SimpTensor:
        combined = SimpTensor(np.concatenate([existing._data, new._data]))
        fused = self.fusion_proj(combined)
        fused = gelu(fused)
        norm = np.sqrt(np.sum(fused._data ** 2) + 1e-8)
        fused._data = fused._data / norm
        return fused

    def _detect_relation(self, embedding1: SimpTensor, embedding2: SimpTensor) -> float:
        combined = SimpTensor(np.concatenate([embedding1._data, embedding2._data]))
        relation_score = self.relation_proj(combined)
        relation_score = sigmoid_tensor(relation_score)
        return relation_score._data[0]

    def add_knowledge(
        self,
        content: str,
        embedding: SimpTensor,
        metadata: Optional[Dict[str, Any]] = None,
        quality_score: float = None,
    ) -> str:
        embedded = self._embed_knowledge(embedding)

        if quality_score is None:
            quality_score = self._assess_quality(embedded)

        existing = self.knowledge_base.query(embedded, top_k=1, threshold=0.85)

        if existing:
            existing_item, similarity = existing[0]
            fused = self._fuse_knowledge(existing_item.embedding, embedded)

            new_confidence = (existing_item.confidence + quality_score) / 2
            new_quality = (existing_item.quality_score + quality_score) / 2

            updated_item = KnowledgeItem(
                id=existing_item.id,
                embedding=fused,
                content=f"{existing_item.content}\n\n{content}",
                metadata={**existing_item.metadata, **(metadata or {})},
                confidence=new_confidence,
                usage_count=existing_item.usage_count,
                last_accessed=datetime.now(),
                status=KnowledgeStatus.ACTIVE,
                quality_score=new_quality,
                related_ids=existing_item.related_ids,
            )
            self.knowledge_base.add(updated_item)
            return existing_item.id
        else:
            item_id = str(uuid.uuid4())[:8]
            new_item = KnowledgeItem(
                id=item_id,
                embedding=embedded,
                content=content,
                metadata=metadata or {},
                confidence=quality_score,
                usage_count=0,
                last_accessed=datetime.now(),
                status=KnowledgeStatus.NEW,
                quality_score=quality_score,
                related_ids=[],
            )
            self.knowledge_base.add(new_item)

            self._find_and_link_relations(item_id, embedded)

            return item_id

    def _find_and_link_relations(self, new_id: str, new_embedding: SimpTensor) -> None:
        existing = self.knowledge_base.query(new_embedding, top_k=5, threshold=0.5)
        for item, similarity in existing:
            if item.id != new_id:
                relation_score = self._detect_relation(new_embedding, item.embedding)
                if relation_score > 0.6:
                    self.knowledge_base.add_relation(new_id, item.id)

    def learn_from_output(
        self,
        input_tensor: SimpTensor,
        output_tensor: SimpTensor,
        confidence: float,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """从模型输出自动学习新知识"""
        if confidence < self.auto_learn_threshold:
            return None

        combined_embedding = input_tensor + output_tensor
        if content is None:
            content = f"从推理中学习: 输入特征 -> 输出特征 (置信度: {confidence:.2f})"

        if metadata is None:
            metadata = {}
        metadata["source"] = "auto_learn"
        metadata["confidence"] = confidence
        metadata["learn_time"] = datetime.now().isoformat()

        quality_score = confidence * 0.7 + self._assess_quality(output_tensor, input_tensor) * 0.3
        item_id = self.add_knowledge(content, combined_embedding, metadata, quality_score)

        self._total_learned += 1
        self._learning_history.append({
            "item_id": item_id,
            "confidence": confidence,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
        })

        return item_id

    def query_knowledge(
        self,
        query: SimpTensor,
        top_k: int = 5,
        threshold: float = 0.3,
        include_related: bool = False,
    ) -> List[Tuple[KnowledgeItem, float]]:
        results = self.knowledge_base.query(query, top_k, threshold)

        if include_related and results:
            related_results = []
            for item, similarity in results:
                relations = self.knowledge_base.query_by_relation(item.id)
                related_results.extend(relations)

            combined = results + related_results
            combined.sort(key=lambda x: x[1], reverse=True)
            seen_ids = set()
            unique_results = []
            for item, similarity in combined:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    unique_results.append((item, similarity))
                    if len(unique_results) >= top_k * 2:
                        break
            return unique_results

        return results

    def multi_hop_reasoning(
        self,
        query: SimpTensor,
        max_hops: int = 3,
        top_k_per_hop: int = 3,
        threshold: float = 0.3,
    ) -> Tuple[SimpTensor, List[Dict[str, Any]]]:
        """多跳知识推理"""
        path_history = []
        current_embedding = query
        visited_ids = set()

        for hop in range(max_hops):
            results = self.knowledge_base.query(current_embedding, top_k=top_k_per_hop, threshold=threshold)

            if not results:
                break

            best_item, best_similarity = results[0]

            if best_item.id in visited_ids:
                if len(results) > 1:
                    best_item, best_similarity = results[1]
                else:
                    break

            visited_ids.add(best_item.id)
            path_history.append({
                "hop": hop,
                "knowledge_id": best_item.id,
                "content": best_item.content[:100] + "..." if len(best_item.content) > 100 else best_item.content,
                "similarity": best_similarity,
                "confidence": best_item.confidence,
            })

            relation_embeddings = []
            for rel_item, rel_sim in self.knowledge_base.query_by_relation(best_item.id):
                if rel_item.id not in visited_ids and rel_sim > threshold:
                    relation_embeddings.append(rel_item.embedding._data)

            if relation_embeddings:
                relation_context = np.mean(relation_embeddings, axis=0)
            else:
                relation_context = best_item.embedding._data

            combined_data = np.concatenate([
                current_embedding._data,
                best_item.embedding._data,
                relation_context,
                current_embedding._data * best_item.embedding._data,
            ])
            combined = SimpTensor(combined_data, requires_grad=True)
            current_embedding = self.multi_hop_proj(combined)
            current_embedding = gelu(current_embedding)

        if path_history:
            enhanced = query + 0.2 * current_embedding
        else:
            enhanced = query

        return enhanced, path_history

    def _clean_stale_knowledge(self) -> int:
        removed_count = 0
        with self._update_lock:
            stale_ids = [
                id for id, item in self.knowledge_base.knowledge.items()
                if item.status == KnowledgeStatus.OUTDATED or (
                    item.status == KnowledgeStatus.STALE and
                    (datetime.now() - item.last_accessed).days > 7
                ) or (
                    item.confidence < 0.1 and
                    item.usage_count == 0 and
                    (datetime.now() - item.creation_time).days > 30
                )
            ]

            for id in stale_ids:
                self.knowledge_base.remove(id)
                removed_count += 1

        return removed_count

    def _update_knowledge_confidence(self) -> None:
        with self._update_lock:
            now = datetime.now()
            for item in self.knowledge_base.knowledge.values():
                if item.status == KnowledgeStatus.NEW:
                    item.status = KnowledgeStatus.ACTIVE

                age_days = (now - item.creation_time).days
                recency_factor = 1.0 / (1.0 + age_days * 0.05)

                usage_factor = min(item.usage_count * 0.05, 1.5)

                quality_factor = item.quality_score

                item.confidence = max(0.01, recency_factor * 0.3 + usage_factor * 0.4 + quality_factor * 0.3)

                if item.confidence < 0.3:
                    item.status = KnowledgeStatus.STALE
                elif item.confidence < 0.1:
                    item.status = KnowledgeStatus.OUTDATED

    def _process_pending_updates(self) -> int:
        processed = 0
        with self._update_lock:
            for update in self._pending_updates:
                embedding = SimpTensor(update["embedding"])
                self.add_knowledge(update["content"], embedding, update.get("metadata"))
                processed += 1
            self._pending_updates = []

        return processed

    def _update_loop(self) -> None:
        while self._running:
            try:
                self._perform_update()
            except Exception as e:
                pass

            time.sleep(self.update_interval)

    def _perform_update(self) -> Dict[str, Any]:
        self._last_update = datetime.now()
        self._update_count += 1

        processed = self._process_pending_updates()
        self._update_knowledge_confidence()
        removed = self._clean_stale_knowledge()

        stats = self.knowledge_base.get_stats()

        return {
            "update_time": self._last_update.isoformat(),
            "update_count": self._update_count,
            "processed_updates": processed,
            "removed_stale": removed,
            "knowledge_stats": stats,
            "total_learned": self._total_learned,
        }

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=5)

    def schedule_update(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._update_lock:
            self._pending_updates.append({
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            })

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "last_update": self._last_update.isoformat(),
            "update_count": self._update_count,
            "update_interval": self.update_interval,
            "pending_updates": len(self._pending_updates),
            "knowledge_stats": self.knowledge_base.get_stats(),
            "total_learned": self._total_learned,
            "auto_learn_threshold": self.auto_learn_threshold,
        }

    def save_knowledge(self, path: str) -> None:
        self.knowledge_base.save(path)

    def load_knowledge(self, path: str) -> None:
        self.knowledge_base.load(path)

    def forward(self, x: SimpTensor, mode: str = "enhance") -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播

        Args:
            x: 输入特征
            mode: 推理模式
                - "enhance": 知识增强（默认）
                - "multi_hop": 多跳推理
                - "query": 仅查询
        """
        if mode == "multi_hop":
            enhanced, path_history = self.multi_hop_reasoning(x)
            info = {
                "mode": "multi_hop",
                "path_length": len(path_history),
                "path": path_history,
            }
            return enhanced, info

        results = self.query_knowledge(x, top_k=5)

        if results and mode != "query":
            knowledge_features = []
            weights = []

            for item, similarity in results:
                weight = similarity * item.confidence
                weighted = item.embedding * weight
                knowledge_features.append(weighted)
                weights.append(weight)

            if weights:
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                aggregated_data = sum(
                    f._data * w for f, w in zip(knowledge_features, normalized_weights)
                )
                aggregated = SimpTensor(aggregated_data)
            else:
                aggregated = zeros(self.dim)
        else:
            aggregated = zeros(self.dim)

        enhanced = x + 0.3 * self.embedding_norm(self.embedding_proj(aggregated))

        info = {
            "mode": mode,
            "num_knowledge_found": len(results),
            "knowledge_ids": [item.id for item, _ in results],
            "similarities": [sim for _, sim in results],
            "confidences": [item.confidence for item, _ in results],
        }

        return enhanced, info

    def __del__(self):
        self.stop()


class ContinuousLearningEngine(KnowledgeSelfIteration):
    """持续学习引擎 - 增强版知识自迭代

    额外功能：
    1. 主动学习：从高置信度输出中学习
    2. 知识图谱构建：维护知识之间的关系
    3. 冲突检测：检测和解决知识冲突
    4. 知识推荐：识别知识盲区并建议补充
    """

    def __init__(
        self,
        dim: int,
        update_interval: int = 15,
    ):
        super().__init__(dim, update_interval)

        self.conflict_proj = Linear(dim * 2, 2)
        self.register_module("conflict_proj", self.conflict_proj)

        self.gap_detection_proj = Linear(dim * 2, 1)
        self.register_module("gap_detection_proj", self.gap_detection_proj)

        self._knowledge_relations: Dict[str, List[str]] = defaultdict(list)
        self._recent_queries: List[Tuple[SimpTensor, float, datetime]] = []
        self._max_recent_queries = 100

    def _detect_conflict(
        self,
        existing: KnowledgeItem,
        new_embedding: SimpTensor,
    ) -> Tuple[bool, float]:
        combined = SimpTensor(np.concatenate([existing.embedding._data, new_embedding._data]))
        logits = self.conflict_proj(combined)
        probs = softmax(logits, dim=0)

        is_conflict = probs._data[1] > 0.7
        conflict_prob = probs._data[1]

        return is_conflict, conflict_prob

    def add_knowledge(
        self,
        content: str,
        embedding: SimpTensor,
        metadata: Optional[Dict[str, Any]] = None,
        quality_score: float = None,
    ) -> str:
        embedded = self._embed_knowledge(embedding)

        existing = self.knowledge_base.query(embedded, top_k=3, threshold=0.6)

        for existing_item, similarity in existing:
            is_conflict, conflict_prob = self._detect_conflict(existing_item, embedded)
            if is_conflict:
                existing_item.status = KnowledgeStatus.CONFLICTED
                existing_item.confidence = max(0.01, existing_item.confidence - conflict_prob * 0.3)

        return super().add_knowledge(content, embedding, metadata, quality_score)

    def _record_query(self, query: SimpTensor, confidence: float) -> None:
        self._recent_queries.append((query, confidence, datetime.now()))
        if len(self._recent_queries) > self._max_recent_queries:
            self._recent_queries = self._recent_queries[-self._max_recent_queries:]

    def suggest_knowledge_gaps(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """识别知识盲区并建议补充"""
        gaps = []
        now = datetime.now()

        for query, confidence, timestamp in self._recent_queries:
            if confidence < 0.5 and (now - timestamp).total_seconds() / 3600 < 24:
                gap_score = self._assess_gap_score(query)
                gaps.append({
                    "query_norm": float(np.linalg.norm(query._data)),
                    "confidence": confidence,
                    "gap_score": gap_score,
                    "timestamp": timestamp.isoformat(),
                })

        gaps.sort(key=lambda x: x["gap_score"], reverse=True)

        return gaps[:top_k]

    def _assess_gap_score(self, query: SimpTensor) -> float:
        results = self.knowledge_base.query(query, top_k=3, threshold=0.1)
        if not results:
            return 1.0

        avg_similarity = sum(sim for _, sim in results) / len(results)
        avg_confidence = sum(item.confidence for item, _ in results) / len(results)

        return (1 - avg_similarity) * 0.5 + (1 - avg_confidence) * 0.5

    def forward(self, x: SimpTensor, mode: str = "enhance") -> Tuple[SimpTensor, Dict[str, Any]]:
        enhanced, info = super().forward(x, mode)

        if "confidence" in info:
            self._record_query(x, info["confidence"])
        else:
            self._record_query(x, 0.7 if info.get("num_knowledge_found", 0) > 0 else 0.3)

        return enhanced, info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

    from SimplCore.engine.tensor import manual_seed, randn

    manual_seed(0)
    print("=== 测试 KnowledgeSelfIteration 增强版 ===")

    ksi = KnowledgeSelfIteration(dim=64, update_interval=1)
    ksi.start()

    embedding1 = randn(64)
    id1 = ksi.add_knowledge("测试知识1：人工智能基础概念", embedding1, {"category": "AI"})
    print(f"添加知识1: {id1}")

    embedding2 = randn(64)
    id2 = ksi.add_knowledge("测试知识2：机器学习算法原理", embedding2, {"category": "ML"})
    print(f"添加知识2: {id2}")

    embedding3 = embedding1 * 0.8 + embedding2 * 0.2
    id3 = ksi.add_knowledge("测试知识3：深度学习与神经网络", embedding3, {"category": "DL"})
    print(f"添加知识3: {id3}")

    print(f"知识1关联: {ksi.knowledge_base.get(id1).related_ids}")
    print(f"知识2关联: {ksi.knowledge_base.get(id2).related_ids}")

    learned_id = ksi.learn_from_output(
        input_tensor=randn(64),
        output_tensor=randn(64),
        confidence=0.85,
        content="自动学习的知识：推理输出模式",
    )
    print(f"自动学习知识: {learned_id}")

    query = randn(64)
    results = ksi.query_knowledge(query, top_k=3, include_related=True)
    print(f"查询结果数(含关联): {len(results)}")

    enhanced, info = ksi(query, mode="enhance")
    print(f"知识增强 - 输入: {query.shape}, 输出: {enhanced.shape}")
    print(f"增强信息: {info}")

    multi_hop_result, hop_info = ksi(query, mode="multi_hop")
    print(f"多跳推理 - 路径长度: {hop_info['path_length']}")

    stats = ksi.get_status()
    print(f"状态统计: total={stats['knowledge_stats']['total']}, learned={stats['total_learned']}")

    ksi.stop()
    print("KnowledgeSelfIteration 增强版测试 PASSED\n")

    print("=== 测试 ContinuousLearningEngine ===")
    cle = ContinuousLearningEngine(dim=64, update_interval=1)
    cle.start()

    emb4 = randn(64)
    id4 = cle.add_knowledge("持续学习知识：自适应学习机制", emb4)
    print(f"添加知识: {id4}")

    y, info_y = cle(emb4)
    print(f"输入: {emb4.shape}, 输出: {y.shape}")

    gaps = cle.suggest_knowledge_gaps()
    print(f"知识盲区建议数: {len(gaps)}")

    cle.stop()
    print("ContinuousLearningEngine 测试 PASSED\n")

    print("=== 所有测试通过 ===")

