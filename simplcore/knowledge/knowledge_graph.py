# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""动态知识图谱模块 - 将知识库升级为图结构表示.

核心功能:
  1. 实体建模 - 支持多种类型的知识实体
  2. 关系建模 - 支持丰富的实体关系类型
  3. 图存储和查询 - 高效的图数据管理
  4. 知识推理 - 基于图结构的推理能力
  5. 动态更新 - 支持知识的增删改
  6. 知识融合 - 从外部数据源融合知识
  7. 可视化支持 - 导出图结构用于可视化

设计目标:
  - 支持百万级实体和关系
  - 支持复杂查询和推理
  - 支持增量学习和动态更新
  - 支持多模态知识表示
  - 支持知识图谱嵌入
"""

from __future__ import annotations

import simplcore.tensor as torch
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict


class KnowledgeEntity:
    """知识实体 - 表示知识图谱中的一个节点."""
    
    def __init__(
        self,
        entity_id: str,
        entity_type: str,
        name: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[torch.Tensor] = None,
    ):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.name = name
        self.description = description
        self.properties = properties or {}
        self.embedding = embedding
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "embedding": self.embedding.numpy().tolist() if self.embedding is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntity":
        """从字典创建实体."""
        embedding = torch.tensor(data["embedding"]) if data.get("embedding") else None
        return cls(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            name=data["name"],
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            embedding=embedding,
        )


class KnowledgeRelation:
    """知识关系 - 表示知识图谱中的一条边."""
    
    RELATION_TYPES = [
        "IS_A",           # 是...的一种
        "PART_OF",        # 是...的一部分
        "HAS_PROPERTY",   # 具有...属性
        "CAUSES",         # 导致
        "FOLLOWS",        # 跟随
        "SIMILAR_TO",     # 类似于
        "OPPOSITE_TO",    # 相反于
        "RELATED_TO",     # 相关于
        "USED_IN",        # 用于
        "DISCOVERED_BY",  # 由...发现
        "APPLIES_TO",     # 适用于
        "DEPENDS_ON",     # 依赖于
        "CONTAINS",       # 包含
        "INTERACTS_WITH", # 与...交互
    ]
    
    def __init__(
        self,
        relation_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.relation_id = relation_id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.weight = weight
        self.description = description
        self.properties = properties or {}
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示."""
        return {
            "relation_id": self.relation_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "description": self.description,
            "properties": self.properties,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeRelation":
        """从字典创建关系."""
        return cls(
            relation_id=data["relation_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            weight=data.get("weight", 1.0),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
        )


class KnowledgeGraph:
    """知识图谱 - 管理实体和关系的图结构."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        
        # 实体存储
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.entity_types: Dict[str, Set[str]] = defaultdict(set)
        
        # 关系存储
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.source_relations: Dict[str, List[str]] = defaultdict(list)
        self.target_relations: Dict[str, List[str]] = defaultdict(list)
        self.type_relations: Dict[str, List[str]] = defaultdict(list)
        
        # 索引
        self.entity_name_index: Dict[str, str] = {}
        
        # 嵌入层
        self.entity_embeddings = torch.nn.Embedding(0, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(len(KnowledgeRelation.RELATION_TYPES), embedding_dim)
        
        self.type_to_idx = {t: i for i, t in enumerate(KnowledgeRelation.RELATION_TYPES)}
        self.idx_to_type = {i: t for i, t in enumerate(KnowledgeRelation.RELATION_TYPES)}
        
        self.next_entity_id = 0
        self.next_relation_id = 0
    
    def add_entity(
        self,
        entity_type: str,
        name: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[torch.Tensor] = None,
    ) -> str:
        """添加实体."""
        entity_id = f"entity_{self.next_entity_id}"
        self.next_entity_id += 1
        
        if embedding is None:
            embedding = torch.randn(self.embedding_dim)
        
        entity = KnowledgeEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            description=description,
            properties=properties,
            embedding=embedding,
        )
        
        self.entities[entity_id] = entity
        self.entity_types[entity_type].add(entity_id)
        self.entity_name_index[name.lower()] = entity_id
        
        return entity_id
    
    def get_entity(self, entity_id: str) -> Optional[KnowledgeEntity]:
        """获取实体."""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[KnowledgeEntity]:
        """通过名称获取实体."""
        entity_id = self.entity_name_index.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None
    
    def delete_entity(self, entity_id: str) -> bool:
        """删除实体及相关关系."""
        if entity_id not in self.entities:
            return False
        
        # 删除相关关系
        for relation_id in list(self.source_relations.get(entity_id, [])):
            self.delete_relation(relation_id)
        
        for relation_id in list(self.target_relations.get(entity_id, [])):
            self.delete_relation(relation_id)
        
        # 删除实体
        entity = self.entities[entity_id]
        del self.entities[entity_id]
        self.entity_types[entity.entity_type].discard(entity_id)
        self.entity_name_index.pop(entity.name.lower(), None)
        
        return True
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """添加关系."""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        if relation_type not in KnowledgeRelation.RELATION_TYPES:
            return None
        
        relation_id = f"relation_{self.next_relation_id}"
        self.next_relation_id += 1
        
        relation = KnowledgeRelation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            description=description,
            properties=properties,
        )
        
        self.relations[relation_id] = relation
        self.source_relations[source_id].append(relation_id)
        self.target_relations[target_id].append(relation_id)
        self.type_relations[relation_type].append(relation_id)
        
        return relation_id
    
    def get_relation(self, relation_id: str) -> Optional[KnowledgeRelation]:
        """获取关系."""
        return self.relations.get(relation_id)
    
    def delete_relation(self, relation_id: str) -> bool:
        """删除关系."""
        if relation_id not in self.relations:
            return False
        
        relation = self.relations[relation_id]
        
        self.source_relations[relation.source_id].remove(relation_id)
        self.target_relations[relation.target_id].remove(relation_id)
        self.type_relations[relation.relation_type].remove(relation_id)
        
        del self.relations[relation_id]
        
        return True
    
    def get_neighbors(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """获取实体的邻居节点."""
        neighbors = []
        
        # 出边
        for relation_id in self.source_relations.get(entity_id, []):
            relation = self.relations[relation_id]
            neighbors.append((relation.target_id, relation.relation_type, relation_id))
        
        # 入边
        for relation_id in self.target_relations.get(entity_id, []):
            relation = self.relations[relation_id]
            neighbors.append((relation.source_id, relation.relation_type, relation_id))
        
        return neighbors
    
    def get_entities_by_type(self, entity_type: str) -> List[KnowledgeEntity]:
        """获取指定类型的实体."""
        entity_ids = self.entity_types.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def query(
        self,
        source_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> List[Tuple[str, str, str]]:
        """查询图谱."""
        results = []
        
        for relation_id, relation in self.relations.items():
            match = True
            
            if source_type is not None:
                source_entity = self.entities.get(relation.source_id)
                if source_entity is None or source_entity.entity_type != source_type:
                    match = False
            
            if relation_type is not None and relation.relation_type != relation_type:
                match = False
            
            if target_type is not None:
                target_entity = self.entities.get(relation.target_id)
                if target_entity is None or target_entity.entity_type != target_type:
                    match = False
            
            if match:
                results.append((relation.source_id, relation_id, relation.target_id))
        
        return results
    
    def multi_hop_query(
        self,
        start_entity_id: str,
        max_hops: int = 3,
        relation_types: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, str, str]]]:
        """多跳查询."""
        paths = []
        visited = set()
        
        def dfs(current_id: str, path: List[Tuple[str, str, str]], hops: int):
            if hops >= max_hops:
                return
            
            visited.add(current_id)
            
            for relation_id in self.source_relations.get(current_id, []):
                relation = self.relations[relation_id]
                
                if relation_types and relation.relation_type not in relation_types:
                    continue
                
                next_id = relation.target_id
                new_path = path + [(current_id, relation_id, next_id)]
                paths.append(new_path)
                
                if next_id not in visited:
                    dfs(next_id, new_path, hops + 1)
            
            visited.remove(current_id)
        
        dfs(start_entity_id, [], 0)
        
        return paths
    
    def infer_relations(
        self,
        entity_id: str,
        target_entity_id: str,
    ) -> List[Dict[str, Any]]:
        """推断两个实体之间的关系."""
        results = []
        
        # 直接关系
        for relation_id in self.source_relations.get(entity_id, []):
            relation = self.relations[relation_id]
            if relation.target_id == target_entity_id:
                results.append({
                    "type": "direct",
                    "relation": relation.to_dict(),
                    "confidence": 1.0,
                })
        
        # 间接关系（通过中间节点）
        neighbors = self.get_neighbors(entity_id)
        for neighbor_id, _, _ in neighbors:
            for relation_id in self.source_relations.get(neighbor_id, []):
                relation = self.relations[relation_id]
                if relation.target_id == target_entity_id:
                    results.append({
                        "type": "indirect",
                        "intermediate": neighbor_id,
                        "relation": relation.to_dict(),
                        "confidence": 0.7,
                    })
        
        return results
    
    def compute_similarity(
        self,
        entity_id1: str,
        entity_id2: str,
        method: str = "cosine",
    ) -> float:
        """计算两个实体的相似度."""
        entity1 = self.entities.get(entity_id1)
        entity2 = self.entities.get(entity_id2)
        
        if entity1 is None or entity2 is None:
            return 0.0
        
        if entity1.embedding is None or entity2.embedding is None:
            return 0.0
        
        if method == "cosine":
            return torch.nn.functional.cosine_similarity(
                entity1.embedding, entity2.embedding, dim=0
            ).item()
        elif method == "euclidean":
            return 1.0 / (1.0 + torch.norm(entity1.embedding - entity2.embedding).item())
        else:
            return 0.0
    
    def find_similar_entities(
        self,
        entity_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """查找相似实体."""
        entity = self.entities.get(entity_id)
        if entity is None or entity.embedding is None:
            return []
        
        similarities = []
        for eid, other in self.entities.items():
            if eid == entity_id or other.embedding is None:
                continue
            
            sim = self.compute_similarity(entity_id, eid)
            similarities.append((eid, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def update_entity_embedding(
        self,
        entity_id: str,
        embedding: torch.Tensor,
    ) -> bool:
        """更新实体嵌入."""
        if entity_id not in self.entities:
            return False
        
        self.entities[entity_id].embedding = embedding
        self.entities[entity_id].updated_at = time.time()
        
        return True
    
    def get_graph_representation(self) -> Dict[str, Any]:
        """获取图的完整表示."""
        return {
            "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
            "relations": {rid: relation.to_dict() for rid, relation in self.relations.items()},
            "entity_types": {etype: list(eids) for etype, eids in self.entity_types.items()},
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
        }
    
    def save(self, filepath: str) -> bool:
        """保存知识图谱到文件."""
        try:
            data = self.get_graph_representation()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def load(self, filepath: str) -> bool:
        """从文件加载知识图谱."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 清空现有数据
            self.entities = {}
            self.entity_types = defaultdict(set)
            self.relations = {}
            self.source_relations = defaultdict(list)
            self.target_relations = defaultdict(list)
            self.type_relations = defaultdict(list)
            self.entity_name_index = {}
            
            # 加载实体
            for eid, entity_data in data.get("entities", {}).items():
                entity = KnowledgeEntity.from_dict(entity_data)
                self.entities[eid] = entity
                self.entity_types[entity.entity_type].add(eid)
                self.entity_name_index[entity.name.lower()] = eid
                
                # 更新ID计数器
                if eid.startswith("entity_"):
                    num = int(eid.split("_")[1])
                    self.next_entity_id = max(self.next_entity_id, num + 1)
            
            # 加载关系
            for rid, relation_data in data.get("relations", {}).items():
                relation = KnowledgeRelation.from_dict(relation_data)
                self.relations[rid] = relation
                self.source_relations[relation.source_id].append(rid)
                self.target_relations[relation.target_id].append(rid)
                self.type_relations[relation.relation_type].append(rid)
                
                if rid.startswith("relation_"):
                    num = int(rid.split("_")[1])
                    self.next_relation_id = max(self.next_relation_id, num + 1)
            
            return True
        except Exception:
            return False


class KnowledgeGraphEmbedding:
    """知识图谱嵌入 - 学习实体和关系的向量表示."""
    
    def __init__(self, graph: KnowledgeGraph, embedding_dim: int = 128):
        self.graph = graph
        self.embedding_dim = embedding_dim
        
        # 实体嵌入矩阵
        self.entity_embeddings = torch.nn.Parameter(
            torch.randn(len(graph.entities), embedding_dim)
        )
        
        # 关系嵌入矩阵
        self.relation_embeddings = torch.nn.Parameter(
            torch.randn(len(graph.type_to_idx), embedding_dim)
        )
        
        # 投影矩阵
        self.projection = torch.nn.Linear(embedding_dim, embedding_dim)
        
        # 损失函数
        self.loss_fn = torch.nn.MSELoss()
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def parameters(self):
        """获取所有可训练参数."""
        return list(self.entity_embeddings) + list(self.relation_embeddings) + list(self.projection.parameters())
    
    def get_entity_embedding(self, entity_id: str) -> torch.Tensor:
        """获取实体嵌入."""
        entity_ids = list(self.graph.entities.keys())
        idx = entity_ids.index(entity_id)
        return self.entity_embeddings[idx]
    
    def get_relation_embedding(self, relation_type: str) -> torch.Tensor:
        """获取关系嵌入."""
        idx = self.graph.type_to_idx.get(relation_type, 0)
        return self.relation_embeddings[idx]
    
    def score_triple(self, head_id: str, relation_type: str, tail_id: str) -> torch.Tensor:
        """计算三元组得分."""
        h = self.get_entity_embedding(head_id)
        r = self.get_relation_embedding(relation_type)
        t = self.get_entity_embedding(tail_id)
        
        # TransE 风格评分
        score = torch.norm(h + r - t)
        
        return score
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        negative_samples: int = 5,
    ) -> Dict[str, Any]:
        """训练知识图谱嵌入."""
        losses = []
        
        entity_ids = list(self.graph.entities.keys())
        relation_types = KnowledgeRelation.RELATION_TYPES
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 构建正样本
            positive_triples = []
            for rid, relation in self.graph.relations.items():
                positive_triples.append((relation.source_id, relation.relation_type, relation.target_id))
            
            # 打乱
            np.random.shuffle(positive_triples)
            
            for i in range(0, len(positive_triples), batch_size):
                batch = positive_triples[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                total_loss = 0.0
                
                for head_id, relation_type, tail_id in batch:
                    # 正样本损失
                    pos_score = self.score_triple(head_id, relation_type, tail_id)
                    total_loss += pos_score
                    
                    # 负样本损失
                    for _ in range(negative_samples):
                        # 随机替换尾节点
                        neg_tail_id = np.random.choice(entity_ids)
                        while neg_tail_id == tail_id:
                            neg_tail_id = np.random.choice(entity_ids)
                        
                        neg_score = self.score_triple(head_id, relation_type, neg_tail_id)
                        margin = 1.0
                        total_loss += max(0, margin - neg_score)
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 更新图中实体的嵌入
        entity_ids = list(self.graph.entities.keys())
        for i, eid in enumerate(entity_ids):
            self.graph.update_entity_embedding(eid, self.entity_embeddings[i].detach())
        
        return {"losses": losses, "final_loss": losses[-1]}


class KnowledgeGraphFusion:
    """知识融合 - 从外部数据源融合知识."""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
    
    def merge_from_knowledge_base(self, knowledge_base: Any) -> Dict[str, Any]:
        """从知识库合并知识."""
        stats = {
            "entities_added": 0,
            "relations_added": 0,
            "entities_updated": 0,
            "total_entries": len(knowledge_base.entries),
        }
        
        for entry in knowledge_base.entries:
            # 创建或更新实体
            existing_entity = self.graph.get_entity_by_name(entry.get("title", ""))
            
            if existing_entity:
                # 更新现有实体
                existing_entity.description = entry.get("content", "")
                existing_entity.properties.update(entry.get("metadata", {}))
                stats["entities_updated"] += 1
                entity_id = existing_entity.entity_id
            else:
                # 创建新实体
                entity_id = self.graph.add_entity(
                    entity_type=entry.get("category", "general"),
                    name=entry.get("title", ""),
                    description=entry.get("content", ""),
                    properties=entry.get("metadata", {}),
                )
                stats["entities_added"] += 1
            
            # 添加关系
            keywords = entry.get("keywords", [])
            for keyword in keywords:
                keyword_entity = self.graph.get_entity_by_name(keyword)
                
                if keyword_entity:
                    # 添加 RELATED_TO 关系
                    self.graph.add_relation(
                        source_id=entity_id,
                        target_id=keyword_entity.entity_id,
                        relation_type="RELATED_TO",
                        weight=0.8,
                        description=f"包含关键词: {keyword}",
                    )
                    stats["relations_added"] += 1
        
        return stats
    
    def add_external_knowledge(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """添加外部知识."""
        stats = {"entities_added": 0, "relations_added": 0}
        
        # 添加实体
        for entity_data in entities:
            self.graph.add_entity(
                entity_type=entity_data.get("type", "general"),
                name=entity_data.get("name", ""),
                description=entity_data.get("description", ""),
                properties=entity_data.get("properties", {}),
            )
            stats["entities_added"] += 1
        
        # 添加关系
        for relation_data in relations:
            self.graph.add_relation(
                source_id=relation_data.get("source_id", ""),
                target_id=relation_data.get("target_id", ""),
                relation_type=relation_data.get("relation_type", "RELATED_TO"),
                weight=relation_data.get("weight", 1.0),
                description=relation_data.get("description", ""),
            )
            stats["relations_added"] += 1
        
        return stats


class KnowledgeGraphVisualizer:
    """知识图谱可视化 - 导出图结构用于可视化."""
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
    
    def export_to_dot(self, filepath: str) -> bool:
        """导出为DOT格式."""
        try:
            dot_content = "digraph KnowledgeGraph {\n"
            dot_content += "    node [shape=box, style=filled, color=\"#E8E8E8\"];\n"
            dot_content += "    edge [color=\"#A0A0A0\"];\n\n"
            
            # 添加实体节点
            for eid, entity in self.graph.entities.items():
                color = self._get_entity_color(entity.entity_type)
                dot_content += f"    {eid} [label=\"{self._escape_label(entity.name)}\", fillcolor=\"{color}\"];\n"
            
            dot_content += "\n"
            
            # 添加关系边
            for rid, relation in self.graph.relations.items():
                dot_content += f"    {relation.source_id} -> {relation.target_id} [label=\"{relation.relation_type}\"];\n"
            
            dot_content += "}\n"
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(dot_content)
            
            return True
        except Exception:
            return False
    
    def export_to_gexf(self, filepath: str) -> bool:
        """导出为GEXF格式."""
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.Element("gexf", version="1.2")
            meta = ET.SubElement(root, "meta", lastmodifieddate=time.strftime("%Y-%m-%d"))
            
            graph = ET.SubElement(root, "graph", mode="static", defaultedgetype="directed")
            
            nodes = ET.SubElement(graph, "nodes")
            for eid, entity in self.graph.entities.items():
                node = ET.SubElement(nodes, "node", id=eid, label=entity.name)
                attvalue = ET.SubElement(node, "attvalue", for_="type", value=entity.entity_type)
            
            edges = ET.SubElement(graph, "edges")
            for rid, relation in self.graph.relations.items():
                edge = ET.SubElement(edges, "edge", id=rid, source=relation.source_id, target=relation.target_id)
                attvalue = ET.SubElement(edge, "attvalue", for_="relation_type", value=relation.relation_type)
            
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding="utf-8", xml_declaration=True)
            
            return True
        except Exception:
            return False
    
    def _get_entity_color(self, entity_type: str) -> str:
        """获取实体类型对应的颜色."""
        color_map = {
            "physics": "#FF6B6B",
            "math": "#4ECDC4",
            "cs": "#45B7D1",
            "biology": "#96CEB4",
            "chemistry": "#FFEAA7",
            "ai": "#DDA0DD",
            "history": "#F39C12",
            "general": "#BDC3C7",
        }
        return color_map.get(entity_type, "#BDC3C7")
    
    def _escape_label(self, label: str) -> str:
        """转义DOT标签."""
        return label.replace('"', '\\"').replace("\n", "\\n")


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("KnowledgeGraph 测试")
    print("=" * 60)
    
    # 创建知识图谱
    kg = KnowledgeGraph(embedding_dim=128)
    
    # 添加实体
    newton_id = kg.add_entity(
        entity_type="physics",
        name="牛顿",
        description="艾萨克·牛顿，英国物理学家，万有引力定律发现者",
        properties={"birth_year": 1643, "nationality": "英国"},
    )
    
    einstein_id = kg.add_entity(
        entity_type="physics",
        name="爱因斯坦",
        description="阿尔伯特·爱因斯坦，相对论创立者",
        properties={"birth_year": 1879, "nationality": "德国"},
    )
    
    gravity_id = kg.add_entity(
        entity_type="physics",
        name="万有引力",
        description="物体之间相互吸引的力",
        properties={"formula": "F = G * m1 * m2 / r^2"},
    )
    
    relativity_id = kg.add_entity(
        entity_type="physics",
        name="相对论",
        description="爱因斯坦提出的物理学理论",
        properties={"year": 1905},
    )
    
    print(f"\n添加实体: {kg.next_entity_id} 个")
    
    # 添加关系
    kg.add_relation(newton_id, gravity_id, "DISCOVERED_BY", weight=1.0)
    kg.add_relation(einstein_id, relativity_id, "DISCOVERED_BY", weight=1.0)
    kg.add_relation(gravity_id, relativity_id, "RELATED_TO", weight=0.8)
    kg.add_relation(newton_id, einstein_id, "RELATED_TO", weight=0.6)
    
    print(f"添加关系: {kg.next_relation_id} 条")
    
    # 查询测试
    print("\n--- 查询测试 ---")
    
    # 获取邻居
    neighbors = kg.get_neighbors(newton_id)
    print(f"牛顿的邻居: {len(neighbors)} 个")
    
    # 查询关系
    query_results = kg.query(source_type="physics", relation_type="DISCOVERED_BY")
    print(f"查询结果: {len(query_results)} 条")
    
    # 多跳查询
    paths = kg.multi_hop_query(newton_id, max_hops=2)
    print(f"多跳路径: {len(paths)} 条")
    
    # 相似度计算
    sim = kg.compute_similarity(newton_id, einstein_id)
    print(f"牛顿与爱因斯坦相似度: {sim:.4f}")
    
    # 相似实体
    similar = kg.find_similar_entities(newton_id, top_k=3)
    print(f"相似实体: {len(similar)} 个")
    
    # 推理测试
    print("\n--- 推理测试 ---")
    inferred = kg.infer_relations(newton_id, relativity_id)
    print(f"推断关系: {len(inferred)} 条")
    
    # 保存和加载
    print("\n--- 保存/加载测试 ---")
    save_path = "./test_knowledge_graph.json"
    kg.save(save_path)
    print(f"图谱已保存到: {save_path}")
    
    new_kg = KnowledgeGraph()
    new_kg.load(save_path)
    print(f"加载后实体数: {len(new_kg.entities)}")
    print(f"加载后关系数: {len(new_kg.relations)}")
    
    # 嵌入训练测试
    print("\n--- 嵌入训练测试 ---")
    embedding_model = KnowledgeGraphEmbedding(kg, embedding_dim=128)
    train_result = embedding_model.train(epochs=20, batch_size=4)
    print(f"训练完成，最终损失: {train_result['final_loss']:.4f}")
    
    # 融合测试
    print("\n--- 融合测试 ---")
    fusion = KnowledgeGraphFusion(kg)
    
    external_entities = [
        {"type": "physics", "name": "量子力学", "description": "描述微观粒子行为的物理学理论"},
        {"type": "physics", "name": "薛定谔", "description": "奥地利物理学家，量子力学奠基人"},
    ]
    
    external_relations = [
        {"source_id": "entity_4", "target_id": "entity_5", "relation_type": "RELATED_TO", "weight": 0.9},
    ]
    
    fusion_stats = fusion.add_external_knowledge(external_entities, external_relations)
    print(f"融合统计: {fusion_stats}")
    
    # 可视化导出
    print("\n--- 可视化导出测试 ---")
    visualizer = KnowledgeGraphVisualizer(kg)
    visualizer.export_to_dot("./test_graph.dot")
    visualizer.export_to_gexf("./test_graph.gexf")
    print("DOT和GEXF文件已导出")
    
    # 清理测试文件
    import os
    for f in [save_path, "./test_graph.dot", "./test_graph.gexf"]:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n" + "=" * 60)
    print("知识图谱测试完成!")
    print("=" * 60)
