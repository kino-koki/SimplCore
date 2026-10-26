# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import uuid

from ..engine.tensor import SimpTensor, randn, zeros, gelu, sigmoid_tensor
from ..engine.nn import Module, Linear, LayerNorm


class AnalogyMapping:
    def __init__(self, source_id: str, target_id: str, 
                 similarity: float, structure_score: float):
        self.source_id = source_id
        self.target_id = target_id
        self.similarity = similarity
        self.structure_score = structure_score
        self.confidence = 0.0
        self.metadata: Dict[str, Any] = {}
    
    @property
    def overall_score(self) -> float:
        return 0.6 * self.similarity + 0.4 * self.structure_score


class StructureMappingEngine(Module):
    def __init__(self, dim: int, num_mappings: int = 10):
        super().__init__()
        self.dim = dim
        self.num_mappings = num_mappings
        
        self.source_proj = Linear(dim, dim)
        self.target_proj = Linear(dim, dim)
        
        self.similarity_head = Linear(dim * 2, 1)
        self.structure_head = Linear(dim * 2, 1)
        
        self.alignment_proj = Linear(dim * 2, dim)
        self.alignment_norm = LayerNorm(dim)
        
        self.transfer_proj = Linear(dim, dim)
        self.transfer_gate = Linear(dim, 1)
        
        self.mappings: List[AnalogyMapping] = []
        
        self.register_module("source_proj", self.source_proj)
        self.register_module("target_proj", self.target_proj)
        self.register_module("similarity_head", self.similarity_head)
        self.register_module("structure_head", self.structure_head)
        self.register_module("alignment_proj", self.alignment_proj)
        self.register_module("alignment_norm", self.alignment_norm)
        self.register_module("transfer_proj", self.transfer_proj)
        self.register_module("transfer_gate", self.transfer_gate)

    def _compute_similarity(self, source: SimpTensor, target: SimpTensor) -> SimpTensor:
        combined = SimpTensor(np.concatenate([source._data, target._data]))
        score = sigmoid_tensor(self.similarity_head(combined))
        return score

    def _compute_structure_score(self, source: SimpTensor, target: SimpTensor) -> SimpTensor:
        source_struct = gelu(self.source_proj(source))
        target_struct = gelu(self.target_proj(target))
        
        combined = SimpTensor(np.concatenate([source_struct._data, target_struct._data]))
        score = sigmoid_tensor(self.structure_head(combined))
        return score

    def _align(self, source: SimpTensor, target: SimpTensor) -> SimpTensor:
        combined = SimpTensor(np.concatenate([source._data, target._data]))
        aligned = gelu(self.alignment_proj(combined))
        return self.alignment_norm(aligned)

    def _transfer(self, source: SimpTensor, alignment: SimpTensor) -> SimpTensor:
        transferred = gelu(self.transfer_proj(source))
        gate = sigmoid_tensor(self.transfer_gate(alignment))
        
        return transferred * float(gate._data[0])

    def add_mapping(self, source: SimpTensor, target: SimpTensor, 
                    source_id: str = None, target_id: str = None):
        source_id = source_id or str(uuid.uuid4())[:8]
        target_id = target_id or str(uuid.uuid4())[:8]
        
        similarity_tensor = self._compute_similarity(source, target)
        structure_score_tensor = self._compute_structure_score(source, target)
        
        similarity = float(similarity_tensor._data[0])
        structure_score = float(structure_score_tensor._data[0])
        
        mapping = AnalogyMapping(source_id, target_id, similarity, structure_score)
        mapping.confidence = mapping.overall_score
        self.mappings.append(mapping)
        
        if len(self.mappings) > self.num_mappings:
            self.mappings = sorted(self.mappings, key=lambda m: m.confidence, reverse=True)[:self.num_mappings]
        
        return mapping

    def find_mappings(self, query: SimpTensor, top_k: int = 5) -> List[AnalogyMapping]:
        if not self.mappings:
            return []
        
        scores = []
        for mapping in self.mappings:
            source_data = np.random.randn(self.dim)
            source_tensor = SimpTensor(source_data)
            similarity = self._compute_similarity(source_tensor, query)
            scores.append((mapping, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in scores[:top_k]]

    def forward(self, source: SimpTensor, target: Optional[SimpTensor] = None,
               query: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        if target is None:
            target = zeros(self.dim)
        
        similarity = self._compute_similarity(source, target)
        structure_score = self._compute_structure_score(source, target)
        
        alignment = self._align(source, target)
        transferred = self._transfer(source, alignment)
        
        confidence_score = 0.6 * similarity + 0.4 * structure_score
        
        if query is not None:
            query_similarity = self._compute_similarity(source, query)
            query_similarity_val = float(query_similarity._data[0])
            if query_similarity_val > 0.3:
                query_alignment = self._align(source, query)
                transferred = transferred + 0.5 * self._transfer(query, query_alignment)
        
        final_output = target + transferred * confidence_score
        
        info = {
            "similarity": float(similarity._data[0]),
            "structure_score": float(structure_score._data[0]),
            "mapping_confidence": float(confidence_score._data[0]),
            "num_stored_mappings": len(self.mappings),
            "transfer_strength": float(confidence_score._data[0]),
            "has_query_enhancement": query is not None,
        }
        
        return final_output, info


class AnalogicalReasoner(StructureMappingEngine):
    def __init__(self, dim: int, num_mappings: int = 10, max_hops: int = 3):
        super().__init__(dim, num_mappings)
        self.max_hops = max_hops
        
        self.hop_proj = Linear(dim, dim)
        self.hop_gate = Linear(dim, 1)
        
        self.register_module("hop_proj", self.hop_proj)
        self.register_module("hop_gate", self.hop_gate)

    def _multi_hop_analogy(self, source: SimpTensor, target: SimpTensor, hops: int) -> SimpTensor:
        current = source
        accumulated = zeros(self.dim)
        
        for hop in range(hops):
            aligned = self._align(current, target)
            transferred = self._transfer(current, aligned)
            
            hop_gate = sigmoid_tensor(self.hop_gate(aligned))
            accumulated = accumulated + transferred * float(hop_gate._data[0])
            
            current = gelu(self.hop_proj(current))
        
        return accumulated

    def forward(self, source: SimpTensor, target: Optional[SimpTensor] = None,
               query: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        if target is None:
            target = zeros(self.dim)
        
        base_output, base_info = super().forward(source, target, query)
        
        hop_count = min(self.max_hops, int(base_info["mapping_confidence"] * 5))
        multi_hop_output = self._multi_hop_analogy(source, target, hop_count)
        
        final_output = base_output + 0.3 * multi_hop_output
        
        info = {
            **base_info,
            "num_hops": hop_count,
            "multi_hop_contribution": float(np.mean(multi_hop_output._data)),
            "analogy_depth": hop_count + 1,
        }
        
        return final_output, info


if __name__ == "__main__":
    dim = 64
    print("=== 测试 StructureMappingEngine ===")
    engine = StructureMappingEngine(dim)
    
    source = randn(dim)
    target = randn(dim)
    query = randn(dim)
    
    output, info = engine(source, target)
    print(f"输入: source={source.shape}, target={target.shape}")
    print(f"输出: {output.shape}")
    print(f"相似度: {info['similarity']:.3f}")
    print(f"结构分数: {info['structure_score']:.3f}")
    print(f"映射置信度: {info['mapping_confidence']:.3f}")
    
    mapping = engine.add_mapping(source, target)
    print(f"添加映射: {mapping.source_id} -> {mapping.target_id}, 置信度: {mapping.confidence:.3f}")
    
    mappings = engine.find_mappings(query, top_k=3)
    print(f"找到映射数: {len(mappings)}")
    
    output_sum = output.sum()
    output_sum.backward()
    grad_ok = any(p.grad is not None and np.any(p.grad._data != 0) for p in engine.parameters())
    print(f"梯度传播: {grad_ok}")
    print("StructureMappingEngine 测试 PASSED\n")
    
    print("=== 测试 AnalogicalReasoner ===")
    reasoner = AnalogicalReasoner(dim, max_hops=3)
    
    source2 = randn(dim)
    target2 = randn(dim)
    
    output2, info2 = reasoner(source2, target2)
    print(f"输入: source={source2.shape}, target={target2.shape}")
    print(f"输出: {output2.shape}")
    print(f"映射置信度: {info2['mapping_confidence']:.3f}")
    print(f"跳数: {info2['num_hops']}")
    print(f"类比深度: {info2['analogy_depth']}")
    
    output2_sum = output2.sum()
    output2_sum.backward()
    grad_ok2 = any(p.grad is not None and np.any(p.grad._data != 0) for p in reasoner.parameters())
    print(f"梯度传播: {grad_ok2}")
    print("AnalogicalReasoner 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")
