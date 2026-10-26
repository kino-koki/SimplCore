# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import uuid

from ..engine.tensor import SimpTensor, randn, zeros, gelu, sigmoid_tensor, tanh
from ..engine.nn import Module, Linear, LayerNorm


class CausalNode:
    def __init__(self, id: str, embedding: SimpTensor):
        self.id = id
        self.embedding = embedding
        self.parents: List[str] = []
        self.children: List[str] = []
        self.effect_strength: Dict[str, float] = {}
        self.confidence: float = 0.0


class CausalGraph:
    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, Dict[str, float]] = {}
    
    def add_node(self, node: CausalNode):
        self.nodes[node.id] = node
        self.edges[node.id] = {}
    
    def add_edge(self, cause_id: str, effect_id: str, strength: float = 0.5):
        if cause_id in self.nodes and effect_id in self.nodes:
            self.nodes[cause_id].children.append(effect_id)
            self.nodes[effect_id].parents.append(cause_id)
            self.nodes[cause_id].effect_strength[effect_id] = strength
            self.edges[cause_id][effect_id] = strength
    
    def get_path(self, start_id: str, end_id: str, max_hops: int = 5) -> List[str]:
        if start_id == end_id:
            return [start_id]
        
        visited = {start_id}
        queue = [(start_id, [start_id])]
        
        while queue:
            current, path = queue.pop(0)
            if len(path) > max_hops:
                continue
            
            for child in self.nodes[current].children:
                if child == end_id:
                    return path + [child]
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))
        
        return []


class CausalReasoner(Module):
    def __init__(self, dim: int, max_hops: int = 3):
        super().__init__()
        self.dim = dim
        self.max_hops = max_hops
        
        self.cause_proj = Linear(dim, dim)
        self.effect_proj = Linear(dim, dim)
        
        self.strength_head = Linear(dim * 2, 1)
        self.confidence_head = Linear(dim, 1)
        
        self.do_proj = Linear(dim, dim)
        self.do_gate = Linear(dim, 1)
        
        self.counterfactual_proj = Linear(dim * 3, dim)
        self.counterfactual_norm = LayerNorm(dim)
        
        self.graph = CausalGraph()
        
        self.register_module("cause_proj", self.cause_proj)
        self.register_module("effect_proj", self.effect_proj)
        self.register_module("strength_head", self.strength_head)
        self.register_module("confidence_head", self.confidence_head)
        self.register_module("do_proj", self.do_proj)
        self.register_module("do_gate", self.do_gate)
        self.register_module("counterfactual_proj", self.counterfactual_proj)
        self.register_module("counterfactual_norm", self.counterfactual_norm)

    def _compute_causal_strength(self, cause: SimpTensor, effect: SimpTensor) -> float:
        cause_feat = gelu(self.cause_proj(cause))
        effect_feat = gelu(self.effect_proj(effect))
        
        combined = SimpTensor(np.concatenate([cause_feat._data, effect_feat._data]))
        strength = sigmoid_tensor(self.strength_head(combined))
        return float(strength._data[0])

    def _assess_confidence(self, x: SimpTensor) -> float:
        confidence = sigmoid_tensor(self.confidence_head(x))
        return float(confidence._data[0])

    def add_causal_relation(self, cause: SimpTensor, effect: SimpTensor,
                           cause_id: str = None, effect_id: str = None):
        cause_id = cause_id or str(uuid.uuid4())[:8]
        effect_id = effect_id or str(uuid.uuid4())[:8]
        
        cause_node = CausalNode(cause_id, cause)
        effect_node = CausalNode(effect_id, effect)
        
        if cause_id not in self.graph.nodes:
            self.graph.add_node(cause_node)
        if effect_id not in self.graph.nodes:
            self.graph.add_node(effect_node)
        
        strength = self._compute_causal_strength(cause, effect)
        self.graph.add_edge(cause_id, effect_id, strength)
        
        return {
            "cause_id": cause_id,
            "effect_id": effect_id,
            "strength": strength,
        }

    def _do_intervention(self, x: SimpTensor, intervention: SimpTensor) -> SimpTensor:
        intervened = gelu(self.do_proj(intervention))
        gate = sigmoid_tensor(self.do_gate(x))
        
        intervened_data = intervened._data * float(gate._data[0])
        return SimpTensor(intervened_data, requires_grad=True)

    def _counterfactual_reason(self, x: SimpTensor, cause: SimpTensor, 
                               alternative: SimpTensor) -> SimpTensor:
        combined = SimpTensor(np.concatenate([x._data, cause._data, alternative._data]))
        counterfactual = gelu(self.counterfactual_proj(combined))
        return self.counterfactual_norm(counterfactual)

    def _traverse_causal_chain(self, start: SimpTensor, max_hops: int = 3) -> Tuple[SimpTensor, List[Dict]]:
        current = start
        chain_info = []
        accumulated_effect = zeros(self.dim)
        
        for hop in range(max_hops):
            effect = gelu(self.effect_proj(current))
            strength = self._compute_causal_strength(current, effect)
            
            accumulated_effect = accumulated_effect + effect * float(strength)
            chain_info.append({
                "hop": hop,
                "strength": float(strength),
                "norm": float(np.linalg.norm(effect._data)),
            })
            
            current = gelu(self.cause_proj(current))
        
        return accumulated_effect, chain_info

    def forward(self, x: SimpTensor, context: Optional[SimpTensor] = None,
               intervention: Optional[SimpTensor] = None,
               counterfactual: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        base_output = gelu(x)
        
        if intervention is not None:
            intervened = self._do_intervention(x, intervention)
            base_output = base_output + 0.5 * intervened
        
        causal_chain, chain_info = self._traverse_causal_chain(base_output, self.max_hops)
        
        if counterfactual is not None and context is not None:
            counterfactual_output = self._counterfactual_reason(base_output, context, counterfactual)
            base_output = base_output + 0.3 * counterfactual_output
        
        final_output = base_output + causal_chain
        
        confidence = self._assess_confidence(final_output)
        
        info = {
            "confidence": float(confidence),
            "num_hops": len(chain_info),
            "chain_strength": float(np.mean([c["strength"] for c in chain_info])) if chain_info else 0.0,
            "has_intervention": intervention is not None,
            "has_counterfactual": counterfactual is not None,
            "num_nodes": len(self.graph.nodes),
            "causal_chain": chain_info,
        }
        
        return final_output, info


class CounterfactualReasoner(CausalReasoner):
    def __init__(self, dim: int, max_hops: int = 3):
        super().__init__(dim, max_hops)
        
        self.alternative_proj = Linear(dim, dim)
        self.contrast_proj = Linear(dim * 2, dim)
        
        self.register_module("alternative_proj", self.alternative_proj)
        self.register_module("contrast_proj", self.contrast_proj)

    def forward(self, x: SimpTensor, context: Optional[SimpTensor] = None,
               intervention: Optional[SimpTensor] = None,
               counterfactual: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        base_output, base_info = super().forward(x, context, intervention, counterfactual)
        
        if counterfactual is not None:
            alternative = gelu(self.alternative_proj(counterfactual))
            
            contrast = SimpTensor(np.concatenate([base_output._data, alternative._data]))
            contrast_output = gelu(self.contrast_proj(contrast))
            
            final_output = base_output + 0.5 * contrast_output
            
            diff = np.mean(np.abs(base_output._data - alternative._data))
            base_info["counterfactual_diff"] = float(diff)
            base_info["contrast_strength"] = float(np.mean(contrast_output._data))
        else:
            final_output = base_output
        
        return final_output, base_info


if __name__ == "__main__":
    dim = 64
    print("=== 测试 CausalReasoner ===")
    reasoner = CausalReasoner(dim, max_hops=3)
    
    x = randn(dim)
    cause = randn(dim)
    effect = randn(dim)
    
    relation = reasoner.add_causal_relation(cause, effect)
    print(f"添加因果关系: {relation['cause_id']} -> {relation['effect_id']}, 强度: {relation['strength']:.3f}")
    
    output, info = reasoner(x)
    print(f"输入: {x.shape}, 输出: {output.shape}")
    print(f"置信度: {info['confidence']:.3f}")
    print(f"跳数: {info['num_hops']}")
    print(f"链条强度: {info['chain_strength']:.3f}")
    print(f"节点数: {info['num_nodes']}")
    
    intervention = randn(dim)
    output2, info2 = reasoner(x, intervention=intervention)
    print(f"干预推理 - 置信度: {info2['confidence']:.3f}, 有干预: {info2['has_intervention']}")
    
    output_sum = output.sum()
    output_sum.backward()
    grad_ok = any(p.grad is not None and np.any(p.grad._data != 0) for p in reasoner.parameters())
    print(f"梯度传播: {grad_ok}")
    print("CausalReasoner 测试 PASSED\n")
    
    print("=== 测试 CounterfactualReasoner ===")
    counter_reasoner = CounterfactualReasoner(dim)
    
    x2 = randn(dim)
    context2 = randn(dim)
    counterfactual2 = randn(dim)
    
    output3, info3 = counter_reasoner(x2, context=context2, counterfactual=counterfactual2)
    print(f"输入: {x2.shape}, 输出: {output3.shape}")
    print(f"置信度: {info3['confidence']:.3f}")
    print(f"有反事实: {info3['has_counterfactual']}")
    print(f"反事实差异: {info3.get('counterfactual_diff', 'N/A')}")
    
    output3_sum = output3.sum()
    output3_sum.backward()
    grad_ok3 = any(p.grad is not None and np.any(p.grad._data != 0) for p in counter_reasoner.parameters())
    print(f"梯度传播: {grad_ok3}")
    print("CounterfactualReasoner 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")
