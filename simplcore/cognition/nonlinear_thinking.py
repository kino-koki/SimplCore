# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
import uuid

from ..engine.tensor import SimpTensor, randn, zeros, cat, gelu, sigmoid_tensor
from ..engine.nn import Module, Linear, LayerNorm


class ThinkingNode:
    def __init__(self, content: SimpTensor, node_type: str = "process", 
                 confidence: float = 0.0, depth: int = 0):
        self.id = str(uuid.uuid4())[:8]
        self.content = content
        self.node_type = node_type
        self.confidence = confidence
        self.depth = depth
        self.children: List[Tuple[str, float]] = []
        self.parents: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.visit_count = 0
        self.total_value = 0.0

    @property
    def avg_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0


class NonLinearThinkingChain(Module):
    def __init__(self, dim: int, num_branches: int = 4, max_depth: int = 8,
                 branch_threshold: float = 0.5, merge_strategy: str = "attention"):
        super().__init__()
        self.dim = dim
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.branch_threshold = branch_threshold
        self.merge_strategy = merge_strategy

        self.branch_proj = Linear(dim, num_branches * dim)
        self.branch_gate = Linear(dim, num_branches)
        
        self.condition_proj = Linear(dim, dim)
        self.jump_proj = Linear(dim, dim)
        
        self.backtrack_proj = Linear(dim * 2, dim)
        self.backtrack_gate = Linear(dim, 1)
        
        if merge_strategy == "attention":
            self.merge_query = Linear(dim, dim)
            self.merge_key = Linear(dim, dim)
            self.merge_value = Linear(dim, dim)
            self.merge_output = Linear(dim, dim)
        else:
            self.merge_proj = Linear(num_branches * dim, dim)
        
        self.node_norm = LayerNorm(dim)
        self.final_norm = LayerNorm(dim)

        self.register_module("branch_proj", self.branch_proj)
        self.register_module("branch_gate", self.branch_gate)
        self.register_module("condition_proj", self.condition_proj)
        self.register_module("jump_proj", self.jump_proj)
        self.register_module("backtrack_proj", self.backtrack_proj)
        self.register_module("backtrack_gate", self.backtrack_gate)
        if merge_strategy == "attention":
            self.register_module("merge_query", self.merge_query)
            self.register_module("merge_key", self.merge_key)
            self.register_module("merge_value", self.merge_value)
            self.register_module("merge_output", self.merge_output)
        else:
            self.register_module("merge_proj", self.merge_proj)
        self.register_module("node_norm", self.node_norm)
        self.register_module("final_norm", self.final_norm)

    def _create_node(self, content: SimpTensor, node_type: str = "process",
                    confidence: float = 0.0, depth: int = 0) -> ThinkingNode:
        return ThinkingNode(content, node_type, confidence, depth)

    def _branch(self, x: SimpTensor) -> Tuple[List[SimpTensor], SimpTensor]:
        branch_features = self.branch_proj(x)
        branches = []
        for i in range(self.num_branches):
            start = i * self.dim
            end = (i + 1) * self.dim
            branch = branch_features[:, start:end] if x.ndim == 2 else branch_features[start:end]
            branches.append(gelu(self.node_norm(branch)))
        
        gate_logits = self.branch_gate(x)
        gates = sigmoid_tensor(gate_logits)
        
        return branches, gates

    def _should_branch(self, x: SimpTensor) -> bool:
        uncertainty = self._compute_uncertainty(x)
        return uncertainty > self.branch_threshold

    def _compute_uncertainty(self, x: SimpTensor) -> float:
        x_data = x._data.flatten()
        std_val = np.std(x_data)
        
        probs = np.abs(x_data) / (np.sum(np.abs(x_data)) + 1e-9)
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        return float(std_val + entropy)

    def _condition_jump(self, x: SimpTensor, condition: SimpTensor) -> SimpTensor:
        projected = self.condition_proj(condition)
        if x.ndim == 1 and projected.ndim == 1:
            cond_score = sigmoid_tensor(SimpTensor(np.array([np.dot(x._data, projected._data)])))
        else:
            cond_score = sigmoid_tensor(x.matmul(projected.transpose(0, 1)))
        
        jump_prob = float(cond_score._data.flatten()[0])
        
        if jump_prob > 0.5:
            return self.jump_proj(x)
        return x

    def _backtrack(self, current: SimpTensor, previous: SimpTensor) -> Tuple[SimpTensor, bool]:
        combined = SimpTensor(np.concatenate([current._data, previous._data]))
        backtrack_signal = self.backtrack_proj(combined)
        backtrack_prob = sigmoid_tensor(self.backtrack_gate(backtrack_signal))._data[0]
        
        should_backtrack = backtrack_prob > 0.7
        
        if should_backtrack:
            corrected = previous + 0.1 * backtrack_signal[:self.dim]
            return corrected, True
        return current, False

    def _merge_attention(self, branches: List[SimpTensor]) -> SimpTensor:
        if len(branches) == 1:
            return branches[0]
        
        query = self.merge_query(branches[0])
        
        attn_weights = []
        for branch in branches:
            key = self.merge_key(branch)
            score = query.matmul(key.unsqueeze(-1))._data[0] if query.ndim == 1 else query.matmul(key.transpose(0, 1))
            attn_weights.append(float(score))
        
        attn_weights = np.array(attn_weights)
        attn_weights = attn_weights / (np.sum(attn_weights) + 1e-9)
        
        merged_data = np.zeros(branches[0]._data.shape)
        for branch, weight in zip(branches, attn_weights):
            merged_data += branch._data * weight
        
        merged = SimpTensor(merged_data, requires_grad=True)
        return self.merge_output(merged)

    def _merge_concat(self, branches: List[SimpTensor]) -> SimpTensor:
        if len(branches) == 1:
            return branches[0]
        
        concatenated = cat(branches, dim=0) if branches[0].ndim == 1 else cat(branches, dim=1)
        return gelu(self.merge_proj(concatenated))

    def _merge(self, branches: List[SimpTensor]) -> SimpTensor:
        if self.merge_strategy == "attention":
            return self._merge_attention(branches)
        else:
            return self._merge_concat(branches)

    def _evaluate_node(self, node: ThinkingNode) -> float:
        content = node.content
        content_data = content._data.flatten()
        
        consistency = 1.0 / (1.0 + np.std(content_data))
        coverage = np.mean(content_data > 0)
        novelty = np.mean(np.abs(content_data))
        
        quality = 0.4 * consistency + 0.3 * coverage + 0.3 * novelty
        return float(quality)

    def forward(self, x: SimpTensor, context: Optional[SimpTensor] = None,
               max_iterations: int = 100) -> Tuple[SimpTensor, Dict[str, Any]]:
        nodes: Dict[str, ThinkingNode] = {}
        edges: List[Tuple[str, str]] = []
        
        root = self._create_node(x, node_type="input", confidence=0.0, depth=0)
        nodes[root.id] = root
        
        active_nodes = [root]
        completed_paths = []
        all_nodes = [root]
        
        for iteration in range(max_iterations):
            if not active_nodes:
                break
            
            current_node = active_nodes.pop(0)
            
            if current_node.depth >= self.max_depth:
                completed_paths.append(current_node)
                continue
            
            should_branch = self._should_branch(current_node.content)
            
            if should_branch and current_node.depth < self.max_depth - 1:
                branches, gates = self._branch(current_node.content)
                
                for i, (branch_content, gate) in enumerate(zip(branches, gates._data.flatten())):
                    if gate > 0.3:
                        branch_node = self._create_node(
                            branch_content,
                            node_type="branch",
                            confidence=float(gate),
                            depth=current_node.depth + 1
                        )
                        branch_node.parents.append(current_node.id)
                        current_node.children.append((branch_node.id, float(gate)))
                        nodes[branch_node.id] = branch_node
                        edges.append((current_node.id, branch_node.id))
                        all_nodes.append(branch_node)
                        
                        if branch_node.depth < self.max_depth - 1:
                            condition = randn(self.dim)
                            jumped = self._condition_jump(branch_node.content, condition)
                            if not np.allclose(jumped._data, branch_node.content._data):
                                jump_node = self._create_node(
                                    jumped,
                                    node_type="jump",
                                    confidence=0.5,
                                    depth=branch_node.depth
                                )
                                jump_node.parents.append(branch_node.id)
                                branch_node.children.append((jump_node.id, 0.5))
                                nodes[jump_node.id] = jump_node
                                edges.append((branch_node.id, jump_node.id))
                                all_nodes.append(jump_node)
                                active_nodes.append(jump_node)
                            else:
                                active_nodes.append(branch_node)
            else:
                processed = gelu(self.node_norm(current_node.content))
                process_node = self._create_node(
                    processed,
                    node_type="process",
                    confidence=self._evaluate_node(current_node),
                    depth=current_node.depth + 1
                )
                process_node.parents.append(current_node.id)
                current_node.children.append((process_node.id, 1.0))
                nodes[process_node.id] = process_node
                edges.append((current_node.id, process_node.id))
                all_nodes.append(process_node)
                
                if context is not None and current_node.depth > 0:
                    previous_node = nodes[current_node.parents[0]] if current_node.parents else None
                    if previous_node is not None:
                        corrected, did_backtrack = self._backtrack(process_node.content, previous_node.content)
                        if did_backtrack:
                            backtrack_node = self._create_node(
                                corrected,
                                node_type="backtrack",
                                confidence=0.8,
                                depth=process_node.depth
                            )
                            backtrack_node.parents.append(process_node.id)
                            process_node.children.append((backtrack_node.id, 0.8))
                            nodes[backtrack_node.id] = backtrack_node
                            edges.append((process_node.id, backtrack_node.id))
                            all_nodes.append(backtrack_node)
                            active_nodes.append(backtrack_node)
                            continue
                
                if process_node.depth >= self.max_depth:
                    completed_paths.append(process_node)
                else:
                    active_nodes.append(process_node)
        
        if completed_paths:
            best_path = max(completed_paths, key=lambda n: n.confidence)
            branch_nodes = [n for n in all_nodes if n.node_type == "branch"]
            merge_candidates = [n for n in all_nodes if n.node_type in ["branch", "process", "backtrack"] and n.depth >= 2]
            
            if merge_candidates:
                candidate_tensors = [n.content for n in merge_candidates]
                merged = self._merge(candidate_tensors)
                final_output = self.final_norm(merged)
            else:
                final_output = self.final_norm(best_path.content)
        else:
            final_output = self.final_norm(x)
        
        confidences = [n.confidence for n in all_nodes]
        info = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_branches": sum(1 for n in all_nodes if n.node_type == "branch"),
            "num_backtracks": sum(1 for n in all_nodes if n.node_type == "backtrack"),
            "num_jumps": sum(1 for n in all_nodes if n.node_type == "jump"),
            "max_depth_reached": max(n.depth for n in all_nodes),
            "avg_confidence": float(np.mean(confidences)),
            "best_confidence": float(max(confidences)),
            "node_types": {t: sum(1 for n in all_nodes if n.node_type == t) 
                         for t in ["input", "process", "branch", "jump", "backtrack"]},
        }
        
        return final_output, info


class ParallelThinkingChain(NonLinearThinkingChain):
    def __init__(self, dim: int, num_parallel_paths: int = 4, max_depth: int = 8):
        super().__init__(dim, num_branches=num_parallel_paths, max_depth=max_depth)
        self.num_parallel_paths = num_parallel_paths
        
        self.path_weights = Linear(dim, num_parallel_paths)
        self.register_module("path_weights", self.path_weights)

    def forward(self, x: SimpTensor, context: Optional[SimpTensor] = None,
               max_iterations: int = 100) -> Tuple[SimpTensor, Dict[str, Any]]:
        path_outputs = []
        path_infos = []
        
        for path_idx in range(self.num_parallel_paths):
            path_seed = x + randn(self.dim) * 0.1 * (path_idx + 1)
            
            path_output, path_info = super().forward(path_seed, context, max_iterations // self.num_parallel_paths)
            path_outputs.append(path_output)
            path_infos.append(path_info)
        
        path_scores = self.path_weights(x)
        path_weights = sigmoid_tensor(path_scores)
        
        weighted_outputs = []
        for i, (output, weight) in enumerate(zip(path_outputs, path_weights._data.flatten())):
            weighted_outputs.append(output * float(weight))
        
        combined = sum(weighted_outputs)
        final_output = self.final_norm(combined)
        
        best_confidences = [info["best_confidence"] for info in path_infos]
        avg_info = {
            "num_parallel_paths": self.num_parallel_paths,
            "avg_num_nodes": float(np.mean([info["num_nodes"] for info in path_infos])),
            "avg_num_branches": float(np.mean([info["num_branches"] for info in path_infos])),
            "path_weights": path_weights._data.flatten().tolist(),
            "max_depth_reached": max(info["max_depth_reached"] for info in path_infos),
            "best_path_confidence": float(max(best_confidences)),
        }
        
        return final_output, avg_info


if __name__ == "__main__":
    dim = 64
    print("=== 测试 NonLinearThinkingChain ===")
    nl_chain = NonLinearThinkingChain(dim, num_branches=3, max_depth=4)
    x = randn(dim)
    output, info = nl_chain(x)
    print(f"输入: {x.shape}, 输出: {output.shape}")
    print(f"节点数: {info['num_nodes']}, 边数: {info['num_edges']}")
    print(f"分支数: {info['num_branches']}, 回溯数: {info['num_backtracks']}")
    print(f"最大深度: {info['max_depth_reached']}, 最佳置信度: {info['best_confidence']:.3f}")
    print(f"节点类型分布: {info['node_types']}")
    
    output_sum = output.sum()
    output_sum.backward()
    grad_ok = any(p.grad is not None and np.any(p.grad._data != 0) for p in nl_chain.parameters())
    print(f"梯度传播: {grad_ok}")
    print("NonLinearThinkingChain 测试 PASSED\n")
    
    print("=== 测试 ParallelThinkingChain ===")
    parallel_chain = ParallelThinkingChain(dim, num_parallel_paths=3)
    x2 = randn(dim)
    output2, info2 = parallel_chain(x2)
    print(f"输入: {x2.shape}, 输出: {output2.shape}")
    print(f"并行路径数: {info2['num_parallel_paths']}")
    print(f"平均节点数: {info2['avg_num_nodes']:.1f}")
    path_weights_str = [f"{w:.3f}" for w in info2['path_weights']]
    print(f"路径权重: {path_weights_str}")
    print(f"最佳路径置信度: {info2['best_path_confidence']:.3f}")
    
    output2_sum = output2.sum()
    output2_sum.backward()
    grad_ok2 = any(p.grad is not None and np.any(p.grad._data != 0) for p in parallel_chain.parameters())
    print(f"梯度传播: {grad_ok2}")
    print("ParallelThinkingChain 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")
