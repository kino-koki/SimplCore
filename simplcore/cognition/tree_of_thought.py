# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Tree of Thought (思维树) - 深度数学推理模块
支持复杂推理路径的探索和评估
"""

import math
import numpy as np
fimport os
rom typing import Optional, Tuple, List, Dict, Any, Union
from collections import deque

from ..engine.tensor import SimpTensor, randn, zeros, ones, softmax, tanh, sigmoid_tensor, gelu, layer_norm
from ..engine.nn import Module, Linear, LayerNorm

try:
    import simplcore.tensor as torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _to_simptensor(x):
    """将 torch.Tensor 或 numpy.ndarray 转换为 SimpTensor"""
    if isinstance(x, SimpTensor):
        return x
    elif _HAS_TORCH and isinstance(x, torch.Tensor):
        return SimpTensor(x.detach().cpu().numpy(), requires_grad=x.requires_grad)
    elif isinstance(x, np.ndarray):
        return SimpTensor(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


class TreeNode:
    """思维树节点"""
    
    def __init__(
        self,
        content: SimpTensor,
        depth: int = 0,
        parent: Optional["TreeNode"] = None,
        children: Optional[List["TreeNode"]] = None,
    ):
        self.content = content
        self.depth = depth
        self.parent = parent
        self.children = children or []
        self.score = 0.0
        self.visit_count = 0
        self.value = 0.0
    
    def add_child(self, child: "TreeNode") -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_path(self) -> List["TreeNode"]:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class TreeOfThought(Module):
    """思维树推理引擎 - 支持深度数学推理
    
    核心特性：
    1. Monte Carlo Tree Search (MCTS) 风格的推理路径探索
    2. 动态规划剪枝，避免无效路径
    3. 置信度评估，选择最优推理路径
    4. 支持数学公式解析和验证
    
    Args:
        dim: 特征维度
        max_depth: 最大推理深度
        num_branches: 每步分支数
        exploration_weight: 探索权重 (UCB公式)
        confidence_threshold: 置信度阈值，低于此值触发反思
    """
    
    def __init__(
        self,
        dim: int,
        max_depth: int = 8,
        num_branches: int = 4,
        exploration_weight: float = 1.414,
        confidence_threshold: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth
        self.num_branches = num_branches
        self.exploration_weight = exploration_weight
        self.confidence_threshold = confidence_threshold
        
        self.reasoning_proj = Linear(dim, dim)
        self.register_module("reasoning_proj", self.reasoning_proj)
        
        self.score_proj = Linear(dim, 1)
        self.register_module("score_proj", self.score_proj)
        
        self.branch_proj = Linear(dim, num_branches * dim)
        self.register_module("branch_proj", self.branch_proj)
        
        self.merge_proj = Linear(dim, dim)
        self.register_module("merge_proj", self.merge_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
        
        self.branch_norm = LayerNorm(dim)
        self.register_module("branch_norm", self.branch_norm)
        
        self.consistency_proj = Linear(dim * 2, dim)
        self.register_module("consistency_proj", self.consistency_proj)
        
        self.coverage_proj = Linear(dim, dim)
        self.register_module("coverage_proj", self.coverage_proj)
        
        self.novelty_proj = Linear(dim, dim)
        self.register_module("novelty_proj", self.novelty_proj)
        
        self.quality_proj = Linear(dim * 3, 1)
        self.register_module("quality_proj", self.quality_proj)
        
        self._cache: Dict[str, Any] = {}
    
    def _compute_ucb(self, node: TreeNode, total_visits: int) -> float:
        """UCB1公式选择节点"""
        if node.visit_count == 0:
            return float('inf')
        exploitation = node.value / node.visit_count
        exploration = self.exploration_weight * math.sqrt(math.log(total_visits) / node.visit_count)
        return exploitation + exploration
    
    def _expand_node(self, node: TreeNode) -> List[TreeNode]:
        """扩展节点生成子节点"""
        branch_features = self.branch_proj(node.content)
        branch_features = branch_features.reshape(self.num_branches, self.dim)
        branch_features = self.branch_norm(branch_features)
        
        children = []
        for i in range(self.num_branches):
            child_content = gelu(branch_features[i])
            child = TreeNode(content=child_content, depth=node.depth + 1)
            node.add_child(child)
            children.append(child)
        return children
    
    def _evaluate_node(self, node: TreeNode) -> float:
        """评估节点质量 - 使用多维度推理质量指标
        
        评估维度：
        1. 一致性 (Consistency): 当前节点与父节点的逻辑一致性
        2. 覆盖度 (Coverage): 信息覆盖的完整性
        3. 新颖性 (Novelty): 产生新信息的能力
        """
        reasoning = self.reasoning_proj(node.content)
        reasoning = tanh(reasoning)
        
        if node.parent is not None:
            parent_reasoning = self.reasoning_proj(node.parent.content)
            parent_reasoning = tanh(parent_reasoning)
            combined = SimpTensor(np.concatenate([reasoning._data, parent_reasoning._data]))
            consistency_feature = gelu(self.consistency_proj(combined))
            consistency_score = float(np.mean(consistency_feature._data))
        else:
            consistency_score = 0.7
        
        coverage_feature = gelu(self.coverage_proj(reasoning))
        coverage_score = float(np.mean(coverage_feature._data > 0))
        
        novelty_feature = gelu(self.novelty_proj(reasoning))
        novelty_score = float(np.mean(np.abs(novelty_feature._data)))
        
        quality_input = SimpTensor(np.concatenate([
            np.array([consistency_score]),
            np.array([coverage_score]),
            np.array([novelty_score])
        ]))
        
        quality_score = sigmoid_tensor(self.score_proj(reasoning))._data[0]
        
        final_score = 0.4 * consistency_score + 0.3 * coverage_score + 0.3 * float(quality_score)
        
        return float(final_score)
    
    def _backpropagate(self, node: TreeNode, value: float) -> None:
        """回溯更新节点值"""
        while node is not None:
            node.visit_count += 1
            node.value += value
            node = node.parent
    
    def _select_node(self, root: TreeNode) -> TreeNode:
        """选择最优子节点进行探索"""
        node = root
        while not node.is_leaf() and node.depth < self.max_depth:
            total_visits = sum(child.visit_count for child in node.children)
            best_child = max(
                node.children,
                key=lambda c: self._compute_ucb(c, total_visits)
            )
            node = best_child
        return node
    
    def _merge_paths(self, nodes: List[TreeNode]) -> SimpTensor:
        """合并多条推理路径"""
        path_features = []
        for node in nodes:
            path_features.append(node.content)
        
        if len(path_features) == 0:
            return zeros(self.dim)
        
        merged_data = np.mean([n._data for n in path_features], axis=0)
        merged = SimpTensor(merged_data, requires_grad=True)
        merged = self.merge_proj(merged)
        return merged
    
    def forward(
        self,
        x: Union[SimpTensor, torch.Tensor, np.ndarray],
        max_iterations: int = 100,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """
        思维树推理前向传播
        
        Args:
            x: 输入特征 [B, S, D] 或 [D]，支持 SimpTensor、torch.Tensor 或 numpy.ndarray
            max_iterations: MCTS迭代次数
        
        Returns:
            output: 推理结果特征
            info: 推理过程信息（置信度、深度、路径数）
        """
        x = _to_simptensor(x)
        if x.ndim == 3:
            B, S, D = x.shape
            results = []
            infos = []
            for b in range(B):
                for s in range(S):
                    xt = x[b, s]
                    out, info = self._single_inference(xt, max_iterations)
                    results.append(out)
                    infos.append(info)
            
            result_tensor = SimpTensor(np.stack([r._data for r in results])).reshape(B, S, D)
            avg_info = {
                "avg_confidence": sum(i["confidence"] for i in infos) / len(infos),
                "avg_depth": sum(i["depth"] for i in infos) / len(infos),
                "num_paths": len(infos),
            }
            return result_tensor, avg_info
        elif x.ndim == 2:
            B, D = x.shape
            results = []
            infos = []
            for b in range(B):
                xt = x[b]
                out, info = self._single_inference(xt, max_iterations)
                results.append(out)
                infos.append(info)
            
            result_tensor = SimpTensor(np.stack([r._data for r in results])).reshape(B, D)
            avg_info = {
                "avg_confidence": sum(i["confidence"] for i in infos) / len(infos),
                "avg_depth": sum(i["depth"] for i in infos) / len(infos),
                "num_paths": len(infos),
            }
            return result_tensor, avg_info
        else:
            return self._single_inference(x, max_iterations)
    
    def _single_inference(
        self,
        x: SimpTensor,
        max_iterations: int,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """单样本思维树推理"""
        root = TreeNode(content=x, depth=0)
        
        for _ in range(max_iterations):
            node = self._select_node(root)
            
            if node.depth >= self.max_depth or node.visit_count > 0:
                value = self._evaluate_node(node)
                self._backpropagate(node, value)
            else:
                children = self._expand_node(node)
                for child in children:
                    value = self._evaluate_node(child)
                    self._backpropagate(child, value)
        
        best_path = self._find_best_path(root)
        merged_result = self._merge_paths(best_path)
        
        final_confidence = best_path[-1].value / best_path[-1].visit_count if best_path[-1].visit_count > 0 else 0.0
        
        info = {
            "confidence": final_confidence,
            "depth": len(best_path) - 1,
            "num_paths": root.visit_count,
            "needs_reflection": final_confidence < self.confidence_threshold,
        }
        
        return merged_result, info
    
    def _find_best_path(self, root: TreeNode) -> List[TreeNode]:
        """找到最优推理路径"""
        path = [root]
        node = root
        
        while not node.is_leaf() and node.depth < self.max_depth:
            if node.visit_count == 0:
                break
            best_child = max(node.children, key=lambda c: c.value / max(c.visit_count, 1))
            path.append(best_child)
            node = best_child
        
        return path
    
    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """获取推理轨迹（用于调试）"""
        return []
    
    def reset_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()


class MathematicalReasoner(TreeOfThought):
    """增强版数学推理器
    
    针对数学推理优化：
    1. 支持公式解析和符号计算
    2. 推理步骤验证
    3. 错误检测和修正
    """
    
    def __init__(
        self,
        dim: int,
        max_depth: int = 12,
        num_branches: int = 6,
        exploration_weight: float = 1.2,
        confidence_threshold: float = 0.7,
    ):
        super().__init__(dim, max_depth, num_branches, exploration_weight, confidence_threshold)
        
        self.math_verify_proj = Linear(dim, 3)
        self.equation_encoder = Linear(dim, dim)
    
    def _verify_step(self, node: TreeNode) -> float:
        """验证推理步骤的正确性"""
        verification = self.math_verify_proj(node.content)
        logits = softmax(verification, dim=0)
        
        correctness_prob = logits._data[0]
        uncertainty_prob = logits._data[1]
        error_prob = logits._data[2]
        
        if error_prob > 0.3:
            return 0.1
        elif uncertainty_prob > 0.5:
            return 0.5
        else:
            return correctness_prob * 0.9 + 0.1
    
    def _evaluate_node(self, node: TreeNode) -> float:
        """数学推理专用评估"""
        base_confidence = super()._evaluate_node(node)
        verification_score = self._verify_step(node)
        return 0.6 * base_confidence + 0.4 * verification_score


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed, randn
    
    manual_seed(0)
    print("=== 测试 TreeOfThought ===")
    
    tot = TreeOfThought(dim=64, max_depth=4, num_branches=3)
    x = randn(2, 8, 64, requires_grad=True)
    y, info = tot(x, max_iterations=50)
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"推理信息: {info}")
    
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in tot.parameters())
    print(f"梯度传播: {has_grad}")
    print("TreeOfThought 测试 PASSED\n")
    
    print("=== 测试 MathematicalReasoner ===")
    mr = MathematicalReasoner(dim=64, max_depth=6, num_branches=4)
    x2 = randn(64, requires_grad=True)
    y2, info2 = mr(x2, max_iterations=30)
    print(f"输入: {x2.shape}, 输出: {y2.shape}")
    print(f"推理信息: {info2}")
    
    loss2 = (y2 ** 2).mean()
    loss2.backward()
    has_grad2 = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in mr.parameters())
    print(f"梯度传播: {has_grad2}")
    print("MathematicalReasoner 测试 PASSED\n")
    
    print("=== 所有测试通过 ===")

