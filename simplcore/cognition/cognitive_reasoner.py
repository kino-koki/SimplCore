# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Cognitive Reasoner (认知推理器) - 整合所有认知模块
实现多模态认知推理，包括思维树、反思、联想、情感决策和知识自迭代
"""

import numpy as np
from typing imimport os
port Optional, Tuple, List, Dict, Any, Callable
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, ones, softmax, sigmoid_tensor, gelu, tanh, abs as tensor_abs
from ..engine.nn import Module, Linear, LayerNorm

from .tree_of_thought import TreeOfThought
from .reflexion import ReflexionModule
from .associative_reasoner import AssociativeReasoner
from .emotional_engine import EmotionalDecisionEngine
from .knowledge_iteration import KnowledgeSelfIteration
from .nonlinear_thinking import NonLinearThinkingChain, ParallelThinkingChain
from .metacognition import MetaCognition, StrategyOptimizer
from .analogical_reasoner import AnalogicalReasoner
from .causal_reasoner import CausalReasoner, CounterfactualReasoner


class ReasoningMode(Enum):
    """推理模式"""
    FAST = "fast"
    DEEP = "deep"
    BALANCED = "balanced"
    CREATIVE = "creative"
    NONLINEAR = "nonlinear"
    META = "meta"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


class CognitivePhase(Enum):
    """认知阶段"""
    INPUT = "input"
    KNOWLEDGE = "knowledge"
    ASSOCIATION = "association"
    REASONING = "reasoning"
    REFLEXION = "reflexion"
    DECISION = "decision"
    OUTPUT = "output"


class CognitiveReasoner(Module):
    """认知推理器 - 整合所有认知模块

    核心架构：
    1. 输入编码：将输入转换为认知特征
    2. 知识增强：使用知识库增强输入特征
    3. 联想推理：基于相似性进行知识关联
    4. 思维树推理：深度推理路径探索
    5. 反思修正：降低幻觉，提高可靠性
    6. 情感决策：基于情感状态优化决策
    7. 输出生成：综合所有模块结果
    8. 知识自迭代：从高置信度输出中学习新知识

    Args:
        dim: 特征维度
        reasoning_mode: 推理模式
        enable_tree_of_thought: 是否启用思维树
        enable_reflexion: 是否启用反思模块
        enable_association: 是否启用联想推理
        enable_emotion: 是否启用情感决策
        enable_knowledge: 是否启用知识自迭代
    """

    def __init__(
        self,
        dim: int,
        reasoning_mode: ReasoningMode = ReasoningMode.BALANCED,
        enable_tree_of_thought: bool = True,
        enable_reflexion: bool = True,
        enable_association: bool = True,
        enable_emotion: bool = True,
        enable_knowledge: bool = True,
        max_reasoning_depth: int = 5,
        confidence_threshold: float = 0.5,
        knowledge_capacity: int = 10000,
    ):
        super().__init__()
        self.dim = dim
        self.reasoning_mode = reasoning_mode
        self.enable_tree_of_thought = enable_tree_of_thought
        self.enable_reflexion = enable_reflexion
        self.enable_association = enable_association
        self.enable_emotion = enable_emotion
        self.enable_knowledge = enable_knowledge
        self.max_reasoning_depth = max_reasoning_depth
        self.confidence_threshold = confidence_threshold

        self.input_proj = Linear(dim, dim)
        self.register_module("input_proj", self.input_proj)

        self.output_proj = Linear(dim * 5, dim)
        self.register_module("output_proj", self.output_proj)

        self.gate_proj = Linear(dim * 2, 5)
        self.register_module("gate_proj", self.gate_proj)

        self.input_norm = LayerNorm(dim)
        self.register_module("input_norm", self.input_norm)

        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)

        self.knowledge_gate = Linear(dim, 1)
        self.register_module("knowledge_gate", self.knowledge_gate)

        if enable_tree_of_thought:
            self.tree_of_thought = TreeOfThought(dim, max_depth=max_reasoning_depth)
            self.register_module("tree_of_thought", self.tree_of_thought)

        if enable_reflexion:
            self.reflexion = ReflexionModule(dim, confidence_threshold=confidence_threshold)
            self.register_module("reflexion", self.reflexion)

        if enable_association:
            self.associative_reasoner = AssociativeReasoner(dim, num_concepts=256)
            self.register_module("associative_reasoner", self.associative_reasoner)

        if enable_emotion:
            self.emotional_engine = EmotionalDecisionEngine(dim)
            self.register_module("emotional_engine", self.emotional_engine)

        if enable_knowledge:
            self.knowledge_iteration = KnowledgeSelfIteration(
                dim=dim,
                knowledge_capacity=knowledge_capacity,
                auto_learn_threshold=confidence_threshold,
            )
            self.register_module("knowledge_iteration", self.knowledge_iteration)

        self.nonlinear_chain = NonLinearThinkingChain(dim, num_branches=4, max_depth=6)
        self.register_module("nonlinear_chain", self.nonlinear_chain)

        self.metacognition = MetaCognition(dim)
        self.register_module("metacognition", self.metacognition)

        self.analogical_reasoner = AnalogicalReasoner(dim)
        self.register_module("analogical_reasoner", self.analogical_reasoner)

        self.causal_reasoner = CausalReasoner(dim)
        self.register_module("causal_reasoner", self.causal_reasoner)

        self._cognitive_history: List[Dict[str, Any]] = []
        self._total_inferences = 0
        self._successful_inferences = 0
        self._knowledge_learned = 0

    def _encode_input(self, x: SimpTensor) -> SimpTensor:
        """编码输入"""
        encoded = self.input_proj(x)
        encoded = gelu(encoded)
        encoded = self.input_norm(encoded)
        return encoded

    def _compute_gates(self, x: SimpTensor, context: Optional[SimpTensor] = None) -> np.ndarray:
        """计算各模块的门控权重"""
        if context is not None:
            combined = SimpTensor(np.concatenate([x._data, context._data]))
        else:
            combined = SimpTensor(np.concatenate([x._data, x._data]))

        gates = self.gate_proj(combined)
        gates = sigmoid_tensor(gates)
        return gates._data

    def _run_knowledge_enhancement(
        self, x: SimpTensor, context: Optional[SimpTensor], mode: str = "enhance"
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """运行知识增强"""
        if not self.enable_knowledge:
            return x, {"num_knowledge_found": 0, "knowledge_ids": [], "path_length": 0}

        return self.knowledge_iteration(x, mode=mode)

    def _run_association(self, x: SimpTensor, context: Optional[SimpTensor]) -> Tuple[SimpTensor, Dict[str, Any]]:
        """运行联想推理"""
        if not self.enable_association:
            return x, {"num_associations": 0}
        return self.associative_reasoner(x, context)

    def _run_reasoning(self, x: SimpTensor, context: Optional[SimpTensor]) -> Tuple[SimpTensor, Dict[str, Any]]:
        """运行推理 - 根据模式选择不同推理引擎"""
        if self.reasoning_mode == ReasoningMode.NONLINEAR:
            return self.nonlinear_chain(x, context)
        elif self.reasoning_mode == ReasoningMode.META:
            return self.metacognition(x, context)
        elif self.reasoning_mode == ReasoningMode.ANALOGICAL:
            source = context if context is not None else x
            return self.analogical_reasoner(source, x)
        elif self.reasoning_mode == ReasoningMode.CAUSAL:
            return self.causal_reasoner(x, context)
        elif not self.enable_tree_of_thought:
            return x, {"confidence": 1.0, "depth": 0}

        if self.reasoning_mode == ReasoningMode.FAST:
            iterations = 5
        elif self.reasoning_mode == ReasoningMode.DEEP:
            iterations = 30
        elif self.reasoning_mode == ReasoningMode.CREATIVE:
            iterations = 20
        else:
            iterations = 15

        return self.tree_of_thought(x, max_iterations=iterations)

    def _run_reflexion(self, x: SimpTensor, context: Optional[SimpTensor]) -> Tuple[SimpTensor, Dict[str, Any]]:
        """运行反思修正"""
        if not self.enable_reflexion:
            return x, {"confidence": 1.0, "is_hallucination": False}
        return self.reflexion(x, context)

    def _run_emotional_decision(self, x: SimpTensor, context: Optional[SimpTensor]) -> Tuple[SimpTensor, Dict[str, Any]]:
        """运行情感决策"""
        if not self.enable_emotion:
            return x, {"emotion_intensity": 0.0, "valence": 0.0}
        return self.emotional_engine(x, context)

    def _learn_from_inference(self, input_tensor: SimpTensor, output_tensor: SimpTensor, confidence: float) -> Optional[str]:
        """从推理结果中学习新知识"""
        if not self.enable_knowledge or confidence < self.confidence_threshold:
            return None

        return self.knowledge_iteration.learn_from_output(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            confidence=confidence,
        )

    def forward(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor] = None,
        mode: Optional[ReasoningMode] = None,
        knowledge_mode: str = "enhance",
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """
        认知推理前向传播

        Args:
            x: 输入特征 [B, S, D] 或 [D]
            context: 上下文信息
            mode: 推理模式（覆盖默认模式）
            knowledge_mode: 知识使用模式
                - "enhance": 知识增强（默认）
                - "multi_hop": 多跳推理
                - "query": 仅查询

        Returns:
            output: 推理结果特征
            info: 推理过程信息
        """
        current_mode = mode or self.reasoning_mode

        if x.ndim == 3:
            B, S, D = x.shape
            results = []
            infos = []

            for b in range(B):
                for s in range(S):
                    xt = x[b, s]
                    out, info = self._single_inference(xt, context, current_mode, knowledge_mode)
                    results.append(out)
                    infos.append(info)

            result_tensor = SimpTensor(np.stack([r._data for r in results])).reshape(B, S, D)

            avg_info = self._aggregate_infos(infos)
            return result_tensor, avg_info
        else:
            return self._single_inference(x, context, current_mode, knowledge_mode)

    def _single_inference(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor],
        mode: ReasoningMode,
        knowledge_mode: str = "enhance",
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """单样本认知推理"""
        self._total_inferences += 1

        original_input = x
        encoded = self._encode_input(x)
        gates = self._compute_gates(encoded, context)

        phase_info = {}

        knowledge_feature, knowledge_info = self._run_knowledge_enhancement(encoded, context, knowledge_mode)
        phase_info[CognitivePhase.KNOWLEDGE.value] = knowledge_info

        assoc_feature, assoc_info = self._run_association(knowledge_feature, context)
        phase_info[CognitivePhase.ASSOCIATION.value] = assoc_info

        reasoning_feature, reasoning_info = self._run_reasoning(assoc_feature, context)
        phase_info[CognitivePhase.REASONING.value] = reasoning_info

        reflexion_feature, reflexion_info = self._run_reflexion(reasoning_feature, context)
        phase_info[CognitivePhase.REFLEXION.value] = reflexion_info

        decision_feature, decision_info = self._run_emotional_decision(reflexion_feature, context)
        phase_info[CognitivePhase.DECISION.value] = decision_info

        features = [knowledge_feature, assoc_feature, reasoning_feature, reflexion_feature, decision_feature]
        gated_features = []

        for i, feat in enumerate(features):
            gate_value = float(gates[i])
            gated = SimpTensor(feat._data * gate_value)
            gated_features.append(gated)

        combined = SimpTensor(np.concatenate([f._data for f in gated_features]))
        output = self.output_proj(combined)
        output = gelu(output)
        output = self.output_norm(output)

        confidence = reflexion_info.get("confidence", 1.0)
        is_hallucination = reflexion_info.get("is_hallucination", False)

        learned_id = self._learn_from_inference(original_input, output, confidence)
        if learned_id:
            self._knowledge_learned += 1

        if confidence >= self.confidence_threshold:
            self._successful_inferences += 1

        info = {
            "confidence": confidence,
            "is_hallucination": is_hallucination,
            "reasoning_mode": mode.value,
            "knowledge_mode": knowledge_mode,
            "gates": gates.tolist(),
            "phases": phase_info,
            "success_rate": self._successful_inferences / max(self._total_inferences, 1),
            "knowledge_learned": learned_id,
            "total_knowledge_learned": self._knowledge_learned,
        }

        self._cognitive_history.append(info)

        return output, info

    def _aggregate_infos(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多个推理的信息"""
        if len(infos) == 0:
            return {}

        avg_confidence = sum(i["confidence"] for i in infos) / len(infos)
        hallucination_rate = sum(1 for i in infos if i["is_hallucination"]) / len(infos)
        avg_gates = np.mean([np.array(i["gates"]) for i in infos], axis=0)
        total_knowledge_learned = sum(1 for i in infos if i["knowledge_learned"])

        return {
            "avg_confidence": avg_confidence,
            "hallucination_rate": hallucination_rate,
            "avg_gates": avg_gates.tolist(),
            "reasoning_mode": infos[0]["reasoning_mode"],
            "knowledge_mode": infos[0]["knowledge_mode"],
            "total_inferences": self._total_inferences,
            "success_rate": self._successful_inferences / max(self._total_inferences, 1),
            "total_knowledge_learned": self._knowledge_learned,
        }

    def add_knowledge(self, content: str, embedding: SimpTensor, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """向知识库添加知识"""
        if not self.enable_knowledge:
            return None
        return self.knowledge_iteration.add_knowledge(content, embedding, metadata)

    def query_knowledge(
        self, query: SimpTensor, top_k: int = 5, threshold: float = 0.3, include_related: bool = False
    ) -> List[Tuple[Any, float]]:
        """查询知识库"""
        if not self.enable_knowledge:
            return []
        return self.knowledge_iteration.query_knowledge(query, top_k, threshold, include_related)

    def get_knowledge_status(self) -> Dict[str, Any]:
        """获取知识自迭代状态"""
        if not self.enable_knowledge:
            return {}
        return self.knowledge_iteration.get_status()

    def start_knowledge_iteration(self) -> None:
        """启动知识自迭代后台线程"""
        if self.enable_knowledge:
            self.knowledge_iteration.start()

    def stop_knowledge_iteration(self) -> None:
        """停止知识自迭代后台线程"""
        if self.enable_knowledge:
            self.knowledge_iteration.stop()

    def set_reasoning_mode(self, mode: ReasoningMode) -> None:
        """设置推理模式"""
        self.reasoning_mode = mode

    def get_cognitive_stats(self) -> Dict[str, float]:
        """获取认知统计信息"""
        if len(self._cognitive_history) == 0:
            return {}

        avg_confidence = sum(i["confidence"] for i in self._cognitive_history) / len(self._cognitive_history)
        hallucination_rate = sum(1 for i in self._cognitive_history if i["is_hallucination"]) / len(self._cognitive_history)

        stats = {
            "total_inferences": self._total_inferences,
            "successful_inferences": self._successful_inferences,
            "success_rate": self._successful_inferences / max(self._total_inferences, 1),
            "avg_confidence": avg_confidence,
            "hallucination_rate": hallucination_rate,
            "history_length": len(self._cognitive_history),
            "knowledge_learned": self._knowledge_learned,
        }

        if self.enable_knowledge:
            stats.update(self.knowledge_iteration.get_status())

        return stats

    def reset_stats(self) -> None:
        """重置统计数据"""
        self._total_inferences = 0
        self._successful_inferences = 0
        self._knowledge_learned = 0
        self._cognitive_history = []

        if self.enable_reflexion:
            self.reflexion.reset_stats()
        if self.enable_emotion:
            self.emotional_engine.reset_history()
        if self.enable_association:
            self.associative_reasoner.reset_memory()


class AdvancedCognitiveReasoner(CognitiveReasoner):
    """高级认知推理器 - 增强版整合框架

    额外功能：
    1. 动态模块路由：根据输入复杂度选择推理路径
    2. 自适应深度：根据置信度动态调整推理深度
    3. 跨模块信息共享：各模块之间共享中间结果
    4. 智能知识使用：根据推理阶段动态调整知识使用策略
    """

    def __init__(
        self,
        dim: int,
        reasoning_mode: ReasoningMode = ReasoningMode.BALANCED,
        adaptive_depth: bool = True,
    ):
        super().__init__(dim, reasoning_mode)
        self.adaptive_depth = adaptive_depth

        self.complexity_proj = Linear(dim, 1)
        self.register_module("complexity_proj", self.complexity_proj)

        self.depth_proj = Linear(dim, 3)
        self.register_module("depth_proj", self.depth_proj)

    def _compute_complexity(self, x: SimpTensor) -> float:
        """计算输入复杂度"""
        complexity = self.complexity_proj(x)
        complexity = sigmoid_tensor(complexity)
        return complexity._data[0]

    def _single_inference(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor],
        mode: ReasoningMode,
        knowledge_mode: str = "enhance",
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """增强版单样本认知推理"""
        complexity = self._compute_complexity(x)

        if self.adaptive_depth and self.enable_tree_of_thought:
            if complexity > 0.7:
                self.tree_of_thought.max_depth = min(self.max_reasoning_depth, 8)
                knowledge_mode = "multi_hop"
            elif complexity < 0.3:
                self.tree_of_thought.max_depth = max(2, self.max_reasoning_depth // 2)
                knowledge_mode = "enhance"

        result, info = super()._single_inference(x, context, mode, knowledge_mode)
        info["input_complexity"] = complexity
        info["adaptive_depth"] = self.tree_of_thought.max_depth if self.enable_tree_of_thought else 0

        return result, info


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

    from SimplCore.engine.tensor import manual_seed, randn

    manual_seed(0)
    print("=== 测试 CognitiveReasoner (含知识自迭代) ===")

    cr = CognitiveReasoner(
        dim=64,
        reasoning_mode=ReasoningMode.BALANCED,
        enable_tree_of_thought=True,
        enable_reflexion=True,
        enable_association=True,
        enable_emotion=True,
        enable_knowledge=True,
    )

    cr.start_knowledge_iteration()

    cr.add_knowledge("基础数学知识：加减乘除运算", randn(64), {"category": "math"})
    cr.add_knowledge("机器学习知识：神经网络原理", randn(64), {"category": "AI"})
    cr.add_knowledge("物理学知识：牛顿运动定律", randn(64), {"category": "physics"})

    x = randn(64, requires_grad=True)
    context = randn(64)

    print("推理模式: BALANCED + 知识增强")
    y, info = cr(x, context, knowledge_mode="enhance")
    print(f"输入: {x.shape}, 输出: {y.shape}")
    print(f"置信度: {info['confidence']:.3f}")
    print(f"知识学习: {info['knowledge_learned']}")
    print(f"知识模式: {info['knowledge_mode']}")

    print("\n推理模式: DEEP + 多跳推理")
    y_deep, info_deep = cr(x, context, mode=ReasoningMode.DEEP, knowledge_mode="multi_hop")
    print(f"输入: {x.shape}, 输出: {y_deep.shape}")
    print(f"置信度: {info_deep['confidence']:.3f}")
    print(f"多跳路径长度: {info_deep['phases']['knowledge']['path_length']}")

    loss = (y_deep ** 2).mean()
    loss.backward()
    has_grad = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in cr.parameters())
    print(f"\n梯度传播: {has_grad}")

    stats = cr.get_cognitive_stats()
    print(f"认知统计: 总推理={stats['total_inferences']}, 成功={stats['successful_inferences']}, 知识学习={stats['knowledge_learned']}")

    knowledge_status = cr.get_knowledge_status()
    print(f"知识库状态: 总知识={knowledge_status['knowledge_stats']['total']}, 自学习={knowledge_status['total_learned']}")

    cr.stop_knowledge_iteration()
    print("CognitiveReasoner 测试 PASSED\n")

    print("=== 测试 AdvancedCognitiveReasoner ===")
    acr = AdvancedCognitiveReasoner(
        dim=64,
        reasoning_mode=ReasoningMode.BALANCED,
        adaptive_depth=True,
    )

    acr.start_knowledge_iteration()

    x2 = randn(64, requires_grad=True)
    context2 = randn(64)
    y2, info2 = acr(x2, context2)
    print(f"输入: {x2.shape}, 输出: {y2.shape}")
    print(f"复杂度: {info2['input_complexity']:.3f}, 自适应深度: {info2['adaptive_depth']}")
    print(f"置信度: {info2['confidence']:.3f}, 知识模式: {info2['knowledge_mode']}")

    loss2 = (y2 ** 2).mean()
    loss2.backward()
    has_grad2 = any(p.grad is not None and np.abs(p.grad._data).sum() > 0 for p in acr.parameters())
    print(f"梯度传播: {has_grad2}")

    acr.stop_knowledge_iteration()
    print("AdvancedCognitiveReasoner 测试 PASSED\n")

    print("=== 所有测试通过 ===")

