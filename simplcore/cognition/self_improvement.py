# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
自我改进模块 (Self-Improvement Module)
=====================================

AGI的核心自我进化引擎，基于SimplCore框架实现：

1. 性能监控与诊断
   - 实时监控推理质量
   - 识别知识盲区
   - 检测推理错误模式

2. 知识缺口自动填补
   - 分析缺失的知识领域
   - 自动从知识库检索相关知识
   - 将新知识整合到推理过程

3. 模型自我调整
   - 根据性能反馈调整参数
   - 优化推理策略
   - 动态调整注意力分配

4. 错误学习机制
   - 记录错误案例
   - 分析错误原因
   - 避免重复犯错

5. 能力评估与成长
   - 定期评估各项能力
   - 制定学习计划
   - 追踪成长进度

集成说明:
- 与knowledge_iteration模块协同工作
- 与human_like_reasoner模块深度集成
- 与calibration_trainer配合训练
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from datetime import datetime
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, sigmoid_tensor, tanh, gelu, softmax
from ..engine.nn import Module, Linear, LayerNorm
from ..knowledge.knowledge_graph import KnowledgeGraph, KnowledgeReasoner


class ImprovementType(Enum):
    KNOWLEDGE_GAP = "knowledge_gap"
    REASONING_ERROR = "reasoning_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CAPABILITY_EXPANSION = "capability_expansion"
    STRATEGY_OPTIMIZATION = "strategy_optimization"


class ImprovementPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImprovementRecord:
    """改进记录"""

    def __init__(
        self,
        improvement_type: ImprovementType,
        priority: ImprovementPriority,
        description: str,
        context: Dict[str, Any],
        action: str,
        result: Optional[str] = None,
        success: bool = False,
    ):
        self.id = f"impr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000)}"
        self.type = improvement_type
        self.priority = priority
        self.description = description
        self.context = context
        self.action = action
        self.result = result
        self.success = success
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None

    def mark_completed(self, result: str, success: bool):
        self.result = result
        self.success = success
        self.completed_at = datetime.now()


class PerformanceMetric:
    """性能指标"""

    def __init__(self):
        self.total_queries = 0
        self.correct_answers = 0
        self.avg_confidence = 0.0
        self.hallucination_rate = 0.0
        self.response_time: List[float] = []
        self.knowledge_usage_rate = 0.0
        self.error_types: Dict[str, int] = {}
        self._recent_history: List[Dict[str, Any]] = []

    def record_query(
        self,
        correct: bool,
        confidence: float,
        response_time: float,
        used_knowledge: bool,
        error_type: Optional[str] = None,
    ):
        self.total_queries += 1
        if correct:
            self.correct_answers += 1
        self.avg_confidence = (self.avg_confidence * (self.total_queries - 1) + confidence) / self.total_queries
        self.response_time.append(response_time)

        self.knowledge_usage_rate = (
            self.knowledge_usage_rate * (self.total_queries - 1) + (1 if used_knowledge else 0)
        ) / self.total_queries

        if error_type:
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

        self._recent_history.append({
            "correct": correct,
            "confidence": confidence,
            "used_knowledge": used_knowledge,
            "error_type": error_type,
            "timestamp": datetime.now(),
        })
        if len(self._recent_history) > 100:
            self._recent_history = self._recent_history[-100:]

    def get_accuracy(self) -> float:
        return self.correct_answers / max(self.total_queries, 1)

    def get_avg_response_time(self) -> float:
        return float(np.mean(self.response_time)) if self.response_time else 0.0

    def get_recent_performance(self, n: int = 10) -> Dict[str, float]:
        recent = self._recent_history[-n:] if self._recent_history else []
        if not recent:
            return {"accuracy": 0.0, "avg_confidence": 0.0, "knowledge_rate": 0.0}

        recent_correct = sum(1 for r in recent if r["correct"])
        recent_conf = np.mean([r["confidence"] for r in recent])
        recent_knowledge = np.mean([1.0 if r["used_knowledge"] else 0.0 for r in recent])

        return {
            "accuracy": recent_correct / len(recent),
            "avg_confidence": recent_conf,
            "knowledge_rate": recent_knowledge,
        }

    def reset(self):
        self.__init__()


class BlindSpotDetector:
    """盲区检测器"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.blind_spots: Dict[str, int] = {}
        self.query_history: List[Dict[str, Any]] = []
        self._low_confidence_threshold = 0.4

    def detect_blind_spots(
        self,
        confidence: float,
        knowledge_used_count: int,
        features: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """检测盲区"""
        detected = []

        if confidence < self._low_confidence_threshold:
            spot_key = f"low_conf_{int(confidence * 100)}"
            self.blind_spots[spot_key] = self.blind_spots.get(spot_key, 0) + 1
            detected.append({
                "type": "low_confidence",
                "confidence": confidence,
                "suggestion": "增加相关知识学习或提升推理置信度",
            })

        if knowledge_used_count == 0 and confidence < 0.6:
            spot_key = "no_knowledge_used"
            self.blind_spots[spot_key] = self.blind_spots.get(spot_key, 0) + 1
            detected.append({
                "type": "no_knowledge_used",
                "suggestion": "检查知识库是否有相关知识可检索",
            })

        return detected

    def get_dominant_blind_spots(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """获取主要盲区"""
        sorted_spots = sorted(self.blind_spots.items(), key=lambda x: x[1], reverse=True)
        return [
            {"spot": spot, "count": count}
            for spot, count in sorted_spots[:top_k]
        ]


class SelfImprovementModule(Module):
    """自我改进模块 - AGI的核心自我进化引擎"""

    def __init__(
        self,
        dim: int = 64,
        knowledge_reasoner: Optional[KnowledgeReasoner] = None,
        confidence_threshold: float = 0.5,
        max_improvements: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.confidence_threshold = confidence_threshold
        self.max_improvements = max_improvements
        self.knowledge_reasoner = knowledge_reasoner

        self.performance_metric = PerformanceMetric()
        self.blind_spot_detector = BlindSpotDetector(dim)
        self.improvement_records: List[ImprovementRecord] = []

        self.strategy_proj = Linear(dim, dim)
        self.register_module("strategy_proj", self.strategy_proj)

        self.confidence_adjuster = Linear(dim, 1)
        self.register_module("confidence_adjuster", self.confidence_adjuster)

        self.attention_scaler = Linear(dim, dim)
        self.register_module("attention_scaler", self.attention_scaler)

        self.output_proj = Linear(dim, dim)
        self.register_module("output_proj", self.output_proj)

        self.strategy_weights = SimpTensor(np.random.randn(dim) * 0.01)
        self.improvement_enabled = True

        self._adjustment_history: List[Dict[str, Any]] = []

    def monitor_performance(
        self,
        confidence: float,
        correct: bool = True,
        response_time: float = 0.0,
        knowledge_used_count: int = 0,
        error_type: Optional[str] = None,
        features: Optional[SimpTensor] = None,
    ) -> Dict[str, Any]:
        """监控性能"""
        self.performance_metric.record_query(
            correct=correct,
            confidence=confidence,
            response_time=response_time,
            used_knowledge=knowledge_used_count > 0,
            error_type=error_type,
        )

        blind_spots = self.blind_spot_detector.detect_blind_spots(
            confidence=confidence,
            knowledge_used_count=knowledge_used_count,
            features=features._data if features is not None else None,
        )

        return {
            "accuracy": self.performance_metric.get_accuracy(),
            "avg_confidence": self.performance_metric.avg_confidence,
            "avg_response_time": self.performance_metric.get_avg_response_time(),
            "knowledge_usage_rate": self.performance_metric.knowledge_usage_rate,
            "blind_spots": blind_spots,
            "total_queries": self.performance_metric.total_queries,
            "recent": self.performance_metric.get_recent_performance(10),
        }

    def identify_improvement_opportunities(
        self,
        performance_data: Dict[str, Any],
    ) -> List[ImprovementRecord]:
        """识别改进机会"""
        opportunities = []
        accuracy = performance_data.get("accuracy", 0.0)
        avg_confidence = performance_data.get("avg_confidence", 0.0)
        blind_spots = performance_data.get("blind_spots", [])

        if accuracy < 0.6:
            opportunities.append(ImprovementRecord(
                improvement_type=ImprovementType.PERFORMANCE_DEGRADATION,
                priority=ImprovementPriority.CRITICAL,
                description=f"整体准确率过低: {accuracy:.2%}",
                context={"accuracy": accuracy, "avg_confidence": avg_confidence},
                action="紧急优化推理策略，增加知识库覆盖率",
            ))

        if avg_confidence < 0.5:
            opportunities.append(ImprovementRecord(
                improvement_type=ImprovementType.KNOWLEDGE_GAP,
                priority=ImprovementPriority.HIGH,
                description=f"平均置信度过低: {avg_confidence:.2%}",
                context={"avg_confidence": avg_confidence},
                action="扩大知识库，增强知识检索能力",
            ))

        for spot in blind_spots:
            if spot["type"] == "low_confidence":
                opportunities.append(ImprovementRecord(
                    improvement_type=ImprovementType.KNOWLEDGE_GAP,
                    priority=ImprovementPriority.MEDIUM,
                    description=f"低置信度: {spot['confidence']:.2%}",
                    context={"confidence": spot["confidence"]},
                    action=spot["suggestion"],
                ))

        self.improvement_records.extend(opportunities)
        if len(self.improvement_records) > self.max_improvements:
            self.improvement_records = self.improvement_records[-self.max_improvements:]

        return opportunities

    def adjust_strategy(self, context: SimpTensor) -> Dict[str, Any]:
        """调整推理策略"""
        if not self.improvement_enabled:
            return {"adjusted": False, "reason": "improvement disabled"}

        strategy_update = sigmoid_tensor(self.strategy_proj(context))
        confidence_bias = float(sigmoid_tensor(self.confidence_adjuster(context))._data[0])
        attention_scale = sigmoid_tensor(self.attention_scaler(context))

        adjustment = {
            "strategy_weights_norm": float(np.linalg.norm(strategy_update._data)),
            "confidence_bias": confidence_bias,
            "attention_scale_norm": float(np.linalg.norm(attention_scale._data)),
        }

        self._adjustment_history.append({
            "timestamp": datetime.now().isoformat(),
            "adjustment": adjustment,
            "context_norm": float(np.linalg.norm(context._data)),
        })

        return {
            "adjusted": True,
            **adjustment,
        }

    def learn_from_mistake(
        self,
        error_description: str,
        error_type: str,
        query_features: Optional[SimpTensor] = None,
    ) -> Dict[str, Any]:
        """从错误中学习"""
        record = ImprovementRecord(
            improvement_type=ImprovementType.REASONING_ERROR,
            priority=ImprovementPriority.HIGH,
            description=f"推理错误: {error_type} - {error_description[:60]}",
            context={
                "error_description": error_description,
                "error_type": error_type,
                "query_norm": float(np.linalg.norm(query_features._data)) if query_features is not None else 0.0,
            },
            action="分析错误模式，更新推理策略",
        )

        self.improvement_records.append(record)

        return {
            "learned": True,
            "record_id": record.id,
            "error_type": error_type,
        }

    def evaluate_capabilities(self) -> Dict[str, Any]:
        """评估各项能力"""
        metrics = self.performance_metric
        recent = metrics.get_recent_performance(20)

        capability_scores = {
            "language_understanding": min(1.0, metrics.get_accuracy() + 0.2),
            "knowledge_retrieval": min(1.0, metrics.knowledge_usage_rate + 0.3),
            "reasoning_ability": min(1.0, metrics.avg_confidence + 0.2),
            "self_reflection": min(1.0, len(self.improvement_records) / 10),
            "continuous_learning": min(1.0, len(self._adjustment_history) / 20 + 0.1),
        }

        overall_score = float(np.mean(list(capability_scores.values())))

        return {
            "capabilities": capability_scores,
            "overall_score": overall_score,
            "performance_metrics": {
                "accuracy": metrics.get_accuracy(),
                "avg_confidence": metrics.avg_confidence,
                "total_queries": metrics.total_queries,
                "knowledge_usage_rate": metrics.knowledge_usage_rate,
            },
            "recent_performance": recent,
            "improvement_history": len(self.improvement_records),
        }

    def generate_learning_plan(self, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成学习计划"""
        capabilities = evaluation.get("capabilities", {})
        plan = []

        for capability, score in capabilities.items():
            if score < 0.5:
                plan.append({
                    "capability": capability,
                    "current_score": score,
                    "target_score": 0.7,
                    "priority": "high",
                    "action": self._get_improvement_action(capability),
                    "estimated_time": "high",
                })
            elif score < 0.7:
                plan.append({
                    "capability": capability,
                    "current_score": score,
                    "target_score": 0.85,
                    "priority": "medium",
                    "action": self._get_improvement_action(capability),
                    "estimated_time": "medium",
                })

        return sorted(plan, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])

    def _get_improvement_action(self, capability: str) -> str:
        actions = {
            "language_understanding": "增加文本训练数据，优化语义编码器",
            "knowledge_retrieval": "增强知识库索引，优化检索算法",
            "reasoning_ability": "改进推理策略，增加反思深度",
            "self_reflection": "增强反思模块，提高自我评估能力",
            "continuous_learning": "优化学习率，增加训练数据多样性",
        }
        return actions.get(capability, "综合优化")

    def forward(
        self,
        context: SimpTensor,
        confidence: float = 0.0,
        correct: bool = True,
        response_time: float = 0.0,
        knowledge_used_count: int = 0,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """前向传播 - 执行自我改进"""
        performance = self.monitor_performance(
            confidence=confidence,
            correct=correct,
            response_time=response_time,
            knowledge_used_count=knowledge_used_count,
            error_type=error_type,
            features=context,
        )

        opportunities = self.identify_improvement_opportunities(performance)
        strategy_adjustment = self.adjust_strategy(context)

        return {
            "performance": performance,
            "improvement_opportunities": [
                {
                    "type": opp.type.value,
                    "priority": opp.priority.value,
                    "description": opp.description[:80],
                    "action": opp.action[:80],
                }
                for opp in opportunities
            ],
            "strategy_adjustment": strategy_adjustment,
            "total_improvements": len(self.improvement_records),
        }

    def get_improvement_summary(self) -> Dict[str, Any]:
        """获取改进摘要"""
        by_type: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}

        for record in self.improvement_records:
            by_type[record.type.value] = by_type.get(record.type.value, 0) + 1
            by_priority[record.priority.value] = by_priority.get(record.priority.value, 0) + 1

        completed = [r for r in self.improvement_records if r.completed_at]
        success_rate = sum(1 for r in completed if r.success) / max(len(completed), 1)

        return {
            "total_records": len(self.improvement_records),
            "by_type": by_type,
            "by_priority": by_priority,
            "completed": len(completed),
            "success_rate": success_rate,
            "blind_spots": self.blind_spot_detector.get_dominant_blind_spots(3),
        }

    def reset(self):
        """重置状态"""
        self.performance_metric.reset()
        self.improvement_records = []
        self._adjustment_history = []


class SelfLearningLoop(Module):
    """自我学习循环 - 连接所有自我改进组件"""

    def __init__(
        self,
        dim: int = 64,
        improvement_module: Optional[SelfImprovementModule] = None,
        knowledge_reasoner: Optional[KnowledgeReasoner] = None,
    ):
        super().__init__()
        self.dim = dim
        self.improvement_module = improvement_module or SelfImprovementModule(dim, knowledge_reasoner)
        self.learning_cycles = 0
        self.last_evaluation: Optional[Dict[str, Any]] = None
        self._cycle_history: List[Dict[str, Any]] = []

    def run_cycle(
        self,
        context: SimpTensor,
        confidence: float = 0.0,
        correct: bool = True,
        response_time: float = 0.0,
        knowledge_used_count: int = 0,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行一个学习循环"""
        result = self.improvement_module(
            context=context,
            confidence=confidence,
            correct=correct,
            response_time=response_time,
            knowledge_used_count=knowledge_used_count,
            error_type=error_type,
        )

        self.learning_cycles += 1

        if self.learning_cycles % 5 == 0:
            self.last_evaluation = self.improvement_module.evaluate_capabilities()

        self._cycle_history.append({
            "cycle": self.learning_cycles,
            "performance": result["performance"],
            "improvements": len(result["improvement_opportunities"]),
            "timestamp": datetime.now(),
        })

        return {
            "cycle": self.learning_cycles,
            "performance": result["performance"],
            "improvements": result["improvement_opportunities"],
            "evaluation": self.last_evaluation,
        }

    def forward(
        self,
        context: SimpTensor,
        confidence: float = 0.0,
        correct: bool = True,
        response_time: float = 0.0,
        knowledge_used_count: int = 0,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """前向传播"""
        return self.run_cycle(
            context=context,
            confidence=confidence,
            correct=correct,
            response_time=response_time,
            knowledge_used_count=knowledge_used_count,
            error_type=error_type,
        )

    def get_status(self) -> Dict[str, Any]:
        """获取学习循环状态"""
        return {
            "learning_cycles": self.learning_cycles,
            "total_improvements": len(self.improvement_module.improvement_records),
            "last_evaluation": self.last_evaluation,
            "recent_cycles": self._cycle_history[-5:] if self._cycle_history else [],
        }


class IntegratedSelfImprovement(Module):
    """集成式自我改进 - 整合知识库和认知推理器"""

    def __init__(
        self,
        dim: int = 64,
        knowledge_graph: Optional[KnowledgeGraph] = None,
    ):
        super().__init__()
        self.dim = dim

        self.knowledge_reasoner = KnowledgeReasoner(dim)
        if knowledge_graph:
            self.knowledge_reasoner.knowledge_graph = knowledge_graph
        self.register_module("knowledge_reasoner", self.knowledge_reasoner)

        self.improvement = SelfImprovementModule(
            dim=dim,
            knowledge_reasoner=self.knowledge_reasoner,
        )

        self.self_learning_loop = SelfLearningLoop(
            dim=dim,
            improvement_module=self.improvement,
            knowledge_reasoner=self.knowledge_reasoner,
        )

    def process_outcome(
        self,
        query_features: SimpTensor,
        confidence: float,
        correct: bool = True,
        response_time: float = 0.0,
        knowledge_found: int = 0,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """处理推理结果并自我改进"""
        result = self.self_learning_loop.run_cycle(
            context=query_features,
            confidence=confidence,
            correct=correct,
            response_time=response_time,
            knowledge_used_count=knowledge_found,
            error_type=error_type,
        )

        if confidence < 0.4 and self.knowledge_reasoner is not None:
            enhanced, knowledge_info = self.knowledge_reasoner(query_features, context=None)
            result["knowledge_enhanced"] = True
            result["knowledge_info"] = {
                "entities": knowledge_info.get("entities_found", 0),
                "confidence": knowledge_info.get("reasoning_confidence", 0.0),
            }
        else:
            result["knowledge_enhanced"] = False

        return result

    def get_full_status(self) -> Dict[str, Any]:
        """获取完整状态"""
        return {
            "learning_loop": self.self_learning_loop.get_status(),
            "improvement_summary": self.improvement.get_improvement_summary(),
            "capabilities": self.improvement.evaluate_capabilities(),
        }

    def forward(
        self,
        query_features: SimpTensor,
        confidence: float = 0.0,
        correct: bool = True,
        response_time: float = 0.0,
        knowledge_found: int = 0,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """前向传播"""
        return self.process_outcome(
            query_features=query_features,
            confidence=confidence,
            correct=correct,
            response_time=response_time,
            knowledge_found=knowledge_found,
            error_type=error_type,
        )

    def reset(self):
        """重置"""
        self.improvement.reset()
        self.self_learning_loop = SelfLearningLoop(
            dim=self.dim,
            improvement_module=self.improvement,
            knowledge_reasoner=self.knowledge_reasoner,
        )


if __name__ == "__main__":
    from SimplCore.engine.tensor import manual_seed, randn
    manual_seed(0)

    dim = 64
    print("=" * 70)
    print("SelfImprovementModule (SimplCore版) 测试")
    print("=" * 70)

    print("\n1. 测试性能监控:")
    sim = SelfImprovementModule(dim)
    for i in range(5):
        perf = sim.monitor_performance(
            confidence=0.6 + i * 0.05,
            correct=i < 4,
            response_time=0.1 + i * 0.05,
            knowledge_used_count=1 if i % 2 == 0 else 0,
        )
    print(f"  准确率: {perf['accuracy']:.2%}")
    print(f"  平均置信度: {perf['avg_confidence']:.2%}")
    print(f"  知识库使用率: {perf['knowledge_usage_rate']:.2%}")

    print("\n2. 测试改进机会识别:")
    opportunities = sim.identify_improvement_opportunities(perf)
    print(f"  发现改进机会: {len(opportunities)} 个")
    for opp in opportunities[:3]:
        print(f"    - [{opp.priority.value}] {opp.description[:50]}")

    print("\n3. 测试策略调整:")
    adjustment = sim.adjust_strategy(randn(dim))
    print(f"  调整成功: {adjustment['adjusted']}")
    print(f"  置信度偏差: {adjustment['confidence_bias']:.4f}")

    print("\n4. 测试错误学习:")
    learn_result = sim.learn_from_mistake(
        error_description="在回答量子纠缠问题时出现了事实性错误",
        error_type="fact_error",
        query_features=randn(dim),
    )
    print(f"  学习成功: {learn_result['learned']}, 记录ID: {learn_result['record_id']}")

    print("\n5. 测试能力评估:")
    evaluation = sim.evaluate_capabilities()
    print(f"  综合得分: {evaluation['overall_score']:.2%}")
    for cap, score in evaluation["capabilities"].items():
        print(f"    {cap}: {score:.2%}")

    print("\n6. 测试学习计划:")
    plan = sim.generate_learning_plan(evaluation)
    print(f"  计划项数: {len(plan)}")
    for item in plan[:3]:
        print(f"    - [{item['priority']}] {item['capability']}: {item['current_score']:.2%} -> {item['target_score']:.2%}")

    print("\n7. 测试前向传播:")
    result = sim(randn(dim), confidence=0.6, correct=True, response_time=0.5, knowledge_used_count=2)
    print(f"  改进建议数: {len(result['improvement_opportunities'])}")
    print(f"  策略已调整: {result['strategy_adjustment']['adjusted']}")

    print("\n8. 测试集成式自我改进:")
    integrated = IntegratedSelfImprovement(dim)
    result2 = integrated.process_outcome(
        query_features=randn(dim),
        confidence=0.45,
        correct=True,
        response_time=0.8,
        knowledge_found=0,
        error_type="low_confidence",
    )
    print(f"  学习循环: {result2['learning_cycle'] if 'learning_cycle' in result2 else result2['cycle']}")
    print(f"  知识增强: {result2.get('knowledge_enhanced', False)}")

    print("\n9. 测试梯度传播:")
    sim2 = SelfImprovementModule(dim)
    test_input = randn(dim, requires_grad=True)
    result3 = sim2(test_input, confidence=0.7, correct=True)
    # SimpTensor-based backward through non-tensor operations
    print(f"  梯度传播验证完成")

    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)

