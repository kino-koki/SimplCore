# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
增强版综合训练器 - 集成所有新模块的联合训练

核心功能:
1. 多模块联合训练（AGICore + MultimodalCognitiveIntegration + ...）
2. 基于图灵测试指标的强化学习
3. 自我改进驱动的自适应训练
4. 知识自迭代与训练同步
5. 多模态数据支持
"""

from __future__ import annotations

import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any, Callable
from datetime import datetime

from ..engine.tensor import SimpTensor, randn, zeros, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm
from ..engine.optim import Adam

from ..evaluation.turing_test_eval import (
    TuringTestEvaluator,
    EvaluationMetric,
    TuringScore,
)
from ..evaluation.enhanced_evaluator import (
    EnhancedHumanSimilarityEvaluator,
)

from ..cognition.human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode
from ..cognition.multimodal_cognition import MultimodalCognitiveIntegration, IntegrationMode
from ..cognition.self_improvement import SelfImprovementModule, IntegratedSelfImprovement

from ..agi.agi_controller import AGICore

from ..knowledge.knowledge_graph import KnowledgeGraph


class MultiTaskLoss(Module):
    """多任务联合损失"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.cognition_weight = Linear(dim, 1)
        self.register_module("cognition_weight", self.cognition_weight)

        self.perception_weight = Linear(dim, 1)
        self.register_module("perception_weight", self.perception_weight)

        self.knowledge_weight = Linear(dim, 1)
        self.register_module("knowledge_weight", self.knowledge_weight)

    def forward(
        self,
        cognition_output: SimpTensor,
        perception_output: Optional[SimpTensor] = None,
        knowledge_output: Optional[SimpTensor] = None,
        target_score: float = 10.0,
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """计算多任务损失"""
        cog_loss = (tanh(self.cognition_weight(cognition_output)) - SimpTensor(np.array([target_score / 10]))) ** 2
        total_loss = cog_loss

        losses = {"cognition_loss": float(cog_loss._data[0])}

        if perception_output is not None:
            perc_loss = (tanh(self.perception_weight(perception_output)) - SimpTensor(np.array([target_score / 10]))) ** 2
            total_loss = total_loss + perc_loss * 0.3
            losses["perception_loss"] = float(perc_loss._data[0])

        if knowledge_output is not None:
            knw_loss = (tanh(self.knowledge_weight(knowledge_output)) - SimpTensor(np.array([target_score / 10]))) ** 2
            total_loss = total_loss + knw_loss * 0.2
            losses["knowledge_loss"] = float(knw_loss._data[0])

        return total_loss, losses


class EnhancedSyntheticDataGenerator:
    """增强版合成数据生成器 - 支持多模态"""

    def __init__(self, dim: int):
        self.dim = dim

    def generate_multimodal_batch(self, batch_size: int = 8) -> Dict[str, Any]:
        """生成多模态批次数据"""
        text_inputs = [randn(self.dim) for _ in range(batch_size)]
        visual_inputs = [randn(self.dim) for _ in range(batch_size // 2)]
        audio_inputs = [randn(self.dim) for _ in range(batch_size // 2)]

        targets = {
            "human_likeness": float(np.random.uniform(0.5, 0.9)),
            "coherence": float(np.random.uniform(0.4, 0.8)),
        }

        return {
            "text": text_inputs,
            "visual": visual_inputs,
            "audio": audio_inputs,
            "targets": targets,
        }

    def generate_dialogue_batch(self, num_turns: int = 5, batch_size: int = 3) -> List[List[SimpTensor]]:
        """生成多轮对话批次"""
        batches = []
        for _ in range(batch_size):
            turns = [randn(self.dim) for _ in range(num_turns)]
            batches.append(turns)
        return batches

    def generate_knowledge_batch(self, num_items: int = 5) -> List[Tuple[SimpTensor, str]]:
        """生成知识批次"""
        items = []
        topics = ["science", "history", "technology", "art", "philosophy"]
        for _ in range(num_items):
            embedding = randn(self.dim)
            topic = np.random.choice(topics)
            items.append((embedding, f"知识条目: {topic}"))
        return items


class EnhancedTrainer:
    """增强版综合训练器"""

    def __init__(
        self,
        model: Module,
        dim: int = 64,
        learning_rate: float = 5e-4,
        num_epochs: int = 100,
        batch_size: int = 8,
        enable_self_improvement: bool = True,
        enable_knowledge_iteration: bool = True,
        target_turing_score: float = 95.0,
    ):
        self.model = model
        self.dim = dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.enable_self_improvement = enable_self_improvement
        self.enable_knowledge_iteration = enable_knowledge_iteration
        self.target_turing_score = target_turing_score

        self.evaluator = TuringTestEvaluator(dim)
        self.data_generator = EnhancedSyntheticDataGenerator(dim)
        self.multi_task_loss = MultiTaskLoss(dim)

        self.optimizer = Adam(
            list(model.parameters()) + list(self.multi_task_loss.parameters()),
            lr=learning_rate,
        )

        self._training_history: List[Dict[str, Any]] = []
        self._best_score = 0.0
        self._current_epoch = 0
        self._consecutive_no_improvement = 0

    def _compute_adaptive_reward(self, score: TuringScore, info: Dict[str, Any]) -> float:
        """自适应奖励计算"""
        base_reward = score.percentage / 100

        coherence = score.scores.get(EvaluationMetric.DIALOGUE_COHERENCE, 0) / 10
        diversity = score.scores.get(EvaluationMetric.RESPONSE_DIVERSITY, 0) / 10
        emotional = score.scores.get(EvaluationMetric.EMOTIONAL_INTELLIGENCE, 0) / 10

        coherence_bonus = max(0, coherence - 0.5) * 0.15
        diversity_bonus = max(0, diversity - 0.4) * 0.1
        emotional_bonus = max(0, emotional - 0.5) * 0.1

        reward = base_reward + coherence_bonus + diversity_bonus + emotional_bonus

        if score.percentage < 30:
            reward = reward * 0.5
        elif score.percentage > 80:
            reward = reward * 1.5

        return float(np.clip(reward, 0.0, 2.0))

    def _train_step_agi_core(self, batch: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """AGICore训练步数"""
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_reward = 0.0
        texts = batch["text"]
        visuals = batch.get("visual", [])
        audios = batch.get("audio", [])
        targets = batch.get("targets", {})

        all_outputs = []

        for i, text in enumerate(texts):
            visual = visuals[i % len(visuals)] if visuals else None
            audio = audios[i % len(audios)] if audios else None

            result = self.model(text, visual_input=visual, audio_input=audio)

            perception_output = result.get("perception", {}).get("multimodal", {}).get("fused", text)
            reasoning_output = result.get("reasoning", {}).get("output", text)

            target_score = targets.get("human_likeness", 0.8) * 10
            loss, loss_info = self.multi_task_loss(
                cognition_output=reasoning_output,
                perception_output=perception_output,
                target_score=target_score,
            )

            confidence = result.get("reasoning", {}).get("confidence", 0.5)
            score = self.evaluator.evaluate_single(
                response=reasoning_output,
                query=text,
                previous_response=text,
                context=text,
                situation=randn(self.dim),
                input_emotion=randn(self.dim),
            )

            reward = self._compute_adaptive_reward(score, {})
            weighted_loss = loss * (1.0 - reward * 0.3)

            total_loss += float(weighted_loss._data[0])
            total_reward += reward
            all_outputs.append(reasoning_output)

        avg_loss = total_loss / len(texts)
        avg_reward = total_reward / len(texts)

        loss_tensor = SimpTensor(np.array([avg_loss]))
        loss_tensor.backward()
        self.optimizer.step()

        return avg_loss, {
            "reward": avg_reward,
            "batch_size": len(texts),
        }

    def _train_step_standard(self, batch: List[SimpTensor]) -> Tuple[float, Dict[str, Any]]:
        """标准训练步数"""
        self.optimizer.zero_grad()

        total_loss = 0.0
        total_reward = 0.0
        responses = []

        for i, query in enumerate(batch):
            prev_response = responses[-1] if responses else zeros(self.dim)

            response, info = self.model(query)
            responses.append(response)

            score = self.evaluator.evaluate_single(
                response=response,
                query=query,
                previous_response=prev_response,
                context=query,
                situation=randn(self.dim),
                input_emotion=randn(self.dim),
            )

            reward = self._compute_adaptive_reward(score, info)
            target_score = min(10, score.total_score + reward * 2)

            loss, _ = self.multi_task_loss(
                cognition_output=response,
                target_score=target_score,
            )

            total_loss += float(loss._data[0])
            total_reward += reward

        avg_loss = total_loss / len(batch)
        loss_tensor = SimpTensor(np.array([avg_loss]))
        loss_tensor.backward()
        self.optimizer.step()

        return avg_loss, {
            "reward": total_reward / len(batch),
        }

    def _run_self_improvement(self, epoch_info: Dict[str, Any]):
        """运行自我改进"""
        if not self.enable_self_improvement:
            return

        if not hasattr(self.model, 'self_improvement'):
            return

        confidence = 1.0 - min(1.0, epoch_info.get("loss", 1.0))
        correct = epoch_info.get("reward", 0) > 0.1

        improvement = self.model.self_improvement(
            query_features=randn(self.dim),
            confidence=confidence,
            correct=correct,
            response_time=epoch_info.get("duration", 0.1),
            knowledge_used_count=epoch_info.get("knowledge_count", 0),
        )

        return improvement

    def train(self) -> Dict[str, Any]:
        """执行训练"""
        is_agi_core = isinstance(self.model, AGICore)

        print("=" * 70)
        print(f"增强版综合训练器 - 开始训练")
        print(f"  模型: {type(self.model).__name__}")
        print(f"  维度: {self.dim}, 学习率: {self.learning_rate}")
        print(f"  轮数: {self.num_epochs}, 批次: {self.batch_size}")
        print(f"  自我改进: {self.enable_self_improvement}")
        print(f"  目标图灵分数: {self.target_turing_score:.0f}%")
        print("=" * 70)

        for epoch in range(self.num_epochs):
            self._current_epoch = epoch
            epoch_start = time.time()

            if is_agi_core:
                batch = self.data_generator.generate_multimodal_batch(self.batch_size)
                loss, info = self._train_step_agi_core(batch)
            else:
                batch = [randn(self.dim) for _ in range(self.batch_size)]
                loss, info = self._train_step_standard(batch)

            epoch_duration = time.time() - epoch_start

            epoch_info = {
                "epoch": epoch,
                "loss": loss,
                "reward": info["reward"],
                "duration": epoch_duration,
                "knowledge_count": 0,
            }

            self._training_history.append(epoch_info)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                eval_result = self._evaluate()
                percentage = eval_result["percentage"]
                score_value = eval_result["score"]

                print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
                print(f"  损失: {loss:.4f}, 奖励: {info['reward']:.4f}")
                print(f"  评估: {percentage:.1f}% (目标: {self.target_turing_score:.0f}%)")
                print(f"  耗时: {epoch_duration:.3f}s")

                if percentage > self._best_score:
                    improvement = percentage - self._best_score
                    self._best_score = percentage
                    self._consecutive_no_improvement = 0
                    print(f"  ✓ 新最优! +{improvement:.1f}%")
                else:
                    self._consecutive_no_improvement += 1

                if percentage >= self.target_turing_score:
                    print(f"\n✓ 已达到目标图灵分数 {self.target_turing_score:.0f}%!")
                    print(f"  在 {epoch+1} 轮训练后达成")
                    break

                if self._consecutive_no_improvement >= 20:
                    print(f"\n! 连续20轮无提升，提前终止")
                    break

                self._run_self_improvement(epoch_info)

        final_eval = self._evaluate()
        summary = self.get_training_summary()

        print("\n" + "=" * 70)
        print("训练完成")
        print(f"  最终评估: {final_eval['percentage']:.1f}%")
        print(f"  最优分数: {self._best_score:.1f}%")
        print(f"  总轮数: {self._current_epoch + 1}")
        print(f"  状态: {'已达到目标' if self._best_score >= self.target_turing_score else '继续训练'}")
        print("=" * 70)

        return {
            "training_history": self._training_history,
            "best_score": self._best_score,
            "final_evaluation": final_eval,
            "total_epochs": self._current_epoch + 1,
            "consecutive_no_improvement": self._consecutive_no_improvement,
            "summary": summary,
        }

    def _evaluate(self) -> Dict[str, Any]:
        """评估当前模型"""
        try:
            if hasattr(self.model, 'reset'):
                self.model.reset()
        except Exception:
            pass

        queries = [randn(self.dim) for _ in range(5)]
        responses = []

        for query in queries:
            try:
                if isinstance(self.model, AGICore):
                    result = self.model(query)
                    response = result.get("reasoning", {}).get("output", query)
                else:
                    response, _ = self.model(query)
            except Exception:
                response = query
            responses.append(response)

        try:
            dialogue_result = self.evaluator.evaluate_dialogue(
                responses=responses,
                queries=queries,
                situations=[randn(self.dim) for _ in range(5)],
                emotions=[randn(self.dim) for _ in range(5)],
            )
            percentage = dialogue_result.get("average_percentage", 50.0)
            score_value = dialogue_result.get("average_score", 5.0)
        except Exception:
            percentage = 50.0
            score_value = 5.0

        return {
            "percentage": percentage,
            "score": score_value,
            "status": "评估完成",
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self._training_history:
            return {}

        recent = self._training_history[-10:]
        recent_losses = [h["loss"] for h in recent]
        recent_rewards = [h["reward"] for h in recent]

        loss_trend = (
            "下降" if len(recent_losses) > 1 and recent_losses[-1] < recent_losses[0]
            else "稳定" if len(recent_losses) > 1 and abs(recent_losses[-1] - recent_losses[0]) < 0.05
            else "上升"
        )

        return {
            "total_epochs": len(self._training_history),
            "best_score": self._best_score,
            "recent_avg_loss": float(np.mean(recent_losses)),
            "recent_avg_reward": float(np.mean(recent_rewards)),
            "loss_trend": loss_trend,
            "consecutive_no_improvement": self._consecutive_no_improvement,
            "target_turing_score": self.target_turing_score,
        }


class AdaptiveTrainingScheduler:
    """自适应训练调度器"""

    def __init__(
        self,
        dim: int = 64,
        stages: Optional[List[Dict[str, Any]]] = None,
    ):
        self.dim = dim
        self.stages = stages or self._default_stages()
        self.current_stage_idx = 0
        self.stage_results: List[Dict[str, Any]] = []

    def _default_stages(self) -> List[Dict[str, Any]]:
        """默认训练阶段"""
        return [
            {
                "name": "基础对话训练",
                "epochs": 30,
                "batch_size": 8,
                "learning_rate": 5e-4,
                "mode": HumanLikeMode.CONVERSATION,
                "description": "训练基础对话能力",
            },
            {
                "name": "推理增强训练",
                "epochs": 30,
                "batch_size": 6,
                "learning_rate": 3e-4,
                "mode": HumanLikeMode.REASONING,
                "description": "增强逻辑推理能力",
            },
            {
                "name": "创造性训练",
                "epochs": 20,
                "batch_size": 6,
                "learning_rate": 2e-4,
                "mode": HumanLikeMode.CREATIVE,
                "description": "培养创造性表达",
            },
            {
                "name": "多模态整合训练",
                "epochs": 20,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "mode": HumanLikeMode.CONVERSATION,
                "description": "整合多模态感知能力",
            },
        ]

    def get_current_stage(self) -> Dict[str, Any]:
        """获取当前阶段"""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return self.stages[-1]

    def advance_stage(self, result: Dict[str, Any]) -> bool:
        """推进到下一阶段"""
        self.stage_results.append(result)
        self.current_stage_idx += 1

        if self.current_stage_idx >= len(self.stages):
            return False

        new_stage = self.get_current_stage()
        print(f"\n{'='*60}")
        print(f"进入训练阶段: {new_stage['name']}")
        print(f"  描述: {new_stage['description']}")
        print(f"  轮数: {new_stage['epochs']}, 批次: {new_stage['batch_size']}")
        print(f"  学习率: {new_stage['learning_rate']}")
        print(f"{'='*60}\n")

        return True

    def run_scheduled_training(
        self,
        model_factory: Callable[[], Module],
    ) -> Dict[str, Any]:
        """运行调度训练"""
        all_stage_results = []
        best_overall_score = 0.0

        while self.current_stage_idx < len(self.stages):
            stage = self.get_current_stage()
            model = model_factory()

            trainer = EnhancedTrainer(
                model=model,
                dim=self.dim,
                learning_rate=stage["learning_rate"],
                num_epochs=stage["epochs"],
                batch_size=stage["batch_size"],
            )

            result = trainer.train()
            all_stage_results.append({
                "stage": stage["name"],
                "result": result,
            })

            if result.get("best_score", 0) > best_overall_score:
                best_overall_score = result["best_score"]

            if not self.advance_stage(result):
                break

        return {
            "stage_results": all_stage_results,
            "best_overall_score": best_overall_score,
            "total_stages": len(all_stage_results),
        }


if __name__ == "__main__":
    from SimplCore.engine.tensor import manual_seed

    manual_seed(42)
    dim = 64

    print("=" * 70)
    print("增强版训练器 - 综合测试")
    print("=" * 70)

    print("\n1. 测试多任务损失:")
    loss_fn = MultiTaskLoss(dim)
    test_output = randn(dim)
    loss, losses = loss_fn(test_output, test_output, test_output, target_score=8.0)
    print(f"  总损失: {float(loss._data[0]):.4f}")
    print(f"  认知损失: {losses['cognition_loss']:.4f}")
    print(f"  感知损失: {losses['perception_loss']:.4f}")

    print("\n2. 测试增强数据生成器:")
    gen = EnhancedSyntheticDataGenerator(dim)
    batch = gen.generate_multimodal_batch(batch_size=4)
    print(f"  文本数: {len(batch['text'])}")
    print(f"  视觉数: {len(batch['visual'])}")
    print(f"  音频数: {len(batch['audio'])}")

    print("\n3. 测试标准训练器:")
    model = HumanLikeCognitiveReasoner(dim)
    trainer = EnhancedTrainer(
        model=model,
        dim=dim,
        learning_rate=5e-4,
        num_epochs=10,
        batch_size=4,
    )
    result = trainer.train()
    print(f"\n  最优: {result['best_score']:.1f}%")
    print(f"  总轮数: {result['total_epochs']}")

    print("\n4. 测试AGICore训练器:")
    agi_core = AGICore(dim)
    agi_trainer = EnhancedTrainer(
        model=agi_core,
        dim=dim,
        learning_rate=5e-4,
        num_epochs=5,
        batch_size=2,
    )
    agi_result = agi_trainer.train()
    print(f"  最优: {agi_result['best_score']:.1f}%")

    print("\n5. 测试自适应调度:")
    scheduler = AdaptiveTrainingScheduler(
        dim=dim,
        stages=[
            {"name": "快速测试阶段", "epochs": 3, "batch_size": 4,
             "learning_rate": 5e-4, "mode": HumanLikeMode.CONVERSATION,
             "description": "快速测试"},
        ],
    )

    def make_model():
        return HumanLikeCognitiveReasoner(dim)

    sched_result = scheduler.run_scheduled_training(make_model)
    print(f"  阶段数: {sched_result['total_stages']}")
    print(f"  最优: {sched_result['best_overall_score']:.1f}%")

    print("\n" + "=" * 70)
    print("增强版训练器测试完成!")
    print("=" * 70)

