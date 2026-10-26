#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
训练校准循环 - 优化类人化模型参数

核心功能：
1. 基于图灵测试指标的强化学习训练
2. 多任务联合训练
3. 自监督训练数据生成
4. 评估驱动的参数优化
5. 渐进式训练策略
"""

import numpy as np
import time
fimport os
rom typing import Optional, Tuple, List, Dict, Any

from ..engine.tensor import SimpTensor, randn, zeros, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm
from ..engine.optim import Adam

from ..evaluation.turing_test_eval import (
    TuringTestEvaluator,
    EvaluationMetric,
    TuringScore,
)
from ..cognition.human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode


class CalibrationLoss(Module):
    """校准损失函数"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.similarity_weight = Linear(dim, 1)
        self.register_module("similarity_weight", self.similarity_weight)
        
        self.coherence_weight = Linear(dim, 1)
        self.register_module("coherence_weight", self.coherence_weight)
        
        self.adaptability_weight = Linear(dim, 1)
        self.register_module("adaptability_weight", self.adaptability_weight)
    
    def forward(
        self,
        response: SimpTensor,
        target_score: float,
        metric_scores: Dict[str, float]
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """计算校准损失"""
        score_feature = tanh(self.similarity_weight(response))
        target_tensor = SimpTensor(np.array([target_score]))
        
        loss = (score_feature - target_tensor) ** 2
        
        weighted_loss = loss * 0.5
        
        if "dialogue_coherence" in metric_scores:
            coherence_penalty = max(0, 5 - metric_scores["dialogue_coherence"])
            weighted_loss = weighted_loss + SimpTensor(np.array([coherence_penalty])) * 0.3
        
        info = {
            "base_loss": float(loss._data[0]),
            "weighted_loss": float(weighted_loss._data[0]),
            "target_score": target_score,
        }
        
        return weighted_loss, info


class SyntheticDataGenerator:
    """合成数据生成器"""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def generate_conversation(self, length: int = 5) -> Tuple[List[SimpTensor], List[SimpTensor]]:
        """生成对话数据"""
        queries = []
        responses = []
        
        for i in range(length):
            query = randn(self.dim)
            response = tanh(query * 0.8 + randn(self.dim) * 0.2)
            queries.append(query)
            responses.append(response)
        
        return queries, responses
    
    def generate_emotional_data(self, num_samples: int = 10) -> Tuple[List[SimpTensor], List[int]]:
        """生成情绪数据"""
        samples = []
        labels = []
        
        for _ in range(num_samples):
            sample = randn(self.dim)
            emotion_label = np.random.randint(0, 13)
            samples.append(sample)
            labels.append(emotion_label)
        
        return samples, labels
    
    def generate_contextual_data(self, num_samples: int = 10) -> List[List[SimpTensor]]:
        """生成上下文数据"""
        contexts = []
        
        for _ in range(num_samples):
            context_length = np.random.randint(2, 8)
            context = [randn(self.dim) for _ in range(context_length)]
            contexts.append(context)
        
        return contexts


class CalibrationTrainer:
    """校准训练器"""
    
    def __init__(
        self,
        model: HumanLikeCognitiveReasoner,
        dim: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        batch_size: int = 8,
    ):
        self.model = model
        self.dim = dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.evaluator = TuringTestEvaluator(dim)
        self.data_generator = SyntheticDataGenerator(dim)
        self.calibration_loss = CalibrationLoss(dim)
        
        self.optimizer = Adam(
            list(model.parameters()) + list(self.calibration_loss.parameters()),
            lr=learning_rate
        )
        
        self._training_history: List[Dict[str, Any]] = []
        self._best_score = 0.0
    
    def _compute_reward(self, score: TuringScore) -> float:
        """计算奖励"""
        reward = score.percentage / 100
        
        coherence_bonus = min(0.2, (score.scores.get(EvaluationMetric.DIALOGUE_COHERENCE, 0) - 4) / 6)
        reward = reward + coherence_bonus
        
        diversity_penalty = min(0.1, (8 - score.scores.get(EvaluationMetric.RESPONSE_DIVERSITY, 0)) / 2)
        reward = reward - diversity_penalty
        
        return reward
    
    def _train_step(self, queries: List[SimpTensor]) -> Tuple[float, Dict[str, Any]]:
        """单步训练"""
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_reward = 0.0
        metric_scores = {}
        
        responses = []
        history_responses = []
        
        for i, query in enumerate(queries):
            prev_response = responses[i-1] if i > 0 else zeros(self.dim)
            situation = randn(self.dim)
            emotion = randn(self.dim)
            
            response, info = self.model(query)
            responses.append(response)
            
            score = self.evaluator.evaluate_single(
                response=response,
                query=query,
                previous_response=prev_response,
                context=query,
                situation=situation,
                input_emotion=emotion,
                history_responses=history_responses.copy()
            )
            
            history_responses.append(response)
            
            reward = self._compute_reward(score)
            target_score = min(10, score.total_score + reward * 2)
            
            loss, loss_info = self.calibration_loss(response, target_score, score.scores)
            
            total_loss += float(loss._data[0])
            total_reward += reward
            
            for metric, value in score.scores.items():
                metric_scores[metric.value] = metric_scores.get(metric.value, 0) + value
        
        total_loss /= len(queries)
        
        avg_loss = SimpTensor(np.array([total_loss]))
        avg_loss.backward()
        
        self.optimizer.step()
        
        for metric in metric_scores:
            metric_scores[metric] /= len(queries)
        
        return total_loss, {
            "reward": total_reward / len(queries),
            "metric_scores": metric_scores,
        }
    
    def train(self) -> Dict[str, Any]:
        """执行训练"""
        print("=== 开始校准训练 ===")
        print(f"学习率: {self.learning_rate}")
        print(f"训练轮数: {self.num_epochs}")
        print(f"批次大小: {self.batch_size}")
        print()
        
        for epoch in range(self.num_epochs):
            queries, _ = self.data_generator.generate_conversation(length=self.batch_size)
            
            loss, info = self._train_step(queries)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                eval_score = self._evaluate()
                
                print(f"Epoch {epoch+1}/{self.num_epochs}:")
                print(f"  损失: {loss:.4f}")
                print(f"  奖励: {info['reward']:.4f}")
                print(f"  评估得分: {eval_score['percentage']:.1f}%")
                print(f"  状态: {eval_score['status']}")
                
                if eval_score['percentage'] > self._best_score:
                    self._best_score = eval_score['percentage']
                    print(f"  新最佳得分: {self._best_score:.1f}%")
                
                print()
            
            self._training_history.append({
                "epoch": epoch,
                "loss": loss,
                "reward": info["reward"],
                "metric_scores": info["metric_scores"],
            })
        
        final_eval = self._evaluate()
        
        return {
            "training_history": self._training_history,
            "best_score": self._best_score,
            "final_evaluation": final_eval,
            "total_epochs": self.num_epochs,
        }
    
    def _evaluate(self) -> Dict[str, Any]:
        """评估当前模型"""
        self.model.reset()
        
        queries = [randn(self.dim) for _ in range(5)]
        responses = []
        
        for query in queries:
            response, _ = self.model(query)
            responses.append(response)
        
        dialogue_result = self.evaluator.evaluate_dialogue(
            responses=responses,
            queries=queries,
            situations=[randn(self.dim) for _ in range(5)],
            emotions=[randn(self.dim) for _ in range(5)]
        )
        
        return {
            "percentage": dialogue_result["average_percentage"],
            "score": dialogue_result["average_score"],
            "status": self._determine_status(dialogue_result["average_percentage"]),
        }
    
    def _determine_status(self, percentage: float) -> str:
        """确定状态"""
        if percentage >= 95:
            return "EXCELLENT"
        elif percentage >= 80:
            return "GOOD"
        elif percentage >= 60:
            return "ACCEPTABLE"
        elif percentage >= 40:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self._training_history:
            return {}
        
        recent = self._training_history[-10:]
        avg_loss = sum(h["loss"] for h in recent) / len(recent)
        avg_reward = sum(h["reward"] for h in recent) / len(recent)
        
        return {
            "total_epochs": len(self._training_history),
            "best_score": self._best_score,
            "recent_avg_loss": avg_loss,
            "recent_avg_reward": avg_reward,
            "final_metrics": self._training_history[-1]["metric_scores"] if self._training_history else {},
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed
    from SimplCore.cognition.human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode
    
    manual_seed(42)
    dim = 64
    
    print("=== 测试校准训练器 ===")
    
    hlcr = HumanLikeCognitiveReasoner(
        dim=dim,
        mode=HumanLikeMode.CONVERSATION,
        enable_nlu=True,
        enable_disambiguation=True,
        enable_dialogue_tracking=True,
        enable_long_range_memory=True,
        enable_emotional_intelligence=True,
        enable_response_variety=True,
        enable_reasoning=True,
    )
    
    trainer = CalibrationTrainer(
        model=hlcr,
        dim=dim,
        learning_rate=5e-4,
        num_epochs=50,
        batch_size=6,
    )
    
    result = trainer.train()
    
    print("=== 训练完成 ===")
    print(f"最佳得分: {result['best_score']:.1f}%")
    print(f"最终评估: {result['final_evaluation']['percentage']:.1f}%")
    print(f"最终状态: {result['final_evaluation']['status']}")
    
    summary = trainer.get_training_summary()
    print(f"\n训练摘要:")
    print(f"  总轮数: {summary['total_epochs']}")
    print(f"  最近平均损失: {summary['recent_avg_loss']:.4f}")
    print(f"  最近平均奖励: {summary['recent_avg_reward']:.4f}")

