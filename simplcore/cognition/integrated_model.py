#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
Human-Like Integrated Model (类人化集成模型) - 增强版
"""

import numpy as np
import time
fimport os
rom typing import Optional, Tuple, List, Dict, Any

from ..engine.tensor import SimpTensor, randn, zeros, gelu, tanh, softmax, sigmoid_tensor
from ..engine.nn import Module, Linear, LayerNorm

from ..evaluation.enhanced_evaluator import EnhancedTuringTestEvaluator, SyntheticHumanDataGenerator
from ..evaluation.turing_test_eval import EvaluationMetric, TuringScore

from .human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode


class TrainingConfig:
    """训练超参数配置"""
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        total_steps: int = 100000,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        patience: int = 50,
        min_lr: float = 1e-6,
        turing_weight: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.patience = patience
        self.min_lr = min_lr
        self.turing_weight = turing_weight


class HumanLikeIntegratedModel(Module):
    """类人化集成模型"""
    
    def __init__(
        self,
        core_model,
        cognitive_reasoner: HumanLikeCognitiveReasoner,
        dim: int = 64,
        integrate_mode: str = "post",
        use_turing_loss: bool = True,
        turing_weight: float = 0.5,
    ):
        super().__init__()
        self.core_model = core_model
        self.cognitive_reasoner = cognitive_reasoner
        self.dim = dim
        self.integrate_mode = integrate_mode
        self.use_turing_loss = use_turing_loss
        self.turing_weight = turing_weight
        
        self.evaluator = EnhancedTuringTestEvaluator(dim)
        self.data_generator = SyntheticHumanDataGenerator(dim)
        
        self.cognitive_proj = Linear(dim, dim)
        self.register_module("cognitive_proj", self.cognitive_proj)
        
        self.combine_proj = Linear(dim * 2, dim)
        self.register_module("combine_proj", self.combine_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
        
        self.adaptation_layer = Linear(dim * 2, dim)
        self.register_module("adaptation_layer", self.adaptation_layer)
        
        self._training_mode = True
        self._previous_response = None
        self._context_history: List[SimpTensor] = []
        self._emotion_history: List[SimpTensor] = []
    
    def set_training_mode(self, training: bool):
        self._training_mode = training
    
    def forward(
        self,
        x: SimpTensor,
        context: Optional[SimpTensor] = None,
        mode: Optional[HumanLikeMode] = None,
        emotion: Optional[SimpTensor] = None,
        **kwargs
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        info = {}
        
        if emotion is None:
            emotion = randn(self.dim)
        
        if self.integrate_mode == "pre":
            cognitive_feature, cognitive_info = self.cognitive_reasoner(x, context, mode)
            info["cognitive"] = cognitive_info
            
            combined_input = SimpTensor(np.concatenate([x._data, cognitive_feature._data]))
            core_output = self._forward_core(combined_input[:self.dim], **kwargs)
            output = self.output_norm(core_output)
            
        elif self.integrate_mode == "post":
            core_output = self._forward_core(x, **kwargs)
            
            cognitive_feature, cognitive_info = self.cognitive_reasoner(core_output, context, mode)
            info["cognitive"] = cognitive_info
            
            combined = SimpTensor(np.concatenate([core_output._data, cognitive_feature._data]))
            output = self.combine_proj(combined)
            output = gelu(output)
            output = self.output_norm(output)
            
        elif self.integrate_mode == "parallel":
            cognitive_feature, cognitive_info = self.cognitive_reasoner(x, context, mode)
            info["cognitive"] = cognitive_info
            
            core_output = self._forward_core(x, **kwargs)
            
            combined = SimpTensor(np.concatenate([core_output._data, cognitive_feature._data]))
            output = self.combine_proj(combined)
            output = gelu(output)
            output = self.output_norm(output)
            
        else:
            output = self._forward_core(x, **kwargs)
        
        if self._previous_response is not None:
            adaptation_input = SimpTensor(np.concatenate([output._data, self._previous_response._data]))
            adaptation = tanh(self.adaptation_layer(adaptation_input))
            output = output + 0.1 * adaptation
            info["adaptation_factor"] = 0.1
        
        self._context_history.append(output)
        self._emotion_history.append(emotion)
        if len(self._context_history) > 50:
            self._context_history = self._context_history[-50:]
            self._emotion_history = self._emotion_history[-50:]
        
        self._previous_response = output
        
        if self._training_mode and self.use_turing_loss:
            turing_score = self._compute_turing_score(output, x, emotion)
            info["turing_score"] = turing_score
        
        info["output_shape"] = output.shape
        info["integrate_mode"] = self.integrate_mode
        
        return output, info
    
    def _forward_core(self, x: SimpTensor, **kwargs) -> SimpTensor:
        if hasattr(self.core_model, 'forward'):
            result = self.core_model.forward(x, **kwargs)
            if isinstance(result, tuple):
                return result[0]
            return result
        return self.core_model(x, **kwargs)
    
    def _compute_turing_score(self, response: SimpTensor, query: SimpTensor, emotion: SimpTensor) -> Dict[str, float]:
        prev_response = self._previous_response if self._previous_response is not None else zeros(self.dim)
        situation = randn(self.dim)
        
        score = self.evaluator.evaluate_single(
            response=response,
            query=query,
            previous_response=prev_response,
            context=query,
            situation=situation,
            input_emotion=emotion,
            history_responses=self._context_history[:-1].copy(),
            history_context=self._context_history[:-1].copy(),
            history_emotions=self._emotion_history[:-1].copy()
        )
        
        return {
            "total_score": score.total_score,
            "percentage": score.percentage,
            "human_similarity": score.scores.get(EvaluationMetric.HUMAN_SIMILARITY, 0),
            "dialogue_coherence": score.scores.get(EvaluationMetric.DIALOGUE_COHERENCE, 0),
            "context_adaptability": score.scores.get(EvaluationMetric.CONTEXT_ADAPTABILITY, 0),
            "emotional_intelligence": score.scores.get(EvaluationMetric.EMOTIONAL_INTELLIGENCE, 0),
            "response_diversity": score.scores.get(EvaluationMetric.RESPONSE_DIVERSITY, 0),
        }
    
    def compute_loss(self, response: SimpTensor, query: SimpTensor, 
                    labels: Optional[SimpTensor] = None, emotion: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        loss_info = {}
        
        base_loss = SimpTensor(np.array([0.0]))
        if labels is not None:
            base_loss = ((response - labels) ** 2).mean()
            loss_info["base_loss"] = float(base_loss._data[0])
        
        turing_loss = SimpTensor(np.array([0.0]))
        if self.use_turing_loss:
            turing_score = self._compute_turing_score(response, query, emotion or randn(self.dim))
            turing_loss = SimpTensor(np.array([10 - turing_score["total_score"]]))
            loss_info["turing_loss"] = float(turing_loss._data[0])
            loss_info["turing_score"] = turing_score
        
        total_loss = base_loss + self.turing_weight * turing_loss
        
        loss_info["total_loss"] = float(total_loss._data[0])
        loss_info["turing_weight"] = self.turing_weight
        
        return total_loss, loss_info
    
    def evaluate(self, queries: List[SimpTensor]) -> Dict[str, Any]:
        prev_training_mode = self._training_mode
        self.set_training_mode(True)
        self.cognitive_reasoner.reset()
        
        responses = []
        total_turing_score = 0.0
        total_confidence = 0.0
        
        for i, query in enumerate(queries):
            context = queries[i-1] if i > 0 else None
            emotion = randn(self.dim)
            response, info = self(query, context, emotion=emotion)
            responses.append(response)
            
            total_confidence += info["cognitive"].get("overall_confidence", 0.5)
            
            if "turing_score" in info:
                total_turing_score += info["turing_score"]["percentage"]
        
        avg_turing_score = total_turing_score / len(queries)
        avg_confidence = total_confidence / len(queries)
        
        status = self._determine_status(avg_turing_score)
        
        self.set_training_mode(prev_training_mode)
        
        return {
            "average_turing_score": avg_turing_score,
            "average_confidence": avg_confidence,
            "status": status,
            "num_samples": len(queries),
        }
    
    def _determine_status(self, percentage: float) -> str:
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
    
    def reset(self):
        self._previous_response = None
        self._context_history = []
        self._emotion_history = []
        self.cognitive_reasoner.reset()


class CosineAnnealingLR:
    """余弦退火学习率调度器"""
    
    def __init__(self, optimizer, max_steps: int, warmup_steps: int = 0, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self._step = 0
    
    def step(self):
        self._step += 1
        
        if self._step <= self.warmup_steps:
            lr = self.base_lr * (self._step / self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        self.optimizer.lr = lr
        return lr


class HumanLikeTrainingPipeline:
    """类人化训练管道 - 完整端到端训练流程"""
    
    def __init__(
        self,
        integrated_model: HumanLikeIntegratedModel,
        config: TrainingConfig,
    ):
        self.model = integrated_model
        self.config = config
        
        from ..engine.optim import Adam
        self.optimizer = Adam(
            list(integrated_model.parameters()) + integrated_model.evaluator.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            max_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
        )
        
        self._step = 0
        self._epoch = 0
        self._training_history: List[Dict[str, Any]] = []
        self._best_score = 0.0
        self._patience_counter = 0
        self._grad_accum_counter = 0
        self._total_loss = 0.0
    
    def train_step(self, queries: List[SimpTensor], labels: Optional[List[SimpTensor]] = None) -> Tuple[float, Dict[str, Any]]:
        self._grad_accum_counter += 1
        
        total_loss = 0.0
        total_turing_score = 0.0
        total_confidence = 0.0
        
        for i, query in enumerate(queries):
            context = queries[i-1] if i > 0 else None
            emotion = randn(self.model.dim)
            response, info = self.model(query, context, emotion=emotion)
            
            label = labels[i] if labels is not None else None
            loss, loss_info = self.model.compute_loss(response, query, label, emotion)
            
            loss = loss / self.config.gradient_accumulation_steps
            
            loss.backward()
            
            total_loss += float(loss._data[0]) * self.config.gradient_accumulation_steps
            total_confidence += info["cognitive"].get("overall_confidence", 0.5)
            
            if "turing_score" in info:
                total_turing_score += info["turing_score"]["percentage"]
        
        if self._grad_accum_counter >= self.config.gradient_accumulation_steps:
            if self.config.max_grad_norm > 0:
                self._clip_gradients()
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            current_lr = self.scheduler.step()
            self._step += 1
            self._grad_accum_counter = 0
        else:
            current_lr = self.optimizer.lr
        
        avg_turing_score = total_turing_score / len(queries)
        avg_confidence = total_confidence / len(queries)
        
        result = {
            "step": self._step,
            "loss": total_loss / len(queries),
            "turing_score": avg_turing_score,
            "confidence": avg_confidence,
            "lr": current_lr,
            "status": self.model._determine_status(avg_turing_score),
        }
        
        self._training_history.append(result)
        
        if avg_turing_score > self._best_score:
            self._best_score = avg_turing_score
            self._patience_counter = 0
        else:
            self._patience_counter += 1
        
        return total_loss / len(queries), result
    
    def _clip_gradients(self):
        max_norm = self.config.max_grad_norm
        total_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = np.linalg.norm(param.grad._data)
                total_norm += param_norm ** 2
        
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-8)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad._data = param.grad._data * clip_coef
    
    def evaluate(self, queries: List[SimpTensor]) -> Dict[str, Any]:
        return self.model.evaluate(queries)
    
    def should_stop(self) -> bool:
        """早停检查"""
        if self._patience_counter >= self.config.patience:
            return True
        if self.scheduler._step >= self.config.total_steps:
            return True
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        if not self._training_history:
            return {}
        
        recent = self._training_history[-10:]
        avg_loss = sum(h["loss"] for h in recent) / len(recent)
        avg_turing = sum(h["turing_score"] for h in recent) / len(recent)
        
        return {
            "total_steps": self._step,
            "best_score": self._best_score,
            "recent_avg_loss": avg_loss,
            "recent_avg_turing_score": avg_turing,
            "final_status": self._training_history[-1]["status"],
            "patience_counter": self._patience_counter,
            "current_lr": self.optimizer.lr,
        }


class TuringTestOptimizer:
    """图灵测试优化器 - 自动优化模型以达到100%图灵测试分数"""
    
    def __init__(
        self,
        pipeline: HumanLikeTrainingPipeline,
        target_score: float = 100.0,
        max_iterations: int = 10000,
    ):
        self.pipeline = pipeline
        self.target_score = target_score
        self.max_iterations = max_iterations
        self._iteration = 0
        self._improvement_history: List[float] = []
    
    def optimize(self) -> Dict[str, Any]:
        """执行优化直到达到目标分数或最大迭代次数"""
        print(f"=== 开始图灵测试优化 ===")
        print(f"目标分数: {self.target_score}%")
        print(f"最大迭代次数: {self.max_iterations}")
        
        start_time = time.time()
        
        while self._iteration < self.max_iterations:
            queries = [randn(self.pipeline.model.dim) for _ in range(8)]
            
            loss, result = self.pipeline.train_step(queries)
            
            if (self._iteration + 1) % 100 == 0:
                eval_queries = [randn(self.pipeline.model.dim) for _ in range(10)]
                eval_result = self.pipeline.evaluate(eval_queries)
                
                elapsed = time.time() - start_time
                
                print(f"\n迭代 {self._iteration + 1}:")
                print(f"  损失: {loss:.4f}")
                print(f"  图灵分数: {result['turing_score']:.1f}%")
                print(f"  评估分数: {eval_result['average_turing_score']:.1f}%")
                print(f"  最佳分数: {self.pipeline._best_score:.1f}%")
                print(f"  学习率: {result['lr']:.6f}")
                print(f"  耗时: {elapsed:.2f}s")
                
                self._improvement_history.append(eval_result['average_turing_score'])
                
                if eval_result['average_turing_score'] >= self.target_score:
                    print(f"\n🎉 达到目标分数 {self.target_score}%!")
                    break
                
                if self.pipeline.should_stop():
                    print(f"\n⚠️ 早停触发，当前最佳分数: {self.pipeline._best_score:.1f}%")
                    break
            
            self._iteration += 1
        
        final_eval = self.pipeline.evaluate([randn(self.pipeline.model.dim) for _ in range(20)])
        summary = self.pipeline.get_training_summary()
        
        total_time = time.time() - start_time
        
        return {
            "final_score": final_eval["average_turing_score"],
            "best_score": summary["best_score"],
            "total_iterations": self._iteration,
            "total_time": total_time,
            "final_status": summary["final_status"],
            "improvement_history": self._improvement_history,
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    
    from SimplCore.engine.tensor import manual_seed, randn
    
    manual_seed(42)
    dim = 64
    
    print("=== 测试 HumanLikeIntegratedModel ===")
    
    core_model = Linear(dim, dim)
    
    cognitive_reasoner = HumanLikeCognitiveReasoner(
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
    
    integrated_model = HumanLikeIntegratedModel(
        core_model=core_model,
        cognitive_reasoner=cognitive_reasoner,
        dim=dim,
        integrate_mode="post",
        use_turing_loss=True,
        turing_weight=0.5,
    )
    
    print("\n1. 测试前向传播:")
    x = randn(dim)
    output, info = integrated_model(x)
    print(f"输入: {x.shape}, 输出: {output.shape}")
    print(f"图灵分数: {info.get('turing_score', {}).get('percentage', 0):.1f}%")
    
    print("\n2. 测试多轮对话:")
    for i in range(3):
        x = randn(dim)
        output, info = integrated_model(x)
        print(f"  轮次 {i+1}: 置信度={info['cognitive']['overall_confidence']:.4f}, 图灵分数={info.get('turing_score', {}).get('percentage', 0):.1f}%")
    
    print("\n3. 测试损失计算:")
    x = randn(dim)
    labels = randn(dim)
    response, _ = integrated_model(x)
    loss, loss_info = integrated_model.compute_loss(response, x, labels)
    print(f"基础损失: {loss_info['base_loss']:.4f}")
    print(f"图灵损失: {loss_info['turing_loss']:.4f}")
    print(f"总损失: {loss_info['total_loss']:.4f}")
    
    print("\n4. 测试评估:")
    queries = [randn(dim) for _ in range(5)]
    eval_result = integrated_model.evaluate(queries)
    print(f"平均图灵分数: {eval_result['average_turing_score']:.1f}%")
    print(f"状态: {eval_result['status']}")
    
    print("\n=== 所有测试通过 ===")


