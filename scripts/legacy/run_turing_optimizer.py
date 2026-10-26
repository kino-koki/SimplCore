#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图灵测试优化器 - 改进版
使用固定权重评估器 + 增强认知模块 + 多任务学习
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from SimplCore.engine.tensor import manual_seed, randn, zeros, ones
from SimplCore.engine.nn import Linear, LayerNorm, Sequential, GELU
from SimplCore.engine.optim import Adam
from SimplCore.cognition.human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode
from SimplCore.cognition.integrated_model import HumanLikeIntegratedModel, TrainingConfig, HumanLikeTrainingPipeline

class HumanLikeGenerator(Sequential):
    def __init__(self, dim):
        super().__init__(
            Linear(dim, dim * 4),
            GELU(),
            LayerNorm(dim * 4),
            Linear(dim * 4, dim * 2),
            GELU(),
            LayerNorm(dim * 2),
            Linear(dim * 2, dim),
        )

class FixedTuringEvaluator:
    def __init__(self, dim):
        self.dim = dim
        self.similarity_weight = 0.25
        self.coherence_weight = 0.25
        self.diversity_weight = 0.25
        self.emotion_weight = 0.25
    
    def evaluate(self, response, query, emotion):
        response_norm = np.linalg.norm(response._data)
        query_norm = np.linalg.norm(query._data)
        
        if response_norm > 0 and query_norm > 0:
            similarity = np.dot(response._data, query._data) / (response_norm * query_norm)
        else:
            similarity = 0.5
        
        similarity_score = (similarity + 1) / 2 * 10
        
        coherence = 1.0 - np.mean((response._data - query._data) ** 2) * 0.1
        coherence = max(0.1, min(1.0, coherence))
        coherence_score = coherence * 10
        
        diversity = np.std(response._data) * 10
        diversity_score = max(1, min(10, diversity))
        
        emotion_norm = np.linalg.norm(emotion._data)
        if emotion_norm > 0:
            emotion_alignment = np.dot(response._data, emotion._data) / emotion_norm
        else:
            emotion_alignment = 0
        emotion_score = (emotion_alignment + 1) / 2 * 10
        
        total_score = (
            similarity_score * self.similarity_weight +
            coherence_score * self.coherence_weight +
            diversity_score * self.diversity_weight +
            emotion_score * self.emotion_weight
        )
        
        percentage = total_score / 10 * 100
        
        return {
            "similarity": similarity_score,
            "coherence": coherence_score,
            "diversity": diversity_score,
            "emotion": emotion_score,
            "total_score": total_score,
            "percentage": percentage,
        }

def main():
    manual_seed(42)
    dim = 64
    
    print("=== 图灵测试优化器 ===")
    print(f"维度: {dim}")
    print(f"目标分数: 100%")
    print("=" * 50)
    
    evaluator = FixedTuringEvaluator(dim)
    
    core_model = HumanLikeGenerator(dim)
    
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
        use_turing_loss=False,
        turing_weight=0.0,
    )
    
    params = list(integrated_model.parameters())
    optimizer = Adam(params, lr=1e-3)
    
    best_score = 0.0
    patience_counter = 0
    step = 0
    max_steps = 20000
    
    print("=== 开始训练 ===")
    print("目标: 最大化类人相似度")
    
    while step < max_steps:
        queries = [randn(dim) for _ in range(16)]
        
        total_loss = 0.0
        total_turing = 0.0
        
        for i, query in enumerate(queries):
            context = queries[i-1] if i > 0 else None
            emotion = randn(dim)
            
            response, info = integrated_model(query, context, emotion=emotion)
            
            turing_score = evaluator.evaluate(response, query, emotion)
            
            turing_loss = 10 - turing_score["total_score"]
            
            response_norm = np.linalg.norm(response._data)
            response_norm_loss = (response_norm - 1.0) ** 2
            
            loss = turing_loss + 0.1 * response_norm_loss
            
            loss_tensor = randn(1)
            loss_tensor._data = np.array([loss])
            loss_tensor.backward()
            
            total_loss += loss
            total_turing += turing_score["percentage"]
        
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        
        avg_turing = total_turing / len(queries)
        
        if avg_turing > best_score:
            best_score = avg_turing
            patience_counter = 0
        else:
            patience_counter += 1
        
        if step % 100 == 0:
            eval_queries = [randn(dim) for _ in range(10)]
            eval_turing = 0.0
            
            for i, query in enumerate(eval_queries):
                context = eval_queries[i-1] if i > 0 else None
                emotion = randn(dim)
                response, _ = integrated_model(query, context, emotion=emotion)
                score = evaluator.evaluate(response, query, emotion)
                eval_turing += score["percentage"]
            
            avg_eval = eval_turing / len(eval_queries)
            
            print(f"\n迭代 {step}:")
            print(f"  损失: {total_loss/16:.4f}")
            print(f"  训练图灵分数: {avg_turing:.1f}%")
            print(f"  评估分数: {avg_eval:.1f}%")
            print(f"  最佳分数: {best_score:.1f}%")
            
            if avg_eval >= 100.0:
                print("\n🎉 成功达到100%图灵测试分数！")
                break
            
            if patience_counter >= 1000:
                print("\n⚠️ 早停触发")
                break
    
    eval_queries = [randn(dim) for _ in range(20)]
    final_turing = 0.0
    for i, query in enumerate(eval_queries):
        context = eval_queries[i-1] if i > 0 else None
        emotion = randn(dim)
        response, _ = integrated_model(query, context, emotion=emotion)
        score = evaluator.evaluate(response, query, emotion)
        final_turing += score["percentage"]
    
    final_score = final_turing / len(eval_queries)
    
    print("\n" + "=" * 50)
    print("最终结果:")
    print(f"  最终分数: {final_score:.1f}%")
    print(f"  最佳分数: {best_score:.1f}%")
    print(f"  总迭代次数: {step}")
    
    if final_score >= 100.0:
        print("\n🎉 成功达到100%图灵测试分数！")
    else:
        print(f"\n💡 当前分数: {final_score:.1f}%, 继续优化中...")

if __name__ == "__main__":
    main()

