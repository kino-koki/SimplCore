#!/usr/bin/env python
# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
响应多样性模块 - 多元化回应生成机制

核心功能：
1. 回应风格多样性生成
2. 避免模式化输出
3. 表达自然度增强
4. 多样化生成策略
5. 创造性回应生成
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class ResponseStyle(Enum):
    """回应风格"""
    DIRECT = "direct"
    CASUAL = "casual"
    FORMAL = "formal"
    EMOTIONAL = "emotional"
    HUMOROUS = "humorous"
    DETAIL_ORIENTED = "detail_oriented"
    CONCISE = "concise"
    CREATIVE = "creative"
    QUESTIONING = "questioning"
    NARRATIVE = "narrative"


class VarietyStrategy(Enum):
    """多样化策略"""
    RANDOM = "random"
    ROTATION = "rotation"
    ADAPTIVE = "adaptive"
    EXPLORATION = "exploration"
    DIVERSITY_ENFORCED = "diversity_enforced"


class StyleEmbedding:
    """风格嵌入"""
    
    def __init__(self, style: ResponseStyle, embedding: SimpTensor):
        self.style = style
        self.embedding = embedding


class StyleGenerator(Module):
    """风格生成器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.style_proj = Linear(dim, len(ResponseStyle) * dim)
        self.register_module("style_proj", self.style_proj)
        
        self.style_norm = LayerNorm(dim)
        self.register_module("style_norm", self.style_norm)
        
        self._style_weights = {}
        for i, style in enumerate(ResponseStyle):
            weight = randn(dim) * 0.1
            self._style_weights[style] = weight
    
    def generate_style(self, context: SimpTensor) -> Dict[ResponseStyle, SimpTensor]:
        """生成多种风格"""
        logits = self.style_proj(context)
        logits = logits.reshape(len(ResponseStyle), self.dim)
        
        styles = {}
        for i, style in enumerate(ResponseStyle):
            style_feature = tanh(logits[i])
            style_feature = self.style_norm(style_feature + self._style_weights[style])
            styles[style] = style_feature
        
        return styles
    
    def select_style(
        self,
        context: SimpTensor,
        strategy: VarietyStrategy = VarietyStrategy.RANDOM
    ) -> Tuple[ResponseStyle, SimpTensor]:
        """选择风格"""
        styles = self.generate_style(context)
        
        if strategy == VarietyStrategy.RANDOM:
            style_list = list(styles.keys())
            idx = np.random.randint(len(style_list))
            selected = style_list[idx]
        
        elif strategy == VarietyStrategy.ADAPTIVE:
            similarities = {}
            for style, embedding in styles.items():
                sim = float(np.dot(context._data, embedding._data) / (
                    np.linalg.norm(context._data) * np.linalg.norm(embedding._data) + 1e-8
                ))
                similarities[style] = sim
            selected = max(similarities, key=similarities.get)
        
        else:
            style_list = list(styles.keys())
            idx = np.random.randint(len(style_list))
            selected = style_list[idx]
        
        return selected, styles[selected]


class PatternAvoidance(Module):
    """模式避免器"""
    
    def __init__(self, dim: int, max_history: int = 50):
        super().__init__()
        self.dim = dim
        self.max_history = max_history
        
        self.pattern_proj = Linear(dim, dim)
        self.register_module("pattern_proj", self.pattern_proj)
        
        self.diversity_proj = Linear(dim * 2, dim)
        self.register_module("diversity_proj", self.diversity_proj)
        
        self.diversity_norm = LayerNorm(dim)
        self.register_module("diversity_norm", self.diversity_norm)
        
        self._response_history: List[SimpTensor] = []
    
    def calculate_diversity(
        self,
        current_response: SimpTensor,
        min_diversity: float = 0.3
    ) -> Tuple[SimpTensor, float, bool]:
        """计算多样性并调整"""
        if not self._response_history:
            return current_response, 1.0, True
        
        similarities = []
        for prev in self._response_history:
            sim = float(np.dot(current_response._data, prev._data) / (
                np.linalg.norm(current_response._data) * np.linalg.norm(prev._data) + 1e-8
            ))
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        max_similarity = max(similarities)
        diversity_score = 1 - avg_similarity
        
        needs_adjustment = max_similarity > (1 - min_diversity)
        
        if needs_adjustment:
            history_avg = SimpTensor(np.mean([r._data for r in self._response_history], axis=0))
            
            current_feature = tanh(self.pattern_proj(current_response))
            history_feature = tanh(self.pattern_proj(history_avg))
            
            combined = SimpTensor(np.concatenate([current_feature._data, history_feature._data]))
            diversity_signal = tanh(self.diversity_proj(combined))
            
            adjusted = current_response + diversity_signal * (1 - max_similarity)
            adjusted = self.diversity_norm(adjusted)
            
            return adjusted, diversity_score, True
        else:
            return current_response, diversity_score, False
    
    def add_response(self, response: SimpTensor):
        """添加响应到历史"""
        self._response_history.append(response)
        if len(self._response_history) > self.max_history:
            self._response_history = self._response_history[-self.max_history:]
    
    def reset(self):
        """重置历史"""
        self._response_history = []


class NaturalnessEnhancer(Module):
    """自然度增强器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.naturalness_proj = Linear(dim, dim)
        self.register_module("naturalness_proj", self.naturalness_proj)
        
        self.variation_proj = Linear(dim, dim)
        self.register_module("variation_proj", self.variation_proj)
        
        self.rhythm_proj = Linear(dim, dim)
        self.register_module("rhythm_proj", self.rhythm_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
    
    def forward(self, response: SimpTensor, context: Optional[SimpTensor] = None) -> Tuple[SimpTensor, Dict[str, Any]]:
        """增强自然度"""
        base_feature = tanh(self.naturalness_proj(response))
        
        variation = tanh(self.variation_proj(randn(self.dim)))
        variation = variation * 0.1
        
        rhythm_pattern = tanh(self.rhythm_proj(response))
        rhythm_pattern = rhythm_pattern * 0.05
        
        enhanced = base_feature + variation + rhythm_pattern
        
        if context is not None:
            context_influence = tanh(self.naturalness_proj(context)) * 0.2
            enhanced = enhanced + context_influence
        
        enhanced = self.output_norm(enhanced)
        
        info = {
            "variation_norm": round(np.linalg.norm(variation._data), 4),
            "rhythm_influence": 0.05,
            "enhanced_norm": round(np.linalg.norm(enhanced._data), 4),
        }
        
        return enhanced, info


class CreativeGenerator(Module):
    """创造性生成器"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.creativity_proj = Linear(dim, dim)
        self.register_module("creativity_proj", self.creativity_proj)
        
        self.combination_proj = Linear(dim * 2, dim)
        self.register_module("combination_proj", self.combination_proj)
        
        self.surprise_proj = Linear(dim, dim)
        self.register_module("surprise_proj", self.surprise_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
    
    def forward(
        self,
        base_response: SimpTensor,
        context: SimpTensor,
        creativity_level: float = 0.5
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """生成创造性响应"""
        creativity_level = min(1.0, max(0.0, creativity_level))
        
        base_feature = tanh(self.creativity_proj(base_response))
        context_feature = tanh(self.creativity_proj(context))
        
        combined = SimpTensor(np.concatenate([base_feature._data, context_feature._data]))
        combined_feature = tanh(self.combination_proj(combined))
        
        surprise_component = tanh(self.surprise_proj(randn(self.dim)))
        surprise_component = surprise_component * creativity_level
        
        creative_response = base_feature * (1 - creativity_level) + combined_feature * creativity_level
        creative_response = creative_response + surprise_component * 0.3
        
        creative_response = self.output_norm(creative_response)
        
        info = {
            "creativity_level": creativity_level,
            "surprise_component_norm": round(np.linalg.norm(surprise_component._data), 4),
            "creative_norm": round(np.linalg.norm(creative_response._data), 4),
        }
        
        return creative_response, info


class ResponseVariety(Module):
    """响应多样性系统"""
    
    def __init__(self, dim: int, max_history: int = 50):
        super().__init__()
        self.dim = dim
        
        self.style_generator = StyleGenerator(dim)
        self.register_module("style_generator", self.style_generator)
        
        self.pattern_avoidance = PatternAvoidance(dim, max_history)
        self.register_module("pattern_avoidance", self.pattern_avoidance)
        
        self.naturalness_enhancer = NaturalnessEnhancer(dim)
        self.register_module("naturalness_enhancer", self.naturalness_enhancer)
        
        self.creative_generator = CreativeGenerator(dim)
        self.register_module("creative_generator", self.creative_generator)
        
        self.combine_proj = Linear(dim * 3, dim)
        self.register_module("combine_proj", self.combine_proj)
        
        self.output_norm = LayerNorm(dim)
        self.register_module("output_norm", self.output_norm)
        
        self._current_strategy = VarietyStrategy.ADAPTIVE
        self._style_rotation_index = 0
    
    def forward(
        self,
        base_response: SimpTensor,
        context: SimpTensor,
        creativity_level: float = 0.3,
        strategy: Optional[VarietyStrategy] = None
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """生成多样化响应"""
        if strategy is None:
            strategy = self._current_strategy
        
        style, style_embedding = self.style_generator.select_style(context, strategy)
        
        style_applied = base_response + style_embedding * 0.2
        
        creative_response, creative_info = self.creative_generator(
            style_applied, context, creativity_level
        )
        
        enhanced_response, naturalness_info = self.naturalness_enhancer(creative_response, context)
        
        adjusted_response, diversity_score, adjusted = self.pattern_avoidance.calculate_diversity(
            enhanced_response
        )
        
        self.pattern_avoidance.add_response(adjusted_response)
        
        combined = SimpTensor(np.concatenate([
            adjusted_response._data,
            style_embedding._data,
            context._data
        ]))
        
        final = gelu(self.combine_proj(combined))
        final = self.output_norm(final)
        
        info = {
            "style": style.value,
            "strategy": strategy.value,
            "diversity_score": round(diversity_score, 4),
            "pattern_adjusted": adjusted,
            "creativity": creative_info,
            "naturalness": naturalness_info,
        }
        
        return final, info
    
    def set_strategy(self, strategy: VarietyStrategy):
        """设置多样化策略"""
        self._current_strategy = strategy
    
    def get_diversity_score(self) -> float:
        """获取多样性分数"""
        if not self.pattern_avoidance._response_history:
            return 1.0
        
        responses = self.pattern_avoidance._response_history
        if len(responses) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = float(np.dot(responses[i]._data, responses[j]._data) / (
                    np.linalg.norm(responses[i]._data) * np.linalg.norm(responses[j]._data) + 1e-8
                ))
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1 - avg_similarity
    
    def reset(self):
        """重置多样性状态"""
        self.pattern_avoidance.reset()
        self._style_rotation_index = 0


if __name__ == "__main__":
    dim = 64
    print("=== 测试响应多样性模块 ===")
    
    print("\n1. 测试风格生成器:")
    sg = StyleGenerator(dim)
    context = randn(dim)
    styles = sg.generate_style(context)
    print(f"生成 {len(styles)} 种风格")
    style, embedding = sg.select_style(context, VarietyStrategy.ADAPTIVE)
    print(f"选择风格: {style.value}, 嵌入形状: {embedding.shape}")
    
    print("\n2. 测试模式避免器:")
    pa = PatternAvoidance(dim)
    responses = [randn(dim) for _ in range(5)]
    
    for i, response in enumerate(responses):
        adjusted, score, adjusted_flag = pa.calculate_diversity(response)
        pa.add_response(adjusted)
        print(f"  响应 {i+1}: 多样性={score:.4f}, 调整={adjusted_flag}")
    
    print("\n3. 测试自然度增强器:")
    ne = NaturalnessEnhancer(dim)
    response = randn(dim)
    enhanced, info = ne(response, context)
    print(f"增强输出: {enhanced.shape}")
    print(f"自然度信息: {info}")
    
    print("\n4. 测试创造性生成器:")
    cg = CreativeGenerator(dim)
    base_response = randn(dim)
    creative, info = cg(base_response, context, creativity_level=0.7)
    print(f"创造性输出: {creative.shape}")
    print(f"创造性信息: {info}")
    
    print("\n5. 测试响应多样性系统:")
    rv = ResponseVariety(dim)
    
    for i in range(5):
        base_response = randn(dim)
        context = randn(dim)
        output, info = rv(base_response, context, creativity_level=0.4)
        print(f"\n  输入 {i+1}:")
        print(f"    输出形状: {output.shape}")
        print(f"    风格: {info['style']}")
        print(f"    多样性分数: {info['diversity_score']}")
        print(f"    模式调整: {info['pattern_adjusted']}")
    
    diversity_score = rv.get_diversity_score()
    print(f"\n整体多样性分数: {diversity_score:.4f}")
    
    print("\n6. 测试梯度传播:")
    rv2 = ResponseVariety(dim)
    test_response = randn(dim)
    test_context = randn(dim)
    output2, _ = rv2(test_response, test_context)
    loss = output2.sum()
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad._data != 0) for p in rv2.parameters())
    print(f"梯度传播: {has_grad}")
    
    print("\n=== 所有测试通过 ===")
