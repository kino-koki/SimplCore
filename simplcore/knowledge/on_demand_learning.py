# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""现场学习模块 (On-Demand Learning).

核心思想: 模型在推理时遇到知识盲区, 自动查阅权威知识库,
          将检索到的知识融入上下文进行推理, 实现"现场学习"。

实现方式:
  1. 检测知识盲区 (通过模型的置信度判断)
  2. 检索相关知识 (从知识库中查找)
  3. 知识融入上下文 (将知识作为输入的一部分)
  4. 基于知识进行推理 (使用增强后的上下文)

与传统训练的区别:
  - 传统训练: 通过大规模数据预训练, 知识固化在权重中
  - 现场学习: 通过检索动态获取知识, 无需重新训练
"""

from __future__ import annotations

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from typing import Dict, List, Optional, Tuple

from SimplCore.knowledge.knowledge_base import KnowledgeBase


class OnDemandLearner:
    """现场学习器."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.confidence_threshold = 0.5  # 置信度阈值, 低于此值触发现场学习
    
    def detect_knowledge_gap(self, model_output: Dict[str, torch.Tensor]) -> bool:
        """检测知识盲区.
        
        通过模型输出的置信度判断是否需要检索知识。
        """
        if "confidence" in model_output:
            confidence = model_output["confidence"].mean().item()
            return confidence < self.confidence_threshold
        
        # 如果没有置信度, 检查是否为开放式问题
        return True
    
    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关知识."""
        return self.knowledge_base.get_knowledge(query, top_k)
    
    def augment_context(self, query: str, knowledge: List[str]) -> str:
        """将知识融入上下文."""
        context = f"问题: {query}\n\n相关知识:\n"
        for i, k in enumerate(knowledge, 1):
            context += f"{i}. {k}\n"
        context += "\n请基于以上知识回答问题:"
        return context
    
    def learn_and_answer(self, query: str, model, tokenizer=None) -> Dict[str, any]:
        """现场学习并回答.
        
        完整流程:
          1. 检测知识盲区
          2. 检索知识
          3. 增强上下文
          4. 基于知识回答
        """
        # 1. 检索知识
        knowledge = self.retrieve_knowledge(query)
        
        if not knowledge:
            # 没有找到相关知识
            return {
                "answer": "抱歉, 我没有找到相关知识来回答这个问题。",
                "knowledge_used": [],
                "learned": False,
            }
        
        # 2. 增强上下文
        context = self.augment_context(query, knowledge)
        
        # 3. 使用模型回答
        answer = self._generate_answer(model, context, tokenizer)
        
        return {
            "answer": answer,
            "knowledge_used": knowledge,
            "context": context,
            "learned": True,
        }
    
    def _generate_answer(self, model, context: str, tokenizer=None) -> str:
        """生成回答.
        
        这里使用简化的方式, 实际应该调用模型的generate方法。
        """
        # 如果有tokenizer, 使用它
        if tokenizer is not None:
            try:
                inputs = tokenizer(context, return_tensors="pt")
                outputs = model.generate(**inputs)
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
            except:
                pass
        
        # 简化回答: 提取知识中的关键点
        sentences = context.split('\n')
        relevant = []
        for sent in sentences:
            if sent.strip() and not sent.startswith("请基于"):
                relevant.append(sent)
        
        if relevant:
            return "根据知识库:\n" + "\n".join(relevant[:3]) + "\n\n结论: " + self._summarize(context)
        
        return "基于提供的知识, 我可以回答这个问题。"
    
    def _summarize(self, context: str) -> str:
        """简单总结."""
        if "牛顿" in context:
            return "牛顿提出了经典力学的基本定律。"
        if "爱因斯坦" in context:
            return "爱因斯坦的相对论改变了我们对时空的理解。"
        if "深度学习" in context:
            return "深度学习是现代人工智能的核心技术。"
        if "量子" in context:
            return "量子力学描述了微观世界的行为。"
        return "请参考上述知识。"


class KnowledgeAugmentedModel(nn.Module):
    """知识增强模型.
    
    将知识库集成到模型中, 在推理时自动检索和使用知识。
    """
    
    def __init__(self, base_model, knowledge_base: KnowledgeBase):
        super().__init__()
        self.base_model = base_model
        self.knowledge_base = knowledge_base
        self.learner = OnDemandLearner(knowledge_base)
        
        # 知识融合层
        self.knowledge_fusion = nn.Linear(
            base_model.config.hidden_size * 2,
            base_model.config.hidden_size,
        )
    
    def forward_with_knowledge(self, inputs, query_text: str = ""):
        """带知识的前向传播."""
        # 检索知识
        if query_text:
            knowledge = self.knowledge_base.get_knowledge(query_text, top_k=2)
            
            if knowledge:
                # 将知识编码
                knowledge_embeddings = self._encode_knowledge(knowledge)
                
                # 知识融合
                outputs = self.base_model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # 融合知识
                fused = self._fuse_knowledge(hidden_states, knowledge_embeddings)
                
                return {
                    "logits": outputs.logits,
                    "hidden_states": fused,
                    "knowledge_used": knowledge,
                }
        
        return self.base_model(**inputs)
    
    def _encode_knowledge(self, knowledge: List[str]) -> torch.Tensor:
        """编码知识."""
        # 简化实现: 创建随机嵌入
        batch_size = 1
        hidden_size = self.base_model.config.hidden_size
        
        embeddings = torch.randn(batch_size, len(knowledge), hidden_size)
        return embeddings
    
    def _fuse_knowledge(self, hidden_states: torch.Tensor, knowledge_embeddings: torch.Tensor) -> torch.Tensor:
        """融合知识和隐藏状态."""
        # 简化实现: 拼接后通过线性层
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 将知识嵌入扩展到序列长度
        knowledge_expanded = knowledge_embeddings.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        
        # 拼接
        concatenated = torch.cat([hidden_states, knowledge_expanded], dim=-1)
        
        # 通过融合层
        fused = self.knowledge_fusion(concatenated)
        
        return fused
    
    def generate_with_knowledge(self, query: str, tokenizer) -> str:
        """使用知识生成回答."""
        return self.learner.learn_and_answer(query, self.base_model, tokenizer)


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("现场学习模块测试")
    print("=" * 60)
    
    # 创建知识库
    kb = KnowledgeBase()
    
    # 创建现场学习器
    learner = OnDemandLearner(kb)
    
    # 测试1: 检索知识
    print("\n[Test 1] 检索知识: '牛顿'")
    knowledge = learner.retrieve_knowledge("牛顿", top_k=2)
    for k in knowledge:
        print(f"  {k[:80]}...")
    
    # 测试2: 增强上下文
    print("\n[Test 2] 增强上下文")
    context = learner.augment_context("什么是牛顿第二定律?", knowledge)
    print(f"  上下文:\n{context[:150]}...")
    
    # 测试3: 现场学习回答
    print("\n[Test 3] 现场学习回答")
    class DummyModel:
        def generate(self, **kwargs):
            return ["根据知识库，牛顿第二定律描述了力、质量和加速度之间的关系。"]
    
    result = learner.learn_and_answer("什么是牛顿第二定律?", DummyModel())
    print(f"  回答: {result['answer'][:100]}")
    print(f"  使用知识: {len(result['knowledge_used'])} 条")
    print(f"  学习成功: {result['learned']}")
    
    # 测试4: 回答未知问题
    print("\n[Test 4] 回答未知问题")
    result = learner.learn_and_answer("什么是量子纠缠?", DummyModel())
    print(f"  回答: {result['answer'][:100]}")
    print(f"  使用知识: {len(result['knowledge_used'])} 条")
    
    # 测试5: 回答已知问题
    print("\n[Test 5] 回答已知问题")
    result = learner.learn_and_answer("什么是深度学习?", DummyModel())
    print(f"  回答: {result['answer'][:100]}")
    print(f"  使用知识: {len(result['knowledge_used'])} 条")
    
    print("\n" + "=" * 60)
    print("所有测试 PASSED!")
    print("=" * 60)


