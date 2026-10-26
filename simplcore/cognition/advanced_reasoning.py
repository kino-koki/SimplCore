# -*- coding: utf-8 -*-
"""高级推理模块 (Advanced Reasoning).

核心推理能力:
  1. 类比推理 (Analogical Reasoning) - 从相似情境中学习
  2. 因果推理 (Causal Reasoning) - 理解因果关系
  3. 元认知 (Metacognition) - 思考自身的思考过程
  4. 演绎推理 (Deductive Reasoning) - 从一般到特殊
  5. 归纳推理 (Inductive Reasoning) - 从特殊到一般
  6. 溯因推理 (Abductive Reasoning) - 寻找最佳解释

与知识库的集成:
  - 在推理时自动检索相关知识
  - 将知识作为推理的基础
  - 验证推理结果是否与知识一致
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from SimplCore.knowledge.knowledge_base import KnowledgeBase


class AnalogicalReasoner:
    """类比推理器."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def find_analogies(self, query: str, source_domain: str, target_domain: str) -> List[Tuple[str, str, float]]:
        """寻找跨域类比."""
        source_knowledge = self.knowledge_base.get_knowledge(source_domain, top_k=5)
        target_knowledge = self.knowledge_base.get_knowledge(target_domain, top_k=5)
        
        analogies = []
        
        for source in source_knowledge:
            for target in target_knowledge:
                similarity = self._compute_similarity(source, target)
                if similarity > 0.3:
                    analogies.append((source, target, similarity))
        
        return sorted(analogies, key=lambda x: -x[2])
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度."""
        words1 = set(text1.lower().replace(' ', ''))
        words2 = set(text2.lower().replace(' ', ''))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def reason(self, query: str) -> Dict[str, Any]:
        """进行类比推理."""
        domains = ["physics", "mathematics", "biology", "computer_science", "chemistry"]
        
        all_analogies = []
        for i, dom1 in enumerate(domains):
            for dom2 in domains[i+1:]:
                analogies = self.find_analogies(query, dom1, dom2)
                all_analogies.extend(analogies)
        
        all_analogies.sort(key=lambda x: -x[2])
        
        return {
            "query": query,
            "analogies": all_analogies[:5],
            "reasoning": "基于类比推理，找到了以下相似概念:",
        }


class CausalReasoner:
    """因果推理器."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def find_causes(self, effect: str) -> List[Tuple[str, float]]:
        """寻找原因."""
        knowledge = self.knowledge_base.get_knowledge(effect, top_k=10)
        
        causes = []
        for k in knowledge:
            confidence = self._extract_causal_confidence(k, effect)
            if confidence > 0.5:
                causes.append((k, confidence))
        
        return sorted(causes, key=lambda x: -x[1])
    
    def find_effects(self, cause: str) -> List[Tuple[str, float]]:
        """寻找结果."""
        knowledge = self.knowledge_base.get_knowledge(cause, top_k=10)
        
        effects = []
        for k in knowledge:
            confidence = self._extract_effect_confidence(k, cause)
            if confidence > 0.5:
                effects.append((k, confidence))
        
        return sorted(effects, key=lambda x: -x[1])
    
    def _extract_causal_confidence(self, knowledge: str, effect: str) -> float:
        """提取因果置信度."""
        keywords = ["导致", "引起", "产生", "造成", "因为", "由于", "因此", "所以"]
        
        confidence = 0.0
        for kw in keywords:
            if kw in knowledge:
                confidence += 0.2
        
        if effect in knowledge:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _extract_effect_confidence(self, knowledge: str, cause: str) -> float:
        """提取效果置信度."""
        keywords = ["导致", "引起", "产生", "造成", "结果", "从而", "使得"]
        
        confidence = 0.0
        for kw in keywords:
            if kw in knowledge:
                confidence += 0.2
        
        if cause in knowledge:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def reason(self, query: str) -> Dict[str, Any]:
        """进行因果推理."""
        cause_keywords = ["为什么", "原因", "为什么会", "什么导致"]
        effect_keywords = ["会怎样", "结果", "后果", "会发生"]
        
        is_cause_query = any(kw in query for kw in cause_keywords)
        
        if is_cause_query:
            causes = self.find_causes(query)
            return {
                "query": query,
                "type": "cause",
                "causes": causes[:5],
                "reasoning": "基于因果推理，找到了以下可能的原因:",
            }
        else:
            effects = self.find_effects(query)
            return {
                "query": query,
                "type": "effect",
                "effects": effects[:5],
                "reasoning": "基于因果推理，找到了以下可能的结果:",
            }


class MetacognitiveReasoner:
    """元认知推理器."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.thought_history = []
    
    def monitor_thinking(self, thought: str, confidence: float) -> None:
        """监控思考过程."""
        self.thought_history.append({
            "thought": thought,
            "confidence": confidence,
            "timestamp": len(self.thought_history),
        })
    
    def evaluate_confidence(self, answer: str, context: str) -> float:
        """评估答案置信度."""
        knowledge = self.knowledge_base.get_knowledge(answer, top_k=5)
        
        score = 0.5
        for k in knowledge:
            if any(word in answer for word in k.split()[:5]):
                score += 0.1
        
        return min(score, 1.0)
    
    def self_correct(self, initial_answer: str, feedback: str) -> str:
        """自我修正."""
        corrected = initial_answer
        
        if "错误" in feedback or "不对" in feedback:
            knowledge = self.knowledge_base.get_knowledge(initial_answer, top_k=3)
            if knowledge:
                corrected = f"基于反馈修正: {knowledge[0]}"
        
        return corrected
    
    def reflect(self) -> Dict[str, Any]:
        """反思思考过程."""
        if not self.thought_history:
            return {"reflection": "暂无思考历史"}
        
        avg_confidence = sum(t["confidence"] for t in self.thought_history) / len(self.thought_history)
        
        return {
            "total_thoughts": len(self.thought_history),
            "average_confidence": avg_confidence,
            "recent_thoughts": self.thought_history[-5:],
            "reflection": self._generate_reflection(avg_confidence),
        }
    
    def _generate_reflection(self, avg_confidence: float) -> str:
        """生成反思文本."""
        if avg_confidence > 0.8:
            return "我的思考过程自信且准确，可以继续保持。"
        elif avg_confidence > 0.5:
            return "我的思考过程有一定信心，但某些部分需要更多验证。"
        else:
            return "我的思考过程信心不足，需要更多知识支持和验证。"
    
    def reason(self, query: str) -> Dict[str, Any]:
        """进行元认知推理."""
        self.monitor_thinking(f"处理查询: {query}", 0.7)
        
        knowledge = self.knowledge_base.get_knowledge(query, top_k=3)
        
        if knowledge:
            self.monitor_thinking(f"找到相关知识: {len(knowledge)} 条", 0.8)
            
            answer = f"基于知识库回答: {knowledge[0][:100]}..."
            confidence = self.evaluate_confidence(answer, query)
            
            self.monitor_thinking(f"生成答案，置信度: {confidence:.2f}", confidence)
            
            return {
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "knowledge_used": knowledge,
                "reflection": self.reflect(),
            }
        
        self.monitor_thinking("未找到相关知识", 0.3)
        
        return {
            "query": query,
            "answer": "暂无足够知识回答此问题",
            "confidence": 0.3,
            "knowledge_used": [],
            "reflection": self.reflect(),
        }


class DeductiveReasoner:
    """演绎推理器."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def deduce(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """从前提推导结论."""
        verified_premises = []
        for premise in premises:
            knowledge = self.knowledge_base.get_knowledge(premise, top_k=3)
            verified_premises.append((premise, len(knowledge) > 0))
        
        validity = self._check_validity(premises, conclusion)
        
        return {
            "premises": verified_premises,
            "conclusion": conclusion,
            "validity": validity,
            "reasoning": "基于演绎推理:",
        }
    
    def _check_validity(self, premises: List[str], conclusion: str) -> float:
        """检查推理有效性."""
        premise_words = set()
        for p in premises:
            premise_words.update(p.lower().split())
        
        conclusion_words = set(conclusion.lower().split())
        
        overlap = premise_words & conclusion_words
        
        if not overlap:
            return 0.0
        
        return len(overlap) / len(conclusion_words)
    
    def reason(self, query: str) -> Dict[str, Any]:
        """进行演绎推理."""
        if "因为" in query:
            parts = query.split("因为")
            conclusion = parts[0].strip()
            premise = parts[1].strip()
            
            result = self.deduce([premise], conclusion)
            return {
                "query": query,
                **result,
            }
        
        knowledge = self.knowledge_base.get_knowledge(query, top_k=3)
        
        return {
            "query": query,
            "premises": knowledge,
            "conclusion": query,
            "validity": 0.7 if knowledge else 0.3,
            "reasoning": "基于知识库进行演绎推理.",
        }


class InductiveReasoner:
    """归纳推理器."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def generalize(self, examples: List[str]) -> Dict[str, Any]:
        """从例子中归纳一般规律."""
        patterns = self._find_common_patterns(examples)
        
        return {
            "examples": examples,
            "generalization": patterns[:3],
            "reasoning": "基于归纳推理，从例子中发现以下规律:",
        }
    
    def _find_common_patterns(self, examples: List[str]) -> List[str]:
        """寻找共同模式."""
        patterns = []
        
        if len(examples) < 2:
            return patterns
        
        all_words = []
        for ex in examples:
            words = ex.lower().split()
            all_words.extend(words)
        
        word_counts = {}
        for word in all_words:
            if len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        common_words = sorted(word_counts.items(), key=lambda x: -x[1])
        patterns = [f"共同概念: {word}" for word, count in common_words[:3] if count >= len(examples) // 2]
        
        return patterns
    
    def reason(self, query: str) -> Dict[str, Any]:
        """进行归纳推理."""
        knowledge = self.knowledge_base.get_knowledge(query, top_k=5)
        
        if knowledge:
            result = self.generalize(knowledge)
            return {
                "query": query,
                **result,
            }
        
        return {
            "query": query,
            "examples": [],
            "generalization": [],
            "reasoning": "暂无足够例子进行归纳.",
        }


class AbductiveReasoner:
    """溯因推理器."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
    
    def find_best_explanation(self, observation: str) -> Dict[str, Any]:
        """寻找最佳解释."""
        knowledge = self.knowledge_base.get_knowledge(observation, top_k=10)
        
        explanations = []
        for k in knowledge:
            score = self._evaluate_explanation(k, observation)
            if score > 0.3:
                explanations.append((k, score))
        
        explanations.sort(key=lambda x: -x[1])
        
        return {
            "observation": observation,
            "explanations": explanations[:3],
            "best_explanation": explanations[0] if explanations else None,
            "reasoning": "基于溯因推理，找到了以下可能的解释:",
        }
    
    def _evaluate_explanation(self, knowledge: str, observation: str) -> float:
        """评估解释的可能性."""
        score = 0.0
        
        obs_words = set(observation.lower().split())
        know_words = set(knowledge.lower().split())
        
        overlap = obs_words & know_words
        if overlap:
            score += 0.5
        
        causal_keywords = ["因为", "导致", "由于", "所以"]
        for kw in causal_keywords:
            if kw in knowledge:
                score += 0.2
        
        return min(score, 1.0)
    
    def reason(self, query: str) -> Dict[str, Any]:
        """进行溯因推理."""
        result = self.find_best_explanation(query)
        result["query"] = query
        return result


class IntegratedReasoner:
    """集成推理器 - 综合所有推理能力."""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.analogical = AnalogicalReasoner(knowledge_base)
        self.causal = CausalReasoner(knowledge_base)
        self.metacognitive = MetacognitiveReasoner(knowledge_base)
        self.deductive = DeductiveReasoner(knowledge_base)
        self.inductive = InductiveReasoner(knowledge_base)
        self.abductive = AbductiveReasoner(knowledge_base)
        
        self.reasoner_map = {
            "analogical": self.analogical,
            "causal": self.causal,
            "deductive": self.deductive,
            "inductive": self.inductive,
            "abductive": self.abductive,
        }
    
    def analyze_query(self, query: str) -> str:
        """分析查询类型，选择合适的推理器."""
        if "类比" in query or "类似" in query or "好比" in query:
            return "analogical"
        if "为什么" in query or "原因" in query or "导致" in query:
            return "causal"
        if "因为" in query or "所以" in query or "因此" in query:
            return "deductive"
        if "总结" in query or "规律" in query or "一般" in query:
            return "inductive"
        if "解释" in query or "为什么" in query or "原因" in query:
            return "abductive"
        
        return "causal"
    
    def reason(self, query: str, reasoner_type: Optional[str] = None) -> Dict[str, Any]:
        """进行综合推理."""
        if reasoner_type is None:
            reasoner_type = self.analyze_query(query)
        
        reasoner = self.reasoner_map.get(reasoner_type, self.causal)
        
        result = reasoner.reason(query)
        
        self.metacognitive.monitor_thinking(f"使用 {reasoner_type} 推理器处理: {query}", 0.7)
        
        result["reasoner_type"] = reasoner_type
        
        if "answer" in result:
            confidence = self.metacognitive.evaluate_confidence(result["answer"], query)
            result["confidence"] = confidence
        
        return result
    
    def multi_reasoning(self, query: str) -> Dict[str, Any]:
        """使用多种推理器进行推理."""
        results = {}
        
        for reasoner_name, reasoner in self.reasoner_map.items():
            try:
                result = reasoner.reason(query)
                results[reasoner_name] = result
            except:
                continue
        
        best_result = None
        best_confidence = 0.0
        
        for name, result in results.items():
            confidence = result.get("confidence", 0.5)
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = {**result, "reasoner_type": name}
        
        return {
            "query": query,
            "multi_results": results,
            "best_result": best_result,
            "metacognition": self.metacognitive.reflect(),
        }


if __name__ == "__main__":
    print("=" * 70)
    print("高级推理模块测试")
    print("=" * 70)
    
    kb = KnowledgeBase()
    
    print("\n[Test 1] 类比推理")
    analogical = AnalogicalReasoner(kb)
    result = analogical.reason("量子纠缠和什么类似?")
    print(f"  查询: {result['query']}")
    print(f"  找到类比: {len(result['analogies'])} 个")
    
    print("\n[Test 2] 因果推理")
    causal = CausalReasoner(kb)
    result = causal.reason("为什么会发生光电效应?")
    print(f"  查询: {result['query']}")
    print(f"  类型: {result['type']}")
    
    print("\n[Test 3] 元认知推理")
    metacog = MetacognitiveReasoner(kb)
    result = metacog.reason("什么是深度学习?")
    print(f"  查询: {result['query']}")
    print(f"  答案: {result['answer'][:80]}")
    print(f"  置信度: {result['confidence']:.2f}")
    
    print("\n[Test 4] 演绎推理")
    deductive = DeductiveReasoner(kb)
    result = deductive.reason("因为牛顿第二定律F=ma")
    print(f"  查询: {result['query']}")
    print(f"  有效性: {result['validity']:.2f}")
    
    print("\n[Test 5] 归纳推理")
    inductive = InductiveReasoner(kb)
    result = inductive.reason("物理学定律")
    print(f"  查询: {result['query']}")
    
    print("\n[Test 6] 溯因推理")
    abductive = AbductiveReasoner(kb)
    result = abductive.reason("为什么苹果会落地?")
    print(f"  查询: {result['query']}")
    
    print("\n[Test 7] 集成推理")
    integrated = IntegratedReasoner(kb)
    result = integrated.reason("什么是量子纠缠?")
    print(f"  查询: {result['query']}")
    print(f"  推理器: {result['reasoner_type']}")
    
    print("\n[Test 8] 多推理器推理")
    result = integrated.multi_reasoning("为什么深度学习效果好?")
    print(f"  查询: {result['query']}")
    print(f"  使用推理器数量: {len(result['multi_results'])}")
    
    print("\n" + "=" * 70)
    print("高级推理模块测试完成!")
    print("=" * 70)

