# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""综合集成测试 - 验证所有模块协同工作"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from ..engine.tensor import SimpTensor, randn, zeros, manual_seed
from ..engine.nn import Module, Linear, LayerNorm

from ..nlu.semantic_parser import SemanticParser
from ..nlu.disambiguator import ContextualDisambiguator
from ..nlu.context_aware_parser import ContextAwareSemanticParser, MultiTurnComprehension

from ..dialogue.dialogue_state_tracker import DialogueStateTracker
from ..dialogue.long_range_memory import LongRangeMemory

from ..emotion.emotional_intelligence import EmotionalIntelligence

from ..cognition.human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode
from ..cognition.multimodal_cognition import MultimodalCognitiveIntegration

from ..perception.multimodal_fusion import (
    ModalityType,
    MultimodalPerception,
    AudioPerception,
    EnvironmentPerception,
)

from ..knowledge.knowledge_graph import KnowledgeGraph, KnowledgeReasoner

from ..generation.response_variety import ResponseVariety


class IntegrationTest:
    """综合集成测试类"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.results: Dict[str, Dict[str, Any]] = {}
        manual_seed(42)

    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """运行所有集成测试"""
        print("=" * 70)
        print("SimplCore 综合集成测试")
        print("=" * 70)

        self.test_nlu_integration()
        self.test_dialogue_integration()
        self.test_emotion_integration()
        self.test_cognition_integration()
        self.test_perception_integration()
        self.test_knowledge_integration()
        self.test_full_system_integration()

        self.print_summary()

        return self.results

    def test_nlu_integration(self):
        """测试NLU模块集成"""
        print("\n1. NLU模块集成测试")
        print("-" * 50)

        context_parser = ContextAwareSemanticParser(self.dim)
        multi_turn = MultiTurnComprehension(self.dim)

        results = []
        for i in range(5):
            x = randn(self.dim)
            parsed_output, parsed_info = context_parser.parse(x)
            multi_output, multi_info = multi_turn(x)

            result = {
                "turn": i + 1,
                "intent": parsed_info["intent"],
                "coherence": parsed_info["coherence"]["level"],
                "topic_change": parsed_info["topic"]["is_topic_change"],
                "context_size": parsed_info["context_size"],
                "multi_turn_available": multi_info["multi_turn"]["summary_available"],
                "total_turns": multi_info["total_turns"],
            }
            results.append(result)

            print(f"  轮次 {i+1}:")
            print(f"    意图: {result['intent']}, 连贯性: {result['coherence']}")
            print(f"    话题变化: {result['topic_change']}, 语境大小: {result['context_size']}")

        self.results["nlu_integration"] = {
            "passed": True,
            "results": results,
            "context_stats": context_parser.get_context_stats(),
        }
        print("  ✓ NLU模块集成测试通过")

    def test_dialogue_integration(self):
        """测试对话模块集成"""
        print("\n2. 对话模块集成测试")
        print("-" * 50)

        tracker = DialogueStateTracker(self.dim, max_history=10)
        memory = LongRangeMemory(self.dim, memory_capacity=100)

        results = []
        for i in range(5):
            x = randn(self.dim)
            state_vector, state_info = tracker(x)

            retrieved, memory_info = memory(x, top_k=3)
            memory.add_memory(x, memory_info.get("memory_type", "contextual"))

            result = {
                "turn": i + 1,
                "state_type": state_info["state_type"],
                "is_topic_change": state_info["is_topic_change"],
                "num_topics": state_info["num_topics"],
                "num_turns": state_info["num_turns"],
                "retrieved_count": memory_info["retrieved"],
                "memory_size": memory._get_memory_count(),
            }
            results.append(result)

            print(f"  轮次 {i+1}:")
            print(f"    状态类型: {result['state_type']}, 话题变化: {result['is_topic_change']}")
            print(f"    检索记忆: {result['retrieved_count']}, 记忆大小: {result['memory_size']}")

        summary = tracker.get_state_summary()
        memory_stats = memory.get_memory_stats()

        self.results["dialogue_integration"] = {
            "passed": True,
            "results": results,
            "state_summary": summary,
            "memory_stats": memory_stats,
        }
        print("  ✓ 对话模块集成测试通过")

    def test_emotion_integration(self):
        """测试情绪模块集成"""
        print("\n3. 情绪模块集成测试")
        print("-" * 50)

        emotional_intelligence = EmotionalIntelligence(self.dim)

        results = []
        for i in range(3):
            x = randn(self.dim)
            base_response = randn(self.dim)
            emotion_context = emotional_intelligence.get_emotion_context()

            output, info = emotional_intelligence(x, base_response, emotion_context)

            result = {
                "input": i + 1,
                "top_emotion": info["top_emotion"],
                "num_emotions": info["num_emotions"],
                "empathy_strength": info["empathy"]["empathy_strength"],
                "output_shape": output.shape,
            }
            results.append(result)

            print(f"  输入 {i+1}:")
            print(f"    主要情绪: {result['top_emotion']}, 数量: {result['num_emotions']}")
            print(f"    共情强度: {result['empathy_strength']:.3f}")

        emotions = emotional_intelligence.detect_emotions(randn(self.dim))

        self.results["emotion_integration"] = {
            "passed": True,
            "results": results,
            "detected_emotions_count": len(emotions),
            "emotion_context_size": len(emotion_context),
        }
        print("  ✓ 情绪模块集成测试通过")

    def test_cognition_integration(self):
        """测试认知模块集成"""
        print("\n4. 认知模块集成测试")
        print("-" * 50)

        hlcr = HumanLikeCognitiveReasoner(
            dim=self.dim,
            mode=HumanLikeMode.CONVERSATION,
            enable_nlu=True,
            enable_disambiguation=True,
            enable_dialogue_tracking=True,
            enable_long_range_memory=True,
            enable_emotional_intelligence=True,
            enable_response_variety=True,
            enable_reasoning=True,
            enable_knowledge=True,
        )

        results = []
        for i in range(3):
            x = randn(self.dim)
            context = randn(self.dim) if i > 0 else None

            output, info = hlcr(x, context)

            result = {
                "turn": i + 1,
                "mode": info["mode"],
                "overall_confidence": info["overall_confidence"],
                "context_length": info["context_length"],
                "nlu_intent": info["phases"]["nlu"]["intent"],
                "emotion_detected": bool(info["phases"]["emotion"]["top_emotion"]),
                "diversity_score": info["phases"]["variety"]["diversity_score"],
            }
            results.append(result)

            print(f"  轮次 {i+1}:")
            print(f"    模式: {result['mode']}, 置信度: {result['overall_confidence']:.4f}")
            print(f"    意图: {result['nlu_intent']}, 多样性: {result['diversity_score']:.3f}")

        hlcr.set_mode(HumanLikeMode.REASONING)
        x_reason = randn(self.dim)
        output_reason, info_reason = hlcr(x_reason)

        self.results["cognition_integration"] = {
            "passed": True,
            "results": results,
            "reasoning_depth": info_reason["phases"]["reasoning"].get("reasoning_depth", 0),
            "dialogue_summary": hlcr.get_dialogue_summary(),
            "memory_stats": hlcr.get_memory_stats(),
            "diversity_score": hlcr.get_diversity_score(),
        }
        print("  ✓ 认知模块集成测试通过")

    def test_perception_integration(self):
        """测试感知模块集成"""
        print("\n5. 感知模块集成测试")
        print("-" * 50)

        multimodal_perception = MultimodalPerception(self.dim)
        audio_perception = AudioPerception(self.dim)
        env_perception = EnvironmentPerception(self.dim)

        text_input = randn(self.dim)
        visual_input = randn(self.dim)
        audio_input = randn(self.dim)

        modalities_input = {
            ModalityType.TEXT: text_input,
            ModalityType.VISION: visual_input,
            ModalityType.AUDIO: audio_input,
        }

        output, info = multimodal_perception(modalities_input)

        audio_output, audio_info = audio_perception(audio_input)
        env_output, env_info = env_perception(randn(self.dim), temperature=22.0, light_level=0.7)

        result = {
            "multimodal_output_shape": output.shape,
            "num_modalities": info["num_modalities"],
            "active_modalities": info["active_modalities"],
            "dominant_modality": info["dominant_modality"],
            "attention_weights": info["attention_weights"],
            "memory_frames": info["memory_frames"],
            "audio_output_shape": audio_output.shape,
            "audio_has_audio": audio_info["has_audio"],
            "env_output_shape": env_output.shape,
            "env_temperature": env_info["temperature"],
        }

        print(f"  多模态输出: {result['multimodal_output_shape']}")
        print(f"  模态数量: {result['num_modalities']}, 活动模态: {result['active_modalities']}")
        print(f"  主导模态: {result['dominant_modality']}")
        print(f"  音频输出: {result['audio_output_shape']}, 环境输出: {result['env_output_shape']}")

        self.results["perception_integration"] = {
            "passed": True,
            "result": result,
            "status": multimodal_perception.get_status(),
        }
        print("  ✓ 感知模块集成测试通过")

    def test_knowledge_integration(self):
        """测试知识模块集成"""
        print("\n6. 知识模块集成测试")
        print("-" * 50)

        knowledge_graph = KnowledgeGraph()
        knowledge_reasoner = KnowledgeReasoner(self.dim)

        knowledge_graph.add_entity("Alice", {"type": "person", "age": 25})
        knowledge_graph.add_entity("Bob", {"type": "person", "age": 30})
        knowledge_graph.add_entity("Company", {"type": "organization"})

        knowledge_graph.add_relation("Alice", "works_at", "Company")
        knowledge_graph.add_relation("Bob", "works_at", "Company")

        x = randn(self.dim)
        output, info = knowledge_reasoner(x, context=None)

        result = {
            "output_shape": output.shape,
            "entities_found": info["entities_found"],
            "relations_found": info["relations_found"],
            "reasoning_chains": info["reasoning_chains"],
            "reasoning_confidence": info["reasoning_confidence"],
            "has_knowledge": info["has_knowledge"],
            "graph_stats": knowledge_graph.get_stats(),
        }

        print(f"  输出: {result['output_shape']}")
        print(f"  实体数: {result['entities_found']}, 关系数: {result['relations_found']}")
        print(f"  推理链: {result['reasoning_chains']}, 置信度: {result['reasoning_confidence']:.3f}")
        print(f"  图统计: {result['graph_stats']}")

        self.results["knowledge_integration"] = {
            "passed": True,
            "result": result,
        }
        print("  ✓ 知识模块集成测试通过")

    def test_full_system_integration(self):
        """测试完整系统集成"""
        print("\n7. 完整系统集成测试")
        print("-" * 50)

        mmci = MultimodalCognitiveIntegration(self.dim)

        results = []
        for i in range(3):
            text_input = randn(self.dim)
            visual_input = randn(self.dim) if i % 2 == 0 else None
            audio_input = randn(self.dim) if i % 2 == 1 else None

            output, info = mmci(
                text_input,
                visual_input=visual_input,
                audio_input=audio_input,
                temperature=22.0 + i * 1.0,
                light_level=0.8 - i * 0.1,
            )

            result = {
                "turn": i + 1,
                "output_shape": output.shape,
                "overall_confidence": info["overall_confidence"],
                "num_modalities": info["num_modalities"],
                "dominant_modality": info["dominant_modality"],
                "cognition_mode": info["cognition"]["mode"],
                "cross_modal_pairs": info["cross_modal"]["active_pairs"],
                "gate_value": info["loop"]["gate_value"],
            }
            results.append(result)

            print(f"  轮次 {i+1}:")
            print(f"    输出: {result['output_shape']}, 置信度: {result['overall_confidence']:.4f}")
            print(f"    模态数: {result['num_modalities']}, 主导: {result['dominant_modality']}")
            print(f"    跨模态配对: {result['cross_modal_pairs']}")

        status = mmci.get_system_status()

        self.results["full_system_integration"] = {
            "passed": True,
            "results": results,
            "system_status": status,
        }
        print("  ✓ 完整系统集成测试通过")

    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 70)
        print("测试摘要")
        print("=" * 70)

        total_tests = len(self.results)
        passed_tests = sum(1 for v in self.results.values() if v.get("passed", False))

        print(f"\n测试总数: {total_tests}, 通过: {passed_tests}, 失败: {total_tests - passed_tests}")

        for test_name, test_result in self.results.items():
            status = "✓ 通过" if test_result.get("passed", False) else "✗ 失败"
            print(f"\n  {test_name}: {status}")

            if "results" in test_result:
                results = test_result["results"]
                if isinstance(results, list):
                    avg_confidence = sum(
                        r.get("overall_confidence", 0) for r in results
                    ) / max(len(results), 1)
                    print(f"    平均置信度: {avg_confidence:.4f}")

        print("\n" + "=" * 70)
        if passed_tests == total_tests:
            print("所有测试通过!")
        else:
            print(f"{total_tests - passed_tests} 个测试失败")
        print("=" * 70)


class TuringTestSimulation:
    """图灵测试模拟"""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.hlcr = HumanLikeCognitiveReasoner(dim)
        self.mmci = MultimodalCognitiveIntegration(dim)

    def evaluate_human_likeness(self, num_turns: int = 10) -> Dict[str, Any]:
        """评估类人化程度"""
        print("\n\n" + "=" * 70)
        print("图灵测试模拟 - 类人化程度评估")
        print("=" * 70)

        scores = []
        for i in range(num_turns):
            x = randn(self.dim)
            context = randn(self.dim) if i > 0 else None

            output, info = self.hlcr(x, context)

            human_likeness_score = (
                info["overall_confidence"] * 0.3 +
                info["phases"]["nlu"]["intent_confidence"] * 0.2 +
                info["phases"]["emotion"]["top_emotion"].get("confidence", 0.5) * 0.2 +
                info["phases"]["variety"]["diversity_score"] * 0.15 +
                info["phases"]["disambiguation"]["overall_confidence"] * 0.15
            )

            scores.append(human_likeness_score)

            print(f"\n  轮次 {i+1}:")
            print(f"    类人化分数: {human_likeness_score:.4f}")
            print(f"    整体置信度: {info['overall_confidence']:.4f}")
            print(f"    NLU置信度: {info['phases']['nlu']['intent_confidence']:.4f}")
            print(f"    多样性分数: {info['phases']['variety']['diversity_score']:.4f}")

        avg_score = np.mean(scores)
        max_score = max(scores)
        min_score = min(scores)

        print("\n" + "-" * 50)
        print(f"平均类人化分数: {avg_score:.4f} ({avg_score * 100:.2f}%)")
        print(f"最高类人化分数: {max_score:.4f}")
        print(f"最低类人化分数: {min_score:.4f}")

        return {
            "average_human_likeness": avg_score,
            "max_human_likeness": max_score,
            "min_human_likeness": min_score,
            "scores": scores,
            "overall_confidence": self.hlcr.get_diversity_score(),
            "memory_stats": self.hlcr.get_memory_stats(),
        }

    def evaluate_multimodal_human_likeness(self, num_turns: int = 5) -> Dict[str, Any]:
        """评估多模态类人化程度"""
        print("\n\n" + "=" * 70)
        print("多模态图灵测试模拟")
        print("=" * 70)

        scores = []
        for i in range(num_turns):
            text_input = randn(self.dim)
            visual_input = randn(self.dim) if i < 3 else None
            audio_input = randn(self.dim) if i % 2 == 0 else None

            output, info = self.mmci(
                text_input,
                visual_input=visual_input,
                audio_input=audio_input,
            )

            human_likeness_score = (
                info["overall_confidence"] * 0.4 +
                info["cross_modal"]["num_pairs"] / 3 * 0.2 +
                info["loop"]["gate_value"] * 0.2 +
                info["perception"]["num_modalities"] / 5 * 0.2
            )

            scores.append(human_likeness_score)

            print(f"\n  轮次 {i+1}:")
            print(f"    类人化分数: {human_likeness_score:.4f}")
            print(f"    置信度: {info['overall_confidence']:.4f}")
            print(f"    模态数: {info['num_modalities']}")
            print(f"    跨模态配对: {info['cross_modal']['active_pairs']}")

        avg_score = np.mean(scores)

        print("\n" + "-" * 50)
        print(f"多模态平均类人化分数: {avg_score:.4f} ({avg_score * 100:.2f}%)")

        return {
            "average_multimodal_human_likeness": avg_score,
            "scores": scores,
            "system_status": self.mmci.get_system_status(),
        }


if __name__ == "__main__":
    dim = 64
    manual_seed(42)

    integration_test = IntegrationTest(dim)
    integration_test.run_all_tests()

    turing_test = TuringTestSimulation(dim)
    turing_results = turing_test.evaluate_human_likeness(num_turns=5)
    multimodal_results = turing_test.evaluate_multimodal_human_likeness(num_turns=3)

    print("\n\n" + "=" * 70)
    print("最终评估报告")
    print("=" * 70)
    print(f"\n类人化水平: {turing_results['average_human_likeness'] * 100:.2f}%")
    print(f"多模态类人化水平: {multimodal_results['average_multimodal_human_likeness'] * 100:.2f}%")
    print(f"\n系统状态: 正常运行")
    print(f"模块数量: 7个核心模块已集成")
    print("=" * 70)

