# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
AGI主循环控制器 (AGI Controller) - SimplCore版
=============================

实现感知-推理-行动-反思的完整闭环，是SimplCore AGI的核心调度系统。

架构设计：
    感知 (Perceive) → 理解 (Understand) → 推理 (Reason) → 决策 (Decide) → 行动 (Act) → 反思 (Reflect)

集成模块：
    1. 多模态感知 - MultimodalPerception, AudioPerception, EnvironmentPerception
    2. 语境感知语义解析 - ContextAwareSemanticParser, MultiTurnComprehension
    3. 对话状态跟踪 - DialogueStateTracker
    4. 长程记忆 - LongRangeMemory
    5. 情绪智能 - EmotionalIntelligence
    6. 类人认知推理 - HumanLikeCognitiveReasoner
    7. 多模态认知整合 - MultimodalCognitiveIntegration
    8. 知识图谱 - KnowledgeGraph, KnowledgeReasoner
    9. 知识自迭代 - KnowledgeSelfIteration
    10. 自我改进 - SelfImprovementModule, SelfLearningLoop
    11. 响应多样性 - ResponseVariety
"""

from __future__ import annotations

import sys
sys.path.insert(0, '.')

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import time

from ..engine.tensor import SimpTensor, randn, zeros, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm

from ..perception.multimodal_fusion import (
    ModalityType,
    MultimodalPerception,
    AudioPerception,
    EnvironmentPerception,
)

from ..nlu.context_aware_parser import (
    ContextAwareSemanticParser,
    MultiTurnComprehension,
    CoherenceLevel,
)

from ..dialogue.dialogue_state_tracker import DialogueStateTracker
from ..dialogue.long_range_memory import LongRangeMemory

from ..emotion.emotional_intelligence import EmotionalIntelligence

from ..cognition.human_like_reasoner import HumanLikeCognitiveReasoner, HumanLikeMode
from ..cognition.multimodal_cognition import (
    MultimodalCognitiveIntegration,
    IntegrationMode,
    CrossModalReasoner,
)

from ..knowledge.knowledge_graph import KnowledgeGraph, KnowledgeReasoner

from ..generation.response_variety import ResponseVariety

from ..cognition.self_improvement import (
    SelfImprovementModule,
    SelfLearningLoop,
    IntegratedSelfImprovement,
)


class AGIState(Enum):
    IDLE = "idle"
    PERCEIVING = "perceiving"
    UNDERSTANDING = "understanding"
    REASONING = "reasoning"
    DECIDING = "deciding"
    ACTING = "acting"
    REFLECTING = "reflecting"
    LEARNING = "learning"
    ERROR = "error"


class EventType(Enum):
    PERCEPTION_COMPLETE = "perception_complete"
    UNDERSTANDING_COMPLETE = "understanding_complete"
    REASONING_COMPLETE = "reasoning_complete"
    DECISION_MADE = "decision_made"
    ACTION_COMPLETE = "action_complete"
    REFLECTION_COMPLETE = "reflection_complete"
    LEARNING_COMPLETE = "learning_complete"
    ERROR_OCCURRED = "error_occurred"
    USER_INPUT = "user_input"


class AGIEvent:
    """AGI事件"""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any] = None,
        timestamp: Optional[datetime] = None,
        priority: int = 0,
    ):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = timestamp or datetime.now()
        self.priority = priority
    
    def __repr__(self):
        return f"AGIEvent({self.event_type.value}, {self.timestamp}, priority={self.priority})"


class AGICore(Module):
    """AGI核心 - 整合所有模块"""
    
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # ===== 感知层 =====
        self.multimodal_perception = MultimodalPerception(dim)
        self.register_module("multimodal_perception", self.multimodal_perception)
        
        self.audio_perception = AudioPerception(dim)
        self.register_module("audio_perception", self.audio_perception)
        
        self.env_perception = EnvironmentPerception(dim)
        self.register_module("env_perception", self.env_perception)
        
        # ===== NLU层 =====
        self.semantic_parser = ContextAwareSemanticParser(dim)
        self.register_module("semantic_parser", self.semantic_parser)
        
        self.multi_turn = MultiTurnComprehension(dim)
        self.register_module("multi_turn", self.multi_turn)
        
        # ===== 对话层 =====
        self.dialogue_tracker = DialogueStateTracker(dim)
        self.register_module("dialogue_tracker", self.dialogue_tracker)
        
        self.long_range_memory = LongRangeMemory(dim)
        self.register_module("long_range_memory", self.long_range_memory)
        
        # ===== 情绪层 =====
        self.emotional_intelligence = EmotionalIntelligence(dim)
        self.register_module("emotional_intelligence", self.emotional_intelligence)
        
        # ===== 认知层 =====
        self.cognitive_reasoner = HumanLikeCognitiveReasoner(dim)
        self.register_module("cognitive_reasoner", self.cognitive_reasoner)
        
        self.multimodal_cognition = MultimodalCognitiveIntegration(dim)
        self.register_module("multimodal_cognition", self.multimodal_cognition)
        
        # ===== 知识层 =====
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_reasoner = KnowledgeReasoner(dim)
        self.register_module("knowledge_reasoner", self.knowledge_reasoner)
        
        # ===== 响应多样性 =====
        self.response_variety = ResponseVariety(dim)
        self.register_module("response_variety", self.response_variety)
        
        # ===== 自我改进 =====
        self.self_improvement = IntegratedSelfImprovement(dim)
        self.register_module("self_improvement", self.self_improvement)
        
        # ===== 输出投影 =====
        self.output_proj = Linear(dim * 2, dim)
        self.register_module("output_proj", self.output_proj)
        
        # ===== 状态 =====
        self._dialogue_history: List[SimpTensor] = []
        self._perception_buffer: List[Dict[str, Any]] = []
        self._environment_state: Dict[str, float] = {"temperature": 25.0, "light_level": 1.0}
    
    def perceive(
        self,
        text_input: Optional[SimpTensor] = None,
        visual_input: Optional[SimpTensor] = None,
        audio_input: Optional[SimpTensor] = None,
    ) -> Dict[str, Any]:
        """感知阶段"""
        results = {}
        
        if text_input is not None:
            parsed, parse_info = self.semantic_parser.parse(text_input)
            comprehension, compre_info = self.multi_turn(text_input)
            results["text"] = {
                "parsed": parsed,
                "intent": parse_info["intent"],
                "intent_confidence": parse_info["intent_confidence"],
                "coherence": parse_info["coherence"],
                "topic": parse_info["topic"],
            }
        
        if visual_input is not None or audio_input is not None or text_input is not None:
            modalities = {}
            if text_input is not None:
                modalities[ModalityType.TEXT] = text_input
            if visual_input is not None:
                modalities[ModalityType.VISION] = visual_input
            if audio_input is not None:
                modalities[ModalityType.AUDIO] = audio_input
            
            if ModalityType.ENVIRONMENT not in modalities:
                env_input = randn(self.dim)
                env_feature, _ = self.env_perception(
                    env_input,
                    temperature=self._environment_state["temperature"],
                    light_level=self._environment_state["light_level"],
                )
                modalities[ModalityType.ENVIRONMENT] = env_feature
            
            fused, fusion_info = self.multimodal_perception(modalities)
            results["multimodal"] = {
                "fused": fused,
                "modalities": fusion_info["active_modalities"],
                "dominant": fusion_info["dominant_modality"],
                "weights": fusion_info["modality_weights"],
            }
        
        self._perception_buffer.append(results)
        if len(self._perception_buffer) > 50:
            self._perception_buffer = self._perception_buffer[-50:]
        
        return results
    
    def reason(
        self,
        query: SimpTensor,
        perception_data: Optional[Dict[str, Any]] = None,
        mode: HumanLikeMode = HumanLikeMode.CONVERSATION,
    ) -> Dict[str, Any]:
        """推理阶段"""
        context = None
        if self._dialogue_history:
            context = self._dialogue_history[-1]
        
        # 类人认知推理
        cognitive_output, cognitive_info = self.cognitive_reasoner(
            query, context=context, mode=mode
        )
        
        # 知识增强
        knowledge_enhanced, knowledge_info = self.knowledge_reasoner(
            cognitive_output, context=context
        )
        
        # 多模态增强（如果有感知数据）
        if perception_data and "multimodal" in perception_data:
            fused = perception_data["multimodal"]["fused"]
            combined = SimpTensor(np.concatenate([knowledge_enhanced._data, fused._data]))
        else:
            combined = SimpTensor(np.concatenate([knowledge_enhanced._data, knowledge_enhanced._data]))
        
        output = gelu(self.output_proj(combined))
        
        overall_confidence = (
            cognitive_info.get("overall_confidence", 0.5) * 0.6 +
            knowledge_info.get("reasoning_confidence", 0.5) * 0.4
        )
        
        return {
            "output": output,
            "confidence": round(overall_confidence, 4),
            "cognitive_info": cognitive_info,
            "knowledge_info": knowledge_info,
        }
    
    def update_dialogue(self, x: SimpTensor):
        """更新对话历史"""
        self._dialogue_history.append(x)
        if len(self._dialogue_history) > 50:
            self._dialogue_history = self._dialogue_history[-50:]
    
    def update_environment(self, temperature: float = None, light_level: float = None):
        """更新环境状态"""
        if temperature is not None:
            self._environment_state["temperature"] = temperature
        if light_level is not None:
            self._environment_state["light_level"] = light_level
    
    def forward(
        self,
        text_input: SimpTensor,
        visual_input: Optional[SimpTensor] = None,
        audio_input: Optional[SimpTensor] = None,
        mode: HumanLikeMode = HumanLikeMode.CONVERSATION,
    ) -> Dict[str, Any]:
        """完整前向传播"""
        perception = self.perceive(text_input, visual_input, audio_input)
        reasoning = self.reason(text_input, perception, mode)
        self.update_dialogue(reasoning["output"])
        
        return {
            "perception": perception,
            "reasoning": reasoning,
            "mode": mode.value,
        }
    
    def reset(self):
        """重置"""
        self._dialogue_history = []
        self._perception_buffer = []
        self._environment_state = {"temperature": 25.0, "light_level": 1.0}
        self.cognitive_reasoner.reset()


class PerceptionLoop:
    """感知循环 - 从环境获取信息"""
    
    def __init__(self, agi_core: AGICore):
        self.core = agi_core
        self.is_running = False
        self.last_perception: Optional[Dict[str, Any]] = None
        self.perception_history: List[Dict[str, Any]] = []
    
    def run(
        self,
        text_input: Optional[SimpTensor] = None,
        visual_input: Optional[SimpTensor] = None,
        audio_input: Optional[SimpTensor] = None,
    ) -> Dict[str, Any]:
        """运行感知循环"""
        self.is_running = True
        
        results = self.core.perceive(text_input, visual_input, audio_input)
        self.last_perception = results
        self.perception_history.append({
            "timestamp": datetime.now(),
            "data": results,
        })
        
        self.is_running = False
        return results


class CognitionLoop:
    """认知循环 - 理解和推理"""
    
    def __init__(self, agi_core: AGICore):
        self.core = agi_core
        self.is_running = False
        self.reasoning_history: List[Dict[str, Any]] = []
    
    def run(
        self,
        query: SimpTensor,
        perception_data: Optional[Dict[str, Any]] = None,
        mode: HumanLikeMode = HumanLikeMode.CONVERSATION,
    ) -> Dict[str, Any]:
        """运行认知循环"""
        self.is_running = True
        
        results = self.core.reason(query, perception_data, mode)
        
        self.reasoning_history.append({
            "timestamp": datetime.now(),
            "results": results,
        })
        
        self.is_running = False
        return results


class ReflectionLoop:
    """反思循环 - 自我评估和改进"""
    
    def __init__(self, agi_core: AGICore):
        self.core = agi_core
        self.is_running = False
        self.reflection_history: List[Dict[str, Any]] = []
    
    def run(
        self,
        query_features: SimpTensor,
        confidence: float = 0.0,
        correct: bool = True,
        response_time: float = 0.0,
        knowledge_found: int = 0,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行反思循环"""
        self.is_running = True
        
        improvement_result = self.core.self_improvement.process_outcome(
            query_features=query_features,
            confidence=confidence,
            correct=correct,
            response_time=response_time,
            knowledge_found=knowledge_found,
            error_type=error_type,
        )
        
        self.reflection_history.append({
            "timestamp": datetime.now(),
            "improvement": improvement_result,
        })
        
        self.is_running = False
        return improvement_result


class AGIController:
    """AGI主控制器 - 协调所有循环"""
    
    def __init__(self, agi_core: AGICore):
        self.core = agi_core
        self.state = AGIState.IDLE
        self.event_queue: List[AGIEvent] = []
        self.running = False
        self.cycle_count = 0
        
        self.perception_loop = PerceptionLoop(agi_core)
        self.cognition_loop = CognitionLoop(agi_core)
        self.reflection_loop = ReflectionLoop(agi_core)
        
        self.callbacks: Dict[EventType, List[Callable]] = {}
        self.error_count = 0
        self.max_errors = 10
        
        self._register_default_callbacks()
    
    def _register_default_callbacks(self):
        """注册默认回调"""
        self.register_callback(EventType.PERCEPTION_COMPLETE, self._on_perception_complete)
        self.register_callback(EventType.REASONING_COMPLETE, self._on_reasoning_complete)
        self.register_callback(EventType.REFLECTION_COMPLETE, self._on_reflection_complete)
        self.register_callback(EventType.ERROR_OCCURRED, self._on_error)
    
    def register_callback(self, event_type: EventType, callback: Callable):
        """注册事件回调"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def emit_event(self, event: AGIEvent):
        """发出事件"""
        self.event_queue.append(event)
        if event.event_type in self.callbacks:
            for callback in self.callbacks[event.event_type]:
                callback(event)
    
    def _on_perception_complete(self, event: AGIEvent):
        pass
    
    def _on_reasoning_complete(self, event: AGIEvent):
        pass
    
    def _on_reflection_complete(self, event: AGIEvent):
        pass
    
    def _on_error(self, event: AGIEvent):
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.stop()
    
    def start(self):
        """启动控制器"""
        self.running = True
        self.state = AGIState.IDLE
        self.cycle_count = 0
        self.error_count = 0
    
    def stop(self):
        """停止控制器"""
        self.running = False
        self.state = AGIState.IDLE
    
    def step(self, text_input: SimpTensor) -> Dict[str, Any]:
        """执行一个完整的AGI循环"""
        if not self.running:
            return {"error": "控制器未启动"}
        
        self.cycle_count += 1
        start_time = datetime.now()
        
        try:
            # 1. 感知阶段
            self.state = AGIState.PERCEIVING
            perception_data = self.perception_loop.run(text_input=text_input)
            self.emit_event(AGIEvent(EventType.PERCEPTION_COMPLETE, {"perception": perception_data}))
            
            # 2. 推理阶段
            self.state = AGIState.REASONING
            reasoning_results = self.cognition_loop.run(text_input, perception_data)
            self.emit_event(AGIEvent(EventType.REASONING_COMPLETE, {"reasoning": reasoning_results}))
            
            # 3. 更新对话
            self.core.update_dialogue(reasoning_results["output"])
            
            # 4. 反思阶段
            self.state = AGIState.REFLECTING
            confidence = reasoning_results.get("confidence", 0.5)
            reflection_result = self.reflection_loop.run(
                query_features=text_input,
                confidence=confidence,
                correct=True,
                response_time=(datetime.now() - start_time).total_seconds(),
                knowledge_found=reasoning_results.get("knowledge_info", {}).get("entities_found", 0),
            )
            self.emit_event(AGIEvent(EventType.REFLECTION_COMPLETE, {"reflection": reflection_result}))
            
            self.state = AGIState.IDLE
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "cycle": self.cycle_count,
                "duration": duration,
                "perception": {
                    "intent": perception_data.get("text", {}).get("intent", "unknown"),
                    "coherence": perception_data.get("text", {}).get("coherence", {}).get("level", "unknown"),
                    "topic_change": perception_data.get("text", {}).get("topic", {}).get("is_topic_change", False),
                },
                "reasoning": {
                    "confidence": confidence,
                    "cognitive_mode": reasoning_results.get("cognitive_info", {}).get("mode", "unknown"),
                },
                "reflection": {
                    "improvements": len(reflection_result.get("improvements", [])),
                    "knowledge_enhanced": reflection_result.get("knowledge_enhanced", False),
                },
            }
        
        except Exception as e:
            self.emit_event(AGIEvent(EventType.ERROR_OCCURRED, {"error": str(e)}))
            return {"error": str(e), "cycle": self.cycle_count}
    
    def process_query(self, query_vec: SimpTensor) -> Dict[str, Any]:
        """处理用户查询"""
        self.emit_event(AGIEvent(EventType.USER_INPUT, {"data": "query_received"}, priority=10))
        
        self.state = AGIState.REASONING
        
        perception = self.core.perceive(text_input=query_vec)
        reasoning = self.core.reason(query_vec, perception)
        
        self.state = AGIState.REFLECTING
        
        self.reflection_loop.run(
            query_features=query_vec,
            confidence=reasoning.get("confidence", 0.7),
            correct=True,
            knowledge_found=reasoning.get("knowledge_info", {}).get("entities_found", 0),
        )
        
        self.state = AGIState.IDLE
        
        return reasoning
    
    def run_n_cycles(self, n: int = 10) -> List[Dict[str, Any]]:
        """运行N个循环"""
        results = []
        self.start()
        
        for i in range(n):
            x = randn(self.core.dim)
            result = self.step(x)
            results.append(result)
            
            if i % 2 == 0:
                status = "OK" if "error" not in result else "ERR"
                print(f"  循环 {i+1}/{n} - {status} - 耗时 {result.get('duration', 0):.3f}s")
        
        self.stop()
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        status = self.core.self_improvement.get_full_status() if hasattr(self.core, 'self_improvement') else {}
        return {
            "running": self.running,
            "state": self.state.value,
            "cycle_count": self.cycle_count,
            "error_count": self.error_count,
            "perception_history": len(self.perception_loop.perception_history),
            "reasoning_history": len(self.cognition_loop.reasoning_history),
            "reflection_history": len(self.reflection_loop.reflection_history),
            "capabilities": status.get("capabilities", {}),
            "improvements": status.get("improvement_summary", {}),
        }


class AGISession:
    """AGI会话管理器"""
    
    def __init__(self, controller: AGIController):
        self.controller = controller
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.queries: List[Dict[str, Any]] = []
        self.total_duration = 0.0
    
    def ask(self, query_vec: SimpTensor) -> Dict[str, Any]:
        """处理查询"""
        start = datetime.now()
        result = self.controller.process_query(query_vec)
        duration = (datetime.now() - start).total_seconds()
        self.total_duration += duration
        
        self.queries.append({
            "result": result,
            "duration": duration,
            "timestamp": datetime.now(),
        })
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        return {
            "session_id": self.session_id,
            "duration": self.total_duration,
            "total_queries": len(self.queries),
            "avg_time": self.total_duration / max(len(self.queries), 1),
            "cycle_count": self.controller.cycle_count,
            "controller_status": self.controller.get_status(),
        }


if __name__ == "__main__":
    from SimplCore.engine.tensor import manual_seed, randn
    manual_seed(0)
    
    dim = 64
    print("=" * 70)
    print("AGIController (SimplCore版) 集成测试")
    print("=" * 70)
    
    print("\n1. 测试AGI核心初始化:")
    core = AGICore(dim)
    controller = AGIController(core)
    print(f"  核心模块数: {len([p for p in dir(core) if not p.startswith('_')])}")
    status = controller.get_status()
    print(f"  状态: {status['state']}, 循环: {status['cycle_count']}")
    
    print("\n2. 测试感知循环:")
    text_input = randn(dim)
    perception = core.perceive(text_input=text_input)
    print(f"  意图: {perception.get('text', {}).get('intent', 'unknown')}")
    print(f"  连贯性: {perception.get('text', {}).get('coherence', {}).get('level', 'unknown')}")
    print(f"  模态数: {len(perception.get('multimodal', {}).get('modalities', []))}")
    
    print("\n3. 测试推理循环:")
    reasoning = core.reason(text_input)
    print(f"  置信度: {reasoning.get('confidence', 0):.4f}")
    print(f"  认知模式: {reasoning.get('cognitive_info', {}).get('mode', 'unknown')}")
    
    print("\n4. 测试完整AGI循环:")
    controller.start()
    for i in range(3):
        x = randn(dim)
        result = controller.step(x)
        if "error" not in result:
            print(f"  循环 {i+1}: 耗时 {result.get('duration', 0):.3f}s, "
                  f"置信度: {result.get('reasoning', {}).get('confidence', 0):.4f}")
    
    print("\n5. 测试会话管理:")
    session = AGISession(controller)
    result1 = session.ask(randn(dim))
    result2 = session.ask(randn(dim))
    summary = session.get_summary()
    print(f"  查询次数: {summary['total_queries']}")
    print(f"  平均耗时: {summary['avg_time']:.3f}s")
    
    print("\n6. 测试多轮连贯性:")
    for i in range(5):
        x = randn(dim)
        perception = core.perceive(text_input=x)
        coherence = perception.get("text", {}).get("coherence", {}).get("level", "unknown")
        print(f"  轮次 {i+1}: 连贯性={coherence}")
    
    print("\n7. 测试状态报告:")
    status = controller.get_status()
    print(f"  运行中: {status['running']}")
    print(f"  总循环: {status['cycle_count']}")
    if "capabilities" in status and isinstance(status["capabilities"], dict):
        for cap, score in status["capabilities"].get("capabilities", {}).items():
            print(f"    {cap}: {score:.2%}")
    
    print("\n" + "=" * 70)
    print("AGIController 集成测试完成!")
    print("=" * 70)

