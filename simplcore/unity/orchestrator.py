# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity —— 自组织智能体协作框架 (编排内核).

SimplUnity 是 SimplCore 的 "神经系统": 它不替代底层模型 (LatticeCore / 世界模型),
而是把它们作为可调度的 "能力单元 (Capability Unit)" 编排起来, 形成
感知 -> 推理 -> 行动 -> 反思 -> 自我模型 的持续闭环.

设计原则
--------
1. **单元即服务**: 每个专家/模块注册为一个 Unit, 通过消息总线 (MessageBus)
   通信; 新增能力只需注册, 不改核心 (可扩展).
2. **闭环优先**: 每一次交互都经过 Perception -> Cognition -> Action -> Feedback,
   而不是单向生成. 这与 AGI_ARCHITECTURE_DESIGN.md 的闭环一致.
3. **与既有组件集成**:
   - 语言/多模态理解: 接入 UnitEmbedding + LatticeCore (无词表, 无注意力).
   - 记忆: 接入 ContextBank (上下文) / PersistentMemory (长期).
   - 世界模型: 接入 WorldModel + RolloutEngine (潜空间推演).
   - 元认知/自我: 接入 metacognition / self_model (待建).
4. **自洽**: 所有消息基于 Unit 表征, 无词表; 调度策略可学习或规则混合.

本文件为骨架 (skeleton): 接口与数据流完整, 具体实现依赖上层模块逐步接入.
当前环境缺少 numpy, 仅做 py_compile 验证; 集成测试需联网安装 numpy 后运行.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# 消息类型 (意图/感知/动作/反思/状态)
MSG_PERCEPT = "percept"
MSG_COGNITION = "cognition"
MSG_ACTION = "action"
MSG_REFLECTION = "reflection"
MSG_SELF = "self_state"


@dataclass
class Message:
    """总线消息: 所有载荷都是 Unit 表征或标量信号, 无词表."""
    kind: str
    source: str
    target: Optional[str]
    payload: object
    timestamp: float = 0.0


class ExpertUnit:
    """SimplUnity 中的任一能力单元 (专家/模块) 基类."""

    def __init__(self, name: str) -> None:
        self.name = name

    def handle(self, msg: "Message") -> Optional["Message"]:
        """处理消息并返回 (可选) 下一条消息. 子类实现."""
        raise NotImplementedError


class MessageBus:
    """轻量发布-订阅总线, 按 target 路由; 支持广播 (target=None)."""

    def __init__(self) -> None:
        self._subs: Dict[str, List[ExpertUnit]] = {}
        self._log: List["Message"] = []

    def register(self, unit: "ExpertUnit") -> None:
        self._subs.setdefault(unit.name, []).append(unit)

    def publish(self, msg: "Message") -> List["Message"]:
        """投递消息, 返回各订阅者的响应."""
        self._log.append(msg)
        out: List["Message"] = []
        if msg.target is None:
            for name, units in self._subs.items():
                for u in units:
                    r = u.handle(msg)
                    if r is not None:
                        out.append(r)
        elif msg.target in self._subs:
            for u in self._subs[msg.target]:
                r = u.handle(msg)
                if r is not None:
                    out.append(r)
        return out


class UnityOrchestrator:
    """SimplUnity 编排器: 串联感知-推理-行动-反思闭环.

    模块协作:
      percept -> [PerceptionUnit] -> cognition -> [CognitionUnit] -> action
              -> [ActionUnit] -> feedback -> [ReflectionUnit] -> 更新 SelfModel
    """

    def __init__(self) -> None:
        self.bus = MessageBus()
        self.units: Dict[str, ExpertUnit] = {}
        self.self_state: dict = {}  # 自我模型状态 (待接 self_model 模块)

    def register(self, unit: "ExpertUnit") -> "UnityOrchestrator":
        self.units[unit.name] = unit
        self.bus.register(unit)
        return self

    def step(self, percept_payload: object) -> List["Message"]:
        """单步闭环: 从一次感知输入推进到反思.

        Args:
            percept_payload: 原始感知 (文本/图像等), 由 PerceptionUnit 内部转 Unit.
        Returns:
            本轮产生的消息列表 (含 action / reflection).
        """
        t = 0.0
        percept = Message(MSG_PERCEPT, "env", "perception", percept_payload, t)
        msgs = self.bus.publish(percept)

        # 取 cognition / action 之后的反馈进入反思
        downstream = [m for m in msgs if m.kind in (MSG_COGNITION, MSG_ACTION)]
        reflection = Message(
            MSG_REFLECTION, "orchestrator", "reflection", downstream, t
        )
        reflected = self.bus.publish(reflection)
        return msgs + reflected

    def run(self, percepts: List[object], rounds: int = 1) -> List["Message"]:
        """多轮闭环运行."""
        history: List["Message"] = []
        for _ in range(rounds):
            for p in percepts:
                history.extend(self.step(p))
        return history


class PerceptionUnit(ExpertUnit):
    """把原始输入转为 Unit 表征 (文本走 UnitCodec, 多模态可扩展)."""

    def __init__(self, embedder=None) -> None:
        super().__init__("perception")
        self.embedder = embedder  # UnitEmbedding 实例或 None

    def handle(self, msg: "Message") -> Optional["Message"]:
        if msg.kind != MSG_PERCEPT:
            return None
        # 文本示例: 若有 embedder 则编码为 Unit 表征
        if self.embedder is not None and isinstance(msg.payload, str):
            vec = self.embedder.encode_and_embed(msg.payload)
            return Message(MSG_COGNITION, self.name, "cognition", vec)
        return Message(MSG_COGNITION, self.name, "cognition", msg.payload)


class CognitionUnit(ExpertUnit):
    """推理单元占位: 接入 LatticeCore / WorldModel (具体前向待接)."""

    def __init__(self, core=None) -> None:
        super().__init__("cognition")
        self.core = core

    def handle(self, msg: "Message") -> Optional["Message"]:
        if msg.kind != MSG_COGNITION:
            return None
        # TODO: 调用 core.forward(units) 产生表征/预测; 此处仅透传
        return Message(MSG_ACTION, self.name, "action", msg.payload)


class ActionUnit(ExpertUnit):
    """行动单元: 把表征转为可观测输出 (文本生成/工具调用等)."""

    def __init__(self, actuator: Optional[Callable] = None) -> None:
        super().__init__("action")
        self.actuator = actuator

    def handle(self, msg: "Message") -> Optional["Message"]:
        if msg.kind != MSG_ACTION:
            return None
        if self.actuator is not None:
            self.actuator(msg.payload)
        return None


class ReflectionUnit(ExpertUnit):
    """反思单元: 汇总本轮 cognition/action, 更新自我模型与记忆衰减触发."""

    def __init__(self, context_bank=None) -> None:
        super().__init__("reflection")
        self.context_bank = context_bank  # ContextBank 实例

    def handle(self, msg: "Message") -> Optional["Message"]:
        if msg.kind != MSG_REFLECTION:
            return None
        # 触发一次长期记忆衰减 (基于本次交互强度)
        if self.context_bank is not None:
            self.context_bank.decay_step()
        return None


def build_simplunity(embedder=None, context_bank=None, core=None) -> "UnityOrchestrator":
    """组装一个最小可用 SimplUnity 实例."""
    uni = UnityOrchestrator()
    uni.register(PerceptionUnit(embedder))
    uni.register(CognitionUnit(core))
    uni.register(ActionUnit())
    uni.register(ReflectionUnit(context_bank))
    return uni
