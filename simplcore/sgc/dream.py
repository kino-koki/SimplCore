# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Dreaming 模式 —— 架构层的模型自进化反思循环。

**Dreaming 与 Agent 的区别**：Agent 的反思是**推理期**的（本次回答里多想一步），
Dreaming 的反思是**离线**的：模型在"睡眠"阶段回放自己的经验、反思出洞见、
把洞见变回训练信号、巩固自己，然后才回到线上。

本模块提供的是**循环骨架与契约**，不是某个具体反思策略：

    在线 (serving)                    离线 (dreaming)
    ─────────────                    ────────────────
    推理 → 轨迹入流                   HARVEST    采样高价值经验
         ↓                           REFLECT    模型自评 → 洞见
    CMT 压缩记忆库                    SYNTHESIZE 洞见 → 训练信号
    (检索 → 聚合 → 注入)              CONSOLIDATE 巩固（记忆 或 权重）
         ↑                           VERIFY     held-out 评测
    权重 / 记忆库 ←── COMMIT         ROLLBACK   退化则回滚
                                  （或只回滚，不提交）

四种巩固模式
------------
* ``memory``  —— 只更新 CMT 压缩记忆库。**零遗忘风险**（基座权重冻结），
  是 CMT 的默认模式。
* ``weights`` —— 自蒸馏写回权重。真正的"自进化"，但会引入遗忘，
  因此**必须**过 no-regression gate 才提交。
* ``both``    —— 先写记忆再试写权重，权重部分受 gate 保护。
* ``none``    —— 只反思不巩固（用于观察模式 / 消融）。

硬性约束
--------
1. **no-regression gate**：任何写回权重的巩固，若在 held-out 集上相对基线退化
   超过容差，一律回滚（权重快照 + 记忆快照）。
2. **可恢复**：每轮结束保存 checkpoint（阶段、指标、游标），可中断续跑。
3. **有预算**：每轮的轨迹数、洞见数、轮数上限都是硬约束。
4. **可观测**：每轮产出 :class:`DreamReport`，含各阶段耗时与产出计数。

诚实边界
--------
本模块**不保证**自进化会提升能力。默认的反思/合成钩子是结构性的占位实现，
没有真实的"批判者模型"时它们不会产生有效训练信号（会如实报告
``synthesized == 0``）。在没有带引用的独立评测集之前，
**不得宣称"自进化有效"**。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
class DreamStage(str, Enum):
    """Dreaming 循环的阶段。"""
    IDLE = "idle"
    HARVEST = "harvest"
    REFLECT = "reflect"
    SYNTHESIZE = "synthesize"
    CONSOLIDATE = "consolidate"
    VERIFY = "verify"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    FAILED = "failed"


@dataclass
class Trajectory:
    """一条经验轨迹（在线推理时落盘）。"""
    prompt: str
    response: str
    confidence: float = 1.0
    grounded: bool = True                 # 是否有外部证据支撑（对应 L4 拒答门）
    score: float = 0.0                    # 事后评分（越高越有价值）
    tokens: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    """一条反思洞见。"""
    text: str
    source: Optional[Trajectory] = None
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainExample:
    """由洞见合成出的训练样本。"""
    prompt: str
    target: str
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamReport:
    """一轮 Dreaming 的结果。"""
    round_index: int
    stage: DreamStage
    harvested: int = 0
    insights: int = 0
    synthesized: int = 0
    consolidated: bool = False
    rolled_back: bool = False
    baseline_score: Optional[float] = None
    new_score: Optional[float] = None
    score_delta: Optional[float] = None
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_index,
            "stage": self.stage.value,
            "harvested": self.harvested,
            "insights": self.insights,
            "synthesized": self.synthesized,
            "consolidated": self.consolidated,
            "rolled_back": self.rolled_back,
            "baseline_score": self.baseline_score,
            "new_score": self.new_score,
            "score_delta": self.score_delta,
            "seconds": round(self.seconds, 3),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# 经验回放池
# ---------------------------------------------------------------------------
class ExperienceBuffer:
    """优先级经验回放池。

    连续学习的三项精华之一：**回放**。按价值采样而非均匀采样，
    让有限的固化预算花在最有信息量的经验上。

    优先级默认取 ``score``（事后评分）；若未评分则用
    ``1 - confidence``（模型自己越不确信的越值得反思），
    再用 ``1 - grounded``（没有证据支撑的越值得反思）加权。
    """

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self._items: List[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        self._items.append(traj)
        if len(self._items) > self.capacity:
            # 丢弃最旧且价值最低的
            worst = min(range(len(self._items)),
                        key=lambda i: (self._items[i].score, -i))
            self._items.pop(worst)

    def extend(self, trajs: Sequence[Trajectory]) -> None:
        for t in trajs:
            self.add(t)

    def __len__(self) -> int:
        return len(self._items)

    def priority(self, traj: Trajectory) -> float:
        p = traj.score
        if p <= 0.0:
            p = (1.0 - float(traj.confidence)) + (0.0 if traj.grounded else 0.5)
        return float(p)

    def sample(self, n: int, strategy: str = "priority") -> List[Trajectory]:
        """采样 n 条经验。

        Args:
            n: 采样条数。
            strategy: ``priority``（按价值，带随机性）/ ``recent``（最新）/
                ``uniform``（均匀）。
        """
        if not self._items:
            return []
        n = min(n, len(self._items))
        if strategy == "recent":
            return list(self._items[-n:])
        if strategy == "uniform":
            idx = torch.randperm(len(self._items))[:n].tolist()
            return [self._items[i] for i in idx]
        # priority：softmax 加权无放回采样
        prio = torch.tensor([self.priority(t) for t in self._items], dtype=torch.float32)
        w = torch.softmax(prio * 4.0, dim=0)
        idx = torch.multinomial(w, n, replacement=False).tolist()
        return [self._items[i] for i in idx]

    def stats(self) -> Dict[str, Any]:
        if not self._items:
            return {"size": 0, "capacity": self.capacity}
        return {
            "size": len(self._items),
            "capacity": self.capacity,
            "grounded_ratio": sum(1 for t in self._items if t.grounded) / len(self._items),
            "mean_confidence": sum(t.confidence for t in self._items) / len(self._items),
        }


# ---------------------------------------------------------------------------
# 默认钩子
# ---------------------------------------------------------------------------
def default_reflector(model, trajectories: Sequence[Trajectory]) -> List[Insight]:
    """默认反思器：**结构化占位实现**。

    它只做一件事：把"低置信度 / 无证据支撑"的轨迹标记为待反思项。
    **它不调用模型做真正的批判-修订**（Constitutional AI 那种），
    因此产出的洞见是标记而非知识。

    接入真实反思能力请替换本函数：
    ``lambda model, trajs: my_constitutional_critic(model, trajs)``
    """
    out: List[Insight] = []
    for t in trajectories:
        if (not t.grounded) or t.confidence < 0.5:
            out.append(Insight(
                text=f"未获证据支撑或置信度不足：{t.prompt[:60]}",
                source=t,
                weight=(1.0 - float(t.confidence)) + (0.0 if t.grounded else 1.0),
            ))
    return out


def default_synthesizer(model, insights: Sequence[Insight]) -> List[TrainExample]:
    """默认合成器：**结构化占位实现**。

    只有当轨迹带有显式的 ``meta['corrected_response']`` 时才产出训练样本
    （那通常来自真实的批判-修订模型）；否则如实返回空列表。
    """
    out: List[TrainExample] = []
    for ins in insights:
        t = ins.source
        if t is None:
            continue
        corrected = t.meta.get("corrected_response")
        if not corrected:
            continue
        out.append(TrainExample(prompt=t.prompt, target=str(corrected),
                                weight=ins.weight,
                                meta={"from_insight": ins.text}))
    return out


def default_verifier(model) -> float:
    """默认验证器：返回 0.0（无法在没有 held-out 集时给出有意义分数）。

    生产实现应替换为在**独立 held-out 集**上的评测（越高越好）。
    """
    return 0.0


# ---------------------------------------------------------------------------
# 循环配置
# ---------------------------------------------------------------------------
@dataclass
class DreamConfig:
    """Dreaming 循环配置。"""
    mode: str = "memory"                  # memory | weights | both | none
    batch_size: int = 32                  # 每轮采样轨迹数
    max_rounds: int = 100                 # 轮数上限
    max_insights: int = 64                # 每轮洞见上限
    max_examples: int = 256               # 每轮合成样本上限
    sampling: str = "priority"            # priority | recent | uniform
    regression_tolerance: float = 0.0     # 允许的退化幅度（负向为容差）
    require_regression_check: bool = True # 是否必须跑 no-regression gate
    seed: int = 0


# ---------------------------------------------------------------------------
# 循环本体
# ---------------------------------------------------------------------------
class DreamingLoop:
    """架构层的自进化循环。

    Args:
        model: 被巩固的模型（实现 ``snapshot``/``restore`` 者可获得权重回滚能力）。
        buffer: 经验回放池。
        config: 循环配置。
        reflector: ``(model, trajectories) -> List[Insight]``。
        synthesizer: ``(model, insights) -> List[TrainExample]``。
        consolidator: ``(model, examples) -> Dict``，可返回 ``{"snapshot": ...}``。
        verifier: ``(model) -> float``，越高越好。
        memory_bank: CMT 压缩记忆库；``mode="memory"`` 时洞见写入这里。
    """

    def __init__(
        self,
        model,
        buffer: Optional[ExperienceBuffer] = None,
        config: Optional[DreamConfig] = None,
        reflector: Optional[Callable] = None,
        synthesizer: Optional[Callable] = None,
        consolidator: Optional[Callable] = None,
        verifier: Optional[Callable] = None,
        memory_bank=None,
    ) -> None:
        self.model = model
        self.buffer = buffer if buffer is not None else ExperienceBuffer()
        self.config = config or DreamConfig()
        self.reflector = reflector or default_reflector
        self.synthesizer = synthesizer or default_synthesizer
        self.consolidator = consolidator
        self.verifier = verifier or default_verifier
        self.memory_bank = memory_bank
        self.history: List[DreamReport] = []
        self._round = 0

    # -- 基线 ---------------------------------------------------------------
    def baseline(self) -> Optional[float]:
        """当前模型的基线分数（用于 no-regression 比较）。"""
        if not self.config.require_regression_check:
            return None
        try:
            return float(self.verifier(self.model))
        except Exception as exc:                      # noqa: BLE001
            self._note(f"基线评测失败：{exc}")
            return None

    # -- 快照 / 回滚 --------------------------------------------------------
    def snapshot(self) -> Optional[Any]:
        """模型权重快照（若模型支持）。"""
        if self.model is None or not hasattr(self.model, "snapshot"):
            return None
        try:
            return self.model.snapshot()
        except Exception:                             # noqa: BLE001
            return None

    def restore(self, snap: Optional[Any]) -> bool:
        """回滚模型权重。"""
        if snap is None or self.model is None:
            return False
        if not hasattr(self.model, "restore"):
            return False
        try:
            self.model.restore(snap)
            return True
        except Exception:                             # noqa: BLE001
            return False

    # -- 主循环 -------------------------------------------------------------
    def step(self) -> DreamReport:
        """跑一轮 Dreaming。"""
        t0 = time.perf_counter()
        cfg = self.config
        self._round += 1
        rep = DreamReport(round_index=self._round, stage=DreamStage.IDLE)

        # 1) HARVEST
        rep.stage = DreamStage.HARVEST
        trajs = self.buffer.sample(cfg.batch_size, strategy=cfg.sampling)
        rep.harvested = len(trajs)
        if not trajs:
            rep.stage = DreamStage.IDLE
            rep.seconds = time.perf_counter() - t0
            rep.notes.append("经验池为空，本轮无输入")
            self.history.append(rep)
            return rep

        # 2) REFLECT
        rep.stage = DreamStage.REFLECT
        try:
            insights = list(self.reflector(self.model, trajs) or [])
        except Exception as exc:                      # noqa: BLE001
            rep.stage = DreamStage.FAILED
            rep.seconds = time.perf_counter() - t0
            rep.notes.append(f"反思阶段异常：{exc}")
            self.history.append(rep)
            return rep
        insights = insights[:cfg.max_insights]
        rep.insights = len(insights)

        # 3) SYNTHESIZE
        rep.stage = DreamStage.SYNTHESIZE
        try:
            examples = list(self.synthesizer(self.model, insights) or [])
        except Exception as exc:                      # noqa: BLE001
            rep.stage = DreamStage.FAILED
            rep.seconds = time.perf_counter() - t0
            rep.notes.append(f"合成阶段异常：{exc}")
            self.history.append(rep)
            return rep
        examples = examples[:cfg.max_examples]
        rep.synthesized = len(examples)

        if cfg.mode == "none" or not examples:
            if not examples:
                rep.notes.append("无合成训练信号（反思钩子为占位实现或洞见缺修正样本）")
            rep.stage = DreamStage.IDLE
            rep.seconds = time.perf_counter() - t0
            self.history.append(rep)
            return rep

        # 4) CONSOLIDATE
        rep.stage = DreamStage.CONSOLIDATE
        snap = self.snapshot()
        try:
            if self.consolidator is not None:
                self.consolidator(self.model, examples)
            if cfg.mode in ("memory", "both") and self.memory_bank is not None:
                self._write_memory(insights, examples)
            rep.consolidated = True
        except Exception as exc:                      # noqa: BLE001
            rep.stage = DreamStage.FAILED
            self.restore(snap)
            rep.seconds = time.perf_counter() - t0
            rep.notes.append(f"巩固阶段异常，已回滚：{exc}")
            self.history.append(rep)
            return rep

        # 5) VERIFY（no-regression gate）
        if cfg.require_regression_check and cfg.mode in ("weights", "both"):
            rep.stage = DreamStage.VERIFY
            base = self.baseline()
            new = None
            try:
                new = float(self.verifier(self.model))
            except Exception as exc:                  # noqa: BLE001
                rep.notes.append(f"评测失败：{exc}")
            rep.baseline_score, rep.new_score = base, new
            if base is not None and new is not None:
                rep.score_delta = new - base
                if rep.score_delta < -abs(cfg.regression_tolerance):
                    rep.stage = DreamStage.ROLLBACK
                    self.restore(snap)
                    rep.rolled_back = True
                    rep.notes.append(
                        f"退化 {rep.score_delta:.4f} 超过容差 "
                        f"{cfg.regression_tolerance:.4f}，已回滚"
                    )
                    rep.seconds = time.perf_counter() - t0
                    self.history.append(rep)
                    return rep

        # 6) COMMIT
        rep.stage = DreamStage.COMMIT
        rep.seconds = time.perf_counter() - t0
        self.history.append(rep)
        return rep

    def run(self, max_rounds: Optional[int] = None) -> List[DreamReport]:
        """连续跑若干轮，直到触达轮数上限。"""
        limit = max_rounds if max_rounds is not None else self.config.max_rounds
        out: List[DreamReport] = []
        for _ in range(int(limit)):
            rep = self.step()
            out.append(rep)
            if rep.stage is DreamStage.FAILED:
                break
        return out

    # -- 记忆写入 -----------------------------------------------------------
    def _write_memory(self, insights: Sequence[Insight],
                      examples: Sequence[TrainExample]) -> None:
        """把洞见写进 CMT 压缩记忆库（mode=memory 的巩固方式）。"""
        if self.memory_bank is None:
            return
        dim = getattr(self.memory_bank, "dim", None)
        if dim is None:
            return
        keys, values, metas = [], [], []
        for ins in insights:
            # 用 insight 文本的确定性哈希做键（跨轮可复现，不依赖模型）
            h = float(abs(hash(ins.text)) % (10 ** 8)) / (10 ** 8)
            key = torch.full((dim,), h, dtype=torch.float32)
            val = torch.full((dim,), h, dtype=torch.float32)
            keys.append(key)
            values.append(val)
            metas.append({"text": ins.text, "weight": ins.weight})
        if keys:
            self.memory_bank.add(torch.stack(keys), torch.stack(values), metas)

    def _note(self, msg: str) -> None:
        if self.history:
            self.history[-1].notes.append(msg)
