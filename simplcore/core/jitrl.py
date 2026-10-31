# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""JitRL 式测试时优势调制 —— 免训练的持续适应层（v1.2，agent 域）。

来源主张
--------
JitRL（Just-In-Time Reinforcement Learning, arXiv:2601.18510, ICML 2026
Oral）证明：维护一个**非参数经验轨迹记忆**，检索相关轨迹即时估计动作
优势（action advantage），并把优势以**加性方式直接调制输出 logits**，
是 KL 约束策略优化目标的精确闭式解。该方法免梯度更新，在 WebArena 上
超过微调基线 WebRL（arXiv:2411.02337）而成本降低 30× 以上。

与 SimplCore 的结构同构（为什么这是"融入"而不是"外挂"）
------------------------------------------------------
JitRL 的三个组件在本架构中各有原生对应，且有一处**比原论文更干净**：

1. **经验记忆** ↔ SMM 情景记忆（`memory.py` 的 CMT 库 / `smm.py`
   `experience()`）。本模块提供轻量独立的 :class:`TrajectoryMemory`
   （轨迹级、动作粒度），与 CMT 互补：CMT 存"知识条目"，这里存
   "(状态, 动作, 结果)" 三元组。
2. **即时优势估计** ↔ O(1) 流式检索。检索是纯字典/余弦相似度运算，
   复杂度与记忆条数成线性、与上下文长度无关——与 O(1) 流式推理
   的成本结构一致。
3. **logits 加性调制** ↔ 原 JitRL 调制的是**外部 LLM API** 的 logits
   （借 logit-bias 接口，受 API 能力限制）；SimplCore 的 logits 出自
   自家 `head`，调制可以做成**一等结构化通路**：`logits + alpha·bias`
   一行，且在模型内部，无需任何 API hack。

与 Reflexion（arXiv:2303.11366）的关系：Reflexion 把失败经验写成语义
反思放 episodic memory（**提示端**，改 prompt）；JitRL 把轨迹优势写成
logit 偏置（**输出端**，改分布）；SimplCore 的 L2 证据门把前缀证据写
成路径权重（**mixer 端**，改混合）。三者是"经验约束生成"在三个位置
的实例化，本模块补齐输出端，与 Dreaming 的 REFLECT 通道（提示端）
正交叠加。

理论口径（诚实声明）
--------------------
* JitRL 的"加性更新 = KL 约束策略优化闭式解"证明属于原论文（其
  假设：优势估计来自检索、温度受 alpha 控制）。本模块**复用其结论**
  并保持同一更新形式，不重述证明。
* **全部是机制接线，无环境实测**：WebArena/Jericho 数字属于原论文的
  外部 API 模型（GPT-4 级 agent）。SimplCore 主干在 30M–2B 量级，
  在任何环境基准上的效果**未验证**——可证伪判据见论文 §7.4（W4），
  评测环境为 WebArena（arXiv:2307.13854）。
* 默认关闭：``alpha=0`` 时所有路径与未接入**逐位一致**（测试锁定），
  经验记忆为空时优势恒 0——无监督缺失即空转的问题（与 L2/L3 闸门
  同一条纪律：调制必须有经验来源才生效）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "TrajectoryMemory", "AdvantageEstimator", "advantage_logit_bias",
    "JitRLController",
]


# ---------------------------------------------------------------------------
# 非参数经验轨迹记忆
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    """单步转移：任务/状态表征、动作（其首 token 的词表 id）、结果奖励。"""
    task_key: Tuple[float, ...]      # 任务/状态表征（嵌入向量，量化后可哈希）
    action_token: int                # 动作首 token 的词表 id（调制落点）
    reward: float                    # 结果奖励（ORM/pass@k/环境判定，0/1 或连续）
    sim_extra: float = 0.0           # 预留：额外相似度信息


@dataclass
class TrajectoryMemory:
    """非参数经验轨迹记忆：``task_key → [Transition]``。

    与 CMT 的分工：CMT 存压缩后的知识条目（语义记忆），这里存原始
    (状态, 动作, 奖励) 三元组（情景/轨迹记忆）——JitRL 的优势估计需要
    **逐动作的结果统计**，压缩会丢掉动作粒度。容量上限 + 最旧淘汰
    （LRU 语义按插入序，与 CMT 的定容原则一致）。
    """

    capacity: int = 100_000
    transitions: List[Transition] = field(default_factory=list)
    _index: Dict[Tuple[float, ...], List[int]] = field(default_factory=dict)

    def add(self, transition: Transition) -> None:
        """记录一条转移；超容量时淘汰最早插入的条目。"""
        if len(self.transitions) >= self.capacity:
            evicted = self.transitions.pop(0)
            bucket = self._index.get(evicted.task_key)
            if bucket is not None:
                # 惰性清理：桶内索引失效由检索端的下界过滤兜底
                bucket[:] = [i - 1 for i in bucket if i > 0]
        idx = len(self.transitions)
        self.transitions.append(transition)
        self._index.setdefault(transition.task_key, []).append(idx)

    def recent(self, limit: int = 0) -> List[Transition]:
        """全部（或最近 ``limit`` 条）转移，供检索器线性扫描。"""
        if limit and limit < len(self.transitions):
            return self.transitions[-limit:]
        return self.transitions

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# 即时优势估计
# ---------------------------------------------------------------------------
class AdvantageEstimator:
    """检索式动作优势：``A(a|s) = Σ_i sim_i·r_i / (Σ_i sim_i + eps)``。

    ``sim_i`` 是当前任务表征与经验轨迹表征的余弦相似度（clip 到
    ``[0, 1]``——只取正相关经验，负相关的轨迹不反向投票，这是 JitRL
    检索优势的保守读法）。``r_i`` 是该轨迹中该动作的结果奖励。记忆为
    空或无相关经验时优势恒 0 —— 无经验即无主张。

    该估计 + 加性 logit 更新 ``logits += alpha·bias`` 即 JitRL 的
    KL 约束策略优化闭式解的更新形式（证明见原论文；本模块保持同一
    形式，``alpha`` 对应其温度参数）。
    """

    def __init__(self, memory: TrajectoryMemory, eps: float = 1e-6,
                 sim_floor: float = 0.0) -> None:
        self.memory = memory
        self.eps = eps
        self.sim_floor = sim_floor

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return num / (na * nb)

    def advantage(self, task_key: Sequence[float],
                  action_token: int) -> float:
        """动作 ``action_token`` 在任务 ``task_key`` 下的检索优势 ∈ [0, 1]。"""
        num = 0.0
        den = 0.0
        for tr in self.memory.transitions:
            if tr.action_token != action_token:
                continue
            sim = self._cosine(task_key, tr.task_key)
            if sim <= self.sim_floor:
                continue
            sim = max(0.0, min(1.0, sim))
            num += sim * tr.reward
            den += sim
        if den <= 0.0:
            return 0.0
        return num / (den + self.eps)

    def action_advantages(self, task_key: Sequence[float],
                          candidate_tokens: Sequence[int]) -> Dict[int, float]:
        """一组候选动作的优势字典（只含记忆中出现过的动作；未出现的
        动作优势为 0，调用方按需补零）。"""
        return {a: self.advantage(task_key, a) for a in set(candidate_tokens)}


# ---------------------------------------------------------------------------
# logit 调制（JitRL 的加性更新）
# ---------------------------------------------------------------------------
def advantage_logit_bias(advantages: Dict[int, float],
                         vocab_size: int, alpha: float) -> Optional[torch.Tensor]:
    """把动作优势编译成词表级 logit 偏置 ``(V,)``（fp32）。

    ``bias[v] = alpha · A(v)``；优势为 0 的动作（含词表里无经验的）偏置
    为 0。返回 None 当 ``alpha == 0`` 或优势全零 —— 调用方跳过加法，
    与未接入逐位一致。``alpha`` 的角色是 JitRL 的温度：越大越信任检索
    优势，KL 距离越远。
    """
    if alpha == 0.0 or not advantages:
        return None
    bias = torch.zeros(vocab_size, dtype=torch.float32)
    touched = False
    for token_id, adv in advantages.items():
        if 0 <= token_id < vocab_size and adv != 0.0:
            bias[token_id] = alpha * adv
            touched = True
    return bias if touched else None


def apply_logit_bias(logits: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
    """对最后一维 logits 加偏置（广播到 batch/时间维）。

    ``bias is None`` 时原样返回（同一张量对象，不复制）——恒等路径。
    """
    if bias is None:
        return logits
    return logits + bias.to(device=logits.device, dtype=logits.dtype)


# ---------------------------------------------------------------------------
# 控制器：把 记录→估计→调制 串成 agent 循环的最小骨架
# ---------------------------------------------------------------------------
class JitRLController:
    """JitRL 循环控制器：环境奖励回填记忆 + 下次任务检索调制。

    用法（与 ``envlearn.py`` 的交互循环对齐）::

        ctrl = JitRLController(vocab_size, alpha=2.0)
        bias = ctrl.bias_for(task_key)            # 检索优势 → logit 偏置
        logits = apply_logit_bias(model_logits, bias)
        ... agent 行动，环境给出 reward ...
        ctrl.record(task_key, action_token, reward)   # 回填记忆

    ``alpha=0`` 时 ``bias_for`` 恒 None：整个控制器退化为纯记录器，
    生成行为与未接入逐位一致。
    """

    def __init__(self, vocab_size: int, alpha: float = 0.0,
                 capacity: int = 100_000, eps: float = 1e-6) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.memory = TrajectoryMemory(capacity=capacity)
        self.estimator = AdvantageEstimator(self.memory, eps=eps)

    def bias_for(self, task_key: Sequence[float],
                 candidate_tokens: Optional[Sequence[int]] = None
                 ) -> Optional[torch.Tensor]:
        """当前任务的优势偏置；``alpha=0`` 或无经验时 None（恒等）。"""
        if self.alpha == 0.0 or len(self.memory) == 0:
            return None
        if candidate_tokens is None:
            candidate_tokens = sorted({
                tr.action_token for tr in self.memory.transitions})
        advs = self.estimator.action_advantages(task_key, candidate_tokens)
        return advantage_logit_bias(advs, self.vocab_size, self.alpha)

    def record(self, task_key: Sequence[float], action_token: int,
               reward: float) -> None:
        """环境判定回填：VERIFIED 成功 → reward=1；失败 → 0（ORM 语义，
        与 ``envlearn``/``code_exec`` 的 verifier 对齐——只信独立判定，
        不信自报）。"""
        self.memory.add(Transition(
            task_key=tuple(float(v) for v in task_key),
            action_token=int(action_token),
            reward=float(reward)))