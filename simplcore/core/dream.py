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
本模块**不保证**自进化会提升能力。默认的反思器做的是**文本层可归因性检查**
（:mod:`simplcore.core.grounding`），不是语义批判：

* 它能发现"这句话在证据里找不到出处"（含数字篡改、否定翻转、跨文档拼接）；
* 它**不能**发现逐字引用但语境误用（反讽、条件句、引述他人观点）；
* 它对语义等价改写偏保守（会把正确改写判成无依据）。

因此 ``synthesized == 0`` 是一个**有意义的结果**：它意味着本轮所有发现都没有
可引用的原文，此时正确行为是补检索，而不是拿模型自评去训练自己。
在没有带引用的独立评测集上实测出提升之前，**不得宣称"自进化有效"**。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .grounding import EvidenceIndex, split_claims
from .memory import TextKeyEncoder


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
    """一条反思洞见。

    ``repair`` 是**证据原文**（不是模型自己写的修订）：只有存在可引用的
    外部片段时才有值。合成器只接受带 ``repair`` 的洞见，这样"自进化"产生的
    训练信号永远可以追回一句人类可核对的原文，而不是模型的自我加冕。
    """
    text: str
    source: Optional[Trajectory] = None
    weight: float = 1.0
    kind: str = "flagged"                 # flagged | unsupported | partially_supported
    repair: Optional[str] = None          # 外部证据原文（训练目标候选）
    evidence: Optional[str] = None        # 被引用的证据片段
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
    #: 巩固**阶段**是否顺利完成（不含"是否真的写入了东西"，看下面两个字段）
    consolidated: bool = False
    #: 本轮是否真的改了模型权重
    wrote_weights: bool = False
    #: 本轮往记忆库写了多少条
    wrote_memory: int = 0
    rolled_back: bool = False
    baseline_score: Optional[float] = None
    new_score: Optional[float] = None
    score_delta: Optional[float] = None
    # 多指标 + 置信下界（no-regression gate 升级）
    metrics_base: Dict[str, float] = field(default_factory=dict)
    metrics_new: Dict[str, float] = field(default_factory=dict)
    ci_lower: Dict[str, float] = field(default_factory=dict)
    failed_metrics: List[str] = field(default_factory=list)
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
            "wrote_weights": self.wrote_weights,
            "wrote_memory": self.wrote_memory,
            "rolled_back": self.rolled_back,
            "baseline_score": self.baseline_score,
            "new_score": self.new_score,
            "score_delta": self.score_delta,
            "metrics_base": self.metrics_base,
            "metrics_new": self.metrics_new,
            "ci_lower": self.ci_lower,
            "failed_metrics": list(self.failed_metrics),
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
# 评测分数容器 + 配对 bootstrap（no-regression gate 的判定基础）
# ---------------------------------------------------------------------------
@dataclass
class EvalScores:
    """一次 held-out 评测的结果：**逐样本**分数，而不是一个均值。

    为什么不能只留均值
    ------------------
    旧实现的 gate 比较两个标量：``new - base < -tolerance`` 就回滚。
    在小评测集上，同一份权重换个随机种子重跑，均值的抖动常常就有几个百分点，
    于是 gate 变成**噪声判定器**：真退化被放过（抖动掩盖），或者一次无关波动
    被当成退化而丢弃了有效巩固。用逐样本分数做**配对 bootstrap** 才能给出
    "delta 的置信下界"，把判定建立在分布上而不是单点上。

    ``metrics`` 的键是指标名（``grounding_f1`` / ``abstain_auroc`` / …），
    值是**同一条样本上**的分数序列。配对 bootstrap 要求 base/new 的样本
    一一对应，因此两者长度取 ``min`` 截断（顺序必须一致 —— 由调用方保证）。
    """

    metrics: Dict[str, List[float]] = field(default_factory=dict)
    #: 非空表示"逐样本分布缺失、已降级"，gate 会把它写进 report.notes
    degraded: str = ""

    @classmethod
    def from_any(cls, value: Any) -> "EvalScores":
        """把 verifier 的任意合法返回归一成本类。

        兼容三种返回：
        * ``float``           —— 单点分数；
        * ``Dict[str, float]``—— 多指标点估计；
        * ``Dict[str, Sequence[float]]`` —— 多指标逐样本分数（**推荐**）。
        """
        if isinstance(value, EvalScores):
            return value
        if isinstance(value, bool):                       # bool 是 int 的子类，先挡掉
            return cls({"score": [float(value)]}, degraded="布尔返回值，仅当作单点分数")
        if isinstance(value, (int, float)):
            return cls({"score": [float(value)]}, degraded="单点分数，无逐样本分布")
        if isinstance(value, dict):
            metrics: Dict[str, List[float]] = {}
            degraded = ""
            for key, val in value.items():
                name = str(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    metrics[name] = [float(val)]
                    degraded = degraded or "部分指标为点估计，其置信下界等于点估计"
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    metrics[name] = [float(x) for x in val]
            if not metrics:
                raise TypeError("verifier 返回的空字典无法用于 no-regression gate")
            return cls(metrics, degraded=degraded)
        raise TypeError(
            f"verifier 必须返回 float / Dict[str, float|Sequence[float]] / EvalScores，"
            f"收到 {type(value).__name__}"
        )

    # -- 视图 ---------------------------------------------------------------
    def aggregate(self) -> Dict[str, float]:
        """每个指标的均值（报告用；判定不用它）。"""
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self.metrics.items()}

    def primary(self, prefer: str = "score") -> Optional[float]:
        """单个代表值，供 ``baseline_score`` / ``new_score`` 这类旧字段使用。"""
        if prefer in self.metrics and self.metrics[prefer]:
            v = self.metrics[prefer]
            return sum(v) / len(v)
        if not self.metrics:
            return None
        first = sorted(self.metrics)[0]
        v = self.metrics[first]
        return sum(v) / len(v) if v else None

    def __len__(self) -> int:
        return max((len(v) for v in self.metrics.values()), default=0)


def paired_bootstrap_ci(
    base: Sequence[float],
    new: Sequence[float],
    level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """配对 bootstrap 的 ``(下界, 上界)``，针对 ``mean(new) - mean(base)``。

    配对（paired）意味着第 i 条样本在 base/new 中是**同一条**，重采样时
    两个分数一起被抽中 —— 这样样本难度带来的方差会相互抵消，只留下
    "同一条样本上新旧权重的差"，判定灵敏度远高于比较两个独立均值。

    长度不等时按 ``min`` 截断（顺序对齐由调用方保证）。
    """
    b = torch.as_tensor(list(base), dtype=torch.float64)
    n = torch.as_tensor(list(new), dtype=torch.float64)
    m = min(int(b.numel()), int(n.numel()))
    if m == 0:
        return (float("nan"), float("nan"))
    b, n = b[:m], n[:m]
    delta = n - b                                        # (m,) 逐样本差
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.randint(0, m, (max(int(n_boot), 1), m), generator=g)
    boot = delta[idx].mean(dim=1)                        # (n_boot,)
    alpha = (1.0 - float(level)) / 2.0
    lower = float(torch.quantile(boot, alpha))
    upper = float(torch.quantile(boot, 1.0 - alpha))
    return (lower, upper)


@dataclass
class GateVerdict:
    """no-regression gate 的判定结果。"""

    passed: bool
    metrics_base: Dict[str, float] = field(default_factory=dict)
    metrics_new: Dict[str, float] = field(default_factory=dict)
    ci_lower: Dict[str, float] = field(default_factory=dict)
    failed_metrics: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def score_delta(self) -> Optional[float]:
        """主指标的点估计差（兼容旧字段）。"""
        keys = sorted(set(self.metrics_base) & set(self.metrics_new))
        if not keys:
            return None
        k = "score" if "score" in keys else keys[0]
        return self.metrics_new[k] - self.metrics_base[k]


def regression_gate(
    base: EvalScores,
    new: EvalScores,
    tolerance: float = 0.0,
    ci_level: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
) -> GateVerdict:
    """多指标 + 置信下界的退化判定。

    规则（每条都是硬约束）：

    1. 对**两边都出现**的每个指标，算配对 bootstrap 的 delta 下界；
    2. 下界低于 ``-|tolerance|`` 即该指标失守；
    3. **任一**指标失守 → 整轮判定为退化（不允许"总分涨了就抵消某项崩掉"）；
    4. 只有一边出现的指标不参与判定（写成 note，不静默丢弃）。

    与旧实现的差别：旧实现只看一个标量点估计，且要求"总分不降"；
    这里允许主指标微降而某项大涨的情形依然通过（只要没有任何指标的
    **置信下界**跌破容差），这正是"不退化"的正确定义。
    """
    tol = -abs(float(tolerance))
    verdict = GateVerdict(passed=True)
    common = sorted(set(base.metrics) & set(new.metrics))
    if not common:
        verdict.passed = False
        verdict.notes.append("base/new 无共同指标，无法判定（按退化处理）")
        return verdict
    for key in common:
        b_vals, n_vals = base.metrics[key], new.metrics[key]
        if not b_vals or not n_vals:
            verdict.notes.append(f"指标 {key} 无样本，跳过")
            continue
        b_agg = sum(b_vals) / len(b_vals)
        n_agg = sum(n_vals) / len(n_vals)
        lower, _upper = paired_bootstrap_ci(b_vals, n_vals, ci_level, n_boot, seed)
        verdict.metrics_base[key] = b_agg
        verdict.metrics_new[key] = n_agg
        verdict.ci_lower[key] = lower
        if not (lower == lower):                          # NaN
            verdict.notes.append(f"指标 {key} 置信区间不可用，跳过")
            verdict.ci_lower.pop(key, None)
            continue
        if lower < tol:
            verdict.failed_metrics.append(key)
            verdict.passed = False
            verdict.notes.append(
                f"指标 {key} 退化：{n_agg:.4f} vs {b_agg:.4f}，"
                f"delta 的 {ci_level:.0%} 下界 {lower:.4f} < {tol:.4f}"
            )
    for key in sorted(set(base.metrics) - set(new.metrics)):
        verdict.notes.append(f"指标 {key} 仅出现在 baseline，未参与判定")
    for key in sorted(set(new.metrics) - set(base.metrics)):
        verdict.notes.append(f"指标 {key} 仅出现在巩固后评测，未参与判定")
    if base.degraded:
        verdict.notes.append(f"baseline 评测降级：{base.degraded}")
    if new.degraded:
        verdict.notes.append(f"巩固后评测降级：{new.degraded}")
    return verdict


# ---------------------------------------------------------------------------
# 默认钩子
# ---------------------------------------------------------------------------
#: 轨迹 meta 里可能承载外部证据的键（按优先级）。
_EVIDENCE_META_KEYS = ("evidence", "context", "contexts", "retrieved",
                       "hits", "documents", "sources")


def _collect_evidence(traj: Trajectory) -> Tuple[str, str]:
    """从轨迹的 ``meta`` 里取出外部证据原文，返回 ``(拼接文本, 来源标识)``。

    支持三种常见投放方式（与 ``abstain.GroundedAnswerer`` 的入参一致）：

    * ``meta["context"] = "整段上下文"``；
    * ``meta["hits"] = [{"text": ..., "source": ...}, ...]``（检索命中）；
    * ``meta["evidence"] = ["片段1", "片段2"]``。

    **从 response 里反推证据是不允许的** —— 那会让"自我反思"退化成
    "把模型自己的话当成事实"，正是抗幻觉要防的那件事。
    """
    for key in _EVIDENCE_META_KEYS:
        raw = traj.meta.get(key)
        if raw is None:
            continue
        pieces: List[str] = []
        srcs: List[str] = []
        if isinstance(raw, str):
            if raw.strip():
                pieces.append(raw)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str):
                    if item.strip():
                        pieces.append(item)
                elif isinstance(item, dict):
                    txt = item.get("text") or item.get("content") or item.get("span")
                    if txt:
                        pieces.append(str(txt))
                    src = item.get("source") or item.get("doc_id") or item.get("id")
                    if src:
                        srcs.append(str(src))
                else:
                    txt = getattr(item, "text", None)
                    if txt:
                        pieces.append(str(txt))
                    src = getattr(item, "source", None)
                    if src:
                        srcs.append(str(src))
        if pieces:
            return "。".join(pieces), (",".join(srcs) if srcs else str(key))
    return "", ""


def default_reflector(
    model,
    trajectories: Sequence[Trajectory],
    min_repair_coverage: float = 0.30,
    max_claims: int = 16,
) -> List[Insight]:
    """默认反思器：**基于外部证据的逐句归因**，不调用模型自评。

    旧实现是把"低置信度 / 无证据"的轨迹打上标记就结束 —— 标记不是知识，
    所以合成器永远拿不到 ``repair``，``synthesized`` 恒为 0，整条自进化链
    是空转的（``docs/THEORETICAL_EDGE.md`` 里如实记着这一点）。

    本实现换了一条不需要外部批判者模型的路：**用证据绑定做归因**
    （:mod:`simplcore.core.grounding`），逐句判定响应里哪些话在证据里找得到。
    ``repair`` 一律取**证据原文**，绝不取模型自己的改写 —— 这样"自进化"
    产出的每一条训练目标都能追回一句人类可以核对的原文。

    三种洞见
    --------
    * ``no_evidence``        —— 轨迹没有任何外部证据可用。**不产 repair**
      （无依据可修），只作标记，用于统计"该补检索"的比例；
    * ``partially_supported``—— 句子与某条证据高度重叠但**未过判据**
      （典型：数字被篡改、否定被翻转、跨文档拼接）。``repair`` = 那条证据原文，
      这是最有价值的信号：错在细节，而不是全无依据；
    * ``unsupported``        —— 证据里找不到对应片段。**不产 repair**
      （没有原文可给），只作标记；
    * ``corrected``          —— 轨迹自带 ``meta['corrected_response']``
      （来自外部批判-修订链路），原样作为目标。

    Args:
        min_repair_coverage: 低于此覆盖率视为"证据里根本没这句话"，
            不给 repair（避免把不相干的证据硬塞成训练目标）。
        max_claims: 每条轨迹最多核验的句子数（预算硬约束）。
    """
    out: List[Insight] = []
    seen: set = set()
    for traj in trajectories:
        # ① 外部修订优先：它已是完整响应，不像 claim 那样需要回填
        corrected = traj.meta.get("corrected_response")
        if corrected:
            out.append(Insight(
                text=f"外部修订：{traj.prompt[:60]}",
                source=traj,
                weight=1.0 + max(0.0, 1.0 - float(traj.confidence)),
                kind="corrected",
                repair=str(corrected),
                evidence=str(corrected),
                meta={"evidence_source": str(traj.meta.get("corrected_source", "external"))},
            ))
            continue

        ctx, src = _collect_evidence(traj)
        if not ctx:
            if (not traj.grounded) or traj.confidence < 0.5:
                out.append(Insight(
                    text=f"未获证据支撑或置信度不足：{traj.prompt[:60]}",
                    source=traj,
                    weight=(1.0 - float(traj.confidence)) + (0.0 if traj.grounded else 1.0),
                    kind="no_evidence",
                    meta={"evidence_source": ""},
                ))
            continue

        index = EvidenceIndex.from_text(ctx, source=src or "trajectory.evidence")
        if not index.spans:
            continue
        bindings = index.bind_all(split_claims(traj.response, max_claims=max_claims))
        for b in bindings:
            if b.supported:
                continue
            if b.span is not None and b.coverage >= float(min_repair_coverage):
                kind = "partially_supported"
                repair: Optional[str] = b.span.text
            else:
                kind = "unsupported"
                repair = None
            dedup_key = (kind, repair or b.claim.text)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # 严重度：越不像证据越该反思；数字篡改与否定翻转额外加权
            sev = max(1.0 - float(b.coverage), 0.0)
            if not b.numbers_ok:
                sev += 0.5
            if not b.polarity_ok:
                sev += 0.25
            out.append(Insight(
                text=f"[{kind}] 句子未获证据支撑：{b.claim.text[:60]}",
                source=traj,
                weight=max(sev, 0.05),
                kind=kind,
                repair=repair,
                evidence=(b.span.text if b.span is not None else None),
                meta={
                    "claim": b.claim.text,
                    "coverage": float(b.coverage),
                    "ngram": float(b.ngram),
                    "numbers_ok": bool(b.numbers_ok),
                    "polarity_ok": bool(b.polarity_ok),
                    "span_id": (b.span.span_id if b.span is not None else None),
                    "evidence_source": (b.span.source if b.span is not None else src),
                },
            ))
    return out


def _replace_claim(response: str, claim: str, repair: str) -> str:
    """把响应里那句不可归因的话替换成证据原文，得到"修订后的响应"。

    直接拿一条证据片段当训练目标会丢掉整段回答的语境（目标变成一句孤立事实），
    自蒸馏时学到的就不是"怎么答"而是"怎么抄"。因此这里保留原响应结构，
    只替换出错的那一句 —— 这是**最小编辑**的修订目标。
    """
    claim = (claim or "").strip()
    if claim and response and claim in response:
        return response.replace(claim, repair, 1)
    if not (response or "").strip():
        return repair
    # 找不到原句（响应被归一化/截断过）：不硬改写，退回证据原文并在 meta 标注
    return repair


def default_synthesizer(model, insights: Sequence[Insight]) -> List[TrainExample]:
    """默认合成器：**只接受带 ``repair``（证据原文）的洞见**。

    这是整条自进化链的诚实性闸门：
    ``repair is None`` 的洞见（``no_evidence`` / ``unsupported``）**一律丢弃** ——
    没有原文可给的时候，"修复它"只能靠模型自己编，那正是要防的东西。

    因此 ``synthesized == 0`` 是一个**有意义的结果**：它意味着本轮所有发现
    都没有可引用的原文，此时正确行为是补检索，而不是拿模型自评去训练自己。
    """
    out: List[TrainExample] = []
    seen: set = set()
    for ins in insights:
        traj = ins.source
        if traj is None or not ins.repair:
            continue
        repair = str(ins.repair).strip()
        if not repair:
            continue
        if ins.kind == "corrected":
            target = repair
        else:
            target = _replace_claim(traj.response, str(ins.meta.get("claim") or ""), repair)
        target = (target or "").strip()
        if not target:
            continue
        key = (traj.prompt, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(TrainExample(
            prompt=traj.prompt,
            target=target,
            weight=ins.weight,
            meta={
                "from_insight": ins.text,
                "kind": ins.kind,
                "claim": ins.meta.get("claim"),
                "evidence": ins.evidence,
                "evidence_source": ins.meta.get("evidence_source"),
                # 供下游审计：目标里的每个事实都应能在 evidence 里查到
                "from_evidence": ins.evidence is not None,
            },
        ))
    return out


def _encode_ids(codec, text: str) -> List[int]:
    """把文本编码成 token id 列表（兼容返回原生 list / HF Encoding 的编解码器）。"""
    if codec is None or not text:
        return []
    out = codec.encode(text)
    if out is None:
        return []
    if isinstance(out, int):
        return [out]
    try:
        return [int(i) for i in out]
    except TypeError:
        return []


def default_consolidator(
    model,
    examples: Sequence[TrainExample],
    *,
    codec=None,
    seq_len: int = 512,
    lr: float = 1e-5,
    epochs: int = 1,
    batch_size: int = 4,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    max_steps: int = 50,
    device=None,
    seed: int = 0,
) -> Dict[str, Any]:
    """默认巩固器：**把训练信号真正写进模型权重**。

    这是支柱 C（"去真实环境写代码 → 验证 → 写回权重"）里唯一缺的那一环。
    :class:`DreamingLoop` 负责决定**该不该写**（no-regression gate）与
    **怎么回滚**（``snapshot`` / ``restore``），但"怎么把样本变成梯度步"
    必须有人做 —— 此前只有调用方自备 ``consolidator``，没有内置实现，
    于是"写回权重"这条链路在默认配置下是断的。

    关键约定
    --------
    * **标签错位一格**。``SimplCoreModel`` 的 loss 是 ``CE(logits[t], targets[t])``
      （模型内部**不做** shift），因此标签必须自己错位：
      ``labels[t] = ids[t+1]``，且只在**目标段**计损失（提示段一律 ``-100``）。
    * **不对一维参数做 weight decay**。振荡基参数（``raw_rho`` / ``raw_theta``）、
      norm、bias、``feed``、``out_scale`` 都是一维或标量；给它们加衰减会直接
      掐死长程记忆（与 ``train.py`` 的同一条纪律）。
    * **样本权重生效**。``TrainExample.weight`` 按逐样本加权，而不是被平均掉。

    Args:
        model: 目标模型（``SimplCoreModel`` 或同接口的 ``nn.Module``）。
        examples: 训练样本；``meta["input_ids"]`` 里若已带编码结果则优先使用。
        codec: 编解码器，需有 ``encode(text) -> list[int]``；
            不传时样本必须自带 ``meta["input_ids"]``。
        seq_len: 截断长度。
        lr: 学习率（**巩固是小步微调**，默认远小于常规 SFT）。
        epochs / batch_size / max_steps: 训练规模上限。
        weight_decay: 权重衰减（只作用于二维及以上参数）。
        grad_clip: 梯度裁剪阈值。
        device: 训练设备；``None`` 表示跟随模型。
        seed: 随机种子（决定 batch 顺序）。

    Returns:
        统计字典：``steps`` / ``loss_before`` / ``loss_after`` / ``examples`` /
        ``tokens`` / ``lr`` / ``grad_norm``。
    """
    import torch.nn.functional as F

    if model is None:
        raise ValueError("default_consolidator 需要一个模型")
    ex_list = list(examples or [])
    if not ex_list:
        return {"steps": 0, "examples": 0, "tokens": 0,
                "loss_before": None, "loss_after": None,
                "lr": lr, "grad_norm": None}

    # ---- 1. 构造 (ids, labels, weight) -----------------------------------
    samples: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    for ex in ex_list:
        ids = ex.meta.get("input_ids") if isinstance(ex.meta, dict) else None
        if ids is None:
            if codec is None:
                raise ValueError(
                    "default_consolidator 需要 codec 来编码样本，或让样本在 "
                    "meta['input_ids'] 里带上已编码的 id"
                )
            p_ids = _encode_ids(codec, ex.prompt)
            t_ids = _encode_ids(codec, ex.target)
            ids = p_ids + t_ids
            prompt_len = len(p_ids)
        else:
            ids = [int(i) for i in ids]
            prompt_len = int((ex.meta or {}).get("prompt_len", 0))
        ids = ids[:seq_len]
        if len(ids) < 2:
            continue
        # labels[t] = ids[t+1]；只从"提示段最后一位"开始计目标 token 的损失
        labels = [-100] * len(ids)
        for t in range(max(prompt_len - 1, 0), len(ids) - 1):
            labels[t] = ids[t + 1]
        if all(v == -100 for v in labels):
            continue                       # 没有目标 token，这条样本学不到东西
        samples.append((torch.tensor(ids, dtype=torch.long),
                        torch.tensor(labels, dtype=torch.long),
                        float(ex.weight)))

    if not samples:
        return {"steps": 0, "examples": 0, "tokens": 0,
                "loss_before": None, "loss_after": None,
                "lr": lr, "grad_norm": None}

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev)

    # ---- 2. 优化器：只对二维及以上参数做 weight decay ----------------------
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(weight_decay)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(lr),
    )

    gen = torch.Generator().manual_seed(int(seed))
    order = list(range(len(samples)))
    was_training = bool(model.training)
    model.train()

    def _loss_on(batch_idx: List[int]) -> torch.Tensor:
        width = max(len(samples[i][0]) for i in batch_idx)
        ids = torch.zeros(len(batch_idx), width, dtype=torch.long)
        labels = torch.full((len(batch_idx), width), -100, dtype=torch.long)
        w = torch.zeros(len(batch_idx), dtype=torch.float32)
        for r, i in enumerate(batch_idx):
            x, y, wi = samples[i]
            ids[r, :x.numel()] = x
            labels[r, :y.numel()] = y
            w[r] = wi
        ids, labels, w = ids.to(dev), labels.to(dev), w.to(dev)
        logits = model(ids)
        V = logits.size(-1)
        ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                             ignore_index=-100, reduction="none").view(labels.shape)
        mask = (labels != -100).to(ce.dtype)
        per_sample = (ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-6), int(mask.sum().item())

    # ---- 3. 训练循环 ------------------------------------------------------
    steps = 0
    grad_norm = None
    loss_before = None
    try:
        with torch.no_grad():
            loss_before, _ = _loss_on(order[:max(1, min(len(order), batch_size))])
        for _ep in range(max(1, int(epochs))):
            perm = torch.randperm(len(order), generator=gen).tolist()
            for s in range(0, len(perm), max(1, int(batch_size))):
                if steps >= max_steps:
                    break
                batch = perm[s:s + max(1, int(batch_size))]
                opt.zero_grad(set_to_none=True)
                loss, _ = _loss_on(batch)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip) if grad_clip else float("inf"))
                grad_norm = float(gn)
                opt.step()
                steps += 1
            if steps >= max_steps:
                break
    finally:
        if not was_training:
            model.eval()

    with torch.no_grad():
        loss_after, tokens = _loss_on(order[:max(1, min(len(order), batch_size))])

    return {
        "steps": steps,
        "examples": len(samples),
        "tokens": tokens,
        "loss_before": None if loss_before is None else float(loss_before),
        "loss_after": None if loss_after is None else float(loss_after),
        "lr": float(lr),
        "grad_norm": grad_norm,
    }


def default_verifier(model) -> float:
    """默认验证器：**恒定返回 0.0**，因此**不能**用于判定退化。

    它存在只是为了让 ``DreamingLoop`` 的签名完整。``DreamingLoop`` 会记住
    "调用方没给 verifier"，并在 ``mode in ("weights", "both")`` 时**拒绝写回
    权重**（见 :meth:`DreamingLoop.step` 的 VERIFY 段）—— 没有独立 held-out    评测集时让 gate 恒通过，等于把 no-regression gate 变成装饰品。

    生产实现应返回逐样本分数::

        lambda model: {"grounding_f1": [...每样本 f1...],
                       "abstain_auroc": [...每样本 auroc...]}
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
    # -- gate 的统计口径（多指标 + 配对 bootstrap） -------------------------
    ci_level: float = 0.95                # 置信水平
    bootstrap_samples: int = 1000         # bootstrap 重采样次数
    seed: int = 0
    # -- 反思/合成的预算 ----------------------------------------------------
    max_claims_per_trajectory: int = 16   # 每条轨迹最多核验的句子数
    min_repair_coverage: float = 0.30     # 低于此覆盖率不给 repair（不给证据原文）
    max_memory_writes: int = 256          # 每轮最多写入记忆库的条目数
    # -- 权重巩固（default_consolidator）的超参 -------------------------------
    consolidate_lr: float = 1e-5          # 巩固是小步微调，远小于常规 SFT
    consolidate_epochs: int = 1
    consolidate_batch_size: int = 4


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
        verifier: ``(model) -> EvalScores | Dict[str, float|Sequence[float]] | float``，
            越高越好。**传入逐样本分数**才能得到置信下界判定。
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
        codec=None,
    ) -> None:
        self.model = model
        self.buffer = buffer if buffer is not None else ExperienceBuffer()
        self.config = config or DreamConfig()
        # 默认钩子需要读 config（预算/阈值），自定义钩子保持原签名不变
        self.reflector = reflector or (
            lambda m, trajs: default_reflector(
                m, trajs,
                min_repair_coverage=self.config.min_repair_coverage,
                max_claims=self.config.max_claims_per_trajectory,
            )
        )
        self.synthesizer = synthesizer or default_synthesizer
        self.consolidator = consolidator
        #: 编解码器（``default_consolidator`` 需要它把文本编码成 token id）
        self.codec = codec
        #: 调用方没给 verifier 时，权重写回必须被拒绝（见 step 的 VERIFY 段）
        self.verifier_is_default = verifier is None
        self.verifier = verifier or default_verifier
        self.memory_bank = memory_bank
        self.history: List[DreamReport] = []
        self._round = 0
        self._encoder = None                       # 延迟构造的 TextKeyEncoder
        self._harvested: List[Trajectory] = []     # 本轮采集到的轨迹（记忆写入用）

    # -- 基线 ---------------------------------------------------------------
    def evaluate(self, rep: Optional[DreamReport] = None) -> Optional[EvalScores]:
        """跑一次 held-out 评测，返回逐样本分数容器。

        评测器异常/返回不可用结构时返回 ``None`` 并记录原因 —— 绝不用
        "常量 0.0" 冒充有效评测（那会让 gate 恒通过）。
        """
        try:
            return EvalScores.from_any(self.verifier(self.model))
        except Exception as exc:                      # noqa: BLE001
            self._note(f"评测失败：{exc}", rep)
            return None

    def baseline(self) -> Optional[float]:
        """当前模型的基线分数（单值视图，保留给旧调用方）。

        多指标情形下取 ``score`` 或字典序第一个指标的均值；做退化判定请用
        :meth:`evaluate` + :func:`regression_gate`，那才有多指标与置信下界。
        """
        if not self.config.require_regression_check:
            return None
        scores = self.evaluate()
        return None if scores is None else scores.primary()

    # -- 快照 / 回滚 --------------------------------------------------------
    @staticmethod
    def _examples_have_ids(examples: Sequence[TrainExample]) -> bool:
        """样本是否自带已编码的 ``meta["input_ids"]``（自带就不需要 codec）。"""
        for ex in examples:
            meta = getattr(ex, "meta", None)
            if isinstance(meta, dict) and meta.get("input_ids") is not None:
                return True
        return False

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

    def memory_snapshot(self) -> Optional[Any]:
        """记忆库快照。

        回滚必须**同时**回滚权重与记忆：只回滚权重会把"这一轮学错的东西"
        留在记忆库里，下一轮继续被检索注入 —— 于是"回滚"看起来生效了，
        错误却还在。
        """
        bank = self.memory_bank
        if bank is None:
            return None
        try:
            return bank.state_dict()
        except Exception:                             # noqa: BLE001
            return None

    def memory_restore(self, snap: Optional[Any]) -> bool:
        """回滚记忆库。"""
        bank = self.memory_bank
        if bank is None or snap is None:
            return False
        try:
            bank.load_state_dict(snap)
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
        self._harvested = list(trajs)
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

        # 记忆巩固与权重巩固是两件事：前者不需要训练样本
        # （"把这次查到的证据存进库"和"这次答错了要不要改权重"无关）。
        # 旧实现把两者绑在同一个 if 上，于是"全部答对"的一轮什么都不写，
        # 记忆库永远是空的 —— 这正是"记忆写了但从不被召回"的另一半原因。
        memory_work = cfg.mode in ("memory", "both")
        do_memory = memory_work and self.memory_bank is not None
        weight_work = (cfg.mode in ("weights", "both") and bool(examples))

        if cfg.mode == "none" or not (do_memory or weight_work or examples):
            if not examples:
                if insights:
                    rep.notes.append(
                        f"{len(insights)} 条洞见均无外部证据原文（repair=None），"
                        f"不合成训练信号 —— 此时正确做法是补检索，而不是用自评训练自己"
                    )
                else:
                    rep.notes.append("无洞见（响应全部可归因或轨迹无证据）")
            rep.stage = DreamStage.IDLE
            rep.seconds = time.perf_counter() - t0
            self.history.append(rep)
            return rep

        # 4) CONSOLIDATE
        rep.stage = DreamStage.CONSOLIDATE
        snap = self.snapshot()
        mem_snap = self.memory_snapshot()
        wrote_weights = False
        try:
            if weight_work:
                # 没给 consolidator 时用**内置**的 default_consolidator ——
                # 此前这里直接跳过，于是报告写着 "consolidated=True"，
                # 而权重一个字节都没变：一份自称成功、实际没做事的报告。
                fn = self.consolidator
                if fn is None:
                    if self.codec is None and not self._examples_have_ids(examples):
                        raise ValueError(
                            "mode=weights 需要 consolidator，或提供 codec "
                            "（或让样本自带 meta['input_ids']）—— 没有它就无法"
                            "把训练信号写进权重"
                        )
                    fn = lambda m, ex: default_consolidator(       # noqa: E731
                        m, ex, codec=self.codec,
                        lr=self.config.consolidate_lr,
                        epochs=self.config.consolidate_epochs,
                        batch_size=self.config.consolidate_batch_size,
                    )
                stats = fn(self.model, examples) or {}
                rep.notes.append(
                    f"权重巩固：{stats.get('steps', '?')} 步，"
                    f"样本 {stats.get('examples', len(examples))}，"
                    f"loss {stats.get('loss_before')} → {stats.get('loss_after')}"
                )
                wrote_weights = True
            if do_memory:
                wrote = self._write_memory(insights, examples)
                rep.wrote_memory = int(wrote)
                rep.notes.append(f"记忆库写入 {wrote} 条（事实 + 修正）")
            elif memory_work:
                rep.notes.append("mode=memory 但未提供 memory_bank，记忆巩固被跳过")
            # ``consolidated`` 说的是**阶段**完成；"到底写了没有"看
            # ``wrote_weights`` / ``wrote_memory``。
            rep.consolidated = True
            rep.wrote_weights = wrote_weights
        except Exception as exc:                      # noqa: BLE001
            rep.stage = DreamStage.FAILED
            self.restore(snap)
            self.memory_restore(mem_snap)
            rep.seconds = time.perf_counter() - t0
            rep.notes.append(f"巩固阶段异常，已回滚：{exc}")
            self.history.append(rep)
            return rep

        # 5) VERIFY（no-regression gate：多指标 + 配对 bootstrap 置信下界）
        if cfg.require_regression_check and weight_work:
            rep.stage = DreamStage.VERIFY
            if self.verifier_is_default:
                # 没有独立评测器时让 gate 恒通过 = 没有 gate。直接拒绝写回权重。
                rep.rolled_back = True
                rep.stage = DreamStage.ROLLBACK
                self.restore(snap)
                self.memory_restore(mem_snap)
                rep.notes.append(
                    "未提供独立 held-out 评测器（verifier 为默认常量），"
                    "no-regression gate 无法判定退化，已拒绝写回权重"
                )
                rep.seconds = time.perf_counter() - t0
                self.history.append(rep)
                return rep

            base = self.evaluate(rep)
            new = None
            if base is not None:
                new = self.evaluate(rep)
            rep.baseline_score = None if base is None else base.primary()
            rep.new_score = None if new is None else new.primary()

            if base is None or new is None:
                # 评测不可用 → 不能声称"没退化"，按退化处理（保守）
                rep.rolled_back = True
                rep.stage = DreamStage.ROLLBACK
                self.restore(snap)
                self.memory_restore(mem_snap)
                rep.notes.append("评测不可用，无法证明未退化，已回滚")
                rep.seconds = time.perf_counter() - t0
                self.history.append(rep)
                return rep

            verdict = regression_gate(
                base, new,
                tolerance=cfg.regression_tolerance,
                ci_level=cfg.ci_level,
                n_boot=cfg.bootstrap_samples,
                seed=cfg.seed,
            )
            rep.metrics_base = dict(verdict.metrics_base)
            rep.metrics_new = dict(verdict.metrics_new)
            rep.ci_lower = dict(verdict.ci_lower)
            rep.failed_metrics = list(verdict.failed_metrics)
            delta = verdict.score_delta
            rep.score_delta = delta
            rep.notes.extend(verdict.notes)

            if not verdict.passed:
                rep.stage = DreamStage.ROLLBACK
                self.restore(snap)
                self.memory_restore(mem_snap)
                rep.rolled_back = True
                rep.notes.append(
                    f"退化指标 {', '.join(verdict.failed_metrics) or '(判定失败)'} "
                    f"超过容差 {abs(cfg.regression_tolerance):.4f}，已回滚权重与记忆"
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
    def _encoder_for(self, dim: int) -> "TextKeyEncoder":
        if self._encoder is None or getattr(self._encoder, "dim", None) != int(dim):
            self._encoder = TextKeyEncoder(int(dim))
        return self._encoder

    def _write_memory(self, insights: Sequence[Insight],
                      examples: Sequence[TrainExample]) -> int:
        """把本轮学到的东西写进 CMT 压缩记忆库（mode=memory 的巩固方式）。

        写两类条目，键空间一致（都是"将来会出现的查询"），值一律是**证据原文**：

        * ``knowledge`` —— 本轮被**证据支撑**的分句。这是记忆库存在的主要理由：
          只存"改过错的地方"会让库变成错题本，而不是知识库；线上新查询检索不到
          任何东西，``aggregate`` 返回零向量，注入等于没注入。
        * ``repair``    —— 被判定不可归因、但**找得到原文**的分句（``repair`` 非空）。
          它记的是"这种问法以前答错过，正确说法是这句原文"。

        键与值的分工
        ------------
        * **键**编码 ``source.prompt`` —— "将来什么样的查询应当召回这条记忆"，
          与线上 ``aggregate(query)`` 处同一空间，检索才对得上；
        * **值**编码证据原文。

        旧实现用 ``float(abs(hash(text)) % 1e8) / 1e8`` 填满整条向量：所有条目的键
        彼此共线（余弦恒为 1.0），检索退化成"永远返回库里任意一条"；且内置
        ``hash()`` 带进程随机盐，重启后同一文本的键就变了。这类缺陷不会让任何
        形状测试失败，只表现为"记忆写了但从不被召回"。
        """
        bank = self.memory_bank
        if bank is None:
            return 0
        dim = getattr(bank, "dim", None)
        if not dim:
            return 0
        enc = self._encoder_for(int(dim))
        cfg = self.config

        # (key_text, value_text, meta) 去重表；同一 (键, 值) 不重复写入
        entries: List[Tuple[str, str, Dict[str, Any]]] = []
        seen: set = set()

        def push(key_text: str, value_text: str, meta: Dict[str, Any]) -> None:
            key_text = (key_text or "").strip()
            value_text = (value_text or "").strip()
            if not key_text or not value_text:
                return
            sig = (key_text, value_text)
            if sig in seen:
                return
            seen.add(sig)
            entries.append((key_text, value_text, meta))

        # ① 事实记录：本轮轨迹里被证据支撑的分句
        for traj in self._harvested:
            if not (traj.grounded or traj.confidence >= 0.5):
                continue
            ctx, src = _collect_evidence(traj)
            if not ctx:
                continue
            index = EvidenceIndex.from_text(ctx, source=src or "trajectory.evidence")
            if not index.spans:
                continue
            spans: List[str] = []
            for b in index.bind_all(
                split_claims(traj.response, max_claims=cfg.max_claims_per_trajectory)
            ):
                if b.supported and b.span is not None and b.span.text not in spans:
                    spans.append(b.span.text)
            if spans:
                push(traj.prompt, "。".join(spans),
                     {"kind": "knowledge", "prompt": traj.prompt,
                      "evidence": "。".join(spans),
                      "evidence_source": src, "weight": 1.0,
                      "n_spans": len(spans), "from_evidence": True})

        # ② 修正记录：有证据原文的洞见
        for ins in insights:
            traj = ins.source
            key_text = ((traj.prompt if traj is not None else "") or ins.text or "")
            content = (ins.repair or ins.evidence or "")
            push(key_text, content, {
                "text": ins.text,
                "weight": float(ins.weight),
                "kind": ins.kind,
                "prompt": key_text.strip(),
                "evidence": content.strip(),
                "evidence_source": ins.meta.get("evidence_source", ""),
                "from_evidence": True,
            })

        if not entries:
            return 0
        entries = entries[: max(int(cfg.max_memory_writes), 1)]

        keys, values, metas = [], [], []
        for key_text, value_text, meta in entries:
            key = enc(key_text)
            if float(key.norm()) <= 0.0:
                # 归一化后为零向量（纯标点/空白提示）：写进去只会污染检索
                continue
            keys.append(key)
            values.append(enc(value_text))
            metas.append(meta)
        if keys:
            bank.add(torch.stack(keys), torch.stack(values), metas)
        return len(keys)

    def _note(self, msg: str, rep: Optional[DreamReport] = None) -> None:
        """记一条备注。``rep`` 未给时落到最近一条报告（step 早期阶段用）。"""
        target = rep if rep is not None else (self.history[-1] if self.history else None)
        if target is not None:
            target.notes.append(msg)
