# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""L4 拒答门 (abstention gate) —— 四层闸门的输出侧闸门。

四层闸门总览（见 ``docs/ARCHITECTURE_PIVOT.md`` §4）::

    L1 选择门   selective    mixer 输入    按内容选择参与位置/通道
    L2 证据门   evidence     mixer 谱基    前缀能量不足时压制长程路径
    L3 不确定门 uncertainty   mixer 输出    长短路径分歧大时回退到局部证据
    L4 拒答门   abstention   输出头        没有外部证据支撑时拒绝作答

L1~L3 作用在 mixer 内部（``simplcore/core/mixer.py``），L4 作用在输出侧：
从最终隐状态预测"该位置的回答是否被证据支撑"，低于阈值即拒答。

设计要点
--------
* **不引入 softmax 注意力**：L4 只是一个 ``Linear(dim -> 1)`` 的标量头。
* **默认不拒答**：偏置初始化为 -2.0（sigmoid≈0.12），未训练时几乎不会触发拒答，
  避免未训练模型行为突变。
* **监督契约**：``grounded`` 标签必须来自**外部证据**（引文、检索命中、
  工具返回值、人工标注），不能由模型自身生成 —— 自监督的"拒答标签"会退化成
  另一个自信度估计，四层闸门就白做了。
* **诚实边界**：四层闸门是**降低幻觉倾向的结构性约束**，不构成"消除幻觉"的保证。
  必须用带引用的评测集实测后才能宣称有效。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grounding import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_NGRAM,
    EvidenceIndex,
    attribution_coverage,
    split_claims,
)


class AbstentionHead(nn.Module):
    """Predicts whether the answer at each position is supported by evidence.

    Args:
        dim: hidden width.
        bias_init: initial bias of the logit. -2.0 gives sigmoid(-2) ~= 0.12,
            i.e. an untrained model almost never abstains.
    """

    def __init__(self, dim: int, bias_init: float = -2.0) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.full((1,), float(bias_init)))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Args: hidden (B, L, D). Returns: logit (B, L)."""
        return (self.proj(hidden) * torch.exp(self.logit_scale) + self.bias).squeeze(-1)

    def extra_repr(self) -> str:
        return f"dim={self.proj.in_features}, bias={float(self.bias):.2f}"


def abstention_loss(
    grounded_logit: torch.Tensor,
    grounded: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy for the grounded/abstain decision.

    Args:
        grounded_logit: (B, L) raw logits from :class:`AbstentionHead`.
        grounded: (B, L) float targets in [0, 1]. 1 = the answer at that
            position **is** supported by external evidence, 0 = it is not.
            Must come from external evidence, never from the model itself.
        weight: optional per-element loss weight.
        pos_weight: optional positive-class weight (for class imbalance).

    Returns:
        Scalar loss.
    """
    target = grounded.to(grounded_logit.dtype)
    if target.shape != grounded_logit.shape:
        raise ValueError(
            f"grounded shape {tuple(target.shape)} != logit shape "
            f"{tuple(grounded_logit.shape)}"
        )
    return F.binary_cross_entropy_with_logits(
        grounded_logit, target, weight=weight, pos_weight=pos_weight
    )


def grounded_probability(grounded_logit: torch.Tensor) -> torch.Tensor:
    """Sigmoid of the grounded logit: probability that the answer is supported."""
    return torch.sigmoid(grounded_logit)


def should_abstain(grounded_logit: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Boolean mask of positions that should be refused.

    A position abstains when the probability that it is grounded falls below
    ``threshold``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    return grounded_probability(grounded_logit) < threshold


def calibrate_temperature(logit: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling for confidence calibration (Guo et al., 2017).

    ``temperature < 1`` sharpens, ``> 1`` softens. Used at inference to
    correct over/under-confidence of the grounded head before thresholding.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logit / temperature


# ===========================================================================
# 证据感知拒答：四态决策 + 校准
# ===========================================================================
# 上面的 AbstentionHead 只给一个标量（隐状态 → 是否有依据）。但真实的拒答
# 决策需要三路信号，缺一不可：
#
#   grounded_prob      模型对自己"有没有依据"的判断（L4 头）
#   attribution_coverage 外部证据对输出的**可归因比例**（grounding.py，机械判据）
#   retrieval_confidence 检索本身是否真的命中了什么东西
#
# 只用第一路 → 退化成自信度；只用第二路 → 无法区分"没有证据"与"检索没做好"。
# 三者组合才能把拒答拆成四种**语义不同**的处置，而不是一个二元开关。
class AbstentionState(str, Enum):
    """四态处置。

    * ``ANSWER``  —— 证据充分、模型判有依据：正常作答；
    * ``HEDGE``   —— 部分归因：作答但必须带限定语（"据 X 所述…"），不得给断言；
    * ``CLARIFY`` —— 检索几乎没命中：问题不在"能不能答"，在"给的信息不够"，
                    应要求补充上下文，而不是假装不知道或硬答；
    * ``ABSTAIN`` —— 有检索命中但输出归因不上：**最危险的一类**，
                    上下文里明明有东西、模型说的却不是它。必须拒答。
    """

    ANSWER = "answer"
    HEDGE = "hedge"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass
class GroundedPolicy:
    """四态决策的阈值与权重（全部可外部覆盖，便于按场景校准）。"""

    tau_grounded: float = 0.50      # L4 头判"有依据"的概率阈值
    coverage_answer: float = 0.80   # 达到此归因覆盖率 → 允许直接作答
    coverage_hedge: float = 0.50    # 达到此覆盖率 → 允许带限定语作答
    tau_retrieval: float = 0.25     # 检索置信低于此 → 判为"信息不足"
    min_coverage: float = DEFAULT_MIN_COVERAGE  # 单句判 supported 的 LCS 阈值
    min_ngram: float = DEFAULT_MIN_NGRAM        # 单句判 supported 的二元组阈值
    w_grounded: float = 0.45
    w_coverage: float = 0.45
    w_retrieval: float = 0.10
    # 只要回答里有**任何一句**归因不上，就禁止给出无保留断言（降级为 HEDGE）。
    # 理由是长度加权覆盖率会被长真句稀释：两真一假时覆盖率仍可能 > 0.8，
    # 但那句假话恰恰是读者会记住的。安全侧一律按"最少保证"处理。
    cap_on_unsupported: bool = True

    def validate(self) -> None:
        if not 0.0 < self.tau_grounded < 1.0:
            raise ValueError("tau_grounded must be in (0,1)")
        if not 0.0 <= self.coverage_hedge <= self.coverage_answer <= 1.0:
            raise ValueError("require coverage_hedge <= coverage_answer in [0,1]")
        if not 0.0 <= self.tau_retrieval <= 1.0:
            raise ValueError("tau_retrieval must be in [0,1]")


@dataclass
class AbstentionDecision:
    """一次拒答决策的完整记录（可审计、可 JSON 化）。"""

    state: AbstentionState
    score: float
    grounded_prob: float
    coverage: float
    retrieval_confidence: float
    reason: str = ""
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "score": round(self.score, 4),
            "grounded_prob": round(self.grounded_prob, 4),
            "coverage": round(self.coverage, 4),
            "retrieval_confidence": round(self.retrieval_confidence, 4),
            "reason": self.reason,
            "message": self.message,
        }


_DEFAULT_MESSAGES = {
    AbstentionState.ANSWER: "",
    AbstentionState.HEDGE: "以下内容仅部分可追溯到你提供的材料，请按线索核对后再采信。",
    AbstentionState.CLARIFY: "现有材料不足以定位答案。请补充相关段落或指明出处，我再作答。",
    AbstentionState.ABSTAIN: "我没有在给定材料里找到支撑这句话的依据，因此不作答。",
}


def evidence_aware_decision(
    grounded_prob: float,
    coverage: float,
    retrieval_confidence: float,
    policy: Optional[GroundedPolicy] = None,
    n_unsupported: int = 0,
) -> AbstentionDecision:
    """三路信号 → 四态处置。

    判定顺序是刻意的（先判"信息够不够"，再判"能不能答"）：

    1. ``retrieval_confidence < tau_retrieval`` 且 ``coverage < coverage_hedge``
       → ``CLARIFY``：什么都没检索到，问题在输入侧；
    2. ``coverage >= coverage_answer`` 且 ``grounded >= tau_grounded``
       → ``ANSWER``；
    3. ``coverage >= coverage_hedge`` → ``HEDGE``：有部分依据，但不许下断言；
    4. 其余 → ``ABSTAIN``：检索有命中、输出却归因不上，最该拦的一类。

    另有**单向降级**规则：``n_unsupported > 0`` 时 ``ANSWER`` 一律降为 ``HEDGE``。
    覆盖率是长度加权聚合，一句短的编造可能被两句长的真话稀释到阈值以上；
    但"有依据"应当是**逐句**性质，不是平均水平。
    """
    pol = policy or GroundedPolicy()
    pol.validate()
    g = float(max(0.0, min(1.0, grounded_prob)))
    c = float(max(0.0, min(1.0, coverage)))
    r = float(max(0.0, min(1.0, retrieval_confidence)))
    score = pol.w_grounded * g + pol.w_coverage * c + pol.w_retrieval * r

    if r < pol.tau_retrieval and c < pol.coverage_hedge:
        state, reason = AbstentionState.CLARIFY, "检索置信低于阈值且归因覆盖不足"
    elif c >= pol.coverage_answer and g >= pol.tau_grounded:
        state, reason = AbstentionState.ANSWER, "归因覆盖充分且模型判有依据"
    elif c >= pol.coverage_hedge:
        state, reason = AbstentionState.HEDGE, "部分归因：允许带限定语作答"
    else:
        state, reason = AbstentionState.ABSTAIN, "有检索命中但输出归因不上"

    if n_unsupported > 0 and pol.cap_on_unsupported:
        note = f"存在 {n_unsupported} 句无归因依据，禁止无保留断言"
        if state is AbstentionState.ANSWER:
            state = AbstentionState.HEDGE
            reason = note
        elif state is AbstentionState.HEDGE:
            reason = f"{reason}；{note}"

    return AbstentionDecision(state=state, score=score, grounded_prob=g,
                              coverage=c, retrieval_confidence=r,
                              reason=reason,
                              message=_DEFAULT_MESSAGES[state])


# ---------------------------------------------------------------------------
# 校准
# ---------------------------------------------------------------------------
def fit_temperature(logits: Sequence[float], labels: Sequence[int],
                    grid: Optional[Sequence[float]] = None) -> float:
    """用网格搜索拟合温度缩放（最小化 BCE），确定性、无第三方依赖。

    Args:
        logits: 未校准的 raw logit 列表。
        labels: 0/1 标签（必须来自外部证据）。
        grid: 候选温度；默认 0.05–10 的对数网格。

    Returns:
        选中的温度（>0）。
    """
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have the same length")
    if not logits:
        return 1.0
    if grid is None:
        grid = [0.05 * (1.35 ** i) for i in range(30)]     # 0.05 … ~12
    z = torch.tensor(list(map(float, logits)), dtype=torch.float32)
    y = torch.tensor(list(map(float, labels)), dtype=torch.float32)
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        if t <= 0:
            continue
        loss = float(F.binary_cross_entropy_with_logits(z / t, y))
        if loss < best_loss - 1e-9:
            best_t, best_loss = float(t), loss
    return best_t


def select_threshold(scores: Sequence[float], labels: Sequence[int],
                     metric: str = "f1") -> Dict[str, float]:
    """在给定分数上扫描阈值，返回最优阈值与该点的混淆统计。

    ``scores`` 越大越"有依据"。判定规则为 ``score >= thr → 正类``。
    用于把 L4 的原始概率变成**可解释的工作点**（而不是拍 0.5）。
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "accuracy": 0.0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    pairs = sorted(zip(map(float, scores), map(int, labels)), key=lambda p: p[0])
    cands = sorted({s for s, _ in pairs})
    best: Dict[str, float] = {}
    for thr in cands:
        tp = fp = tn = fn = 0
        for s, y in pairs:
            pred = 1 if s >= thr else 0
            if pred and y:
                tp += 1
            elif pred and not y:
                fp += 1
            elif not pred and not y:
                tn += 1
            else:
                fn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(pairs)
        val = {"precision": prec, "recall": rec, "f1": f1,
               "accuracy": acc}[metric]
        row = {"threshold": thr, "precision": prec, "recall": rec, "f1": f1,
               "accuracy": acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        if not best or val > best[metric] + 1e-12:
            best = row
    return best


def expected_calibration_error(probs: Sequence[float], labels: Sequence[int],
                               n_bins: int = 10) -> float:
    """ECE（Guo et al., 2017）：置信度与准确率的分箱平均偏差。"""
    if not probs or len(probs) != len(labels):
        return 0.0
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        p = max(0.0, min(1.0, float(p)))
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, int(y)))
    n = float(len(probs))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(conf - acc)
    return ece


# ---------------------------------------------------------------------------
# 运行时管线：检索 → 生成 → 归因 → 四态决策
# ---------------------------------------------------------------------------
@dataclass
class GroundedResult:
    """一次带证据的回答。``text`` 可能已被替换为拒答/澄清话术。"""

    text: str
    decision: AbstentionDecision
    attribution: Dict[str, Any] = field(default_factory=dict)
    raw_answer: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "raw_answer": self.raw_answer,
            "decision": self.decision.as_dict(),
            "attribution": self.attribution,
        }


class GroundedAnswerer:
    """把三层接起来的运行时外壳（不绑定具体模型）。

    Args:
        generate: ``(prompt, context) -> str``，模型生成回调。
        retriever: ``(prompt) -> List[hit]``，命中需含 ``text`` 字段。
        grounded_scorer: ``(prompt, answer) -> float``，可选；不给时使用
            **文本级代理**（见下），并在 ``attribution`` 里标注 ``grounded_source="proxy"``。
        policy: 四态阈值。

    ``grounded_scorer`` 缺省时的代理是"最佳单句覆盖率"，它与
    ``attribution_coverage`` 相关但不等价，属于**未训练 L4 头时的占位**。
    接入真实 ``AbstentionHead`` 后应显式传入 scorer，代理值不可用于对外结论。
    """

    def __init__(self, generate: Callable[[str, str], str],
                 retriever: Optional[Callable[[str], Sequence[Any]]] = None,
                 grounded_scorer: Optional[Callable[[str, str], float]] = None,
                 policy: Optional[GroundedPolicy] = None) -> None:
        self.generate = generate
        self.retriever = retriever
        self.grounded_scorer = grounded_scorer
        self.policy = policy or GroundedPolicy()

    def answer(self, prompt: str, context: Optional[str] = None,
               hits: Optional[Sequence[Any]] = None) -> GroundedResult:
        if hits is None and self.retriever is not None:
            hits = self.retriever(prompt)
        if hits is None and context:
            index = EvidenceIndex.from_text(context, source="context")
        elif hits is not None:
            index = EvidenceIndex.from_retrieval(hits)
            if context:
                for sp in EvidenceIndex.from_text(context, source="context").spans:
                    index.spans.append(sp)
        else:
            index = EvidenceIndex(source="none")

        ctx_text = context or "\n".join(sp.text for sp in index.spans)
        raw = self.generate(prompt, ctx_text)
        bindings = index.bind_all(split_claims(raw), self.policy.min_coverage,
                                  self.policy.min_ngram)
        cov = attribution_coverage(bindings)

        if self.grounded_scorer is not None:
            g = float(self.grounded_scorer(prompt, raw))
            g_src = "head"
        else:
            g = max((b.coverage for b in bindings), default=0.0)
            g_src = "proxy"

        # 检索置信：真命中的证据片段数（饱和到 1），反映"检索有没有捞到东西"
        n_spans = len(index)
        rc = 0.0 if n_spans == 0 else min(1.0, n_spans / 3.0)

        n_unsupported = sum(1 for b in bindings if not b.supported)
        decision = evidence_aware_decision(g, cov, rc, self.policy,
                                           n_unsupported=n_unsupported)
        text = raw
        if decision.state is not AbstentionState.ANSWER and decision.message:
            text = f"{decision.message}\n\n{raw}" if decision.state is AbstentionState.HEDGE else decision.message

        return GroundedResult(
            text=text, decision=decision, raw_answer=raw,
            attribution={
                "grounded_source": g_src,
                "n_claims": len(bindings),
                "n_unsupported": n_unsupported,
                "bindings": [b.as_dict() for b in bindings],
            },
        )
