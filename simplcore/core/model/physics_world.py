# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""物理世界模型（0.1-α，**实验功能，opt-in 默认关闭**）.

解决的问题：文本/多模态模型的抗幻觉闸门目前有两个信号源——L2 证据门
（外部证据）与 L3 不确定门（内部分歧）；**两者都不检查「内容是否符合
物理规律」**。一个模型可以说「他把手举高 3 米然后又下降了 2 米，现在
手在 6 米处」——没有证据冲突、模型自己也不确定度异常，但物理上矛盾。

本模块引入第三个**正交信号源**：物理一致性。

* 结构：物理量轨迹抽取 → 规则约束检查（连续性/守恒/运动学）→
  相容分数 → 幻觉信号；可学习的 ``PhysicsPredictor`` 头从 O(1) 状态
  预测下一时刻物理量（与 ``world_probe`` 同一测量文化）。
* 互补（不是替代）：L2 管「有没有证据」、L3 管「模型确不确定」、
  本模块管「说的和物理规律矛不矛盾」——三路信号正交，可独立消融。
* 接入：``hallucination_signal`` 供调用方与 L4 拒答头阈值联用；
  ``bias_fn`` 适配 ``SimplCoreModel.generate(logit_bias_fn=...)`` 的
  既有钩子（物理不一致时给 EOS 正偏置——倾向收尾，零侵入生成路径）。

**opt-in 纪律（与 evidence_gate 同级）**：``SimplCoreConfig.
use_physics_world_model`` 默认 ``False``；**tuan-2b 预设完全不使用**
（预设不含此开关为 True；训练主路径零引用，守护测试锁定）。
只有显式置 True 才接线——未开时零行为差异、零依赖。

诚实边界
--------
* 轨迹抽取器（``RegexQuantityExtractor``）是**结构化占位**：只识别
  阿拉伯数字 + 有限单位/动词线索，中文数字、隐式指代、复杂事件结构
  均不解析——替换点明确（传自定义 ``extractor`` 即可）；
* 规则约束是**手工物理先验**（一阶近似），不是学习到的动力学；
* ``PhysicsPredictor`` 未训练时输出无意义（标准初始化，无零初始化
  纪律——它不是残差流注入，是测量头）；未在真实数据上实测；
* 全模块不构成「模型理解物理」的宣称——它检查的是**文本描述的量
  与手工约束的相容性**。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["PhysicsEvent", "RegexQuantityExtractor", "PhysicsConstraints",
           "PhysicsPredictor", "PhysicsWorldModel"]


# ---------------------------------------------------------------------------
# 物理事件与抽取器
# ---------------------------------------------------------------------------
@dataclass
class PhysicsEvent:
    """从文本抽取的一条物理量轨迹事件（span 可溯源到「第几个字」）."""
    timestamp: float          # 序内时间（句子序号；抽取器的时间粒度）
    quantity: float           # 数值（默认口径：以「米」计的位置/长度量）
    kind: str                 # 量类型：position / delta / count
    span: Tuple[int, int]     # 字符偏移 [start, end)（可寻址引用）
    text: str = ""            # 原文片段（审计用）


@dataclass
class _Sent:
    start: int
    text: str


class RegexQuantityExtractor:
    """结构化占位抽取器：数字 + 位置/移动动词线索 → 物理量轨迹.

    只支持阿拉伯数字（中文数字/隐式指代不解析——诚实边界）。识别两类量：
    * ``position``：「在/位于/高度 … N 米」—— 绝对位置量；
    * ``delta``：   「上升/下降/前进/后退 … N 米」—— 相对位移量
      （带方向符号，下降/后退为负）。
    ``kind`` 与符号规则是可替换的占位实现（构造参数注入自定义正则集）。
    """

    _NUM = r"(\d+(?:\.\d+)?)\s*(?:米|m)(?![0-9])"
    _POS = re.compile(r"(?:在|位于|高度|深度|距离)[^\d。；;]{0,8}" + _NUM)
    _DELTA_UP = re.compile(r"(?:上升|升高|上涨|增加|前进|加了?)[^\d。；;]{0,4}" + _NUM)
    _DELTA_DOWN = re.compile(r"(?:下降|降低|下跌|减少|后退|减了?)[^\d。；;]{0,4}" + _NUM)

    def __init__(self, pos_pattern: Optional[re.Pattern] = None,
                 up_pattern: Optional[re.Pattern] = None,
                 down_pattern: Optional[re.Pattern] = None,
                 sentence_split: str = "。；;！？!?\n") -> None:
        self.pos = pos_pattern or self._POS
        self.up = up_pattern or self._DELTA_UP
        self.down = down_pattern or self._DELTA_DOWN
        self.sentence_split = sentence_split

    def __call__(self, text: str) -> List[PhysicsEvent]:
        """可调用别名：自定义 extractor 只需实现 ``callable(text)->events``。"""
        return self.extract(text)

    def extract(self, text: str) -> List[PhysicsEvent]:
        events: List[PhysicsEvent] = []
        for si, sent in enumerate(_split_sentences(text, self.sentence_split)):
            for m in self.pos.finditer(sent.text):
                events.append(PhysicsEvent(
                    timestamp=float(si), quantity=float(m.group(1)),
                    kind="position",
                    span=(sent.start + m.start(1), sent.start + m.end(1)),
                    text=sent.text[m.start():m.end()]))
            for m in self.up.finditer(sent.text):
                events.append(PhysicsEvent(
                    timestamp=float(si), quantity=float(m.group(1)),
                    kind="delta",
                    span=(sent.start + m.start(1), sent.start + m.end(1)),
                    text=sent.text[m.start():m.end()]))
            for m in self.down.finditer(sent.text):
                events.append(PhysicsEvent(
                    timestamp=float(si), quantity=-float(m.group(1)),
                    kind="delta",
                    span=(sent.start + m.start(1), sent.start + m.end(1)),
                    text=sent.text[m.start():m.end()]))
        events.sort(key=lambda e: e.span[0])
        return events


def _split_sentences(text: str, seps: str) -> List[_Sent]:
    out: List[_Sent] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in seps:
            if i > start:
                out.append(_Sent(start, text[start:i]))
            start = i + 1
    if start < len(text):
        out.append(_Sent(start, text[start:]))
    return out


# ---------------------------------------------------------------------------
# 规则约束（手工物理先验，一阶近似）
# ---------------------------------------------------------------------------
class PhysicsConstraints:
    """每条约束返回 [0,1] 相容度：1 = 完全相容，0 = 完全矛盾."""

    @staticmethod
    def continuity(positions: Sequence[float], max_jump: float = 10.0) -> float:
        """位置不跳变：相邻位置差超过 ``max_jump``（默认 10 m / 句）即违规.

        分数 = 1 − (超出比例的最大值)，全平滑 → 1.0。
        """
        if len(positions) < 2:
            return 1.0
        worst = 0.0
        for a, b in zip(positions, positions[1:]):
            jump = abs(b - a) - max_jump
            worst = max(worst, jump / max(max_jump, 1e-9))
        return max(0.0, 1.0 - worst)

    @staticmethod
    def conservation(deltas: Sequence[float], tolerance: float = 0.0) -> float:
        """守恒检查：**只有位移记录、没有位置锚点**时，位移应回到锚点附近.

        实现口径：δ 序列若从未出现 position 锚点（由调用方过滤），则约束
        退化为「净位移不应超出单步位移和的合理范围」——恒真。真正的守恒
        断言在 :meth:`net_displacement_consistent`（有锚点时才启用）。
        """
        return 1.0

    @staticmethod
    def net_displacement_consistent(start: float, deltas: Sequence[float],
                                    end: float, tol: float = 1e-6) -> float:
        """x_end − x_start = Σδ（运动学闭合约束；误差按 |Σδ| 归一）."""
        total = sum(deltas)
        denom = max(abs(total), abs(end - start), 1.0)
        return max(0.0, 1.0 - abs((start + total) - end) / denom)

    @staticmethod
    def kinematics(deltas: Sequence[float], durations: Sequence[float],
                   tol: float = 1e-6) -> float:
        """速度一致性：v_i = δ_i/τ_i，加速度 |a| = |Δv/τ| 无超限.

        ``durations`` 缺省全 1（按句粒度）。返回 1 − (最大超限比例)。
        """
        if len(deltas) != len(durations) or len(deltas) < 2:
            return 1.0
        vs = [d / max(t, tol) for d, t in zip(deltas, durations)]
        worst = 0.0
        max_acc = 10.0                                    # m/s² 量级先验
        for v0, v1 in zip(vs, vs[1:]):
            a = abs(v1 - v0)                              # 按句粒度 τ=1
            worst = max(worst, (a - max_acc) / max_acc)
        return max(0.0, 1.0 - worst)


# ---------------------------------------------------------------------------
# 可学习预测头（测量文化：未训练 = 无意义，如实标注）
# ---------------------------------------------------------------------------
class PhysicsPredictor(nn.Module):
    """O(1) 状态 → 下一时刻物理量的线性读出（与 world_probe 同一口径）.

    未训练时输出无意义；训练目标由调用方构造（合成弹道/守恒系统），
    本模块不自带训练闭环。
    """

    def __init__(self, dim: int, n_quantities: int = 1) -> None:
        super().__init__()
        self.head = nn.Linear(dim, n_quantities)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., dim) O(1) 状态 → (..., n_quantities) 预测物理量。"""
        return self.head(h)


# ---------------------------------------------------------------------------
# 组合：物理世界模型（opt-in）
# ---------------------------------------------------------------------------
class PhysicsWorldModel(nn.Module):
    """物理一致性判据 + 可学习预测头（实验件，默认关闭）.

    用法::

        pwm = PhysicsWorldModel(dim=model.config.dim)
        signal = pwm.hallucination_signal(text)   # [0,1]，越高越可疑
        # 与 L4 拒答头联用：signal 超阈值 → 拒答/收尾
        # 与生成钩子联用：
        model.generate(idx, logit_bias_fn=pwm.bias_fn(text, threshold=0.5))
    """

    def __init__(self, dim: int, max_jump: float = 10.0,
                 extractor: Optional[Callable[[str], List[PhysicsEvent]]] = None,
                 predictor: Optional[PhysicsPredictor] = None) -> None:
        super().__init__()
        self.max_jump = float(max_jump)
        self.extractor = extractor or RegexQuantityExtractor()
        self.predictor = predictor or PhysicsPredictor(dim)

    # -- 抽取 + 检查 ------------------------------------------------------------
    def extract(self, text: str) -> List[PhysicsEvent]:
        return self.extractor(text)

    def check(self, text: str) -> Dict[str, float]:
        """各约束相容度；无物理量轨迹时全部返回 1.0（无信号 ≠ 相容，
        note 说明「本次无可检查内容」）。"""
        events = self.extract(text)
        out = {"continuity": 1.0, "kinematics": 1.0,
               "net_displacement": 1.0, "n_events": float(len(events))}
        if len(events) < 2:
            return out
        pos = [e.quantity for e in events if e.kind == "position"]
        deltas = [e.quantity for e in events if e.kind == "delta"]
        if len(pos) >= 2:
            out["continuity"] = PhysicsConstraints.continuity(pos, self.max_jump)
        if len(deltas) >= 2:
            out["kinematics"] = PhysicsConstraints.kinematics(
                deltas, [1.0] * len(deltas))
        # 位移闭合：有位置锚点 + 位移混合时检查（事件流按 span 排序后，
        # position 之间的 delta 求和应等于位置差）
        if pos and deltas:
            out["net_displacement"] = self._closure(events, pos)
        return out

    def _closure(self, events: List[PhysicsEvent], pos: List[float]) -> float:
        """位移闭合：任意相邻两个 position 之间的 delta 和 ≈ 位置差."""
        worst = 1.0
        anchor = None
        acc = 0.0
        for e in events:
            if e.kind == "position":
                if anchor is not None:
                    worst = min(worst, PhysicsConstraints
                                .net_displacement_consistent(anchor, [acc],
                                                             e.quantity))
                anchor, acc = e.quantity, 0.0
            else:
                acc += e.quantity
        return worst

    # -- 信号 -------------------------------------------------------------------
    def hallucination_signal(self, text: str) -> float:
        """物理幻觉信号 ∈ [0,1]：1 − min(各约束相容度)。无可查内容 → 0."""
        checks = self.check(text)
        active = [v for k, v in checks.items()
                  if k != "n_events" and not (k in ("continuity", "kinematics",
                                                   "net_displacement")
                                              and checks["n_events"] < 2)]
        return 1.0 - min(active) if active else 0.0

    # -- 生成钩子适配（零侵入：logit_bias_fn 已有机制） ---------------------------
    def bias_fn(self, text: str, threshold: float = 0.5,
                eos_token_id: Optional[int] = None
                ) -> Callable[[torch.Tensor], Optional[torch.Tensor]]:
        """适配 ``generate(logit_bias_fn=...)``：信号超阈值 → EOS 正偏置.

        物理不一致时倾向「收尾」——保守策略：不试图改写内容，只提前止损。
        ``eos_token_id`` 缺省取 vocab 末位（调用方应显式传入真实 EOS）。
        """
        signal = self.hallucination_signal(text)
        fired = signal > threshold

        def _bias(logits: torch.Tensor) -> Optional[torch.Tensor]:
            if not fired:
                return None
            eos = eos_token_id if eos_token_id is not None else logits.shape[-1] - 1
            bias = torch.zeros_like(logits)
            bias[..., eos] = 2.0 * (signal - threshold) / max(1.0 - threshold, 1e-9)
            return bias
        return _bias

    # -- 可学习预测头（测量口径） --------------------------------------------------
    def predict_next(self, h: torch.Tensor) -> torch.Tensor:
        """O(1) 状态 → 下一时刻物理量。**未训练时输出无意义**."""
        return self.predictor(h)

    def audit(self, text: str) -> Dict[str, Any]:
        """完整审计账本：事件轨迹（含 span）+ 约束分数 + 信号."""
        return {
            "events": [{"t": e.timestamp, "q": e.quantity, "kind": e.kind,
                        "span": list(e.span), "text": e.text}
                       for e in self.extract(text)],
            "checks": self.check(text),
            "hallucination_signal": round(self.hallucination_signal(text), 6),
            "note": "实验功能（opt-in）：规则约束为手工先验；extractor 为"
                    "结构化占位（阿拉伯数字）；不构成「模型理解物理」宣称",
        }


def main(argv: Optional[List[str]] = None) -> int:
    """自检：一致 vs 矛盾的两段文本，信号应显著区分（机制冒烟）。"""
    pwm = PhysicsWorldModel(dim=16)
    good = "他在 2 米处停住。之后位于 3 米高度。随后在 4 米处。"
    bad = "他在 2 米处停住。之后位于 3 米高度。突然位于 30 米处。"
    s_good, s_bad = pwm.hallucination_signal(good), pwm.hallucination_signal(bad)
    print(f"相容文本 signal={s_good:.3f}\n矛盾文本 signal={s_bad:.3f}")
    print(pwm.audit(bad)["checks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())