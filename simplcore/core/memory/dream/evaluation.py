# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""held-out 评测与 no-regression 闸门：EvalScores / GateVerdict / bootstrap。"""
from __future__ import annotations



from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch




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

    判定口径（诚实边界）：本 gate 证明的是「**不退步**」——那是写回权重的
    **必要条件**，**不是**「越用越好」的证据。泛化提升的判据只能是
    **与经验/巩固信号不重叠的 held-out 评测**（见
    ``ReflectionConfig.heldout_eval`` 与 ``ReflectionReport.heldout_delta``）。
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

