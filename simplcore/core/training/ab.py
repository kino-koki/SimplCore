# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""A/B 对照的统计层：配对比较、精确检验、参数效率外推。

这套东西存在的理由
------------------
"1B 抵 3B~7B"这类说法如果不写成**受控比较 + 不确定性**，就只是宣传。
本模块只做三件事，而且每件都刻意保守：

1. **配对比较**：两侧跑的是同一套题（同一 prompt、同一检查器、同一贪心
   解码）。因此正确做法不是"比两个通过率"，而是看**逐题的通过/失败翻转**
   —— 只有"一边过一边不过"的题携带信息，"两边都过"的题对差值毫无贡献。
2. **精确检验而不是正态近似**：基础套件只有 10 题，DeLong / z 检验
   在小样本上的名义 p 值会严重低估错误率。这里用 McNemar **精确**
   二项检验，样本量不足时它会诚实地给不出显著性。
3. **参数效率只在有插值支撑时才报**：跨规模外推必须至少有两个规模点，
   且要说明插值方式。单点数据只能报"该点的比值"，不能报"曲线"。

明确不做：把 ``task_pass_rate`` 的差值当成"能力差"。10 道题上的
0.1 差值只有一个题的证据量，它连方向都可能是错的。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------
def _binom_two_sided_p(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）。

    在 H0（两种架构逐题通过概率相同）下，翻转题里"只 A 过"与"只 B 过"
    各占一半，即 ``b ~ Binomial(b + c, 0.5)``。精确 p 值是**两倍的单侧尾**
    再截断到 1，而不是 ``2 * P(X <= min(b,c))`` 的粗暴近似 —— 当 ``b == c``
    时后者会给出 1.0，恰好正确；但 ``b + c`` 为奇数时容易算到 >1。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar(b: int, c: int) -> Dict[str, Any]:
    """返回 ``{"b","c","n_discordant","p_value","significant"}``。

    ``significant`` 用 alpha=0.05。``n_discordant`` 太小（<5）时即使
    p 值很小也不该当结论 —— 那是一个题翻转就能改变的世界。
    """
    p = _binom_two_sided_p(b, c)
    return {
        "b_only_first_passes": b,
        "c_only_second_passes": c,
        "n_discordant": b + c,
        "p_value": p,
        "significant": bool(p < 0.05 and (b + c) >= 5),
    }


def bootstrap_diff_ci(a: Sequence[float], b: Sequence[float],
                      alpha: float = 0.05, n_boot: int = 4000,
                      seed: int = 0) -> Dict[str, Any]:
    """配对 bootstrap：``mean(a) - mean(b)`` 的百分位置信区间。

    配对重采样（同一下标同时取两侧），因为两侧跑的是同一套题。
    非配对重采样会把"题目难度"这一项当成噪声，区间被无谓拉宽。
    """
    if len(a) != len(b):
        raise ValueError("配对比较要求等长")
    n = len(a)
    if n == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = random.Random(seed)
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    point = sum(diffs) / n
    boots: List[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(math.floor(alpha / 2 * len(boots)))]
    hi = boots[int(math.ceil((1 - alpha / 2) * len(boots))) - 1]
    return {
        "diff": point,
        "ci_low": lo,
        "ci_high": hi,
        "n": n,
        "n_boot": int(n_boot),
        # 区间跨 0 就是不显著，不要再用 p 值挑一个好看的
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


# ---------------------------------------------------------------------------
# 逐题配对结果
# ---------------------------------------------------------------------------
@dataclass
class PairedOutcome:
    """两个模型在同一套题上的逐题结果。

    只接受 **按 task id 对齐** 的字典：如果一边少了一道题（比如生成崩了
    导致该题被跳过），必须显式报出来，而不是按位置对齐 —— 按位置对齐会
    把"少一道题"变成"题目错位"，差值会完全失真。
    """

    first: str
    second: str
    ids: List[str] = field(default_factory=list)
    pass_first: List[float] = field(default_factory=list)
    pass_second: List[float] = field(default_factory=list)

    @classmethod
    def from_reports(cls, first: str, second: str,
                     rep_a, rep_b) -> "PairedOutcome":
        a = {r["id"]: 1.0 if r["passed"] else 0.0 for r in rep_a.per_task}
        b = {r["id"]: 1.0 if r["passed"] else 0.0 for r in rep_b.per_task}
        ids = [k for k in a if k in b]
        missing = sorted(set(a) ^ set(b))
        if missing:
            raise ValueError(f"两侧题目不对齐，缺失：{missing}")
        return cls(first=first, second=second, ids=ids,
                   pass_first=[a[k] for k in ids],
                   pass_second=[b[k] for k in ids])

    # -- 统计 ---------------------------------------------------------------
    def rate_first(self) -> float:
        return sum(self.pass_first) / max(len(self.pass_first), 1)

    def rate_second(self) -> float:
        return sum(self.pass_second) / max(len(self.pass_second), 1)

    def discordant(self) -> Tuple[int, int]:
        """``(只第一个过, 只第二个过)``。"""
        b = sum(1 for x, y in zip(self.pass_first, self.pass_second) if x > y)
        c = sum(1 for x, y in zip(self.pass_first, self.pass_second) if y > x)
        return b, c

    def flipped_ids(self) -> List[Dict[str, Any]]:
        out = []
        for i, (x, y) in enumerate(zip(self.pass_first, self.pass_second)):
            if x != y:
                out.append({"id": self.ids[i],
                            "first_passed": bool(x),
                            "second_passed": bool(y)})
        return out

    def summary(self, n_boot: int = 4000, seed: int = 0) -> Dict[str, Any]:
        b, c = self.discordant()
        ci = bootstrap_diff_ci(self.pass_first, self.pass_second,
                               n_boot=n_boot, seed=seed)
        return {
            "first": self.first,
            "second": self.second,
            "n_tasks": len(self.ids),
            "rate_first": round(self.rate_first(), 4),
            "rate_second": round(self.rate_second(), 4),
            "delta": round(ci["diff"], 4),
            "ci95": [round(ci["ci_low"], 4), round(ci["ci_high"], 4)],
            "excludes_zero": ci["excludes_zero"],
            "mcnemar": mcnemar(b, c),
            "flipped": self.flipped_ids(),
            # 10 题里翻转 1 题就能让差值是 0.1，读报告时先看这一行
            "evidence_weight": f"{b + c}/{len(self.ids)} 题出现翻转",
        }


# ---------------------------------------------------------------------------
# 参数效率
# ---------------------------------------------------------------------------
def param_efficiency(points: Sequence[Dict[str, Any]],
                     metric: str = "task_pass_rate") -> Dict[str, Any]:
    """从多档规模的实测结果里算"参数效率比"。

    Args:
        points: 每档一个字典，必须含 ``"name"``、``"params"``、``metric``。
            其中 ``metric`` 是**同一套题**上的通过率。

    算法：把 ``params`` 升序排好，对每个"低参模型"的点，找"高参模型"
    折线里**第一次超过**它分数的位置，线性插值得到等价参数量，比值即
    ``等效参数量 / 低参模型参数量``。

    必须声明的三件事（报告里会自动带上）：
      * 至少两档参数量才能插值，一档只能报点值；
      * 分数非单调时插值会跨过多个交点，取**第一个**是为了保守
        （更高的等价参数意味着更强的说法，不能默认取）；
      * 这是**内插**，超出最高档的外推一律不给。
    """
    fam: Dict[str, List[Dict[str, Any]]] = {}
    for p in points:
        fam.setdefault(str(p["name"]), []).append(p)

    out: Dict[str, Any] = {"metric": metric, "families": {},
                           "comparisons": [], "caveats": []}
    for name, pts in fam.items():
        pts = sorted(pts, key=lambda d: float(d["params"]))
        out["families"][name] = [
            {"params": int(d["params"]), metric: float(d[metric])} for d in pts
        ]

    low = sorted([p for p in points if str(p["name"]) == "sgc"],
                 key=lambda d: float(d["params"]))
    high = sorted([p for p in points if str(p["name"]) != "sgc"],
                  key=lambda d: float(d["params"]))
    if len(low) < 2 or len(high) < 2:
        out["caveats"].append(
            "每种架构至少需要两档参数量才能插值；当前不足以给出效率比。")
        return out

    for lo_pt in low:
        target = float(lo_pt[metric])
        need = None
        for i in range(len(high) - 1):
            y0, y1 = float(high[i][metric]), float(high[i + 1][metric])
            if y0 <= target <= y1 and y1 > y0:
                x0, x1 = float(high[i]["params"]), float(high[i + 1]["params"])
                t = (target - y0) / (y1 - y0)
                need = x0 + t * (x1 - x0)
                break
        if need is None:
            out["comparisons"].append({
                "sgc_params": int(lo_pt["params"]),
                "sgc_score": target,
                "equivalent_params": None,
                "ratio": None,
                "note": ("同族任何规模都没达到该分数 —— 不能报效率比"
                         if target > max(float(p[metric]) for p in high)
                         else "分数不在同族插值区间内"),
            })
            continue
        ratio = need / float(lo_pt["params"])
        out["comparisons"].append({
            "sgc_params": int(lo_pt["params"]),
            "sgc_score": target,
            "equivalent_params": int(round(need)),
            "ratio": round(ratio, 3),
            "note": "内插得到，未外推",
        })
        out["caveats"].append(
            f"{int(lo_pt['params'])/1e6:.1f}M 的效率比来自"
            f"{int(need)/1e6:.1f}M 处的插值；两端的实测点越近，这个数字越可信。")

    out["caveats"].append(
        "效率比是**同数据同预算同评测**下的内插比值，不是'能力等价'的证明；"
        "换数据集或换预算都必须重测。")
    return out


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------
def ab_report(paired: Sequence[PairedOutcome],
              meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把多组配对结果合成一份 A/B 报告。"""
    return {
        "meta": meta or {},
        "pairs": [p.summary() for p in paired],
        "notes": [
            "基础套件只有 10 题：0.1 的差值等于 1 题，读结论前先看 evidence_weight。",
            "所有分数来自贪心解码，未做多 seed 平均；采样方差未计入区间。",
            "配对 bootstrap 只反映**题目抽样**的不确定性，不反映训练随机性。",
        ],
    }
