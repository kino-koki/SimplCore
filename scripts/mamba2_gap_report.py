#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mamba2_gap_report.py — sgc vs Mamba2 真实语料差距的趋势报告.

汇总三档同 token 前哨（1M / 4M / 16M，char 级中文语料，D=128/4 层，
等 token 直训）的 sgc/mamba2 val PPL 比值，给出：
  * 差距随数据规模的收敛趋势（1.77× → 1.15× → 16M 实测）；
  * 对数线性外推到 1B×24h 预算（~0.46B tokens）的差距预测区间
    —— 外推是【一阶趋势外推】，只用于战役规划，不构成能力主张；
  * 超越判据状态：ratio < 1.0 即"同 token 超越"。

用法：python scripts/mamba2_gap_report.py [--json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts" / "binding_probe"


def load(name):
    p = A / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    lm1 = load("real_lm.json")
    lm4 = load("real_lm_scale.json")
    lm16 = load("real_lm_scale16.json")

    rows = []
    if lm1 and lm1.get("sgc", {}).get("val_ppl") and lm1.get("mamba2"):
        rows.append(("1M", lm1["sgc"]["val_ppl"], lm1["mamba2"]["val_ppl"],
                     lm1["sgc"].get("tokens")))
    if lm4 and lm4.get("sgc", {}).get("val_ppl") and lm4.get("mamba2"):
        rows.append(("4M", lm4["sgc"]["val_ppl"], lm4["mamba2"]["val_ppl"],
                     lm4["sgc"].get("tokens")))
    if lm16 and lm16.get("sgc", {}).get("val_ppl") and lm16.get("mamba2"):
        rows.append(("16M", lm16["sgc"]["val_ppl"], lm16["mamba2"]["val_ppl"],
                     lm16["sgc"].get("tokens")))

    out = {"rows": [], "extrapolation": None, "verdict": None}
    print("== sgc / Mamba2 同 token 差距趋势（真实语料 char 级前哨）==")
    pts = []
    for name, sp, mp, toks in rows:
        ratio = sp / mp
        # log-token 轴上的累计 token（用于外推）
        pts.append((math.log2(toks / 1e6), math.log(ratio)))
        out["rows"].append({"scale": name, "tokens": toks,
                            "sgc_ppl": sp, "mamba2_ppl": mp,
                            "ratio": round(ratio, 3)})
        print(f"  {name:>4}: sgc {sp:.2f} / mamba2 {mp:.2f} = {ratio:.3f}x")

    if len(pts) >= 2:
        # 对数线性最小二乘：log(ratio) ~ a*log2(tokens/1M) + b
        n = len(pts)
        sx = sum(p[0] for p in pts) / n
        sy = sum(p[1] for p in pts) / n
        sxx = sum(p[0] * p[0] for p in pts) / n
        sxy = sum(p[0] * p[1] for p in pts) / n
        denom = (sxx - sx * sx)
        if abs(denom) > 1e-12:
            a = (sxy - sx * sy) / denom
            b = sy - a * sx
            # 1B×24h 预算 ~0.46B tokens => log2(460) ≈ 8.84
            x1b = math.log2(460.0)
            pred = math.exp(a * x1b + b)
            out["extrapolation"] = {
                "slope_per_doubling": round(a, 4),
                "predicted_ratio_at_0_46B": round(pred, 3),
                "note": ("对数线性外推，仅用于战役规划；1B/word-level/真实"
                         "分词与 char 级前哨不同构，实机实测为准")}
            print(f"  趋势斜率: {a:+.4f} log-ratio / 数据翻倍")
            print(f"  外推 0.46B tokens（1B×24h）: ratio ≈ {pred:.3f}")
            if pred < 1.0:
                print("  判定: 趋势上超越 Mamba2（ratio < 1）——需 1B 实机验证")
            else:
                gap_pct = (pred - 1.0) * 100
                print(f"  判定: 趋势外推仍差 {gap_pct:.0f}%——需机制改进或更多 tokens")
            out["verdict"] = ("trend-parity" if abs(pred - 1.0) < 0.05 else
                              ("trend-beat" if pred < 1.0 else "trend-gap"))

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
