# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""把多个 run 的 **验证损失曲线** 按 step 对齐成一张表（只读 report.json，不加载模型）。

用法::

    python scripts/compare_curves.py runs/ab_sgc_5m runs/ab_attn_5m runs/srf5m_fair
    python scripts/compare_curves.py runs/* --to-step 1000 --json runs/curves.json

与 ``compare_dialogue_runs.py`` 的分工：那个靠重新跑对话评测，慢但测的是"会不会
说话"；这个只读训练时已经存下来的 val_loss，秒出，测的是"学得快不快"。两者都
需要，因为架构差异可能在一个上显现、在另一个上被淹没。

存在的理由：这张表曾经被手算错过一次 —— 两组参数量看着都在 5.7M，实际一组的
``num_bases`` 被 smoke 模式截成了 16、另一组是 32，记忆容量差一倍。**参数总量
相等不等于配置可比**，所以这个脚本会在配置不一致时直接喊出来，而不是默默出表。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# 配置一致性检查分三组：全集里只有"这个 mixer 真的会读"的字段才参与比较。
# 曾经把 num_heads 也算进去，结果 sgc（num_heads 是残留字段，谱核根本不读）
# 和 attn（5 头）被误报成"配置不同"——噪音会让人开始忽略这些警告，那比没有更糟。
_KEYS_ALWAYS = ("dim", "num_layers", "hidden")
_KEYS_SSM = ("num_bases",)                 # sgc / srf 都有记忆基
_KEYS_SRF = ("srf_chunk",)                 # 只有 srf 读
_KEYS_ATTN = ("num_heads", "num_kv_heads", "attn_window")
# 参数总量允许的相对偏差；超过就不是等参对照了。
_PARAM_TOL = 1.0  # %


def _uses_attn(cfg: Dict) -> bool:
    return cfg.get("mixer") == "attn" or bool(cfg.get("reason_attn_every"))


def _uses_ssm(cfg: Dict) -> bool:
    return cfg.get("mixer") in ("sgc", "srf")


def compare_keys(rc: Dict, cc: Dict) -> List[str]:
    keys = list(_KEYS_ALWAYS)
    if _uses_ssm(rc) and _uses_ssm(cc):
        keys += _KEYS_SSM
    if rc.get("mixer") == "srf" and cc.get("mixer") == "srf":
        keys += _KEYS_SRF
    if _uses_attn(rc) and _uses_attn(cc):
        keys += _KEYS_ATTN
    return keys


def load(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "report.json")
    if not os.path.isfile(path):
        raise SystemExit(f"[error] 找不到 {path}（训练没跑完？）")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label(run_dir: str, rep: Dict) -> str:
    """列名：优先用 mixer，同 mixer 多个 run 时退回目录名。"""
    cfg = rep.get("config") or {}
    mx = cfg.get("mixer") or "?"
    extra = []
    if cfg.get("reason_attn_every"):
        extra.append(f"ra{cfg['reason_attn_every']}")
    if mx == "srf" and cfg.get("srf_chunk"):
        extra.append(f"C{cfg['srf_chunk']}")
    base = mx + ("+" + "+".join(extra) if extra else "")
    return base


def val_curve(rep: Dict) -> Dict[int, float]:
    return {int(p["step"]): float(p["val_loss"])
            for p in rep.get("curve", []) if "val_loss" in p}


def check_fair(runs: List[Dict]) -> List[str]:
    """返回警告列表；空列表 = 这些 run 之间只有 mixer 不同。"""
    warns = []
    if len(runs) < 2:
        return warns
    ref_name, ref = runs[0]
    for name, rep in runs[1:]:
        rc, cc = ref.get("config", {}), rep.get("config", {})
        diffs = [f"{k}: {rc.get(k)}→{cc.get(k)}"
                 for k in compare_keys(rc, cc)
                 if rc.get(k) != cc.get(k) and (rc.get(k) is not None or cc.get(k) is not None)]
        if diffs:
            warns.append(f"{ref_name} vs {name} 配置不同 → " + "; ".join(diffs))
        p0, p1 = ref.get("params"), rep.get("params")
        if p0 and p1:
            dev = abs(p1 - p0) / p0 * 100
            if dev > _PARAM_TOL:
                warns.append(f"{ref_name} vs {name} 参数量差 {dev:.2f}%"
                             f"（{p0:,} vs {p1:,}），超过 {_PARAM_TOL}% 容差")
    return warns


def main() -> int:
    ap = argparse.ArgumentParser(description="按 step 对齐多个 run 的验证损失")
    ap.add_argument("runs", nargs="+", help="run 目录（含 report.json）")
    ap.add_argument("--to-step", type=int, default=None, help="只显示到该 step（便于公平截断）")
    ap.add_argument("--json", default=None, help="把对齐结果存成 json")
    ap.add_argument("--no-fairness", action="store_true", help="跳过配置一致性检查")
    args = ap.parse_args()

    loaded = [(d, load(d)) for d in args.runs]
    names, curves, meta = {}, {}, {}
    for d, rep in loaded:
        lb = label(d, rep)
        if lb in names.values():              # 同 mixer 撞名 → 用目录名区分
            lb = lb + "@" + os.path.basename(d.rstrip("/\\"))
        names[d] = lb
        curves[d] = val_curve(rep)
        cfg = rep.get("config") or {}
        meta[d] = dict(label=lb, params=rep.get("params"), steps=rep.get("steps"),
                       best=rep.get("best_val_loss"), nb=cfg.get("num_bases"),
                       chunk=cfg.get("srf_chunk"), hidden=cfg.get("hidden"))

    # ---- 概览 ----
    def _fmt(v, spec="", dash="-"):
        return format(v, spec) if v is not None else dash

    print("\n== run 概览 ==")
    print(f"{'run':<14} {'mixer':<16} {'params':>12} {'steps':>7} {'best_val':>9} "
          f"{'nb':>4} {'C':>5} {'ffn':>5}")
    for d, _ in loaded:
        m = meta[d]
        short = os.path.basename(d.rstrip("/\\"))
        print(f"{short:<14} {m['label']:<16} "
              f"{_fmt(m['params'], ','):>12} {_fmt(m['steps']):>7} "
              f"{_fmt(m['best'], '.4f'):>9} {_fmt(m['nb']):>4} "
              f"{_fmt(m['chunk']):>5} {_fmt(m['hidden']):>5}")

    # ---- 公平性 ----
    if not args.no_fairness:
        warns = check_fair(loaded)
        if warns:
            print("\n!! 配置不可比，下面的表不能当 A/B 结论用：")
            for w in warns:
                print("   -", w)
        else:
            print("\n[公平性] 各 run 配置一致，差异只来自 mixer。")

    # ---- 对齐表 ----
    steps = sorted(set().union(*[set(c) for c in curves.values()]))
    if args.to_step:
        steps = [s for s in steps if s <= args.to_step]
    head = "step".rjust(6) + "".join(f"{names[d]:>14}" for d, _ in loaded)
    print("\n== 验证损失（按 step 对齐） ==")
    print(head)
    rows = []
    for s in steps:
        cells, vals = [], []
        for d, _ in loaded:
            v = curves[d].get(s)
            vals.append(v)
            cells.append(f"{v:.4f}".rjust(14) if v is not None else "-".rjust(14))
        print(f"{s:>6}" + "".join(cells))
        rows.append({"step": s,
                     **{names[d]: curves[d].get(s) for d, _ in loaded}})

    # ---- 末尾差值（相对第一个 run）----
    if len(loaded) >= 2 and rows:
        last = rows[-1]
        ref = loaded[0]
        print(f"\n== 末行差值（相对 {names[ref[0]]}） ==")
        for d, _ in loaded[1:]:
            a, b = last.get(names[ref[0]]), last.get(names[d])
            if a is not None and b is not None:
                print(f"  {names[d]:<16} {b - a:+.4f}  ({(b - a) / a * 100:+.1f}%)")

    payload = {"runs": [{"dir": d, "label": names[d], "params": rep.get("params"),
                         "steps": rep.get("steps"), "best_val_loss": rep.get("best_val_loss")}
                        for d, rep in loaded],
               "curve": rows}
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[ok] 写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
