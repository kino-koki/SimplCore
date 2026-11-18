#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fill_measurements.py — 从训练 run 产出自动回填 24h·4090D 实测口径模板。

用法（实机上）：
    python scripts/fill_measurements.py \
        --run-dir runs/tuan-2b-4090d \
        --out artifacts/4090d/phase0_measurements.json \
        [--peak-tflops 157.0]

口径（与 artifacts/4090d/measurement_template.json 的 caliber 一致）：
  * tokens/s = (micro_batch × seq_len) / step 秒，取 metrics.jsonl 稳态中位数
    （跳过前 10 步预热）；
  * MFU = tokens/s × F_tok / (peak_tflops × 1e12)，F_tok 用
    scaling.train_flops_per_token(cfg, seq) 解析式；
  * 显存峰值：从 metrics.jsonl 的 mem 列读（trainer 落盘口径），缺列则 null
    ——绝不编造；
  * 回填只写 measured，estimate 保持模板原值；同字段两者都有时自动算
    deviation_pct 并在 >20% 时把 cause 标记为 "REVIEW"（诚实规则，
    由人补写原因）。

本脚本不建模型、不碰 GPU；CPU 上对任意 run 目录可跑。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEMPLATE = ROOT / "artifacts" / "4090d" / "measurement_template.json"


def read_metrics(run_dir: Path):
    rows = []
    mp = run_dir / "metrics.jsonl"
    if not mp.exists():
        return rows
    for line in mp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def steady_median(rows, key):
    """跳过前 10 步预热后取中位数；缺列返回 None。"""
    vals = [r[key] for r in rows[10:] if key in r and r[key] is not None]
    if not vals:
        return None
    try:
        return statistics.median(vals)
    except statistics.StatisticsError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="artifacts/4090d/phase0_measurements.json")
    ap.add_argument("--peak-tflops", type=float, default=157.0,
                    help="4090D bf16 稠密峰值（规格估算；实测可覆盖）")
    ap.add_argument("--preset", default="tuan-2b")
    ap.add_argument("--seq-len", type=int, default=4096)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out)
    doc = json.loads(TEMPLATE.read_text(encoding="utf-8"))

    # run.json（trainer 启动头）给出实际形状
    run_json = run_dir / "run.json"
    micro, seq, n_params = args.seq_len, args.seq_len, None
    if run_json.exists():
        head = json.loads(run_json.read_text(encoding="utf-8"))
        micro = int(head.get("micro_batch", micro))
        seq = int(head.get("seq_len_initial", seq))
        n_params = head.get("n_params")
        doc["phase1"]["config_locked"]["micro_batch"] = {"measured": micro}
        doc["phase1"]["config_locked"]["train_chunk"] = {
            "measured": head.get("seq_len_max")}
        locked = head.get("argv") or []
        doc["phase1"]["config_locked"]["optimizer"]["measured"] = (
            "adafactor" if any("adafactor" in str(a) for a in locked)
            else ("adamw" if any("adamw" in str(a) for a in locked) else None))

    rows = read_metrics(run_dir)
    if not rows:
        print(f"[fill] no metrics.jsonl under {run_dir} —— 只回填 run.json 可得项")
    # trainer 实际落盘键：tokens_per_s（整 log 窗口口径，main.py 已防
    # "末步 dt 冒充窗口吞吐" 的虚高）与 peak_mem_gb（GiB×1e9 口径）。
    tps = steady_median(rows, "tokens_per_s")
    if tps is not None:
        doc["phase0"]["throughput"]["tokens_per_sec"] = {
            "measured": round(tps, 1),
            "caliber": doc["phase0"]["throughput"]["tokens_per_sec"]["caliber"]}
        # MFU 需要 F_tok：解析式（不建模型）
        try:
            from simplcore.core.config import get_preset
            from simplcore.core.scaling import train_flops_per_token
            cfg = get_preset(args.preset)
            f_tok = float(train_flops_per_token(cfg, args.seq_len))
            mfu = tps * f_tok / (args.peak_tflops * 1e12)
            doc["phase0"]["throughput"]["mfu"] = {
                "measured": round(mfu, 4),
                "caliber": doc["phase0"]["throughput"]["mfu"]["caliber"]}
        except Exception as exc:                    # 缺依赖时保持 null
            print(f"[fill] MFU skipped: {exc}")
    mem = steady_median(rows, "peak_mem_gb")
    if mem is not None:
        doc["phase1"]["memory"]["mem_peak_gib"] = {"measured": round(mem, 2)}
    if rows:
        last = rows[-1]
        toks = last.get("tokens")
        if toks is not None:
            doc["phase1"]["budget"]["tokens_total"] = {"measured": int(toks)}
            if n_params:
                doc["phase1"]["budget"]["tokens_per_param"] = {
                    "measured": round(int(toks) / n_params, 4)}

    # 诚实规则：estimate 与 measured 并存 → deviation；>20% 标 REVIEW
    devs = []

    def walk(node, path=""):
        if isinstance(node, dict):
            if "measured" in node and "estimate" in node \
                    and node["measured"] is not None \
                    and node["estimate"]:
                est, mea = float(node["estimate"]), float(node["measured"])
                d = abs(mea - est) / est * 100.0
                flag = "REVIEW" if d > 20.0 else "ok"
                devs.append({"field": path, "estimate": est,
                             "measured": mea,
                             "deviation_pct": round(d, 2), "cause": flag})
            for k, v in node.items():
                walk(v, f"{path}/{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    doc["deviations"]["rows"] = devs

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    n_filled = sum(1 for d in devs)
    print(f"[fill] wrote {out_path} | deviations tracked: {n_filled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
