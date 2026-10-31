# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""GATE-2：长上下文课程过半时的 needle 召回测量门。

为什么是"测量门"而不是"质量门"
--------------------------------
论文轴 D 明确承认 O(1) 状态在精确检索上**必输**给能注意的 TS。GATE-2 的目标
不是要求 SGC 赢，而是回答一个此前没人实测的问题：**4M 课程训到一半时，
needle 召回到底是多少**。因此：

* ``recall < chance`` → ``below_chance``：大概率是管线/模型坏了（硬失败候选）；
* ``chance <= recall < 2×chance`` → ``at_chance``：符合"压缩状态丢精确副本"的预期；
* ``recall >= 2×chance`` → ``pass``：模型确实在部分检索，记录但不据此宣称质量。

阈值 2× 是工程门限，不是理论结果；报告同时给出逐深度最差桶与机会水平，
避免"平均分掩盖某个深度全错"。

与训练循环的关系
----------------
``train.py`` 在课程首次进入 1M 阶段（以及可选的 4M 阶段）调用
:func:`evaluate_gate2`，用 ``mode="chunked"`` 评分（``forward_chunked``），
不物化 4M 的整段核/激活。默认只记录；``--gate2-strict`` 才在
``below_chance`` 时硬停。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .longctx_eval import NeedleCorpus, score_trial

__all__ = [
    "DEFAULT_LENGTHS",
    "DEFAULT_DEPTHS",
    "PASS_MULTIPLIER",
    "summarize_gate2",
    "evaluate_gate2",
]

DEFAULT_LENGTHS: Sequence[int] = (262_144, 1_048_576)
DEFAULT_DEPTHS: Sequence[float] = (0.25, 0.5, 0.75)
#: "pass" 门限：召回至少是机会水平的 2 倍。低于 1× 视为管线异常。
PASS_MULTIPLIER = 2.0


def summarize_gate2(trials: Sequence[Dict[str, Any]], *,
                    n_values: int) -> Dict[str, Any]:
    """把逐 trial 结果汇总成 GATE-2 判定（纯函数，便于测试）。"""
    if not trials:
        raise ValueError("trials must not be empty")
    n = len(trials)
    n_ok = sum(1 for t in trials if t.get("correct"))
    recall = n_ok / n
    chance = 1.0 / max(int(n_values), 1)
    by_depth: Dict[float, Dict[str, Any]] = {}
    for t in trials:
        d = float(t.get("depth", 0.0))
        bucket = by_depth.setdefault(d, {"n": 0, "ok": 0})
        bucket["n"] += 1
        bucket["ok"] += int(bool(t.get("correct")))
    depth_recall = {d: b["ok"] / b["n"] for d, b in by_depth.items()}
    worst_depth = min(depth_recall.values())
    if recall < chance:
        status = "below_chance"
    elif recall < PASS_MULTIPLIER * chance:
        status = "at_chance"
    else:
        status = "pass"
    return {
        "n_trials": n,
        "n_correct": n_ok,
        "recall": recall,
        "chance": chance,
        "pass_threshold": PASS_MULTIPLIER * chance,
        "worst_depth_recall": worst_depth,
        "per_depth": {str(k): v for k, v in sorted(depth_recall.items())},
        "status": status,
        "passed": status in ("pass", "at_chance"),
    }


def evaluate_gate2(model: Any, *,
                   lengths: Sequence[int] = DEFAULT_LENGTHS,
                   depths: Sequence[float] = DEFAULT_DEPTHS,
                   trials_per_depth: int = 1,
                   chunk_size: int = 4096,
                   seed: int = 2026,
                   out_path: Optional[Any] = None) -> Dict[str, Any]:
    """在给定模型上跑 GATE-2，返回并（可选）落盘报告。

    ``model`` 必须是已加载权重的 ``SimplCoreModel``（训练中传 ``raw_model()``），
    且所有 mixer 支持 ``forward_chunk``；否则直接报错，不静默退回整段。
    """
    model.eval()
    rows: List[Dict[str, Any]] = []
    all_trials: List[Dict[str, Any]] = []
    for L in lengths:
        L = int(L)
        corpus = NeedleCorpus(L, seed=seed)
        t0 = time.time()
        trials: List[Dict[str, Any]] = []
        for di, d in enumerate(depths):
            for j in range(int(trials_per_depth)):
                idx = di * int(trials_per_depth) + j
                tr = corpus.make_trial(idx, float(d))
                r = score_trial(model, tr, mode="chunked",
                                chunk_size=chunk_size)
                r["depth"] = float(d)
                trials.append(r)
                all_trials.append(r)
        summary = summarize_gate2(trials, n_values=corpus.n_values)
        rows.append({
            "seq_len": L,
            "chunk_size": int(chunk_size),
            "seconds": round(time.time() - t0, 2),
            **summary,
        })

    overall = summarize_gate2(all_trials,
                              n_values=NeedleCorpus(64).n_values)
    report = {
        "protocol": "GATE-2",
        "report_version": 1,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lengths": [int(L) for L in lengths],
        "depths": [float(d) for d in depths],
        "trials_per_depth": int(trials_per_depth),
        "chunk_size": int(chunk_size),
        "overall": overall,
        "per_length": rows,
        "status": overall["status"],
        "passed": overall["passed"],
        "note": ("GATE-2 是测量门：轴 D 承认 O(1) 状态在精确检索上必输。"
                 "below_chance=管线/模型异常；at_chance=符合压缩预期；"
                 "pass=部分检索（不据此宣称质量优势）。"),
    }
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return report

def main(argv: Optional[Sequence[str]] = None) -> int:
    """独立跑 GATE-2（4M 验收用）：从 checkpoint 加载，不启动训练。"""
    ap = argparse.ArgumentParser(description="GATE-2 needle 召回测量门")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lengths", default="1048576,4000000")
    ap.add_argument("--depths", default="0.25,0.5,0.75")
    ap.add_argument("--trials-per-depth", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    from .config import SimplCoreConfig
    from .model import SimplCoreModel

    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    cfg = SimplCoreConfig(**ckpt["config"])
    model = SimplCoreModel(cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device).eval()

    rep = evaluate_gate2(
        model,
        lengths=[int(x) for x in args.lengths.split(",") if x.strip()],
        depths=[float(x) for x in args.depths.split(",") if x.strip()],
        trials_per_depth=args.trials_per_depth,
        chunk_size=args.chunk_size,
        out_path=args.out or str(Path(args.checkpoint).with_suffix(".gate2.json")),
    )
    print(f"[gate2] {rep['status']} recall={rep['overall']['recall']:.3f} "
          f"chance={rep['overall']['chance']:.3f} "
          f"worst_depth={rep['overall']['worst_depth_recall']:.3f}")
    return 0 if rep["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())