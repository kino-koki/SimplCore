# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""把多个 run 目录的对话评测放在同一张表里比较（不重训）。

用法::

    python scripts/compare_dialogue_runs.py runs/talk2 runs/srf_5m --ckpt best.pt

存在的理由：A/B 的结论只能来自"同数据、同步数、同判据"的三次评测并排看。
手工跑三遍 ``eval_checkpoint.py`` 再复制粘贴，一定会漏掉某一列或者拿错
checkpoint —— 那个错误比没有结论更糟。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplcore.core.chat import ChatTemplate, SpecialTokenCodec
from simplcore.core.config import SimplCoreConfig
from simplcore.core.dialogue_eval import (BASIC_DIALOGUE_SUITE,
                                         CHAT_DIALOGUE_SUITE,
                                         CHAT_TOPIC_SUITE,
                                         evaluate_dialogue)
from simplcore.core.model import SimplCoreModel
from simplcore.core.tokenization import CharVocab

_SUITES = {"basic": BASIC_DIALOGUE_SUITE,
           "chat": CHAT_DIALOGUE_SUITE,
           "topic": CHAT_TOPIC_SUITE}


def evaluate(run_dir: str, ckpt_name: str, suites):
    vocab = CharVocab.load(os.path.join(run_dir, "charvocab.json"))
    codec = SpecialTokenCodec(vocab)
    ckpt = torch.load(os.path.join(run_dir, ckpt_name),
                      map_location="cpu", weights_only=False)
    cfg = SimplCoreConfig.from_dict(ckpt["config"])
    model = SimplCoreModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = {"run": os.path.basename(run_dir.rstrip("/\\")),
           "mixer": cfg.mixer,
           "params": sum(p.numel() for p in model.parameters()),
           "step": ckpt.get("step"),
           "val_best": ckpt.get("best_val")}
    tmpl = ChatTemplate()
    for name, tasks in suites.items():
        rep = evaluate_dialogue(model, codec, tasks, tmpl)
        s = rep.summary()
        out[name] = s["task_pass_rate"]
        out[f"{name}_fmt"] = s["format_ok"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--suite", action="append", choices=sorted(_SUITES))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    suites = {k: _SUITES[k] for k in (args.suite or ["topic", "chat", "basic"])}
    rows = [evaluate(r, args.ckpt, suites) for r in args.runs]

    cols = list(suites)
    print(f"{'run':>12s} {'mixer':>6s} {'params':>11s} {'step':>6s} {'val':>7s}  "
          + "  ".join(f"{c:>7s}" for c in cols))
    print("-" * (52 + 9 * len(cols)))
    for r in rows:
        vals = "  ".join(f"{r.get(c, float('nan')):>7.3f}" for c in cols)
        vb = r.get("val_best")
        vb_s = f"{vb:.4f}" if isinstance(vb, float) else "—"
        print(f"{r['run']:>12s} {r['mixer']:>6s} {r['params']:>11,} "
              f"{str(r['step']):>6s} {vb_s:>7s}  {vals}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print(f"\n写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
