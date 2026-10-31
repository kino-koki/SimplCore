# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""从一个已保存的 checkpoint 直接跑评测（不重训）。

为什么需要它
------------
调整评测套件（例如换掉含词表外字符的题）时，重训一次要半小时以上，
而结论其实只取决于 checkpoint + 套件。把这两者解耦，改判据就不用重训。

用法::

    python scripts/eval_checkpoint.py runs/talk1 --suite basic
    python scripts/eval_checkpoint.py runs/talk1 --suite chat --suite basic
    python scripts/eval_checkpoint.py runs/mem_fix2 --suite memory

约定目录里必须有 ``charvocab.json`` 与 ``last.pt``（训练脚本都会写）；
``--suite memory`` 还需要 ``memory_encoder.json``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplcore.core.chat import (ChatTemplate, SpecialTokenCodec,
                                load_memory_encoder)
from simplcore.core.config import SimplCoreConfig
from simplcore.core.dialogue_eval import (BASIC_DIALOGUE_SUITE,
                                         CHAT_DIALOGUE_SUITE,
                                         CHAT_TOPIC_SUITE,
                                         MEMORY_DIALOGUE_SUITE,
                                         evaluate_dialogue, format_report,
                                         memory_ablation)
from simplcore.core.model import SimplCoreModel
from simplcore.core.tokenization import CharVocab

_SUITES = {
    "basic": lambda: BASIC_DIALOGUE_SUITE,
    "chat": lambda: CHAT_DIALOGUE_SUITE,
    "topic": lambda: CHAT_TOPIC_SUITE,
    "memory": lambda: MEMORY_DIALOGUE_SUITE,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="checkpoint 评测")
    ap.add_argument("run_dir", help="训练输出目录（含 charvocab.json / last.pt）")
    ap.add_argument("--suite", action="append", choices=sorted(_SUITES),
                    help="可重复；默认 basic")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--json", default=None, help="把结果另存为 JSON")
    args = ap.parse_args(argv)

    suites = args.suite or ["basic"]
    vocab = CharVocab.load(os.path.join(args.run_dir, "charvocab.json"))
    codec = SpecialTokenCodec(vocab)

    ckpt = torch.load(os.path.join(args.run_dir, args.ckpt),
                      map_location="cpu", weights_only=False)
    cfg = SimplCoreConfig.from_dict(ckpt["config"])
    model = SimplCoreModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    template = ChatTemplate()
    out = {"run": args.run_dir, "ckpt": args.ckpt,
           "step": ckpt.get("step"), "params": sum(p.numel() for p in model.parameters())}

    for name in suites:
        tasks = _SUITES[name]()

        if name == "memory":
            enc_path = os.path.join(args.run_dir, "memory_encoder.json")
            enc = (load_memory_encoder(enc_path, dim=cfg.dim)
                   if os.path.exists(enc_path) else None)
            rep = memory_ablation(model, codec, tasks, template,
                                  memory_encoder=enc)
            print(f"\n[memory] {rep['verdict']}")
            for r in rep["per_task"]:
                print(f"  {r['id']:<14} 记忆 {'OK ' if r['with_memory'] else 'FAIL'}"
                      f"  无记忆 {'OK ' if r['without_memory'] else 'FAIL'}"
                      f"  {r['reply_with']!r}")
            out[name] = rep
            continue

        rep = evaluate_dialogue(model, codec, tasks, template)
        print(f"\n[{name}]")
        print(format_report(rep))
        out[name] = rep.as_dict()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
