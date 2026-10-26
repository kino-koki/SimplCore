# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Export JSON training configs from the SGC presets.

Keeping the configs generated (rather than hand-maintained) guarantees they cannot
drift away from `simplcore/sgc/config.py`.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simplcore.sgc.config import (  # noqa: E402
    SGC_PRESETS,
    SGCConfig,
    analytic_params,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="override the vocabulary size (must match the trained tokenizer)")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    names = args.only or [n for n in SGC_PRESETS if not n.startswith(("debug", "toy"))]
    rows = []
    for name in names:
        cfg = SGCConfig.from_dict(SGC_PRESETS[name].to_dict())
        if args.vocab_size:
            cfg.vocab_size = args.vocab_size
        path = os.path.join(args.out, name.replace(".", "_") + ".json")
        cfg.to_json(path)
        rows.append((name, analytic_params(cfg), path))
    print(f"{'preset':14s} {'params':>13s}  file")
    for name, n, path in rows:
        print(f"{name:14s} {n:13,d}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
