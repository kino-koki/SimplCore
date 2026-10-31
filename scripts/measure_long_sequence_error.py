#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测量 fp32 双形式误差随 L 的曲线（补论文 §6(d) 的长序列实测缺口）。

用法：
    python scripts/measure_long_sequence_error.py
    python scripts/measure_long_sequence_error.py --lengths 512,1024,2048,4096,8192

输出：`runs/numerical/long_sequence_error.json`（含每个 L 的绝对/相对误差、
配置、时间戳）。这是原创清单 A 的验证集，不是训练实验。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from simplcore.core.sgc import SpectralGlobalConv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192")
    ap.add_argument("--bases", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--decay-max", type=float, default=8192.0)
    ap.add_argument("--out", default="runs/numerical/long_sequence_error.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    m = SpectralGlobalConv(dim=args.dim, num_bases=args.bases,
                           decay_max=args.decay_max, local_kernel=4,
                           content_gate=True, evidence_gate=True,
                           uncertainty_gate=True).eval()
    rows = []
    for L in [int(x) for x in args.lengths.split(",") if x.strip()]:
        x = torch.randn(1, L, args.dim)
        t0 = time.time()
        with torch.no_grad():
            full = m(x)
            st = m.init_state(1, x.device, x.dtype)
            outs = []
            for t in range(L):
                y, st = m.step(x[:, t:t + 1], st)
                outs.append(y)
            streamed = torch.cat(outs, dim=1)
        d = (full - streamed).abs()
        rows.append({
            "length": L,
            "max_abs": float(d.max()),
            "max_rel": float(d.max() / full.abs().max()),
            "seconds": round(time.time() - t0, 2),
            "exceeds_4e-5": bool(float(d.max()) > 4e-5),
        })
        print(f"L={L:>7,} max_abs={rows[-1]['max_abs']:.3e} "
              f"max_rel={rows[-1]['max_rel']:.3e} "
              f"({'超 4e-5' if rows[-1]['exceeds_4e-5'] else '<=4e-5'})")

    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {"num_bases": args.bases, "dim": args.dim,
                   "decay_max": args.decay_max, "local_kernel": 4,
                   "evidence_gate": True, "uncertainty_gate": True},
        "rows": rows,
        "note": ("fp32 双形式误差；<4e-5 只在短序列成立。"
                 "界与缓解见 docs/NUMERICAL_STABILITY.md"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"[measure] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())