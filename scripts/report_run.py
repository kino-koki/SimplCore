# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Turn a training run directory into a readable report and loss-curve SVG.

Reads ``<run>/metrics.jsonl`` (written by ``simplcore.sgc.train``) plus
``run.json``/``summary.json`` and emits:

  * ``report.md``  -- tables of throughput, train loss and validation perplexity,
                      with the token budget actually consumed;
  * ``curve.svg``  -- train loss and val perplexity vs tokens seen.

No plotting library is used: the SVG is written directly, so the report works on a
machine with nothing but the standard library and numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def load(run_dir: str):
    metrics = []
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no metrics.jsonl in {run_dir}")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    run = {}
    for name in ("run.json", "summary.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            run[name.split(".")[0]] = json.load(open(p, encoding="utf-8"))
    return metrics, run


def _svg(points, colour, width, height, pad, xmax, ymin, ymax, label):
    """points: list of (x, y). Returns an SVG polyline string."""
    if not points:
        return ""
    def sx(x):
        return pad + (width - 2 * pad) * (x / xmax if xmax else 0)

    def sy(y):
        span = (ymax - ymin) or 1.0
        return height - pad - (height - 2 * pad) * ((y - ymin) / span)

    pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    return (f'<polyline fill="none" stroke="{colour}" stroke-width="2" '
            f'points="{pts}" aria-label="{label}" />')


def write_svg(train_loss, val_ppl, out_path, n_params, tokens):
    W, H, PAD = 900, 420, 60
    xs = [t for t, _ in train_loss] + [t for t, _ in val_ppl]
    xmax = max(xs) if xs else 1
    ys = [v for _, v in train_loss]
    ymin, ymax = (min(ys), max(ys)) if ys else (0, 1)
    if ymax - ymin < 1e-6:
        ymax = ymin + 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{PAD}" y="24" font-size="15" font-weight="bold">'
        f'SGC training on real Chinese corpus</text>',
        f'<text x="{PAD}" y="42" fill="#555">params={n_params:,}  '
        f'tokens_seen={tokens:,}  (train loss, lower is better)</text>',
        # axes
        f'<line x1="{PAD}" y1="{H-PAD}" x2="{W-PAD}" y2="{H-PAD}" stroke="#333"/>',
        f'<line x1="{PAD}" y1="{PAD}" x2="{PAD}" y2="{H-PAD}" stroke="#333"/>',
        f'<text x="{PAD}" y="{H-PAD+20}">0</text>',
        f'<text x="{W-PAD-60}" y="{H-PAD+20}">tokens {xmax:,}</text>',
        f'<text x="6" y="{PAD+8}">{ymax:.2f}</text>',
        f'<text x="6" y="{H-PAD}">{ymin:.2f}</text>',
        _svg(train_loss, "#1f77b4", W, H, PAD, xmax, ymin, ymax, "train loss"),
    ]
    # validation perplexity on a secondary label list (printed as text, not scaled)
    if val_ppl:
        parts.append(f'<text x="{PAD+20}" y="{PAD+20}" fill="#d62728">'
                     f'val ppl: ' + ", ".join(f"{p:.0f}" for _, p in val_ppl) + '</text>')
    parts.append(f'<text x="{W-PAD-190}" y="{PAD+20}" fill="#1f77b4">'
                 f'train loss</text>')
    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="defaults to <run_dir>")
    args = ap.parse_args()
    out_dir = args.out or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    metrics, run = load(args.run_dir)
    # ``eval`` records carry no token count of their own; carry the last known one
    # forward so both series share an x-axis.
    last_tokens = 0
    for m in metrics:
        if "tokens" in m:
            last_tokens = m["tokens"]
        else:
            m["tokens"] = last_tokens
    train = [(m["tokens"], m["loss"]) for m in metrics if "loss" in m]
    val = [(m["tokens"], m["val_ppl"]) for m in metrics if "val_ppl" in m]
    tps = [m["tokens_per_s"] for m in metrics if "tokens_per_s" in m]

    tokens = train[-1][0] if train else 0
    n_params = run.get("run", {}).get("n_params", 0)

    lines = ["# SGC training run report", ""]
    if "run" in run:
        r = run["run"]
        lines += [
            f"- parameters: **{r.get('n_params', 0):,}**",
            f"- device: `{r.get('device')}`  dtype: `{r.get('amp_dtype')}`",
            f"- seq_len: {r.get('seq_len')}  micro_batch: {r.get('micro_batch')}  "
            f"grad_accum: {r.get('grad_accum')}  tokens/step: {r.get('tokens_per_step'):,}",
            f"- data: `{r.get('data_dir')}`",
            "",
        ]
    if tokens:
        lines.append(f"- tokens consumed: **{tokens:,}**")
    if tps:
        lines.append(f"- throughput: median {sorted(tps)[len(tps)//2]:,.0f} tokens/s "
                     f"(min {min(tps):,.0f}, max {max(tps):,.0f})")
    lines.append("")

    if train:
        lines += ["## Train loss", "", "| tokens | loss |", "|---|---|"]
        step = max(1, len(train) // 12)
        for t, v in train[::step] + [train[-1]]:
            lines.append(f"| {t:,} | {v:.4f} |")
        lines += ["", f"first={train[0][1]:.4f} -> last={train[-1][1]:.4f} "
                      f"(Δ={train[-1][1]-train[0][1]:+.4f})", ""]
    if val:
        lines += ["## Validation", "", "| tokens | val loss | val ppl |", "|---|---|---|"]
        for m in metrics:
            if "val_ppl" in m:
                lines.append(f"| {m['tokens']:,} | {m['val_loss']:.4f} | {m['val_ppl']:.2f} |")
        lines.append("")

    md = os.path.join(out_dir, "report.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    svg = os.path.join(out_dir, "curve.svg")
    write_svg(train, val, svg, n_params, tokens)
    print(f"wrote {md}")
    print(f"wrote {svg}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
