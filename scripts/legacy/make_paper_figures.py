# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""生成论文级 SVG 图表（无第三方依赖，纯标准库）。

从各 bench_*.json 读取实测数据，输出矢量 SVG 图：
  * fig_scaling_curve.svg   —— 规模扩展曲线（val_ppl vs 参数量）
  * fig_ablation.svg        —— 组件消融柱状图（各变体 val_ppl）
  * fig_rfp_latency.svg     —— RFP vs 注意力延迟曲线（log-log）

用法: python make_paper_figures.py [输出目录]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

W, H = 720, 460
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 78, 24, 40, 56
PLOT_W = W - MARGIN_L - MARGIN_R
PLOT_H = H - MARGIN_T - MARGIN_B

PALETTE = {
    "simplcore": "#2563eb",
    "baseline": "#dc2626",
    "rfp": "#2563eb",
    "attn": "#dc2626",
    "full": "#2563eb",
    "variant": "#64748b",
}


def _svg_header() -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">')


def _axes(xmin, xmax, ymin, ymax, xlabel, ylabel, xticks, yticks, xlog=False, ylog=False):
    """绘制坐标轴与刻度，返回 (x_scale, y_scale) 映射函数。"""
    def _x(v):
        if xlog:
            return MARGIN_L + PLOT_W * (math.log10(v) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
        return MARGIN_L + PLOT_W * (v - xmin) / (xmax - xmin)

    def _y(v):
        if ylog:
            return MARGIN_T + PLOT_H * (1 - (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin)))
        return MARGIN_T + PLOT_H * (1 - (v - ymin) / (ymax - ymin))

    parts = [f'<line x1="{MARGIN_L}" y1="{MARGIN_T + PLOT_H}" x2="{MARGIN_L + PLOT_W}" y2="{MARGIN_T + PLOT_H}" stroke="#94a3b8" stroke-width="1"/>',
             f'<line x1="{MARGIN_L}" y1="{MARGIN_T}" x2="{MARGIN_L}" y2="{MARGIN_T + PLOT_H}" stroke="#94a3b8" stroke-width="1"/>',
             f'<text x="{MARGIN_L + PLOT_W / 2}" y="{H - 14}" text-anchor="middle" font-size="13" fill="#334155">{xlabel}</text>',
             f'<text x="16" y="{MARGIN_T + PLOT_H / 2}" text-anchor="middle" font-size="13" fill="#334155" transform="rotate(-90 16 {MARGIN_T + PLOT_H / 2})">{ylabel}</text>']
    for v in xticks:
        parts.append(f'<line x1="{_x(v):.1f}" y1="{MARGIN_T + PLOT_H}" x2="{_x(v):.1f}" y2="{MARGIN_T + PLOT_H + 4}" stroke="#94a3b8" stroke-width="1"/>')
        parts.append(f'<text x="{_x(v):.1f}" y="{MARGIN_T + PLOT_H + 18}" text-anchor="middle" font-size="11" fill="#475569">{v}</text>')
    for v in yticks:
        parts.append(f'<line x1="{MARGIN_L - 4}" y1="{_y(v):.1f}" x2="{MARGIN_L}" y2="{_y(v):.1f}" stroke="#94a3b8" stroke-width="1"/>')
        parts.append(f'<text x="{MARGIN_L - 8}" y="{_y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{v}</text>')
    parts.append('')
    return _x, _y, parts


def _title(text: str) -> str:
    return f'<text x="{W / 2}" y="24" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">{text}</text>'


def _line_series(xs, ys, xmap, ymap, color, label, dashed=False, marker=True):
    pts = " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y in zip(xs, ys))
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    parts = [f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>']
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="4.5" fill="{color}"/>')
    parts.append(f'<text x="{xmap(xs[-1]) + 6:.1f}" y="{ymap(ys[-1]) - 6:.1f}" font-size="12" fill="{color}">{label}</text>')
    return parts


def _bar_chart(cats, vals, xmap, ymap, colors, labels=None):
    n = len(cats)
    bw = min(PLOT_W / n * 0.6, 64)
    parts = []
    for i, (c, v) in enumerate(zip(cats, vals)):
        cx = MARGIN_L + PLOT_W * (i + 0.5) / n
        y0 = ymap(0)
        y1 = ymap(v)
        parts.append(f'<rect x="{cx - bw / 2:.1f}" y="{min(y0, y1):.1f}" width="{bw:.1f}" height="{abs(y1 - y0):.1f}" fill="{colors[i]}" rx="3"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y1 - 6:.1f}" text-anchor="middle" font-size="12" font-weight="bold" fill="#0f172a">{v}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{MARGIN_T + PLOT_H + 18}" text-anchor="middle" font-size="11" fill="#475569">{c}</text>')
        if labels:
            parts.append(f'<text x="{cx:.1f}" y="{y1 + 16:.1f}" text-anchor="middle" font-size="10" fill="#64748b">{labels[i]}</text>')
    return parts


def fig_scaling(data: dict) -> str:
    rows = data["results"]
    # 主曲线：匹配 200 步预算的三个点（H=64/96/128@200）；H=128@400 作为补充点
    main = [r for r in rows if r.get("steps", 200) == 200]
    supp = [r for r in rows if r.get("steps", 200) != 200]
    xs = [r["params"] for r in main]
    ys = [r["val_perplexity"] for r in main]
    xmin, xmax = 1.5e5, 1.1e6
    ymin, ymax = 1.20, 1.30
    _x, _y, parts = _axes(xmin, xmax, ymin, ymax, "参数量 (params)", "验证集困惑度 (val_ppl ↓)",
                          xticks=[2e5, 4e5, 6e5, 8e5, 1e6], yticks=[1.20, 1.22, 1.24, 1.26, 1.28, 1.30], xlog=True)
    parts.append(_title("规模扩展：无注意力架构困惑度随参数量下降"))
    parts += _line_series(xs, ys, _x, _y, PALETTE["simplcore"], "SimplCore (200 步)")
    for r in main:
        parts.append(f'<text x="{_x(r["params"]) + 6:.1f}" y="{_y(r["val_perplexity"]) - 8:.1f}" font-size="11" fill="#334155">H={r["hidden"]},L={r["layers"]}</text>')
    for r in supp:
        parts.append(f'<circle cx="{_x(r["params"]):.1f}" cy="{_y(r["val_perplexity"]):.1f}" r="5" fill="none" stroke="#16a34a" stroke-width="2.5"/>')
        parts.append(f'<text x="{_x(r["params"]) + 6:.1f}" y="{_y(r["val_perplexity"]) - 8:.1f}" font-size="11" fill="#16a34a">H=128,L=3 (400 步, 收敛)</text>')
    parts.append(f'<text x="{MARGIN_L}" y="{MARGIN_T - 6}" font-size="11" fill="#64748b">val_ppl 越低越好；H=128@200 步欠训练（虚线趋势），延至 400 步收敛到 1.2294</text>')
    parts.append('</svg>')
    return _svg_header() + "\n".join(parts)


def fig_ablation(data: dict) -> str:
    rows = data["results"]
    order = ["full", "no_recurrence", "no_dmc", "no_rfp", "no_conv"]
    cats = [r["variant"] for r in rows if r["variant"] in order]
    vals = [r["val_perplexity"] for r in rows if r["variant"] in order]
    labels = [f'+{round((r["val_perplexity"] / rows[0]["val_perplexity"] - 1) * 100, 1)}%' if r["variant"] != "full" else "基线"
              for r in rows if r["variant"] in order]
    colors = [PALETTE["full"] if c == "full" else PALETTE["variant"] for c in cats]
    ymin, ymax = 1.22, 1.34
    _x, _y, parts = _axes(0, 1, ymin, ymax, "变体", "验证集困惑度 (val_ppl ↓)",
                          xticks=[], yticks=[1.22, 1.24, 1.26, 1.28, 1.30, 1.32, 1.34])
    parts.append(_title("组件消融：逐组件归因（真实中文语料）"))
    parts += _bar_chart(cats, vals, _x, _y, colors, labels)
    parts.append(f'<text x="{MARGIN_L}" y="{MARGIN_T - 6}" font-size="11" fill="#64748b">移除因果卷积损失最大 (+6.6%)；RFP/DMC/递归各贡献互补增益，完整架构最优</text>')
    parts.append('</svg>')
    return _svg_header() + "\n".join(parts)


def fig_rfp_latency(data: dict) -> str:
    rows = data["rows"]
    ns = [r["seq_len"] for r in rows]
    rfp = [r["rfp_ms"] for r in rows]
    attn = [r["attn_ms"] for r in rows]
    xmin, xmax = 64, 8192
    ymin, ymax = 0.05, 300
    _x, _y, parts = _axes(xmin, xmax, ymin, ymax, "序列长度 n (log)", "前向延迟 ms (log)",
                          xticks=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
                          yticks=[0.1, 1, 10, 100], xlog=True, ylog=True)
    parts.append(_title("谱全局卷积 (RFP) vs 标准注意力：延迟标度"))
    parts += _line_series(ns, rfp, _x, _y, PALETTE["rfp"], "RFP (O(n log n))")
    parts += _line_series(ns, attn, _x, _y, PALETTE["attn"], "SDPA 注意力 (O(n²))", dashed=True)
    parts.append(f'<text x="{MARGIN_L}" y="{MARGIN_T - 6}" font-size="11" fill="#64748b">n=4096 时 RFP 快 16.2×；差距随 n 单调扩大（CPU, hidden=256, no_grad）</text>')
    parts.append('</svg>')
    return _svg_header() + "\n".join(parts)


def fig_learning_curve(data: dict) -> str:
    """SimplCore vs Baseline 训练损失曲线（真实中文语料）。"""
    rows = {r["model"]: r for r in data["results"]}
    ymin, ymax = 0.0, 6.5
    _x, _y, parts = _axes(0, data["steps"], ymin, ymax, "训练步数", "训练损失 (train loss ↓)",
                          xticks=[0, 50, 100, 150, 200, 250, 300], yticks=[0, 1, 2, 3, 4, 5, 6])
    parts.append(_title("学习曲线：SimplCore（无注意力）vs Transformer 基线"))
    for name, color in [("SimplCore", PALETTE["simplcore"]), ("Baseline", PALETTE["baseline"])]:
        r = rows.get(name)
        if not r or not r.get("loss_curve"):
            continue
        xs = [p["step"] for p in r["loss_curve"]]
        ys = [p["loss"] for p in r["loss_curve"]]
        parts += _line_series(xs, ys, _x, _y, color, f"{name} (val_ppl={r['val_perplexity']})",
                              dashed=(name == "Baseline"))
    parts.append(f'<text x="{MARGIN_L}" y="{MARGIN_T - 6}" font-size="11" fill="#64748b">'
                 f'Same 数据/协议；SimplCore 收敛更快且 val_ppl 更低（{rows.get("SimplCore", {}).get("val_perplexity")} vs {rows.get("Baseline", {}).get("val_perplexity")}）</text>')
    parts.append('</svg>')
    return _svg_header() + "\n".join(parts)


def main() -> None:
    root = Path(__file__).resolve().parent
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "docs" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("fig_scaling_curve.svg", fig_scaling, json.loads((root / "bench_scaling.json").read_text(encoding="utf-8"))),
        ("fig_ablation.svg", fig_ablation, json.loads((root / "bench_ablation.json").read_text(encoding="utf-8"))),
        ("fig_rfp_latency.svg", fig_rfp_latency, json.loads((root / "bench_rfp_vs_attention.json").read_text(encoding="utf-8"))),
        ("fig_learning_curve.svg", fig_learning_curve, json.loads((root / "bench_corpus_ppl.json").read_text(encoding="utf-8"))),
    ]
    for name, fn, data in jobs:
        (outdir / name).write_text(fn(data), encoding="utf-8")
        print(f"已生成 {outdir / name}")


if __name__ == "__main__":
    main()
