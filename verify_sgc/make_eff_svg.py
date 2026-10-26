# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""根据 results/efficiency_timing.json 生成效率实测 SVG (手绘, 无 matplotlib 依赖)."""
from __future__ import annotations
import os, json

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
data = json.load(open(os.path.join(RESULTS, "efficiency_timing.json")))
rows = data["timing"]
by_len = {}
for r in rows:
    by_len.setdefault(r["seq_len"], {})[r["mixer"]] = r["median_s"]
Ls = sorted(by_len)

W, H = 680, 420
pad_l, pad_r, pad_t, pad_b = 60, 30, 50, 60
plot_w = W - pad_l - pad_r
plot_h = H - pad_t - pad_b

# 最大时间用于纵轴缩放
max_t = max(v for d in by_len.values() for v in d.values())
ymax = max_t * 1.1

def x(i):  # 等距 x 位置
    return pad_l + (plot_w * i / (len(Ls) - 1)) if len(Ls) > 1 else pad_l + plot_w / 2
def y(v):
    return pad_t + plot_h * (1 - v / ymax)

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif">')
# 背景
svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
# 标题
svg.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a1a2e">SGC-v2 vs 因果注意力: 单步(前+反)耗时 vs 序列长度</text>')
# 轴
svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#333" stroke-width="1.5"/>')
svg.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#333" stroke-width="1.5"/>')
# y 轴刻度
for k in range(6):
    v = ymax * k / 5
    yy = y(v)
    svg.append(f'<line x1="{pad_l-4}" y1="{yy:.1f}" x2="{pad_l+plot_w}" y2="{yy:.1f}" stroke="#eee"/>')
    svg.append(f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="#666">{v:.1f}s</text>')
# x 轴标签
for i, L in enumerate(Ls):
    xx = x(i)
    svg.append(f'<text x="{xx:.1f}" y="{pad_t+plot_h+18}" text-anchor="middle" font-size="11" fill="#333">L={L}</text>')
# 柱状: 每个 L 两根柱 (attn / sgc)
bw = plot_w / (len(Ls) * 3)
for i, L in enumerate(Ls):
    cx = x(i)
    a_t = by_len[L].get("attn"); s_t = by_len[L].get("sgc")
    if a_t is not None:
        xa = cx - bw - 2
        svg.append(f'<rect x="{xa:.1f}" y="{y(a_t):.1f}" width="{bw:.1f}" height="{pad_t+plot_h-y(a_t):.1f}" fill="#e74c3c"/>')
        svg.append(f'<text x="{xa+bw/2:.1f}" y="{y(a_t)-4:.1f}" text-anchor="middle" font-size="9" fill="#e74c3c">{a_t:.2f}</text>')
    if s_t is not None:
        xs = cx + 2
        svg.append(f'<rect x="{xs:.1f}" y="{y(s_t):.1f}" width="{bw:.1f}" height="{pad_t+plot_h-y(s_t):.1f}" fill="#2e86de"/>')
        svg.append(f'<text x="{xs+bw/2:.1f}" y="{y(s_t)-4:.1f}" text-anchor="middle" font-size="9" fill="#2e86de">{s_t:.2f}</text>')
# 图例
svg.append(f'<rect x="{pad_l+10}" y="{pad_t+8}" width="12" height="12" fill="#e74c3c"/><text x="{pad_l+26}" y="{pad_t+18}" font-size="11" fill="#333">注意力 O(n²)</text>')
svg.append(f'<rect x="{pad_l+120}" y="{pad_t+8}" width="12" height="12" fill="#2e86de"/><text x="{pad_l+136}" y="{pad_t+18}" font-size="11" fill="#333">SGC-v2 O(n log n)</text>')
# 加速标注
for i, L in enumerate(Ls):
    a_t = by_len[L].get("attn"); s_t = by_len[L].get("sgc")
    if a_t and s_t:
        sp = (a_t / s_t - 1) * 100
        svg.append(f'<text x="{x(i):.1f}" y="{pad_t+plot_h+38:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#1a7a3c">SGC 快 {sp:.0f}%</text>')
svg.append('</svg>')

out = os.path.join(RESULTS, "efficiency_bars.svg")
open(out, "w").write("\n".join(svg))
print("saved ->", out)
