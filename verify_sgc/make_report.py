"""
读取 results/results.json，生成纯 SVG 对比曲线图 + HTML 报告（不依赖 matplotlib）。
输出: results/curve_loss.svg, results/curve_acc.svg, results/report.html
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
with open(os.path.join(RES, "results.json")) as f:
    data = json.load(f)

W, H = 680, 360
def svg_linechart(title, series, ylabel, ymax=None, ymin=0.0, fmt="{:.2f}"):
    """series: list of (label, color, [(x,y),...])"""
    pad_l, pad_r, pad_t, pad_b = 60, 20, 40, 40
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    all_y = [y for _, _, pts in series for _, y in pts]
    all_x = [x for _, _, pts in series for x, _ in pts]
    if not all_y:
        return ""
    xmax = max(all_x) if all_x else 1
    xmin = min(all_x) if all_x else 0
    if ymax is None:
        ymax = max(all_y) * 1.05 if all_y else 1
    ymin = min(ymin, min(all_y)) if all_y else 0
    def X(x): return pad_l + (x - xmin) / (xmax - xmin + 1e-9) * plot_w
    def Y(y): return pad_t + (ymax - y) / (ymax - ymin + 1e-9) * plot_h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="monospace" font-size="11">']
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#0f1420"/>')
    parts.append(f'<text x="{W/2}" y="20" fill="#e6e6e6" text-anchor="middle" font-size="13">{title}</text>')
    # 网格 + y 轴刻度
    for i in range(5):
        yy = ymin + (ymax - ymin) * i / 4
        py = Y(yy)
        parts.append(f'<line x1="{pad_l}" y1="{py:.1f}" x2="{W-pad_r}" y2="{py:.1f}" stroke="#2a3142"/>')
        parts.append(f'<text x="{pad_l-6}" y="{py+4:.1f}" fill="#9aa4b2" text-anchor="end">{fmt.format(yy)}</text>')
    # x 轴刻度
    for i in range(5):
        xx = xmin + (xmax - xmin) * i / 4
        px = X(xx)
        parts.append(f'<text x="{px:.1f}" y="{H-pad_b+16}" fill="#9aa4b2" text-anchor="middle">{int(xx)}</text>')
    parts.append(f'<text x="{pad_l-44}" y="{pad_t-14}" fill="#9aa4b2">{ylabel}</text>')
    parts.append(f'<text x="{W-pad_r}" y="{H-pad_b+34}" fill="#9aa4b2" text-anchor="end">step</text>')
    # 折线
    for label, color, pts in series:
        if not pts: continue
        d = " ".join(f"L{x:.1f},{Y(y):.1f}" if i else f"M{x:.1f},{Y(y):.1f}" for i,(x,y) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    # 图例
    ly = pad_t + 6
    for label, color, pts in series:
        parts.append(f'<rect x="{pad_l+6}" y="{ly}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{pad_l+20}" y="{ly+9}" fill="#e6e6e6">{label}</text>')
        ly += 16
    parts.append('</svg>')
    return "\n".join(parts)

# 准备序列
attn = next((d for d in data if d["mixer"] == "attn"), None)
sgc = next((d for d in data if d["mixer"] == "sgc"), None)

loss_series, acc_series = [], []
if attn:
    loss_series.append(("Attention loss", "#ff6b6b", [(h["step"], h["loss"]) for h in attn["history"]]))
    acc_series.append(("Attention val_acc", "#ff6b6b", [(h["step"], h["val_acc"]*100) for h in attn["history"]]))
if sgc:
    loss_series.append(("SGC loss", "#4dd4ff", [(h["step"], h["loss"]) for h in sgc["history"]]))
    acc_series.append(("SGC val_acc", "#4dd4ff", [(h["step"], h["val_acc"]*100) for h in sgc["history"]]))

loss_svg = svg_linechart("Train Loss", loss_series, "loss", ymax=None)
acc_svg = svg_linechart("Validation Accuracy (selective copy)", acc_series, "acc %", ymin=0, ymax=100, fmt="{:.0f}")

with open(os.path.join(RES, "curve_loss.svg"), "w") as f: f.write(loss_svg)
with open(os.path.join(RES, "curve_acc.svg"), "w") as f: f.write(acc_svg)

# HTML
def row(d):
    if not d: return ""
    return f"<tr><td>{d['tag']}</td><td>{d['params']:,}</td><td>{d['final_val_acc']*100:.2f}%</td><td>{d['time_s']:.0f}s</td></tr>"

a_acc = attn["final_val_acc"]*100 if attn else float('nan')
s_acc = sgc["final_val_acc"]*100 if sgc else float('nan')
delta = (s_acc - a_acc) if (attn and sgc) else 0
better = "SGC" if s_acc > a_acc else ("Attention" if a_acc > s_acc else "tie")

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<style>body{{background:#0b0e14;color:#e6e6e6;font-family:system-ui,monospace;max-width:760px;margin:24px auto;padding:0 16px}}
h1{{font-size:20px}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #2a3142;padding:6px 10px;text-align:left}}
th{{background:#161b26}}.card{{background:#11151f;border:1px solid #2a3142;border-radius:8px;padding:10px;margin:12px 0}}
.k{{color:#9aa4b2}}.v{{color:#4dd4ff;font-weight:bold}}</style></head><body>
<h1>SGC vs Attention 下游有效性验证</h1>
<p class="k">任务：Selective Copy（长程依赖探针）。两模型仅混合层不同（Attention / SGC），其余骨架一致。</p>
<div class="card"><table>
<tr><th>模型</th><th>参数量</th><th>最终验证准确率</th><th>训练耗时</th></tr>
{row(attn)}{row(sgc)}</table></div>
<div class="card"><span class="k">相对表现：</span><span class="v">{better}</span>
<span class="k">　Δ = </span><span class="v">{delta:+.2f} pp</span></div>
<div class="card">{loss_svg}</div>
<div class="card">{acc_svg}</div>
<p class="k">说明：纯自研 simplcore.tensor 后端（CPU/NumPy），模型规模经缩减以在当前环境跑完。
SGC 为谱全局卷积（FFT），Attention 为因果自注意力。两者均可见全序列（全局混合），公平对比表达能力。</p>
</body></html>"""
with open(os.path.join(RES, "report.html"), "w") as f: f.write(html)
print("report written:", os.path.join(RES, "report.html"))
print(f"final_acc  attn={a_acc:.2f}%  sgc={s_acc:.2f}%  better={better} delta={delta:+.2f}pp")
