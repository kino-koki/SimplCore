"""根据 results/v2_800.json (或指定的 json) 生成 SGC vs Attention 对比 SVG 图。"""
import os, sys, json, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
res_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "v2_800.json")
with open(res_file) as f:
    data = json.load(f)
attn = data["attn"]; sgc = data["sgc"]
aa = st.mean(r["acc"] for r in attn) * 100
sa = st.mean(r["acc"] for r in sgc) * 100
al = st.mean(r["loss"] for r in attn)
sl = st.mean(r["loss"] for r in sgc)
at = st.mean(r["time"] for r in attn)
stt = st.mean(r["time"] for r in sgc)

W, H = 680, 260
# 三组条形: acc(%), loss(x100), time(s) 各自归一化
def bar(x, y, w, h, color, label, val):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{color}"/>'
            f'<text x="{x+w/2}" y="{y+h+16}" fill="#e6e6e6" font-size="13" text-anchor="middle">{label}</text>'
            f'<text x="{x+w/2}" y="{y-6}" fill="#fff" font-size="12" text-anchor="middle">{val}</text>')

groups = [
    ("val_acc(%)", aa, sa, 0, 50, "%.2f"),
    ("loss(x100)", al*100, sl*100, 0, 200, "%.1f"),
    ("time(s)", at, stt, 0, 200, "%.0f"),
]
x0 = 90; gap = 200
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
svg.append(f'<rect width="{W}" height="{H}" fill="#0f1115"/>')
svg.append(f'<text x="20" y="28" fill="#7ee787" font-size="15">SGC v2 vs Attention (多seed平均)</text>')
for i, (name, av, sv, lo, hi, fmt) in enumerate(groups):
    gx = x0 + i * gap
    # 归一化到高度
    def yn(v):
        return 180 - (v - lo) / (hi - lo) * 150 if hi > lo else 180
    svg.append(bar(gx, yn(av), 60, 180 - yn(av), "#58a6ff", "Attn", fmt % av))
    svg.append(bar(gx + 75, yn(sv), 60, 180 - yn(sv), "#7ee787", "SGC", fmt % sv))
    svg.append(f'<text x="{gx+65}" y="210" fill="#e6e6e6" font-size="12" text-anchor="middle">{name}</text>')
svg.append('</svg>')
out = os.path.join(HERE, "results", "compare_bars.svg")
with open(out, "w") as f:
    f.write("\n".join(svg))
print("wrote", out)
