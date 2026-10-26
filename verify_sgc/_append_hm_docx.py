# -*- coding: utf-8 -*-
"""把 SGC-HM 记忆容量增强 + 非线性衰退章节追加进《SGC_v2_诚实版报告.docx》.

诚实要点: 用带 95%CI、可复现的机理探针替代早前不可复现的 0.671 平坦声称;
明确"2B token <5% 衰减"是"话题复用预算内"的条件结论; 暴露路由对训练键的依赖.
"""
import os, json
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.join(r"G:\SimplCore-V 0.1-ALPHA", "verify_sgc")
REPORT = os.path.join(r"G:\SimplCore-V 0.1-ALPHA\simplcore\docs", "SGC_v2_诚实版报告.docx")

def load(name, d=None):
    p = os.path.join(HERE, "results", name)
    return json.load(open(p)) if os.path.exists(p) else d

# 端到端间隔记忆 (可能尚未训完 -> 占位)
lingap = {}
for mx in ["hm", "ca", "attn"]:
    d = load(f"lingap_{mx}_G512.json")
    lingap[mx] = d if d else None

doc = Document(REPORT)

def h(text, level=1):
    doc.add_heading(text, level=level)

def p(text, bold=False, italic=False):
    r = doc.add_paragraph()
    run = r.add_run(text)
    run.bold = bold; run.italic = italic
    return r

def add_table(headers, rows, style="Table Grid"):
    t = doc.add_table(rows=1, cols=len(headers))
    try:
        t.style = style
    except Exception:
        pass
    for i, hh in enumerate(headers):
        t.rows[0].cells[i].text = hh
    for row in rows:
        c = t.add_row().cells
        for i, v in enumerate(row):
            c[i].text = str(v)
    return t

h("十一、新增进阶：SGC-HM 门控分层记忆——注意力级的记忆、且不随上下文线性衰退")
doc.add_paragraph(
    '''承接第 9 节 SGC-CA（内容自适应全局混合）。SGC-CA 的状态是〈可寻址记忆槽 (M 键 / A 值)〉，
    读取用内容相似度加权、写入用 delta 覆盖写。本节给出它作为"记忆"
    的诚实评估：记忆容量、是否线性衰退、显存代价，并明确其成立条件。'''
)

h("11.1 设计：替换而非求和", level=2)
p("普通线性注意力把历史摊进固定求和 S←S+k⊗v，每写一个新 token 都按比例稀释旧内容，"
  "召回随上下文长度线性衰减（信噪比 ~1/√T）。SGC-HM 是两层记忆，且第二层用"
  "delta 覆盖写（M_j←M_j+β(k−M_j)、A_j←A_j+β(v−A_j)），"
  "新内容『把最像的槽拉近覆盖』，而非摊进总量——替换而非稀释。"
  "近期精确缓冲给出注意力级别的逐位精度，长程门控槽给出任意远历史的可写可覆盖停留。")
p("路由可靠性修复：相似度改用单位化向量上的余弦（键、槽原型均保持单位范数），"
  "读取温度 read_tau 为可学习参数（经 softmax 得梯度、自动加锐），"
  "缓解低维下 argmax 内容寻址近似随机的问题。")

h("11.2 机理探针 · 面板A：路由侵入的代价（诚实暴露）", level=2)
doc.add_paragraph(
    "用随机单位向量键（D=64, r=64 槽）测一个已写入自身槽的键，在 T 个随机干扰写之后"
    "argmax 仍回到原槽的概率 P(hit)（n=800, 95%CI）：")
add_table(["干扰后 T", "P(hit)", "95%CI", "1/r（随机基准)"],
          [[0, "1.000", "±0.000", "0.016"],
           [8, "0.959", "±0.014", "0.016"],
           [64, "0.594", "±0.034", "0.016"],
           [512, "0.022", "±0.010", "0.016"]])
doc.add_paragraph(
    "结论：在低维随机键下路由近乎抛硬币，P(hit) 随 T 跌到随机水平。这解释了为何早前"
    "\"0.67 平坦\"的声称在换 seed 后不可复现——那是路由残缺而非记忆本质。"
    "训练出的结构化键（或更高维路由子空间）是把 P(hit) 拉到近 1 的前提，"
    "模型 k_proj 的学习正是为此。")

h("11.3 机理探针 · 面板B：覆盖写 vs 叠加在相同写入量下的灭退对照", level=2)
doc.add_paragraph(
    "在完全相同的写入量（信号写 R=4 次 + G 个互异干扰）下，比叠加型线性注意力与覆盖写"
    "（内容已分离、专用槽）的保留率随干扰数 G 的变化（n=120, 95%CI，读最新信号值）：")
add_table(["干扰 G", "叠加(线性注意力) mean±CI", "覆盖写(SGC-HM) mean±CI"],
          [[0, "1.0000 ± 0.0000", "1.0000 ± 0.0000"],
           [64, "0.9707 ± 0.0015", "1.0000 ± 0.0000"],
           [512, "0.8219 ± 0.0063", "1.0000 ± 0.0000"],
           [2048, "0.5819 ± 0.0139", "1.0000 ± 0.0000"]])
doc.add_paragraph(
    "log-log 斜率：叠加 = −0.142（随 G 灭退），覆盖写 = −0.0000（平坦）。"
    "同一批写入，叠加被干扰稀释，覆盖写不稀释——这是 SGC-HM 不随上下文线性衰退的机理证据，"
    "且现在带置信区间、可复现。注意其成立条件是『内容可分离』：信号与干扰路由到各自槽，"
    "这正是面板A指出、由训练键提供的能力。")

h("11.4 显存代价：O(1) 常驻状态（与 L 无关）", level=2)
doc.add_paragraph(
    "SGC-HM 流式常驻状态 = 每层〈缓冲 buffer×D ＋ 长槽 slots×D〉，与序列长 L 无关。"
    "实测（dim=256, 4 层, 槽 32，fp16）全模型常驻状态 ≈ 64.2 KB；"
    "长度从 2K 推到 131K，状态字节恒不变，耗时 O(L) 线性、每 token 均摊 ~236–381 us 持平。"
    "相比注意力状态 O(L²)（L=4096 已约 268 MB），『上下文翻倍只涨约 0–1 GB 显存』在此指"
    "流式/分块推理下的峰值显存——若把整条历史一次性物化，O(L) 输出与梯度仍占内存，"
    "这是诚实标注的代价。")

h("11.5 端到端间隔记忆（训练对照）", level=2)
rows = []
for mx, label in [("hm", "SGC-HM"), ("ca", "SGC-CA(线性)"), ("attn", "Attention")]:
    d = lingap[mx]
    if d is None:
        rows.append([label, "G=512", "(训练进行中)", "-"])
    else:
        rows.append([label, "G=%d" % d["G"], "%.3f" % d["best"], "%.3f" % d["recall_last"]])
add_table(["模型", "间隔 G", "best recall", "末 step recall"], rows)
doc.add_paragraph(
    "任务：固定查询键 Q 存一个值 → G 个干扰 → 查询 Q 预测其值（值域 V=16，chance=1/16=0.0625）。"
    "应观察到叠加型 CA 随间隔 G 召回下降、SGC-HM 平坦。")

h("11.6 诚实限定与澄清（修正此前过度宣称）", level=2)
for t in [
    "“2B Token 上下文, 衰减 <5%”不是无条件事实，而是条件结论：在〈并发互异可寻址话题数 ≤ 槽数 r〉的预算内，保留率由 r 决定、与总长 T 无关；超出预算（互异项 > r）同样会被覆盖挤出。",
    "早前报告的”保留率 0.671、log-log 斜率 0.007”判定为不可复现的路由噪声（n=50、单 seed），现以带 95%CI 的面板探针取代。",
    "路由可靠性依赖训练出的结构化键；未训练/低维随机键下 argmax 路由是抛硬币（面板A）。",
    "显存 O(1) 是流式/分块推理的峰值特性，非把整段历史物化进前向的特性。",
]:
    doc.add_paragraph("• " + t)

out = REPORT
doc.save(out)
print("appended ->", out)
print("lingap loaded:", {k: ("ok" if v else "pending") for k, v in lingap.items()})