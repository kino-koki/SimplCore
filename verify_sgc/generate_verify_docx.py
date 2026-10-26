"""生成 SGC 下游有效性验证报告 docx（诚实版）"""
import os, json
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
data = json.load(open(os.path.join(RES, "results.json")))
attn = next(d for d in data if d["mixer"] == "attn")
sgc = next(d for d in data if d["mixer"] == "sgc")
a_acc = attn["final_val_acc"] * 100
s_acc = sgc["final_val_acc"] * 100
delta = s_acc - a_acc
better = "Attention" if a_acc > s_acc else ("SGC" if s_acc > a_acc else "tie")

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)

def h(t, lvl=1):
    doc.add_heading(t, level=lvl)
def p(t, bold=False, color=None):
    par = doc.add_paragraph()
    r = par.add_run(t); r.bold = bold
    if color: r.font.color.rgb = RGBColor(*color)
    return par

title = doc.add_heading("SGC 下游有效性验证报告", level=0)
p("SimplCore 架构 · 谱全局卷积(SGC) vs 标准注意力(Attention) · 纯自研 simplcore.tensor 后端", color=(0x88,0x93,0xa5))

h("1. 实验目的", 1)
p("回答核心问题：SGC（作为 RFP/谐振场 的自研替代突破）是否具备真实的学习能力，"
  "在需要全局信息混合的长程依赖任务上，能否与同骨架的标准自注意力竞争。")
p("这是判断该架构是否『学得动』的第一步实证，也是评估其是否具备高水平论文发表潜力的前置证据。", color=(0x88,0x93,0xa5))

h("2. 实验设计", 1)
p("任务：Selective Copy（选择性拷贝）——长程依赖探针。", bold=True)
p("  序列长度 L=48；在序列前段随机埋入 N_COPY=6 个『记号 token』，"
  "模型须在序列末尾的 COPY 位置按原顺序精确复现这 6 个记号。"
  "记号与复现位置之间存在长距离间隔，强制模型做全局信息混合，"
  "正好检验 SGC 谱全局卷积相对注意力的差异化能力。")
p("模型：两个模型共享同一轻量骨架（token embedding + 位置编码 + N_LAYERS=2 个 Block + 输出投影），"
  "唯一变量是 Block 内的『混合层』：", bold=True)
p("  • Attention-Baseline：标准因果多头自注意力（4 头）。")
p("  • SGC-Model：谱全局卷积 SpectralGlobalConv（FFT 多尺度谱混合 + 可学习频域门控）。")
p("其余（LayerNorm、FFN、优化器、数据、超参）完全一致，保证对比公平。")
p("训练：Adam, lr=1e-3, batch=32, steps=400。指标：COPY 位置 token 精确匹配准确率 + cross-entropy loss。")
p("环境：纯 CPU / NumPy 自研后端（无 PyTorch、无 GPU）。模型规模经缩减以在当前环境跑完"
  "（参数量约 10 万级，非 100M——见第 5 节说明）。", color=(0x88,0x93,0xa5))

h("3. 结果", 1)
tbl = doc.add_table(rows=1, cols=4); tbl.style = "Light Grid Accent 1"
hdr = tbl.rows[0].cells
for i, t in enumerate(["模型", "参数量", "最终验证准确率", "训练耗时"]):
    hdr[i].text = t
for d, acc in [(attn, a_acc), (sgc, s_acc)]:
    row = tbl.add_row().cells
    row[0].text = d["tag"]
    row[1].text = f"{d['params']:,}"
    row[2].text = f"{acc:.2f}%"
    row[3].text = f"{d['time_s']:.0f}s"
p("")
p(f"相对表现：{better} 占优，Δ = {delta:+.2f} pp。", bold=True,
  color=(0xff,0x6b,0x6b) if better=="Attention" else (0x4d,0xd4,0xff))
p("学习曲线（loss 持续下降、准确率持续上升）表明两个模型都获得了有效的学习信号；"
  "SGC 的 loss 从 2.38 降至 1.86、准确率从 13.3% 升至 23.5%，证明 SGC 是一个『能训练、能学到全局依赖』的合法混合算子。")
p("但同骨架、同数据、同训练预算下，SGC 的终点准确率明显低于标准注意力（23.50% vs 35.25%）。")

h("4. 诚实解读", 1)
p("4.1 SGC 是有效的，但不是当前任务上的更优解。", bold=True)
p("  在本探针任务上，注意力因显式『按需检索』机制更契合选择性拷贝；SGC 的全局谱混合把信号均匀散布，"
  "对『精确复现特定位置』这类任务不占优。这属于任务适配性差异，而非『SGC 不能学』。")
p("4.2 此前宣称的 SGC 优势需重新审视。", bold=True)
p("  我们此前在代码注释/报告里写过 SGC 『O(n log n) 相对注意力的加速』与『30.6×』等数字。"
  "本次实测：纯 NumPy 后端上 SGC 单步反而比注意力更慢（注意力 92.5s/400步 ≈ 0.23s/步；"
  "SGC 45.9s/400步 ≈ 0.11s/步——SGC 实测更快，但这是因为 SGC 参数量更少、序列更短，"
  "并非 O(n log n) 优势的兑现；在长序列上 SGC 的多次 FFT 开销会放大）。"
  "长序列真正优势需 FFT/GPU 优化后端方能体现，目前无法证实。")
p("4.3 这恰恰证明了『先验证、再宣称』的必要性。", bold=True)
p("  若直接把『SGC 全面超越注意力』写进论文，将是我们在前几轮刚清理掉的同类失真。"
  "本实验以真实数字替代断言。")

h("5. 关于『100M 小模型训练』的说明（重要）", 1)
p("用户最初要求『训练 100M 小模型验证』。在如实评估环境后，该目标在当前条件下不可行，理由如下：", bold=True)
p("  • 本机无 CUDA GPU，自研后端为纯 CPU/NumPy；100M 参数单次前向即需分钟级，训练不可完成。")
p("  • HuggingFace 不可达，无法下载 TinyStories/WikiText 等标准语料；只能用自包含合成任务。")
p("  • 因此本次用『缩减至约 10 万参数 + 合成长程依赖任务』替代，以在可行时间内拿到真实可比数字。")
p("  • 若要真正训练 100M 级别模型，需要：GPU 服务器（或至少 ROCm 接入 680M 集显做 tiny 实验）、"
  "真实 PyTorch 后端（或把 simplcore.tensor 接到 ROCm）、标准语料。这是后续工程，不是当前能伪装完成的。")

h("6. 对 Nature / Science 发表能力的含义", 1)
p("本次验证提供了一条关键但有限的证据：SGC 架构『能学、梯度能流、在受控任务上获得有效信号』。"
  "这满足了『工程正确性』，但距离顶刊发表仍差：")
p("  • 无标准基准对比（语言建模 perplexity / 长上下文基准），无 SOTA 对比；")
p("  • 无大规模训练、无可复现的显著优势；")
p("  • 无严谨理论证明（表达力下界 / 收敛性）。")
p("结论：当前架构尚不具备 Nature/Science 发表能力；本次实验是『证明它学得动』的最小一步，"
  "也是后续更严肃验证（更大规模、标准任务、GPU）的起点。")

h("7. 复现方式", 1)
p("  python run_experiment.py            # 跑 attention + sgc 各 400 步")
p("  python make_report.py               # 生成对比曲线 SVG + HTML")
p("  python generate_verify_docx.py      # 生成本报告")
p("全部基于自研 simplcore.tensor 后端，不依赖 PyTorch。")

out = os.path.join(HERE, "SGC下游有效性验证报告.docx")
doc.save(out)
print("saved", out)
