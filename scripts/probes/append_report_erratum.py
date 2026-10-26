# -*- coding: utf-8 -*-
"""Append an erratum section to the SGC-v2 honest report.

The 2026-10 audit found that the report's most load-bearing negative evidence
("independent-test SGC 2.67% / Attention 12.89% / Gated 1.50%", cited to
`SGC_v2_门控消融_诚实报告.md`) has no source: that markdown does not exist and the
three numbers appear in no file in the repository. The report must carry the
correction rather than silently keep the claim.
"""
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

PATH = r"simplcore\docs\SGC_v2_诚实版报告.docx"
d = docx.Document(PATH)

d.add_page_break()
h = d.add_heading("十二、勘误（2026-10 独立性审计）", level=1)
p = d.add_paragraph()
r = p.add_run(
    "本节由代码级独立性审计追加。审计对报告中每一条结论逐一回溯到 "
    "verify_sgc/results/ 下的原始 JSON；下表只列出**未能回溯成功**或**已被证伪**的条目。"
    "本节优先级高于正文：凡与本节冲突者，以本节为准。")
r.italic = True

d.add_heading("12.1 无来源、应予撤回的结论（最严重）", level=2)
d.add_paragraph(
    "正文 §3.4、§4.1、§5、§6 反复引用的「独立测试集 test_acc：SGC 2.67% / "
    "Gated-SGC 1.50% / Attention 12.89%」，其引文指向 "
    "simplcore/docs/SGC_v2_门控消融_诚实报告.md。经全仓库检索：")
for t in [
    "该 markdown 文件在本仓库中**不存在**；",
    "字符串“2.67%”“12.89%”“1.50%”在**任何**文件中都不出现；",
    "因此该组数字**没有任何可复现来源**。",
]:
    d.add_paragraph("\u2022 " + t)
p = d.add_paragraph()
r = p.add_run("处置：撤回该组数字。")
r.bold = True
d.add_paragraph(
    "影响：这是本报告中最具分量的**负面**证据，正文 §3.4 门控消融、§4.1「关键矛盾一」、"
    "§5 结论第 2/3 条、§6 数据来源清单均建立其上。撤回后，"
    "「SGC 在更严格独立测试协议下崩坏至随机水平」这一说法**不再有支撑**，"
    "§4.1 所述“两种结论不可调和”的矛盾也随之消解——"
    "剩下的只是同分布 fresh-batch 上 SGC≈注意力≈36%（v2_1200.json，可信）。"
    "在重新生成独立测试协议数据之前，§3.4/§4.1/§5/§6 的相应表述应视为无效。")

d.add_heading("12.2 与原始 JSON 不符的引用", level=2)
d.add_paragraph(
    "§3.4 门控消融表第一行引用 gated_compare.json。核对该文件自身记录的参数为 "
    "dim=32, layers=1, heads=2, seq=64, rank=4, batch=4, **steps=5**, wd=0.0："
    "仅 5 步训练，train_loss 在 ln(vocab) 附近游走。报告把它与真实的 800 步实验并列，"
    "未披露步数。此外该文件是以 --mixers sgc,gated_sgc 生成的，"
    "**attn 与 sgc_nogate 两条对照臂完全缺失**，即该消融从未完整做过。")

d.add_heading("12.3 §11.2 / §11.3 / §11.4 的证据状态", level=2)
d.add_paragraph(
    "§11.2、§11.3 的机理探针数值并非来自任何 JSON，而只作为字符串字面量存在于 "
    "生成报告的脚本 _append_hm_docx.py 中。审计重跑了 _probe_retention.py，"
    "**数值可精确复现**（含斜率 −0.142 / −0.0000），即结论为真但未落盘。"
    "处置：结论保留，但“可复现”应改为“已由审计独立重跑复现”，并应补交 JSON。")
d.add_paragraph(
    "§11.4 的显存数字（“常驻状态 ≈ 64.2 KB”“每 token 236–381 µs”）同样是 "
    "_append_hm_docx.py 中的字面量，取自 _stream_memproof.py 的**解析打印**，不是测量。"
    "更关键的是：**该报告所依赖的显存测量工具在本机完全失效**——"
    "GetProcessMemoryInfo 返回 0 且 GetLastError()=122，"
    "stream_infer.py 与 _memcap.py 打印的每一行“峰值RSS”都是 0.0 MB，"
    "memcap.json 的 attn_delta 列全为 0。审计独立复现了该失效（分配 8 MB 后重读仍为 0）。"
    "处置：§11.4 的“O(1) 显存实测”应撤回；仅“每 token 耗时线性、约 101 µs”一项可用。")

d.add_heading("12.4 §11.5「训练进行中」——现已实跑完成，脚本有缺陷", level=2)
d.add_paragraph(
    "审计把 §11.5 的三行跑到了终点（G=512，700 步）："
    "hm 905.0s / ca 535.2s / attn 862.3s，三者 loss 分别为 2.787 / 2.782 / 2.780，"
    "**全部高于 ln(16)=2.773**，recall 0.000 / 0.031 / 0.031，chance=0.0625。")
d.add_paragraph(
    "根因不是 mixer 而是脚本：_train_lingap.py 对全部 L 个位置计交叉熵，"
    "而序列结构 [Q, v, 干扰×G, Q] 中除最后一位外，其余位置的“下一个 token”是"
    "**与历史无关的随机干扰**，在 16 类上不可学。514/515 个不可学位置把损失顶在 "
    "ln(16) 之上，连能精确检索任意历史 token 的注意力也同样失败。"
    "==> 该实验此前“训练不收敛”的现象是测量脚本的产物，不能用来评价任何 mixer。")
p = d.add_paragraph()
r = p.add_run("已修复：")
r.bold = True
d.add_paragraph(
    "只对查询位置计损失（F.cross_entropy(logits[:, -1, :], tv)）。"
    "修正后的结果为 results/lingap_{hm,ca,attn}_G512.json。")

d.add_heading("12.5 其他已证实缺陷", level=2)
tbl = d.add_table(rows=1, cols=3)
try:
    tbl.style = "Table Grid"
except Exception:
    pass
hdr = tbl.rows[0].cells
for i, t in enumerate(["缺陷", "证据", "状态"]):
    hdr[i].text = t
rows = [
    ("GatedAssocMixer 非因果 + 非确定",
     "sgc_v3.py:190 每次前向用 torch.randn 重置状态；同一输入两次前向差 9.7e-6，"
     "扰动未来会改变过去——两者同源（随机初值不一致伪装成未来泄漏）",
     "已修复：改用固定 generator 播种；实测确定性 0.0、因果性 0.0"),
    ("SGC_HM 峰值不是 O(1)",
     "sgc_v3.py:292-301 块内做 buffer×chunk 的注意力，峰值 O(buffer·chunk)",
     "待修：应改述为 O(buffer·chunk)，或改为逐 token 精确缓冲"),
    ("ContentAdaptiveMixer 输出依赖内部 chunk",
     "Q·S 未归一化；实测 chunk 8 vs 16 的 max|Δ| = 1.3。"
     "各脚本 ca_chunk 取 128/256/512/64 不等",
     "待修：ca 的历史结果跨脚本不可比"),
    ("sgc / sgcv3 有静默上下文上限",
     "self.low[:L, :] 截断；L > max_len 时退化为固定窗，不报错",
     "已在 SGC 消除（simplcore/sgc）"),
    ("冒烟测试用未移位目标",
     "sgc_pytorch.py:196、sgc_v3.py:391 把输入本身当 target，"
     "tie-embedding 模型只需回声即得 loss=0.0，测试恒过；"
     "同仓库 mixers.py:277-279 反而正确移位",
     "待修：应移位并断言 loss > 0"),
    ("语料实为仓库自身日志",
     "build_corpus.py 采集 24 文件 115,255 B / 74,610 字符，"
     "其中真实中文语料仅 12,694 字符(17%)，其余 83% 是构建日志与 markdown；"
     "test 集 65% 来自单个文件 PAPER_SPEC.md",
     "已改用 FineWeb-Edu-Chinese 真实语料"),
    ("无结果溯源",
     "complexity_scaling.json 不含任何 args；_verify_ca/_probe_retention/"
     "_stream_memproof/_ar_quick 均不写 JSON",
     "待修：统一 runner 落盘 {args, hash, host, wall_s, metric}"),
]
for a, b, c in rows:
    cells = tbl.add_row().cells
    cells[0].text = a
    cells[1].text = b
    cells[2].text = c

d.add_heading("12.6 经审计确认可信、予以保留的结论", level=2)
for t in [
    "§3.1 长序列效率（efficiency_timing.json，逐格精确匹配）；",
    "§3.2 低熵选择性任务准确率持平 35.90% vs 36.27%（v2_1200.json）；",
    "§3.2 loss 两轮结论相反（v2_1200 与 results_v2_causal 逐格匹配）；",
    "§3.3 高熵真实语料泛化更差（pytorch_compare_wd0.json 六格全部匹配）"
    "——注意该语料的实际质量见 12.5，其“真实语料”应降级为“微型混合语料”；",
    "§9.1 四路对比全部 20 格匹配（honest_*.json）；",
    "§9.2 复杂度缩放全部 15 格匹配（complexity_scaling.json）"
    "——但“SGC-CA 在 seq=2048 反超注意力”仅为前向、单 mixer，"
    "端到端实测为 ca 991 ms/step vs attn 499 ms/step，**不成立**，应撤回；",
    "§10.1 最大模型训练全部 12 格匹配（honest_max_ca512.json）；",
    "文档级切分确实无泄漏（0/77 val、0/101 test 的 128 字符切片出现在 train 中）"
    "——但窗口重叠 4×，582 个 train 窗口仅含约 149k 去重 token。",
]:
    d.add_paragraph("\u2022 " + t)

d.add_heading("12.7 总体诚信结论（修订）", level=2)
d.add_paragraph(
    "撤回 12.1 的无源数字、并剔除 §9.2 的端到端效率主张之后，"
    "本报告**不再包含任何“SGC 优于或持平注意力”的正面泛化主张**。"
    "剩下的可信内容为：(a) O(n log n) 长序列效率（对照标准因果注意力，实测）；"
    "(b) 参数量更少；(c) 固定位置核在高熵语料上泛化更差（微型语料实测）；"
    "(d) 内容自适应混合（CA）在微型语料上修复了该崩塌，但其结果依赖未申报的 chunk 参数；"
    "(e) SGC-HM 的覆盖写 vs 叠加的机理探针（已由审计独立复现）。"
    "据此，本报告应定位为**内部技术备忘**，不构成任何可发表的实证主张。")

out = r"simplcore\docs\SGC_v2_诚实版报告.docx"
d.save(out)
print("erratum appended ->", out)
print("paragraphs:", len(d.paragraphs), " tables:", len(d.tables))


