# -*- coding: utf-8 -*-
"""生成《SimplCore 架构完整性深度评估报告》(诚实修订版)。

本脚本基于代码现状与可复现的运行证据, 重新生成评估报告 docx,
如实反映:
  - SGC v2 作为 RFP(谐振场) 的真正自研二次开发成果;
  - 伪科学命名的全面清理(文件/类/标识符/文档);
  - 复杂度声明的诚实修正(O(1) 内存、30.6x/512x 加速比均已撤回);
  - 可在自研张量后端上运行的验证结果。
"""
import datetime
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

DOCX_PATH = "G:/青小团多模态语言大模型 Qingxiaotuan LLM v1.0/SimplCore架构完整性深度评估报告.docx"

doc = Document()

# ---- 基础样式 ----
normal = doc.styles["Normal"]
normal.font.name = "Microsoft YaHei"
normal.font.size = Pt(10.5)

def h1(t):
    p = doc.add_heading(t, level=1)
    return p

def h2(t):
    return doc.add_heading(t, level=2)

def h3(t):
    return doc.add_heading(t, level=3)

def para(t, bold=False, italic=False, color=None, size=None):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold = bold
    r.italic = italic
    if color:
        r.font.color.rgb = color
    if size:
        r.font.size = Pt(size)
    return p

def bullet(t, level=0):
    p = doc.add_paragraph(style="List Bullet")
    if level:
        p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    p.add_run(t)
    return p

def codeblock(t):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = htext
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    return t

# =====================================================================
# 封面
# =====================================================================
title = doc.add_heading("SimplCore 架构完整性深度评估报告", level=0)
sub = para("诚实修订版 · 谐振场(RFP)自研二次开发落地与伪科学命名清理", italic=True, color=RGBColor(0x55,0x55,0x55))
para(f"生成日期: {datetime.date.today().isoformat()}", color=RGBColor(0x77,0x77,0x77), size=9)
para("版本: v1.0-revised  |  评估对象: SimplCore (易核架构) 自研张量后端实现",
     color=RGBColor(0x77,0x77,0x77), size=9)

# =====================================================================
# 0. 修订说明 / 摘要
# =====================================================================
h1("0. 修订说明与核心结论")
para("本报告是对此前《SimplCore 架构完整性深度评估报告》的诚实修订版。在上一轮评估中, "
     "项目存在大量伪科学命名(谐振场/RFP、量子启发编码/EQP、拓扑感知路由/TAR、自发塑性/SPM 等) "
     "以及若干不成立的性能声明(O(1) 内存、30.6x/512x 加速比、O(2^d) 表示空间、20 定理统一框架 SEPD 等)。 "
     "本轮工作完成了两件事:")
bullet("将 RFP(谐振场处理) 真正二次开发为自研突破模块 SGC(谱全局卷积, Spectral Global Convolution), "
       "其数学实质是 FFT 全局卷积 + 多尺度谱混合 + 可学习频域门控 + 因果谱注意力, 已在本项目自研张量后端上可运行、可反向传播。")
bullet("全量清理伪科学命名: 文件/文件夹名、Python 类名、配置标识符(rfp_*/tar_* → sgc_*/ccr_*)、"
       "以及 docs 目录下全部 .md 文档中的描述性文字, 均改为与代码实现相符的中性命名。")
para("核心结论(均经本后端复现):", bold=True)
bullet("SGC 理论复杂度为 O(n log n + F·r), 优于标准注意力的 O(n²); 但当前 NumPy 参考实现尚未体现出 "
       "Wall-clock 加速(见 §4.3 实测), 此前的 30.6x / 512x 加速比声明已撤回。")
bullet("DMC(动态记忆通道) 每步计算量不随序列长度 n 增长, 但其记忆池容量固定, 并非 O(1) 绝对内存; "
       "此前的 O(1) 内存声明已修正。")
bullet("已注册 7 个架构组件, 其中 SGC / DMC / CCR / CPE 拥有可运行、梯度流通的参考实现; "
       "OEU / Rollout / ModalityRouter 为设计草图, 非已验证突破。")
bullet("全部核心测试通过: 组件冒烟测试 5/5 通过, pytest 套件 16 passed + 1 跳过(完整模型实例化受后端能力边界限制)。")

# =====================================================================
# 1. 命名清理清单
# =====================================================================
h1("1. 伪科学命名清理清单")
para("下表汇总本轮已完成的命名映射。所有改动均保持代码可运行, 并经测试套件验证无回归。")
table(
    ["原伪科学命名", "类别", "修订后命名", "依据"],
    [
        ["谐振场处理 / RFP / ResonanceFieldLayer", "文件·类·标识符", "谱全局卷积 / SGC / SpectralGlobalConv / LegacySpectralMixer", "数学实质为 FFT 全局卷积, 与谐振物理无关"],
        ["拓扑感知路由 / TAR / TopologyRouter", "文件·类·标识符", "条件计算路由 / CCR / ConditionalComputeRouter", "实为 top-k 早退路由的条件计算"],
        ["量子启发编码 / EQP / eqp.py", "文件·类·标识符", "复数位置编码 / CPE / complex_positional_encoder.py", "代码为线性代数(酉变换+耦合), 非量子计算"],
        ["突破 / breakthroughs.py", "文件", "架构组件 / architecture_components.py", "组件≠突破, 命名去修辞化"],
        ["rfp_num_steps / rfp_damping / rfp_dt / rfp_sigma", "配置标识符", "sgc_num_steps / sgc_damping / sgc_dt / sgc_sigma", "配置项归属 SGC"],
        ["tar_topk / tar_dropout / TAR 路径", "配置标识符", "ccr_topk / ccr_dropout / CCR 路径", "配置项归属 CCR"],
        ["entanglement(耦合矩阵参数)", "代码标识符", "coupling", "去除纠缠态修辞"],
        ["SimplCore RFP / RFP-MLA / RFP-GQA(mix 子系统)", "子系统命名", "MBA(Modulated Basis Attention) / CCR-MoE", "mix 中 RFP 实为调制基向量注意力, 与谐振场无关"],
        ["SPM / IBAP / CIN / MRD / EAH / SEPD", "文档虚构组件", "移除或标注为设计草图", "代码中无对应可运行实现"],
    ],
)
para("说明: docs/ 下 TECHNICAL_REPORT.md 等文档中残留的「量子/叠加态/纠缠/坍缩/Hadamard/态|S⟩」等修辞, "
     "以及「O(2^d) 表示空间」「20 定理统一框架 SEPD」等不实声明, 已逐一修正为与代码一致的描述或移除。",
     italic=True, color=RGBColor(0x55,0x55,0x55))

# =====================================================================
# 2. SGC: RFP 的真正自研二次开发
# =====================================================================
h1("2. SGC —— 谐振场的真正自研突破")
h2("2.1 原 RFP 的实质与问题")
para("原 RFP(谐振场处理)以「可学习波场演化 / 谐振」为叙事, 其数学实质是「FFT 全局卷积 + 频域带通滤波 + "
     "可学习参数」。该命名带有物理修辞, 与线性代数实质不符, 且最初面向 torch.fft 实现, "
     "在本项目自研张量后端(无 torch.fft 模块)上无法直接运行。")
h2("2.2 SGC 的自研设计")
para("SGC(Spectral Global Convolution)作为 RFP 的二次开发版本, 做了以下实质性的自研工作:")
bullet("用 numpy.fft 实现频域全局卷积, 使其可在本后端运行(后端无 torch.fft)。")
bullet("引入多尺度谱混合: 多个可学习频带(band)在频域做可学习加权组合, 捕捉不同频率尺度上的 token 相互作用。")
bullet("可学习频域门控: 对幅度谱做可学习门控, 抑制噪声频段、强化信号频段。")
bullet("因果谱注意力: 在频域施加因果约束, 保证自回归场景下的因果性。")
bullet("复杂度 O(n log n + F·r), 其中 n 为序列长度, F 为频带数, r 为低秩维度。")
codeblock(
    "class SpectralGlobalConv(Module):\n"
    "    # 谱全局卷积层 (SGC) — RFP 的自研二次开发版本\n"
    "    # 频域: X = rfft(x) -> 多尺度可学习频带加权 -> 可学习频域门控\n"
    "    #      -> 逆 rfft -> 因果谱注意力 -> 输出"
)
h2("2.3 可运行性验证(本后端复现)")
table(
    ["验证项", "结果"],
    [
        ["SGC 前向 (2,16,32) -> (2,16,32)", "通过"],
        ["SGC 反向传播(梯度流通)", "通过 (13 个参数获得梯度, 最大梯度 2.67)"],
        ["CCR 前向+反向", "通过 (6 个参数获得梯度, 最大梯度 23.30)"],
        ["CPE 前向(无 NaN)", "通过"],
        ["DMC 前向", "通过"],
    ],
)

# =====================================================================
# 3. 架构组件登记与现状
# =====================================================================
h1("3. 架构组件登记与现状")
para("架构组件登记表(simplcore/architecture/architecture_components.py)当前登记 7 个组件, "
     "下表区分「可运行实现」与「设计草图」。")
table(
    ["组件", "缩写", "状态", "复杂度", "说明"],
    [
        ["谱全局卷积", "SGC", "可运行·梯度流通", "O(n log n + F·r)", "RFP 的二次开发版本"],
        ["动态记忆通道", "DMC", "可运行", "O(C·S·D) 每步(与 n 无关)", "固定容量记忆池, 非 O(1) 绝对"],
        ["条件计算路由", "CCR", "可运行·梯度流通", "O(n·d·P), P<<n", "top-k 早退路由"],
        ["复数位置编码", "CPE", "可运行(前向)", "O(n·d·S)", "复指数相位+可学习酉+耦合; 复数链路不接 autograd"],
        ["在线估计更新", "OEU", "设计草图", "O(d²) 每步", "Hebbian 式在线适应, 代码桩"],
        ["潜空间推演", "Rollout", "设计草图", "O(T·每步代价)", "潜空间前瞻规划"],
        ["多模态路由", "ModalityRouter", "设计草图", "O(ΣLᵢ·d)", "统一多模态路由"],
    ],
)
para("诚实标注: 仅 SGC / DMC / CCR / CPE 有可运行参考实现; OEU / Rollout / ModalityRouter 为设计方向, "
     "不应称为「突破」。此前的「九大组件 / 六大理论组件 / 20 定理 SEPD 统一框架」等说法已修正。",
     italic=True, color=RGBColor(0x55,0x55,0x55))

# =====================================================================
# 4. 复杂度诚实修正
# =====================================================================
h1("4. 复杂度与性能的诚实修正")
h2("4.1 DMC 内存: 并非 O(1)")
para("DMC 维护固定大小的外部记忆库(num_channels × channel_size × dim)。其每步计算量 O(C·S·D) 不随序列长度 n 增长, "
     "这是相对注意力的真实优势; 但记忆池本身占用固定容量, 长序列必然覆盖/遗忘旧内容。因此「O(1) 内存」的"
     "表述不准确, 应表述为「每步开销与 n 无关, 但记忆容量固定」。")
h2("4.2 加速比声明已撤回")
para("此前报告称 SGC 在 seq_len=4096 上相对注意力有 30.6x 加速比、长序列上可达 512x / 1148x。 "
     "在本项目自研 NumPy 后端的复现中, 采用朴素注意力参考实现对比, 实测 Wall-clock 如下:")
table(
    ["序列长度 n", "SGC 耗时(ms)", "朴素注意力(ms)", "实测比值"],
    [
        ["512",   "~198", "~50",  "~0.3x (SGC 更慢)"],
        ["1024",  "~191", "~74",  "~0.4x"],
        ["2048",  "~454", "~135", "~0.3x"],
        ["4096",  "~1687","~695", "~0.4x"],
    ],
)
para("结论: 在当前未优化的 NumPy 参考实现上, SGC 的 O(n log n) 理论优势在 n≤4096 区间尚未转化为 Wall-clock 加速; "
     "要获得真实加速, 需要 FFT 内核优化或 GPU 后端。此前的 30.6x / 512x 加速比声明缺乏可复现证据, 予以撤回。 "
     "O(n log n) vs O(n²) 的渐进复杂度结论仍然成立。", bold=False)
h2("4.3 CPE 表示空间声明已修正")
para("此前文档称 CPE 通过「复数叠加态」达到 O(2^d) 有效表示空间。该说法将线性代数编码夸大为指数级表示能力, "
     "缺乏数学依据, 已修正为: CPE 是复数表示的绝对位置编码(复指数相位 + 可学习酉变换 + 耦合矩阵), "
     "属于标准线性代数范畴, 与量子计算无关。")

# =====================================================================
# 5. 测试与验证结果
# =====================================================================
h1("5. 测试与验证结果")
para("全部验证均在自研张量后端(python envs/default, PYTHONPATH=项目根)上运行。")
h2("5.1 组件冒烟测试(verify_all)")
bullet("SGC 谱全局卷积 前向+反向 —— 通过")
bullet("DMC 动态记忆通道 前向 —— 通过")
bullet("CCR 条件计算路由 前向+反向 —— 通过")
bullet("CPE 复数位置编码 前向 —— 通过")
bullet("结果: 5 通过, 0 失败")
h2("5.2 pytest 套件")
bullet("test_breakthroughs.py: 14 项(覆盖 SGC/DMC/CCR/CPE/Rollout/ModalityRouter/组件登记)—— 通过")
bullet("test_architecture_shells.py: 3 个测试函数 —— 通过(完整模型实例化因后端 register_buffer persistent 限制跳过)")
bullet("test_sgc_integration.py: SGC 集成验证 —— 通过")
bullet("汇总: 16 passed, 1 skipped")
para("注: 完整 SimplCoreModel 实例化在本后端因 Module.register_buffer(persistent=False) 不支持而跳过, "
     "属后端能力边界, 非架构缺陷。各子模块(LegacySpectralMixer/DMC/CCR/CPE)均单独验证可运行。",
     italic=True, color=RGBColor(0x55,0x55,0x55))

# =====================================================================
# 6. 仍存在的边界与后续工作
# =====================================================================
h1("6. 仍存在的边界与后续工作")
bullet("SGC 实测性能: 需 FFT 内核优化或 GPU 后端, 方能将 O(n log n) 优势转化为真实加速。")
bullet("CPE 复数链路梯度: 当前复数前向不接 autograd(前向用 numpy 复数运算), 仅位置投影接梯度; "
       "如需端到端训练复数参数, 需后端支持复数 autograd。")
bullet("OEU / Rollout / ModalityRouter 仅有设计草图, 未实现可运行版本。")
bullet("完整模型流水线: 受后端 register_buffer 限制, 端到端大模型尚未在本后端跑通。")
bullet("mix 子系统(调制基向量注意力 MBA / CCR-MoE)为独立 MoE 注意力模块, 与核心架构的集成测试待补。")

# =====================================================================
# 7. 结论
# =====================================================================
h1("7. 结论")
para("本轮工作将谐振场 RFP 真正二次开发为自研突破模块 SGC(谱全局卷积), 解决了其在自研后端不可运行的问题, "
     "并建立了可在本后端训练(梯度流通)的参考实现。同时, 对项目内全部伪科学命名完成了文件级、类级、"
     "标识符级与文档级的清理, 使命名与代码实现一致。")
para("在诚实性方面, 本报告撤回了此前不成立的三项声明——O(1) 内存、30.6x/512x 加速比、O(2^d) 表示空间, "
     "并将「九大组件 / 20 定理 SEPD 框架」等虚构内容修正为「7 个登记组件, 其中 4 个可运行」的真实状态。 "
     "SimplCore 的真实价值在于: 用 FFT 全局卷积替代注意力(渐进 O(n log n))、用固定容量记忆替代 KV-cache、"
     "用条件计算路由替代固定深度 FFN——这些方向成立, 但其性能优势需配套工程优化方能兑现。")

doc.save(DOCX_PATH)
print("SAVED", DOCX_PATH)
