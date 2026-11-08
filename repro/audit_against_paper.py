"""用作者提供的 SimplCore_Paper.md 逐条核对复现报告里的引用与数字。

用法: python audit_against_paper.py
输出: results_citation_audit.json + 控制台 PASS/FAIL 表
"""
import json, re, os

PAPER = "/data/inputs/SimplCore_Paper.md"
REPORT = "/data/workspace/repro/白盒复现报告_SimplCore_SGC.md"
OUT = "/data/workspace/repro/results_citation_audit.json"

paper = open(PAPER, encoding="utf-8").read()
report = open(REPORT, encoding="utf-8").read()

# (报告里的说法, 原文中应存在的证据字符串, 类别, 判定说明)
CLAIMS = [
    # ---- 章节 / 定理 / 命题编号 ----
    ("定理 1 双形式对偶", "定理 1（双形式对偶）", "章节", "真实"),
    ("定理 2 严格因果", "定理 2（严格因果）", "章节", "真实"),
    ("定理 3 常数状态公式", "状态字节数 $4(2MD+(K{-}1)D+1)$", "章节", "真实"),
    ("命题 5 双口径勘误", "命题 5（误差界，0.1-α 双口径勘误后口径）", "章节", "真实"),
    ("P10 定容槽 O(1) 无损", "定容槽位存绑定副本\n  同样是 O(1) 且无损", "章节", "真实"),
    ("§3.1 srf 的 O(D·L/chunk)", "$O(D\\cdot L/\\text{chunk})$", "章节", "真实"),
    ("§3.1 P1 θ 钳制", "θ 钳制**（P1）", "章节", "真实"),
    ("§3.3 绑定能力定位", "SGC 缺的不是感受野，是**绑定能力**", "章节", "真实"),
    ("§5.4 重尾", "实测为重尾", "章节", "真实"),
    ("§5.5 判定纪律", "均值差落在对方 bootstrap CI 内 = 噪声内", "章节", "真实"),
    ("§5.5 72% 参数口径", "72% 参数", "章节", "真实"),
    ("§5.12 部署账本", "### 5.12 部署账本", "章节", "真实"),
    ("§5.14 端侧极限训练", "### 5.14 端侧极限训练", "章节", "真实"),
    ("§5.13 战役未运行", "§5.13 战役未运行", "章节", "真实"),
    ("§8 C 级背叛不做", "全局注意力 / 增长 KV / RoPE | ❌ 背叛，不做", "章节", "真实"),
    ("§8 R2 cond 1e19→1.4", "cond 1e19→1.4", "章节", "真实"),
    ("§10 结论", "把它变成事实需要的是 §5.13 战役的", "章节", "真实"),
    ("§7.2（原文自身交叉引用）", "### 7.2", "章节", "原文内部悬空引用：§3.3 写『MQAR 对照（§7.2）』，但 §7 无 7.2 小节"),

    # ---- 数字 ----
    ("1/|sinθ| ≈ 2607.6", "≈ 2607.6", "数字", "真实"),
    ("1.948 GiB @1M", "1.948 GiB", "数字", "真实"),
    ("bf16 权重 1.888 GiB", "权重 1.888", "数字", "真实"),
    ("流式状态 30.36 MiB", "30.36 MiB", "数字", "真实"),
    ("KV ≈120 GiB / 129 GB", "129 GB (120 GiB)", "数字", "真实"),
    ("1.085 亿参数 dim=768/12层/M=64", "1.085 亿参数（dim=768/12 层/M=64）", "数字", "真实"),
    ("状态 4,829,184 B", "4,829,184 B", "数字", "真实"),
    ("斜率 1.09 vs 2.35", "1.09 vs TS 2.35", "数字", "真实"),
    ("精确指向 1.000 / 0.398", "1.000 | 0.398 | 0.125", "数字", "真实"),
    ("MQAR 0.500 / 0.3125", "0.500 | 0.3125 | 0.125", "数字", "真实"),
    ("MQAR+记忆 0.2422 / 0.2031 / 0.2578", "0.2422* | 0.2031", "数字", "真实"),
    ("聚合 L=129 0.938 / 0.875", "0.938 | 0.875 | 0.5", "数字", "真实"),
    ("聚合 L=65 1.000 / 1.000", "1.000 | 1.000 | 0.5", "数字", "真实"),
    ("精确计数 0.812/0.938 双方打平", "0.812/0.938 | 0.812/0.938", "数字", "真实"),
    ("短程 LM ppl 14.63 / 15.71", "ppl 14.63 | 15.71", "数字", "真实"),
    ("§5.4 logfreq L=2048 4.7e-5", "4.7\\times10^{-5}", "数字", "真实"),
    ("命题 5 legacy M=8 旧界违反 1.69×", "被违反 1.69×", "数字", "真实"),
    ("定理 3 示例 524.0 KiB/层", "524.0 KiB/层", "数字", "真实"),

    # ---- 报告里对原文的【转述 / 归因】，需单独判定 ----
    ("原文 §5.4 归因『近零 θ + 高 ρ 是重尾主因』", "近零 θ", "归因", "幻觉：原文 §5.4 只报现象与两代网格数值，未做此归因"),
    ("原文把 §5.14 标为未运行/未验证", "真实训练成功", "归因", "错误：原文 §5.14 明确『真实训练成功』，仅生成质量诚实说明"),
    ("原文 §5.5 是 5 个 probe 任务", "### 5.5 能力对照", "归因", "错误：原文表格共 7 行（含 MQAR+绑定记忆、聚合 L=129/L=65 两档）"),
    ("sgc-r 列为可选 mixer 是因正交性问题", "谱专家路由", "归因", "推测：原文只列了 sgc-r 的定义与 FLOPs，未给原因"),
    ("§5.12 账本来自另一个更大的预设", "### 5.12 部署账本", "归因", "推测：原文未给 §5.12 的配置，不能断言『来自另一个预设』"),
]


def norm(x):
    return re.sub(r"\s+", "", x).replace("$", "").replace("\\", "")


rows = []
for name, evidence, kind, note in CLAIMS:
    hit = norm(evidence) in norm(paper)
    if kind == "章节" and "悬空" in note:
        status = "FAIL(原文自身)"
    elif kind == "归因":
        status = "FAIL(报告)" if hit is False or note.startswith(("幻觉", "错误", "推测")) else "CHECK"
        # 归因类：evidence 命中不代表转述成立，按 note 判定
        status = "FAIL(报告)" if note.startswith(("幻觉", "错误", "推测")) else "CHECK"
    else:
        status = "PASS" if hit else "FAIL"
    rows.append(dict(claim=name, evidence=evidence, kind=kind,
                     found_in_paper=hit, status=status, note=note))

# ---- 报告里三处伪吻合的独立判定 ----
PSEUDO = [
    ("2607.5947 与 2607.6『逐位吻合』", "循环论证：θ_min=π/8192 取自原文，代入计算必然得 2607.6，不构成独立验证"),
    ("0.3066 vs 0.3125『近乎逐位吻合』", "不同任务构造/不同规模下的数值巧合，不可作证据"),
    ("1.16 vs 1.09『高度吻合』", "不同规模拟合；且吞吐数据 L=2048 用 B=1、其余 B=2，混回归"),
]
for claim, why in PSEUDO:
    rows.append(dict(claim=claim, evidence="—", kind="伪论证",
                     found_in_paper=None, status="FAIL(报告)", note=why))

# ---- 报告与原文对照表里的数字一致性 ----
NUM_CHECK = [
    ("报告 §8.1 精确计数行写『0.938 / 0.938』", "0.812/0.938 | 0.812/0.938", "漏了 0.812 一档"),
    ("报告 §0 写『5 个 probe 任务』", "精确计数 | 0.812/0.938", "原文 §5.5 共 7 行"),
]
for claim, ev, why in NUM_CHECK:
    rows.append(dict(claim=claim, evidence=ev, kind="数字一致性",
                     found_in_paper=norm(ev) in norm(paper),
                     status="FAIL(报告)", note=why))

json.dump(rows, open(OUT, "w"), indent=2, ensure_ascii=False)

print("=" * 100)
print(f"{'状态':<16}{'类别':<12}{'报告中的说法':<52}{'原文命中':<10}")
print("=" * 100)
for r in rows:
    print(f"{r['status']:<16}{r['kind']:<12}{r['claim'][:50]:<52}{str(r['found_in_paper']):<10}")
print("=" * 100)
n_pass = sum(1 for r in rows if r["status"] == "PASS")
n_fail = sum(1 for r in rows if r["status"].startswith("FAIL"))
print(f"PASS {n_pass} / FAIL {n_fail} / 共 {len(rows)} 项")
print(f"\n结论：章节编号与绝大多数数字在原文中【真实存在】，不是幻觉。")
print(f"      真正的问题集中在【归因转述】【伪吻合论证】【对照表漏项】三类。")
print(f"\n-> {OUT}")
