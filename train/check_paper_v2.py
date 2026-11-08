"""0.2-β 主稿（含理论形式化 + 相关工作核实）一致性自检，扩展版。"""
import json
import re

s = open("PAPER_NEW.md", encoding="utf-8").read()
lines = s.splitlines()
checks = []

def ck(name, cond, detail=""):
    checks.append(dict(name=name, passed=bool(cond), detail=detail))
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))

# 已核实的 arXiv 白名单（2026-11-05 逐条检索核对）
WHITELIST = {
    "2111.00396", "2208.04933", "2302.10866", "2312.00752", "2405.21060",
    "2312.06635", "2406.06484", "2006.16236", "2305.13048", "2407.04620",
    "2307.02486", "2403.19887", "2312.04927", "2402.01032", "2404.08819",
}
ids = set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", s)) | set(
    re.findall(r"\b(\d{4}\.\d{4,5})\b", s))
stray = ids - WHITELIST

# 1 overclaim
abstract = s[s.find("## 摘要"):s.find("## Abstract")]
conc = s[s.find("## 9 结论"):s.find("## 附录 A")]
for b in ["超越 Transformer", "替代注意力", "唯一成本可行", "等效 4–7B"]:
    ck(f"摘要不含「{b}」", b not in abstract)
ck("标题不含「替代注意力」", "替代注意力" not in lines[0])
ck("摘要无 [C] 标记", "[C]" not in abstract)
ck("结论无 [C] 标记", "[C]" not in conc)

# 2 引文纪律
ck("无未核实 arXiv ID（白名单外为 0）", len(stray) == 0,
   f"白名单外: {sorted(stray)}" if stray else f"{len(ids & WHITELIST)} 个已核实 ID")
ck("无 doi.org / 10.xxxx 形式", not re.search(r"doi\.org|10\.\d{4}/", s))
ck("明确不主张首次发现且禁用语仅出现在列举段",
   "不主张首次发现" in s and "不可辩护的表述" in s
   and s.count("首次证明静态卷积做不了内容寻址") == 1)

# 3 理论升级
ck("B1 不再标为观察性命题", "命题 B1（观察性命题，实证支撑）" not in s)
ck("B1 标为严格证明", "命题 B1（严格证明" in s)
ck("B1 含可加分离与矛盾式", "可加分离" in s and "→ 矛盾" in s)
ck("B1 适用范围已写明", "适用范围" in s and "未在此证明" in s)
ck("B1.1 明确标为非定理", "推论 B1.1（解释性框架，非定理）" in s)
ck("B4 已标为引理 + 严格证明", "引理 B4（严格证明" in s and "$\\square$" in s)
ck("B4 注明已知来源 S5/Mamba", "S5 (arXiv:2208.04933)" in s and "Mamba (arXiv:2312.00752)" in s)
ck("B2/B3 明确不是定理", "B2 / B3 仍是实证观察 + 有限样本估计量，不是定理" in s)
ck("0.2484 略高于 0.2476 的估计量说明保留", "略**高于**" in s and "估计量" in s)
ck("理论文档路径已挂", "docs/EXPRESSIVITY_BOUNDARIES.md" in s)
ck("B1 数值注脚路径已挂", "tools/boundary_probe/results/b1_corollary.json" in s)

# 4 相关工作
ck("§8 已含统一分类表的已核实条目", "统一分类表" in s and "逐条核实" in s)
ck("§8 明确「不是实验结果表」", "不是实验结果表" in s)
ck("§8 明确定量对标缺口", "等参数 / 等算力定量对标" in s and "完全未做" in s)
ck("相关工作文档路径已挂", "docs/RELATED_WORK.md" in s)
ck("7.3(12) 标为文献已核实", "文献条目已核，定量对标未做" in s)

# 5 附录 C 状态
ck("附录 C 形式化标为部分完成", "B1 / B4 已严格证明" in s and "🟡 **部分完成**" in s)
ck("附录 C 新增定量对标 P0 行", "| P0 | **等参数 / 等算力定量对标实验**" in s)
ck("附录 C 仍标不宜投稿", "不宜直接投稿" in s)

# 6 证据分级
for lv in ["[R]", "[E]", "[A]", "[C]"]:
    ck(f"证据等级 {lv} 已定义并使用", lv in s)
ck("含 AI 使用声明模板", "附录 D：AI 使用声明" in s)
ck("fp32 无条件逐位已撤回", "撤回" in s and "容差内一致" in s)
ck("归档路径自洽", "SimplCore_Paper_v0.1alpha.md" in s)

# 7 关键数字齐全
for pair in [("0.2484", "0.2476"), ("0.5430", "0.4478"), ("10.817", "10.217"),
             ("0.8672", "1.0000"), ("1.92", "1.04"), ("0.9720", "17.67"),
             ("1.948", "120 GiB"), ("0.1094", "0.1575")]:
    ck(f"数字对 {pair[0]}/{pair[1]} 均在稿内", pair[0] in s and pair[1] in s)

n = sum(c["passed"] for c in checks)
out = dict(total=len(checks), passed=n, failed=len(checks) - n,
           whitelist_size=len(WHITELIST), ids_used=sorted(ids & WHITELIST),
           stray_ids=sorted(stray), checks=checks)
json.dump(out, open("results/paper_consistency.json", "w"), indent=1, ensure_ascii=False)
print(f"\n==> {n}/{len(checks)} PASS, {len(checks)-n} FAIL")
print("行数", len(lines), "字节", len(s.encode()))
