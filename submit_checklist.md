# 010 · submit_checklist.md — 匿名化 / 合规 / 冻结清单

> **状态：PHASE 7 挂起。** 两个独立原因，任一未解除都不得提交：
> 1. ICML 2027 CFP / Author Instructions 未发布（`cfp_lock.md`）；
> 2. Phase 6 的 60% 内部质量门为 `NOT EVALUABLE`（`review_sim.md`）。
>
> 技能包 §2.4 原句，交付时原样附：
> **页数、披露、互审条款以当年 CFP 为准；本轮按已公布信息 + 往年惯例预排，不构成合规承诺。**

## 0. 提交包的唯一内容

只允许两个 PDF 进入提交包：

| 文件 | 页数 | 内容 |
|---|---:|---|
| `paper_main.pdf` | **8**（7 页正文 + 1 页 Broader Impact/References） | §1–§8 + Broader Impact + References，无附录 |
| `paper_appendix.pdf` | **9**（1 页封面 + 8 页附录） | Appendix A–J，全部完整解析证明与原始数据表 |

`simplcore.pdf`（16 页合并版）**只作本地阅读副本，不进提交包**。
内部工作文件（`cfp_lock.md` / `claim_registry.md` / `proof_audit.md` / `p0_scan.md` / `number_freeze.md` / `review_sim.md` / `rebuttal_prep.md` / 本文件 / 全部 JSON / 生成器 / 探针脚本）**一律不随稿提交**。

> 版式提示：ICML 2026 惯例是「单份主 PDF + 可选 supplementary」。本轮双 PDF 拆分是技能包 §0.5 的输出契约；2027 提交系统开放后必须按其要求重新打包，可能需要合并回单 PDF。此项列为 §3 的硬项。

## 1. 已完成的合规项（本轮机器复核）

| 项 | 结果 | 证据 |
|---|---|---|
| 双盲：正文/附录无作者、机构、致谢 | ✅ | `split_check.json` banned fragments = [] |
| PDF 元数据匿名 | ✅ | title/author/subject 重写，`creator`/`producer` 清空 |
| 自识别路径 / 源码文件名 / 内部编号 | ✅ 0 | `.py` / `.json` / `results/` / `docs/` / `core/arch` / `github.com` / `kino-koki` 扫描 = 0 |
| 对审稿人直接喊话 / 预演 rebuttal 元话术 | ✅ 0 | 7 类措辞扫描 = 0 |
| 撤回登记册编号、check 计数作为贡献 | ✅ 0 | Appendix B 只作 soundness 记录；`45/45` 等不进 §1.2 |
| Broader Impact 在 References 之前 | ✅ | `paper_main.pdf` 第 8 页 |
| 主文页数在上限内 | ✅ 7 页正文（2026 上限 8） | `desk_check_v7.py` |
| 实体 / TeX 碎片 / 乱码 | ✅ 0 | 45/45 desk-check |
| 摘要与全文一致（universal vs class-M 分开） | ✅ | `p0_scan.md` 第 6 项 |
| 数字单源、表 caption 自包含、chance 唯一 | ✅ | `number_freeze.md` |
| 编号唯一、P0 清零 | ✅ | `claim_registry.md` / `p0_scan.md` |
| 只投一篇、无同主题切碎投稿 | ✅ 本轮承诺 | 见 §3 硬项 |

## 2. 挂起 / 未完成（必须逐条解除）

| # | 项 | 状态 | 解除条件 |
|---|---|---|---|
| H1 | 2027 官方 `.sty` 替换 + 页数复核 | 🔴 未发布 | CFP 发布后取 `icml2027.sty`，重建并重核 8 页上限、references 是否计入 |
| H2 | 作者侧 LLM 使用披露 | 🔴 政策未发布 | 按当年政策重写；**不写「AI 未产生任何内容」类绝对句**；说明方法论层面的显著使用 |
| H3 | reciprocal reviewer 提名 | 🔴 待办 | 截稿前 **4 周**确认被提名人资格与 OpenReview 记录；不排到截稿后 |
| H4 | 投 ≥4 篇者须实际参与评审 | 🔴 待办 | 若本轮投稿总数达阈值，须落实 reviewer/AC/SAC 身份 |
| H5 | concurrent submission / thinly sliced 自查 | 🟡 部分 | v7 已删旧 ledger、本轮删正文重复证明；截稿前须确认无同主题并行投稿，若有须互相引用并提供匿名 PDF |
| H6 | **References 逐条核验** | 🔴 **本轮未做** | 14 条逐条核对作者、标题、venue、arXiv ID/DOI；v7 已统一 venue 并修 Zoology→ICLR 2024，但**不声称已核验**；幻觉引用可直接 desk reject |
| H7 | 匿名补充材料冻结 | 🟡 待办 | 代码/数据 zip 在截稿时固定，截止后不可改；文件名、注释、PDF 元数据同步清理 |
| H8 | 摘要与全文逐句对齐的最后一轮人工复核 | 🟡 待办 | 45/45 是字符串检查，不是数学证明 |
| H9 | 3 名独立 blind review | 🔴 未做 | 见 `review_sim.md` §5；门实测 `NOT EVALUABLE` |
| H10 | 双 PDF → 单 PDF 重新打包 | 🟡 待办 | 取决 2027 提交系统 |
| H11 | 相机就绪版（+1 页、lay summary、COI） | ⬜ 录用后 | 2026 惯例；2027 待核验 |
| H12 | 实验 significance 短板 | ⬜ 未补 | ≥10M token LM / 同算力强基线 / 真实 MQAR 多层；**不承诺补，也不伪装已补** |

## 3. 截稿前 1–2 周执行顺序

1. 取官方 CFP → 更新 `cfp_lock.md` 全部 `待复核/UNCONFIRMED` 字段；
2. 用 `icml2027.sty` 重建 `paper_main.pdf` / `paper_appendix.pdf`，复核页数与 ≤10 MB；
3. 逐条核验 References（H6）；
4. 确认 reciprocal reviewer（H3/H4）与 concurrent submission 自查（H5）；
5. 冻结匿名补充材料（H7）；
6. 按当年政策重写 LLM 披露（H2）；
7. 做最后一轮摘要—全文逐句对齐（H8）；
8. 若 2027 沿用 2026 的投稿前 LLM 反馈通道，用一轮外部视角预演审稿；
9. **不先投草稿**：ICML 2026 起录用稿会公开原始投稿版与审稿意见，`先投再改` = 公开社死。

## 4. 不主张清单（提交前再确认一次）

- 不作 LM parity / perplexity / 生成可读性 / 吞吐绝对值主张
- 不作与 attention 的等价、优越、低劣或表达力优势主张
- 不作 SOTA、新序列模型家族、通用自进化主张
- 不承诺任何录用概率；60% 门只作内部质量代理指标

## 5. 判定

**Phase 7：挂起。** 两个 PDF 已按技能包输出契约备好，但在 H1/H2/H3/H6/H9 解除之前，不构成可提交件。

## v7.2 清单更新

- **H6 References**：[4] 已改真 S5、[11] 已改 arXiv:2102.02611，另有 [4]/[8]/[7]/[9]/[11] + S4D 编号联网核验；**[10] Hyena venue 仍未取得独立权威页面，保留待复核**，故 H6 仍为「部分完成」，不勾全绿。
- **H2 LLM 披露**：Broader Impact 已补中性一句（实现辅助 + 语言编辑，研究问题/证明/结论由人类作者指定并核验），并明写「须按 ICML 2027 author instructions 复核」。代码文件内**未添加**任何 AI 声明（按用户此前要求）。
- **页数**：`paper_main.pdf` 9 页 = 8 页正文 + 1 页 Broader Impact/References；按 ICML 2026 惯例正文仍在上限内。2027 `.sty` 发布后须重核。
- **Phase 7 仍挂起**：CFP 未发布、独立 blind review 未做、双 PDF 重打包待定。
