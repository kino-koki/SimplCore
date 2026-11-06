# 稿件与证据链索引

> 本文件说明仓库里各类文档的角色，避免把工作稿、复现报告与投稿稿混为一谈。

## 唯一投稿稿

| 文件 | 说明 |
|---|---|
| `simplcore.pdf` | 英文、匿名、ICML 版式。主文 8 页上限内，References / Broader Impact / Appendix 单列。**只有这一份是稿。** |

## 日志类文档（由原 markdown 论文稿整合而来）

| 文件 | 说明 |
|---|---|
| `CHANGELOG.md` | 更新日志：各阶段实际进展与实测结论，含 `[待补]` 清单 |
| `REVISION_LOG.md` | 修订日志：版本序列、定理重标、撤回记录、数字口径修正、格式合规 |
| `SimplCore_Paper.md` | 稿件状态说明（原工作稿已归档，不再是论文正文） |

## 历史稿归档（只读，不作结论引用）

| 文件 | 版本 |
|---|---|
| `docs/SimplCore_Paper_v0.1alpha.md` | 0.1-α 原始技术报告，逐字节归档 |
| `docs/SimplCore_Paper_v0.2beta.md` | 0.2-β 边界刻画版，逐字节归档 |

## 架构内部理论层（不是稿，不改现有源码逻辑）

| 文件 | 说明 |
|---|---|
| `simplcore/core/theory/statements.py` | 15 个历史命题（含 T16 撤销标注） |
| `simplcore/core/theory/smm_sgc.py` | **CT6/CT7/CT5-L2 完整证明 + CT1–CT7 收束表 + 猜想判定** |
| `simplcore/core/theory/bridge.py` | SMM×SGC 只读接线规格 + 零阶算子（默认关闭） |
| `simplcore/core/theory/verify.py` | 15 命题：9 定理 / 3 证伪 / 2 猜想 / 1 撤销 |
| `simplcore/core/theory/verify_smm_sgc.py` | CT6/CT7/CT5-L2 + 前提非空检查，7/7 |
| `docs/CONJECTURE_15_THEOREM_UPGRADE.md` | 上一轮 15 命题完整证明与 RSI/SMM 核实 |
| `docs/SMM_SGC_SELF_EVOLUTION.md` | 上一轮：四条审稿意见回应、CT6/CT7 完整证明、收束表 |
| `docs/THEOREM_30_AUDIT_49.md` | **本轮主文档：30 条定理（6 组）、19 项定级、49 项全量判定** |
| `simplcore/core/theory/new_theorems15.py` | NT01–NT15 完整解析证明 |
| `simplcore/core/theory/new_theorems30.py` | **NT16–NT30 完整解析证明 + THEOREMS30** |
| `simplcore/core/theory/verify15.py` | 第一梯队一致性检查 15/15 |
| `simplcore/core/theory/verify49.py` | **49 项全量审计 49/49** |
| `simplcore/core/theory/ledger49.py` | 49 项判定台账（单一真源） |
| `artifacts/theorem49_results.json` | 49 项判定原始结果 |
| `docs/simplcore_v0.2beta.pdf` | 0.2-β 旧稿归档 |
| `docs/build_simplcore_pdf_v3.py` | v3 生成器归档 |
| `artifacts/theorem15_results.json`、`artifacts/smm_sgc_results.json` | 验证原始结果 |

定理编号：历史 T7–T16 已收束为 CT1–CT7，并进一步演进为投稿稿的
**NT01–NT30**（6 组）；`simplcore.pdf` v4 **已并入**，附录 A 给全部 30 条完整证明，
附录 G 给 49 项判定台账。见 `REVISION_LOG.md` §6.1 / §6.2 / §6.3。

## 证据链（数据与报告，不是稿）

| 类别 | 位置 |
|---|---|
| 结构性质、守护、网格、吞吐原始 JSON | `repro/results_*.json` |
| 引用逐条核对、证据分级台账、一致性自检 | `repro/results_citation_audit.json`、`repro/evidence_ledger.md`、`repro/results_*.json` |
| 上游源码待修清单与真实包索引 | `repro/upstream_fixlist.md`、`docs/` 相关说明 |
| SFSC / FD-CAO 判别性前哨 | `docs/SFSC_FDCAO_PILOT.md`、`tools/sfsc_fdcao_probe/` |
| 表达力边界证明与相关工作核实 | `docs/EXPRESSIVITY_BOUNDARIES.md`、`docs/RELATED_WORK.md` |
| 自研字典、张量谱卷积、物理边界的全部脚本与原始结果 | `tools/native_probe/`（含 `results/` 下全部 JSON、`README_*.md` 报告） |

`tools/native_probe/README_*.md` 是分阶段实验报告，记录当时口径；若与
`simplcore.pdf` 冲突，以投稿稿与 `REVISION_LOG.md` 的撤回记录为准。

## 使用约定

1. 引用结论时引用 `simplcore.pdf`；引用数字时引用对应原始 JSON。
2. 任何「已验证 / 猜想 / 撤回」的口径问题，查 `REVISION_LOG.md`。
3. 新实验只往 `CHANGELOG.md` 追加，不改动历史条目。
4. 不把脚本断言计数当作定理编号；不把有限样本观察写成定理。
