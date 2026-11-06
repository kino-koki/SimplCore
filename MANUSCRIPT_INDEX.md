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
