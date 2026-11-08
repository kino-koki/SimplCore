# ICML 投稿包 · 索引（2026-11-04 最终版）

## 本轮新增（证据分级）

用户口径：**只有已测试可复现的才写成结论；不可复现的统一按猜想写。**

| 文件 | 内容 |
|---|---|
| **`ICML_evidence_graded.md`**（=《ICML_证据分级版.md》） | **本轮主交付**：论文骨架，正文只写 29 项已验证结论（逐条 `[Vxx]`），14 项猜想集中 §3 |
| `证据分级台账.md` | 43 项人类可读总表（29 PASS / 0 FAIL / 14 猜想） |
| `results_verification.json` | 一键验证的机器可读台账 |
| `verify_all.py` | **一键复现验证**：从代码重跑 R/A 类、从原始记录独立重算 S 类 |
| `make_evidence_graded.py` | 从台账生成分级版文稿（保证表格与数据一致） |
| `check_grading.py` | 学术一致性自动检查：18 PASS / 0 FAIL |
| `reproduce.sh` | 训练类重跑命令清单（本轮未重训，如实标注） |
| `results_grading_check.json` | 一致性检查结果 |

## 三道检查结果

```
audit_against_paper.py   46 项引用核对：35 PASS / 11 FAIL（全部报告侧，已修订）
verify_all.py            43 项证据分级：29 PASS / 0 FAIL / 14 猜想
check_grading.py         18 项学术一致性：18 PASS / 0 FAIL
```

## 之前两轮的交付（保留）

- `ICML_revised_report.md`：22 处修订的完整报告 + §16 核对章节
- `ICML_checklist.md`：投稿核查清单（A/C 完成、B/D 待补、G 证据分级完成）
- `results_citation_audit.json`、`audit_against_paper.py`、`make_icml_revision.py`
- 全部 `results_*.json`、`nn.py` / `sgc_core.py` / 各实验脚本（原始数据未删改）
- `excluded/`：被打断的半成品，已排除出投稿材料

## 关于「完全可复现」的诚实边界

29 项 PASS 中，R 类（从代码重跑）与 A 类（解析重算）是字面意义的完全复现；
S 类是**从原始 held-out `correct` 列表独立重算统计结论**（bootstrap / 回归），
不是重新训练模型。模型断点已清理，训练类结果的重跑命令在 `reproduce.sh`，
本轮未重训——这一点已写入分级版 §5，未掩盖。

## 仍需作者补（数据完整性，非诚信问题）

≥5 seed、等步数/等算力对照、平凡基线、合法消融、统一 batch 重测斜率、
相关工作 DOI 核对、commit hash 与复现脚本、双盲匿名化、目标年份 CfP 条款。
见分级版 §6。补不齐须在 Limitations 如实标注。
