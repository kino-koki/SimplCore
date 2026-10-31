# .archive —— 历史归档区

本目录存放**已退役、不再参与 CI / 构建 / 文档主张**的历史材料。
归档物保留原文与 git 重命名历史，仅供追溯；一律不得作为当前架构的能力证据引用。

## 2026-10 架构整改归档清单

| 归档物 | 原位置 | 说明 |
|--------|--------|------|
| `java-high-performance/` | `java-high-performance/` | SCArch 谱卷积 Java 平行原型（约 5.3 万行，NON-NORMATIVE）。与 Python 规范来源无自动同步机制，数值语义分叉风险长期存在，整体移出工作主线。其内部 README 与隔离测试（`test_impl_isolation.py`）继续生效。 |
| `qxt_java_legacy/java/` | `src/main/java/` | 早期 QXT 模型遗留代码（`com.luckyos.quickly`，`QXTModelExpanded` 系列），与 SimplCore v3.0 产品线无引用关系，从未参与打包。 |
| `docs_2026_08_24/` | — | 2026-08 文档快照（报告/计划/专利披露等）。 |
| `simplcore_docs/` | — | 更早期的技术报告与设计稿。 |
| `superpowers/`、`learnings/` | — | 历史 spec/计划与经验记录。 |

## 约定

- 归档 = 冻结：内容不再更新，不再修复，不再解读为"现行架构"。
- 恢复使用必须先通过 conformance 实测并重新评审，不得直接复活。
- Java 原型若未来重启，应从归档检出后独立建分支，不得移回工作主线
  （该约束由 `simplcore/core/tests/test_impl_isolation.py` 守护）。