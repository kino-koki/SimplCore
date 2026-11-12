# 稿件状态

**本文件不再是论文正文。** 它曾是 0.1-α / 0.2-β 工作稿的载体，现已按
「原稿留档作证据链、另立投稿稿」的方式整理：

- 唯一投稿稿：[`simplcore.pdf`](simplcore.pdf)（英文、匿名、ICML 版式）
- 实际进展与实测结论：[`CHANGELOG.md`](CHANGELOG.md)
- 口径变更、定理重标与撤回记录：[`REVISION_LOG.md`](REVISION_LOG.md)
- 全部文档角色与证据链索引：[`MANUSCRIPT_INDEX.md`](MANUSCRIPT_INDEX.md)
- 历史稿逐字节归档：`docs/SimplCore_Paper_v0.1alpha.md`、
  `docs/SimplCore_Paper_v0.2beta.md`

其余 `README_*.md`、复现报告、原始 JSON 均为证据链材料，不是稿件。
若旧稿表述与投稿稿冲突，以 `simplcore.pdf` 与 `REVISION_LOG.md` 为准。

投稿前硬门槛（语言建模主实验、≥5 seed 配对 CI、等算力对标等）仍未补齐，
见 `CHANGELOG.md` 末节。

0.1-α 二次审计：命题 5 的双口径勘误已锁定为 k·u·pert 主口径，旧稿中的裸 u 口径仅作存档对照；对数频率网格的保守修正因子为 1/|sin θ| ≈ 2607.6，旧值 5.13 乃勘误。
