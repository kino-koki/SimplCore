# `simplcore.pdf` 重排版报告

日期：2026-11-08 · 输出：`simplcore.pdf` / `paper_main.pdf` / `paper_appendix.pdf`

## 走的是哪条路

**重排既有投稿稿（重建流程），不是套通用报告模板从零生成。** 这份稿的硬约束是 ICML 式 US Letter 双栏、主文 8 页上限、匿名、全部数字由原始 JSON 注入；换成通用 HTML 报告模板会同时破坏这四条。所以重写的是渲染层（`build_simplcore_pdf_v8.py` 里的 `Doc` 类与几处版面调用），**内容层一个结论、一个数字、一条定理都没动**。

重建前先逐页渲染定位异常，且刻意用了全新预览路径——按路径缓存的旧预览会让人把上几轮的旧版排版误当成当前稿。

## 修掉的七类排版缺陷

| # | 缺陷 | 修法 |
|---|---|---|
| 1 | **首页标题第三行与 "Anonymous submission" 重叠**（基线只差 9 pt，15 pt 与 9.5 pt 字形互压） | 标题块重排，匿名行下移 23 pt；实测两框垂直间距 **12.13 pt** |
| 2 | **Table 1（首页精确差异表）挤爆**：6.5 pt、两窄列、每格 5–7 行，几乎不可读 | 单元格措辞压缩为每格 2–3 行、列宽 `[112,131]`、字号 6.9、行距 8.4、表后留白 5 pt；事实与引用编号一字未改 |
| 3 | **表格被拦腰劈到另一栏/另一页**，长标识符（URL、`arXiv:` 号）溢出到相邻列 | `table()` 加 **keep-together**（整表放不下才换栏/换页）、**字符级断词**、每行绘制前 `ensure`、表格线偏移 2.5 pt |
| 4 | **长章节标题溢出右边界**：`Appendix E …` 曾画到 x=598.7，`Appendix I …` 甚至到 612.7（超出页宽） | `heading()` 按栏宽自动换行；现在所有标题都在版心内 |
| 5 | **Figure 2 依赖图两个框的文字互相压住**（原先按「字符数 > 34」断行，不按实际字宽） | 改为按 `stringWidth` 断行、字号 6.2→6.6、行距 7→7.6 |
| 6 | **§2 公式段被两端对齐拉成忽宽忽窄的字距**；并残留一句重复尾巴 `absorbed by the unit normalisation in H6.` | 两段公式文字设 `justify=False`；`para()` 的对齐额外字距封顶为半个空格；删重复句 |
| 7 | **第 9 页只填满左栏，右栏全空；References 行距 1 pt、字距拉伸、`[10]/[11]` 挤在一起** | References 改成双栏均衡（[1]–[7] 左 / [8]–[14] 右）、字号 8.4、行距 11、段后 3 pt、不对齐 |

顺带统一了 PDF 元数据标题（原先元数据写着另一个旧标题，与首页标题不一致）。

## 校验

| 检查 | 结果 |
|---|---|
| 生成器 `py_compile` | PASS |
| PDF 重建 | **18 页，主文正文止于第 8 页**（8 页限内） |
| 几何核验（全部 18 页逐 span） | **0 处越界 / 0 处 <6 pt 字 / 0 处同行重叠** |
| 标题–匿名行垂直间距 | 12.13 pt，分离 |
| desk-check（D1–D7 + P + P1） | **46/46 PASS** |
| 拆分投稿包 | **11/11 PASS**，禁用片段 0；`paper_main.pdf` 9 页 / `paper_appendix.pdf` 10 页 |

> 两个页数口径：生成器报「主文正文止于第 8 页」，desk-check / 拆分把 Broader Impact 与 References 计入第 9 页；**正文仍在 8 页预算内**。

## 页数为什么从 17 变 18

排版舒展后附录自然多分一页：长标题换行占行、References 双栏、表格 keep-together 不再把行挤到栏底。**没有删内容，也没有注水**；`paper_appendix.pdf` 10 页（含封面）仍不计主文页数。附录最后一页（Table 34 续表 + "End of artifact"）只占左栏上半，是内容量决定的尾部留白，不是排版错误。若你坚持总页数 17，可以再压附录表格行距，但会牺牲刚修好的可读性——不建议。

## 没动、也没圆的

- 未新增或删除任何定理、命题、猜想、实验数字；所有数值仍由 `results/*.json` 注入。
- 未跑训练、未做迁移、未推仓；`GH_TOKEN` 不在环境中。
- 当前稿以磁盘上现存的生成器为准（其文本含上一轮那五处审稿修正）；本次只重排版式。
- ICML 2027 官方 `.sty`、CFP、截稿日、reciprocal reviewer、LLM 使用披露仍未发布/未办；≥10M token LM、同算力强基线、真实 MQAR、TS2SC 端到端迁移仍一项未补。
- 不作 LM parity / SOTA / 过稿率 / 吞吐主张。

## 可复现

```bash
cd /data/workspace
python3 build_simplcore_pdf_v8.py          # -> simplcore.pdf
python3 tools/eval_baselines/desk_check_v7.py
python3 split_submission.py                 # -> paper_main.pdf / paper_appendix.pdf
```

补丁与备份：`relayout_v8.py`、`relayout_v8_fix2.py`、`build_simplcore_pdf_v8_prerelayout.py`（重排前生成器，可一键回退）。
