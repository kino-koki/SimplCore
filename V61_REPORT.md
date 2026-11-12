# v6.1：Theorem B 三维守恒 + 完结中断轮

> 用户指令：「续」——接着上一轮被中断的收尾工作。
> 训练/实验任务全程冻结；零 LM 训练、零 ppl、零生成样本、零私有权重字节。

## 0. 上一轮真实状态（先对齐，不重写历史）

上一轮被工具轮次上限打断时，**Theorem A、论文 v6、P0 清理、验证器、报告其实都已落盘，但从未推送**：

- 远端 `main` 仍是 `7158641`（v5.2，2026-11-07T09:49:46Z）
- 本地 `CHANGELOG.md` / `REVISION_LOG.md` 里**没有** v6 段（`push_v60.py` 从未执行）
- 4 份 `info_bound_*.json` witness 都在本地，未进仓库

所以本轮第一件事是「把已完成的工作收尾并推仓」，第二件事才是继续深化。

## 1. 新贡献：Theorem B（信息 / 串行深度 / 状态阶守恒）

主文 §4.12，前提、断言、完整证明、表格、witness、边界**全在正文同一节**，不靠附录。

| | 结论 |
|---|---|
| (a) | `B < log2|A|` 时任何深度/状态阶/核都不可行（Theorem A(b)） |
| (b) | `B >= log2|A|` 且 `P = 2` 时，计算图含长度 **H** 的串行内容匹配依赖链；**该数目与层数、核参数、输入侧 bit 数无关** |
| (c) | H 次匹配可以放在 H 层栈（1 bit role）或 1 层 + H 步读出链（hop-index 标注）；两者都精确，都是 H 次串行 |
| (d) | 把串行深度压到 1 必须让状态枚举 2^H 个端点；(H+1) 阶 witness 有 `d^(H+1)` 元素，**指数爆炸** |
| (e) | 因此 `(B, 串行深度, 状态阶)` 前沿是**离散阶跃**：输入 bit 只把计算搬家，不消除计算 |

这一条把上一轮「bit-深度直角前沿」升级为**三维守恒**，也把 reviewer 最致命的
「the format solves the task」进一步钉死：格式给的是信息，不是匹配次数。

**判别性 witness 130/130**（`train/superposition_probe/theoremB_probe.py`，纯 NumPy）：
5 seeds、H=1..4、四种 regime、穷举 bit 扫描。实测

```
H=1  serial(role/hop/tensor)=1/1/1   state=256/256/256
H=2  serial=2/2/1                    state=512/512/4096
H=3  serial=3/3/1                    state=768/768/65536
H=4  serial=4/4/1                    state=1024/1024/1048576
```

三种构造 acc 全 1.0000；`no_bits` 最优 acc = 2^(B−H)，阶跃恰在 B=H。

## 2. 修掉的硬伤（都是真问题，不是润色）

| 项 | 修前 | 修后 |
|---|---|---|
| 章节编号 | `3.7` 两次、`4.8` 两次，顺序 4.8→4.9→4.10→4.8→5 | 3.7/3.8/3.9/3.10 与 4.8/4.9/4.10/4.11/4.12，**无重复** |
| 附录 A.31 H5 | 「exactly one, **necessary and sufficient**」无条件，与 Theorem A(b) 直接冲突 | 改为「1 bit/token 是**充分载体**；必要的是总预算 `B >= log2\|A\|`」，R3 保留为 \|A\|=2 最坏情况特例 |
| Theorem A(b) 显式族 | 「m 个正交码行编码端点」表述含糊，端点数推导不成立 | 改为**二叉链构造**：每跳一个未标注的二选一 bit，H 跳 ⇒ 2^H 个可见输入相同但答案不同的实例，V ≥ 2^H 时端点互异 |
| References | 15 条省略作者，格式不规范 | 逐条从 **arXiv 官方 abs 页**核验作者/标题，补真实第一作者 + et al.，格式统一；核验记录落 `repro/arxiv_reference_audit.json` |
| PDF metadata | 仍是旧标题 | 与新标题一致 |
| 主文页数 | 7 页 | **8 页**（顶格，靠 Theorem B 的证明与表格填满，无注水） |

顺带说明：正文里 `necessary and sufficient` 在附录还剩 1 次，是 NT01 关于
「order-2 状态在严格正核下单跳检索必要且充分」的合法陈述，与 role bit 无关，保留。

## 3. 回归与自检

- Theorem A 验证器 **18/18**；Theorem B witness **130/130**
- `nt31 / one_bit_theorem / role_lemma / codebook_corollary / verify49 / verify_smm_sgc / verify` 全部 rc=0
- PDF **25 页，主文 8 页**（8 页限内），Broader Impact(p9) < References(p10) < Appendix
- 主文内部编号 `W9 / NT31 / 17/17 / EXACTLY` 全 **0**；实体残留 `&lt;` `&gt;`、TeX 碎片 `_{` `^{`、自识别路径（`sgc.py` / `loop.py` / `.json` / `results/` / `docs/` / `/data/workspace`）全 **0**
- PDF sha256 前缀 `8496532fc1ebb7fc54c5`

> 排版说明：PyMuPDF 对逐词 `drawString` 的文本提取偶尔会把一行拆成逐词行；
> 渲染图目视确认是提取假象，正文排版本身正常。

## 4. 诚实边界（不圆）

- **Theorem B 的串行深度陈述只在「一次双线性缩并至多解一次内容匹配」的计算模型内成立**，明写不排除其他算法范式。
- `(H+1)` 阶状态是**下界 witness / 查表构造**，论文里明确标注「NOT a proposed layer」。
- 仍是合成内容链、one-hot 正交码、单层/多层小规模、5 seeds；**没有**语言建模、val ppl、生成可读性、实测吞吐对标。
- **P1 剩余的最大缺口未补**：小、公平、≥5 seed 配对 CI 的同条件对照（MQAR / GLA / DeltaNet / Mamba-2 / 2 层 Transformer）。按上一轮评审意见，**没有**硬塞 ≥10M token 不等算力大实验，§5/§7 明写「deliberately did NOT do」。
- 不承诺过稿率。本轮改变的是贡献类型（信息下界 + 设计规则）与稿子的一致性，不是「已经能打」。

## 5. 交付物

| 文件 | 内容 |
|---|---|
| `simplcore.pdf` | v6.1 投稿稿，主文 8 页 |
| `build_simplcore_pdf_v5.py` / `patch_v61.py` | 唯一生成器与本轮修订脚本 |
| `V61_REPORT.md` / `V60_INFO_BOUND_REPORT.md` | 本轮与上一轮报告 |
| `train/superposition_probe/theoremB_probe.py` + `results/theoremB_results.json` | Theorem B 全部 witness 代码与原始数据 |
| `theory_work/info_bound.py` / `nt31.py` + 对应 results | Theorem A 证明收紧、口径冲突修复 |
| `repro/arxiv_reference_audit.json` | 15 条 arXiv 引用核验记录 |
| `CHANGELOG.md` / `REVISION_LOG.md` | v6.1 与 v6 登记 |

私有权重字节与 GitHub token 一律未进仓库。
