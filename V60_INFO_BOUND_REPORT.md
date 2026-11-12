# 本轮：把最大软肋做成核心定理（Theorem A），换贡献类型

> 用户判断「修错是防守，换定位才是进攻」——这一轮按此执行。
> 主交付是**新定理 A：输入侧角色信息的紧下界**，不是又一轮打磨。
> 训练/实验任务全程冻结；零 LM 训练、零 ppl、零生成样本、零私有权重字节。

---

## 0. 先说最容易被质疑的一点：这不是空定理

定理 A 的 (a)(b) 是**普适信息论**：不可区分性 + pigeonhole，对**任何确定性模型**成立，
不需要 LTI、不需要层数、不需要状态阶、不需要 read-out 假设。

如果假装它们依赖静态核，那就是用户警告过的「前提过窄导致空定理」。
所以论文里明写：

> Parts (a),(b) are UNIVERSAL ... we do NOT claim they are new information theory.

架构独特性放在 (c)(d)(e)(f)，这才是本文增量：

- **(c) 位置是被迫的**：在静态 LTI 类里，核系数时不变且内容无关，唯一的输入依赖通道是
  固定 embedding 与绝对位置；任意 filler 间隔下位置与角色分配独立 ⇒ 输出对角色不变。
  想把 bit 塞进核里 ⇒ 系数随输入变 ⇒ 时变 ⇒ 由 T4（Toeplitz IFF 时不变）丢 FFT/IIR 对偶。
  **要保对偶，bit 只能在输入端。**
- **(d) 紧性 + 三载体守恒**：|A|=2^m 的显式族使 B ≥ m；达到下界的三种载体
  （显式 tag / 码本分区 / 位置格式先验）每个 ambiguous token 都恰好 1 bit；
  只有码本分区既无标注又保对偶。格式先验把 bit 存在**输入分布假设**里，
  filler 一出现假设失效 ⇒ 有效 bit 归零 ⇒ 定量解释 1.0000 → 0.3438。
- **(e) 直角前沿**：B < log2|A| 时**任何深度**都不可行；B ≥ log2|A| 时最少深度仍恰为 H。
  bit 与层数之间不是平滑 trade-off 曲线，是阶跃。这条比 5×5 网格有信息量。
- **(f) overshoot 不是架构定律**：H_L > H 的失败来自末层读出 + 无 stop/select；
  保留中间层输出可使 H_L ≥ H 充分，下界 H 不变。

---

## 1. 与旧结论的口径冲突，已处理

生成器里已存在 **Lemma R3「恰好 1 bit 必要」**（对手 pair permutation，0 bit 在
某个合法数据集上必错）。定理 A 不能与它打架：

| | R3（旧） | Theorem A（新） |
|---|---|---|
| 断言 | 恰好 1 bit 必要 | B ≥ log2|A| bit 必要 |
| 范围 | 构造性存在、|A|=2、**最坏情况**（accuracy 0） | 任意 |A|=2^m、均匀分布、acc ≤ 2^(B-m) |
| 1 bit/token | 被当作必要 | 只是**充分载体之一**，既非必要也非信息量最小 |
| role tag | 隐含必要 | 码本分区证明 tag 完全不必要 |

R3 **没被推翻**，是 |A|=2 的最坏情况特例；定理 A 推广它并**撤回「恰好 1 bit」这个
过度断言**。这层关系写进定理 SCOPE (viii)、摘要、§3.7、§1.2。

---

## 2. 判别性 witness（不是模型性能，是结构断言的检验）

| 检验 | 结果 |
|---|---|
| **P2 前提可证伪**：2^m 个角色分配产生的 X 逐位相同 | max dev **0.0**，m=2,3,4 全过 |
| 穷举 bit 扫描 m=1..4 | acc = 2^(B-m)，**恰在 B=m 跳到 1.0000**；m=4: 0.0625/0.125/0.25/0.5/1.0 |
| 真实静态 LTI 前向 H_L=1..8，B<m | 同一 side-channel 纤维内输出**逐位相同**（dev 0.0），深度 1..8 完全不补偿 |
| B≥m 时充分深度 | 只有 H_L=H 达 1.0（复用已验证 grid）；select-H 读出使 H_L>H 也充分 |
| 三载体 | tag 1.0000 / codebook 1.0000（Toeplitz、FFT-IIR 2.89e−16）/ format prior 0.2969 |

验证器：**Theorem A 18/18**、witness 判定 **12/12**；
NT31 17/17、R1 12/12、R2 17/17 全部重跑通过。
witness 里「信息论最优解码器」与「架构实测」两种口径**分开标注**，不混用。

---

## 3. 论文 v6：换的是贡献类型，不只是措辞

`simplcore.pdf` **24 页，主文 7 页**（8 页限内，留 1 页余量）。

| 位置 | 改动 |
|---|---|
| 标题 | 《Where the Bits Must Live: A Tight Input-Side Information Bound for Fixed-State Spectral Models》 |
| 摘要 | 五句式重写，第一句即定理 A；明写 (a)(b) 普适、撤回「恰好 1 bit」、无 LM 主张 |
| Table 1（首页） | 新增 Theorem A 精确增量行；左列引用改规范编号 |
| §1.2 贡献 | 30 条 → **4 条**（1 信息下界 + 2 紧界 + 1 构造 + witness）；明写诚实不是原创性得分点 |
| §3 标题 | Six Core Theorems → **Four Core Results** |
| **§3.7（新）** | Theorem A 完整前提 P1–P5 同页陈述、claim (a)–(f)、proof route、scope、novelty 边界 |
| §3.1 / A.1、A.13 | **修证明漏洞**：dim ker>0 不能直接挑「扰动目标坐标」的 Δv；改为显式构造 Δv₁=β₂u, Δv₂=−β₁u ∈ ker L 且改变目标块 |
| §3.2 H5 / Claim(d) | 「EXACTLY ONE BIT」→「至少 log2|A| bit」；(d) 限定在 H1–H7 + 末层读出下 |
| §3.3 T3 | 归一化统一为 `C Cᵀ = I`（原文 `R I` / `I` 混用） |
| §4.8（新） | bit 扫描表、深度不补偿、三载体守恒、P2 前提检验 |
| §6 | **删掉与 Table 1 重复的 Table 2**（原文自己写「here we state the position once more」） |
| §7 | 删掉与主线脱节的 NT24「零阶权重写入速率」社区预测，换成定理 A 的可证伪 bit 下界 |
| §8 | 结论重写为信息下界主线 |
| Appendix A.0 | **Theorem A 完整解析证明**（6 处 QED + 8 条 SCOPE + R3 关系）放最前 |
| Appendix G | 51 项全量台账 → 一页 Statement Classification |
| Broader Impact | 移到 References **之前**，改成定理 A 口径 |
| References | 15 条统一格式；联网核验 arXiv ID ↔ 标题/venue；作者名省略并说明（避免编造） |

### P0 逐条处置

| P0 项 | 状态 |
|---|---|
| 摘要「necessary and sufficient」无条件和 §4.4 dense no-role 1.0000 冲突 | ✅ 改为 layout-conditional + 总 bit 下界 |
| A.1/A.13 的 ker → Δv 跳步 | ✅ 显式构造 ker 向量 |
| 码归一化四套混用 | ✅ 统一 `CCᵀ=I` |
| 表格 / chance / overshoot 口径 | ✅ §4.1 表保留并标清；§4.8 每列标含义 |
| 引用与匿名：幻觉引用、自识别路径 | ✅ 核验 + 规范化；`sgc.py`/`loop.py`/`core/arch`/`.json`/`results/`/`docs/` **全篇 0 次** |
| Broader Impact 排在 References 后 | ✅ 前移 |
| 全大写、W9/NT31/17/17 内部编号 | ✅ 正文全部 0 次 |
| self-evolution 工程日志整包（NT14/15/20/21/22/23/24） | ✅ 从附录定理循环过滤；附录 F/G 相关行改写或删 |
| 版式：`<b>` 原样画入表格、§1.1 跨栏孤立行 | ✅ 两个真渲染 bug 修掉，首页重渲染确认 |

**另修一个中断轮次留下的硬 bug**：生成器 §3.2 有重复 `%` 格式化会抛
`TypeError: not all arguments converted`，本轮修好才能编译。

---

## 4. 诚实边界（不圆）

- **定理 A 不声称首次发现信息论**，也不声称「输入格式能带任务信息」是我们先看到的。
  novelty 是**定量下界 + 位置被 Toeplitz 对偶强制 + 三载体守恒 + 直角前沿**。
- (c) 否定的是**单 FFT 卷积路径**，不是「不存在任何 O(L log L) 算法」；证明域是
  T4 的二阶递推（c_t ≠ 0）。
- 码本分区仍是**编码层归纳偏置**（KEY/VALUE 是不同 id）；主张的是无运行时标注、
  无位置规则、无格式先验，**不主张「无任何结构约定」**。
- 「1 bit/token」是充分载体，**不是信息量最小载体**；tight 的是总 B ≥ log2|A|。
- 定理 A 的 (a) 是**均匀分布**下的 ≤1/|A|；R3 是**最坏情况**下的 0，两者互补不矛盾。
- 精度只到 L=128；吞吐是未优化 NumPy 参考实现；合成内容链任务，**不是语言能力结论**。
- **P1 剩余**：小、公平、带 ≥5 seed 配对 CI 的同条件对照（MQAR / GLA / DeltaNet /
  Mamba-2 / 2 层 Transformer）——这一轮**没跑**。按建议，没有硬塞 ≥10M token、
  val ppl、生成可读性、吞吐对比这四类不等算力大实验；§7 明写「deliberately did NOT do」。

---

## 5. 关于过稿率

不承诺 60%。这一轮完成的是用户指定的「先做这道」：**把最值钱的软肋做成核心定理**，
并把定位从「边界刻画 + 我们很诚实」转成「一个紧的信息下界 + 别人能拿去用的设计规则」。
P0 清完后稿子不再会被 20 分钟 desk-check 打掉；P1 的小对照仍是下一个决定分数的动作。

---

## 6. 交付物

| 文件 | 内容 |
|---|---|
| `simplcore.pdf` / `build_simplcore_pdf_v5.py` | v6 投稿稿与唯一生成器 |
| `V60_INFO_BOUND_REPORT.md` | 本报告 |
| `theory_work/info_bound.py` | Theorem A 全文（前提/证明/边界/novelty/用途）+ 18 项验证器 |
| `theory_work/results/info_bound_*.json` | 定理全文与验证结果 |
| `tools/native_probe/info_bound_probe.py` | 三组 witness 全部代码 |
| `tools/native_probe/results/info_bound_*.json` | 4 份原始数据 + 12/12 判定 |
| `CHANGELOG.md` / `REVISION_LOG.md` | 进展与修订登记 |

私有权重字节、GitHub token 一律未推仓。
