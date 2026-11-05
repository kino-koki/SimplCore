# 相关工作与统一对照表（0.2-β）

> **文档性质**：`SimplCore_Paper.md` §8 的展开版。
> **核实纪律**：下列每一条都在 2026-11-05 通过检索核对了**标题 + arXiv ID +
> 发表年份/venue**。凡核不到明确 arXiv ID 的（如 CKConv 等）**一律不列入本表**，
> 只在正文以类别名提及。**投稿前仍需逐条核对正式出版版本与 BibTeX 字段**——
> 本表解决的是「幽灵文献」风险，不替代引文格式核验。
>
> ⚠️ **本文尚未做等参数 / 等算力定量对标**。本表是**定性分类**，
> 不是实验结果表。主稿 §8 与附录 C 已把「补统一对照实验」列为 P0 待办。

---

## 1. 统一对照表

分类轴取自本文的四条边界：**状态是否常数**、**核是否输入相关**、
**训练路径是否可用 FFT/卷积**、**是否具备 $(q,k)$ 内容寻址**。

| 工作 | arXiv | 年份 / venue | 状态 | 核输入相关 | 训练路径 | 内容寻址 |
|---|---|---|---|---|---|---|
| **S4** — *Efficiently Modeling Long Sequences with Structured State Spaces* | 2111.00396 | 2021 / ICLR 2022 | O(1) LTI | ❌ | FFT 卷积 | ❌ |
| **S5** — *Simplified State Space Layers for Sequence Modeling* | 2208.04933 | 2022 | O(1) MIMO LTI | ❌ | 并行扫描（弃卷积） | ❌ |
| **Hyena** — *Hyena Hierarchy: Towards Larger Convolutional Language Models* | 2302.10866 | 2023 | 隐式长卷积 | **部分**（data-controlled gating） | FFT + 门控 | 部分 |
| **Mamba** — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* | 2312.00752 | 2023 / COLM 2024 | O(1) selective | ✅ | 并行前缀扫描（卷积视图失效） | ✅（选择性） |
| **Mamba-2 / SSD** — *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* | 2405.21060 | 2024 / ICML 2024 | O(1) semiseparable | ✅ | 分块矩阵乘 + 递推 | ✅（与线性注意力对偶） |
| **GLA** — *Gated Linear Attention Transformers with Hardware-Efficient Training* | 2312.06635 | 2023 | O(1) 矩阵状态 | ✅（数据依赖门） | I/O 感知分块 | ✅ |
| **DeltaNet** — *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* | 2406.06484 | 2024 | O(1) 矩阵状态 | ✅（delta rule） | Householder 分块并行 | ✅（联想召回增强） |
| **Linear Transformer** — *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* | 2006.16236 | 2020 / ICML 2020 | $\phi(K)^\top V$ 常数状态 | ✅ | 线性 / RNN 递推 | ✅ |
| **RWKV** — *RWKV: Reinventing RNNs for the Transformer Era* | 2305.13048 | 2023 / EMNLP 2023 | O(1) 向量状态 | 部分（v4 固定衰减 / v6 数据依赖） | 并行训练 / RNN 推理 | 部分 |
| **TTT** — *Learning to (Learn at Test Time): RNNs with Expressive Hidden States* | 2407.04620 | 2024 | 隐藏状态本身是模型 | ✅ | 测试时梯度步 | ✅ |
| **LongNet** — *LongNet: Scaling Transformers to 1,000,000,000 Tokens* | 2307.02486 | 2023 | KV 随 $L$ 增长（稀疏 attention） | ✅ | 膨胀 attention | ✅ |
| **Jamba** — *Jamba: A Hybrid Transformer-Mamba Language Model* | 2403.19887 | 2024 | 混合（attention + Mamba） | ✅ | 混合 | ✅ |
| **Zoology / MQAR** — *Zoology: Measuring and Improving Recall in Efficient Language Models* | 2312.04927 | 2023 / ICLR 2024 | 评测协议（非架构） | — | — | 定义 MQAR，82% 差距由回忆解释 |
| **Repeat After Me** — *Repeat After Me: Transformers are Better than State Space Models at Copying* | 2402.01032 | 2024 | 理论 + 实验（GSSM） | — | — | 固定大小状态复制上界 |
| **The Illusion of State in State-Space Models** | 2404.08819 | 2024 / ICML 2024 | 复杂度类分析 | — | — | S4/Mamba 限于 $\mathsf{TC}^0$ |
| **本文：SimplCore / SGC** | 仓内 `SimplCore_Paper.md` | 2026 | **O(1) 静态谱核** | ❌ | **FFT 整段 / IIR 流式双形式** | ❌（结构性缺失） |
| 本文 SFSC（猜想，已证伪对偶） | 仓内 `docs/SFSC_FDCAO_PILOT.md` | 2026 | O(1) LTV | ✅ | **FFT 失效 → $O(LM)$ scan** | ❌（实测与静态持平） |
| 本文 FD-CAO（猜想，部分成立） | 同上 | 2026 | O(1) 频率槽 | ✅ | $O(LM)$ | ✅（等参数仍输 attention） |

> 表中「❌/✅」是**机制层定性判断**，不是实测性能结论。
> 「本文」三行不占用外部文献条目，全部指向仓内文件（主稿附录 A）。

---

## 2. 分组解读

### 2.1 结构化 SSM / 长卷积（S4、S5、Hyena）

S4 确立了「LTI SSM = 递推 / 卷积双视图 + FFT 近线性训练」，
与本文定理 1 同源；S4 论文自己也承认「作为 LTI 系统，动力学不依赖输入内容」，
并留有与 Transformer 的语言建模差距。S5 明确指出**卷积视图要求时不变**，
因此改用纯时域并行扫描——这正是本文引理 B4 的已知来源。
Hyena 用「隐式长卷积 + 数据控制门控」在不完全放弃 FFT 的前提下引入输入相关，
是本文 SFSC 猜想最接近的既有工作；差别在于 Hyena 的门控是对角矩阵而非修改
振荡器参数 $\rho,\theta$。

**本文与这一组的关系**：SGC 把振荡基字典与**分路径逐位口径**（流式 fp32 逐位 /
整段 fp32 容差）显式分开，并给出频率网格塌缩的数值–表征双重解释；
同时用 B4 说明「一旦让核随上下文变，就必然离开 FFT 路径」，
把 S5/Mamba 的已知工程结论写成可检查的引理。

### 2.2 选择性 / 线性注意力状态（Mamba、Mamba-2、GLA、DeltaNet、Linear Transformer、RWKV、TTT）

这一组全部通过**输入相关的状态更新**获得 $(q,k)$ 内容寻址能力，
代价是放弃 LTI 卷积视图。Mamba-2/SSD 进一步证明一类结构化 SSM 与线性注意力
形式等价——这支撑本文推论 B1.1（内容寻址状态与 attention 同源）。
DeltaNet 报「1.3B / 100B token 上超过 Mamba、GLA，但 hybrid 仍需 attention 层
才能超过强 Transformer 基线」，与本文 §6.4 的分工依据同向。

**本文与这一组的关系**：FD-CAO 本质是把 $\varphi(k_s)v_s^\top$ 压缩到 $M$ 个
固定频率槽（见主稿 §5.5「边界张力」）。因此**本文不声称 FD-CAO 是全新机制**，
只把它作为「在 O(1) 频率槽内补内容寻址」的一个可证伪实例，
并如实报告等参数下仍输低秩 attention（$0.8672$ vs $1.0000$）。

### 2.3 混合与长上下文（LongNet、Jamba）

两者都保留 attention（稀疏或 interleaved），属于本文 §6.4 讨论的 hybrid 方向。
**本文不主张 hybrid 已被验证**，只把 B1 + §5.5 作为「为何需要分工」的理论动机。
LongNet 与 Jamba 也说明：本文的「无注意力」约束是**自设边界**，不是社区共识。

### 2.4 表达力边界的理论与评测（Zoology、Repeat After Me、Illusion of State）

这是与本文**最容易被混淆**的一组，必须正面写清：

- Zoology 定义 MQAR，并把门控卷积与 attention 的 ppl 差距 82% 归因于联想回忆；
  其 hybrid 弥合 97.4% 差距。
- Repeat After Me 证明固定大小状态的 GSSM 无法复制超过隐状态比特数的串，
  并给出 Transformer 的构造性优势。
- Illusion of State 把 S4/Mamba 的表达力界定在 $\mathsf{TC}^0$，
  指出「状态」在表达力意义上是幻觉。

**因此「静态 / 常数状态模型在内容寻址与复制上弱于 attention」不是本文首次发现。**
本文的理论贡献是：把这一已知结论归约到 SGC 输出的一个可检查代数性质
（可加分离），给出几行证明；再把它与「选择性破坏卷积视图」绑定到
同一条架构约束，得到对偶性与上下文相关核的互斥性。
完整表述见 `docs/EXPRESSIVITY_BOUNDARIES.md` §5。

---

## 3. 明确缺口（投稿前必须补）

| # | 缺口 | 状态 |
|---|---|---|
| 1 | **等参数 / 等算力定量对标**：在相同 token 预算、相同参数量下，把 SGC 与 Mamba-2、GLA、DeltaNet、Hyena、TS 放在同一张 ppl + 生成质量表里 | ⬜ 未做，P0 |
| 2 | 上述 15 条引文的**正式出版版本与 BibTeX 字段**逐条核验（arXiv ID 已核，venue/页码未核） | ⬜ 部分完成 |
| 3 | 与 $\mathsf{TC}^0$ 表达力结果的正式理论对接（Merrill et al., ICML 2024） | ⬜ 未做 |
| 4 | CKConv 等连续核卷积工作的可核验条目（本轮未核到明确 arXiv ID，故未列入） | ⬜ 未做 |
| 5 | ICML 2025–2026 同类工作的系统检索（本轮只核到 2024 及以前的经典条目；检索结果中出现一篇 ICML 2026 同名延伸工作，因无可核验编号未列入） | ⬜ 未做 |

> ⚠️ 主稿 §8 已把「没有统一对照实验表」写成明确缺口而非措辞遗漏。
> 本表只补齐**文献条目与定性分类**，**不替代**定量对标实验。
> 在补齐 #1 之前，投稿仍属「不宜直接投稿」（主稿 §7 / 附录 C）。
