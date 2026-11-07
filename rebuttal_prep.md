# 09 · rebuttal_prep.md — 三条必杀质疑的答复骨架

> 技能包 §P2：三条必杀质疑必须在正文里就堵死，等 rebuttal 再解释基本来不及。
> 本文件给出**答复骨架**，不是替审稿人写的答案；每条都标注「正文已答 / 部分回答」。
> Rebuttal 只做三件事：soundness 补前提、originality 逐条回差异表、significance 重申搜索空间封闭性意义。
> 绝不用大段文字重复论文已有内容。

## Q1 — "Isn't this just linear attention with extra steps?"

**审稿人真实意图**：判断 novelty 是否只是已知状态 `Σ k_τ v_τᵀ` 的换皮。

**正文已答**（§3.3.1 + Table 1）：
- `ρ=1` 且单极点时，Construction C 的状态**确实**退化为线性注意力状态；论文自己承认，不回避。
- `ρ<1`、多极点时状态为 `Σ_τ K(lag) k_τ v_τᵀ`，保留 SGC 独有的衰减谱；Toeplitz / 单 FFT / O(1) 状态仍成立。
- **不宣称脱离线性注意力范式**。本稿贡献是「在静态 LTI 核约束下，用张量状态 + 缩并读出实现内容寻址的构造」+「输入侧信息预算的精确下界」，不是新序列模型家族。

**Rebuttal 骨架（originality）**：
> 我们同意在 ρ=1 单极点极限下状态与线性注意力同构，论文 §3.3.1 已明确写出。差异不在状态形式，而在约束与结论：本文固定核时不变（P6 证明这等价于 Toeplitz），并在该约束下量化「精确检索还缺多少输入侧 bit」。Table 1 逐条列出与 S4/S5/Zoology/Mamba/GLA 的已知结果和精确增量；我们不主张这些已知观察为本文首创。

**不得做**：不把「已知 LTI 要求时不变」写成新定理；不把「fixed-state recall 弱」写成新发现。

## Q2 — "The retrieval is solved by the format, not by the model."

**审稿人真实意图**：判断下界是不是为定理量身定做的任务，且 tightness witness 有没有偷换量。

**正文已答**（Theorem A(d) + §2 + Appendix A.3/H/I）：
- `B = Σ_t log₂|π⁻¹(v_t)|` 把「实际字母表相对 flat 参考的超出量」**计入预算**。字母表分区 `V→2V` 给每个非空 token 恰好 `log₂(2V_K/V_K)=1 bit`，**这个 bit 在 B 里**。
- 三个载体（显式 tag / 码本分区 / 位置先验）中，前两个向 B 加 bit；第三个是**支持集限制**，作用于 `|A(v)|` 而非 B，在任意 filler 下失效（实测 0.3438, 5 seeds）。
- S4 同量检查：`governance_s4.json` 机器验证下界与紧性见证使用同一个 B；按 v6 的旧定义反事实运行会在 m=1..4 产生 4 条 `|A|=2^m > 2^B=1` 矛盾。
- **主动写明**：tightness 在 binary-chain witness 上是紧的；对原始 content-chain 任务，分区给 8 bits 而 `max|A| ≤ 5`（`log₂|A| ≤ 2.32`），因此是**充分但不紧**，Appendix H / Table 32 不隐藏这一点。

**Rebuttal 骨架（soundness）**：
> 审稿人指出的正是 v6 的硬错，我们已在 §2 改写 B 的定义并把紧性见证的 bit 计入同一个量。附录 H 给出可执行 bit-accounting 表与旧定义的反事实矛盾。若仍有疑虑，Table 32 明确区分「本任务上充分非紧」与「binary-chain 上紧」，我们不把前者包装成主任务成就。

**不得做**：不用 "information does not disappear; it changes carrier" 之类话术替代公式；不把 0.3438 与 0.2969 混为同一配置。

## Q3 — "Toy synthetic, zero training, no real LM evidence."

**审稿人真实意图**：significance 与实用性。

**正文部分回答**（§4.12 / §4.13 / §5 / Broader Impact）：
- 1500 步 Adam、从随机初始化、5 seeds、paired CI：层数下界在训练后仍成立（Table 15）。
- 零训练解析构造 1.0000 被明确标注为「特解 lookup table 风险」，不冒充方法泛化。
- §5 明写：**不作 LM parity / perplexity / 生成可读性 / 吞吐 / SOTA 主张**；§4.6 的 W12 撤回后，全文不作任何相对 attention 的等价或优越主张。
- Limitations 预埋可证伪预测（NT24/CT7 速率上界）：任何声称「长上下文买到更快学习速率」的运行都必须满足 `T·w·log₂V_K` bit 上界。

**未回答的部分（必须承认）**：
- 无 ≥10M token 同算力 LM 主实验；
- 无 Mamba-2 / GLA / DeltaNet / Hyena / Transformer 同场强基线复现（同场参考实现未收敛，Table 只作复现审计，不作排名）；
- 无真实 MQAR 多层、无可学习码本退化曲线。

**Rebuttal 骨架（significance）**：
> 本文的贡献类型是边界刻画 + 构造性反例，不是新 SOTA 架构。若定理成立，它对架构搜索空间的封闭性意义是：在保持静态 LTI 对偶与 O(1) 状态的前提下，精确检索所缺的输入侧信息有可计算下界，且该下界不能被深度或状态阶补偿。我们不在本文中主张语言建模能力，也未提供该类证据；这是明确的 scope，不是遗漏的对比实验。

**不得做**：不为应付评审硬塞不等算力的大规模对比（会被抓「不公平基线」，反而坐实「作者在假装这是架构论文」）；不把「我们很诚实」当 originality 得分点。

## Rebuttal 三段式顺序（技能包 §P2）

1. **Soundness**：只补一个缺失的前提证明或口径澄清，一段话 + 一个表/公式定位。
2. **Originality**：逐条回 Table 1 差异表，引用具体论文与具体增量（用 `iff` / `sharp bound` 这类精确语言，不用 "we study" / "we improve"）。
3. **Significance**：重申搜索空间封闭性；不重复论文已有内容，不新增实验承诺。

## 预埋钩子（Limitations 中已有的可证伪预测）

> 长上下文经验写回权重的信息受 `T·w·log₂V_K` bit 上界约束，与 L 和 bank 容量无关。任何声称长上下文买到更快学习速率的运行都必须满足该界；我们邀请社区检验它。

姿态是「提供靶子」，不是「请求原谅」。
