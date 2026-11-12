# 模块索引（simplcore/core/）

逐模块职责说明。每个模块一段。源码路径均以 `simplcore/core/` 为前缀。

> **命名**：架构叫 **SimplCore**；SGC（`sgc.py`）是 SimplCore 内部的一个 mixer 模块，
> 不是架构的名字。类名一律 `SimplCore*`。

---

## 核心 mixer（全部非注意力）

- **`arch/flux_ledger.py` — FluxLedger 独立研究候选**：自包含 `SimplCoreFluxLanguageModel`，
  不导入或包装现有 SimplCore mixer/model；每层维护固定大小的自适应基线与惊异尺度，
  以当前输入的创新量更新并读取状态。支持流式分块续接，状态尺寸不随序列长度增长。
  目前仅通过单元测试验证因果性、整段/分块一致性和数值稳定性；不是训练完成模型，
  不代表优于当前实现或 Mamba2，也不宣称机制具备文献层面的原创性。
- **`sgc.py`** — SGC mixer。`SpectralGlobalConv`：FFT 因果谱全局卷积 + 阻尼振荡基字典 +
  深度因果卷积 + 内容门控 + L2/L3 闸门；`fft_causal_conv` / `depthwise_causal_conv`
  实现并行/流式两种卷积；`LayerStreamState` / `SGCStreamState` 是 O(1) 流式状态；
  **`forward_chunk(x, state)`** 是第三种形式：块内并行 FFT、块间用 `(y1, y2)` 的
  齐次响应（含 θ→0 重根极限）续接历史，激活 `O(chunk·D)`，供 `--train-chunk`
  的截断 BPTT 使用；`forward_chunk` 与 `forward` 数值等价由 `test_chunked_training` 守护。
  `rho_param="logspace"`（opt-in）把极点改为 `ρ=exp(−softplus(raw))`，可表达 4M
  所需的 `0.99999975`（默认 `sigmoid` 保持旧 checkpoint 逐位兼容）。
  **`RoutedSpectralGlobalConv`（v1.2，`mixer="sgc-r"`）**：谱专家路由变体——M 个基按
  频率分 G 组，逐 token 路由器 top-k 注入（MoE 路由非注意力）；路由门只作用于输入
  注入端，O(1) 流式状态与 `SpectralGlobalConv` 逐字节相同；FFT/IIR 对偶由
  `test_sgc_routed` 按 `test_mixer_math` 同等强度守护，细节见 `SGC_CORE.md` §10。
  `RMSNorm` / `make_norm` 是归一化；`_init_oscillator_grid`（v1.2 默认 logfreq +
  `_assert_no_common_zero` 公共零点守护）/ `_logit` 是振荡基初始化。
- **`jitrl.py`** — JitRL 式测试时优势调制（v1.2，agent 域免训练持续适应）：
  `TrajectoryMemory`（非参数经验轨迹库）+ `AdvantageEstimator`（检索式动作优势）
  + `advantage_logit_bias`/`apply_logit_bias`（加性 logit 调制——KL 约束策略
  优化的闭式更新形式，JitRL arXiv:2601.18510）；`SimplCoreModel.generate(logit_bias_fn=...)`
  是一等接入钩子（默认 None 逐位恒等，测试锁定）。与 Reflexion arXiv:2303.11366
  （提示端）和 L2 证据门（mixer 端）构成"经验约束生成"三位置实例化。
  **无环境实测**（WebArena 判据见论文 §7.4 W4）。
- **`srf.py`** — `SpectralResonanceFlow`：SGC + `SelectiveSpectralSSM`（输入相关 Δ）的
  selective 谱振流混合器（`mixer="srf"`）。
- **`ssm.py`** — `SelectiveSpectralSSM` / `SSMState`：复对角 SSM（复特征值，可表达周期记忆），
  `Δ_t = softplus(W_dt·h_t)` 由输入预测步长；chunked scan 实现（块长 `srf_chunk`）。
- **`selective.py`** — `GatedSpectralConv2`：门控增强多头谱基 mixer
  （GLA/Mamba2 式 γ/β 通道独立双门 + 每 head 独立谱基）；与 SGC 共用闸门数学。
- **`model.py`** — `SimplCoreModel`（顶层 LM，embedding→blocks→ln_f→head + 可选
  abstain_head / memory_in）、`SimplCoreBlock`（mixer→FFN）、`SimplCoreForCausalLM`、
  `SwiGLU`、`_make_block_for_config`（按 mixer 路由）。**`forward_chunk_step`** 跑
  单块并返回 `(logits, CE-sum, n_tok, state)`，训练循环逐块前向+**立即 backward**
  （激活 `O(chunk·D)`，块间状态 detach）；`forward_chunked` 是评估/推理便利接口
  （保留全部 logits，不适合带梯度训练）。多卡 DDP 的 reducer 假定一次 forward
  对应一次 backward，故 `--train-chunk` 在多卡下被 `train.py` 显式拒绝。
- **`config.py`** — `SimplCoreConfig`（全量字段 + 校验 + `from_codec`/`from_json`）与规模预设
  `SIMPLCORE_PRESETS`：`tuan-30m/150m/350m/1b/3b`、`srf-30m`（参数量解析计算）。
  `validate()` 的 mixer 白名单来自 **mixer 注册表**，只放行非注意力 mixer。
- **`registry.py`** — **mixer 注册表（单一事实源）**：`MixerSpec`
  （`build` 构造 / `family` 解析参数家族 / `respects_content_gate` /
  `respects_local_conv` 变体语义）+ `mixer_kinds`（validate 白名单来源）+
  `build_mixer`（model 查表构造）。新增 mixer 只需在此登记一次，
  配置校验、模型构造、参数公式三处自动对齐；
  `purity.assert_registry_pure` 保证注册表内不含注意力实现。

## 抗幻觉与记忆

- **`gates.py`** — 闸门数学**唯一事实来源**：`prefix_mean_energy`、`evidence_weights` /
  `evidence_weight_step`、`uncertainty_weights` / `joint_disagreement`。
  L2 统计量只依赖过去 → 并行 `cumsum` 与流式运行和逐位一致。
- **`grounding.py`** — 逐字归因（LCS 覆盖 + bigram 命中 + **数字逐值核对** + **否定词计数**）、
  `Claim` 字符跨度、`token_labels_from_bindings`、`grounded_label_tensor`。
- **`abstain.py`** — `AbstentionHead`（L4 拒答头）、`GroundedAnswerer`、策略
  （ANSWER/HEDGE/CLARIFY/ABSTAIN 单向降级）、`fit_temperature` / `select_threshold` /
  `expected_calibration_error`。
- **`memory.py`** — CMT 压缩记忆库：固定大小记忆库 + 内容寻址读写（梯度可达）；
  `TextKeyEncoder`（字符 3-gram + 带符号 blake2b + L2 归一化）避免键共线与加盐哈希。
  注入点为残差流加性项（无 KV、无注意力）。
- **`dream.py`** — 自进化反思/合成（证据驱动逐句归因 + 最小编辑修订）；no-regression gate
  （配对 bootstrap 置信下界，失守即回滚权重与记忆）；**`default_consolidator`** ——
  内置的权重巩固器（标签错位一格、只对目标段计损失、样本权重生效、
  不对振荡基参数做 weight decay），让"写回权重"这条链路默认就是通的；
  `ReflectionReport.wrote_weights` / `wrote_memory` 如实报告是否真的写了。
- **`code_exec.py`** — 代码执行沙箱（`LocalPythonSandbox`）+ `pass@k` 验证器
  （`CodeExecutionVerifier`），为 Dreaming 循环提供独立的代码正确性判据。

## 对话与编解码

- **`chat.py`** — `SpecialTokenCodec`（追加式特殊 token）、`ChatTemplate`、`SFTDataset`
  （**错位一格** `labels[1:]`）、贪心打包。
- **`sft.py`** — SFT 训练步（掩码、损失只在查询位置计）。
- **`dialogue_data.py`** — 对话数据校对与 collation（含记忆行/专属行处理）。
- **`dialogue_eval.py`** — 规则化对话评测套件（`stop_rate`/`turn_leak_rate`/`empty_rate`/
  `format_ok` 等，无评审模型）。
- **`dialogue_runtime.py`** — `DialogMemory` / `ChatSession` / `MemoryHit` / `Reply`：
  运行时记忆检索与会话管理。
- **`tokenization.py`** — `Tokenizer` 统一接口与 `make_codec` 工厂（char/byte/bpe）。
- **`tokenizer.py`** — **语料拟合**型分词工具链：`train_bpe`（字节对、GPT-2 字节约定）、
  `train_char` / `train_tokenizer`、`load_tokenizer` / `save_tokenizer`、
  `FittedCharTokenizer`（拟合型字符表 —— 与 `tokenization.py` 的免训练
  `CharTokenizer` **语义不同、名字也不同**）、`compression_ratio`、回合无损校验。

## 规模、外推与互操作

- **`scaling.py`** — `plan_budget()` 反解规模档、嵌入占比 ≤25% 约束；
  长程 FLOPs / KV 字节 / 流式状态三本账（SimplCore 无 KV，长程项 `O(D log L)`）。
- **`extrapolation.py`** — `extrapolation_probe`（固定 L0 训练 + 1/2/4/8× 零微调评测）、
  `decay_ablation`（扫描 `decay_max`）。
- **`numerical_stability.py`** — 原创点 A：含 `ρ/θ/L` 显式依赖的二阶 IIR 误差界
  （`single_basis_error_bound` / `composite_error_bound`）、误差感知基网格
  （`error_aware_grid`：长基稀疏 + 相位抬升，L=2048 误差降 2.2×）、
  由误差预算反解的 `resync_period`。文档 `docs/NUMERICAL_STABILITY.md`。
- **`retrieval_bound.py`** — 原创点 B 的可证伪起点：把"状态容量"与"线性读出寻址"
  拆开算（`capacity_ratio` / `linear_readout_snr`）。1M/30.4MiB 的容量比 ≈6.1×
  （信息论够），瓶颈是固定维线性读出的干扰（SNR≈2.8）。文档 `docs/RETRIEVAL_BOUND.md`。
- **`hf_bridge.py`** — 零第三方 HF 互操作：`migrate_embeddings_and_ffn`（1:1 搬运）、
  `init_simplcore_from_hf`、`simplcore_stream_state_for_eval`（流式状态摘要适配）、
  `sgc_to_hf_eval_bridge`、`MigrationReport`。**源的注意力/位置编码一律不碰。**
- **`purity.py`** — 架构纯洁性守卫，**三层判定**：`assert_pure_config`（建模前挡配置）、
  `assert_attention_free`（建模后遍历 `named_modules()`，并检查 `q_proj/k_proj/v_proj`
  参数与 `forward` 源码里的注意力 API）、`scan_source_attention_free`（AST 源码级：
  类名/参数/`self.x` 赋值/注意力 API 调用，改名绕过无效）。`assert_architecture_pure`
  是三层聚合入口。TS 对照臂（`equal_budget.py` / `throughput_eval.py`）可在文件头用
  `purity: allow-attention-reference` 声明引用豁免，但**定义级检查仍然生效**。
- **`theory.py`** — 理论优势的可执行验证：把 `docs/THEORETICAL_EDGE.md` 的每条
  机制层主张变成断言（34 项），CLI `python -m simplcore.core.eval.claims_audit`；
  由 `tests/test_claims_audit.py` 守护（含"文档点名的模块必须都有检查"的断言）。

## 训练 / 评测 / 数据

- **`train.py`** — 训练循环（bf16/TF32、梯度累积、warmup+cosine、断点续训、JSONL 指标）；
  目标硬件 1× AMD Instinct MI250。
- **`data.py`** — 语料构建、文档级无泄漏切分、打包（修复后不静默丢 token）。
- **`eval.py`** — 探针任务（长程检索、可解性）、流式一致性、显存记账
  （张量记账，非恒 0 的 RSS 探测）、吞吐。

## 高层控制器（加速 / 推理 / 环境学习）

包裹 `SimplCoreModel` 的三层"会做事"的控制器，全部非注意力、复用 O(1) 流式状态、
Dreaming 巩固与不回归门、纯度守卫。三者互不耦合、可单独使用，也可串成
"加速 → 深度推理 → 环境学习"一条链。

- **`accel.py`** — 思考加速：`AdaptiveDepth` 包裹 `SimplCoreModel`，逐 token 按
  "必要性分数"（熵/置信度）决定跑 `min_depth` 还是 `max_depth` 层；
  `skip_threshold=-inf` 时**逐位退化**为恒全算力（须与 `model.stream_step` 一致），
  `ComputeLedger` 只做结构性加速比记账（blocks_budget/blocks_run），不依赖硬件计时。
- **`reasoning.py`** — 深度推理：`IterativeRefiner` 跑"提示 → 多步思考 → 答案"迭代循环；
  `policy` 支持 `fixed_steps`（固定步数）与 `stabilize`（候选答案连续 `patience` 步
  不变即早停）；`tool` 钩子把外部观察（如 `envlearn.LocalCodeEnvironment`）流回状态，
  正是深度推理与环境学习的**接缝**；`ReasoningTrace` 记录结构化轨迹；
  `stabilized()` 为早停判定的纯函数。
- **`envlearn.py`** — 环境学习：`WeightBudget`（1B 基座冻结 + ≤2B 可学，总上限 3B，
  `vram_params()` 只算可学部分即显存优势来源）、`AdapterBlock`/`LearnedAdapter`
  （无状态、无注意力的门控残差适配器）、`ForwardAdapter`（冻结基座 + 只训适配器）、
  `LocalCodeEnvironment`（复用 `code_exec` 沙箱）、`summarize_failures`（失败轨迹 →
  少量训练样本）、`EnvironmentLearner.stage`（交互 → 总结 → 蒸馏 → 环境验证 →
  no-regression gate → 退化回滚，复用 `dream.default_consolidator`）。
- **`simplcore/unity/`（SimplUnity）** — 主干冻结 + 路由 + 多专家的模块化层：
  `UnityModel`（`spawn_expert`/`publish_expert`，预算守卫）、`ExpertRouter`、
  `Expert`；`UnitySelfEvolution(num_workers=N)` 后台并行训练候选专家，
  `register_domain(..., eval_loader, max_regression)` 的 **GATE-3 无回归门**
  拒绝退化候选；`ModuleBundle` 支持专家级导出/跨主干拼接（`base_fingerprint`
  校验 + 预算守卫）；`ExperienceLoader` 把域经验文本转 next-token batch；
  `UNITY_PRESETS` 给 `tuan-2b` 4×~50M 专家的预算口径。三个月路线见
  `docs/SIMPLUNITY_3MONTH_PLAN.md`，入口 `scripts/unity_evolve_2b.py`。
- **`nonlinear.py`** — 非线性思考：`LatentRefiner` 包裹 `SimplCoreModel`，在
  **latent 空间**做可微迭代精化（`LatentRefineCell` 门控非线性残差单元，
  down 零初始化 = 初始恒等），ACT 式 `HaltingHead` 按停止概率自适应收手；
  `max_iters=0`/`enabled=False` 时与 `model.stream_step` **逐位一致**；
  思考不新增流式状态（`stream_bytes` 与原模型相同）；`RefineLedger` 记
  iters_budget/iters_run 与 ponder cost；训练路径 CE + ponder 正则（鼓励早停）。
  与 `reasoning.py` 互补：文本层思考可读可插工具，latent 层思考端到端可微。
- **`smm.py`** — **Simpl Model Memory**：最大自主学习整合架构（接线板）。三层记忆 =
  工作记忆（O(1) 流式状态）→ 情景记忆（`memory.CompressedMemoryBank` 检索注入）→
  语义记忆（`dream.default_consolidator` 蒸馏写权 + no-regression gate 回滚）；
  `experience()` 一次交互落两处（经验池 + 情景记忆），`recall()` 检索聚合，
  `answer()` 带记忆 O(1) 流式生成，`sleep()` 跑一轮 Dreaming 巩固闭环；
  `SMMLedger` 全生命周期记账，`stats()` 出三层记忆全量账目，`state_dict()`
  持久化情景层。`SMMConfig(think_max_iters>0)` 时 `answer()` 走
  `nonlinear.LatentRefiner` 的 latent 思考流式路径（记忆注入与思考在同一
  O(1) 状态上叠加；关闭时与基座逐位一致）。整合件不放松 dream.py 的任何硬约束。

> **编排契约**：`training_integration.ENHANCEMENT_SPECS` 是增强挂载的单一事实源
> （`kind=wrapper|eval|post`、`supersedes`、`consumes`）。未知增强名直接报错；
> 同位置替代写入 `em.superseded` 并由 `train.py` 打印 WARNING；`bio` 消费 `unity`
> 是唯一合法叠加。新增增强只需登记 spec + builder，不再在工厂里堆 if/elif。

> 诚实边界：三者都是**可运行的框架 / 控制流骨架**，不是"已训好的加速/推理/环境学习模型"。
> 机制正确 ≠ 质量达标——未训的玩具模型只会吐无意义的循环文本。与 `theory.py` 的诚实边界一致。

## 战役与运行时契约（1B→3B）

- **`backend_contract.py`** — 自研 CUDA/C++ 后端算子契约（**单一事实源**）：16 个必需内核
  （mixer 三件套 / 四层闸门 / SwiGLU / AdamW 冻结掩码 / O(1) 流式状态管理 / 2 GCD 切分）、
  精度策略（bf16 计算 + fp32 累加与优化器状态）、三条逐位一致性
  （FFT↔IIR `<4e-5`、分块携带、回滚 byte-exact）、禁止项（注意力家族 / KV cache / 位置表）。
  `conformance_checks()` 的 8 项硬检查被 `campaign.build_plan` 的 Phase 0 自动继承。
  运行时 = 自研后端；PyTorch 退役出战役路径，torch 参考实现保留为冻结数值 oracle。
- **`conformance_runner.py`** — 后端一致性**跑批器**（契约的可执行验收，Phase 0 对接脚本）：
  后端把 16 内核包装成 `BackendAdapter` 子类，`--adapter 包.模块:类名` 一条命令执行
  9 项检查（8 项 conformance + C-rollback）——算子完备性冒烟、并行(FFT) vs 流式(IIR)
  完整 mixer `max|Δ|<4e-5`、分块 vs 整段扫描**逐位**、冻结位 p/m/v 逐位不动 +
  可训位对照手工 AdamW、bf16 输入/fp32 累加 vs fp64（相对 ≤1e-2，K=4096 下
  纯 bf16 累加 ~9% 混不过去）、2 分片 all-reduce 逐位、注册表禁止名审查、
  状态 serialize 往返逐位、快照→回滚→续跑逐位。`TorchReferenceAdapter` 是
  参考数学的适配器化（oracle 自检用）；**每个检查都有破坏性变异的负向测试**
  证明它真能抓住对应缺陷。CLI `python -m simplcore.core.conformance_runner`。
- **`adapter_model.py`** — **纯算子重组模型**（16 算子契约的完备性证明）：
  `AdapterModel` 用且仅用 `BackendAdapter` 内核搭出完整 SimplCore
  （嵌入/归一化/gemm/swiglu/残差/内容门/长程并行与流式/局部卷积/证据与
  不确定门/CE），`load_native` 直接吃 `SimplCoreModel.state_dict()`；
  守护测试证明与原生模型**三重等价**（并行前向、流式前向 vs 逐 token
  `stream_step`、梯度）+ **契约覆盖审计**（`RecordingAdapter` 记录每次
  内核调用，前向算子全覆盖且无契约外内核）。契约若缺内核，在这里暴露，
  而不是后端集成期。
- **`trainer.py`** — **Phase 1/2 训练驱动**（驱动与内核分离）：分块流式
  训练（截断 BPTT，梯度窗口=chunk，显存 O(chunk) 与 L 无关——战役分块
  路径）+ `ContextCurriculum` 上下文课程（按 token 预算推进阶梯）+
  优化器走 `adamw_frozen_mask`/`weight_decay_exempt`（冻结位 p/m/v 逐位
  不动）+ `maybe_commit` no-regression gate（更优才提交，退化超容差
  逐位回滚，快照经 `stream_state_serialize` byte-exact）+ 训练账本。
  今天跑在 torch 参考适配器上（tiny 实测 loss 3.50→0.26、课程 16→32→64
  自动升档）；后端过 conformance 后**换适配器即换运行时，驱动零改动**。
  CLI `python -m simplcore.core.trainer`（tiny 演示实跑）。
- **`equal_budget.py`** — **等预算对照 harness（验收协议 X-equal-budget 的可执行化）**：
  SimplCore vs 标准 Transformer 或 Hugging Face Mamba2（对照臂均自包含于本模块，
  **不进运行时/注册表**，架构纯洁性不受影响）。训练和验证严格使用不同 split；
  synthetic smoke 用独立固定 seed 数据，不能授权质量结论。两臂训练消费**同一确定性
  窗口流**（`WindowStream`，同 seed 逐位相同 → 等 token 由构造满足）；Transformer
  臂 FLOPs 账独立逐项推导
  （`6·N_nonemb + 12·s·D·L`，attention 项单列），与 SC 的 `scaling.train_flops_per_token`
  同口径可除；两类基线参数量均按非 embedding 参数对齐（≤5% 才通过门槛）；
  held-out **逐窗配对** + bootstrap CI + 最差窗口入报告（禁止 average-only）。
  Mamba2 FLOPs 账不伪造，当前只支持等 token；Hugging Face causal-LM 自动 shift labels
  不用于训练 loss，避免与 SimplCore 标签语义错一位。`--seeds 11 22 33` 跑至少三组
  固定数据的重复实验，按 seed 聚合不确定性并校验每次运行的 train/val 内容指纹一致；
  冻结语料每个 shard 必须提供 SHA-256，train/val 共享 shard 会被拒绝。
  单 seed 或合成语料不会给出无条件质量结论。
  CLI `simplcore-equal-budget`
  （产出 `runs/eval_equal_token.json`，即 ShowcaseItem S4 的复验脚本）。
- **`longctx_eval.py`** — **长上下文检索评测（X-longctx-retrieval + X-stream-consistency
  的可执行化，ShowcaseItem S5/S7 脚本）**：确定性 needle 语料（K 个 needle 不同
  深度防"只记最后一个"、key→value 映射 seed 绑定全语料一致、argmax==value 机器
  可判无自报自评）；**流式==整段是闸门不是注释**——同一 trial 并行与 O(1) 流式
  各跑一遍，argmax 逐 trial 相同且 logit 差 < 4e-5 才算过；**长 L 的账先于跑**——
  TS KV（`2·L·layers·D·2B`）超过 `max_device_mem` 就如实标记不可跑、账面照报、
  闸门不参与判定（1B 形态 1M：SC 状态 30.36 MiB vs TS KV 120 GiB，2M/4M 240/480 GiB
  单卡不可行）；`--ledger-only` 一条命令出 1M/2M/4M 可行性账 + S7 信息论边界
  （needle 原文字节 vs O(1) 状态字节 vs TS KV 三行并列）；near-chance 警告——
  未训练模型的"≥基线"PASS 自动附带不构成能力证据的标注并关闭宣称许可。
  CLI `python -m simplcore.core.longctx_eval --lengths 1048576,2097152,4194304 --ledger-only`。
- **`gate2.py`** — **GATE-2 needle 召回测量门（原创清单 P3）**：课程首次进入
  1M 时用 `score_trial(mode="chunked")`（`forward_chunked`）跑确定性 needle 语料，
  不物化整段 `(L,D)` 核/激活；按机会水平分档（`below_chance`/`at_chance`/`pass`），
  同时报逐深度最差桶。训练默认只记录（`--gate2`），`--gate2-strict` 才在
  `below_chance` 硬停；4M 验收走
  `python -m simplcore.core.gate2 --checkpoint ... --lengths 1048576,4000000`。
  文档 `docs/GATE2_NEEDLE.md`。
- **`code_passk_eval.py`** — **代码 pass@k 对照（X-code-passk 的可执行化，S6 脚本）**：
  实现 Codex 论文的**无偏估计量** `1 - C(n-c,k)/C(n,k)`（对数域稳定，与精确组合版
  互证——`code_exec` 的 `pass_rate` 是"每样本平均分"，与 pass@k 不是同一口径，
  报告里两者分开标注互不冒充）；判分只认子进程**真实执行**（异常/超时/断言失败
  一律不计过，自报自评在结构上不可能）；两臂**同一套 8 题确定性 held-out 题**
  逐题对齐（复用 `ab.py`：缺题显式报错、题级 0/1 走 McNemar 精确检验、题级连续
  pass@k 走配对 bootstrap CI）；两臂只通过 `(prompt)->List[str]` generator 接触，
  同 k 同 timeout（等预算口径的推理侧对应物）；`--demo` 固定候选池跑通协议
  （demo=True 进报告、宣称许可关闭）。CLI `python -m simplcore.core.code_passk_eval --demo`。
- **`throughput_eval.py`** — **解码吞吐对照（X-throughput 的可执行化，S3 脚本）**：
  TS 臂是**真 KV-cache 增量解码器**（`TSDecoder`：prefill 建 cache、每步只算新
  query——守护测试钉死 prefill/step logits 与整段前向一致，慢慢在真实算法上，
  不是稻草人）；计时报 p10/p50/p90 + min/max 分布（禁 average-only）；**闸门只判
  可测部分**——SC 延迟平坦性用 p50(L_max)/p50(L_min) ≤ 1.5 的比值判据（对 CPU
  计时噪声稳健，最小二乘斜率被噪声证明脆弱、已弃用），1M 吞吐倍数由**解析账
  分列支撑**（KV 120 GiB vs 状态 30 MiB，flops 账注明 TS 每步为读全量 KV 受
  内存带宽约束、flops 是下界）；放不下的 L 如实标记不可跑、不跑假对照。
  CLI `python -m simplcore.core.throughput_eval --lengths 128,512,2048`。
- **`phase1_train.py`** — **Phase 1 训练执行器**（数据闸门 P0-budget-frozen 的数据部分）：
  packed 语料（mmap 分片 + manifest + sha256 指纹）→ 课程窗口流 →
  `StreamingTrainer` → 固定口径评估/no-regression gate → 报告。无泄漏是结构性的
  （train/val 分片交集强校验）；token 游标跨课程档连续消费（换档重切不回头，
  epoch 计数入报告）；`claim_permitted` 在合成语料下强制 False。
  CLI `python -m simplcore.core.phase1_train`（合成演示；`--corpus` 指向真实 packed 语料）。
- **`backend_bridge.py`** — **后端 C-ABI 桥接层**：16 算子 C ABI v1 规范
  （符号 `sc_*`、`(ptr, shape, rank)` 缓冲约定、错误码、NULL 可空张量）+
  `CtypesBackendAdapter`（加载 `.so`/`.dll`、符号完备性校验、contiguity/dtype
  编组、悬指针保活）+ 宿主侧算子原生实现（序列化与参考 byte 兼容）。
  真实 CUDA 后端到位后 `load_backend(path)` drop-in，先过 conformance 再进训练。
  测试用 `FakeCLib` 经**真实 ctypes 指针**读写缓冲验证编组（数学委托参考适配器），
  含 AdapterModel 全链路 parity。
- **`paper_audit.py`** — **论文/战役账目交叉审计**：把 SimplCore_Paper.md 与
  campaign 里的数字当"引用"、代码当"事实源"，逐项重算对照（部署账/流式状态/
  FLOPs/KV/Chinchilla/硬停日历/证明数/守卫数/测试数/CLI 数），任一不符非零码退出。
  曾当场抓出 §7 "TS KV ≈60 GB" 只计了 K/V 一半（单一事实源 120 GiB）。
  **状态字节断言（S01–S05）必须声明配置**（preset 或 M/D/layers/K/batch），
  按量级分档：相对偏差 ≤0.1% PASS、≤1% WARN、>1% FAIL、≥10× 或未声明配置硬失败；
  文档里出现未登记的"流式/常驻状态 X 单位"数字同样硬失败（防止不同配置的
  状态字节被并列引用后再次造成量级误读）。
  CLI `python -m simplcore.core.paper_audit`（产出 runs/paper_audit.json）。
- **`deploy_budget.py`** — 部署侧 HBM 账本：**1M 上下文完整放下 ≤ 2 GiB、翻倍增量 ≤ 1 GiB
  （结构性为 0）**。四项占用（bf16 权重 1.888 GiB + O(1) 流式状态 **30.36 MiB**
  （tuan-1b，batch=1，fp32，15 层）+
  流式 prefill 单 chunk workspace + logits）全部与总上下文 L 无关——
  1M/2M/4M/8M 占用逐字节相同。CLI `python -m simplcore.core.deploy_budget`。
- **`mathproof.py`** — 数学证明模块（symbolic + fp64，无硬件依赖）：13 项证明把架构的
  数学主张变成可机器验证的命题 —— 闭式阻尼振荡核 ↔ 二阶 IIR 递推（sympy 符号 + fp64）、
  伴随矩阵幂/谱半径、**分块携带律与叠加原理**（`chunked_iir_scan` 的合同数学）、
  FFT N=2L 无混叠、BIBO 稳定性与核 ℓ₁ 界（含 fp32 sigmoid 饱和的 ρ 卫兵验证）、
  mixer 级流式==并行（fp64）、证据门恒等式、闸门有界性（三角不等式）、
  时移协变性（门关）/故意非协变（门开）、O(1) 状态**充分性**（马尔可夫性）、
  复杂度公式逐项重推导审计、HBM 账本代数、冻结基座梯度隔离。
  CLI `python -m simplcore.core.mathproof`（13/13 通过才算绿）。
- **`campaign.py`** — 1B→3B 战役计划（纯解析账本，不建模型）：HBM 四路径账本
  （优化器 16 B/参数、O(1) 流式、分块流式训练、朴素整段 FFT 对照）；
  "上下文翻倍增量 ≤ 1 GiB"主张裁决（可行路径增量恒 0，朴素路径自 512K 起爆）；
  Phase 0 冻结 → Phase 1（1B 基座，≤30 天硬停，~26.6B tokens @25% MFU）→
  立即复盘（失败轨迹→训练样本）→ Phase 2（≤2B 适配器，≤28 天硬停，~10.2B tokens）；
  最终验收协议 = 与 3B Transformer 的**真实输出**对照（等 token/FLOPs 口径），
  显式禁止注意力层面机制内对比、自报自评、cherry-picking。

## 已删除的模块（历史归档）

以下模块曾存在于产品线，现已**整体移除**（源码与测试见 `.trash/attn_removed/`）：

| 已删除 | 原用途 |
|--------|--------|
| `CausalAttention`（`model.py` 内） | 标准因果 MHA + RoPE 对照基线 |
| `reason.py` / `ReasoningAttention` | 残差流门控因果注意力（固定环形 KV） |
| `hybrid.py` / `HybridBlock` | SGC 全局层与滑窗注意力交替（LongNet/Jamba 路线） |
| `attention.py` / `attention_yarn.py` | GQA / GQA+YaRN 滑窗注意力 |
| `_rope.py` / `rope_scales.py` | RoPE 助手与 none/linear/yarn 外推缩放 |

早期路线原型（自研张量引擎、attention_res 对照层、共振场、数值引擎第二版）
整体归档在 `research/`，**不发布、不参与 CI**。

## 测试（`simplcore/core/tests/`）

守护测试，分组覆盖：数学/双形式/因果/梯度（`test_mixer_math`）、
闸门（`test_gates`、`test_gates_selective`）、流式与断点（`test_stream_checkpoint`）、
无词表（`test_tokenization*`）、HF 桥（`test_hf_bridge`）、记忆与 Dreaming
（`test_dream`、`test_dream_evolution`、`test_dream_code_integration`）、
SRFN（`test_srf`）、规模账本（`test_scaling`）、外推（`test_extrapolation`）、
架构纯洁性（`test_architecture_purity`）、理论守卫（`test_claims_audit`）、
权重巩固（`test_consolidation`）、可复现（`test_reproducibility`）、
对话（`test_chat`、`test_dialogue_*`）、代码沙箱（`test_code_exec`）、
思考加速（`test_accel`）、深度推理（`test_reasoning`）、环境学习（`test_envlearn`）、
增强编排契约（`test_enhancement_registry`：未知名报错/替代关系显式化/bio 消费 unity）、
主训练窗口预算（`test_train_window_budget`：4M×2048=30.5 GiB 必须被闸门拒绝）、
分块训练内核（`test_chunked_training`：chunked ≡ 整段 fp32 等价、chunk=L 梯度逐位、
重根极限支、状态定长）、
数值稳定性（`test_numerical_stability`：界不低估、重同步周期自洽、误差网格 2.2×）、
长序列误差（`test_long_sequence_error`：`<4e-5` 不外推到 L=2048）、
检索容量账（`test_retrieval_bound`：容量与寻址分离）、
SimplUnity 模块化（`test_simplunity_modular`：bundle 跨主干拼接/指纹拒绝、
多 worker 后台训练 + GATE-3、经验 loader、2B Unity 预算）、
GATE-2（`test_gate2`：分档判定、逐深度最差桶、chunked 评分与 parallel 一致）、
非 Python 实现与记忆分层隔离（`test_impl_isolation`）、
等预算对照（`test_equal_budget`：FLOPs 账逐项、TS 因果性、参数对齐、窗口流
确定性、两臂 token 恒等、配对统计自洽、诚实性字段、CLI schema）。
