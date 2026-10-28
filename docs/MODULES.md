# 模块索引（simplcore/core/）

逐模块职责说明。每个模块一段。源码路径均以 `simplcore/core/` 为前缀。

> **命名**：架构叫 **SimplCore**；SGC（`sgc.py`）是 SimplCore 内部的一个 mixer 模块，
> 不是架构的名字。类名一律 `SimplCore*`。

---

## 核心 mixer（全部非注意力）

- **`sgc.py`** — SGC mixer。`SpectralGlobalConv`：FFT 因果谱全局卷积 + 阻尼振荡基字典 +
  深度因果卷积 + 内容门控 + L2/L3 闸门；`fft_causal_conv` / `depthwise_causal_conv`
  实现并行/流式两种卷积；`LayerStreamState` / `SGCStreamState` 是 O(1) 流式状态；
  `RMSNorm` / `make_norm` 是归一化；`_init_oscillator_grid` / `_logit` 是振荡基初始化。
- **`srf.py`** — `SpectralResonanceFlow`：SGC + `SelectiveSpectralSSM`（输入相关 Δ）的
  selective 谱振流混合器（`mixer="srf"`）。
- **`ssm.py`** — `SelectiveSpectralSSM` / `SSMState`：复对角 SSM（复特征值，可表达周期记忆），
  `Δ_t = softplus(W_dt·h_t)` 由输入预测步长；chunked scan 实现（块长 `srf_chunk`）。
- **`selective.py`** — `GatedSpectralConv2`：门控增强多头谱基 mixer
  （GLA/Mamba2 式 γ/β 通道独立双门 + 每 head 独立谱基）；与 SGC 共用闸门数学。
- **`model.py`** — `SimplCoreModel`（顶层 LM，embedding→blocks→ln_f→head + 可选
  abstain_head / memory_in）、`SimplCoreBlock`（mixer→FFN）、`SimplCoreForCausalLM`、
  `SwiGLU`、`_make_block_for_config`（按 mixer 路由）。
- **`config.py`** — `SimplCoreConfig`（全量字段 + 校验 + `from_codec`/`from_json`）与规模预设
  `SIMPLCORE_PRESETS`：`sgc-30m/150m/350m/1b/3b`、`srf-30m`（参数量解析计算）。
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
  `DreamReport.wrote_weights` / `wrote_memory` 如实报告是否真的写了。
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
- **`hf_bridge.py`** — 零第三方 HF 互操作：`migrate_embeddings_and_ffn`（1:1 搬运）、
  `init_simplcore_from_hf`、`simplcore_stream_state_for_eval`（流式状态摘要适配）、
  `sgc_to_hf_eval_bridge`、`MigrationReport`。**源的注意力/位置编码一律不碰。**
- **`purity.py`** — 架构纯洁性守卫：`assert_pure_config`（建模前）、
  `assert_attention_free`（建模后遍历 `named_modules()`）、`attention_like_modules`。
- **`theory.py`** — 理论优势的可执行验证：把 `docs/THEORETICAL_EDGE.md` 的每条
  机制层主张变成断言（25 项），CLI `python -m simplcore.core.theory`；
  由 `tests/test_theory.py` 守护（含"文档点名的模块必须都有检查"的断言）。

## 训练 / 评测 / 数据

- **`train.py`** — 训练循环（bf16/TF32、梯度累积、warmup+cosine、断点续训、JSONL 指标）；
  目标硬件 1× AMD Instinct MI250。
- **`data.py`** — 语料构建、文档级无泄漏切分、打包（修复后不静默丢 token）。
- **`eval.py`** — 探针任务（长程检索、可解性）、流式一致性、显存记账
  （张量记账，非恒 0 的 RSS 探测）、吞吐。

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
架构纯洁性（`test_architecture_purity`）、理论守卫（`test_theory`）、
权重巩固（`test_consolidation`）、可复现（`test_reproducibility`）、
对话（`test_chat`、`test_dialogue_*`）、代码沙箱（`test_code_exec`）。
