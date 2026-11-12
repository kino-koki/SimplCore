# SimplCore 总体架构设计

> ⚠️ **实验阶段**：本架构（SimplCore-0.1-α）仅处于实验阶段，未经大规模训练验证；
> 若非确实想尝试非注意力方案，建议直接使用 Mamba 等成熟架构进行开发与模型设计。

本文档描述 `simplcore/core/` 主线架构的设计、模块划分与数据/控制流。所有结论以源码为
准（`arch/sgc.py` / `model/model.py` / `model/config.py` / `arch/srf.py` / `arch/ssm.py` /
`arch/selective.py` / `arch/gates.py` 等），无"文字创新"。

> **包结构（0.1-α 结构化拆分，2026-10）**：`core/` 曾是 60+ 文件的平铺包，现已按
> 依赖方向拆为八个子包（依赖只能自上而下，个别受控跨界见下）：
>
> | 子包 | 内容 | 依赖 |
> |------|------|------|
> | `core/arch/` | mixer 数学核心：sgc / srf / ssm / selective / gates / registry / purity / accel / numerical_stability / abstain / grounding | （零 core 内依赖） |
> | `core/model/` | 配置与模型：config / model / adapter_model / export / hf_bridge / nonlinear / tokenization / tokenizer / **multimodal（图像 patchify + 音频帧下采样 → 软 token 拼接，零模态时与纯文本逐位一致；流式走 stream_step 的 inputs_embeds 入口）** / **physics_world（物理世界模型：实验功能 opt-in 默认关，tuan-2b 等全部预设不使用；物理一致性判据与 L2/L3 闸门正交互补）** | → arch |
> | `core/memory/` | 记忆与自进化：memory / dream/（子包）/ code_exec / smm / **sidecar（三层精确引用：L1 旁路档案 + L2 指针取回，O(1) 寻址器 + O(N) 精确档案在 Host RAM）** / sidecar_exact_attn（局部精确注意力实验件，purity 定义级白名单豁免） | → model |
> | `core/dialogue/` | 对话域：chat / dialogue_data / dialogue_runtime / dialogue_eval / reasoning | → memory, model |
> | `core/training/` | 训练域：data/（子包）/ train/（子包）/ trainer / phase1_train / training_integration / sft / ab / seqlen_curriculum / gate2 / envlearn | → model, memory, eval* |
> | `core/eval/` | 评测与账本：eval / probes / bench / memprobe / scaling / equal_budget / longctx_eval / throughput_eval / extrapolation / code_passk_eval / claims_audit / mathproof / paper_audit / retrieval_bound / **world_probe（O(1) 状态的未来信息线性可读出探针：fp64 闭式岭回归 + 置换/多数类基线，测量而非宣称）** | → 各层 |
> | `core/backend/` | 后端契约：backend_contract / backend_bridge / conformance_runner | → arch |
> | `core/deploy/` | 战役与部署：campaign / deploy_budget / sprint30 | → backend, model, training, eval |
>
> 实验性模块（jitrl / pilot / conditions_ledger）隔离在 `simplcore/lab/`，
> **core 不得 import lab**（`test_impl_isolation.py` 守护）。
> 受控跨界（模块级无环，仅包级互引）：`training/gate2.py → eval/longctx_eval`、
> `eval/equal_budget.py → training/data`、`eval/paper_audit·mathproof → deploy/campaign`。
> 旧平铺路径（``simplcore.core.sgc`` 等）由 ``_compat_aliases.py`` 的 sys.modules
> 别名继续兼容；同名子包（model/memory/eval）的 ``__init__`` 另有全量 re-export。
> 新代码一律使用新路径。

> **命名约定**：架构叫 **SimplCore**；`SGC` 只是 SimplCore 内部的一个 mixer 模块
> （`simplcore/core/arch/sgc.py`，`SpectralGlobalConv`）。类名一律 `SimplCore*`
> （`SimplCoreConfig` / `SimplCoreModel` / `SimplCoreBlock` / `SimplCoreForCausalLM`）。

---

## 1. 设计原则

1. **彻底非注意力**。产品线内**不含任何** softmax 自注意力模块 ——
   `CausalAttention` / `ReasoningAttention` / `SlidingWindowAttention` /
   GQA-YaRN 注意力 / RoPE 均已删除，历史归档在 `.trash/attn_removed/`，
   早期路线原型归档在 `research/`。
   `SimplCoreConfig.validate()` 只接受非注意力 mixer，`purity.py` 提供
   配置 + 实例双重断言，把这条原则变成 CI 硬约束。
2. **因果性是第一公民**。任何全局算子都必须通过"扰动末位 token → 前文逐位不变"
   的守护测试，否则视为缺陷（历史上 RFP 非因果、mix 三处非因果均因此被揪出并修复）。
3. **训练/推理双形式严格等价**。并行（FFT）与流式（IIR）必须数值一致
   （fp32 `max|Δ|<4e-5`），流式状态与序列长度无关。
4. **无位置嵌入表、无 `max_len`**。因果卷积本身编码位置，核由闭式/递推定义于任意 τ，
   因此训练长度之外的长度天然有定义。
5. **结构保证优于性能宣称**。参数量 `bit-exact` 解析、流式==整段逐位、因果性、
   外推表有界 —— 这些是可证明的结构性质；困惑度收益/质量曲线必须实测，
   未测前不写进正文。

理论优势的逐模块论证见 **`docs/THEORETICAL_EDGE.md`**。

---

## 2. 顶层数据流

```
tok_emb(idx) ──► drop
   └─► for each SimplCoreBlock:
          x = mixer(x)                # SGC / SRFN / GatedSpectralConv2（非注意力）
          x = ffn(x)                  # SwiGLU + 残差
       (可选) 记忆注入：x = x + memory_in(retrieved_memory)
   └─► ln_f(x) ──► head(x) ──► logits
   (可选) abstain_head(x) ──► 拒答 logit
```

- `SimplCoreModel` 持有 `tok_emb`、`blocks`（ModuleList）、`ln_f`、`head`
  （可 tied 到 `tok_emb`）、可选 `abstain_head`、可选 `memory_in`。
- 记忆注入点是**残差流**：SimplCore 没有 KV 缓存，残差流是它原生的上下文注入位置；
  若传了 `memory` 但未开 `use_memory`，会**当场报错**而不是静默丢弃。

---

## 3. Mixer 分类（同一套 config 切换，全部非注意力）

| `mixer=` | 模块 | 全局交互 | 局部交互 | 注意力 | 流式状态 |
|----------|------|----------|----------|--------|----------|
| `sgc`（默认） | `SpectralGlobalConv` | FFT 谱全局卷积（阻尼振荡基） | 深度因果卷积 | 无 | O(1) IIR |
| `sgc-nogate` / `sgc-nolocal` | 同上（开关） | 同上 | 关门控 / 关局部 | 无 | O(1) |
| `sgc-r` | `RoutedSpectralGlobalConv` | SGC + 谱专家路由（M 基分 G 组，逐 token top-k 注入；MoE 路由非注意力，v1.2） | 深度因果卷积 | 无 | O(1)（与 sgc 逐字节同） |
| `srf` | `SpectralResonanceFlow` | SGC + `SelectiveSpectralSSM`（输入相关 Δ） | 深度因果卷积 | 无 | O(1) chunked scan |
| `snn-sgc`（实验，显式 opt-in） | `SpikingSpectralGlobalConv` | SGC + 有符号 LIF 三值脉冲读出（surrogate-gradient） | SGC 深度因果卷积 | 无 | O(1)，每层额外 `D` 个 fp32 膜电位 |

**已删除的取值**：`attn` / `hybrid*` / `reason_attn_every` —— 注意力族整体移除，
配置层直接拒绝（`ValueError`）。历史实现归档在 `.trash/attn_removed/`，
早期路线原型（含注意力对照层 `attn_res.py`）归档在 `research/architecture/`。

**GatedSpectralConv2**（`selective.py`，`gate_ab=True`）在 SGC 骨架上叠加
GLA/Mamba2 式通道独立双门 `γ`（时间常数内容化）/ `β`（写入门），
与 SGC 共用 `gates.py` 的四层闸门数学。

`snn-sgc` 是基于共享模型接口的实验 mixer，不是默认主干切换，也不改变现有
`sgc` checkpoint/配置的默认行为。SGC 连续输出保持为主残差，脉冲分支以低增益
作为可训练修正项；训练时使用直通 surrogate 梯度，流式状态另存每层一个 fp32
膜电位向量。当前实现仍执行稠密 SGC 与张量算子，不代表稀疏硬件加速或能耗优势。
它可与模型的常规记忆注入和 SMM 共用模型 API；`gate_ab=True` 与 `snn-sgc` 暂不
兼容，配置会显式拒绝。

---

## 4. SGC 与 SRFN（核心，无注意力）

详见 `SGC_CORE.md`。要点：

- **核 = 阻尼振荡基字典**：`K[τ,d] = w_d·1{τ=0} + Σ_m C[m,d]·ρ_m^τ·sin((τ+1)·θ_m)`，
  `ρ_m,θ_m` 可学，`M` 默认 64（1.3B/3B 为 128），无 `max_len`。
- **双形式**：训练用 FFT 卷积 `O(L log L)`；推理用二阶 IIR 递推 `O(1)` 状态。
- **内容门控**（selective）：有效核随内容变化，不引入 Q/K/V softmax。
- **局部路径**：宽度 `local_kernel`（默认 4）的深度因果卷积，补尖锐局部结构。
- **四层闸门**（L1 选择门 / L2 证据门 / L3 不确定门 / L4 拒答头）降低幻觉倾向。

SRFN（`mixer="srf"`）在 SGC 之上叠加 `SelectiveSpectralSSM`：复对角 SSM
（复特征值 `λ_m = ρ_m e^{iθ_m}`，可表达**周期记忆**），
步长 `Δ_t = softplus(W_dt·h_t)` 由输入预测，使"这个位置该记久一点"可由模型决定。
时变核用 **chunk 化下三角 Toeplitz 矩阵**批量求值（一个 chunk 一次矩阵乘，
而非 C 次 Python 步）。CPU 上 `srf` 比 `sgc` 慢约 2–2.8×，但流式状态仍是 O(1)。

---

## 5. 流式推理（O(1) 状态）

`SimplCoreModel.init_stream → stream_step → head` 单 token 推进：

- SGC/SSM：`LayerStreamState` 持有二阶 IIR 状态 `y1,y2 (B,M,D)` + 局部卷积缓冲
  + L2 证据门前缀能量和 `energy_sum`；**与 L 无关**。
- `stream_bytes(batch)` 精确核算常驻字节数；`checkpoint_stream/restore_stream`
  序列化为 plain dict（含 `energy_sum`），可 `torch.save` 持久化
  "断点流"，restore 后继续 `stream_step` 逐位一致。
- `supports_streaming` 属性：仅当所有 mixer 都有 `init_state/step`（或 `init_stream/
  stream_step`）时为 `True`。**所有** SimplCore mixer 均为 `True`
  —— 这与 Transformer 的"注意力基线没有 O(1) 状态"形成结构性差异。

---

## 6. 记忆（CMT）与自进化

- `memory.py`：CMT 式压缩记忆库 —— 固定大小记忆库 + 内容寻址读写，梯度可达；
  记忆检索后投影为 `memory_in`（零初始化 → 初始恒等）注入残差流。
  CMT 原文在注意力层注入 KV，SimplCore 没有注意力与 KV cache，
  因此注入点改为**残差流上的加性记忆项** —— 这是本架构原生的等价物。
- `dream.py`：自进化反思/合成（证据驱动逐句归因 + 最小编辑修订）；默认配置会
  **拒绝写回权重**以防退化的"幻觉式自进化"，这是有意的安全默认。
- `code_exec.py`：代码执行沙箱 + `pass@k` 验证器 ——
  给 Dreaming 循环提供**独立的**代码正确性判据（没有它，`mode="weights"`
  必然 ROLLBACK，因为"让 gate 恒通过"等于没有 gate）。
- `dream.py::default_consolidator`：**内置的权重巩固器** —— 把训练信号变成梯度步
  （标签错位一格、只对目标段计损失、样本权重生效、**不对振荡基参数做 weight decay**）。
  没有它时 `mode="weights"` 需要调用方自备 consolidator，链路默认是断的；
  `ReflectionReport.wrote_weights` / `wrote_memory` 如实报告"到底写了没有"。
- 诚实边界：记忆键是字面 n-gram 而非语义嵌入；"自进化会变强"**未证明**；
  代码沙箱是**进程级**，不是安全边界。
- 目标函数层面的理论动机（为什么 O(1) 状态与世界模型互补、"越用越好"为何
  仍属未证明）见论文 §3.5——本节只陈述机制，动机与边界以论文为准。

---

## 7. 抗幻觉四层闸门

| 层 | 名称 | 拦截失败模式 | 实现 |
|----|------|--------------|------|
| L1 | 选择门 selective | 该参与/不该参与的乱参与 | mixer 内容门控 |
| L2 | 证据门 evidence | 依据不存在的远端上下文编造 | `gates.py` 因果前缀能量统计 |
| L3 | 不确定门 uncertainty | 高熵区过度自信 | `gates.py` 长/短路径分歧 |
| L4 | 拒答门 abstention | 无依据时强行作答 | `abstain.py` grounded 头 + 策略 |

- `gates.py` 是闸门数学**唯一事实来源**，SGC 与 `selective.py` 共用，
  避免"换 mixer 分支抗幻觉静默消失"。
- L2 的统计量只依赖过去：并行路径用 `cumsum`，流式路径用常驻标量运行和，
  **两者逐位一致** → 闸门不会在流式与整段推理之间漂移。
- `grounding.py` 提供逐字归因（LCS 覆盖 + bigram 命中 + **数字逐值核对** +
  **否定词计数**），`abstain.py` 做温度校准 / 阈值选择 / ECE 报告。
- 诚实边界：grounding 判**文本可归因性**不判语义蕴含；L4 头需外部证据标签才能训练。

---

## 8. 编解码（无词表模式）

两个模块，职责**不重叠**：

* **`tokenization.py`** —— **免训练**的 codec 接口层：`Tokenizer` / `CharTokenizer`
  （固定字符集）/ `ByteTokenizer` / `BPETokenizer` / `make_codec`（纯 Python 零第三方）。
  模型、对话、训练链路用的是这一套；`SimplCoreConfig.from_codec` 自动填 `vocab_size`，
  同一套权重可切换字符/字节/BPE 序列，不重新初始化词表参数。
* **`tokenizer.py`** —— **语料拟合**的训练工具链：`train_bpe` / `train_char` /
  `train_tokenizer` / `load_tokenizer` / `save_tokenizer`，以及拟合型字符表
  `FittedCharTokenizer`。

> 历史上两个模块都导出过 `CharTokenizer` 但语义不同，已把拟合型改名为
> `FittedCharTokenizer` —— 否则 `from simplcore.core import CharTokenizer` 与
> `from simplcore.core.tokenizer import CharTokenizer` 会拿到不同的东西。

---

## 9. 互操作（HF 桥）

`hf_bridge.py`（零第三方依赖）：**保守 1:1 搬运** embedding / FFN / norms / head，
形状错配一律 skip 并记录，不污染权重。源的注意力 QKV / 位置编码**一律不碰**
（SimplCore 没有对应接收端，强行投影只会污染谱全局卷积核）。
`sgc_to_hf_eval_bridge` 提供 HF 评测路径，透传 O(1) 状态字节数。
迁移后 O(L log L) / O(1) 流式 / 因果性**全部原样继承**。

---

## 10. 训练 / 评测 / 诚实纪律

- `train.py` / `data.py`：bf16/TF32、梯度累积、warmup+cosine、断点续训、JSONL 指标、
  文档级无泄漏切分。目标硬件 1× AMD Instinct MI250（128 GB HBM2e）。
  **训练窗口预算**：主路径有两种前向模式。
  * **整段并行**（默认，`--train-chunk 0`）：每个 block 物化 `(L, D)` fp32 核
    （`sgc.build_kernel`），训练显存 `O(L·D)`；`--max-kernel-gib`（默认 4 GiB）
    在启动时按课程**最大长度**预检并显式拒绝超预算窗口（4M×2048 = 30.5 GiB）。
  * **分块流式**（`--train-chunk N`）：`SpectralGlobalConv.forward_chunk` 用
    上一块末状态 `(y1, y2)` 的**齐次响应**续接历史，块内仍是并行 FFT，激活
    `O(chunk·D)`；训练循环逐块调用 `SimplCoreModel.forward_chunk_step` 并**立即
    backward**，块间状态按截断 BPTT detach（梯度窗口 = chunk，前向无损）。
    多卡 DDP 的 reducer 假定"一次 forward 对应一次 backward"，因此该路径目前
    只支持单卡，多卡会显式拒绝；`trainer.py` 的 `StreamingTrainer` 是另一套后端契约驱动。
  两条路径都保持"核在任意长度有定义"；区别只在**训练显存结构**，
  不再把架构性质写成主路径能力。
- `eval.py`：探针任务（长程检索、可解性）、流式一致性、显存记账
  （张量记账 + `cuda.max_memory_allocated`，不再用恒为 0 的 RSS 探测）。
- `scaling.py`：嵌入占比 ≤25% 约束 + `plan_budget()` 反解规模档
  （30M/150M/350M/1B/3B）+ 长程 FLOPs / KV / 流式状态三本账。
- `extrapolation.py`：`extrapolation_probe`（固定 L0 训练 + 1/2/4/8× 零微调评测）+
  `decay_ablation`（扫描 `decay_max`）。
- `purity.py`：架构纯洁性守卫（配置 + 实例双重断言）。
- **诚实状态**：所有"已验证"结论必须有可复现脚本与守护测试；失败的实验明确标注、
  不混入正面主张；历史失真（RFP 非因果、mix 非因果、语料造假、无来源数字）一律
  记入诚实状态表并作废相关结论。
