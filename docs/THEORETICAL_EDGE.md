# 理论优势论证 —— SimplCore 相对传统 Transformer 架构

> 本文逐模块论证 **SimplCore**（`simplcore/core/`）在**机制层面**相对传统
> Transformer 架构的优势，并明确列出每条优势对应的**守护测试**与**适用边界**。
> 所有数字均由 `simplcore/core/eval/scaling.py` 的账本函数实算得出（脚本见 §6），
> 不是估算。

---

## 0. 论点与对照口径

**论点**：SimplCore 不是"Transformer 换个 mixer"，而是在**四个正交维度**上
替换了 Transformer 的核心机制，每一条都可在代码里指出、可被测试守护：

| 维度 | Transformer 的做法 | SimplCore 的做法 | 带来的优势 |
|------|-------------------|------------------|-----------|
| 混合算子 | `softmax(QKᵀ)V`，内容寻址的**两两比较** | 阻尼振荡基**谱卷积**，位置相对核 | 训练 `O(L log L)`、推理 `O(1)`、状态与 L 无关 |
| 位置编码 | 位置嵌入表 / RoPE / ALiBi | **因果卷积核本身编码位置** | 无位置表 → 无 `max_len` 截断 → 长度外推天然有定义 |
| 流式状态 | KV cache `O(L·D)`，随上下文线性膨胀 | 常驻振荡器态 `2·M·D`，**与 L 无关** | 解码显存与吞吐不随上下文退化 |
| 不确定性 | 无显式表达，靠 softmax 熵间接反映 | **四层闸门**（L1 选择 / L2 证据 / L3 不确定 / L4 拒答） | 幻觉可被结构性地拦截，而不是"希望它别幻觉" |

**对照口径**：本文所说的"传统 Transformer 架构"= 至少满足以下之一的结构：
(1) 含 softmax 自注意力；(2) 维护随序列长度线性增长的 KV cache；
(3) 依赖显式位置编码机制；(4) 用两两相似度做 token 交互。
SimplCore 四条**全部不满足**。

---

## 1. 逐模块论证

### 1.1 SGC mixer —— `simplcore/core/arch/sgc.py`

**Transformer 对应**：`softmax(QKᵀ)V` 的全局 token 交互。

**SimplCore 机制**：用 M 个**阻尼振荡基**参数化因果全局卷积核

```
psi_m[τ] = ρ_m^τ · sin((τ+1)·θ_m),      τ ≥ 0
K[τ, d] = w_d · 1{τ=0} + Σ_m C[m, d] · psi_m[τ]
```

并给出**对偶形式（dual form）**：

* 并行/训练：`y[t] = Σ_{s≤t} K[t−s]·u[s]`，FFT 计算 → `O(L log L)`；
* 流式/推理：二阶 IIR 递推（每基 2 个状态）
  ```
  y_m[t] = a_m·y_m[t−1] + b_m·y_m[t−2] + sin(θ_m)·u[t]
  a_m = 2ρ_m cos(θ_m),  b_m = −ρ_m²
  ```
  → 常驻状态 `2·M·D`，**与 L 无关**，每 token `O(1)`。

**理论优势**：

1. **内容寻址 → 位置相对核**。注意力按内容挑位置，核按内容变化需要重算；
   谱基把"记住多久"编码成 `ρ_m`，"记什么频率"编码成 `θ_m`，
   一组 M 个基可组合出任意长的衰减振荡响应，且**每个基的衰减尺度是可学习的**。
2. **双形式 bit-exact**。同一套权重在训练（FFT）与推理（IIR）下给出相同结果
   （fp32 实测 `max|Δ| < 4e-5`），因此"训练/推理不一致"这一类最难查的 bug
   在结构上被消除。
3. **内容门控**。`u = W_in·norm(x)·σ(W_g·norm(x))`，有效核随内容变化
   （GLA/GSS 家族的 selective 思想），但门是**逐位置瞬时乘法**，
   不进核、不增状态 → 不改变 `O(L log L)` / `O(1)` 的复杂度阶。
4. **无位置嵌入表**。位置由 `τ`（延迟）隐式编码，训练长度之外的长度同样有定义。

**守护测试**：`tests/test_mixer_math.py`（双形式等价、因果性、梯度可达、长度外推）、
`tests/test_gates.py`、`tests/test_selective.py`。

**变体 SGC-R（v1.2，`mixer="sgc-r"`）**：在共享同一压缩通路上补 MiTA 意义的
routing 分支——M 个振荡基按频率分 G 组"谱专家"，逐 token top-k 注入。理论优势
论证：deformable routed experts 给"内容无关核"补内容自适应的**基组选择**，
而路由门只作用于输入注入端 ⇒ O(1) 状态与 FFT/IIR 对偶不变（`test_sgc_routed`
按同等强度守护）；被切换掉的专家记忆在状态里齐次衰减不丢失（reusable top-k）。
细节与边界见 `SGC_CORE.md` §10；质量效果未实测（与 SGC 主线同属待验证）。

---

### 1.2 SRFN mixer —— `simplcore/core/arch/srf.py` + `simplcore/core/arch/ssm.py`

**Transformer 对应**：注意力只能"读"历史，不能决定"记多久"。

**SimplCore 机制**：把 SGC 的谱基升级为**时变状态空间**——

* `A` 保持对角、特征值为**复数** `λ_m = ρ_m e^{iθ_m}`（阻尼振荡子），
  因此能表示**周期记忆**（纯衰减的 S4D/Mamba 做不到）；
* `Δ_t` 由输入预测 → 递推变成时变 → **FFT 技巧失效**，
  改用 **chunk 化下三角 Toeplitz 矩阵**（一个 chunk 一次批量矩阵乘，
  而不是 C 次 Python 步）；
* 三条支路分工：selective SSM 管慢变长记忆、深度因果卷积管 1~3 token 的精确局部模式、
  两层门控管证据/不确定性。

**理论优势**：

1. **选择性**：`Δ_t` 随输入变化 → 模型可以"这一步记久一点，下一步就衰减"。
   固定核（纯 SGC/S4）做不到这件事，这正是 Mamba 的核心贡献。
2. **周期性记忆**：复特征值振荡子提供**可调的周期**，能表达"每隔 N 个 token 重复"
   类的结构；纯衰减 SSM 只能表达"越来越旧"。
3. **chunk 化不牺牲并行度**：时变核仍以 `O(L·D/chunk)` 的批量矩阵乘完成，
   而不是退化成 `O(L)` 次串行 Python 循环。

**守护测试**：`tests/test_srf.py`（chunk 并行 vs 逐 token 递推 `atol=1e-5`、
Delta 输入依赖性、无未来泄漏、状态字节数与 L 无关）。

---

### 1.3 GatedSpectralConv2 —— `simplcore/core/arch/selective.py`

**Transformer 对应**：GQA/MQA 通过**头维度**共享 KV 来省状态。

**SimplCore 机制**：GLA / Mamba2 式**通道独立双门**：

* `γ(t,c) = σ(W_γ h[t,c])` —— Mamba2 的时间常数内容化，逐位置逐通道调制衰减写入；
* `β(t,c) = σ(W_β h[t,c])` —— GatedDeltaNet 的写入门，逐位置逐通道调制状态写入。

**理论优势**：

1. **状态压缩的粒度更细**。注意力只能在**头**这一层做共享；本实现做到
   **通道**这一层 —— 通道独立意味着不同通道可以有完全不同的记忆半衰期。
2. **门不进核、不增状态**。两个门都是逐位置瞬时乘法，
   FFT 形式 `u_gated = in_proj(h)·σ(gate)·σ(γ)·σ(β)` 不改变长程项结构，
   IIR 路径只需把递推输入端换成 `c_m·u_gated[t]`，状态形状不变
   → 两条路径天然逐位一致。

**守护测试**：`tests/test_selective.py`、`tests/test_gates_selective.py`。

---

### 1.4 四层闸门 —— `gates.py` / `abstain.py` / `grounding.py`

**Transformer 对应**：没有对应物。Transformer 的"置信度"是 softmax 熵的副产品，
无法表达"这句话没有证据"。

**SimplCore 机制**：

```
L1 选择门   selective    mixer 输入    按内容选择参与位置/通道
L2 证据门   evidence     mixer 谱基    前缀能量不足时压制长程路径
L3 不确定门 uncertainty   mixer 输出    长短路径分歧大时回退到局部证据
L4 拒答门   abstention   输出头        没有外部证据支撑时拒绝作答
```

**理论优势**：

1. **L2 的因果前缀语义**：`c_t = (1/t)·Σ_{s≤t}‖u_s‖²` 只依赖过去，
   并行路径用 `cumsum`、流式路径用常驻标量运行和（`energy_sum`，字节数恒定），
   **两者逐位一致** → 闸门不会在流式与整段推理之间产生行为漂移。
2. **L4 不是 softmax**：`Linear(dim → 1)` 标量头，无注意力、无 `QKᵀ`。
3. **监督来自外部**：`grounding.py` 用**字符级最长公共子序列覆盖率 + 二元组覆盖率
   + 数字核对 + 否定词极性核对**四个互补判据，从候选证据片段里
   产出 `grounded` 标签。自监督的"拒答标签"会退化成置信度估计的平方，
   四层闸门就白做了 —— 所以标签必须来自外部证据。

**守护测试**：`tests/test_gates.py`、`tests/test_gates_selective.py`、
`tests/test_grounding.py`、`tests/test_dialogue_runtime.py`（L4 拒答决策）。

---

### 1.5 CMT 压缩记忆库 —— `simplcore/core/memory/memory.py`

**Transformer 对应**：CMT 原文在**注意力层**注入 KV。

**SimplCore 机制**：SimplCore 没有注意力也没有 KV cache（流式状态是 `2·M·D` 的
O(1) 振荡器态），因此把注入点改为 **残差流上的加性记忆项** —— 这是本架构原生的等价物。

**理论优势**：

1. **注入点与架构一致**：不引入 softmax 注意力，也不破坏 `O(1)` 流式不变式。
2. **无灾难性遗忘**：CMT 的基座权重冻结，新知识进外部记忆库；
   记忆检索与聚合（self-matching + top-aggregation）与基座解耦。
3. **只对增量计损失**：`MemoryAwareObjective` 只对"记忆相对基座的增量"计损失，
   避免记忆库退化成基座的第二个拷贝。

**诚实边界**：本模块只负责记忆的**编码/存储/检索/聚合**；
"记忆内容对不对"由 Dreaming 循环的 no-regression gate 负责。

---

### 1.6 Dreaming 自进化循环 —— `simplcore/core/memory/dream/`（0.1-α 命名降调：`ReflectionLoop`）

**Transformer 对应**：无。Agent 的反思是**推理期**的（本次回答里多想一步），
Dreaming 的反思是**离线**的。

**SimplCore 机制**：

```
在线 (serving)                    离线 (dreaming)
─────────────                    ────────────────
推理 → 轨迹入流                   HARVEST    采样高价值经验
     ↓                           REFLECT    模型自评 → 洞见
CMT 压缩记忆库                    SYNTHESIZE 洞见 → 训练信号
(检索 → 聚合 → 注入)              CONSOLIDATE 巩固（记忆 或 权重)
     ↑                           VERIFY     held-out 评测
权重 / 记忆库 ←── COMMIT         ROLLBACK   退化则回滚
```

**理论优势**：

1. **离线反思不污染线上延迟**。洞见在"睡眠"阶段生成，线上只付推理成本。
2. **无回归门（no-regression gate）**：配对 bootstrap CI 判定"这次固化是否真的更好"，
   不是"看着像更好"。退化就 ROLLBACK（快照/恢复），保证模型**不会变差**。
3. **可回滚**：snapshot/restore 让"改权重"这件事变成可撤销操作 ——
   `test_consolidation.py` 逐位断言回滚后权重与快照完全一致。
4. **巩固真的会写权重**。`default_consolidator` 把训练信号变成梯度步：
   标签错位一格（模型 loss 内部不做 shift）、只对目标段计损失、
   样本权重生效、**不对振荡基参数做 weight decay**（那会掐死长程记忆）。
   此前这条链路在默认配置下是**断的** —— 只有调用方自备 `consolidator` 才写得进去，
   而现成的测试用的是空 consolidator，等于"报告说成功、权重一个字节没变"。
   现在：内置实现 + `ReflectionReport.wrote_weights` 如实报告是否真的写了。

---

### 1.7 代码执行沙箱 + pass@k 验证器 —— `simplcore/core/memory/code_exec.py`

**Transformer 对应**：无。

**问题**：支柱 C（"去真实环境写代码 → 写回权重"）缺的不是训练代码，
而是**能判定代码对错的独立评测器**。没有独立 verifier 时，
`ReflectionLoop` 在 `mode="weights"` 下必须直接 ROLLBACK ——
因为"让 gate 恒通过"等于没有 gate。

**SimplCore 机制**：

* `LocalPythonSandbox` —— 隔离子进程执行，带超时与输出截断；
* `extract_code` —— 从模型输出剥出 ` ```python ` 代码块；
* `CodeProblem` —— 一道 held-out 题（prompt + 断言测试）；
* `CodeExecutionVerifier` —— `(model) -> {"pass_rate": [逐题 0..1]}`，
  正好满足 `EvalScores.from_any` 的"逐样本分数"要求，
  可直接用于 no-regression gate 的配对 bootstrap。

**理论优势**：**代码能力来自外部环境的真实反馈，而不是自评**。
这让"学会写代码"成为一个**可验证的收敛过程**，而不是一个自我循环的幻觉。

**诚实边界**：这是**进程级**沙箱，不是安全边界（无 OS 级隔离、不拦网络、不拦文件系统）。

---

### 1.8 长上下文外推工具 —— `simplcore/core/eval/extrapolation.py`

**Transformer 对应**：RoPE / ALiBi / NTK 缩放 / YaRN —— 都是**外推补丁**，
因为位置编码表或旋转频率在训练长度外"没有定义"。

**SimplCore 机制**：因果卷积核本身编码位置，**没有位置表**，
所以"训练长度之外的长度"天然有定义。工具提供两把可复现的尺子：

1. **长度外推探针** `extrapolation_probe`：在固定短训练长度 `L0` 上训练可解任务，
   再在 `L0` 的 1/2/4/8/16 倍长度上**零微调**评测，
   报告准确率 + 困惑度 + 流式状态字节数（应恒定）。
2. **衰减消融** `decay_ablation`：扫描 `decay_max`（振荡基时间尺度先验），
   报告训练困惑度 + 2× 外推准确率。

**理论优势**：外推不是"调参调出来的"，而是**结构上不需要补丁**。

---

### 1.9 算力账本 —— `simplcore/core/eval/scaling.py`

**Transformer 对应**：`6·N·D` 这个粗略近似 + KV cache 估算。

**SimplCore 机制**：把长程项与线性项**分开算**，并区分训练/解码两个口径：

* 训练长程项（每层每 token）：`2·D·M + 10·D·log2(2L) + 6·D + 22·D`
  （核构建 + rfft/irfft + 频域复乘 + 逐点激活）→ **亚线性**（除 log 项外全是常数）；
* 解码每 token：`(5·M + 22)·D` 每层（IIR 递推 5 FLOPs × M·D 个状态元素
  + 逐点激活）→ **与 L 完全无关**；
  （v3.1 修正：旧版分别漏了核构建 `2·D·M`、把递推误写为 `6·D` 漏乘基数 M ——
  由 `mathproof.py` P11 逐项重推导守护；结论不变且更强。）
* KV cache 字节数：**恒为 0**；
* 流式状态字节数：`((2D + D·K + 1)·4)·num_layers + 4D` → **与 L 完全无关**。

**理论优势**：账本本身是"小模型抵大模型"机制的第一环 ——
同样的 FLOPs 预算，SimplCore 的长程项更便宜 → **能喂更多 token**，
而小模型的能力几乎完全由 token 数决定。

---

### 1.10 架构纯洁性守卫 —— `simplcore/core/arch/purity.py`

**Transformer 对应**：无。

**机制**：两道断言把"脱离 Transformer"变成 CI 可执行的硬约束：

* `assert_pure_config(cfg)` —— 建模前挡掉会引入注意力的配置；
* `assert_attention_free(model)` —— 建模后遍历 `named_modules()`，
  确认实例化出来的模型里**真的没有**任何 `*Attention*` / `*attn*` 模块。

再加 `SimplCoreConfig.validate()` 只允许注册表白名单（当前 `sgc` / `sgc-nogate` / `sgc-nolocal` / `sgc-r` / `srf`，单一事实源 `registry.mixer_kinds`）。

**为什么重要**："这套架构不属于 Transformer 框架"如果只写在文档里就会悄悄失效 ——
某次为了追指标打开一个注意力开关，注意力就回来了，而 loss 曲线完全正常。
守卫让它在合并前直接失败。

---

### 1.11 思考加速 —— `simplcore/core/arch/accel.py`

**Transformer 对应**：对整条序列/所有 token 跑满全部层数是固定成本；没有"简单 token 少算、难 token 多算"的机制。

**SimplCore 机制**：`AdaptiveDepth` 包裹 `SimplCoreModel`，对每个 token：

* 先跑第 0 层（锚点）算"必要性分数"（熵 / 置信度）；
* 若分数超过 `skip_threshold` 则只跑 `min_depth` 层，否则跑 `max_depth` 层；
* `ComputeLedger` 记录 `blocks_budget`（全算力预算）与 `blocks_run`（实际跑的层数），
  结构性加速比 = `blocks_budget / blocks_run`，**不依赖任何硬件计时**。

**理论优势**：

1. **逐 token 自适应深度**。重要性随 token 不同而不同，简单位置少算、难位置多算，
   平均算力按内容分布而非固定峰值。
2. **可验证的退化点**。`skip_threshold = -inf` 时永远走 `max_depth`，
   输出必须与 `model.stream_step` **逐位一致**——这把"加速没改模型行为"变成一条
   可执行断言（见 §6 的 `accel.py` 检查项）。
3. **不引入注意力、不改 O(1) 流式不变式**；加速器自身也被 `purity` 守卫。

**诚实边界**：加速比是**结构性上限**（基于"少跑的层"），不是实测 wall-clock 提速；
真实收益取决于 token 难度分布与实现开销。

---

### 1.12 深度推理 —— `simplcore/core/dialogue/reasoning.py`

**Transformer 对应**：o1 / R1 式"先想后答"靠注意力 KV cache 累积思考步，
思考越多、KV 越长、显存越炸。

**SimplCore 机制**：`IterativeRefiner` 跑"提示 → 多步思考 → 答案"循环，`policy`：

* `fixed_steps` —— 固定走 `max_steps` 步；
* `stabilize` —— 候选答案连续 `patience` 步不变即早停（**最大的加速器**：
  简单题想 1~2 步就够，不必烧满 `max_steps`）。

`tool` 钩子把外部观察（如 `envlearn.LocalCodeEnvironment` 的代码执行结果）流回状态，
这正是深度推理与环境学习的接缝。`ReasoningTrace` 记录结构化轨迹，`stabilized()`
是早停判定的纯函数。

**理论优势**：

1. **O(1) 流式状态天然适合多步思考**。思考步只是不断往状态里喂 token，状态始终常驻大小，
   不会因"想了 N 步"而爆显存（对比注意力 KV cache 随步数线性膨胀）。
2. **`stabilize` 早停由"候选答案稳定"驱动**，缩短的是思考步数，与单步算力解耦
   （单步算力由 `AdaptiveDepth` 负责）。
3. **不引入注意力**。

**诚实边界**：本模块是**控制流骨架**，不是"已训好的推理模型"。它保证循环正确、步数可控、
流式状态不变、可插工具；至于模型有没有被训出真正的多步推理能力，与 `theory.py` 的诚实边界一致——
**机制正确 ≠ 质量达标**，未训的玩具模型只会吐无意义的循环文本。

---

### 1.13 环境学习 —— `simplcore/core/training/envlearn.py`

**Transformer 对应**：传统全量训练把整份权重都放进优化器状态（Adam 的 m/v 是参数的 2 倍），
显存随**总权重**增长；且"一次性把数据灌进去"容易过拟合/灾难性遗忘。

**SimplCore 机制**（用户给定的 1B 基座 + ≤2B 可学、总上限 3B 规格）：

* `WeightBudget`：1B 基座**冻结**（`requires_grad_(False)`），只对新增的 ≤2B 适配器
  反向与优化；`vram_params()` 只算可学部分 → 显存只随**可学权重**增长，不随总权重增长。
* 分阶段蒸馏（pillars）：**① 环境交互**（`LocalCodeEnvironment` 执行生成的代码）、
  **② 总结**（失败轨迹 → 少量正确实现样本，真正"学习并总结"而非"背下来"）、
  **③ 蒸馏进权重**（`dream.default_consolidator`，只覆盖适配器）、
  **④ no-regression gate**（环境验证器打分、配对 bootstrap 判退化、退化回滚快照）、
  **⑤ 预算守卫**（分配/增长时硬性拒绝超 3B / 超 2B 可学）。
* `AdapterBlock` / `LearnedAdapter`：per-token 门控残差（SwiGLU 式），无状态、无注意力。

**理论优势**：

1. **显存/时间占用优于传统全量训练**：冻结 1B 基座，优化器状态只覆盖可学适配器，
   这正是规格里"对显存、时间占用相较于传统训练更加有优势"的来由。
2. **分阶段 = 真正学习**：不一次性背数据，而是交互 → 总结 → 蒸馏 → 验证 → 回滚，
   退化的固化会被门拦下并回滚。
3. **不引入注意力**；适配器被 `purity` 守卫。

**诚实边界**：本模块是**可运行的框架**；它保证"交互→总结→蒸馏→回滚→预算"这条链真实、可测、
且只训适配器。至于一个 1B+2B 的真实模型是否训出质量优势，**未训练、未评测**，
与 `theory.py` / `dream.py` 的诚实边界一致。`LocalCodeEnvironment` 复用 `code_exec`
进程级沙箱，上生产应换容器/gVisor。

---

## 2. 复杂度对照（实算，dim=4096 / 32 层 / M=64）

### 2.1 训练期长程 FLOPs / token

| 序列长度 | SimplCore 长程项 | Transformer 长程项 | 倍数 |
|---------|------------------|-------------------|------|
| 2,048 | 1.085e+08 | 3.221e+09 | **30×** |
| 8,192 | 1.164e+08 | 1.288e+10 | **111×** |
| 32,768 | 1.243e+08 | 5.154e+10 | **415×** |
| 131,072 | 1.321e+08 | 2.062e+11 | **1,560×** |

（v3.1 修正后的口径：长程项含核构建与逐点激活，比旧表更诚实地偏大。）

SimplCore 的长程项随 `log L` 增长，Transformer 随 `L` 线性增长 ——
**上下文越长，差距越大，且差距无上界**。

### 2.2 解码期单序列占用（L = 32,768）

| | 显存 | 说明 |
|---|------|------|
| Transformer KV cache | **32.00 GiB** | `2·D·L·layers·4B`，示例 D=4096, layers=32, L=32768 |
| SimplCore 流式状态 | **3.07 MiB** | 示例 M=64, D=512, 12 层, batch=1, fp32；与 L 无关 |
| 倍数 | **≈10,670×** | 跨规模示例，不用于论证质量优势 |

同时，SimplCore 的**解码每 token FLOPs 在 L = 2048 / 32768 / 131072 时完全相同**
（1.6170e+10，v3.1 修正 mixer 常数后），而 Transformer 与 L 成正比。

### 2.3 预置规模（`SIMPLCORE_PRESETS`）

| preset | 参数量 |
|--------|--------|
| `tuan-30m` | 30,085,298 |
| `tuan-150m` | 131,401,020 |
| `tuan-350m` | 348,798,564 |
| `tuan-1b` | 1,013,407,563 |
| `tuan-3b` | 3,238,718,112 |
| `srf-30m` | 30,254,268 |

---

## 3. 优势总表

| 模块 | Transformer 对应 | SimplCore 机制 | 理论优势 | 守护测试 |
|------|------------------|----------------|---------|---------|
| `sgc.py` | `softmax(QKᵀ)V` | 阻尼振荡基谱卷积 + FFT↔IIR 对偶 | `O(L log L)` 训练 / `O(1)` 推理 / 状态与 L 无关 / bit-exact 双形式 | `test_mixer_math.py` |
| `srf.py` `ssm.py` | 固定记忆时长的 SSM | 输入相关 `Δ_t` + 复特征值振荡子 + chunk 化 Toeplitz | 记忆时长可选择 / 可表达周期记忆 / 保持批量并行 | `test_srf.py` |
| `selective.py` | GQA 的头级共享 | GLA/Mamba2 通道级双门 `γ`/`β` | 状态压缩粒度到通道级 / 不进核故不改复杂度阶 | `test_selective.py` |
| `gates.py` | softmax 熵 | L2 证据门 + L3 不确定门（共享数学） | 因果前缀语义 / 流式与整段逐位一致 | `test_gates.py` |
| `abstain.py` | 无 | L4 拒答头（`Linear(dim→1)`） | 结构性拒答而非"希望别幻觉" | `test_gates.py` / `test_dialogue_runtime.py` |
| `grounding.py` | 无 | LCS + bigram + 数字 + 极性 四判据归因 | 拒答标签来自外部证据，可训练 | `test_grounding.py` |
| `memory.py` | CMT 在注意力层注入 KV | 残差流加性记忆项 | 与架构一致 / 破坏 O(1) 不变式为零 / 无灾难性遗忘 | `test_dream.py` / `test_dream_evolution.py` |
| `dream.py` | 无（Agent 反思是推理期的） | 离线反思 + no-regression gate + 回滚 + 内置巩固器 | 离线不污染线上延迟 / 固化可撤销 / 不会变差 / 权重真的被写进去 | `test_dream*.py` / `test_consolidation.py` |
| `code_exec.py` | 无 | 沙箱 + pass@k 验证器 | 代码能力来自真实环境反馈，可验证收敛 | `test_code_exec.py` |
| `extrapolation.py` | RoPE/ALiBi/YaRN 补丁 | 无位置表 | 外推结构上成立，无需补丁 | `test_extrapolation.py` |
| `scaling.py` | `6ND` 粗估 | 长程项与线性项分离、训练/解码分口径 | 可证"同算力喂更多 token" | `test_scaling.py` |
| `purity.py` | 无 | 配置 + 实例双重断言 | 脱离 Transformer 成为 CI 硬约束 | `test_architecture_purity.py` |
| `accel.py` | 无（Transformer 对所有 token 跑满层数） | 逐 token 按必要性分数选 `min_depth`/`max_depth`；`skip_threshold=-inf` 逐位退化 | 逐 token 自适应深度 / 结构性加速比可记账 / 退化点可验证 | `test_accel.py` |
| `reasoning.py` | KV cache 随思考步线性膨胀 | 迭代式思考循环 + `fixed_steps`/`stabilize` 早停 + `tool` 钩子 | O(1) 流式状态天然适合多步思考 / 早停缩短思考步数 | `test_reasoning.py` |
| `envlearn.py` | 全量训练显存随总权重增长、易背数据 | 1B 冻结 + ≤2B 可学适配器（总上限 3B）+ 分阶段蒸馏 + no-regression gate | 显存只随可学权重增长 / 分阶段真正学习 / 退化回滚 | `test_envlearn.py` |
| `eval/claims_audit.py` | 无 | 把上表每条主张变成可执行断言 | 主张不再只是文档，改代码会被 CI 抓住 | `test_claims_audit.py` |

> 上表每一行都有对应的**可执行检查**（`simplcore.core.eval.claims_audit.py`），
> 共 34 项，见 §6。`test_claims_audit.py` 还会断言本表点名的 15 个模块
> （`theory.py` 自身除外 —— 它就是那个验证器）每一个都被自动验证过 ——
> 文档与代码不会各说各话。

---

## 4. 诚实边界（Transformer 仍然更强的地方）

本文只论证**机制层面**的优势。以下差异是真实存在的，不应被淡化：

1. **同参数量的实证质量未验证**。简化注意力（线性注意力 / 状态空间模型）在
   同参数量下的语言建模质量**并非自动超过**标准注意力 —— 注意力有"按内容精确寻址"
   的能力，而固定基字典是"按频率匹配"的近似。SimplCore 的长程项是
   **M 个可学习衰减振荡基**，表达力上限低于 `softmax(QKᵀ)` 的精确寻址。
   **本文论证的是复杂度、状态与机制优势，不是质量优势。**
2. **质量证据缺失**。未与 S4 / Mamba / Hyena / GLA 在 LRA 等公认基准做公平对比。
3. **1B 以上规模从未实际训练**。配置与链路已打通，曲线待测。
4. **长上下文外推只有"输出有限"证据**，无质量证据（`extrapolation.py` 就是为了补这块）。
5. **代码沙箱是进程级**，不是安全边界。
6. **四层闸门的监督依赖外部证据来源**；没有高质量证据源时，L4 训不起来。

因此：**不得据此宣称任何 SOTA**。架构选型应基于上述机制差异与实测曲线共同决定。

---

## 5. 相关文档

| 文档 | 内容 |
|------|------|
| `docs/ARCHITECTURE.md` | 总体架构设计（数据流 / mixer 分类 / 流式 / 记忆 / 编解码） |
| `docs/SGC_CORE.md` | SGC mixer 深度解析（振荡基字典 / FFT↔IIR 对偶 / 内容门控 / 四层闸门） |
| `docs/MODULES.md` | 逐模块索引 |
| `docs/TRAINING_BUDGET_MI250.md` | 1× MI250 上的训练预算估算 |

---

## 6. 复现本文数字

本文的每条主张都有一个**可执行**的对应项，在 `simplcore.core.eval.claims_audit.py` 里：

```bash
# 跑完全部机制层检查（34 项），打印对齐表格；有失败项时退出码为 1
python -m simplcore.core.eval.claims_audit

# 机器可读
python -m simplcore.core.eval.claims_audit --json

# 逐条打印
python -m simplcore.core.eval.claims_audit --verbose
```

它由 `tests/test_theory.py` 守护，并额外断言：**文档 §3 点名的 15 个模块
每一个都有对应检查**（防止"文档写了、代码里没人验"）。

```bash
python -m pytest simplcore/core/tests/test_theory.py -q   # 理论守卫
python -m pytest simplcore/core/tests/test_scaling.py -q  # 复杂度断言
python -m pytest simplcore/core/tests/test_mixer_math.py -q   # 双形式等价 / 因果性
python -m pytest simplcore/core/tests/test_architecture_purity.py -q   # 无注意力硬约束
python -m pytest simplcore/core/tests/test_mathproof.py -q    # 13 项数学证明（符号 + fp64）
python -m simplcore.core.mathproof                            # 数学证明 CLI 报告
```

§2 的数字由 `simplcore.core.scaling` 的 `train_flops_per_token` /
`infer_flops_per_token` / `stream_state_bytes` / `kv_cache_bytes_per_token`
对 `dim=4096, num_layers=32, num_bases=64, local_kernel=4` 的配置实算得出。

---

## 实测对照（0.1-α 独立复现，2026-10-31）—— 负面结果照账

> 口径：D=64、1–2 层、600–3000 步、1–3 seeds、合成任务、**无显著性检验**。
> 足以区分结构性强弱（1.000 vs 0.398 是数量级差异），不足以支撑任何定量的
> 「谁更好」。真实答案等 §5.13 战役运行。协议固化于 `eval/axis_probe.py`
> （X-axis-probe，`purity: allow-attention-reference` 对照臂）。

| 任务 | 能力类型 | attention | SGC | chance | 判定 |
|------|----------|-----------|-----|--------|------|
| 精确指向（2 层） | 指向某一位置 | 1.000 | 0.398 | 0.125 | **SGC 明确劣**（且低于局部卷积 0.427） |
| 聚合 L=129 | 前缀统计 | 0.938 | 0.875 | 0.5 | attention 略优 |
| 聚合 L=65（长跑） | 前缀统计 | 1.000 | 1.000 | 0.5 | 打平 |
| 精确计数（一票之差） | 近精确 | 0.812 / 0.938 | 0.812 / 0.938 | 0.5 | 逐位打平 |
| 短程 LM（Markov） | 语言建模 | ppl 14.63 | 15.71 | V=32 | attention 优 7% |

参数量口径：SGC 14,881 vs attention 30,209（同 D，SGC 少一个 D² 投影）——
对照对 SGC 有利，即便如此也没赢。

三条结论（与 §7.2 轴注记、`paper_audit` 交叉引用）：

1. **精确指向是 SGC 当前谱压缩状态的结构性劣势**，不是调参问题
   （0.398 vs 1.000，甚至低于局部卷积 0.427）。**后续 MQAR 实验把缺陷
   定位得更准：不是感受野，是 key→value 绑定**（MQAR G=4：attention
   0.500 vs SGC 0.3125 vs chance 0.125，聚合捷径被阻断后差距依旧）。
   **架构响应已落地两路**：`memory/sidecar.py` 三层精确引用（O(N) 旁路）
   与 `memory/binding_memory.py` 定容绑定记忆（R1，O(1) 内修绑定——
   见下方路线分级）。注意"常数状态建模必输"的原表述已被修正：
   O(1) 只意味着不随 L 增长，定容绑定副本同样是 O(1)——轴 D 论证的
   修正见 `SimplCore_Paper.md` §7.2 注记。
2. **轴 C"等效 4–7B"的通过率优势未复现**：聚合任务 attention 0.938 vs
   SGC 0.875；SGC 仅 loss 更低、训练快 32%。降级为「约一半参数同等质量 +
   成本优势」。§7.4 判据（4M 检索+摘要 vs YaRN TS-5B）仍未运行——本节是
   前哨信号，不是判据结果。
3. **"优于"是错的维度**。SGC 拿到的是效率轴，且是可证明的结构性质：
   常驻状态 36,868 B 四长度逐字节相同 vs attention 1M 上下文 ~120 GiB KV；
   训练缩放实测斜率 1.09 vs 2.35；流式路径未来泄漏逐位为 0。代价就是
   第 1 行。**engineering trade-off，不是"更好的模型"**——与 §7.2 轴 D
   自认"该轴必输"一致。

本文其余各节的主张不变：它们全部是**可运行断言**（结构性质），与本节的
**能力对照实测**（有输有赢）分属两个口径，不得互相混用。

---

## 修复路线分级（0.1-α，2026-10-31 复现报告）—— 一致性判据先行

**架构忠诚度的一致性判据**（取代"有没有 softmax"的表面口径）：
**状态是否与 L 无关？是否存在 O(L²) 两两交互？** 反 attention 的实质是反
「状态随 L 增长 + O(L²) 交互」，不是反 softmax 算子本身——定容检索是
softmax over S 个槽、O(S·D)、无 L 依赖，**不背叛架构**。

| 级别 | 路线 | 状态 |
|------|------|------|
| **A（完全一致，优先）** | **R1 定容绑定记忆**：写入单元从单 token 改为局部窗口绑定（depthwise 局部卷积天然覆盖相邻 k/v），槽内 key/value 分开存，FIFO 环形，状态严格 O(1)——修复轴 D 的正路 | ✅ 已落地 `memory/binding_memory.py` + **外部独立实现实测（二期）**：MQAR 两 seed 均提升 SGC（+27%）、72% 参数**在噪声内追平** attention（均值 0.2578 vs 0.2422，未超越——attention seed SD≈0.066）；3000 步退化（硬 argmin 不可微）的结构性修复已落地：`write_mode="soft"` 可微门控写入 |
| **A（完全一致，低成本）** | **R2 修频率网格**：logfreq 实测秩缺失（M=16 rank 14/16、M=128 rank 63/128，cond 2e17，与命题 5 病态同源） | ✅ 已落地 `sgc_init_mode="hybrid"`（线性均匀+黄金比扰动，满秩 cond 1.4，含端点锚定；数值秩守护 `_assert_numerical_rank` + 负向变异验证）；logfreq 默认行为不变 |
| **A（完全一致）** | **R3 分层多尺度 decay_max**：层 i 使用 decay_max·scale^i（深层更长记忆） | ✅ 已落地 `decay_scale`（opt-in；scale=1.0 逐位不变） |
| **A（完全一致）** | **R4 记忆升一等公民**：CMT 从"接线"提为核心 + 可微写入门 | ⬜ 规划（大件，另立战役） |
| **B（可用但须标注）** | **R5 滑窗注意力混合**：窗口固定 ⇒ 状态仍 O(1)，严格说不背叛，但削弱"纯 attention-free"叙事 | ⬜ 定位为务实 fallback，与 core 线**分开宣称**（Paper §3.4 混合路线既有框架内） |
| **C（背叛，不做）** | 全局注意力 / 随 L 增长的 KV cache / RoPE-YaRN | ❌ 那就成了 TS，效率主张全部消失 |

**价值定位（复现报告第五节，采纳为正式口径）**：正确的问题不是"质量上
打赢"（§7 自述推测、§5.8 自承不显著），而是**长上下文可行性**——
1M 上下文 SGC 1.948 GiB vs TS KV cache ~120 GiB 是**数量级差异**。
正式宣称口径：**"在 X 长度以上唯一成本可行的选择"**，而非"更好的模型"。

**诚实声明**：复现原型的定容记忆首版失败（SGC 0.3125 → 加记忆 0.1875，
三种写入策略含 oracle 几乎相同）——卡点定位在**读取侧缺绑定**（非写入、
非容量）；该原型另有未解决的梯度异常（mem_Wq 恒 0），故该失败是
**未验证**而非**已证否**。R1 的绑定写入即对此卡点的修复假设，
检索质量数值待训练验证。完整实验记录见 `docs/REPRODUCTION_REPORT.md`。
