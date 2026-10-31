# 生物计算主义（Biological Computationalism）· 架构集成设计文档

> 状态：**已实现并接入训练闭环（opt-in，默认关闭，零回归）**。本文件把 Milinković & Aru
> (2026, *Neurosci. Biobehav. Rev.* 181, 106524) 的三条核心计算原则映射为 SimplCore
> 的可训练机制，并说明实现与诚实边界。
> 配套代码：`simplcore/core/bio_computation.py` + `train.py --enhance bio`。

---

## 0. 论文在说什么（与架构的对应关系）

论文《On biological and artificial consciousness: A case for biological
computationalism》点明当前数字 AI 与生物计算之间的核心鸿沟，并给出生物计算区别于数字
计算的**三条必要计算特征**（作者明确说这不限于碳基，只要非生物系统能复现这些计算属性即可）：

| 论文原则 | 核心主张 | 在本架构的落地点 |
| --- | --- | --- |
| **① 混合计算**（Hybrid computation） | 离散事件（神经脉冲）嵌入连续值过程（电场 / 离子流 / 局部场电位 / 树突整合），二者**相互构成**、不可抽象剥离 | SGC 是连续谱卷积（连续底物），token 是离散事件 → 用一个**常驻 O(1) 连续场**把二者耦合 |
| **② 尺度不可分**（Scale-inseparability / heterarchy） | 离子通道→树突→环路→全脑各尺度**双向耦合**：低尺度内生生成高尺度，高尺度同时约束低尺度（连续共决定），不存在「真正发生计算的特权尺度」 | 原本分离的 micro（token/路由）/ meso（连续场）/ macro（seq-len 课程）三尺度，被一个**可训练耦合器**双向连结 |
| **③ 代谢约束**（Metabolic / energetic constraints） | 代谢预算（脑占体重 2% 却耗 20% 能量）主动塑造学习、稳定性与信息流 | 一个**代谢账本 + 能量预算**把「能耗」做成可被反向传播塑形的信号：超额能耗产生效率损失，预算紧张时自动收缩场增益 |

**诚实边界**：这是把论文的**计算风格**做成机制实验件，**绝非声称模型「有意识」**。论文本身
也只论证这些是意识的*可能*必要条件，且明确可被非生物系统复现。我们做的是「如果数字系统
想逼近生物计算风格，架构上该怎么落地」的最小可行实现。

---

## 1. 支柱 ①：混合计算 —— `ContinuousField`

离散 token 处理嵌入连续场动力之中：

```
对每个位置 t（序列内 unroll）：
    spike_t   = spike(h_t)                       # 离散 → 连续：token 隐状态向场注入脉冲
    phi_{t+1} = (1 - dt·leak)·phi_t               # 耗散（生物连续场是耗散系统，保稳定）
                + dt·tanh(W·phi_t)                # 连续演化（神经场动力学）
                + spike_t                          # 脉冲注入
    g_t       = σ(gain(phi_t)) · scale            # 连续 → 离散：场调制 token 处理
    h'_t      = h_t + g_t ⊗ phi_t                 # 调制后的隐状态 → 头 → logits
```

* **O(1) 常驻态**：`phi` 跨 chunk 仅以 `(B, D)` 携带（detach 一次），与序列长度无关——
  与 SGC 流式不变量一致（无 max_len、长度无关）。
* **梯度贯通**：序列内 unroll 梯度贯通（场动力学可训）；chunk 边界才 detach（内存 O(1)）。
* **耗散项 `leak`**（默认 0.05）：保证 4M 上下文下场不会无限累积、数值稳定，也让代谢预算有意义。
* 验证：见 `test_bio.py::test_continuous_field_runs_and_backprops` / `test_continuous_field_state_is_o1_and_carries`。

---

## 2. 支柱 ②：尺度不可分 —— `MultiscaleCoupling`

把原本分离的 micro / meso / macro 三尺度用**可训练权重**双向连结（论文："lower scales
endogenously generate higher scales; higher scales simultaneously constrain lower
scales through continuous co-determination"）：

* **macro → meso**：`curriculum_frac`（训练课程进度 0→1，由 `train.py` 每步注入）经
  `macro_to_dt`（Linear）调制场的连续演化步长 `dt` —— 训练编排（高尺度）连续地约束
  连续动力（中尺度）。
* **meso → micro**：上一 chunk 的**场能量**经 `meso_to_gain`（Linear）调制 token 门增益
  缩放 —— 连续态（中尺度）约束离散处理（微尺度）。
* **micro → meso**：token 脉冲注入场（已在 `ContinuousField` 内）—— 离散事件（微尺度）
  内生地生成连续场（中尺度）。
* **meso → macro**：场「就绪度」（场能量）作为指标暴露给训练循环（中尺度反馈高尺度）。
* **meso/macro → micro（与 Unity 组合时）**：场能量与课程进度共同决定路由温度
  `router.temperature`（`BioEnhancer._route_temperature`），即中/高尺度连续约束离散路由——
  默认（bio 关、frac=0、场≈0）温度≈1.0，**零回归**。

所有耦合系数均可训练 ⇒ 「算法即底物」：尺度纠缠进权重，而非数字系统那样把算法与物理实现
干净分层。验证：`test_coupling_macro_to_meso_changes_dt` / `test_coupling_meso_to_micro_changes_gain` /
`test_bio_curriculum_frac_modulates_field_dt`。

---

## 3. 支柱 ③：代谢约束 —— `MetabolicMeter` + `EnergyBudget`

把「能耗」做成可被反向传播塑形的信号：

* 每个 token 的**归一化代谢能耗** `E = 1.0（基座底价）+ ‖phi‖/D（连续动力能耗）+ 专家激活占比（若 Unity）`。
* **代谢效率损失** `w·relu(E − budget)`：超额能耗受罚 ⇒ 模型被逼在预算内完成预测
  （"metabolism structures and guides information flow"）。
* **能量感知门控**：当 `‖phi‖/D > budget` 时自动收缩 token 门增益 ⇒ 稳定性 / 信息流被代谢塑造。
* 账本 `MetabolicMeter`：`last_energy / field_energy / waste / budget`，每步写入 `metrics.jsonl`
  的 `bio_energy / bio_field_energy / bio_waste / bio_curriculum_frac`，可观测跨尺度耦合与代谢压力。

旋钮（`train.py` CLI，均有零回归默认）：

| 旋钮 | 默认 | 含义 |
| --- | --- | --- |
| `--bio-energy-weight` | 0.01 | 代谢效率损失权重（超额能耗受罚强度） |
| `--bio-metabolic-budget` | 1.5 | 每 token 归一化代谢能量预算（超过即受罚） |
| `--bio-field-dt` | 0.5 | 连续场基础演化步长（macro 课程会动态调制） |
| `--bio-field-scale` | 1.0 | 连续场→token 调制门增益基础缩放 |

验证：`test_metabolic_meter_records_and_penalizes`。

---

## 4. 集成方式（opt-in，零回归）

`BioEnhancer` 是 `nn.Module`，经 `build_enhanced_model` 接入训练闭环：

* **包裹基座**（无 `unity`）：`em.model = BioEnhancer(base, cfg)`，前向在基座 ln_f 隐状态上施加
  连续场调制，基座 + 场 + 耦合器参数全部进优化器。
* **包裹 SimplUnity**（同时开 `unity`）：`BioEnhancer(model=None, unity=UnityModel)`，复用
  Unity 的**专家调制后**隐状态与路由，meso/macro 调节路由温度；**不重复登记基座参数**
  （基座在 Unity 内已冻结）。
* 与 `ponder`/`envlearn`/`unity` 同为「模型替换型」增强，**last-wins**：bio 覆盖前序模型替换型增强，
  直接包裹原始基座（或已构建的 UnityModel）。
* `train.py` 循环内每步注入 `curriculum_frac`（macro 尺度）；`metrics.jsonl` 记录生物账本。

零回归验证：`test_bio.py::test_bio_off_is_identity_to_base`（不进 enhance 时与基座逐位一致）。

---

## 5. 生成期也跑混合计算（`stream_step` / `generate`）

**问题**：训练走 `BioEnhancer.forward`（整段并行），而生成走基座的 `stream_step`（逐 token）——
两者不是同一条路径，所以开 `bio` 后**生成期会退回基座**，连续场完全失效。

**解法**：把连续场做成**逐 token 可步进**的递推，并在 `BioEnhancer` 上补齐流式接口。

| 新增 | 位置 | 说明 |
| --- | --- | --- |
| `ContinuousField.step(h_t)` | `bio_computation.py` | 单 token 场递推，与 `forward_sequence` 内循环**逐位一致**（段内 phi 连续、无 detach） |
| `ContinuousField.reset(B, …)` | 同上 | 序列起点复位常驻场（`init_stream` 调用） |
| `BioEnhancer.init_stream` | 同上 | 底层 O(1) 递推状态 + 场复位 + 段首缓存耦合 `dt`/`gain` |
| `BioEnhancer.stream_step` | 同上 | 底层隐状态 → 场单步调制 → 头；返回下一 token logits |
| `BioEnhancer.generate` / `generate_text` | 同上 | 与 `SimplCoreModel.generate` 同签名，O(1) 流式路径也跑混合计算 |
| `BioEnhancer.checkpoint_stream` / `restore_stream` | 同上 | 流式状态可持久化（含 `phi`），断点恢复后逐位一致 |
| `UnityModel.init_stream` / `stream_step_hidden` / `stream_step` | `unity/model.py` | Unity 的 O(1) 流式；`stream_step_hidden` 返回**专家调制后**隐状态供 bio 施加场调制 |
| `UnityModel.forward(return_expert_hidden=True)` | 同上 | 训练前向与流式对齐（此前 bio+unity 组合漏掉了专家贡献） |

**段粒度一致性**：训练时 `dt`/`gain` 每 chunk 算一次；流式时改为**每段（序列）算一次**
（`init_stream` 时缓存），与训练语义一致——耦合系数在段内恒定，场本身逐 token 演化。

**理论守卫**：`test_bio_stream.py::test_stream_matches_forward_base` 断言
「逐 token 流式 logits」与「整段前向 logits」逐位一致（atol 1e-5）；
`test_checkpoint_restore_stream_roundtrip` 断言断点恢复后与不间断流式逐位一致。

---

## 6. meso → macro 反馈：连续场就绪度驱动的自适应课程

论文的尺度不可分是**双向**的：macro（训练编排）约束 meso（连续场），而 meso 也应当反过来
约束 macro。此前只有前者（`curriculum_frac → dt`），这条边是空的。

**就绪度探针** `BioEnhancer.readiness() ∈ [0,1]`（回答「模型现在能承受更长的上下文了吗」）：

* 场能量冲破代谢天花板（`budget × D`）→ **0.0**：系统代谢失衡（场在暴涨）⇒ 当前长度
  已超出吸收能力；
* 场稳定（瞬时能量 ≈ EMA）且代谢余量充足 → 接近 1.0；
* 场仍在快速增长 → 稳定度下降 → 就绪度下降。

**自适应课程** `SeqLenCurriculum.promote(step, readiness, gate_threshold, grace_steps)`：

* 调度边界是**最早**可能推进的时刻（绝不提前，避免长度失控）；
* `readiness < gate_threshold` → **推迟**拉长（模型还没消化当前长度时硬拉到 4M
  只会把长上下文表征练废——这正是 4M 课程的真实风险）；
* 越过边界超过 `grace_steps` 仍不达标 → **强制推进**，避免课程死锁
  （生物体的发育也不能无限期停滞）。

**旋钮**（`train.py` CLI，均有零回归默认）：

| 旋钮 | 默认 | 含义 |
| --- | --- | --- |
| `--bio-adaptive-curriculum` | 关（flag） | 打开自适应课程（需同时 `--enhance bio` 才有就绪度信号） |
| `--bio-readiness-gate` | 0.5 | 就绪度阈值：低于该值推迟阶段推进 |
| `--bio-readiness-grace` | 200 | 宽限步数：越过边界超过该步数仍未达阈值则强制推进 |

**metrics.jsonl** 新增 `bio_readiness`（当前就绪度）与 `bio_curriculum_stage`（已达成阶段）。

实测（`tuan-30m`，40 步，课程 64→128→256，阈值 0.5 / 宽限 5）：
就绪度从 1.0 降到 0.16（场暴涨）再回升到 0.52（场趋稳）；第一阶段边界在 step 12，
因就绪度 0.455 未达标而**推迟到 step 17 才推进**；第二阶段靠宽限期强制推进。

---

## 7. 诚实边界与下一步

* 这是**计算风格**的机制实验件，不声称意识、不声称质量优势；机制已接线 ≠ 训出质量优势（MI250 战役验证）。
* 生成期混合计算已接入（§5），但**只覆盖 bio 自身**：`unity`/`ponder` 的生成路径
  仍各自走自己的流式实现；`--enhance bio,unity` 组合下生成走 bio 的 Unity 流式路径。
* 4M 上下文下连续场的耗散稳定性已用 `leak` 保证，但真实 4M 训练显存/数值仍待 MI250 实测。
* 自适应课程只做**推迟**，不做**提前**（提前拉长风险大于收益）；若未来要提前推进，
  需要额外的离线上游任务验证，不能靠就绪度单指标拍板。
* 可选下一步：把就绪度信号接到 loss 权重（代谢压力直接塑形学习率）、给 Unity 的
  后台自进化加代谢预算（专家生长也受能量上限约束）。

