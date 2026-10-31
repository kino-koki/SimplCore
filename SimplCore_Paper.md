# SimplCore-0.1-α：以阻尼振荡基谱卷积替代注意力——具备 FFT/IIR 精确对偶的常数状态序列建模

> ⚠️ **实验阶段声明**：本架构仅处于实验阶段，未经大规模训练验证。如果不是
> 真的想尝试非注意力方案，建议直接使用 Mamba 等成熟架构进行开发与模型设计。
> 本文所有"优于/劣于"的对照结论都标注了口径与局限；**本文的价值主张不是
> "更好的模型"，而是"在足够长的上下文上唯一成本可行的选择"**（§7）。

## 摘要

SimplCore 是一套**可训练、彻底非注意力**的序列建模架构。其核心算子 SGC
（Spectral Global Convolution，谱全局卷积）用阻尼振荡基字典上的因果谱卷积
替代 softmax 自注意力：训练走 FFT 整段形式（$O(D\log L)$），推理走二阶 IIR
递推（$O(1)$ 常数状态），**无位置嵌入表、无 KV 缓存、无最大长度截断**。
三条可机器验证的结构性质（全部有守护测试与 13 项数学证明支撑）：

1. **双形式精确对偶**：FFT 整段与 IIR 流式在 fp32 下逐位一致（守护测试锁定）；
2. **严格因果**：扰动未来 token，过去输出逐位不变（精确 0，非容差）；
3. **常数状态**：流式状态字节数与序列长度无关，1M 上下文部署 1.948 GiB
   （同规模 TS 的 KV cache ≈ 129 GB (120 GiB)，数量级差异）。

本文同时如实报告代价与负面结果：**逐字精确指向与联想召回（MQAR）是
SGC 的结构性劣势**（attention 1.000 / 0.500 vs SGC 0.398 / 0.3125），
根因是 key→value 绑定能力缺失而非感受野——已通过两条正交路线修复
（三层精确引用旁路 + 定容绑定记忆），修复后 MQAR **在噪声内追平**
attention（72% 参数，未超越）；聚合型任务上"等效 4–7B"的通过率优势
**未复现**（0.875 vs 0.938），降级为"约一半参数同等质量 + 成本优势"。
命题 5 的实测数字存在双口径勘误（§4.5）。**版本 0.1-α 是结构性质
（可证）与能力对照（有输有赢）严格分账的版本。**

## Abstract

SimplCore is a trainable, strictly attention-free sequence modeling stack.
SGC replaces softmax self-attention with causal spectral convolution over a
damped-oscillator basis dictionary: FFT parallel form for training
($O(D\log L)$), second-order IIR recurrence for inference ($O(1)$ state),
with no positional table, no KV cache, and no length cap. Three structural
properties are machine-verified: bitwise dual-form equivalence, strict
causality (future perturbation leaves the past bit-identical), and
length-independent streaming state (1.948 GiB @ 1M context vs ≈120 GiB KV).
We equally report the costs: exact pointing and MQAR association-recall are
structural weaknesses of the spectral-compression state (fixed by two
orthogonal routes: a three-tier exact-reference sidecar and bounded binding
memory — post-fix MQAR **ties** attention at 72% parameters, not beyond);
the aggregation-axis "equivalent 4–7B" pass-rate claim did not reproduce.
The correct claim is feasibility ("the only cost-viable choice beyond X
tokens"), not superiority. **This is an experimental release (0.1-α).**

## 1 引言

**命名纪律**：架构叫 **SimplCore**；SGC 只是 SimplCore 内部的一个 mixer
模块（`simplcore/core/arch/sgc.py`）。仓库另有 SRFN（selective SSM 谱卷积）
与 GatedSpectralConv2（GLA/Mamba2 式通道门）两个非注意力 mixer，同一套
config 切换。0.1-α 起对拟人化命名做了降调（`DeepReasoner→IterativeRefiner`、
`DreamingLoop→ReflectionLoop` 等，旧名保留为兼容别名），并明确：

| 是 | 不是 |
|----|------|
| 一套可运行的 PyTorch 非注意力序列混合器 | 已证明 SOTA 的模型 |
| 结构性质可机器验证（双形式/因果/常数状态） | 大规模训练验证过的系统 |
| O(1) 常驻状态 + O(N) 精确档案（Host RAM）的**分层**架构 | "O(1) 通吃"的系统 |
| engineering trade-off（效率轴赢、精确轴认输后修复） | "更好的模型" |

**仓库结构**（0.1-α 结构化拆分）：`core/` 按依赖方向分八个子包——
`arch`（mixer 数学）/`model`（配置与模型、**multimodal** 多模态入口）/
`memory`（记忆、**sidecar** 三层引用、**binding_memory** 定容绑定、
**dream/** 反思巩固、smm）/`dialogue`/`training`（data、train 子包）/
`eval`（评测与账本、**world_probe**）/`backend`/`deploy`；实验性模块
（jitrl/pilot/conditions_ledger）隔离在 `simplcore/lab/`；已退役的 Java
平行原型归档 `.archive/`。全量守护测试 **1127 通过 / 6 环境跳过 /
0 失败**（2026-10-31）。

## 2 记号与问题设定

与 v1.1 一致：因果语言建模，输入 token 流 $x_{1:L}$，模型每位置输出
$\logits_t$；"状态"指推理时需要常驻的中间量。对照基线为标准 Transformer
（TS）：KV cache 随 $L$ 线性增长，注意力为 $O(L^2)$ 两两交互。

## 3 方法

### 3.1 mixer 家族（全部非注意力）

| `mixer=` | 混合器 | 流式状态 | 训练 FLOPs 长程项 |
|---|---|---|---|
| `sgc`（默认） | 谱全局卷积 + FFT/深度卷积 + 内容门控 | O(1) IIR | $O(D\log L)$ |
| `sgc-r` | 谱专家路由（M 基分 G 组 + top-k 注入） | O(1)（与 sgc 逐字节同） | $O(G\cdot D\log L)$ |
| `srf` | SGC + selective SSM（输入相关 Δ） | O(1) chunked scan | $O(D\cdot L/\text{chunk})$ |

`attn`/`hybrid*` 取值**已删除**：`SimplCoreConfig.validate()` 只接受非注意力
mixer。四个参数级卫兵（全部 opt-in 默认关或默认安全）：

- **θ 钳制**（P1）：命题 5 误差界含 $1/|\sin\theta_m|$，θ 参数层钳制
  $[\theta_{\text{floor}}, \pi-10^{-3}]$，下界跟随 decay_max（logfreq 配对
  设计不被钳）；
- **evidence/uncertainty 闸门**：默认 OFF（无监督信号时是静默衰减器）；
- **物理世界模型**（`use_physics_world_model`，实验）：物理一致性判据，
  与 L2/L3 闸门正交的第三抗幻觉信号源——**tuan-2b 等全部预设不使用**；
- **decay_scale**（R3，实验）：分层多尺度衰减，层 $i$ 使用
  $\text{decay\_max}\cdot s^i$。

### 3.2 三层精确引用（L0/L1/L2）——「第几个字」的架构答案

逐字精确引用物理上需要 O(N) 存储（香农，无例外）。架构**不让 O(1) 做精确
检索**，分工为：

| 层 | 内容 | 存储 |
|----|------|------|
| **L0 常驻** | gist 匹配 + 指针头（`PointerHead`，$\approx2D^2$ 参数，与 N 无关） | O(1)，HBM |
| **L1 旁路** | 原文逐 token 精确副本 + 块 gist 索引（O(N/B)） | O(N)，Host RAM/SSD（4M tokens：int32 16 MiB） |
| **L2 取回** | O(1) 状态 → 指针分布 → top-k → 精确读取 → 零初始化注入 | 取回 O(k·B·D)，与 L 无关 |

**定位改口**：O(1) 状态的价值不是"存储器"，是**选择性寻址器**。
正式口径：**O(1) 常驻状态（HBM），O(N) 精确档案（主机内存/SSD）**——
档案不进 GCD，§5.12 的 HBM 账本不受影响。

### 3.3 定容绑定记忆（R1）——修复 key→value 绑定缺陷

MQAR 对照（§7.2）定位：SGC 缺的不是感受野，是**绑定能力**。第一版定容
记忆原型失败的真卡点在读取（槽内 $h_t$ 既当 key 又当 value）。R1 的修复：
**槽内 key/value 分开投影** + **局部窗口绑定写入**（宽 W 的相邻窗口前半进
key 侧、后半进 value 侧——"键 值"恰是相邻结构）+ FIFO 环形定容槽。
0.1-α 二期新增 **soft 可微门控写入**（`write_mode="soft"`）：逐槽 sigmoid
混合替代硬 argmin——修复 3000 步训练退化（不可微写入的槽位分配坍塌无
梯度信号）。

### 3.4 生态扩展（全部 opt-in 或独立入口）

**多模态**（`model/multimodal.py`）：图像 patchify / 音频帧下采样 → 软 token
拼接进无位置表序列（模态前缀因果在前）；`stream_step(inputs_embeds=…)`
提供模态前缀的 O(1) prefill；无模态时与纯文本路径**逐位一致**。
**世界模型容量探针**（`eval/world_probe.py`）：O(1) 递推状态的线性可读出
测量（fp64 闭式岭回归 + 置换/多数类基线）——测量而非宣称。
**SimplUnity**：冻结基座 + 路由门 + 多专家（负载均衡/z-loss），0.1-α 补齐
统一生成入口 `UnityModel.generate()`（O(1) 流式 MoE 采样）。
**物理世界模型**（实验）：物理一致性判据（连续性/守恒/运动学），
`tuan-2b` 等全部预设不使用。

## 4 理论结果

四条定理与七条命题的完整陈述见 `docs/SGC_CORE.md` 与
`simplcore/core/eval/mathproof.py`（13 项机器可验证证明：符号 + fp64）。
核心结论与 0.1-α 状态：

- **定理 1（双形式对偶）**：FFT 整段 == IIR 流式（fp32 短序列逐位一致，
  守护测试锁定；长序列偏差见 §5.4 勘误）。
- **定理 2（严格因果）**：流式路径未来泄漏逐位为 0。
- **定理 3（O(1) 状态）**：状态字节数 $4(2MD+(K{-}1)D+1)$ 与 $L$ 无关
  （M=64, D=1024, K=4 → 524.0 KiB/层；`mathproof` P12 硬断言）。
- **命题 5（误差界，0.1-α 双口径勘误后口径）**：舍入误差注入系数为 1
  （非 $\sin\theta_m$），故误差冲激响应为 $\psi_m[\tau]/|\sin\theta_m|$，
  界为 $\delta_{\max}/((1-\rho)|\sin\theta|)$。**口径纪律（0.1-α 二次审计
  发现并修正；logfreq M=64 默认网格实测 1/|sinθ| = 1/sin(π/8192) ≈ 2607.6，
  与 §5.4 保守性警告的 decay_max/π 口径一致）**：误差界必须用随 $|\tilde y|$ 重构的
  $\delta\le k\cdot u\cdot\max_t(|a||\tilde y_1|+\cdots)$（口径 B）——
  裸 $u$ 常数（口径 A）只在 $|\tilde y|\le1$ 成立，**不是界**。口径 B 实测：
  旧界（无 $1/|\sin\theta|$ 因子）在 legacy M=8 网格被违反 1.69×、
  修正界成立；论文旧数字 5.79×/5.13 来自 legacy M=8 + 未标口径（该网格
  已被构造期守护拒绝）。
- **P10（有损上限的适用范围修正）**：有限常驻状态对**无界** $L$ 的原文
  必有损（信息论，仍真）；但**任务绑定对数有限时，定容槽位存绑定副本
  同样是 O(1) 且无损**——"常数状态建模必输"不成立，"SGC 当前的谱压缩
  状态必输"才是正确表述（可修的工程缺陷）。

## 5 实验

### 5.1–5.3 结构性质验证

8 项数学核心全过（外部独立复现确认：双形式对偶、流式逐位零泄漏、
状态字节数四长度逐字节相同、训练缩放斜率 1.09 vs TS 2.35）。

### 5.4 双形式偏差（两代网格勘误后口径）

fp32 下 $L\ge2048$ 的双形式偏差随长度增长且**实测为重尾**
（`test_long_sequence_error.py`，**D=64**）：legacy 均匀网格（已弃用、
被构造期守护拒绝）在 $L=2048/4096/8192$ 为
$3.5/7.4/13\times10^{-4}$；现行 logfreq 默认网格 $L=2048$ 仅 $4.7\times10^{-5}$。
独立复现补测：L=64K 相对误差最坏 $\sim10^{-3}$、中位数 $10^{-5}$——
重尾印证"长序列 fp32 数值稳健性**未验证**"，4M 逐位一致主张以 Phase 0
conformance 实测为准，不得由短序列外推。

### 5.5 能力对照（X-axis-probe，可复现协议 `eval/axis_probe.py`）

| 任务 | attention | SGC | chance | 判定 |
|------|-----------|-----|--------|------|
| 精确指向（2 层） | 1.000 | 0.398 | 0.125 | **SGC 结构性劣势**（低于局部卷积 0.427） |
| MQAR G=4 | 0.500 | 0.3125 | 0.125 | 绑定缺陷定位 |
| MQAR G=4 + 绑定记忆 | 0.2422* | 0.2031 | 0.125 | SGC+记忆 **0.2578**：噪声内**追平**（72% 参数，未超越*） |
| 聚合 L=129 | 0.938 | 0.875 | 0.5 | attention 略优（"等效 4–7B"未复现） |
| 聚合 L=65 | 1.000 | 1.000 | 0.5 | 打平 |
| 精确计数 | 0.812/0.938 | 0.812/0.938 | 0.5 | 逐位打平 |
| 短程 LM（Markov） | ppl 14.63 | 15.71 | — | attention 优 7% |

\* 单批 128 样本、2 seed；attention seed SD≈0.066，差异 0.0156 在噪声内。
**判定纪律**：均值差落在对方 bootstrap CI 内 = 噪声内（`--seeds` + CI）。

### 5.12 部署账本

1M 上下文完整推理 **1.948 GiB ≤ 2 GB**（bf16 权重 1.888 + O(1) 流式状态
30.36 MiB + 融合单缓冲 workspace）；1M→8M 翻倍增量**恒为 0**。

### 5.14 端侧极限训练（外部实测，`docs/EDGE_TRAINING_REPORT.md`）

**2 GB 内存机器上 1.085 亿参数（dim=768/12 层/M=64）真实训练成功**：
Adafactor（factored 二阶矩，状态 O(m+n)，0.87 GB）+ 梯度检查点 +
micro-batch=1 + 梯度累积。300 步（repo 真实语料 309 万字符）：
train −63.8%、val −43.8%（同步降 = 真学习）、|g| 27.2→3.2 无爆炸；
**O(1) 状态字节 4,829,184 B 全程恒定（定理 3 的工程级证据）**；
双形式偏差 2.384e-05 @ L=96（训练不破坏对偶性）；L=1024 logits 全有限。
**诚实**：生成仍是噪声——0.00056 tok/param，距 Chinchilla 最优差四个
数量级（1/35700）；但生成出了语料真实术语"谱全局卷积"，统计结构已被
学习。意义：**150M @ 2GB 可训**（端侧/边缘叙事）、Adafactor 路径已
落地为 `--optimizer adafactor`（守护测试 `test_adafactor.py`）。

## 6 诚实边界：尚未证明的事（0.1-α 版）

1. 大规模训练有效性（§5.13 战役未运行）；
2. 与前沿线性架构的定量对比；
3. **长上下文 fp32 数值稳健性**（重尾实测，见 §5.4）；
4. **绑定记忆的检索质量数值**（机制与梯度已验证；MQAR 追平未超越，
   差异在噪声内）；
5. "越用越好"（R1 反思巩固的泛化判据已收紧为 held-out——
   `ReflectionConfig.heldout_eval`，gate 只证"不退步"）；
6. 多模态/物理世界模型/世界探针的**任务质量**（接口、形状、梯度、
   守护已立，未在真实数据训练验证）。

## 7 分轴等效参数量（推测；轴 C 降级、轴 D 论证修正）

| 轴 | 等效 TS 参数 | 状态 |
|----|--------------|------|
| A 原生 4M 训练 | 无上界（TS 算力不可行） | 解析账本 |
| B 通用短上下文 | 2–4B | 公开证据外推 |
| C 长上下文聚合 | ~~4–7B~~ → **降级：约一半参数同质量 + 成本优势** | **通过率优势未复现**（0.875 vs 0.938） |
| D 精确检索 | ~~必输（信息论）~~ → **谱压缩状态必输（可修）**；绑定记忆/旁路修复后 MQAR 追平 | MQAR + 精确指向实测 |
| E 域内代码 | +1–2B / 域外 ≈0 | verifier 边界 |

**价值定位（正式口径）**：SGC 的结构优势是**长上下文可行性**——1M 上下文
1.948 GiB vs TS KV cache ≈ 129 GB (120 GiB) 是数量级差异。宣称口径："在 X 长度以上唯一成本
可行的选择"，而非"更好的模型"。

## 8 修复路线分级（一致性判据：状态与 L 无关 + 无 O(L²) 交互）

| 级别 | 路线 | 状态 |
|------|------|------|
| A | R1 定容绑定记忆（soft 可微写入） | ✅ 已落地（追平未超越） |
| A | R2 hybrid 频率网格（满秩，cond 1e19→1.4） | ✅ 已落地 |
| A | R3 分层多尺度 decay_max | ✅ 已落地 |
| A | R4 记忆一等公民（可微写入门） | ⬜ 规划 |
| B | R5 滑窗注意力混合（窗口固定，状态仍 O(1)） | ⬜ fallback，分开宣称 |
| C | 全局注意力 / 增长 KV / RoPE | ❌ 背叛，不做 |

## 9 相关工作与生态

结构化 SSM/长卷积（S4/S5/Hyena/Mamba/GLA/CKConv）、混合架构
（LongNet/Jamba）、连续知识（CMT）——v1.1 相关工作节不变。0.1-α 新增
生态组件均遵循同一纪律：**性质可测、口径分账、负面结果照账**。

## 10 结论

SimplCore-0.1-α 的清单：精确对偶双形式、严格因果、常数状态、
三层精确引用、定容绑定记忆（追平未超越）、多模态入口、物理一致性
第三信号源、8 项数学核心外部复现为真。同等清晰的 is NOT 清单：
不是更好的模型（聚合未复现、精确轴先输后平）、长序列 fp32 稳健性
未验证、大规模训练有效性未证明。把它变成事实需要的是 §5.13 战役的
一次真实运行，而不是更多的文字。

## 附录 A：数据来源索引

一切实测数字可追溯：`scripts/audit_numbers_20261031.py`（双口径统一后）、
`eval/axis_probe.py`（X-axis-probe 协议）、`docs/REPRODUCTION_REPORT.md`
与 `docs/REPRODUCTION_REPORT_PHASE2.md`（复现报告一/二期，含作废修正）、
`docs/THEORETICAL_EDGE.md`（主张断言 34 项 + 实测对照 + 路线分级）。

## 附录 B：复现命令

```bash
# 全量守护（1127 通过 / 6 环境跳过）
python -m pytest -q
# 数学证明 13 项 / 主张断言 34 项 / 轴对照协议
python -m simplcore.core.eval.mathproof
python -m simplcore.core.eval.claims_audit --verbose
python -m simplcore.core.eval.axis_probe --seeds 0 1 2 3 4 --steps 3000
# 数值审计（命题 5 双口径 + 网格零点 + 状态字节账）
python scripts/audit_numbers_20261031.py
# 训练吞吐基准
python scripts/bench_train_throughput.py --seq-lens 512 2048
```
