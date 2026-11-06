# 常数状态非注意力序列模型的表达力边界：以 SimplCore / SGC 为例

> **版本与定位声明（0.2-β，ICML 2027 投稿准备稿）**
>
> 本文**不主张**「以谱卷积替代注意力」或「超越 Transformer」。本文的研究问题是：
> **在严格不引入 $N\times N$ 交互、不引入 KV cache、保持 $O(1)$ 常驻状态的前提下，
> 一类基于阻尼振荡基的静态谱卷积模型，能表达什么、不能表达什么，边界在哪里。**
>
> 本稿由 0.1-α 技术报告重构而来。重构的动因是：原稿以「架构 + 能力对照」为叙事，
> 而多轮独立实验（含本仓与仓库外白盒）反复得到同一结论——**该架构在内容寻址
> 与生成质量轴上的劣势是结构性的，不是调参可绕过的**。把这类结果如实组织成
> 「边界刻画」，比维持一个已被自己证伪的替代叙事更准确，也更有信息量。
>
> **证据纪律**：本文所有数字标注来源等级
> `[R]` 本仓可重跑、`[E]` 外部白盒复现（单 seed / 口径可能不同）、
> `[A]` 解析推导或账本、`[C]` 猜想或未验证主张。
> **`[C]` 一律不进摘要、不进结论、不当作实验证据。**
> 未规模化训练；本稿**不是**「已验证有效的语言模型」论文。

---

## 摘要

高效序列建模的一个核心问题是：常数状态的非注意力模型，其能力边界由什么决定？
我们以 SimplCore 的谱全局卷积（SGC）为对象，给出一组**结构性 + 实证性**的边界结果。

SGC 用阻尼振荡基上的因果卷积替代成对打分：训练走 FFT 整段形式
（$O(D\log L)$），推理走二阶 IIR 递推（$O(1)$ 状态），无位置表、无 KV cache。
三条结构性质可机器验证 `[R]`：双形式对偶（流式路径 fp32 逐位、整段路径容差内一致）、
严格因果、状态字节与 $L$ 无关。

在此之上，本文的核心结论是**四条边界**：

1. **函数类不匹配**（§4.1）：SGC 的核 $K(\tau)$ 只依赖相对位置，attention 的核
   $K(i,j)=f(q_i,k_j)$ 依赖成对内容。精确核 oracle 实测：即使核形状 100% 正确，
   中层注入余弦仍 $\approx 0$（$-0.03\sim0.06$）`[E]`——差距不是自由度不足，
   是函数类不同。
2. **静态核天花板已被吃满**（§4.3、§5.3）：在 1.7B-Instruct 的迁移拟合中，
   28 层末层残差流余弦 v4 $=0.2484$，静态输入无关因果卷积 oracle $=0.2476$，
   差 $0.0008$ `[E]`。单层同输入对照可达 $0.5430$ `[E]`，说明**误差逐层累积**
   而非单层无能；指标全面改善的同时生成仍是词堆砌、不成句 `[E]`。
3. **时变参数必然破坏 FFT 对偶**（§4.4、§5.5）：把 $\rho,\theta$ 改为状态反馈
   （SFSC）会把系统变成线性时变（LTV），冲激响应 $K(t,s)$ 非 Toeplitz；
   数值检验 `is_toeplitz=false`、单次平均核 FFT 近似相对误差 $0.972$ `[R]`。
   因此「保持对偶」与「核随上下文变化」互斥，后者只能走 $O(LM)$ 递推，
   $O(L\log L)$ 主张失效。
4. **内容寻址可以补进 O(1) 状态，但等参数下仍不敌 attention**（§5.5）：
   相位共振方案 FD-CAO 在单跳内容寻址任务上把静态核的 $0.4160$ 提到 $0.8672$，
   而等参数低秩 attention（$5{,}152$ vs $5{,}217$ 参数）为 $1.0000$ `[R]`。

此外，47 万参数原生 SGC 在本仓真实语料上完成 $3.07$M token 前哨训练 `[R]`：
val ppl $10.817$ 优于同预算 Transformer $11.281$，输出分布蒸馏进一步到 $10.217$；
但**三臂生成均不成句**，`SUPPORTED` 自动判据已人工降级为 PARTIAL。

本文的定位是**边界刻画 + 负面结果 + 可复现系统**，不是 SOTA 主张。
所有未验证事项集中在 §7，投稿前必须补齐。

## Abstract

What determines the capability boundary of constant-state, attention-free
sequence models? We study SimplCore's Spectral Global Convolution (SGC),
which replaces pairwise scoring with causal convolution over a damped-oscillator
basis: an FFT parallel form for training ($O(D\log L)$) and a second-order IIR
recurrence for inference ($O(1)$ state), with no positional table and no KV
cache. Three structural properties are machine-verified: dual-form equivalence
(bitwise on the streaming path, fp32 tolerance on the parallel path), strict
causality, and length-independent state.

Our central results are four **boundaries**. (1) Function-class mismatch: the
SGC kernel $K(\tau)$ is position-only whereas attention's $K(i,j)=f(q_i,k_j)$
is content-dependent; an exact-kernel oracle gives mid-layer injection cosine
$\approx 0$ even with a perfectly shaped kernel. (2) A saturated static-kernel
ceiling: on 1.7B-Instruct weight-transfer fitting, the 28-layer residual-stream
cosine reaches $0.2484$ against a $0.2476$ oracle, while a same-input
single-layer control reaches $0.5430$ — the loss is layer-wise accumulation,
and improved alignment does **not** translate into fluent generation.
(3) Time-varying oscillator parameters destroy FFT duality: an LTV system has
a non-Toeplitz impulse response (measured FFT-proxy relative error $0.972$),
so "context-dependent kernel" and "FFT/IIR duality" are mutually exclusive.
(4) Content addressing can be added to an $O(1)$ state — a phase-resonance
variant lifts single-hop accuracy from $0.4160$ to $0.8672$ — but an
equal-parameter low-rank attention baseline still reaches $1.0000$.

A 470K-parameter native SGC pilot ($3.07$M tokens on this repo's real text)
gives val ppl $10.817$ vs a same-budget Transformer's $11.281$ and $10.217$
with output-distribution distillation; **generation remains non-sentential on
all three arms**. This is a boundary-characterization and informative-negative-
result paper, not a superiority claim. All unverified items are listed in §7.

---

## 1 引言

### 1.1 研究问题

过去数年，常数状态序列模型（SSM、线性注意力、长卷积）主要在回答
「能不能以更低成本逼近 attention」。这类工作通常以 perplexity 或下游任务的
单点提升为成功判据，较少系统回答一个更基础的问题：

> **当架构被约束为「无 $N\times N$ 交互、无 KV cache、$O(1)$ 状态」时，
> 哪些能力是结构性可达的，哪些是结构性不可达的？边界由什么机制决定？**

这个问题有独立价值：它决定了 hybrid 设计中「哪些层必须保留 attention」、
迁移方案中「哪些能力不可能靠搬权重获得」，以及数值稳定性设计的优先级。
SimplCore 提供了一个干净的研究对象——它的核是**静态、输入无关、平移不变**
的阻尼振荡基，函数类边界比 selective SSM 或线性注意力更锐利，因而更适合
做边界刻画。

### 1.2 贡献

| # | 贡献 | 证据等级 |
|---|---|---|
| C1 | 三条可机器验证的结构性质及**分路径口径**修正（撤回「fp32 无条件逐位一致」） | `[R]` |
| C2 | 静态谱卷积与 attention 的**函数类不匹配**：输出「可加分离」的严格证明（命题 B1）+ 精确核 oracle 实证定位；推论统一解释各修复路线为何必然走向内容寻址 | `[A]`+`[E]` |
| C3 | 静态核天花板的精确测定 + 单层/多层分解（误差逐层累积，非单层无能） | `[E]` |
| C4 | 「时变参数 ⟹ 非 Toeplitz ⟹ FFT 对偶失效」的**引理 B4 严格证明** + 数值确认，及「对偶 ↔ 上下文相关核」互斥性 | `[R]`+`[A]` |
| C5 | 两个第一性原理猜想（SFSC / FD-CAO）的可证伪实现与六臂等骨架对照 | `[R]` |
| C6 | 47 万参数原生训练 + 输出分布蒸馏的三臂前哨，及「指标–能力脱钩」现象 | `[R]` |
| C7 | 频率网格塌缩的统一解释：**数值病态与表征容量塌缩是同一件事的两面** | `[R]`+`[A]` |
| C8 | 相关工作的逐条文献核实与统一分类表；**明确划出与已知表达力理论工作的关系，不主张首次发现** | `[A]` |

**非贡献（明确排除）**：本文不声称 SOTA、不声称大规模训练有效性、
不声称在任何轴上超越 attention、不声称 1M 上下文部署已实测。

### 1.3 定位纪律

| 是 | 不是 |
|---|---|
| 一套可运行的 PyTorch 非注意力序列混合器 | 已证明 SOTA 的模型 |
| 结构性质可机器验证（对偶/因果/常数状态） | 大规模训练验证过的系统 |
| 一类架构的**表达力边界刻画** | 「attention 的替代品」 |
| 效率轴上的解析账本 + 前哨级实证 | 部署级长上下文实测 |
| 负面结果如实归档（含被自己证伪的猜想） | 只报成功的 Marketing paper |

**命名纪律**：架构叫 **SimplCore**；SGC 是内部 mixer 模块
（`simplcore/core/arch/sgc.py`）。仓库另有 SRFN、`GatedSpectralConv2`
等非注意力 mixer，同一 config 切换；`attn`/`hybrid*` 取值已从
`SimplCoreConfig.validate()` 删除。

---

## 2 记号与问题设定

因果语言建模，输入 $x_{1:L}$，每位置输出 $\mathrm{logits}_t$。
「状态」指推理时需常驻的中间量。对照基线 TS 为标准 Transformer：
KV cache 随 $L$ 线性增长，注意力为 $O(L^2)$ 两两交互。

本文区分两类核：

- **静态谱核**（SGC）：$K_{\text{SGC}}(\tau)=\rho^\tau\sin((\tau+1)\theta)$，
  只依赖相对延迟 $\tau=t-s$，与输入内容无关；整段形式等价于 Toeplitz 卷积。
- **内容核**（attention）：$K_{\text{attn}}(i,j)=\mathrm{softmax}_j(q_i^\top k_j/\sqrt{d})$，
  依赖每一对 $(i,j)$ 的内容。

---

## 3 架构与结构性质

### 3.1 SGC 算子

| `mixer=` | 混合器 | 流式状态 | 训练长程项 |
|---|---|---|---|
| `sgc`（默认） | 谱全局卷积 + FFT / 深度卷积 + 内容门控 | O(1) IIR | $O(D\log L)$ |
| `sgc-r` | 谱专家路由（M 基分 G 组 + top-k 注入） | O(1)（与 sgc 逐字节同） | $O(G\cdot D\log L)$ |
| `srf` | SGC + selective 输入相关 Δ | O(1) chunked scan | $O(D\cdot L/\text{chunk})$ |

四个参数级卫兵（默认关或默认安全）：θ 钳制（P1）、evidence/uncertainty
闸门（默认 OFF）、物理世界模型（实验，预设均不使用）、`decay_scale`（R3，实验）。

### 3.2 频率网格与表征容量 `[R]`

默认 `logfreq` 网格把 $\theta_{\min}$ 设计在 $\pi/\mathrm{decay\_max}$
（$\mathrm{decay}=8192$ 时 $3.8\times10^{-4}$），低频端相邻基在有限 $\tau$
窗口内近似共线：$M=32$ 字典数值秩 **24/32**、`cond(\Psi)=1.92\times10^{16}$。
改为线性均匀 $\theta\in[0.3,\pi-0.3]$ 后秩满 **32/32**、cond **1.04**。
**零成本**：参数量、状态字节、FLOPs、流式 step 时延全部不变
（`test_linear_costs_nothing` / `test_default_mode_unchanged`）。

> **统一解释（C7）**：近零 θ 不只是数值病态（命题 5 的 $1/|\sin\theta|$ 发散），
> 它同时是**表征容量塌缩**。把 $\theta_{\min}$ 抬起来，数值误差与字典秩
> 一起改善——这解释了 §5.7 的重尾现象与 R2 hybrid 网格的动因是同一件事的两面。
>
> ⚠️ **口径限定**：`cond` 的绝对值依赖定义（SVD / 列归一化 Gram /
> fp32 未归一化 Gram，见 `docs/frequency_grid_fix.md` §3、§10）。
> 「linear 是最优解」只在已扫过的网格组合与对应口径下成立；含 $\rho$ 的
> 口径下线性均匀因 π/2 镜像冗余对未必优于其他网格，非对称 θ 网格仍待扫 `[C]`。

### 3.3 三条结构性质（分路径口径） `[R]`

- **定理 1（双形式对偶）**：流式 IIR 路径与闭式核 fp32 逐位一致；
  FFT 整段路径 fp32 为**容差内一致**（短序列 $<10^{-5}$，偏差随 $L$ 增长，
  §5.7），fp64 下数学等价（P01/P06）。
  **撤回**「fp32 无条件逐位一致」的旧表述。fp32 在 $L=64$ 即
  `bitwise=False`（rel_err $1.15\times10^{-5}$），显式 fp64 才 $2.43\times10^{-14}$。
- **定理 2（严格因果）**：流式路径未来扰动对过去输出逐位为 0。
- **定理 3（O(1) 状态）**：$4(2MD+(K{-}1)D+1)$ 与 $L$ 无关。
  **口径注**：$+1$ 是 L2 证据门开启时常驻的每序列 fp32 标量，**不是结构常数**；
  默认 OFF 时为 $4(2MD+(K{-}1)D)$（§5.8 的 $4{,}829{,}184$ B 即此口径）。

---

## 4 表达力边界：理论与结构性结果（核心章节）

### 4.1 函数类不匹配（边界一）

**命题 B1（严格证明，`[A]`）** 静态核输出可写成

$$
y_L=\underbrace{\sum_{\tau=1}^{L-1}K(\tau)u_{L-\tau}}_{\displaystyle A\;(\text{与 }q\text{ 无关})}
+\;\underbrace{K(0)E(q)}_{\displaystyle B(q)} ,
$$

即**历史摘要 $A$ 与 query 项 $B(q)$ 可加分离**。对任意线性 read-out $W$，
取两个 query $q_1,q_2$ 指向不同目标位置 $j_1\ne j_2$，若输出正确则

$$
WA+K(0)WE(q_1)=E(v_{j_1}),\qquad WA+K(0)WE(q_2)=E(v_{j_2}).
$$

相减得 $K(0)W(E(q_1)-E(q_2))=E(v_{j_1})-E(v_{j_2})$：左边与 value 赋值
**无关**，右边随 value 任意变化 → 矛盾。故在「key 线性无关、value 可独立取值、
线性 read-out」下，不存在与输入无关的 $K$ 能实现内容寻址。
完整证明、前提条件与适用范围见 `docs/EXPRESSIVITY_BOUNDARIES.md` §1。

> **推论 B1.1（解释性框架，非定理）** 要打破可加分离，输出必须含有
> $\sum_p\langle\phi(q),\psi(k_p)\rangle v_p$ 型的**成对匹配项**。
> 这把此前所有路线统一起来：加长记忆 / 调 $\rho$ / 正交网格 / 逐通道 `coef`
> 都只改固定的 $A$（❌）；输入相关门控若不产生 $(q,k_p)$ 匹配同样 ❌；
> 线性注意力 $S_t=\sum_s\varphi(k_s)v_s^\top$、CAS、FD-CAO、
> softmax attention 与 hybrid 则全部 ✅。**各修复路线必然汇合到内容寻址状态，
> 不是巧合，是 B1 的函数类后果。**

⚠️ **适用范围**：B1 严格成立于**单个静态核 mixer + 线性 read-out**。
向「含逐位置非线性 FFN 的深层堆叠」的推广**未在此证明**（逐位置非线性可产生
历史 token 之间的交叉项）。本文不声称「任意深度 SGC 都做不了内容寻址」；
实证与已知理论工作支持该推广方向，形式化仍列 §7.1(6)。同理，输入相关门控
（selective / SFSC）不受 B1 约束，受的是 §4.4 的 B4。

**实证定位** `[E]`（`docs/TS2SC_Q17_v4_final.md` §5）：把每层实测
attention profile 直接当作 FIR 核（**核形状 100% 正确**），28 层残差流末层
余弦仍只有 $0.2476$；中层注入余弦 $-0.03\sim0.06$。真实 attention 的输出
方向由 $q_i\cdot k_j$ 内容匹配决定，平均 profile 与它近乎正交。
**「对齐 attention 输出」这个目标本身存在很低的上界，与自由度、参数量、
网格质量无关。** 数值注脚（最有利设置下静态仍失败、仅乘积但抹平位置也失败、
$(q,k_p)$ 匹配加权塌到 0）见 `tools/boundary_probe/results/b1_corollary.json`。

**已知结果关系**：「静态/常数状态模型在联想回忆、复制与状态跟踪上弱于
attention」并非本文首次发现——Zoology (arXiv:2312.04927)、Repeat After Me
(arXiv:2402.01032)、The Illusion of State (arXiv:2404.08819, ICML 2024)
已给出实证与 $\mathsf{TC}^0$ 复杂度类结论。**本文的新增是把该结论归约到
SGC 输出的一个可检查代数性质并给出几行证明，再与 B4 绑定到同一条架构约束
（§6.3）。** 详见 `docs/EXPRESSIVITY_BOUNDARIES.md` §5 与 `docs/RELATED_WORK.md` §2.4。

推论：任何以「让 SGC 内部长得像 attention」为目标的迁移/蒸馏方案，
其可优化空间受 B1 约束；正确做法是改为**只对齐输出分布**（§5.4）。

### 4.2 误差逐层累积（边界二） `[E]`

静态核并非「单层无能」。同输入单层对照（`q17_oracle_head.py`）：

| 单层静态核注入 cos（同一 $x$） | |
|---|---:|
| 全局平均核（跨 16 头抹平） | $0.4478$ |
| 逐头核（16 头各自 profile） | $0.5430$ |

而 28 层累积后为 $0.2484$。**单层能到 0.54，是逐层误差把它压到 0.248。**
这一区分很重要：它把「静态核完全没用」这个过强结论修正为
「静态核在单层有足够的局部对齐度，但误差沿深度累积且无机制纠正」。

### 4.3 静态核天花板（边界三） `[E]`

在 Qwen3-1.7B-Instruct 的解析权重迁移中，以 `ck3` 为基线、唯一变量改为
逐 KV 头 LS 拟合的逐通道 `coef (M,D)`（`SGC.coef` 原生即 $(32,2048)$，
v1–v3 曾被 `view(M,1).expand(M,D)` 压成 32 个自由度）：

| 指标 | v1 | v3 | **v4** | 静态核 oracle |
|---|---:|---:|---:|---:|
| L0 注入 cos | 0.8191 | 0.9041 | **0.9205** | — |
| L4 饱和范数 | 94825 | 78233 | **66264** | — |
| L27 残差流 cos | 0.0518 | 0.2221 | **0.2484** | **0.2476** |

v4 与 oracle 差 $0.0008$，在数值噪声量级。**该上界已被吃满。**
逐头核在 holdout 上仅 $0.2774\to0.3068$（$+0.029$），远小于 in-sample
的 $+0.095$——头间模式不跨文本稳定，本身即说明它由内容决定、静态化不了。

**关键负面结果**：指标全面改善，生成仍是词堆砌
（v1 `IPHERIPHER…` → v3 `的\n\n好像…` → v4 `\nOkay发现\n嗯发现宫`），
不成句、不是回答。**残差流余弦是迁移的代理指标，不是语言能力指标**——
优化它不是在优化想要的东西（§6.1）。

预检还排除了「加 10 档 $\rho^\tau$ 衰减基」：可把 $\Sigma K/\Sigma p$
从 $0.178$ 拉回 $1.000$、中层核拟合 $0.15\to0.69$，但只修幅度（v3 RMS 已
标定到 $0.993$），方向补不上；有数值证据的无效改动已主动取消。

> ⚠️ 口径说明：$0.2484$ 略**高于** $0.2476$，说明该 oracle 是有限样本/有限层的
> 估计量而非形式化硬上界。**B1（§4.1）与 B4（§4.4）已给出严格证明；
> B2 / B3 仍是实证观察 + 有限样本估计量，不是定理。** 因此「天花板被吃满」
> 的准确表述是：在该 oracle 定义下可优化余量在数值噪声量级，
> **不是**「已证明该族模型表达力上界为 0.2476」。B1 向深层非线性网络的推广
> 与 B2/B3 的形式化仍列 §7.1(6)。见 `docs/EXPRESSIVITY_BOUNDARIES.md` §3。

### 4.4 时变参数必然破坏 FFT 对偶（边界四） `[R]`

为检验「让核随上下文呼吸」这一直觉，我们实现了状态反馈谱卷积（SFSC）：
$\rho_t,\theta_t$ 由状态指数移动平均经小矩阵调制，递推仍为二阶 IIR，
状态大小不变。猜想声称「双形式对偶仍成立，可视为时变 FFT 的递推版本」。

**该推理不成立。** 线性时变（LTV）系统的冲激响应 $K(t,s)$ 不再只依赖
$t-s$，即核矩阵**非 Toeplitz**，单次 FFT（频域逐点相乘）无法表达它。
`check_toeplitz()` 用随机时变 $\rho_t,\theta_t$ 直接构造 $K$：

```
toeplitz_mean_abs_deviation = 17.67       is_toeplitz = false
fft_avg_kernel_rel_error    = 0.9720      （用平均核做 FFT 卷积的相对误差）
K[5,0:6] = [-10.4376, 8.4158, 13.6868, -11.9257, 0.081, 15.9999]
```

同一相对延迟在不同绝对时刻的系数完全不同。**这不是实现 bug，是 LTV 的结构事实。**

**引理 B4（严格证明，`[A]`）** 对二阶递推 $y_t=a_t y_{t-1}+b_t y_{t-2}+c_t u_t$
（$a_t=2\rho_t\cos\theta_t,\;b_t=-\rho_t^2,\;c_t=\sin\theta_t$），单位脉冲响应满足
$K(s{+}1,s)=a_{s+1}c_s$、$K(s{+}2,s{+}1)=a_{s+2}c_{s+1}$。若核 Toeplitz，
同一延迟 $\tau=1$ 要求二者相等；状态反馈使 $a_t,c_t$ 随 $t$ 变化，一般不等，
故 $K(t,s)$ 非 $t-s$ 的函数，不存在单一频域滤波器 $\hat K$。$\square$
完整推导见 `docs/EXPRESSIVITY_BOUNDARIES.md` §2。

⚠️ **这也不是首次发现**：S5 (arXiv:2208.04933) 已指出「卷积视图要求系统时不变」，
Mamba (arXiv:2312.00752) 明确「选择性破坏纯卷积视图，必须用并行前缀扫描」。
本文新增的是**在 SGC 具体二阶递推上写成可检查的引理 + 数值确认**，
并把它与 B1 并列为同一条架构约束的两个后果（§6.3）。

直接后果：SFSC 不能走 FFT 整段训练路径，只能走 $O(LM)$ 递推 scan
（与 `srf` 同档）；**$O(L\log L)$ 主张失效**，定理 1 的对偶性不再成立
（除非把「对偶」重新定义为「同一递推的两种执行顺序」），
`backend/` 的 16-op FFT 契约与逐位一致性守护对时变核全部不适用。

> **推论（本文最硬的互斥性结论）**：要保住 FFT 对偶与 $O(L\log L)$，
> $\rho,\theta$ 就不能随 $t$ 变——那正好回到静态核（B1 约束）；
> 要核随上下文变，就得放弃对偶与 FFT 路径。**二者不可兼得。**

### 4.5 误差界与 θ 钳制口径 `[R]`

命题 5（0.1-α 双口径勘误后）：舍入误差注入系数为 1（非 $\sin\theta_m$），
误差冲激响应 $\psi_m[\tau]/|\sin\theta_m|$，界
$\delta_{\max}/((1-\rho)|\sin\theta|)$。**口径纪律**：必须用随
$|\tilde y|$ 重构的 $\delta\le k\cdot u\cdot\max_t(\cdots)$（口径 B）；
裸 $u$ 常数（口径 A）只在 $|\tilde y|\le1$ 成立，**不是界**。
口径 B 实测：旧界（无 $1/|\sin\theta|$）在 legacy M=8 网格被违反 $1.69\times$，
修正界成立；论文旧数字 $5.79\times/5.13$ 来自已被构造期守护拒绝的
legacy 网格且未标口径，**不再引用**。

**P1 卫兵口径修正**：θ 钳制上界 $\pi-10^{-3}$ 绝对；下界随
`decay_max` 让步 $\min(10^{-3},\pi/\mathrm{decay\_max})$，
**不提供固定 $10^{-3}$ 保证**。默认档误差界放大约 $2608\times$、
`tuan-2b`（$4\times10^6$）档约 $1.27\times10^6\times$；
`mathproof` P05 已改为按 $(64,8192,16384,4\times10^6)$ 逐档扫描并如实
打印各档 $1/|\sin\text{floor}|$，不再用小配置代表生产配置。

### 4.6 O(1) 状态的信息论边界（P10 修正） `[A]`

有限常驻状态对**无界** $L$ 的原文必有损（信息论，仍真）；
但**任务绑定对数有限时，定容槽位存绑定副本同样是 O(1) 且无损**。
所以正确表述不是「常数状态建模必输」，而是
**「SGC 当前的谱压缩状态在内容寻址上必输」**——这是可修的工程缺陷，
不是常数状态范式的信息论宿命。三层精确引用（L0 常驻 O(1) 指针 /
L1 旁路 O(N) Host RAM 精确档案 / L2 取回）即按此分工：
**O(1) 状态的价值不是「存储器」，是「选择性寻址器」**。

---

## 5 实验

### 5.0 证据分级（先看这张表）

| 等级 | 含义 | 本文对应内容 |
|---|---|---|
| `[R]` | 本仓有脚本 + 原始数据，可重跑 | 结构性质 / 网格秩 / 47 万参数三臂 / SFSC-FDCAO 六臂 / Toeplitz 检验 / B1 数值注脚 / 守护测试 |
| `[E]` | 仓库外白盒或迁移报告，口径可能不同 | 静态核天花板 / 单层 oracle / §5.6 能力对照表 / osc 周期轴收益 |
| `[A]` | 解析推导或账本，非部署实测 | 1.948 GiB @ 1M / KV 对比 / 斜率 / 命题 5 界 |
| `[C]` | 猜想、未验证、未复现 | 分轴等效参数量 / 4M 逐位 / 端侧规模化 / 多模态任务质量 |

> ⚠️ **必须点破的一处冲突**：0.1-α 曾把「osc 周期轴反超
> （SGC linear30 1500 步 $0.9844$ vs TS 3000 步 $0.8535$，
> bootstrap $+0.1309$ CI $[+0.0996,+0.1621]$）」当作正面结果 `[E]`。
> 本仓随后落地的 `eval/osc_probe.py` + `artifacts/osc_probe_raw.json`
> **未复现该收益**：7 种网格落在 $0.1094\sim0.1354$，全部**低于**
> 「复制上一个 token」平凡基线 $0.1575$；配对差 CI 跨 0
> （`docs/frequency_grid_fix.md` §9）。
> **因此本稿不把「周期轴反超 attention」列为已验证结论**，只保留为
> 「外部白盒单 seed 正面信号，本仓小预算前哨未复现，待按同等预算重跑」。
> 脚本到位 ≠ 结论到位——这条纪律适用于本文所有 `[E]` 项。

### 5.1 结构性质与守护测试 `[R]`

8 项数学核心全过（外部独立复现：双形式对偶、流式逐位零泄漏、
状态字节四长度逐字节相同、训练缩放斜率 $1.09$ vs TS $2.35$）。
仓库自报全量守护 **1127 通过 / 6 环境跳过 / 0 失败**（2026-10-31）；
本稿作者侧未在带 torch 的完整环境重跑该全量套件，见 §7。

### 5.2 频率网格：秩与条件数 `[R]`

| 网格 | 数值秩 | cond($\Psi$, SVD, L=512) | θ_min |
|---|---:|---:|---:|
| legacy 均匀 | 被构造期守护拒绝 | — | — |
| `logfreq`（默认） | 24/32 | $1.92\times10^{16}$ | $3.8\times10^{-4}$ |
| `hybrid` | 32/32 | $6.52$ | — |
| `linear` | 32/32 | **$1.04$** | $0.3397$ |

三种口径（SVD / 列归一化 Gram / fp32 未归一化 Gram）的对齐见
`docs/frequency_grid_fix.md` §3、§10；**改善方向一致，绝对值不可跨口径比较**。
`tests/test_linear_grid.py` 10 项守护（含零成本、FFT/IIR 对偶、跨块一致、
长长度核有限性）通过 `[R]`。

### 5.3 迁移拟合：静态核天花板 `[E]`

见 §4.2–4.3。全部数字来自 `docs/TS2SC_Q17_v4_final.md` 及其交付物
（`q17_oracle*.py`、`TS2SC_Q17/*.json`、168 个逐头 profile、ck4）。
**该实验是迁移拟合，不是语言建模训练**；生成失败是结论的一部分，不是缺陷。

### 5.4 原生训练与输出分布蒸馏（本轮新增） `[R]`

为回答「此前失败的是迁移还是训练本身」，在**本仓真实文本语料**
（700 个文件、$5.57$M 字符，字符级 $V=512$，train/val $=5.29$M/$278$K）
上做三臂等预算前哨（`docs/TRANSFER_DISTILL_REPORT.md`、
`tools/transfer_distill/`）。纯 NumPy 独立实现，无 torch：

| 臂 | 参数量 | val ppl | KL(T=1) vs teacher | 3-gram 重复率 |
|---|---:|---:|---:|---:|
| TS from scratch | 524,928 | 11.281 | 0.0000 | 0.057 |
| SGC from scratch | 436,096 | 10.817 | 0.7395 | 0.034 |
| SGC + 输出分布 KD | 436,096 | **10.217** | **0.4739** | 0.000 |

$3.07$M token/臂（3000 步 × B=16 × L=64），`T=5`，α 从 0.7 线性降到 0.3，
每位置蒸馏，teacher 冻结无反向。fp64 中心差分梯度检验通过；
FFT/IIR 双形式在训练后权重下 L=32/64/128/256 的 `max_abs`
$6.56\text{e-}7/9.54\text{e-}7/2.35\text{e-}6/5.98\text{e-}6$，
流式状态 **68,608 B 恒定** `[R]`。

**判定：PARTIAL，不是 SUPPORTED。** 自动判据
（`ratio\le1.10$ 且 rep3 改善）过松，人工降级理由：

1. **生成未过线**：三臂输出均为代码碎片堆砌，不成句
   （例：KD 臂 `for keyss\cposl |"public void testTimeMillibuditer...`）。
   top-k 采样下 rep3 接近 0 区分不了「连贯」与「不重复的词堆砌」。
2. **核心混淆未排除**：`sgc` scratch 本身（$10.817$）就优于 `ts`（$11.281$），
   KD 的 $-5.55\%$ 收益可能只是多了一个正则项。判别性对照应为
   **随机未训练 teacher**；该臂 `kdrand` 只跑到 1000 步（$16.10$ vs `kd` $13.03$），
   **未跑完，不能定论**，标 ⬜。
3. **预算不足**：$3.07$M token/臂，按 TS 臂参数的 Chinchilla $20\times$
   应为 $10.50$M，仅到 $29\%$；字符级 $V=512$、$L=64$、2 层。

**这条实验证伪了「SGC 从头训练始终失败」的说法**——失败的是
「拟合 attention」，不是训练本身。但它**不证明规模化有效**。

### 5.5 两个第一性原理猜想的判别性检验（本轮新增） `[R]`

`tools/sfsc_fdcao_probe/`，六臂共享 embedding/残差/分类头，只换 mixer，
纯 NumPy fp64 手写 BPTT，`D=64,M=16,B=32`，held-out 512，单 seed：

| 臂 | mixer 参数 | `bind1`（单跳，chance 1/8） | `bind`（两跳 MQAR） |
|---|---:|---:|---:|
| `static` | 1,056 | 0.4160 | 0.2422 |
| `sfsc` | 3,136 | 0.4141 | 0.2500 |
| `fdcao` | 5,217 | **0.8672** | 0.2402 |
| `fdcao_p` | 5,217 | 0.6113 | 0.7656 |
| `attn` | 17,440 | **1.0000** | 0.2617 |
| `attn_lr`（秩 8） | **5,152** | **1.0000** | 0.2402 |

六臂 fp64 梯度检验 6/6 PASS（最坏 $3.19\times10^{-4}$，阈值 $5\times10^{-4}$）。
另见 §4.4 的 Toeplitz 证伪。

**三处别误读**：

1. `bind` 上 `attn` 只有 0.2617 是**单层 attention 一跳不够**（任务需两跳），
   基线本身失效，**不是 FD-CAO 赢了**。
2. `fdcao_p` 把「相邻 (key,value) 成对」先验显式写进写入相位，
   是任务先验注入；换到 `bind1` 立刻从 0.7656 掉到 0.6113，自证。
3. `bind1` 才是有效上界：等参数 `attn_lr`（5,152 vs 5,217）**1.0000**
   vs FD-CAO **0.8672** → **超过静态核，没超过 attention**。

**SFSC 判定**：两个绑定任务均与静态核持平，且 §4.4 已证伪其 FFT 对偶主张；
`mod`（上下文调制）任务只跑 2/6 臂（$0.5000$ / $0.5352$），
且该任务正确标签只依赖 A/B 一阶占比、静态核全局平均本来就能算，
**不构成判别性检验**，需重做。**SFSC 目前无实证支持。**

**边界张力（如实）**：FD-CAO 的状态累积
$\sum_s\cos(\theta_m-\phi_s)\,v_s$ 是「把成对打分压缩到 M 个固定频率槽」，
与线性注意力式内容寻址状态同属一类概念；它守住了
「无 $N\times N$、无 KV cache、O(1) 状态」，但**没有守住
「完全不借鉴外部方案」**。作者需自行裁定纯粹性边界的口径。

### 5.6 能力对照与负面结果 `[E]`

0.1-α 的 X-axis-probe 表格（精确指向 1.000 vs 0.398；
MQAR G=4 0.500 vs 0.3125；聚合 L=129 0.938 vs 0.875；
短程 LM ppl 14.63 vs 15.71）**全部来自外部白盒复现**，
两代数字需在同一任务定义下重跑才能对齐。
`eval/axis_probe.py` 已补 `mqar_g4`（4 对 key→value、8 类、chance 0.125），
附录 B 命令现可跑出同口径前哨 `[R]`。

**已确认的负面结果**（不因叙事需要而删）：

- 精确指向 / MQAR / 联想召回是 SGC 的结构性劣势，根因是 key→value
  绑定能力缺失，**不是感受野不足**（加长记忆窗口至 $\tau=9999$ 无改善）。
- 定容绑定记忆（R1：key/value 分开投影 + soft 可微写入）在噪声内
  **追平** attention（72% 参数，单批 128 样本 / 2 seed，
  attention seed SD $\approx0.066$，差异 0.0156 在噪声内），**未超越**。
- 聚合型任务「等效 4–7B」的通过率优势**未复现**（0.875 vs 0.938）。
- 加长记忆 / 输入相关门控 / 频率作地址三条路线均已实测排除
  （Gram 非对角余弦 $0.98\sim0.9999$，基不可作地址）。

### 5.7 数值稳健性与部署账本 `[A]`

fp32 下 $L\ge2048$ 双形式偏差随长度增长且**实测重尾**
（`test_long_sequence_error.py`，**D=64**）：legacy 网格在
$L=2048/4096/8192$ 为 $3.5/7.4/13\times10^{-4}$；现行 logfreq 默认网格
$L=2048$ 仅 $4.7\times10^{-5}$ `[R]`。独立复现补测 L=64K 相对误差最坏
$\sim10^{-3}$、中位数 $10^{-5}$ `[E]`。**4M 逐位一致主张以 Phase 0
conformance 实测为准，不得由短序列外推** `[C]`。

解析账本：1M 上下文完整推理 $1.948$ GiB $\le2$ GB
（bf16 权重 $1.888$ + O(1) 流式状态 $30.36$ MiB + 融合单缓冲 workspace）；
1M→8M 翻倍增量恒为 0；同规模 TS KV cache $\approx129$ GB (120 GiB)。
**口径注**：$1.888$ GiB 权重对应约 10.1 亿参数的更大预设，
**不是** §5.8 的 1.085 亿参数（dim=768/12 层/M=64）；两节不同规模，
不可互相反推。**这是解析账本，不是部署实测** `[A]`。

### 5.8 端侧极限训练（外部实测） `[E]`

2 GB 内存机器上 1.085 亿参数（dim=768/12 层/M=64）真实训练成功：
Adafactor + 梯度检查点 + micro-batch=1 + 梯度累积；300 步
（repo 真实语料 309 万字符）train $-63.8\%$、val $-43.8\%$（同步降 = 真学习）、
$|g|$ $27.2\to3.2$ 无爆炸；O(1) 状态 $4{,}829{,}184$ B 全程恒定；
双形式偏差 $2.384\times10^{-5}$ @ L=96；L=1024 logits 全有限。

**诚实**：生成仍是噪声——$0.00056$ tok/param，距 Chinchilla 最优差四个
数量级（$1/35700$）；但生成出了语料真实术语「谱全局卷积」，统计结构已被学习。
意义限于「150M @ 2GB 可训」的端侧叙事，**不构成语言能力证据**。

---

## 6 讨论：边界意味着什么

### 6.1 指标—能力脱钩

本工作最一致的经验规律是：**代理指标的改善没有转化成语言能力**。
v1→v4 的 L27 余弦 $0.0518\to0.2484$、L0 $0.8191\to0.9205$、
饱和范数 $94825\to66264$ 全面改善，生成仍从死循环退化成词堆砌；
§5.4 的 val ppl $11.281\to10.217$、KL $0.7395\to0.4739$ 同样伴随
三臂均不成句。

因此本文主张：**评估非注意力架构时，残差流余弦与 ppl 都不能单独作为
「可用」的判据**；必须同时报告生成可读性与任务级指标。
这也解释了为什么「对齐 attention 输出」在 B1 下是个错误目标——
它优化的是可测但不想要的东西。

### 6.2 成本轴与质量轴必须分账

SGC 唯一**稳定成立**的优势是成本：`O(1)$ 状态、`O(L\log L)$ 静态核路径、
解析账本上 1M 上下文 $1.948$ GiB vs KV $120$ GiB（§5.7 `[A]`）、
训练吞吐斜率 $1.09$ vs TS $2.35$ `[E]`。
质量轴上，本文所有有效对照里**没有任何一项超过等参数 attention**。
把两轴分开陈述，比用「等效 4–7B」之类的推测去补质量轴更诚实——
该推测已被 0.875 vs 0.938 证伪，本稿将其整体移出结论与摘要。

### 6.3 纯粹性的代价

§4.4 给出一个清晰的互斥：**FFT 对偶 / $O(L\log L)$ ↔ 核随上下文变化**。
§5.5 给出另一个：**守住无内容寻址状态 ↔ 等参数下追不上 attention**。
这两条不是「还没调好」，是架构约束的可验证后果。
若目标是「在内容寻址轴上不输 attention」，则必须引入内容依赖的写入机制
（FD-CAO 或 hybrid），并明确接受它带来的口径变化。
本文不替读者做这个取舍，只把代价列清楚。

### 6.4 对 hybrid 设计的含义

B1 与 §5.5 合起来给出一个可操作的分工依据：
周期/连续信号与长程衰减是平移不变核的天然主场（**但注意 §5.0：
该轴的「反超」证据本仓未复现，目前只能作为待验证假设** `[C]`），
内容寻址轴必须交给内容依赖机制。
这为「少量 attention 层 + 多数 SGC 层」的 hybrid 提供了理论动机，
但本文**不主张** hybrid 已被验证——它是 §7 的待补实验。

---

## 7 局限与未验证事项（投稿前硬门槛）

> 本节按 ICML 评审口径组织。**以下每一项在投稿前必须补齐或在文中如实标注为未验证；
> 不补齐则不应投稿。**

### 7.1 实验严谨性 🔴

1. **全部主实验单 seed**，无误差棒、无 seed 间方差、无配对 bootstrap。
   §5.4 / §5.5 / §5.6 均需 $\ge5$ seed + CI。
2. **无同 token 预算的语言建模主实验**：需 $\ge10$M token、
   同参数 Transformer / Mamba2 / GLA 等强基线、val ppl + 生成可读性双主指标。
   §5.4 的 $3.07$M 前哨仅到 Chinchilla $20\times$ 的 $29\%$。
3. **判别性对照未完成**：`kdrand`（随机 teacher）只到 1000 步；
   `mod` 剩 4 臂、`osc` 六臂全未跑；SFSC/FD-CAO 未测流式 IIR 一致性、
   4M 长上下文、吞吐。
4. **`mod` 任务设计失效**：一阶统计可解，需重做为二阶/层次调制任务。
5. **FD-CAO 的 key→θ 动态映射未实现**（当前 $\theta_m$ 为固定可学参数），
   容量上界可能被低估；多层混合堆叠未验证。
6. **形式化工作部分完成**：B1（可加分离，§4.1）与 B4（LTV 非 Toeplitz，§4.4）
   已给出严格证明并配数值注脚（`docs/EXPRESSIVITY_BOUNDARIES.md`）。
   **仍待**：① B1 向深层 + 逐位置非线性 FFN 的推广；② B2/B3 的严格陈述
   或明确标注为有限样本估计量；③ 推论 B1.1 的必要性证明；
   ④ 与 $\mathsf{TC}^0$ 复杂度类结果的正式对接。补齐前不得把 B2/B3 写成定理。
7. **本稿作者侧未重跑全量 pytest**（沙盒无 torch / 无 GPU）；
   1127 通过为仓库自报数字。

### 7.2 口径与一致性 🟠

8. 白盒实现（`tools/transfer_distill/`、`tools/sfsc_fdcao_probe/`）与
   上游 PyTorch 算子**不是同一份代码**：核归一化
   （$\rho^\tau\sin((\tau+1)\theta)$ vs 含 $1/\sin\theta$ 的旧白盒）、
   ρ 网格、θ 上界、linear 栅格均不同。训练数字不可跨实现互换。
9. 条件数绝对值依赖口径（SVD / Gram / fp32 数值条件数），见 §3.2。
10. §5.6 的能力对照表与 osc 收益均为 `[E]`，本仓 `osc_probe` 未复现；
    两代数字需在同一任务定义下重跑对齐。
11. §5.7 / §5.8 是不同规模预设，参数量不可互推。

### 7.3 定位与写作 🟡

12. **相关工作：文献条目已核，定量对标未做**。15 条代表工作的标题 +
    arXiv ID + 年份/venue 已逐条核实并列入 `docs/RELATED_WORK.md` 的统一分类表；
    核不到明确编号的条目（如 CKConv）一律不列。**仍待**：
    ① 逐条核对正式出版版本与 BibTeX 字段；② ICML 2025–2026 同类工作的系统检索；
    ③ **等参数 / 等算力定量对标实验**（当前完全缺失，P0）。
    见 §8 与 `docs/RELATED_WORK.md` §3。
13. **技术报告无公开 DOI/arXiv**：若作为「已发表文献」引用会触发不可检索风险。
    本稿定位为「本文提出的架构 + 可复现仓库」，附录 A 全部指向仓内文件。
14. **AI 使用声明**：见附录 D，投稿前按目标 venue CFP 填写并放到正确位置。
15. **可复现包**：匿名 repo、requirements、固定 commit hash、
    多 seed 一键脚本、compute 报告（CPU/GPU 小时数）。

### 7.4 明确不做的事

- 不把 `[C]` 级推测写进摘要或结论；
- 不用「等效 4–7B」「唯一成本可行」「超越 attention」等表述；
- 不把「指标改善」包装成「语言能力」；
- 不把「脚本到位」写成「结论到位」；
- 不引全局 attention / 增长 KV / RoPE 作为本架构的修复路线。

---

## 8 相关工作与待补对标

`docs/RELATED_WORK.md` 给出**逐条核实**的统一分类表（15 条，每条含标题 +
arXiv ID + 年份/venue），分类轴为本文四条边界：状态是否常数、核是否输入相关、
训练路径是否可用 FFT/卷积、是否具备 $(q,k)$ 内容寻址。核不到明确 arXiv ID 的
条目一律不列，以避免幽灵文献。

| 类别 | 已核实代表工作（arXiv ID） | 与本文关系 |
|---|---|---|
| 结构化 SSM / 长卷积 | S4 (2111.00396)、S5 (2208.04933)、Hyena (2302.10866) | 同为 LTI / 输入调制核；S5 已指出「卷积视图要求时不变」，是引理 B4 的已知来源 |
| 选择性 / 线性注意力状态 | Mamba (2312.00752)、Mamba-2/SSD (2405.21060, ICML 2024)、GLA (2312.06635)、DeltaNet (2406.06484)、Linear Transformer (2006.16236, ICML 2020)、RWKV (2305.13048, EMNLP 2023)、TTT (2407.04620) | 全部通过输入相关状态更新获得内容寻址，代价是放弃 LTI 卷积视图；SSD 证明一类 SSM 与线性注意力形式等价，支撑推论 B1.1 |
| 混合与长上下文 | LongNet (2307.02486)、Jamba (2403.19887) | §6.4 的分工依据；本文不主张 hybrid 已验证 |
| 表达力边界的理论与评测 | Zoology/MQAR (2312.04927, ICLR 2024)、Repeat After Me (2402.01032)、The Illusion of State (2404.08819, ICML 2024) | **与本文最易混淆的一组**：这些工作已给出「静态/常数状态模型在联想回忆、复制、状态跟踪上弱于 attention」的实证与 $\mathsf{TC}^0$ 复杂度类结论 |

**novelty 边界（必须正面写清）**：本文**不主张首次发现**上述任何一条边界。
理论贡献是三点：**① 形式化收紧**——把已知结论归约到 SGC 输出的可加分离性质
（命题 B1）与二阶递推的非 Toeplitz 性质（引理 B4），给出可检查的证明；
**② 绑定**——把两条边界绑到同一条架构约束，得到「FFT 对偶 ↔ 上下文相关核」
互斥性（§6.3），这是单看任一已知结果都得不到的；
**③ 落地证据链**——在一个具体架构上把理论、精确核 oracle、$0.2484$ 天花板、
47 万参数三臂、SFSC/FD-CAO 六臂连成可复现的一条链。
不可辩护的表述（「首次证明静态卷积做不了内容寻址」「首次发现时变核无 FFT 对偶」
「证明表达力上界为 0.2476」）一律不用。详见 `docs/EXPRESSIVITY_BOUNDARIES.md` §5。

**明确缺口**：上表是**定性分类 + 文献核实**，不是实验结果表。
**等参数 / 等算力定量对标（SGC vs Mamba-2 / GLA / DeltaNet / Hyena / TS，
同 token 预算下 ppl + 生成质量）完全未做**，是投稿前的 P0 硬门槛（§7.3(12)、
附录 C）。在补齐之前，本节不构成「已与前沿工作对标」的证据。

---

## 9 结论

本文把 SimplCore / SGC 从「注意力替代方案」重新定位为
**常数状态非注意力序列模型的表达力边界研究对象**，并给出四条边界：

1. 静态平移不变核与内容依赖核是不同函数类：输出的可加分离性使内容寻址
   结构性不可达（命题 B1，严格证明），「对齐 attention 输出」这个目标本身
   有很低的上界（精确核 oracle 中层余弦 $\approx0$）；
2. 静态核天花板在 1.7B 迁移中被吃满（$0.2484$ vs $0.2476$），
   单层可达 $0.5430$ 说明瓶颈是逐层累积而非单层无能，
   且指标改善不转化为生成能力；
3. 让核随上下文变化必然破坏 FFT 对偶（非 Toeplitz，FFT 代理误差 $0.972$），
   「对偶 / $O(L\log L)$」与「上下文相关核」互斥；
4. 内容寻址可补进 O(1) 状态（单跳 $0.4160\to0.8672$），
   但等参数下仍不敌 attention（$1.0000$）。

同时，三条结构性质可机器验证，47 万参数原生训练前哨证明
「训练本身可学」而「拟合 attention 失败」是两件事；
所有生成结果仍不成句。

**同等清晰的 is NOT 清单**：不是更好的模型、不是规模化验证过的系统、
不是 SOTA、不是「超越 attention」、不是 hybrid 已验证、
不是 1M 上下文部署实测、不是形式化的表达力上界定理。

本文的价值在于把一类架构「能做什么、不能做什么、代价是什么」说清楚，
并把每一条失败留成可复现的证据。把它变成规模化事实，需要的是
§7 清单里的真实运行，而不是更多的文字。

---

## 附录 A：证据来源索引（全部指向仓内文件）

| 内容 | 路径 |
|---|---|
| 数学证明 13 项 / 主张断言 34 项 | `simplcore/core/eval/mathproof.py`、`eval/claims_audit.py` |
| 双形式 / 数值审计 / 状态字节账 | `scripts/audit_numbers_20261031.py`、`test_long_sequence_error.py` |
| 频率网格秩、条件数、零成本守护 | `docs/frequency_grid_fix.md`、`tests/test_linear_grid.py` |
| 静态核天花板 / 单层 oracle / v4 | `docs/TS2SC_Q17_v4_final.md`、`TS2SC_Q17/*.json`、`q17_oracle*.py` |
| 47 万参数三臂训练 | `docs/TRANSFER_DISTILL_REPORT.md`、`tools/transfer_distill/results/` |
| SFSC / FD-CAO 六臂、Toeplitz 证伪 | `docs/SFSC_FDCAO_PILOT.md`、`tools/sfsc_fdcao_probe/results/` |
| osc 未复现收益的原始结果 | `artifacts/osc_probe_raw.json`、`docs/frequency_grid_fix.md` §9 |
| 修复落地状态（哪些修了、哪些没修） | `docs/upstream_fixlist_applied.md`、`docs/FIX_STATUS_20261105.md` |
| 端侧训练 / 部署账本 | `docs/EDGE_TRAINING_REPORT.md`、`docs/AMD680M_migration_feasibility.md` |
| 能力对照协议 | `eval/axis_probe.py`（含 `mqar_g4`）、`eval/osc_probe.py` |
| **B1 / B4 形式化证明 + 适用范围 + 已知工作关系** | `docs/EXPRESSIVITY_BOUNDARIES.md` |
| **B1 可加分离数值注脚** | `tools/boundary_probe/b1_additive_separation.py`、`results/b1_corollary.json` |
| **相关工作逐条核实与统一分类表** | `docs/RELATED_WORK.md` |

> 前一版本（0.1-α 原稿）保留于 `docs/SimplCore_Paper_v0.1alpha.md`，
> 供口径追溯；本稿为替换后的主稿。

## 附录 B：复现命令

```bash
# 结构性质与守护（需 torch / 完整依赖）
python -m pytest -q
python -m simplcore.core.eval.mathproof
python -m simplcore.core.eval.claims_audit --verbose
python scripts/audit_numbers_20261031.py
python scripts/bench_train_throughput.py --seq-lens 512 2048

# 能力对照（含 mqar_g4）/ 网格前哨
python -m simplcore.core.eval.axis_probe --seeds 0 1 2 3 4 --steps 3000
python -m simplcore.core.eval.osc_probe

# 47 万参数三臂（纯 NumPy，无 torch）
cd tools/transfer_distill && python gradcheck64.py
python run.py --arm ts  --total 3000 --chunk 500 --tag ts  --lr 6e-4   # ×6
python run.py --arm sgc --total 3000 --chunk 500 --tag sgc --lr 6e-4   # ×6
python run.py --arm kd  --total 3000 --chunk 500 --tag kd --teacher ts --lr 6e-4  # ×6
python run.py --final

# SFSC / FD-CAO 六臂（纯 NumPy，断点续跑）
cd tools/sfsc_fdcao_probe
python sfsc_fdcao.py --mode gradcheck     # 6/6 PASS
python sfsc_fdcao.py --mode toeplitz      # is_toeplitz=false
python sfsc_fdcao.py --mode all           # 全任务 × 全臂 × seed 0/1/2

# B1 可加分离下界数值注脚（纯 NumPy，秒级）
cd tools/boundary_probe && python b1_additive_separation.py
```

## 附录 C：投稿前待办清单（ICML 2027）

| 优先级 | 事项 | 状态 |
|---|---|---|
| P0 | ≥5 seed + 配对 bootstrap（§5.4/§5.5/§5.6） | ⬜ |
| P0 | 同 token 预算语言建模主实验（≥10M token）+ 强基线 | ⬜ |
| P0 | 相关工作逐条核 DOI / arXiv ID，补统一对照表 | 🟡 **部分完成**：15 条标题+arXiv ID+venue 已核（`docs/RELATED_WORK.md`）；正式出版版本/BibTeX、ICML 2025–26 检索、**定量对标实验**仍待 |
| P0 | B1–B3 的形式化证明或理论对接 | 🟡 **部分完成**：B1 / B4 已严格证明 + 数值注脚（`docs/EXPRESSIVITY_BOUNDARIES.md`）；B1 深层推广、B2/B3、B1.1 必要性、TC0 对接仍待 |
| P0 | **等参数 / 等算力定量对标实验**（SGC vs Mamba-2 / GLA / DeltaNet / Hyena / TS） | ⬜ |
| P1 | `kdrand` 补完 3000 步；`mod` 重做并重跑六臂；`osc` 六臂 | ⬜ |
| P1 | FD-CAO key→θ 动态映射、多层堆叠、流式一致性 | ⬜ |
| P1 | 全量 pytest 在带 torch 环境重跑并记录版本 | ⬜ |
| P1 | 匿名可复现包 + 固定 commit + compute 报告 | ⬜ |
| P2 | 附录 D 的 AI 使用声明按当年 CFP 填写 | ⬜ |
| P2 | 内部预审 + rebuttal 预案 | ⬜ |

**当前状态：不宜直接投稿。** 详见 §7。

## 附录 D：AI 使用声明模板（投稿前按目标 venue CFP 填写）

> **人工智能使用声明（占位，投稿前填写）**
>
> 本研究的研究问题、架构猜想、模块设计与全部结论判断均由作者提出。
> 作者在实现阶段使用了生成式 AI 工具（工具名称 / 版本 / 访问日期：______），
> 具体用于：① 参考实现的编码辅助；② 实验脚本与统计图画图；
> ③ 文稿初稿的语言组织与结构整理。生成式 AI **未**用于确定研究问题、
> 提出架构猜想、生成实验数据、判断实验结论或生成参考文献。
>
> 作者对 AI 生成的全部代码逐项复核：算子经 fp64 有限差分梯度检验，
> 全部实验在公开环境重新执行并留存原始结果文件，全部引用经
> DOI / 数据库逐条核验。作者对论文全部内容承担完全责任。
> 代码与原始数据见 ______。
>
> 按 ICML 惯例，与**方法实现 / 实验设置**相关的 LLM 使用应写在
> experimental setup；纯语法润色一般无需单独声明。
> 以投稿当年 CfP 为准，不得照抄往届条款文本。
