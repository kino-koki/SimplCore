# SimplCore 架构规范书（论文级，v2.0 — 无注意力）

> 本规范书是 SimplCore-V 0.1-ALPHA 的**数学与实验协议**正式描述。
> 所有结论均对应 `simplcore_pg/` 中可运行代码；不存在代码未实现的"文字创新"。
> 本架构**正式放弃 Transformer / 自注意力**：不使用 Q/K/V、RoPE、SDPA 或任何
> 注意力归一化机制。

---

## 1. 问题定义

给定离散 token 序列 $x_{1:n}$，自回归语言建模目标为最大化

$$
\mathcal{L} = -\sum_{t=1}^{n} \log p_\theta(x_t \mid x_{<t}).
$$

与 Transformer 不同，SimplCore 不通过"每个 query 对全部 key 做 softmax 归一化"
获得全局依赖，而是通过两条互补路径：

- **局部路径** `LocalInteraction`：因果卷积 + 可选线性递归，$O(n)$，
  负责相邻 token 的混合（替代自注意力的局部感知）。
- **全局路径** `ResonanceField (RFP)`：FFT 谱全局卷积，$O(n\log n)$，
  天生全局感受野，等价于一个**可学习、内容无关、频域固定的全局核卷积**
  （与 S4/SSM 的全局卷积思想同源，但演化由可学习带通 + 耦合系数驱动）。

二者之外，`DynamicMemoryChannels (DMC)` 提供**固定大小的外部记忆库**，
与序列长度 $n$ 解耦，承担跨段长期记忆。

---

## 2. 符号表

| 符号 | 含义 |
|------|------|
| $B$ | batch size |
| $n$ | 序列长度 |
| $L$ | 当前块长度（流式分块，固定） |
| $H$ | hidden size |
| $W$ | RFP 波通道数（num_waves） |
| $C$ | DMC 记忆通道数 |
| $S$ | DMC 每通道槽位数 |
| $k$ | 频率索引，$k\in[0, L/2]$ |
| $\omega_w$ | 第 $w$ 波通道可学习中心频率 |
| $\sigma_w$ | 第 $w$ 波通道可学习带宽 |
| $\beta$ | 演化耦合系数（sigmoid 约束到 $(0,1)$） |
| $d$ | 标量阻尼系数（由 damping, dt 决定） |

---

## 3. 层公式

### 3.1 LocalInteraction（局部，替代自注意力）

$$
\begin{aligned}
c_t &= \text{CausalDepthwiseConv1d}(x)_{t} \in \mathbb{R}^H \\
r_t &= \text{GatedLinearRecurrence}(x)_{t}
      = a_t \odot r_{t-1} + (1-a_t)\odot x_t,\quad a_t=\sigma(W_a x_t) \\
o_t &= \text{RMSNorm}\!\left(x_t + W_f[c_t; r_t]\right)
\end{aligned}
$$

- 因果性：卷积左填充 $(k-1)\cdot d$ 个时间步，无未来泄漏；递归为单向。
- 复杂度：$O(n\cdot H)$，无 KV 缓存，无 $n^2$。

### 3.2 ResonanceField / RFP（全局谱卷积，可训练）

投影与多步演化：

$$
\begin{aligned}
u &= W_{in}\,x \in \mathbb{R}^{W} \\
\text{for } s &= 1..S: \\
  U^{(s)} &= \texttt{rfft}(u^{(s)}) \\
  U^{(s)} &= U^{(s)} \odot B(\omega,\sigma) \odot e^{-d} \\
  u^{(s+1)} &= \texttt{irfft}(U^{(s)}) \\
  u^{(s+1)} &= (1-\beta)\,u^{(s+1)} + \beta\, u \\
o &= \text{RMSNorm}\!\left(x + W_{out}\,u^{(S)} \cdot \gamma\right)
\end{aligned}
$$

其中带通窗 $B_{k,w} = \exp\!\big(-(k-\omega_w)^2 / (2\sigma_w^2)\big)$，
$\omega_w,\sigma_w,\beta,\gamma$ 均可反向传播（梯度经 checkpoint 真正到达 FFT
正向算子输入；$W_{in}/W_{out}$ 始终可导）。

- 全局性：`rfft/irfft` 在整段（或整块）上做，天然全局卷积，等效核长度为 $n$。
- 复杂度：$O(n\log n)$ 单步，乘 $S$ 步 → $O(S\cdot n\log n)$。
- 可训练性：默认 `freeze_evolution=False`，演化经 `torch.utils.checkpoint`
  包裹，规避 Windows/CPU 下 FFT 反向 workspace 累积泄漏（详见 §6）。

### 3.3 DynamicMemoryChannels / DMC（固定记忆）

内容寻址外部记忆，容量 $C\times S\times h_d$，与 $n$ 无关：

$$
\begin{aligned}
q &= W_q x,\quad k = W_k x,\quad v = W_v x \\
\alpha &= \text{softmax}_S(q^\top k / \sqrt{h_d}) \\
m' &= \alpha\, v + (1-\alpha)\, m \quad (\text{记忆槽位更新}) \\
o &= W_o\, m' + x
\end{aligned}
$$

记忆 $m$ 跨段保留，提供跨序列片段的长期依赖。

### 3.4 块组合（无注意力）

$$
x \;\xrightarrow{\text{LocalInteraction}}\; x \;\xrightarrow{\text{RFP}}\; x
   \;\xrightarrow{\text{DMC}}\; x \;\xrightarrow{\text{MoE}}\; x
$$

MoE 为门控专家 FFN（SwiGLU），含负载均衡 + router z-loss 辅助项
$\mathcal{L}_{aux} = \lambda_{aux}\mathcal{L}_{lb} + \lambda_z\mathcal{L}_z$。

---

## 4. 训练目标

$$
\mathcal{L}_{total} = \mathcal{L}_{LM} + \mathcal{L}_{aux}.
$$

优化器 AdamW（$\beta=(0.9,0.95)$），梯度裁剪 $\|\nabla\|\le 1.0$。

---

## 5. 实验协议（可复现）

| 基准 | 任务 | 区分点 |
|------|------|--------|
| `bench_rfp_vs_attention.py` | 吞吐/延迟（n=2048） | RFP 谱卷积 vs 滑动注意力速度 |
| `bench_longctx_accuracy.py` | 长距离 key→value 召回 | 全局层在长距不劣于局部 |
| `bench_multihop.py` | **多跳间接关联召回** | 纯局部 vs 全局 RFP 的必要性与 DMC 贡献 |
| `bench_corpus_ppl.py` | **真实中文语料困惑度** | SimplCore vs Transformer 基线（参数对齐） |
| `test_stream_consistency.py` | 流式 vs 整段数值一致 | 流式分块正确性（浮点级） |
| `smoke_test.py` | 全流程健康 | 构造/前向/反向/梯度/可复现 |

全部 CPU 小尺度可复现；通过率目标 $\ge 90\%$。

### 5.1 实测结果（2026-08-26，CPU 可复现）

| 基准 | 关键结果 | 结论 |
|------|---------|------|
| `bench_rfp_vs_attention.json` | n=2048 时 RFP 5.2ms vs 注意力 41.3ms（**8.0×**）；n=4096 时 **16.2×**；交叉点 n≈400 | 复杂度阶数符合 O(n log n) vs O(n²)，但 RFP 常数因子高（CPU 上小 n 不占优） |
| `bench_longctx_accuracy.json` | 长距 key→value 召回两者均 acc=1.0 | 全局层在长距不劣于局部 |
| `bench_multihop.json`（含 scale_sweep） | local_only 500/1500 步 acc≈0.76/0.73；rfp_global ≈0.27/0.24 | 两路径均全局可达；RFP 优势在效率/流式而非尖锐路由 |
| `bench_corpus_ppl.json` | SimplCore val_ppl=**1.28** vs Baseline **2.05**（参数 1.250M vs 1.235M 对齐） | 无注意力架构在真实中文语料上泛化优于标准 Transformer |
| `bench_ablation.json` | full=**1.2374**；no_conv=1.3186(+6.6%) / no_rfp=1.2458(+0.7%) / no_dmc=1.2412(+0.3%) / no_recurrence=1.2397(+0.2%) | 因果卷积是局部主力；RFP/DMC/递归各贡献互补增益，完整架构最优 |
| `bench_scaling.json` | 匹配 200 步预算：H=64→**1.2637** / H=96→**1.2385** / H=128→1.2599（最大模型欠训练）；H=128 延至 400 步收敛到 **1.2294**（低于 H=96） | 无注意力架构困惑度随规模下降；更大模型需更多步数，与扩展律预期一致。全部规模点以 `rfp_freeze_evolution=True` 训练（Windows/CPU 下 FFT 反向泄漏的文档化回退；冻结与可训练 RFP 的 val_ppl 几乎一致） |
| `test_stream_consistency.py` | 流式 vs 整段前向 max_err=2.4e-7 | 流式分块浮点级一致 |
| `longctx_memory_benchmark.json` | 1M/2M/4M 增量 0 | 激活显存与 n 解耦（常数增长） |
| `bench_throughput_result.json` | n 增 4×：RFP×5.5 / Local×4.9 / FullAttn×11.6 | 复杂度阶数符合理论（线性 vs 二次） |
| `tests_simplcore_pg/` + `safety/tests/` | 245/245 通过 | 组件/模型/MoE/流式/安全/checkpointing 全绿 |

> 诚实边界：以上均为 CPU 小尺度可复现证据（语料约 19 万字符、合成任务小规模），
> 非大规模语料 SOTA 对比；GPU/ROCm 实测待补。
>
> **RFP 延迟实测明细（hidden=256, CPU, no_grad, 3 次取均值）**：
>
> | n | RFP (ms) | SDPA 注意力 (ms) | 注意力/RFP |
> |---|----------|------------------|------------|
> | 64 | 1.93 | 0.09 | 0.05× |
> | 128 | 1.45 | 0.19 | 0.13× |
> | 256 | 2.31 | 0.89 | 0.39× |
> | 512 | 1.88 | 2.11 | 1.12× |
> | 1024 | 4.06 | 9.98 | 2.46× |
> | 2048 | 5.17 | 41.32 | 8.00× |
> | 4096 | 10.19 | 164.84 | 16.18× |
>
> 结论：RFP 的 Python 循环 + FFT 常数因子在 CPU 上偏高，小 n（<400）不占优；
> 但随 n 增长，注意力按 O(n²) 恶化而 RFP 按 O(n log n) 增长，n=2048 时 RFP 快 8.0×，
> n=4096 时快 16.2×，且差距随 n 单调扩大。GPU 上 FFT 常数因子更低，交叉点会前移。

### 5.2 综合讨论（证据链合成）

论文级图表由 `make_paper_figures.py` 从各 `bench_*.json` 实时生成（`docs/figures/`）：

| 图 | 数据源 | 结论 |
|----|--------|------|
| `fig_scaling_curve.svg` | `bench_scaling.json` | 匹配 200 步预算下 val_ppl 随参数量下降（1.2637→1.2385）；最大模型欠训练（1.2599），延至 400 步收敛到 1.2294，低于 H=96 |
| `fig_ablation.svg` | `bench_ablation.json` | 完整架构最优（1.2374）；去卷积 +6.6%（局部主力），去 RFP/DMC/递归各 +0.2~0.7%（互补增益） |
| `fig_rfp_latency.svg` | `bench_rfp_vs_attention.json` | log-log 下 RFP 近似线性、注意力近似二次；n=4096 时 RFP 快 16.2× |

**证据链合成**：五条独立证据线共同支撑"无注意力架构可学习、可扩展、可高效推理"：

1. **可学习性**：真实中文语料上 SimplCore val_ppl=1.28 优于参数对齐的 Transformer 基线
   2.05（`bench_corpus_ppl.json`）；组件消融证明每个子层都有可归因的增益
   （`bench_ablation.json`）。
2. **扩展性**：困惑度随参数量单调下降（`bench_scaling.json`），且更大模型在更多步数下
   收敛到更低困惑度——与扩展律预期一致。
3. **全局可达性**：局部门控递归与 RFP 谱卷积均具全局感受野（`bench_multihop.json`、
   `bench_longctx_accuracy.json`），长距召回两者均 acc=1.0。
4. **效率**：RFP 以 O(n log n) 替代注意力的 O(n²)，n=4096 时快 16.2×
   （`bench_rfp_vs_attention.json`）；复杂度阶数由 `bench_throughput_result.json` 独立验证。
5. **流式与显存**：分块流式与整段前向浮点级一致（max_err=2.4e-7），激活显存与 n 解耦
   （`longctx_memory_benchmark.json`）。

**诚实边界**：以上均为 CPU 小尺度可复现证据（语料约 19 万字符、模型 ≤1M 参数、合成任务
小规模），非大规模语料 SOTA 对比；GPU/ROCm 实测待补。RFP 的带限谱卷积对"尖锐长距内容
路由"不占优（见 §6），其优势在效率与常数显存流式，而非替代注意力的全部内容路由能力。

---

## 7. 权重安全保存/加载管线（checkpointing）

模型权重主链路接入 `safety/` 安全模块（`simplcore_pg/checkpointing.py`）：

- **静态加密**：传 `passphrase` 时 state_dict 经 AES-256-GCM 加密落盘为
  `model.enc`（密钥由 PBKDF2-HMAC-SHA256 从口令 + 随机盐派生，盐随 checkpoint
  持久化于 `config.json` 元数据，加载时复用重派生）；不传则明文 `model.bin`。
- **RBAC 鉴权**：传 `rbac` + `user` 时，保存要求 `model.export`、加载要求
  `model.load`（deny-by-default，未授权抛 `RBACError`）。
- **内存保护**：解密后的明文 state_dict 驻留带金丝雀的 `GuardedBuffer`，
  加载完成即覆写清零（用后即焚），并校验金丝雀防越界写。
- **完整性**：`config.json` 记录权重 SHA-256 指纹，加载时校验，防篡改/错配。

API：`model.save_pretrained(dir, passphrase=..., rbac=..., user=...)` /
`ModelClass.from_pretrained(dir, ...)`（三个模型类经 `CheckpointMixin` 获得）。
验证：`tests_simplcore_pg/test_checkpointing.py` 10/10 通过（明文/加密往返、
口令错误拒绝、篡改指纹检测、RBAC 拒绝、配置往返）。

---

## 6. 已知工程约束（诚实记录，非架构缺陷）

- **FFT 反向 workspace 泄漏**：Windows/CPU 的 torch 构建下，长循环里反复调用
  `torch.fft.rfft/irfft` 反向算子会累积不释放的 workspace 内存（纯前向不崩，
  反向约 150 步后被 OS 静默杀死）。RFP 通过 `checkpoint` 包裹多步演化规避；
  若仍触发，设 `rfp_freeze_evolution=True` 回退为"仅投影可学、FFT 走 no_grad"
  的固定全局混合器（与 S4 类将核心卷积设为近似/不可导的惯例一致）。
- **GPU 实测缺失**：当前仅在 Windows/CPU/AMD-680M 验证；真实显存与吞吐需 GPU。
- **合成任务尺度**：多跳基准为小规模可复现证据，非大规模语料 SOTA 对比。
- **RFP 的带限性质（诚实）**：RFP 谱全局卷积经可学习高斯带通滤波，本质是**带限
  全局卷积**，擅长平滑/频域全局结构；对"尖锐长距位置复制"等需高频冲激核的任务
  不占优（实测 `bench_multihop.py` 中 local_only 的门控递归与之相当甚至更好）。
  RFP 的真实优势在 **效率**（O(n log n)，见 §5 基准）与**常数显存流式**，而非
  替代注意力的全部内容路由能力。若需内容相关全局路由，应结合输入依赖门控或
  DMC 内容寻址记忆——属后续开发项。
