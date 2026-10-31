# SGC 核心深度解析（SpectralGlobalConv）

本文档逐层拆解 `simplcore/core/arch/sgc.py::SpectralGlobalConv` 的实现。所有公式与代码对应，
无"文字创新"。

---

## 1. 为什么是"谱全局卷积"

标准因果自注意力的全局交互是 `softmax(QKᵀ)V`——对每一对 token 做内容相关归一化，
复杂度 `O(L²H)`，长序列下 KV 缓存随 `L` 线性膨胀。SGC 用一条**内容无关、频域固定**
的全局核替代它：全局感受野由核的形状决定，不随内容重算注意力，也不随 `L` 膨胀状态。

核由 **M 个阻尼振荡基**的线性组合给出：

```
ψ_m[τ] = ρ_m^τ · sin((τ + 1) · θ_m)        τ ≥ 0
K[τ, d] = Σ_m  coef[m, d] · ψ_m[τ]
```

- `ρ_m ∈ (0,1)`：衰减率（越大记越久）；`θ_m ∈ (0,π)`：角频率。
- `coef[m,d]`：每基每通道的混合系数。
- `τ` 是相对距离，核定义在**任意**长度上——没有 `max_len` 截断。

初始化：`ρ_m = exp(−1/λ_m)`，`λ_m` 在 `[1, decay_max]` 上**对数等距**，
一次性覆盖从 1 个 token 到 `decay_max` 的所有时间尺度；`θ_m` 在 `(0,π)` 等距。
`coef` 用小方差随机初始化（**不能为 0**，否则 `∂K/∂ρ = coef·∂ψ/∂ρ = 0`，
振荡器参数永远收不到梯度——这是一个被守护测试覆盖的初始化陷阱）。

> 这直接解决了 SGC v2 的"时间基秩瓶颈"：v2 的 `K = low@up` 秩 ≤ 8，整条长核只有
> 8 个独立时间基函数，无法按内容路由；SGC 的字典是带衰减先验的 M 维，默认 64/128。

---

## 2. 完整前向公式

```
h      = norm(x)                                  # RMSNorm / LayerNorm
u      = in_proj(h) ⊙ sigmoid(gate_proj(h))        # 内容门控（selective）
long   = fft_causal_conv(u, K)                     # 长程，O(L log L)
local  = depthwise_causal_conv(u, W_loc)           # 短程，O(L·K)
long   = long · evidence_gate(u)                   # L2 证据门（因果前缀能量）
long   = long · uncertainty_gate(long, local)      # L3 不确定门（长/短分歧）
y      = long + local + feed · u                    # feed: 逐通道瞬时项
out    = out_proj(y)
return x + out_scale · out                          # 残差 + 可学标量缩放
```

- `feed`（逐通道对角项，init=1）让 block 起点接近恒等，训练初期稳定。
- `out_scale`（标量，init=1）是 GPT-2 式残差输出缩放。
- **无 Q/K/V、无 softmax、无 RoPE、无位置嵌入表**。

---

## 3. 双形式：训练 FFT，推理 IIR

`ψ_m` 是二阶线性系统的冲激响应，因此同一算子有两个严格等价的实现。

### 3.1 并行 / 训练形式（`fft_causal_conv`）

把 `K` 显式构造为 `(L, D)`，零填充到 `N = 2L`（避免循环卷积混叠），频域相乘后逆变换：

```
cX = rfft(xw, n=2L, dim=1)
cK = rfft(kw, n=2L, dim=0).unsqueeze(0)
Y  = irfft(cX * cK, n=2L, dim=1)[:, :L, :]        # O(L log L)
```

`build_kernel(length)` 对任意 `length` 物化 `K`（`torch.arange(length)` 递推），
这就是"无 `max_len`"的来源。

### 3.2 流式 / 推理形式（`step`）

二阶 IIR 递推，每个基只有 2 个状态：

```
a_m = 2 ρ_m cos θ_m,  b_m = −ρ_m²,  c_m = sin θ_m
y_m[t] = a_m · y_m[t−1] + b_m · y_m[t−2] + c_m · u[t]
long[t] = Σ_m coef[m,:] · y_m[t]                   # O(1) 状态、O(1) 计算
```

- 流式常驻状态 = `2·M·D` 个浮点 + `(local_kernel−1)·D` 个卷积缓冲 + 1 个 `energy_sum`
  标量，**与序列长度无关**。
- 例：M=64, D=1024 → 每层 `2·64·1024 + 3·1024 + 1 = 134,145` 浮点 = 536,580 B ≈ **524.0 KiB**
  （fp32；`streaming_state_bytes()` 实测一致），24 层约 **12.29 MiB（12.88 MB）**（batch=1）。
  模型级常驻再加 CMT 压缩记忆 `(B,D)`=1024 浮点=4 KiB，即 **528.0 KiB**/层等效
  （`streaming_state_bytes()` 不含 memory，属模型级单份——两口径勿混）。
  （勘误 2026-10-28：本行旧版误写为"132 KB / 3.2 MB"——132 KB 实为
  `M=64, D=256` 的量级；M 因子曾被漏计。`scaling.stream_state_bytes`
  已同步修正，并由 mathproof P12 的 ≤2 GiB 硬断言守护。
  勘误 2026-10-31：旧版总数误算为"131,145 浮点 ≈ 512 KiB"——加法错误
  且把 `3·1024+1` 项漏在总数外；正确为 134,145 浮点 = 524.0 KiB，
  实现复现脚本 `scripts/audit_numbers_20261031.py`。）
- 实测 fp32 双形式误差（`M=64, D=64`，含门控；`tests/test_long_sequence_error.py`）。
  **v1.2 口径（logfreq 默认网格，2026-10-31）**：`L=64 1.9e-6`、`128 7.0e-6`、
  `256 1.3e-5`、`512 1.7e-5`、`1024 2.9e-5`、`2048 4.7e-5`——4e-5 打包阈值
  在 logfreq 网格下到 L=2048 才被越过（legacy 网格下 L=1024 即超阈）。
  **历史口径（legacy 均匀网格，已弃用）**：`L=17 6e-7`、`64 6e-7`、`129 7e-6`、
  `512 5e-5`、`1024 1.1e-4`、`2048 3.5e-4`、`4096 7.4e-4`、`8192 1.3e-3`。
  该网格因公共零点（tau=2M(M+1)-1）已被构造期拒绝，误差特征不可复现——
  数字必须绑定网格口径的原因即此。
  因此 **`<4e-5` 只适用于 `L ≲ 512` 的短序列**，不是全长度保证。误差随长基
  （`ρ→1`）累积，界与缓解手段见 `docs/NUMERICAL_STABILITY.md`；不要把这组数
  写成"全长度 fp32 等价"。

> 诚实边界：这里的"O(1) 显存"指**流式/分块推理的常驻状态**。整段前向（默认训练模式）
> 仍会把激活随 `L` 物化——这是 FFT 并行形式的固有代价。训练侧另有
> `--train-chunk` 路径：块内并行 FFT、块间用状态 `(y1, y2)` 的齐次响应续接，
> 激活降到 `O(chunk·D)`；块间按截断 BPTT detach（前向无损、梯度窗口 = chunk）。

---

## 4. 局部路径：深度因果卷积

长程字典擅长平滑长距结构，但不擅长尖锐局部结构。叠加宽度 `local_kernel`（默认 4）
的深度因果卷积 `groups=D`：

- 左填充保证因果；`K[0] = I`（init 时 `weight[:,0,-1]=1`，其余 0）让起点为恒等。
- 流式状态仅 `(local_kernel−1)·D` 个缓冲，仍与 `L` 无关。
- 同一实现 `depthwise_causal_conv(u, w, buf)` 同时服务并行（`buf=None`）与流式
  （携带 `buf`），流式只回传新位置的输出，避免重复发射缓冲位置。

---

## 5. 内容门控（selective）

```
g = sigmoid(W_g · norm(x))          # 逐位置、逐通道、随内容变化
u = W_in · norm(x) · g
```

有效核因此随内容变化（GLA / GSS 家族既有做法），针对"位置相对核无法按内容路由"的
已实测瓶颈，**不引入任何 Q/K/V softmax 注意力**，复杂度阶不变。
`sgc-nogate` 变体可关闭以做消融。

---

## 6. 四层抗幻觉闸门（L1–L3 在 mixer 内）

### L1 选择门（selective）
即 §5 的内容门控：决定"哪些通道/位置参与"。"该参与的没参与 / 不该参与的乱参与"由它拦截。

### L2 证据门（核级抑制，因果前缀）
证据定义为门控后输入的**因果前缀平均能量**：

```
c_t    = (1/t) · Σ_{s≤t} ||u_s||²
logit  = gain · log1p(c_t) + pos_gain · log1p(t) + bias
e_t    = sigmoid(logit)                # (0,1)
long_t ← e_t · long_t
```

- 响应形状**结构性**（证据越足越开），只有强度/偏置可学；用**标量参数**而非 `Linear`，
  避免全零权重让闸门退化成常数，也避免被上层 `_init_weights` 重施 normal 冲掉。
- 统计量是因果前缀量：并行路用 `cumsum`，流式路用运行和，两者一致
  （守护测试 `test_gates_stream_matches_parallel`）。
- 递推状态新增一个 `energy_sum` 标量（batch×4 字节，与 L 无关）；漏存它会导致 restore 后
  闸门从 0 重累积，流式不再逐位一致——`checkpoint_stream` 已补此标量。

### L3 不确定门（长短路径分歧）
长程谱路径与短程卷积是同一上下文的两个独立估计，分歧是不确定性代理：

```
d_t = ||long_t − local_t||₁ / (||long_t||₁ + ||local_t||₁ + ε)
w_t = 1 − sigmoid(γ) · d_t
long_t ← w_t · long_t
```

分歧大 → 压制长程 → 回退到局部证据。

### L4 拒答门（输出侧，在 `abstain.py`）
从最终隐状态预测"该位置回答是否被外部证据支撑"，低于阈值即拒答。详见
`ARCHITECTURE.md` §8。

> 诚实边界：四层闸门是**降低幻觉倾向的结构性约束**，不是"消除幻觉"的保证。
> `grounded` 标签必须来自外部证据（引文/检索/工具返回值），用模型自身生成标签
> 会退化成另一个置信度估计，闸门就白做了。

---

## 7. 复杂度与实测

| 序列长度 | SGC（单层, CPU） | 标准因果注意力 | 加速比 |
|----------|------------------|----------------|--------|
| 512 | 3.41 ms | 6.36 ms | 1.86× |
| 1024 | 6.18 ms | 21.83 ms | 3.53× |
| 2048 | 12.82 ms | 72.26 ms | 5.64× |

端到端拟合：时延 log-log 斜率 ≈ 0.997（≈线性），显存瞬时字节随 `L` 精确线性
（8 MB / 128 token）。流式常驻状态在 `L ∈ {64,256,1024,2048}` 上完全相同
（并与 `stream_bytes()` 解析式断言一致），每 token 字节按 `1/L` 下降。

> 诚实注记：多线程下斜率会在 0.05–1.12 抖动，因此上述测量在单线程 + 多次取最小值下完成，
> 报告斜率而非断言"线性/二次"。

---

## 8. 关键守护测试（结构保证）

- `test_mixer_math.py`（25 项）：数学正确、双形式等价、因果性、梯度可达。
- `test_gates_selective.py`（14 项）：L2/L3 不破坏 FFT↔IIR 对偶、梯度可达、
  只依赖因果前缀、流式==整段逐位、参数量 `bit-exact`。
- 因果性：扰动 `u[t+1:]` 不改变 `w[:t+1]`；扰动末位 token 不改变前文输出。
- 初始化陷阱：`coef` 不能为 0，否则振荡器参数梯度恒 0（有专门守护测试）。

---

## 9. SGC vs SGC v2 逐条对照

| 维度 | SGC v2 | SGC |
|------|--------|-----|
| 核参数化 | 任意低秩 `L×r @ r×D`，秩 ≤ 8 | M 个阻尼振荡基 + 有界局部 FIR |
| 时间基数量 | ≤ 8 | M（64/128），覆盖所有时间尺度 |
| 感受野 | `> max_len` 静默退化为固定窗 | 无 `max_len`；递推定义于任意 τ |
| 流式 | 无 | 二阶 IIR，O(1) 状态，与 L 无关 |
| 内容路由 | 无（位置相对核） | 输入相关门控（selective） |
| 位置嵌入 | 有 `pos_emb` 表 | 无 |
| 复杂度 | O(L log L) | O(L log L) 训练 / O(1) 每 token 流式 |

---

## 10. SGC-R：谱专家路由（MiTA 核心主张的非注意力融入）

`mixer="sgc-r"`（`RoutedSpectralGlobalConv`，2026-10-31 引入）把 MiTA
（Mixture-of-Top-k Attention, arXiv:2602.01219, NeurIPS 2026）的核心主张
以**纯 SGC 机制**吸收进来，不引入任何 token↔token 成对打分、softmax
相似度矩阵或 KV 载体 —— 不滑向 TS/hybrid。

### 10.1 MiTA 快权重分类法下的定位

MiTA 把高效序列混合统一为"让 N 宽输入诱导快权重 MLP 可扩展"的两条路径：
routing（稀疏路由）与 compression（压缩 shared expert）。据此：

| MiTA 概念 | SGC-R 对应物 |
|-----------|--------------|
| narrow shared expert（compression） | SGC 原有 FFT 全局通路（N → M 低秩振荡基，M=64/128） |
| deformable routed experts（routing） | M 个基按频率连续分 G 组"谱专家"，逐 token top-k 选择注入 |
| reusable top-k set | 路由门只作用于**输入注入端**；被切换掉的专家记忆在 O(1) 状态里齐次衰减、不丢失 |
| 涌现 token 剪枝 | 未路由组的注入为 0（结构性等效），负载均衡由 switch aux 正则 |

即：SGC 原本是分类法里"纯 compression 极端"（这正是长上下文精确检索弱、
需要 GATE-2 门的结构性原因），SGC-R 补齐 routing 分支。

### 10.2 机制与对偶性

* 路由器 `route_proj: D → G`，逐 token 产出 top-k hard 权重（集合内为
  renormalized softmax，集合外为 0）；训练用 straight-through（前向 hard、
  反向 dense softmax 梯度），`eval`/流式恒 pure hard —— 训练前向与 O(1)
  流式推理在 fp32 精度内是同一决策。
* 专家 g 的有效输入是 `u_g(t) = w_g(t)·u(t)`。因 `w_g(t)` 是**当前 token
  的因果函数**（与 content_gate/path_gate 同构），FFT 并行形式（对各
  `u_g` 因果卷积求和）与 IIR 流式形式（把 `w_g(t)·u(t)` 注入第 g 组递推、
  其余组纯齐次推进）是同一算子。
* O(1) 流式状态与 sgc **逐字节相同**（(B,M,D)×2 + conv_buf + 证据门标量）。

### 10.3 预算与纯度

* 训练核按组即建即弃：峰值核内存仍是 1×(T,D)（`train_window_bytes` 口径
  不变）；G 倍开销在 `u_g` 激活内存（属激活预算）。`--max-kernel-gib`
  闸门语义对 sgc-r 不变。
* purity 三层守卫（配置/实例/源码 AST）对 `sgc-r` 全量生效：`RoutedSpectralGlobalConv`
  无任何注意力符号，MoE 路由（token→固定专家集合的离散选择）不属于
  `ATTENTION_SYMBOLS` 判定口径。
* 负载均衡 aux（`router_aux_loss()`，switch 风格）由训练侧显式取用；
  model/train 主路径不自动加，现有训练行为逐位不变。

### 10.4 守护测试与实证边界

* `test_sgc_routed.py`（25 项，`test_mixer_math` 同等强度）：O(L²) 分组
  显式参考、整段↔流式逐位、分块↔整段逐位、状态不随 L 增长且字节与 sgc
  相同、严格因果（含路由决策因果性）、全参数梯度、ST 生效、aux 有限、
  注册表/解析参数账目/纯洁性接线、非法配置拒绝。
* **已验证**：上述数学与接线性质（fp32 CPU，atol=1e-4）。
* **未验证（诚实边界）**：SGC-R 的质量效果没有任何训练或对照实验支撑
  ——routing 分支是否如 MiTA 框架预测那样改善 GATE-2 型精确检索，
  需等预算 A/B（sgc vs sgc-r vs 对照臂）出数；MI250 主干战役（tuan-2b，
  2026-11-30）按原 `sgc` 配置训练，不受本次变更影响。SGC-R 的训练循环
  aux 接入与规模化训练列为遗留项（SimplUnity 12 月增强窗口）。
