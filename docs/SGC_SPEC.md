# SGC 规格说明（Spectral Global Convolution, version 2.0）

本文档是 `simplcore/sgc/` 的规范与设计依据。所有数字均来自本仓库可复现的实测脚本，
标注为「实测」的结论可用文中给出的命令复现；标注为「待验证」的结论本章不写入正文。

---

## 1. 为什么需要 SGC

SGC 的前身（`verify_sgc/sgc_pytorch.py::SpectralGlobalConvV2`，下称 **v2**）是一个
用 FFT 实现的因果全局卷积：

```
y[t] = Σ_{s≤t} K[t−s] · x[s],      K = low(L, r) @ up(r, D)
```

它的效率优势是真实的（实测 L=2048 时比注意力快 5.6×，见 §6），但审计发现了三个
**结构性**缺陷；SGC 是针对这三个缺陷的重写，而不是调参。

### 缺陷 1：感受野被静默截断

v2 用 `low[:L] @ up` 构核，而 `low` 的行数固定为 `max_len`。当 `L > max_len` 时：

* 不报错、不警告；
* 实测（`.probe` 脚本，`max_len=16, L=48`）：算子退化为**宽度 16 的固定窗卷积**，
  与"全局卷积"的定义不符（`max|model − conv(16-tap)| = 1.9e-06`，即它精确等于
  截断核的卷积）。

这意味着 v2 的"全局感受野"完全依赖调用方把 `max_len` 设得足够大，是一个隐式契约。

### 缺陷 2：时间基秩瓶颈（最关键）

`K = low @ up` 使核矩阵 `K ∈ R^{L×D}` 的秩 ≤ `r`（默认 8）。实测：

```
rank param   1 -> rank(K^T) =   1
rank param   8 -> rank(K^T) =   8
rank param  64 -> rank(K^T) =  64
```

即**整条长 512/4096 的卷积核最多只有 8 个独立时间基函数**。这是 v2 在高熵真实语料上
"训练集拟合更深、测试集泛化更差"（train ppl 1.10 / test ppl 11.10）并且无法按内容
路由的**结构性**原因——8 维时间子空间不足以表达"按内容选择前文特定位置"。

### 缺陷 3：核谱每步重算

v2 每次前向都对核重新做一次 `rfft`（实测占卷积耗时 14.6%）。核与输入无关时可缓存。

### 缺陷 4：没有流式路径

v2 只有整段前向。任何"常数显存流式"的说法在 v2 里都没有对应实现。

---

## 2. SGC 的设计

### 2.1 位置路径：阻尼振荡基字典

核由 **M 个阻尼振荡基**的线性组合给出：

```
ψ_m[τ] = ρ_m^τ · sin((τ + 1) · θ_m),                 τ ≥ 0
K[τ, d] = Σ_m C[m, d] · ψ_m[τ]
```

参数只有 `M` 个衰减率 `ρ_m`、`M` 个角频率 `θ_m` 与 `M×D` 的混合系数 `C`。
初始化：`ρ_m = exp(−1/λ_m)`，`λ_m` 在 `[1, decay_max]` 上**对数等距**，因此该字典
一次性覆盖从 1 个 token 到 `decay_max` 的**所有时间尺度**；`θ_m` 在 `(0, π)` 上等距。
`C` 用小方差随机初始化（**不能为 0**，否则 `∂K/∂ρ = C·∂ψ/∂ρ = 0`，振荡器参数永远
收不到梯度——这是一个容易踩的初始化陷阱，已由
`test_oscillator_params_get_gradient_through_kernel` 守护）。

这直接解决缺陷 2：时间基不再是任意低秩子空间，而是**带衰减先验的 M 维字典**，
`M` 默认 64（1.3B/3B 配置为 128）。

### 2.2 对偶形式：训练用 FFT，推理用 O(1) 递推

`ψ_m` 是二阶线性系统的冲激响应，因此同一算子有两个严格等价的实现：

**并行 / 训练形式**（`forward`）：把 `K` 显式构造为 `(L, D)`，零填充到 `2L` 做 FFT 卷积：

```
y = irfft( rfft(x, 2L) · rfft(K_pad, 2L) )[:L]        O(L log L)
```

**流式 / 推理形式**（`step`）：二阶 IIR 递推，每个基只有 2 个状态：

```
a_m = 2 ρ_m cos θ_m,  b_m = −ρ_m²,  c_m = sin θ_m
y_m[t] = a_m · y_m[t−1] + b_m · y_m[t−2] + c_m · u[t]
long[t] = Σ_m C[m, :] · y_m[t]                        O(1) 状态、O(1) 计算
```

两者数值一致（实测 `max|Δ| < 4e-5`（fp32），见 `test_streaming_matches_parallel`，
覆盖 L ∈ {1, 4, 17, 64, 129}）。

流式常驻状态 = `2 · M · D` 个浮点 + `(local_kernel − 1) · D` 个缓冲，
**与序列长度无关**。例如 M=64、D=1024：每层 132 KB（fp32）、24 层约 3.2 MB（batch=1）。

> 诚实边界：这里的"O(1) 显存"指**流式/分块推理的常驻状态**。若把整段历史一次性物化
> 进前向（训练模式），激活仍随 `L` 增长——这是 FFT 并行形式的固有代价。

### 2.3 短程路径：有界深度卷积

长程字典擅长平滑的长距离结构，但不擅长**尖锐的局部**结构。因此叠加一个宽度
`local_kernel`（默认 4）的深度因果卷积（`groups = D`），以 `K[0] = I` 初始化。
它的流式状态只有 `(local_kernel − 1) · D` 个数，仍然与 `L` 无关。

### 2.4 内容门控（selective）

```
g = sigmoid(W_g · norm(x))          # 逐位置、逐通道、随内容变化
u = W_in · norm(x) · g              # 门控后的卷积输入
```

有效核因此随内容变化（GLA / GSS 家族的既有做法），直接针对"位置相对核无法按内容路由"
这一已实测瓶颈，且**不引入任何 Q/K/V softmax 注意力**，复杂度阶不变。

### 2.5 完整公式

```
h        = norm(x)
u        = W_in · h ⊙ sigmoid(W_g · h)                 # 内容门控
long     = causal_fft_conv(u, K)                       # 长程，O(L log L)
local    = depthwise_causal_conv(u, W_loc)             # 短程，O(L·K)
y        = long + local + feed ⊙ u                     # feed 为逐通道瞬时项
out      = W_out · y    (可选再乘输出门 sigmoid(W_og·h))
return   x + out_scale ⊙ out
```

**没有位置嵌入表**：因果卷积本身即编码位置，因此不存在 `max_len` 之类的长度上限，
训练长度之外的长度天然有定义（实测 L=1024 前向有限，见
`test_length_extrapolation_beyond_training_length`）。

### 2.6 四层闸门（抗幻觉）

内容门控（§2.4）解决"按内容选择参与"，但不解决"**当模型其实不知道时，不要编**"。
四层闸门逐层收紧，每层对应一种不同的失败模式：

| 层 | 名称 | 拦截的失败模式 | 作用位置 | 实现 |
|---|---|---|---|---|
| L1 | 选择门 selective | 该参与的没参与 / 不该参与的乱参与 | mixer 输入 | `SpectralGlobalConv.content_gate` |
| L2 | 证据门 evidence | 依据不存在的远端上下文编造 | mixer 谱基 | `SpectralGlobalConv.evidence_gate` |
| L3 | 不确定门 uncertainty | 高熵区域输出过度自信的确定值 | mixer 输出 | `SpectralGlobalConv.uncertainty_gate` |
| L4 | 拒答门 abstention | 无依据时强行给出答案 | 输出头 | `simplcore/sgc/abstain.py` |

**L2 证据门（核级抑制，因果前缀）**

证据定义为门控后输入的**因果前缀平均能量**：

```
c_t    = (1/t) · Σ_{s≤t} ||u_s||²
logit  = gain · log1p(c_t) + pos_gain · log1p(t) + bias
e_t    = sigmoid(logit)                # 逐位置标量，(0,1)
long_t ← e_t · long_t
```

* 闸门响应形状是**结构性的**（证据越足越开），只有强度/偏置可学 —— 用标量参数
  而非 `Linear`，避免全零权重让闸门退化成常数，也避免被上层 `_init_weights` 冲掉。
* **对偶形式不被破坏**：统计量是因果前缀量，平行路径用 `cumsum`、流式路径用
  运行和，两者一致（守护测试 `test_gates_stream_matches_parallel`）。

**L3 不确定门（长短路径分歧）**

长程谱路径与短程深度卷积是同一上下文的两个独立估计，其分歧是不确定性的代理：

```
d_t = ||long_t − local_t||₁ / (||long_t||₁ + ||local_t||₁ + ε)
w_t = 1 − sigmoid(γ) · d_t
long_t ← w_t · long_t
```

分歧大 → 压制长程 → 模型回退到局部证据。

**L4 拒答门（输出侧）**

从最终隐状态预测"该位置的回答是否被外部证据支撑"，低于阈值即拒答：

```
grounded_logit = Linear(dim→1)(hidden) · exp(logit_scale) + bias
L_cal = BCEWithLogits(grounded_logit, grounded)
推理： if sigmoid(grounded_logit) < τ → 拒答
```

偏置初始化 −2.0（sigmoid≈0.12），未训练时几乎不拒答。

**四条硬约束**（守护测试 `simplcore/sgc/tests/test_gates.py`，18 项）：

1. 每层**可消融**（`config.evidence_gate` / `uncertainty_gate` / `use_abstain`）；
2. L2/L3 **不破坏 FFT↔IIR 对偶形式**（流式 == 整段，fp32 `max|Δ| < 4e-5`）；
3. 全链路**严格因果**（未来扰动不改变过去）；
4. 闸门参数**梯度可达**（防静默零梯度）。

> **诚实边界**：四层闸门是**降低幻觉倾向的结构性约束**，不构成"消除幻觉"的保证。
> `grounded` 标签必须来自**外部证据**（引文、检索命中、工具返回值、人工标注）；
> 用模型自身生成标签会退化成另一个置信度估计，四层闸门就白做了。
> 未在带引用的评测集上实测前，**不得宣称抗幻觉有效**。

---

## 3. 与 SGC v2 的逐条对照

| 维度 | SGC v2 | SGC |
|------|--------|---------|
| 核参数化 | 任意低秩 `L×r @ r×D`，秩 ≤ 8 | M 个阻尼振荡基 + 有界局部 FIR |
| 时间基数量 | ≤ r（默认 8） | M（默认 64/128），覆盖所有时间尺度 |
| 感受野 | `> max_len` 静默退化为固定窗 | 无 `max_len`；递推定义于任意 τ |
| 流式 | 无 | 二阶 IIR，O(1) 状态，与 L 无关 |
| 内容路由 | 无（位置相对核） | 输入相关门控（selective） |
| 位置嵌入 | 有 `pos_emb` 表 | 无 |
| 核谱重算 | 每步重算全部 | 流式路径不需要 |
| 因果性测试 | 无 | 6 项（含"未来扰动不改变过去"） |
| 复杂度 | O(L log L) | O(L log L) 训练 / O(1)-每 token 流式 |

---

## 4. API

```python
from simplcore.sgc import SGCConfig, SGCModel, get_preset

cfg = get_preset("sgc-150m")        # 或 SGCConfig(vocab_size=..., dim=..., ...)
model = SGCModel(cfg)

logits           = model(idx)                       # (B, L, V)
logits, loss     = model(idx, targets)              # targets 中 -100 被忽略

# 流式推理：每 token 代价与显存都与上下文长度无关
state = model.init_stream(batch=1)
logits, state = model.stream_step(tokens[:, t], state)
out = model.generate(prompt, max_new_tokens=64, temperature=0.8, top_p=0.9)

model.stream_bytes(batch=1)          # 常驻流式状态的精确字节数
```

消融开关（`SGCConfig.mixer`）：`sgc`（完整）、`sgc-nogate`（去内容门控）、
`sgc-nolocal`（去短程卷积）、`attn`（同骨架因果注意力 + RoPE 基线）、
`hybrid` / `hybrid-gqa` / `hybrid-yarn`（**SGC Hybrid**：全局谱卷积 + 局部滑窗
注意力交替堆叠，`attn_every` 控制密度、`attn_window` 控制滑窗；`num_kv_heads`
启用 GQA 省 KV、`gate_ab` 启用 γ/β 双门 + 多头谱基、`rope_scale_strategy` /
`rope_scale_factor` 启用 YaRN/NTK 位置缩放外推。见 `simplcore/sgc/hybrid.py`
及 §4.2）。

### 4.1 SGC 混合架构（`mixer="hybrid"`）

谱全局卷积擅长平滑长程结构（谱带限），但对尖锐的局部 token 级路由（检索、指代）
不如局部注意力直接。SGC Hybrid 把两条路径**交替堆叠**（LongNet / Jamba 路线）：

```
层 i 按 (i+1) % attn_every 切换：
  命中 -> SlidingWindowAttention   O(L·W)   每 head 只看 W 个 key（因果滑窗 + RoPE）
  否则 -> SpectralGlobalConv2      O(L log L) 全局谱卷积 + 内容门控
```

* **不引入 O(L²) 全注意力，也不引入随 L 增长的 KV 缓存**：滑窗因果注意力的流式状态
  是 `(B, H, W, head_dim) × 2`（环形缓存 + 逐位重套 RoPE），**与 L 无关**；
  SGC 层仍 O(1)。`stream_bytes` 精确核算二者之和。
* **严格因果**继承（扰动未来，前文逐位不变，`test_hybrid.py` 守护）；
  窗口 `W >= L` 时退化为普通因果注意力（已实测逐位一致）。
* 预设 `hybrid-150m` / `hybrid-1_3b`（默认 `attn_every=4`，1/4 层用注意力）；
  参数量由 `analytic_hybrid_params` 解析并 bit-exact 守护。
* `SGCModel` 在 `mixer="hybrid"` 时自动路由到 `HybridBlock`，`init_stream`/
  `stream_step`/`stream_bytes` 无缝支持混合层（SGC 走 `step`、滑窗走 `stream_step`）。

### 4.2 SGC Hybrid 对齐当前最强注意力体系（`attention.py` + `attention_yarn.py` + `selective.py` + `rope_scales.py`）

四个结构升级，全部经 `SGCModel` 的 `mixer="hybrid"` 路由（新 config 字段
`num_kv_heads` / `gate_ab` / `rope_scale_strategy` / `rope_scale_factor`）：

* **GQA（Grouped-Query Attention，`attention.py`）**：滑窗注意力的 K/V 头数
  `num_kv_heads < num_heads`（整除），多组 query 共享 KV —— Llama2/3-70B 的标配。
  流式 KV 缓存只按 kv 头存（`(B, G, W, hd)×2`），省 `H/G` 倍字节；
  前向与 :class:`SlidingWindowAttention` 同骨架（合并 qkv 权重 1:1 可对照），
  流式 == 整段逐位一致（`test_attention.py` 8 项守护）。
* **门控增强多头谱基（`selective.py`）**：GLA/Mamba2 式 channel 独立 γ/β 双门
  （逐位置瞬时，不进核、不增状态，FFT 形式不变）+ 每 head 独立谱基系数组
  `coef (M, G, head_dim)`；O(1) 状态字节数与单头**完全相同**（`M·G·hd == M·D`），
  流式 == FFT 整段逐位一致（`test_selective.py` 6 项守护）。
* **YaRN/NTK 位置缩放外推（`rope_scales.py`）**：RoPE 表生成支持 `none` /
  `linear`（位置压缩 1/factor）/ `yarn`（NTK 平滑混合：低频保留、高频压缩
  `1/factor`，Pozzobon 2024 smooth 公式）。滑窗注意力 + 缩放表流式路径同步支持
  （环形 KV + 按绝对位置重套缩放 RoPE），**流式 == 整段逐位一致**
  （`test_rope_scales.py` 7 项守护）。
* **GQA + YaRN 组合层（`attention_yarn.py`，`GQASlidingWindowAttentionYaRN`）**：
  把 GQA 的合并 qkv 布局（`[q|k|v] = D + 2·G·hd`，KV 缓存按 `G` 头存，省
  `H/G` 倍）与 YaRN/NTK 逐分量位置缩放表**组合**——Llama3 系列的标配
  （GQA + 128K YaRN 外推）在 SGC 混合架构里的落地。路由规则：
  `rope_scale_strategy in ("linear","yarn")` 且 `num_kv_heads < num_heads`
  时走 `GQASlidingWindowAttentionYaRN`（而非之前的 MHA 布局 YaRN）；
  `strategy="none"` 且 `factor=1.0` 时逐位等于 `GQASlidingWindowAttention`，
  MHA 退化（`G=H`）时逐位等于 `SlidingWindowAttentionYaRN`（同权重 1:1）。
  流式路径按**绝对位置**取缩放表，环形 KV 只按 `G` 头存，
  **流式 == 整段逐位一致**（`test_attention_yarn.py` 10 项守护）。

**诚实边界**：四者均为**结构保证**（KV 省量精确核算、流式==整段逐位、因果、
参数量 bit-exact、外推表有界有限），具体困惑度收益需 A100 训练实测；
参数量由 `analytic_hybrid_params` / `analytic_gqa_attn_params` /
`analytic_gqa_yarn_attn_params` / `analytic_gated_spectral_params` 解析并
bit-exact 守护（`test_integration_2_1.py` 4 项集成测试覆盖 8 种组合，
`test_attention_yarn.py` 10 项专项守护）。

### 4.3 SGC Hybrid 无词表模式：可插拔编解码器 + GQA HF 权重迁移

**动机**：传统 Transformer 把 tokenize 词表（BPE/SentencePiece）烧死在架构里——
词表 ID 直接就是 embedding 索引，输出层必须与词表 1:1 绑定；换一个词表
就要重新初始化词表参数、重训。SGC Hybrid 把编解码器**从架构里摘出去**，
变成可插拔的一等组件：

* **`tokenization.py`**（纯 Python，零第三方）：
  * `Tokenizer` 统一接口：`encode(text) -> List[int]` / `decode(ids) -> str` /
    `size()` / `max_token_len()`；
  * `CharTokenizer`：Unicode 码点直映（默认字符集 ~ASCII 可打印 + CJK 标点，
    `size() < 300`，嵌入开销极小），无需外部词表文件；
  * `ByteTokenizer`：UTF-8 字节流，词表仅 256 + 2 特殊符，多字节 CJK 无损；
  * `BPETokenizer`：标准 byte-pair 算法（sennrich 2016），GPT-2 字节约定
    （字节 `b` → `chr(b)`），支持 `from_json` / `from_hf`（HF tokenizers 库
    导出的 `vocab.json` + `merges.txt` 双文件）加载，**字节流无损**
    （encode → token 字符串拼接 → 字节 → 源文本，逐位一致）；
* **`SGCConfig.from_codec(codec, ...)`**：自动填 `vocab_size = codec.size()`
  + `codec_kind` / `codec_kwargs`（检查点序列化还原 codec），模型 mixer/FFN
  结构对词表大小零假设；
* **`SGCModel.forward(text, codec=...)` / `generate_text(text, codec, ...)`**：
  文本输入路径（codec 编码 → ID → 模型）与 ID 输入 logit 逐位一致；
  生成在 ID 空间完成（O(1) 状态流式），最后 codec.decode 回文本；
* **同一套模型权重可切换字符/字节/BPE 序列**：不重新初始化词表参数，
  只需在推理时传入对应的 codec（嵌入层尺寸由 codec.size() 决定）。

**GQA HF 权重迁移（`hf_bridge.attention_gqa_warmstart`）**：
* Llama 风格独立 `q_proj`/`k_proj`/`v_proj` → GQA 合并 qkv（逐层搬运，
  `layer=li` 优先匹配该层键，找不到回退全局第一个匹配）；
* MHA→GQA（`G < H`）：k/v 段取前 `G*hd` 行（截断 + note 记录，record-not-corrupt）；
* GQA→GQA：k/v 段 1:1 搬运（逐位一致）；
* 形状错配一律 skip + 记录，绝不硬塞；
* `init_sgc_from_hf` 增 `cfg_override` 参数，`num_kv_heads` 或
  `rope_scale_strategy in ("linear","yarn")` 时自动触发 GQA 暖启动。

**O(1) 流式状态 checkpoint/restore（`SGCModel.checkpoint_stream` /
`restore_stream`）**：
* 把混合堆叠的 O(1) 流式状态（SGC 层 IIR 二阶递推 `LayerStreamState`
  + 滑窗注意力层环形 KV `dict`）序列化为 plain dict
  （`{"length": int, "layers": [per-layer dict or None]}`）；
* 可 `torch.save` / 写 safetensors 持久化"断点流"；
* `restore_stream` 重建后继续 `stream_step`，与不序列化直接生成逐位一致
  （`test_stream_checkpoint.py` 5 项守护）；
* GQA 环形 KV 缓存按 `num_kv_heads` 头存，checkpoint 后 dict 中
  `k/v` shape 第 1 维 = G（GQA 省 KV 在断点流路径同样保持）。

**诚实边界**：无词表模式是**结构保证**（编解码器可换、嵌入/头尺寸由 codec
决定、字节无损、参数量 bit-exact），具体困惑度收益需 A100 训练实测；
GQA 权重迁移是**初值迁移**（只改初值，不改结构），形状错配一律跳过
（`test_hf_bridge_gqa.py` 6 项 + `test_tokenization.py` 15 项 +
`test_stream_checkpoint.py` 5 项守护）。

---

## 5. 规模配置（1 × A100 80GB）

参数数为解析计算值（`python scripts/export_configs.py`），不含任何估算猜测。

| 预设 | dim | layers | M | 参数量 | bf16 权重 |
|------|-----|--------|---|--------|-----------|
| `sgc-150m` | 768 | 12 | 64 | 131.4 M | 0.26 GB |
| `sgc-350m` | 1024 | 20 | 64 | 348.8 M | 0.70 GB |
| `sgc-1_3b` | 2048 | 20 | 128 | 1.329 B | 2.66 GB |
| `sgc-3b` | 2560 | 32 | 128 | 3.239 B | 6.48 GB |
| `attn-150m` / `attn-350m` | 同宽同深 | | — | 137.8 M / 368.4 M | 对照基线 |

**先测再训**：在任何配置上运行前，先用
`python scripts/probe_batch.py --config configs/<preset>.json --seq-len 4096`
实测最大 micro-batch 与峰值显存，再据此设置 `--grad-accum`。凭算术猜测显存是长跑
在第 3000 步崩掉的典型原因。

---

## 6. 效率与复杂度（实测）

**单层对齐对比**（`verify_sgc/sgc_pytorch.py`，CPU，dim=256）：

| 序列长度 | SGC | 标准因果注意力 | 加速比 |
|----------|-----|----------------|--------|
| 512 | 3.41 ms | 6.36 ms | 1.86× |
| 1024 | 6.18 ms | 21.83 ms | 3.53× |
| 2048 | 12.82 ms | 72.26 ms | 5.64× |

**端到端 SGC 复杂度拟合**（`simplcore/sgc/eval.py`，dim=128×2 层，batch=8，
单线程、7 次取最小值降噪，fwd+bwd）：

| seq_len | 单步 ms | tokens/s | 记账峰值显存 MB | 参数/梯度/瞬时 MB |
|---------|---------|----------|------------------|-------------------|
| 128 | 167.72 | 6105 | 15.398 | 2.898 / 2.398 / 8.000 |
| 256 | 444.77 | 4605 | 27.898 | 2.898 / 2.398 / 16.000 |
| 512 | 841.77 | 4866 | 52.898 | 2.898 / 2.398 / 32.000 |
| 1024 | 1822.49 | 4495 | 102.898 | 2.898 / 2.398 / 64.000 |

* 时延 log-log 斜率 **0.9973（R²=0.9995）**；
* 显存 log-log 斜率 **0.9144（R²=0.9994）**，瞬时字节随 L **精确线性**（8 MB / 128 token）。

> 诚实注记：在**多线程**下同一配置拟合出的斜率在 0.05–1.12 之间抖动，
> 因此该测量必须单线程 + 多次取最小值才稳定。报告斜率而不是断言"线性/二次"，
> 正是因为这个原因。

**流式常驻状态**（`eval.py streaming`，dim=128×2 层，M=16）：

| 序列长度 | 常驻状态字节 | 每 token 字节 | 单步 ms |
|----------|--------------|----------------|---------|
| 64 | 35,840 | 373 | 14.1 |
| 256 | 35,840 | 124 | 11.3 |
| 1024 | 35,840 | 34 | 11.9 |
| 2048 | 35,840 | 17 | 15.2 |

状态字节在四个长度上**完全相同**（并与解析式 `stream_bytes()` 断言一致），
每 token 字节精确按 `1/L` 下降，单步时延斜率 0.0133（近平坦）。
这是"O(1) 状态流式"的**有效测量**，取代了 `verify_sgc` 中因
`GetProcessMemoryInfo` 失效而恒为 0 的旧"显存实测"。

---

## 7. 复现命令

```powershell
# 数学正确性 + 双形式等价 + 因果性 + 梯度可达（25 项）
python -m pytest simplcore/sgc/tests/test_mixer_math.py -q

# 真实语料构建（分词器 + 打包分片）
python scripts/build_corpus.py --parquet-dir data/raw/fineweb_zh --out data/processed/sgc_zh

# 目标机显存/吞吐标定
python scripts/probe_batch.py --config configs/sgc-1_3b.json --seq-len 4096

# 训练
python -m simplcore.sgc.train --config configs/sgc-1_3b.json --data-dir data/processed/sgc_zh `
  --seq-len 4096 --micro-batch 4 --global-batch-tokens 2000000 --max-steps 100000
```

---

## 8. 本地真实语料收敛验证（CPU，本机）

目的不是刷分，而是证明**这条链路真的能训练**：真实中文语料、真实 BPE 分词器、
真实文档级切分、真实梯度更新、真实验证困惑度。

**语料**（`scripts/build_corpus.py`，来源 FineWeb-Edu-Chinese 5 个分片 1.36 GB）：

| 划分 | 文档数 | token 数 | 4096 窗口数 |
|------|--------|----------|--------------|
| train | 1,264,508 | 467,718,144 | 114,189 |
| val | 2,539 | 913,408 | 223 |
| test | 2,539 | 929,792 | 227 |
| **合计** | **1,269,586** | **469,561,344** | **114,639** |

* 原始字符 850.22 M，压缩比 **1.8106 chars/token**（32k BPE，采样测得 1.8371）；
* 文档保留率 81.0%（丢弃原因：too_short 298,015 / too_noisy 159 / 其他 35）；
* **切分单位是整篇文档**（`"split_unit": "document"`），不存在窗口级泄漏；
* 分片 SHA-256 记入 `index.json`，数据集加载时校验。

**训练**（17,983,045 参数，dim=320×5 层，M=32，seq_len=256）：

| 已消费 token | train loss | val ppl |
|--------------|-----------|---------|
| 8,192 | 10.4378 | — |
| 409,600 | 9.3595 | 9,280.0 |
| 655,360 | 8.5763 | — |
| 737,280 | 8.3783 | — |

* 随机初始化的理论值 `ln(32000) = 10.37`，起点 10.4378 与之吻合；
* 吞吐中位数 **1,626 tokens/s**（CPU，8 核 16 线程，seq 256）；
* 报告见 `runs/local_proof/report.md` 与 `runs/local_proof/curve.svg`
  （由 `python scripts/report_run.py runs/local_proof` 生成）。

**这证明了什么**：链路可用、损失真实下降、梯度健康、语料与分词器正确。
**这不证明什么**：17.98M 参数、不到 1M token 的运行**不是**任何性能主张。
本机（AMD Radeon 680M 核显 + 13.75 GB 内存 + CPU 版 torch）的吞吐上限约
1.6k tokens/s，即每小时约 580 万 token——比 1.3B 模型所需的数据量低三个数量级。
真正的训练必须在 1×A100 80GB 上按 `docs/SGC_A100_RUNBOOK.md` 执行。

---

## 9. 尚未证明的部分（不得对外宣称）

1. **没有任何一次 1B 以上规模的实测训练**。1.329B/3.239B 配置只做了参数解析计算、
   构建与前向/反向冒烟；本机（AMD 680M 核显 + 13.75 GB 内存 + CPU 版 torch）物理上
   无法实例化 3B。A100 上的 loss 曲线、吞吐、显存均为**待测**。
2. **未与 S4 / Mamba / Hyena / GLA 等在 LRA 等公认基准上做公平对比**。当前的
   效率对比基线只有标准因果注意力。
3. **未在真实大规模语料上证明优于 Transformer**。本机可跑的规模只足以证明
   "能收敛、梯度健康、数学正确"，不足以证明任何 SOTA 主张。
4. **长上下文外推只有"输出有限"的证据**，没有质量证据（训练 4K、推理 32K 时
   困惑度是否保持，未测）。**工具已就位**（``simplcore/sgc/extrapolation.py``
   的 ``extrapolation_probe``：固定 L0 训练 + 1/2/4/8× 零微调评测 + O(1) 状态字节
   守护），但 A100 上的真实曲线仍待跑（CPU 预算下小模型不足以收敛到可解水平，
   见守护测试的诚实边界说明）。SGC Hybrid 的 YaRN/NTK 位置缩放
   （``rope_scales.py``，§4.2）为注意力层补上了外推的结构保证
   （缩放表有界有限 + 流式==整段逐位一致），但外推质量曲线仍需实测。
5. **``decay_max`` 的必要性未做消融**。默认 8192/16384 是按目标训练长度设定的先验，
   尚未验证更大或更小的取值对最终困惑度的影响。**工具已就位**
   （``decay_ablation``：扫描 decay_max → 训练困惑度 + 2×外推准确率/困惑度曲线），
   实测数值仍待 A100。
