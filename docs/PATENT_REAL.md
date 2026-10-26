# SimplCore 专利级创新点（仅真实存在于代码的组件）

> 诚实边界：本文件只记录 `simplcore_pg/` 中**真实实现且梯度可达**的组件。
> 早期 `PATENT_INNOVATIONS.md` 中描述的"基向量余弦调制公式"
> `modulated_base[j] = base[j]*cos(ω*π+j*0.01)+...` 在真实代码中**不存在**
> （`resonance.py` 无 `base/cos/sin/freqScale/phaseShift` 调制），已从本文档删除。
> 命名（"复数稀疏编码""自适应谱全局卷积"等）是修辞，底层为已知技术的变体组合。

---

## 1. ResonanceField (RFP) — 谱全局卷积处理器（可训练）

**真实实现**：`simplcore_pg/resonance.py`
**机制**：隐状态投影到频域多波通道 → 可学习高斯带通滤波 + 阻尼 + 多步波场演化 → 逆 FFT 回时域。
**可专利点（真实）**：
- 用 FFT 实现**全局卷积作为注意力的完全替代路径**（自 v2.0 起架构不再含任何自注意力），
  复杂度 O(n log n)，无 Q/K/V/RoPE/SDPA；
- 频率中心 `omega`、带宽 `sigma`、耦合系数 `beta`、输出缩放 `out_scale` 均为可学习参数，
  且多步演化经 `torch.utils.checkpoint` 包裹，**梯度真正到达 FFT 正向输入**（可训练）；
- 与局部算子 + 外部记忆组合于同一残差流，且全程可微。

**代码证据**：`ResonanceField.forward` 使用 `torch.fft`，参数 `omega/sigma/beta/out_scale` 与
`in_proj/out_proj` 均为 `nn.Parameter`；`smoke_test.py` 验证 `in_proj.weight.grad > 0`。

---

## 2. DynamicMemoryChannels (DMC) — 动态记忆通道

**真实实现**：`simplcore_pg/memory.py`
**机制**：固定大小可学习记忆库 `(C, S, hd)`，经 query/key/value 投影做内容寻址读取。
**可专利点（真实）**：
- 记忆库大小固定，不随序列长度增长（O(1) 空间）；
- 记忆库 `memory_bank` 为 `nn.Parameter`，训练时更新；
- 与 RFP 共同提供"全局依赖 + 固定记忆"，支撑长上下文流式。

**代码证据**：`smoke_test.py` 验证 `memory_bank.grad > 0`。

---

## 3. LocalInteraction — 原生局部交互层（替代自注意力）

**真实实现**：`simplcore_pg/localmix.py`
**机制**：因果深度可分离卷积（`CausalDepthwiseConv1d`，可带空洞 dilation）+ 可选门控线性递归
（`GatedLinearRecurrence`），门控融合，承担"邻近 token 混合"。
**可专利点（真实）**：
- 完全无注意力 / Q-K-V / RoPE / SDPA；局部依赖由卷积 + 线性递归承担，复杂度 O(n)；
- 因果性由左填充保证，无需任何位置编码；
- 与 RFP 全局卷积互补（局部 + 全局），构成无注意力架构的两条路径。

**代码证据**：`LocalInteraction.forward` 调用 `CausalDepthwiseConv1d` 与 `GatedLinearRecurrence`。

---

## 4. SimplCoreMoEBlock — 组合块（无注意力）

**真实实现**：`simplcore_pg/model_moe.py`
**机制**：`x → LocalInteraction → RFP → DMC → MoELayer`，各子层受残差 + RMSNorm 包裹。
**可专利点（真实）**：自研子层（LocalInteraction/RFP/DMC）默认启用并接入生产级 MoE，参数账一致；
自 v2.0 起彻底移除 SelfAttention。

---

## 5. 长上下文流式状态传递

**真实实现**：`simplcore_pg/model_stream.py`（`stream_init`/`stream_step`/`generate_stream`）
**机制**：跨 chunk 仅传递固定大小 `StreamState`（RFP 频域 + DMC 记忆），**无 KV 缓存**
（注意力已移除）。
**可专利点（真实）**：激活显存随序列长度常数增长的设计（见 `docs/LONG_CONTEXT_STREAMING.md`）。

---

## 不列入本文档的"文字创新"（已剔除）

- "复数稀疏编码 / Quantum-Inspired"：代码无复数语义态测量机制，仅为命名修辞；
- "基向量余弦调制公式"：代码中不存在；
- "零 PyTorch 依赖纯原生引擎"：主线依赖 torch；
- "512× 理论加速比"：无对应实测；
- "O(1) 持久记忆 mmap"：未在主线实现；
- 任何"首创/颠覆注意力"的叙事：RFP 是 S4/H3 类谱全局卷积的变体，LocalInteraction 是标准
  卷积 + 线性递归的组合；二者都是已知技术的 SimplCore 化实现，非凭空首创。

这些若未来真实实现，再补入本文档并附代码证据。
