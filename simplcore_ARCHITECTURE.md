# SimplCore AI 架构技术文档（v2.0 — 无注意力）

**最后更新**: 2026-08-24
**作者**: SimplCore 开发组（六安市青奈科技有限责任公司）
**修订说明**: v2.0 起**正式放弃 Transformer / 自注意力路线**。所有 Q/K/V、RoPE、SDPA
已被移除，局部交互由 `LocalInteraction`（因果卷积 + 线性递归）承担，全局依赖由
`ResonanceField (RFP)` 谱全局卷积承担。本版所有结论以 `simplcore_pg/` 真实代码与
`smoke_test.py` / `bench_*.py` 实测为准，无"文字创新"。

---

## 1. 这是什么

SimplCore 是一套**以 PyTorch 为主线的可运行 AI 架构**，核心是用两条互补路径替代
注意力：

- **LocalInteraction（局部）**：因果深度可分离卷积 + 可选门控线性递归，$O(n)$，
  承担"邻近 token 混合"（原本由自注意力负责的部分）。
- **ResonanceField (RFP)（全局）**：基于 FFT 的谱全局卷积，全局感受野，复杂度 $O(n\log n)$
  （替代自注意力"全局 token 交互"的职责）。

二者与 `DynamicMemoryChannels (DMC)`（固定外部记忆）、`MoE` 组合成
`SimplCoreMoEBlock`：

```
x → LocalInteraction → RFP → DMC → MoE
```

> 诚实边界：RFP 是"FFT 全局卷积（S4/H3 类）"的变体实现，DMC 是"外部记忆（NTM 类）"
> 的变体实现，LocalInteraction 是标准卷积 + 线性递归的组合。命名是修辞，底层是已知
> 技术组合的 SimplCore 化实现，不是凭空首创。

---

## 2. 为什么放弃 Transformer 注意力

| 维度 | 自注意力 | SimplCore (Local+RFP) |
|------|----------|------------------------|
| 全局交互 | softmax(QKᵀ) 归一化 | 频域全局卷积（无归一化、无 Q/K/V） |
| 序列复杂度 | $O(n^2 H)$ | $O(n\log n)$（RFP）+ $O(nH)$（Local） |
| 长序列显存 | KV 缓存 $O(n)$ 或近似注意力额外项 | 分块固定状态，与 $n$ 解耦 |
| 位置信息 | 需 RoPE/ALiBi 等外挂 | 因果卷积/递归天然有序，无需位置编码 |
| 训练稳定性 | softmax 饱和、注意力熵崩 | 无 softmax，梯度路径更直接 |

RFP 的全局卷积等价于一个**内容无关、频域固定的全局核**，避免了注意力对每一对
token 做归一化的二次开销，也避免了长序列下 KV 缓存线性膨胀。

---

## 3. 目录结构（真实）

```
simplcore_pg/                  # 生产级主线（PyTorch），本文档主要描述对象
  model_moe.py                 # SimplCoreMoEModel + SimplCoreMoEBlock（Local+RFP+DMC+MoE）
  localmix.py                  # LocalInteraction（因果卷积 + 线性递归，替代注意力）
  resonance.py                 # ResonanceField（FFT 波场演化，O(n log n)，可训练）
  memory.py                    # DynamicMemoryChannels（可训练外部记忆）
  config.py                    # SimplCoreConfig（含 attention-free 预设）
  moe.py / routing.py / block.py / model.py
  longctx.py / model_stream.py # 长上下文流式（固定状态 + 分块，无 KV 缓存）
  cuda/simplcore_kernels.cu    # 真实 CUDA 内核源码（本机无 CUDA 工具链，未编译验证）
  native_mm.py                 # 原生多模态（token-free）
  training/ / eval/ / benchmarks.py
simplcore/                     # 早期/探索性 Python 包（tensor 自研引擎、world model 等）
java-high-performance/         # 并行 Java 实现（非主线）
```

**主线语言**: Python（`simplcore_pg/`，依赖 PyTorch）。

---

## 4. 核心子层（均已在 simplcore_pg 实现且梯度可达）

### 4.1 LocalInteraction（局部，替代注意力）— `localmix.py`
- `CausalDepthwiseConv1d`：逐通道一维因果卷积（可带空洞 dilation 扩大感受野），
  左填充保证因果，参数量 $O(k\cdot H)$。
- `GatedLinearRecurrence`（可选）：$h_t = a_t\odot h_{t-1} + (1-a_t)\odot x_t$，
  单向递归，复杂度 $O(nH)$，无全局 token 交互、无 softmax。
- 二者门控融合，输出经 RMSNorm 残差。纯张量运算、无数据相关控制流，兼容
  `torch.compile` / FX。

### 4.2 ResonanceField (RFP) — `resonance.py`
FFT 波场演化：隐状态投影到频域多波通道，按可学习高斯带通（中心频率 $\omega_w$ +
带宽 $\sigma_w$）+ 阻尼演化若干步，逆变换回时域。
- 复杂度 $O(n\log n)$，全局感受野；
- **可训练**：`omega/sigma/beta/out_scale` 与 `in_proj/out_proj` 均参与反向传播；
  多步演化经 `torch.utils.checkpoint` 包裹，梯度真正到达 FFT 正向输入；
- `freeze_evolution=True` 时可回退为"仅投影可学、FFT 走 no_grad"的固定全局混合器
  （规避 FFT 反向泄漏，详见 §6）。

### 4.3 DynamicMemoryChannels (DMC) — `memory.py`
可学习记忆库 `(C, S, hd)` + 内容寻址读取（query/key/value 投影 + output 投影）。
- 记忆库固定大小，不随序列长度增长；
- 梯度可达（已实测 `memory_bank.grad > 0`）。

### 4.4 MoE — `moe.py`
门控专家 FFN（SwiGLU），含负载均衡 loss + router z-loss 辅助项。

---

## 5. 长上下文流式（无注意力）— `longctx.py`

流式块 `LocalInteraction → RFP(分块谱) → DMC(固定记忆) → MoE`，跨块仅传递
**固定大小**状态（RFP 频域状态 `(B, Lf, W)` + DMC 记忆 `(C,S,hd)`），使激活峰值
与序列长度 $n$ 解耦：

- RFP 分块频域演化：序列切成固定 `chunk_size`，每块 FFT 演化，块间维护固定频域状态；
- 无 KV 缓存（注意力已移除）；
- 复杂度单层 $O(n\log(\text{chunk}) + n\cdot k)$，相对标准 Transformer $O(n^2H)$
  在长序列下数量级占优，且峰值显存与 $n$ 解耦。

---

## 6. 已知工程约束（诚实记录）

- **FFT 反向 workspace 泄漏**：Windows/CPU 的 torch 构建下长循环 FFT 反向会累积不释放
  内存；RFP 用 checkpoint 规避，必要时 `rfp_freeze_evolution=True` 回退。
- **GPU 实测缺失**：当前仅 Windows/CPU/AMD-680M 验证；真实显存/吞吐需 GPU。
- **合成基准尺度**：多跳/长程基准为小规模可复现证据，非大规模语料 SOTA 对比。

---

## 7. 可复现性

见 `docs/REPRODUCIBILITY.md` 与 `smoke_test.py`。所有 `bench_*.py` 输出 JSON，
`test_stream_consistency.py` 验证流式数值一致性。通过率目标 $\ge 90\%$。
