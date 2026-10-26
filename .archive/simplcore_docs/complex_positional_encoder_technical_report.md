# 复数位置编码 (Complex Positional Encoder, CPE) — 技术报告

**版本**: v2.0
**日期**: 2026-08-22
**作者**: 青小团团队（Qingxiaotuan LLM V1.0）
**架构**: SimplCore（易核架构）
**实现语言**: Python（自研 simplcore.tensor 后端）

---

## 一、技术概述

### 1.1 核心问题

位置编码是大语言模型表达序列顺序信息的关键。标准绝对/相对位置编码（如正弦编码、RoPE）以实数张量表示位置；复数表示则可在频域以相位（复指数）自然刻画周期性与相对位移。

### 1.2 CPE 方案

复数位置编码 (Complex Positional Encoder, CPE) 用复数表示构造位置相关的复振幅：

- 通过相位编码将位置映射为复指数 `exp(i·φ(pos))`；
- 施加可学习的酉变换 `U = exp(i·H)`（H 为 Hermite 矩阵），实现位置间特征的线性混合；
- 施加可学习的成对耦合矩阵，建模基频位置间的相互作用；
- 将复态（实部 + 虚部）投影到模型维度 D，以残差方式注入输入。

> **诚实声明**：自研 NumPy 后端当前无复数自动微分，因此复数链路（相位编码 → 酉变换 → 耦合）以 numpy 前向计算，返回的编码张量不携带梯度图。原语组合层（见 §1.3）完全可训练。接入复数 autograd 后，复数链路亦可端到端训练。

### 1.3 原语组合层 (_PrimitiveComposer)

为验证“参数效率 + 可训练”路线，模型另含原语组合层：

- 维护一组可学习 primitive 原语库；
- 经路由网络 top-k 选择原语并以归一化权重组合，生成等效更宽的表示；
- 该层所有参数在 simplcore.tensor 后端上梯度可流（top-k 不可微处已用 scatter-add 手动反向实现）。

---

## 二、数学基础

### 2.1 相位编码

设基频数量 `n`，复数空间维度 `H = 2^n`。基态 `|k⟩` 的频率由可学习 `phase_freqs` 组合：

```
freq[k] = Σ_j phase_freqs[j] · bit_j(k)
```

位置 `pos` 的复振幅：

```
ψ(pos, k) = exp(i · 2π · pos / H · (1 + freq[k]) · k) / sqrt(H)
```

### 2.2 可学习酉变换

Hermite 矩阵 `H_mat = (W + W†) / 2`，酉矩阵 `U = exp(i · H_mat)`。

自研后端无 `np.linalg.matrix_exp`（numpy 2.x 已移除），改用特征分解实现：

```
eigvals, eigvecs = eig(H_mat)
U = eigvecs · diag(exp(i · eigvals)) · inv(eigvecs)
```

对每行状态施加 `U` 后再施加耦合矩阵，并重归一化为单位模长。

### 2.3 幅值分布

将输入视为复振幅的实/虚表示，计算 `p_i ∝ |x_i|²` 并归一化，用于分析各维度激活强度（统计用途，非量子测量）。

---

## 三、实现结构

| 模块 | 说明 |
|------|------|
| `ComplexPositionalEncoder` | 复数位置编码主类（相位编码 + 酉变换 + 耦合） |
| `_PrimitiveComposer` | 原语组合层（top-k 路由 + 组合 + FFN），完全可训练 |
| `SimplCoreWithCPE` | 集成 CPE 与原语组合的轻量语言模型 |
| `compute_parameter_efficiency` | 参数效率估算（对比标准 Transformer） |

---

## 四、验证结果

通过 `simplcore/tests/verify_complex_positional_encoder.py` 验证：

| 测试项 | 结果 | 说明 |
|--------|------|------|
| CPE 前向形状 | ✅ | (B, L, D) → (B, L, D) |
| 无 NaN | ✅ | 前向输出数值稳定 |
| 幅值分布归一化 | ✅ | 概率和 = 1 |
| SimplCoreWithCPE 前向 | ✅ | (2,24) → (2,24,512) |
| SimplCoreWithCPE 反向 | ✅ | 梯度流通畅 |
| 参数效率估算 | ✅ | 见 `compute_parameter_efficiency` |
| 原语组合层可训练 | ✅ | 60 步 SGD 后 loss 下降，参数梯度流通畅 |

---

## 五、命名说明

本文档中的“复数 / 耦合 / 幅值”等术语均为线性代数与信号处理的标准表述，与量子计算无关。早期版本曾使用“EQP / 谐振场 / 拓扑”等夸大或误导命名，现已统一更名为中性、可自研验证的表述。
