# MI250 单卡 · 1 个月训练预算账本

> 目的：在买卡前把"1 个月最多能训多大"算清楚，避免买了卡才发现目标不可达。
> 所有数字都是**估算**，以实机 `scripts/bench_streaming.py` 与
> `python -m simplcore.core.eval throughput` 的吞吐实测为准。

## 1. 硬件口径（1× MI250 OAM，2 GCD）

| 项目 | 数值 |
|---|---|
| BF16 / FP16 矩阵峰值 | 362.1 TFLOPS |
| HBM2e | 128 GB（2×64 GB，每 GCD 64 GB） |
| 显存带宽 | 3.2 TB/s（1.6 TB/s per GCD） |
| CU / 时钟 | 208 CU / 1.7 GHz |
| FP8 | 无（CDNA2），目标精度 BF16 |

MI250 是**双 GCD 封装**，ROCm 通常暴露为两张卡。训练要用 FSDP / ZeRO-2/3
跨 2 个 GCD 切分，不能当成单设备 128 GB 用。

## 2. 计算模型

训练算力近似：

```
FLOPs ≈ 6 · N · D        (N=参数量, D=训练 token 数)
```

30 天墙钟，扣除调试/评测/断点重启，**有效训练占比按 65%**：

```
T_eff = 30 × 86400 × 0.65 = 1,684,800 s
```

SimplCore 的 FFT 全局卷积是 O(L log L)，在 4K 上下文下相对 6ND 的矩阵乘占比很小，
因此 6ND 仍是合理口径；它的架构优势主要体现在**长上下文与 O(1) 流式推理**，
不改变"1 个月能训多大"的量级。

## 3. 参数上限（算力约束）

| MFU | 有效 TFLOPS | 1 个月总 FLOPs | 1B 可训 token | Chinchilla 上限（20 tok/param） | 欠训上限（10 tok/param） |
|---|---|---|---|---|---|
| 15% | 54.3 | 9.15×10^19 | 15.3B | **0.87B** | 1.23B |
| 25% | 90.5 | 1.53×10^20 | 25.4B | **1.13B** | 1.59B |
| 35% | 126.7 | 2.14×10^20 | 35.6B | **1.33B** | 1.89B |

- **推荐目标（Chinchilla 质量）：`tuan-1b`（1013M），训练 20–35B tokens。**
- **Chinchilla 质量上限：约 1.3B。**
- **2B 是"欠训交付档"**：`tuan-2b`（2024M）在 25% MFU 下 30 天只能喂
  **~12.4B tokens（6.1 tok/param）**，约为 20× 的三分之一。它可以在 30 天内
  跑完 4M 课程并产出可用主干（见 `docs/TRAINING_2B_MI250_30D.md`），
  但**不得宣称 Chinchilla-optimal**；要 40B tokens 需 ~92 天。
- **不建议在单卡 30 天内超过 2B**：3B+ 只能喂 ~4 tok/param，质量会崩。

## 4. 显存约束（不是瓶颈）

Adam 混合精度训练的状态约 **16 B/参数**：

```
bf16 权重 2 + bf16 梯度 2 + fp32 master 4 + Adam m 4 + Adam v 4 = 16 B
```

| 参数量 | 优化器状态 | 128 GB 是否够 |
|---|---|---|
| 1B | 16 GB | 非常宽裕 |
| 3B | 48 GB | 够（+激活/碎片） |
| 4B | 64 GB | 勉强（激活要 checkpoint） |

结论：**显存允许到 ~4B，但算力只喂得动 ~1.3B。** 买卡后不要按显存定模型大小。

## 5. 结论与前置条件

1. 目标锁定 **1B / 20–35B tokens**，不贪大。
2. 上卡第一件事：实测 MFU —— 用 `scripts/bench_streaming.py` 与
   `python -m simplcore.core.eval throughput`。SimplCore 含 FFT/卷积，
   ROCm 上可能低于纯 matmul 场景；若实测只有 15%，就把 token 预算降到 ~15B。
   训练循环已自动套用 ROCm 调优（`scripts/train_tuan_mi250.sh` 一键起：max-autotune 编译 /
   展开式显存段 / 双 GCD DDP），但 MFU 实测仍以此为准，不靠调优"虚增"。
3. 对照口径：**不与 Transformer 做无限定宣称**。本仓库已移除注意力基线，
   因此"1B SimplCore 抵 3~7B Transformer"必须按
   **等 token / 等 FLOPs 的能力对齐**来验证，而不是拿"1B + 工具"去对
   "7B 裸模型"做无限定宣称。机制层面的优势（复杂度、状态、闸门）见
   `docs/THEORETICAL_EDGE.md`；质量层面的证据**尚未取得**。
4. 1 个月含调试，实际产出可能低于表值；建议预留 1 周做 bring-up。
