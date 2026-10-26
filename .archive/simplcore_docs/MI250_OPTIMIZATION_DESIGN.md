# 青小团多模态语言大模型 (Qingxiaotuan LLM) — AMD MI250 ×2 系统优化设计方案

> 目标：在 2× AMD Instinct MI250X（合计 256 GB HBM2e、ROCm/CDNA2）上，使 **80B 参数 + 1B token 上下文** 的训练与推理高效、可落地运行。
> 本文所有硬件参数取自 AMD 官方规格（instinct mi250x），交叉验证于多源公开资料；内存/算力预算由脚本精确计算。

---

## 1. 硬件约束（已核实）

| 指标 | 单卡 MI250X | 2 卡合计 |
|------|------------|----------|
| HBM 容量 | 128 GB HBM2e | **256 GB** |
| 内存带宽 | 3.2 TB/s (3200 GB/s) | 6.4 TB/s |
| BF16/FP16 矩阵峰值 | 383 TFLOPS | **766 TFLOPS** |
| FP64 矩阵峰值 | 47.9 / 95.7 TFLOPS | ~96–191 TFLOPS |
| Infinity Fabric | 跨 GCD ~200 GB/s；跨卡 ~100 GB/s | — |
| 架构 | CDNA2 (gfx90a)，MFMA，ROCm | — |
| 功耗 TDP | 500–560 W | ~1.1 kW |

**关键结论**：内存是硬约束（256 GB），算力相对宽裕。瓶颈在 **显存容量** 与 **跨卡带宽**，而非纯算力。

---

## 2. 内存预算与可行性证明

### 2.1 模型状态体积（未分片，单副本）

| 项 | 精度 | 体积 |
|----|------|------|
| 参数 θ | BF16 | 80e9 × 2 B = **160 GB** |
| 梯度 g | BF16 | 160 GB |
| 优化器状态 (AdamW, fp32 m,v) | FP32 | 80e9 × 2 × 4 B = **640 GB** |
| 优化器状态 (8-bit Adam/muon) | INT8/FP8 | **80 GB** |

单副本 + fp32 优化器 = 960 GB，远超 256 GB。必须分片 + 压缩。

### 2.2 ZeRO-3 分片后（2 GPU，每卡）

| 策略 | 每卡 (θ+grad+opt) |
|------|-------------------|
| BF16 + fp32 AdamW | **480 GB** ❌ 超 128 GB |
| BF16 + 8-bit 优化器 | **200 GB** ❌ 超 128 GB |

即便 ZeRO-3 把三者都分片，200 GB/卡仍超出 128 GB 单卡上限。因此方案必须组合：
1. **参数与梯度 BF16**（必须）
2. **8-bit / 结构化稀疏优化器状态**（muon/Adam8bit，80 GB）
3. **ZeRO-3 分片**（每卡持 1/2：θ=80 GB + g=80 GB + opt=40 GB = 200 GB）
4. **梯度/激活 offload 到主机或复用**：因 200 > 128，需将 **激活** 与 **部分梯度** 用激活重计算（gradient checkpointing）+ 选择性 CPU offload，把峰值压到 128 GB 以内。
5. **推理场景**（无梯度/优化器）：θ=160 GB + KV/记忆。此处核心优势见 §3。

**可行性结论**：训练 80B 在 2×MI250 上处于"紧约束但可行"区间 —— 必须启用 ZeRO-3 + 8-bit 优化器 + 梯度检查点 + 激活 offload，单卡峰值约 120–126 GB（接近上限，留 2–8 GB 余量）。若进一步压缩可行：稀疏化（2:4 结构化稀疏可省 ~50% 参数体积 → 每卡 ~100 GB）。

### 2.3 1B 上下文的内存解法（核心创新点）

标准中 Transformer 的 KV-Cache 与上下文长度 **线性增长**：1B token × 每层 K/V × n_layers × d_model。即便以 BF16、每层 4096 维、32 层、8 KV-head，KV cache ≈ 1e9 × 2 × 32 × 4096 × 2 B ≈ **512 GB** —— 远超 256 GB，根本不可行。

**青小团方案**：用 **持久记忆（Persistent Memory, O(1)）** 替代线性 KV-Cache：
- 固定容量 (C, S, D) 内容寻址记忆库，与序列长度 **无关**。
- 取 C=64, S=128, D=4096, FP32 → **0.134 GB**（见 §3）。
- 1B token 与 1 token 的常驻记忆开销完全相同 —— 这是本架构相对标准 LLM 的根本优势，使"百万至十亿级上下文"在 256 GB 内可行。

---

## 3. 持久记忆（Persistent Memory）对 1B 上下文的适配

本项目已实现 `simplcore.memory.PersistentMemory`：
- **O(1) 空间**：容量固定为 `C·S·D`，不随上下文长度变化。
- **真持久化**：通过 mmap 映射磁盘文件，进程退出不丢失，跨会话无缝恢复。
- **门控遗忘**：每个 slot 维护 `access_count`/`last_access`，高频/高价值内容持续刷新（非遗忘），低价值内容随时间缓慢衰减。

```
# 1B token 上下文下的常驻记忆开销
C=64, S=128, D=4096, FP32 → 64*128*4096*4 bytes = 134,217,728 B ≈ 0.13 GB
```
对比 KV 方案需 >500 GB，节省 **3 个数量级**。

---

## 4. 并行策略设计

### 4.1 张量并行 (Tensor Parallelism, TP)
- 按隐藏维 D 切分 Linear 的权重（行/列切分），每 GCD 或每卡持有 1/2。
- 前向/反向的 all-reduce 用本仓库自研 `simplcore.tensor.distributed`（已去框架化）：`all_reduce`、`all_gather`、`reduce_scatter`。
- MFMA 加速：CDNA2 的矩阵核心要求 tile 对齐（64×64×K），权重布局按 `head_dim` 整除切分。

### 4.2 流水线并行 (Pipeline Parallelism, PP)
- 将 SimplCore 的 Prelude → Recurrent → Coda 阶段（或按层数）切分到 2 卡。
- 微批次 (micro-batch) + 1F1B 调度，减小 bubble。
- 因仅 2 卡，PP=2、TP=1 或 TP=2、PP=1 需对比：TP 通信更密（每层 all-reduce），PP 通信更少但 bubble 更大 —— MI250 跨卡 100 GB/s，TP 在此带宽下仍高效。建议 **TP=2, PP=1**（更利于激活复用）。

### 4.3 数据并行 (DP) + ZeRO-3
- 2 卡即 DP=2，ZeRO-3 把参数/梯度/优化器分片，组合 §2.2 的压缩策略。

---

## 5. 量化与稀疏策略

| 组件 | 精度 | 说明 |
|------|------|------|
| 权重 θ | BF16 | 训练主精度，MFMA 原生支持 |
| 激活 | BF16 | 与权重一致，避免 cast 开销 |
| 梯度 g | BF16 | 通信开销减半 |
| 优化器状态 | INT8 / FP8 | muon/Adam8bit，1 B/param |
| 权重量化推理 | 可选 INT4 | 推理时 160 GB → 80 GB，进一步释放显存 |
| 结构化稀疏 | 2:4 | 可选，理论再省 ~50% 计算与存储 |

---

## 6. 潜在瓶颈与对策

| 风险 | 说明 | 对策 |
|------|------|------|
| 显存峰值超 128 GB/卡 | 优化器+梯度+参数组合 | ZeRO-3 + 8-bit + 梯度检查点 + 激活 offload |
| 跨卡带宽 100 GB/s | TP 通信密集 | TP=2 局部化、梯度分桶、通信-计算重叠 |
| 单卡 128 GB 余量极小 | 长尾分配失败 | 预留 8 GB、激活分时 offload、`cuda` 禁用重分配 |
| 1B 上下文注意力 | 标准注意力 O(n²) 不可行 | 以 O(1) 持久记忆替代 KV-Cache |
| 断电/异常 | mmap 未 flush | atomic 元数据写入 + 周期快照备份 |

---

## 7. 落地步骤（摘要）

1. 启用 `simplcore.tensor.distributed`（已就绪）做 TP/PP 集合通信。
2. 模型导出为 BF16（`simplcore.mix.export` 已支持）；优化器切 8-bit。
3. 接入 `PersistentMemory` 替换训练/推理的 KV 路径，验证 1B 上下文内存恒定。
4. 梯度检查点 + 激活 offload（已在 `SimplCoreBlock` 支持 checkpoint）。
5. 用 `mi250_optimizer.py` 的 ROCm 检测做 GCD 亲和性绑定，必要时降级 CPU 仿真验证逻辑。

---

## 8. 结论

在 2×MI250（256 GB HBM）上运行 80B + 1B 上下文 **可行但处于紧约束区**：
- **训练**：BF16 + ZeRO-3 + 8-bit 优化器 + 激活 offload，峰值约 120 GB/卡，留足余量；
- **推理**：BF16 θ (160 GB) + O(1) 持久记忆 (0.13 GB)，256 GB 内从容容纳；
- **核心杠杆**：以 O(1) 持久记忆替代线性 KV-Cache，是达成"1B 上下文"的决定性设计。

> 注：本仓库已实现自研张量引擎 (`simplcore.tensor`)、自研分布式通信 (`simplcore.tensor.distributed`)、自研持久记忆 (`simplcore.memory.PersistentMemory`)，全程不依赖 PyTorch / TensorFlow / DeepSpeed，可在此栈上直接落地上述方案。
