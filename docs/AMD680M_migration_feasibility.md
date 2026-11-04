# AMD Radeon 680M（4GB 分配显存）能否完整跑 TS2SC 迁移工作流

**结论：原样照搬不行，适配后可以，而且迁移这一阶段反而是最不挑硬件的。**

硬件口径先纠一处：680M 是 RDNA2 Rembrandt **iGPU / UMA**，12 CU、768 SP、2.2GHz，FP16 6.76 TFLOPS、FP32 3.38 TFLOPS，共享 LPDDR5/DDR5、标称带宽 102.4 GB/s [citation:8][citation:13]。BIOS 里划的「4GB 专用显存」不是物理显存，是从系统内存预占的 carveout；有实测把分配从 1GB 调到 16GB，推理速度**没有变化** [citation:11]。真正的瓶颈是系统内存带宽（APU 与 CPU 争用，实测约 50–65 GB/s [citation:14]），不是那 4GB 数字。

## 分阶段判定

| 阶段 | 判定 | 依据 / 需要的改动 |
|---|---|---|
| 权重下载、分片重组、流式 mmap 读取 | ✅ 直接可跑 | 纯 CPU + 磁盘；本沙箱已用 42 分片 + 跨片 mmap 跑通 311 张量 |
| 解析修复（linear 频率网格、DC-bias） | ✅ 直接可跑 | 小矩阵闭式解，无显存压力 |
| teacher 前向基线（1.7B bf16 = 4.06 GB） | ⚠️ 需适配 | 刚好超 4GB；改 Q4（~1.1 GB）或继续流式 mmap。本沙箱 2.4GB 内存已用后者跑通 |
| **TS2SC 迁移（逐层解析拟合）** | ✅ **最宽松** | 用的是 ridge 闭式解 + 逐位直搬，**不是梯度训练**，不产生梯度/优化器状态；逐层流式，峰值 ≈ 单层权重 + 几段激活，4GB 绰绰有余 |
| 生成验证 | ✅ Q4 后可行 | llama.cpp Vulkan 在 680M 上实测 Gemma-3-1B：prompt ~150 tok/s、eval ~60 tok/s [citation:6] |
| 全量微调 / SFT | ❌ 不可行 | 1.7B 全量 bf16 混合精度需 16–24 GB；4GB 只能做 QLoRA，且本工作流已证明 SFT 对生成无效 |
| 加视觉编码器（+0.55B） | ⚠️ 需 Q4 | 语言 1.7B + 视觉 0.55B = 2.25B，bf16 约 4.5GB 超了；Q4 后约 1.3GB，装得下 |

## 三个必须改的地方

**① 弃 ROCm，走 Vulkan。** 680M 是 `gfx1035`，**不在 ROCm 官方支持表** [citation:12]。社区用 `HSA_OVERRIDE_GFX_VERSION=10.3.0` 伪装成 gfx1030，但已有明确报告：矩阵乘法看着正常，神经网络却输出**负的 label 索引和负概率**，连单层 linear 都算错 [citation:4]。这条路线不可信。Vulkan 是 680M 上验证过的正路：llama.cpp Vulkan 后端在 680M 上跑 1B 模型 ~60 tok/s，甚至 30B MoE Q4 也能到 13 tok/s（靠系统内存） [citation:6][citation:10][citation:14]。编译时开 `GGML_VULKAN`，运行时按显存调 `-ngl`，别无脑全卸载 [citation:5]。

**② Q4 量化。** 2.25B bf16 是 4.5GB，Q4_K_M 约 1.3GB。680M 上 Q8→Q4 的速度差是数量级的（31B 模型 Q8 只有 2.3 tok/s，Q4 能到 18 tok/s） [citation:11]。

**③ SGC 算子得自己写。** 这是最关键、也最容易被忽略的一条：**llama.cpp 不认 SimplCore 的 SGC mixer**。它的 `ggml` 图里没有「阻尼振荡基 + IIR 递推」这个算子。两条路：
- 写自定义 Vulkan compute shader（FFT/IIR 对偶保持，O(1) 状态可映射到 GPU buffer）
- 或者 SGC 留在 CPU，只把 FFN / embedding 的大 matmul 交给 Vulkan —— 混合执行，跨设备拷贝会吃掉一部分收益

## 「完整迁移」的准确说法

| | 判定 |
|---|---|
| 按当前沙箱脚本原样（PyTorch fp32 + 流式 mmap + CPU 全前向） | ❌ 4GB 装不下，且 ROCm 不稳 |
| 换 instruct 基座 + 解析迁移 + DC-bias + Q4 + Vulkan（含自定义 SGC） | ✅ 可行，预计 2.25B Q4 在 680M 上约 20–40 tok/s |
| 想在 680M 上从零训 1.7B 或做全量 SFT | ❌ 不可能 |

一个反直觉但重要的点：**用户担心的「迁移」恰恰是最不吃显存的阶段**。TS2SC 是解析直接迁移（ridge 闭式解 + 逐位直搬，零梯度），不是蒸馏也不是训练；真正吃资源的 teacher 前向和生成验证，都可以靠流式 mmap 或 Q4 解决。所以 680M 的瓶颈不在迁移算法，在**软件栈（ROCm→Vulkan）和自定义算子**。

## 本地先行

按用户要求，迁移先在沙箱本地跑通再谈 680M 移植。当前进度见 `Q17/` 与 `TS2SC_Q17/`：

- 42 个分片全部到位（4.064 GB），跨片 mmap 流式读取器 `q17_reader.py` 已验证 311/311 张量、index 零未映射
- `q17_teacher.py` 已用 chat template + no-think 预填跑通 **Qwen3-1.7B-Instruct 原版基线**，首次拿到正常中文输出

## 参考

- 680M 规格 [citation:8][citation:13]；UMA 分配无关 [citation:11]；带宽实测 [citation:14]
- ROCm gfx1035 不支持 / 伪装出错 [citation:12][citation:4][citation:7]
- Vulkan 实测 [citation:6][citation:10][citation:5]；量化对比 [citation:11]
