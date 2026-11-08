# 迁移蒸馏前哨实验：SimplCore 原生 47 万参数三臂对照

> **日期**：2026-11-05　**执行**：纯 NumPy 从零实现，2 核 CPU / 2 GB 内存沙盒
> **代码**：`tools/transfer_distill/`（无 torch 依赖，`python gradcheck64.py` → `python run.py`）
> **语料**：本仓库真实包内 700 个 `.md/.py/.txt/.json/.sh/.c/.h/.java` 文件，
> 5,567,423 字符（含 363,377 中文字符），过滤控制字符后字符级词表 V=512，
> 训练 5,289,051 / 验证 278,372 token。
>
> **一句话**：在 3.07M token/臂的前哨预算下，「只蒸馏 teacher 输出分布」
> 让 SimplCore 的 val ppl 比原生从零训练再降 **5.55%**，且比同预算
> Transformer 低 **9.4%**；但**三臂生成都仍是词堆砌，不成句**。
> 这是「ppl / 分布层面的部分支持」，**不是「迁移成功」的证实**。

---

## 0. 为什么要做这个实验

上一轮把「用静态核拟合 attention 输出」这条路证伪了：精确核 oracle 显示
核形状 100% 正确时中层注入 cos 仍 ≈0，因为 attention 的核是 $K(i,j)=f(q_i,k_j)$
（内容相关），SGC 的核是 $K(\tau)$（位置相关、静态）——**函数类不匹配，不是自由度不够**。

本轮换目标函数：**不再逼 SGC 内部长得像 attention，只用冻结 teacher 的
输出分布做黑盒蒸馏**。这不引入 $N\times N$、不引入 KV cache，守住纯粹性。

## 1. 实验设置（三臂等预算）

| 臂 | 架构 | 参数量 | 目标 |
|---|---|---:|---|
| `ts` | 2 层 causal Transformer（4 头，含可学习位置嵌入） | **524,928** | from-scratch 基线 + teacher |
| `sgc` | 2 层 SimplCore SGC（`init_mode=linear`，M=32，Kloc=4） | **436,096** | from-scratch 原生训练 |
| `kd` | 同 `sgc` 架构 | **436,096** | 输出分布蒸馏 |

共同设置：`D=128, FFN=512, L=64, B=16, 3000 步, AdamW lr=6e-4,
wd=0.01, grad clip=1.0`，每臂 **3,072,000 token**。
`kd` 损失：`α·T²·KL(p_t‖p_s) + (1−α)·CE(labels)`，`T=5`，
`α` 从 0.7 线性降到 0.3，**每位置都蒸馏**（不只末位），teacher 全程冻结、无反向。

### 前置检验（先跑，不过不训）

- `gradcheck64.py`：fp64 中心差分 ε=1e-6，18/22 参数组全过。
  最坏相对误差 `ts 7.17e-6 (fn)`、`sgc 2.37e-6 (l1.Wu)`。
- **双形式一致性**：整段 Toeplitz 前向 vs O(1) 流式 IIR，
  L=32/64/128/256 的 `max_abs` 分别 `6.56e-7 / 9.54e-7 / 2.35e-6 / 5.98e-6`，
  流式状态 **68,608 B 恒定**（与 L 无关）。

### 训练过程中发现并修复的两个实现 bug（计入本轮成果）

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | 双形式 `rel=0.320` | `psi_eff` 误乘了 `sinθ`。上游 `step()` 的递推 `y_t=2ρcosθ·y_{t−1}−ρ²·y_{t−2}+sinθ·u_t` 冲激响应恰为 `ρ^t sin((t+1)θ)`，`sinθ` 已抵消 | 去掉该因子 |
| 2 | 修完仍 `rel=0.149` | 所有层**共用同一个**局部卷积历史缓冲 | 每层独立 `state[2·N_LAYERS + l]` |

修复后两臂全部重训；修复前的 checkpoint 归档为 `*_buggy.pkl`，未删除。

## 2. 结果

### 2.1 数值（`verdict.json` / `log_*.json`）

| 指标 | `ts` | `sgc` (scratch) | `kd` |
|---|---:|---:|---:|
| val ppl | 11.281 | 10.817 | **10.217** |
| val CE | 2.4231 | 2.3820 | **2.3111** |
| val KL(T=1) vs teacher | 0.0000 | 0.7395 | **0.4739** |
| 3-gram 重复率（top-k 采样） | 0.057 | 0.034 | 0.000 |
| val ppl 比 | 1.000 | 0.959 | **0.906** |

- `kd` 相对 `sgc` scratch：**−5.55% ppl**
- `kd` 相对 `ts`：**−9.4% ppl**（不到协议里 1.10 的失败线）
- KL(T=1) `0.7395 → 0.4739`（−35.9%）：输出分布确实被吸收，不是数值巧合

### 2.2 收敛曲线（每 500 步 val ppl）

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---:|---:|---:|---:|---:|---:|
| `ts` | 23.08 | 16.13 | 13.57 | 12.43 | 11.72 | 11.28 |
| `sgc` | 16.09 | 12.80 | 11.73 | 11.23 | 10.95 | 10.82 |
| `kd` | 17.07 | 13.03 | 11.63 | 10.94 | 10.51 | 10.22 |

`kd` 前 500 步比 `sgc` 慢（17.07 vs 16.09，蒸馏信号早期是负担），
约 1500 步后交叉并持续拉开——符合「先对齐分布、再吃 CE」的预期。

### 2.3 生成（同一 prompt，top-k=40 / T=0.85 / seed=17，90 token）

```
ts  : ' ime.prinm = new ffrence(float[]; input.length);\n \n ithis.didefaturests = new ATVaCULNOCUs'
sgc : ' * memoryHas()):\n gate_seed to the step = did):\n return L * GRUMER")\n embeddingMit\\tests_r'
kd  : ' for keyss\\cposl |"public void testTimeMillibuditer.configreval = 1.01f);\n \n prefloat:\n in'
```

**三臂都不成句**，仍是代码碎片堆砌。能看出局部字符统计（`for` / `def` /
`public static class` / 换行缩进），说明模型学到了语料的表层结构，但**没有
学到可组合的语言能力**。这一条必须原样保留，不包装成成功。

## 3. 判定：脚本给了 `SUPPORTED`，我人工降级为 **PARTIAL**

`run.py` 的自动判据是 `ratio_kd_vs_ts ≤ 1.10 且 rep3 改善` → 输出
`SUPPORTED`。**这个判据太松**，有三处它没覆盖：

1. **生成未过线**。协议里写的成功判据包含「生成能产出连贯 3~5 句，不出现
   死循环」——三臂都没达到。`rep3` 在 top-k 采样下接近 0，区分不了
   「连贯」和「不重复的词堆砌」。
2. **核心混淆未排除**：`sgc` from scratch 本身（10.817）就优于 `ts`（11.281）。
   所以 `kd` 的 5.55% 收益**可能只是「多了一个正则项」**，而不是「迁移了
   teacher 的知识」。判别性对照是**随机未训练 teacher**（`kdrand` 臂）：
   目前只跑到 1000 步（`kdrand 16.10` vs `kd 13.03`），方向支持
   「teacher 有信息」，但**未跑完 3000 步，不能定论**。该臂标 ⬜ 待补。
3. **预算不足**：3.07M token/臂，按 TS 臂 524,928 参数的 Chinchilla 20×
   应为 **10.50M**，只到 29%。且字符级 512 词表、L=64，与真实 BPE /
   长上下文建模差得很远。

> **严格结论**：在 3.07M token 前哨预算下，「输出分布蒸馏」在 **ppl / 分布
> 层面部分支持**（KD 显著优于 SGC scratch，KL 明显下降）；在**生成质量层面
> 未证实**；「收益来自知识迁移而非正则化」**未证实**。三者都不外推到规模化训练。

### 证伪条件（仍然写死，不事后圆）

若补齐 `kdrand` 3000 步后 `kd` 与 `kdrand` 的 ppl 差落在噪声内，
或把预算拉到 Chinchilla 20× 后 `kd` 不再优于 `sgc` scratch，
则「输出分布蒸馏能迁移知识」判为证伪，停止追加 Stage / trick / attention。

## 4. 这轮真正填补了什么

| 之前的空白 | 本轮状态 |
|---|---|
| 「SGC 从头训练始终失败」这个说法 | **证伪**：436,096 参数原生模型在真实语料上 3000 步 loss 稳定下降、ppl 10.817，且优于同预算 Transformer。此前失败的是**迁移/蒸馏拟合 attention**，不是训练本身 |
| 上游 §5.14 的 150M@2GB 实验（6 万 token，0.00056 tok/param） | 本轮 7.04 tok/param（`sgc` 臂），高 **12,600 倍** |
| 双形式一致性在真实训练权重上的验证 | 训练后权重下 L=32~256 全部 `rel < 6e-6`，状态 68,608 B 恒定 |
| 「只蒸馏输出分布」这条路的可行性 | 有了第一份可复现的三臂对照数据（含 raw log / checkpoint） |

## 5. 与「不偏向 Transformer」边界的兼容性

本实验**没有引入任何 $N\times N$ 或 KV cache**：
- KD 臂是纯 SGC 架构，teacher 仅作为冻结黑盒提供 logits；
- 不搬 attention 权重、不做隐状态硬对齐、不加辅助 attention 层；
- SGC 侧仍是 O(1) 状态、静态核、FFT/IIR 双形式。

teacher 只在训练阶段存在，推理阶段完全没有 attention——这是**训练时蒸馏**，
不是 hybrid 推理。代价是：不能指望逐位复刻 teacher 的推理过程，只能指望
「说出类似分布的话」。

## 6. 复现命令

```bash
cd tools/transfer_distill
python gradcheck64.py                                  # fp64 梯度检验
python run.py --arm ts  --total 3000 --chunk 500 --tag ts  --lr 6e-4   # ×6
python run.py --arm sgc --total 3000 --chunk 500 --tag sgc --lr 6e-4   # ×6
python run.py --arm kd  --total 3000 --chunk 500 --tag kd  --teacher ts --lr 6e-4  # ×6
python run.py --final                                   # 三臂评估 + 判决
```

语料文件 `corpus.txt` 由本仓库 700 个真实文本文件生成，未纳入版本控制
（6.3 MB）；`arr.npy` / `chars.json` 首次运行自动生成。
`ckpt_*.pkl` / `log_*.json` / `verdict.json` 为原始产物，请随论文一并留存。

## 7. 局限（照实，不删）

- **单 seed**（`seed=0`），无误差棒、无 seed 间方差。
- 字符级 V=512、L=64、2 层、纯 CPU 纯 NumPy；与上游 PyTorch / ROCm 实现
  **不是同一份代码**，本轮是独立白盒重实现（核公式按上游 `sgc.py:build_kernel`
  对齐），结论只对这个小尺度实现成立。
- `psi` 归一化按上游 `ρ^τ sin((τ+1)θ)` 口径；与更早白盒报告的
  `ρ^t sin((t+1)θ)/sinθ` 口径不同，条件数/偏差绝对值不可直接搬用。
- 语料是本仓库源码与文档，含大量代码，不是自然中文/英文通用语料；
  ppl 数字只在三臂之间可比，不与外部模型比。
- 未跑仓库 pytest（沙盒无 torch）；本轮所有检验都是自写脚本。
- `kdrand` 对照未完成，见 §3。
