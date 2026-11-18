# RTX 4090D-24GB 训练战役（2B 全参 × 24h）

> **用户硬指标**：单张 RTX 4090D-24GB，**24h 以内训练完整 2B 参数**，极致优化。
> **账本**：`python -m simplcore.core.campaign`（解析计算，不占显存，可在任何机器审计）
> **启动器**：`scripts/train_tuan_2b_4090d.sh`（Phase 0 dry-run / Phase 1 24h 训练）
> **状态**：计划与训练栈已就绪；**所有 4090D 实测数字待实机填充**（本机无卡），
> **实测口径模板**：`docs/RTX4090D_MEASUREMENTS.md` +
> `artifacts/4090d/measurement_template.json`（自动回填
> `scripts/fill_measurements.py`；每字段带 caliber，>20% 偏差强制写原因）。
> 本文所有吞吐/显存数字分"解析估算"与"实机实测"两栏，不得混用。
> 历史 MI250 战役文档已按用户指令删除（2026-11-18，见 REVISION_LOG.md）。

## 1. 硬件与口径

| 项 | 值 | 来源 |
|----|----|------|
| 显存 | 24 GB GDDR6X（CUDA 上下文自占 ~0.5 GiB） | 规格 |
| bf16 稠密峰值 | ~157 TFLOPS（4090 公称 165.2 × 0.95） | 规格估算，**实机以 Phase 0 实测为准** |
| 显存带宽 | 1008 GB/s | 规格 |
| 训练精度 | bf16 autocast 前向 + **fp32 参数/梯度**（TF32 tensor core） | `apply_cuda_tuning` |
| 优化器 | **Adafactor**（因子化二阶矩，β1=0） | 24GB 硬闸门，见 §2 |

## 2. 显存账（24GB 定容，解析）

2B 参数（tuan-2b 预设：dim 2048 / 31 层 / FFN 4.0 / vocab 32000，
2023M total）的单卡显存账——**优化器路径选择是硬闸门，不是建议**：

| 路径 | 公式 | 2B 总量 | 24GB 可行性 |
|------|------|---------|-------------|
| AdamW 混合精度 | 16 B/参数 | 32.4 GiB | ✗ **硬性不可行** |
| **Adafactor 因子化** | fp32 参数 4 + fp32 梯度 4 + 行/列因子 O(m+n)≈0 ≈ **8 B/参数**（**已含**参数与梯度，勿重复叠加） | ~15.1 GiB + 激活 | ✓ **唯一可行** |
| + chunk 激活（micro=1，4096 tokens×8 tensors×bf16×31 层） | O(chunk)，与总 L 无关 | ~3.9 GiB | |
| **合计（chunk=4096，micro=1）** | | **~19.0 GiB** | ✓（实测闸门 ≤23 GiB） |
| （micro=2 时） | 激活 7.8 GiB | ~22.9 GiB | ✓ 极限档（OOM 即降 micro=1） |
| 朴素整段 FFT @512K | 激活随 L 线性 | >24 GiB | ✗ 不可行，仅作对照 |

公式实现：`simplcore/core/deploy/campaign.py` 的
`chunked_train_bytes(cfg, params, chunk, bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)`；
守护测试 `test_tuan_2b_budget.py` 把上表钉在算术上。

OOM 排障顺序：**降 chunk 4096→2048**（激活减半）→ 开 `--grad-checkpoint`
（重算换显存，~30% 计算开销）→ 降 micro-batch。**不要**降 global batch tokens
（有效 batch 决定收敛口径）。

## 3. 24h token 预算（欠训档，诚实口径）

```
有效 FLOPs(24h) = 86400 s × util 0.65 × 157e12 × MFU 0.30 ≈ 2.6e18
每 token FLOPs  ≈ train_flops_per_token(tuan-2b, 4096)      （scaling.py 解析式）
token 预算      ≈ 0.2B   （tokens/param ≈ 0.10）
```

* 这是**欠训档**：Chinchilla-optimal 的 2B 需要 ~40B tokens（20×N）≈
  **55 个 24h**。24h 训的是"完整 2B 参数"（全参参与更新、loss 稳定下降、
  checkpoint/评测链路完整交付），**不是** Chinchilla-complete 的 2B；
* 汇报口径（验收协议 X-equal-budget）：一切对照在**等 token / 等 FLOPs**
  下进行，`tokens/param` 实测值随结果一起报告，禁止把欠训档说成训练充分；
* Phase 0 实测吞吐后修正预算：`python -m simplcore.core.campaign --mfu <实测>`。

## 4. 极致优化清单（4090D 专项）

训练栈 `simplcore/core/training/train`（`apply_cuda_tuning` 已自动启用）：

| # | 优化 | 开关 | 说明 |
|---|------|------|------|
| 1 | bf16 autocast + fp32 主精度 | `--dtype bf16` | Ada tensor core；无 GradScaler 需要 |
| 2 | TF32 matmul | 自动 | fp32 GEMM 走 TF32（fp32 累加精度） |
| 3 | bf16 低精度归约 | 自动 | `allow_bf16_reduced_precision_reduction=True` |
| 4 | 展开式显存段 | 自动 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，24h 长跑防碎片 OOM |
| 5 | cuDNN benchmark | 自动 | 固定形状下选最快算法 |
| 6 | torch.compile max-autotune | `--compile` | inductor 内核自动调优；首步编译开销几十秒 |
| 7 | CUDA Graph（可选） | `--cuda-graph` | 固定形状单步 fwd+bwd 捕获，消 kernel 启动开销（与 compile/grad-checkpoint/NCM 互斥） |
| 8 | 分块流式训练 | `--train-chunk 4096` | 截断 BPTT：激活 O(chunk)；跨块状态携带逐位一致（P0-chunked-carry 闸门） |
| 9 | Adafactor 因子化 | `--optimizer adafactor` | 24GB 上 2B 全参的唯一可行优化器 |
| 10 | 梯度累积 | `--global-batch-tokens` | 有效 batch 与显存解耦 |
| 11 | 数据管线 | `--num-workers` | pin_memory + persistent_workers + non_blocking H2D |
| 12 | 24h 硬停 | `--time-budget-min 1440` | soft stop：落盘 checkpoint 后退出 |

CUDA Graph 与 `--compile` 互斥（二选一实测收益后锁定，Phase 0 步骤 3）。

## 5. Phase 0：栈验收清单（≤1 天硬停）

1. `bash scripts/train_tuan_2b_4090d.sh --dry-run`：随机 token 100 步，
   **OOM-free 且显存峰值 ≤ 23 GiB**（硬闸门 P0-oom-free-dryrun）；
2. 真实数据 ≥200 步：实测 tokens/s、MFU、显存峰值——修正 24h token 预算；
3. `--compile` vs `--cuda-graph` vs eager 三态各测 ≥50 步，锁定制胜组合；
4. 语料与词表冻结（SHA-256 manifest），文档级无泄漏切分；
5. 后端契约一致性（`conformance_checks` 16 项硬闸门，自动并入 Phase 0）。

## 6. Phase 1：24h 训练（硬停）

```bash
bash scripts/train_tuan_2b_4090d.sh          # 前台；或 nohup 后台 + 日志
```

* checkpoint 每 30 个 log 窗口落盘 `last.pt`（可断点续训 `--resume`）；
* eval 每 60 步（held-out loss 曲线必须稳定无发散——硬闸门 P1-loss-stable）；
* `--time-budget-min 1440` 到点软停：落盘最优 checkpoint 后退出；
* 训完立即复盘（不占训练窗口）：吞吐/MFU/token 预算达成率入档、
  held-out ppl + 长上下文检索快照、**tokens/param 实测值**（欠训档诚实标注）。

## 7. 验收协议（真实输出，禁止机制内对比）

与等参数 Transformer 基线**等 token / 等 FLOPs** 对照：held-out ppl、
长上下文检索（chunked 流式）、解码吞吐、流式==整段一致性。
禁止：注意力图等机制内对比、无限定宣称、自报自评、只报平均的 cherry-picking、
把 24h 欠训档说成训练充分。完整清单见
`python -m simplcore.core.campaign`（protocol + forbidden + 8 条可复验演示包）。

## 8. 诚实边界

* 本文所有 4090D 数字除规格值外均为**解析估算**，实机数字待 Phase 0/1 填充；
* 精度栈有定理契约（NT39–NT44，theory 账本 63 项）：bf16 递推舍入有界且与
  t 无关、fp32 梯度累积线性误差预算、Adafactor 因子几何收敛、欠训档是硬件
  预算定理（tokens/param ≤ 0.58 @ MFU=1）、且全部量化层共同保住下降方向
  （cos ≥ √(1−δ²)，tight）；
* 24h 单卡口径下 tokens/param ≈ 0.1——**欠训档**，不构成任何能力/SOTA 主张；
* 长上下文课程（`_CTX_2B_4M`）的 4M 末段在 24h 战役中按 `until_frac`
  相对口径缩放；4M 长程项的每 token FLOPs 增量 <1%（log L 优势，测试锁定）；
* 优化器选择（Adafactor）由 24GB 定容**硬性决定**：fp32 主精度保数值稳定，
  代价是无动量项（β1=0）——收敛性质以实测曲线为准。
