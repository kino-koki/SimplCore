# Tuan-2B × 30 天 × 单 MI250：可执行训练计划

> 目标：**单张 AMD Instinct MI250（2×64 GB GCD），30 天内完成 2B 主干 + 4M 上下文课程**。
> 结论先说：**算力只够 2B × ~12.4B tokens**，约为 Chinchilla 20×（40B）的三分之一。
> 因此本文把"完全训练"定义为：**跑完 4M 课程、产出可用的 2B 主干与 4M 可运行/基础检索能力**，
> 而不是"达到 2B 的计算最优"。要 40B tokens 需 ~92 天，或 3 张同档卡。

---

## 1. 算力核算（决定一切的数字）

| 项 | 值 |
|---|---|
| MI250 BF16/FP16 矩阵峰值 | 362.1 TFLOPS（整卡，双 GCD 合计） |
| 有效训练占比（扣调试/评测/重启） | 0.65 |
| 目标 MFU | 25%（ROCm + FFT + 长上下文的保守值） |
| 30 天有效 FLOPs | `30 × 86400 × 0.65 × 362.1e12 × 0.25 = 1.525e20` |
| `tuan-2b` 每 token FLOPs（4K） | 11.83e9 |
| 每 token FLOPs（4M） | 11.85e9（FFT 的 `log L` 项只增加 **0.16%**） |
| **30 天 token 上限** | **~12.9B** |
| 本计划排课 | **~12.4B**（留 4% 余量给评估/重启） |

关键事实：**4M 上下文对算力的影响几乎为零**（长程项随 `log L`，不是线性）。真正的约束是
**数据与窗口数**，不是 FLOPs。

## 2. 模型口径（`tuan-2b`）

- 总参数 **2.024B**，非嵌入 **1.959B**，嵌入占比 **3.2%**。
- `dim=2048, layers=31, num_bases=128, ffn_ratio=4.0, num_heads=16`。
- 与 `tuan-1b` 同宽（kernel/调优可复用），深度翻倍；宽深比 66。
- **极点参数化**：`decay_max=4e6`、`sgc_init_mode="logfreq"`、
  `sgc_rho_param="logspace"`。4M 需要 `ρ≈0.99999975`；旧 sigmoid 初始化在
  `decay_max=4e6` 时仍被 `_logit` 夹到 `0.9999`（实际只铺到 1e4），
  log-space 才能让最慢基真正覆盖 4M（`test_rho_logspace` 守护）。
- 每 GCD 显存（chunk=4096, micro=2）：
  - AdamW 状态 ~31.3 GiB + bf16 权重 4.0 + 梯度 4.0
  - 单块激活 ~8.3 GiB + FFT workspace ~1 GiB
  - **合计 ~49 GiB / 64 GiB GCD**，余 ~15 GiB 给碎片与临时量。

## 3. 30 天课程（与 `scripts/train_tuan_2b_mi250.sh` 一致）

| 阶段 | seq_len | micro | tokens/step | steps | tokens | 估时@25% MFU |
|---|---|---|---|---|---|---|
| S1 | 4K | 2 | 1.00M | 2300 | 2.30B | ~3.5d |
| S2 | 16K | 2 | 1.02M | 2300 | 2.34B | ~3.6d |
| S3 | 64K | 2 | 1.05M | 2300 | 2.41B | ~3.7d |
| S4 | 256K | 1 | 1.05M | 1950 | 2.04B | ~4.4d |
| S5 | 1M | 1 | 1.05M | 1950 | 2.04B | ~4.6d |
| S6 | 4M | 1 | 4.00M | 325 | 1.30B | ~5.0d |
| 合计 | | | | 11125 | **12.43B** | **~25d 纯算力** |

- 每段用 `--resume runs/tuan-2b/last.pt` 接续，step 全局累计。
- 梯度窗口固定 `--train-chunk 4096`；`micro × chunk ≤ 8192`，保证单块激活可控。
- S6 每步 4M tokens：**325 个 4M 窗口** ≈ 1.3B tokens。这是"4M 可运行 + 基础检索"，
  不是"4M 完全收敛"。
- 若 MFU 实测 <15%，按 §7 降档，不要硬跑满 30 天换来一个欠训的 2B。

## 4. 数据要求（比算力更容易成为瓶颈）

- packed 语料（`simplcore.core.data` 的 manifest），train ≥ **14B tokens**，val ≥ 100M。
- 4M 阶段要求**连续窗口**：manifest 的文档拼接必须允许 4M token 不断档；否则 S6 的训练
  窗口会跨空区，长程样本名存实亡。
- 磁盘：每份 checkpoint ≈ **36 GiB**（权重 4 + 优化器 32）。至少 **600 GB** 本地 SSD；
  只保留 `last.pt` + `best.pt`，中间步删。
- 数据质量闸门：`read_manifest` 的 sha256、split 无交集校验必须通过；`claim_permitted`
  在合成语料下为 False，不得用合成数据对外宣称能力。

## 5. 上卡步骤

```bash
# 0) 环境自检（ROCm 应暴露 2 张 GCD）
python -c "import torch; print(torch.version.hip, torch.cuda.device_count())"

# 1) 先定 micro-batch / chunk（不要跳过；2B 的 64GB GCD 余量只有 ~15GB）
python scripts/probe_batch.py --preset tuan-2b --seq-len 4096

# 2) 单阶段 50 步做吞吐/MFU 标定（编译已禁用，chunk 路径不走 compile）
bash scripts/train_tuan_2b_mi250.sh \
  --data-dir data/processed/tuan_zh --out-dir runs/tuan-2b-probe \
  --seq-len 4096 --max-minutes 30

# 3) 正式 30 天（分段自动 resume）
bash scripts/train_tuan_2b_mi250.sh \
  --data-dir data/processed/tuan_zh --out-dir runs/tuan-2b

# 4) 只跑某一阶段（调试/续跑）
bash scripts/train_tuan_2b_mi250.sh --data-dir ... --seq-len 1048576
```

**为什么不用 `--compile`**：`--train-chunk` 与 `torch.compile` 互斥（自定义逐块前向不会被
编译），且 `--cuda-graph` 也不兼容。MI250 的提速改由以下项承担：
`expandable_segments`、固定 `chunk` 形状、`--train-chunk` 降低激活、DDP 双 GCD、以及
每块前向的 FFT 批量形状固定。

## 6. 监控与硬停

- `runs/tuan-2b/metrics.jsonl`：每 `--eval-every` 记录 `loss / tokens_per_s / grad_norm /
  step_s`；用 `tokens` 字段核对 token 预算，而不是用步数。
- 每 1000 步落 `last.pt`；`best.pt` 只按 val loss。
- **硬停**：`--max-minutes 43200`（30 天）在总预算耗尽时写完 checkpoint 退出；
  段级失败（NaN、OOM、数据校验）立即停，不自动降档。
- **GATE-2（1M 阶段自动触发）**：`scripts/train_tuan_2b_mi250.sh` 在 S5 加
  `--gate2`，训练首次进入 1M 时跑一次 needle 召回，写 `runs/tuan-2b/gate2.json`。
  默认只记录，不阻塞（`below_chance` 才提示管线异常）；4M 验收用离线 CLI
  `python -m simplcore.core.gate2 --checkpoint runs/tuan-2b/last.pt --lengths 1048576,4000000`。
  指标与阈值见 `docs/GATE2_NEEDLE.md`：`at_chance` 是轴 D 的预期，不是失败。
- 每日检查：`loss` 是否单调（含噪声）、`grad_norm` 是否爆炸、`tokens/s` 是否掉到标定值的 70%。

## 7. 降档预案（按优先级，不挪用 4M 阶段）

1. **MFU < 15%**：把 `tuan-2b` 换成 `tuan-1_3b`（同 4M 课程，token 上限 ~19B），
   或在 2B 上把 token 目标降到 ~8B 并明确标注欠训。
2. **显存不足/OOM**：`--chunk 2048`（激活减半，吞吐降 ~15%），或 S1-S3 用 `micro=1`。
3. **4M 数据窗口不足**：S6 降到 1M，4M 只保留推理/评测；不得用重复窗口凑 token。
4. **ROCm FFT 异常慢**：先确认 `fft_dtype=float32` 与 `chunk` 形状固定；若仍慢，
   把 S5/S6 的比例下调，保住 S1-S3 的通用能力。

## 8. 验收（30 天结束时必须能拿出的东西）

- `last.pt` / `best.pt` + `metrics.jsonl` + `run.json`（含实际 tokens、MFU、峰值显存）。
- 4M 推理：`python -m simplcore.core.deploy_budget`（HBM 账）；
  `python -m simplcore.core.gate2 --checkpoint runs/tuan-2b/last.pt
  --lengths 1048576,4000000 --chunk-size 4096 --device cuda`
  （needle 召回；`at_chance` 是预期结果，记录而不改写为"赢"）。
- 流式一致性：`test_stream_checkpoint` + 实际 4M 上下文逐 token 生成日志。
- **诚实标注**：val loss、token 数、4M 窗口数、以及"未达 Chinchilla"的事实。

## 9. 风险与未实测项

| 风险 | 状态 | 处置 |
|---|---|---|
| DDP + 分块逐块 backward（no_sync 跨块/跨 micro-batch） | 代码路径已实现，**CPU 单进程等价已验证；2 GCD 实机未跑** | 上卡第一步用 50 步 probe 验证 loss/grad_norm 与单进程一致 |
| ROCm 上 `torch.fft` 的吞吐 | 未知 | probe 阶段测 tokens/s；低于 15% MFU 触发降档 |
| 4M 连续窗口的数据供给 | 取决于语料 | 先跑 manifest 校验；不足则 S6 降 1M |
| checkpoint 36 GiB × 写入频率 | 磁盘/IO 压力 | `--save-every 1000`，只留 last/best |
| 2B 在 64GB GCD 的余量 | 估算 ~15 GiB | 先 `probe_batch`，OOM 则 `--chunk 2048` |

> 本计划的确定性来自算术（FLOPs/tokens/显存），不来自"跑过"。当前仓库环境没有 MI250，
> 上卡前不得把本文件当作实测结果引用。