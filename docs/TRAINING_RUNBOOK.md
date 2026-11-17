# SimplCore 训练运行手册（单张 RTX 4090D-24GB）

> 适用对象：拿到一台 **RTX 4090D（24 GB）** 机器、仓库全新克隆、
> **`data/` 与 `runs/` 都是空的**的人。
> 目标：从零到"loss 在下降、checkpoint 在落盘、显存不炸"，全程只有一条路径，
> 不含分支猜测。
>
> 架构与理论优势见 `docs/ARCHITECTURE.md` 与 `docs/THEORETICAL_EDGE.md`；
> **24h 能训多大**见 `docs/RTX4090D_TRAINING.md` 与
> `python -m simplcore.core.campaign`。
> 本手册只讲**操作顺序**与**排障**，不重复推导。
> （历史 MI250/A100 运行手册已随 2026-11-18 硬件迁移删除。）

---

## 0. 前置检查

```powershell
.\.venv311\Scripts\python.exe -V          # 期望 3.11.x
.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望看到 `True` 与 `NVIDIA GeForce RTX 4090D`（或同级 Ada 卡）。
若 `False`，先解决驱动，**不要**继续往下走 —— 后面每一步都在 GPU 上。

> 注：训练路径是 `simplcore/core/` 的 **PyTorch** 实现
> （`simplcore/core/training/train/`），不依赖自研 `simplcore/tensor` 引擎。

## 1. Phase 0：栈验收（≤1 天硬停）

```powershell
# 1) dry-run：随机 token 100 步，验证显存/吞吐链路（不落数据）
bash scripts/train_tuan_2b_4090d.sh --dry-run

# 2) 打印战役账本（显存 ~19 GiB 估算 / 24h token 预算 / 全部闸门）
python -m simplcore.core.campaign
```

通过判据（硬闸门）：dry-run OOM-free 且显存峰值 ≤ 23 GiB；
`campaign` 的 `schedule.holds == true`。

## 2. 数据打包

```powershell
python scripts/build_corpus.py --out data/packed    # SHA-256 manifest 自动落盘
```

打包产物的 `manifest.json` 记录 vocab_size 与分片哈希；训练启动时会校验
（`--skip-hash-check` 可跳，但 24h 长跑不建议）。

## 3. Phase 1：24h 训练（硬停）

```powershell
bash scripts/train_tuan_2b_4090d.sh
```

关键参数已固定在启动器里（勿随意改，改动要过战役闸门）：

| 参数 | 值 | 为什么 |
|------|----|--------|
| `--optimizer adafactor` | 因子化二阶矩 | **24GB 上 2B 全参唯一可行**（AdamW 16B/参 = 32 GiB 直接爆） |
| `--dtype bf16` | bf16 前向 + fp32 参数/梯度 | TF32/bf16 tensor core；IIR 递推恒 fp32（数值可信前提） |
| `--train-chunk 4096` | 分块流式 | 激活 O(chunk)，与总上下文 L 无关 |
| `--global-batch-tokens 65536` | 有效 batch | 与显存解耦（梯度累积） |
| `--time-budget-min 1440` | **24h 硬停** | soft stop：落盘 checkpoint 后退出 |

## 4. 断点续训 / 排障

* **OOM**：降 `TRAIN_CHUNK=2048` 重启（激活减半）→ 开 `--grad-checkpoint` →
  降 `MICRO_BATCH=1`。不要动 `--global-batch-tokens`。
* **中断续跑**：启动器自带 `--resume runs/tuan-2b-4090d/last.pt`，直接重跑同一命令。
* **loss 发散**：先查 `runs/tuan-2b-4090d/metrics.jsonl` 的 lr 与 grad-norm 列；
  warmup 200 步内属正常爬坡；确认 `--grad-clip 1.0` 生效。
* **吞吐远低于预算**：跑 `python scripts/bench_train_throughput.py` 对照
  fp32/bf16 与形状缩放；确认 `--compile` 生效（日志第一段有
  `torch.compile enabled`）。

## 5. 训后复盘（立即，不占训练窗口）

1. 汇总 `metrics.jsonl`：实测 tokens/s、MFU、显存峰值 vs 解析账本；
2. held-out ppl + 长上下文检索快照（与等 token 基线对照，协议
   `X-equal-budget`）；
3. **tokens/param 实测值入档**（NT42：24h×4090D×2B 在任何 MFU 下都是欠训档，
   该数字是定理要求的汇报义务，不是可选项）。
