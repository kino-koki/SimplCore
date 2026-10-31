# SimplCore 训练运行手册（1 × AMD Instinct MI250 / A100 80GB）

> 适用对象：拿到一台 MI250（或 A100 80GB）机器、仓库全新克隆、
> **`data/` 与 `runs/` 都是空的**的人。
> 目标：从零到"loss 在下降、checkpoint 在落盘、显存不炸"，全程只有一条路径，
> 不含分支猜测。
>
> 架构与理论优势见 `docs/ARCHITECTURE.md` 与 `docs/THEORETICAL_EDGE.md`；
> **1 个月能训多大**见 `docs/TRAINING_BUDGET_MI250.md`。
> 本手册只讲**操作顺序**与**排障**，不重复推导。

---

## 0. 前置检查

```powershell
# 唯一受支持的环境（README 锁定）
.\.venv311\Scripts\python.exe -V          # 期望 3.11.x
.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- **MI250（ROCm）**：期望看到 `True` 与 `AMD Instinct MI250`。
  注意 MI250 是**双 GCD 封装**，ROCm 通常暴露为两张卡 —— 训练要用
  FSDP / ZeRO-2/3 跨 2 个 GCD 切分，不能当成单设备 128 GB 用。
- **A100**：期望看到 `True` 与 `A100-SXM4-80GB`（或 `A100-PCIE-80GB`）。

若 `False`，先解决驱动，**不要**继续往下走 —— 后面每一步都在 GPU 上。

> 注：训练路径是 `simplcore/core/` 的 **PyTorch** 实现（`simplcore/core/training/train/.py`），
> 不依赖自研 `simplcore/tensor` 引擎 —— 该引擎仍缺可微 `rfft/irfft`，
> 只在 `research/` 里作为历史原型保留。

---

## 1. 重建语料（仓库清理后必做）

两步：

```powershell
# 1) 拉原始语料（约 1.4 GB，走 hf-mirror）
python scripts/download_corpus.py --out data/raw/fineweb_zh

# 2) 训练分词器 + 清洗 + 打包定长分片（产出 tokenizer.json / *.bin / index.json）
python scripts/build_corpus.py --parquet-dir data/raw/fineweb_zh --out data/processed/tuan_zh `
  --vocab-size 32000 --seq-len 4096 --val-frac 0.005
```

验收：`data/processed/tuan_zh/index.json` 里 `seq_len=4096`、`vocab_size=32000`，
且 `train-00000.bin` 等分片带 SHA-256。分片哈希用于训练启动时的完整性校验，
**不要手工删改**（否则用 `--skip-hash-check` 启动，但那就失去了防损坏能力）。

`simplcore/core/training/train/.py` 会按顺序找 `data/processed/tuan_zh` → `data/processed/tuan_zh_smoke`，
所以默认路径建好即可省掉 `--data-dir`。

---

## 2. 先测显存，再定 batch（**不要跳过**）

```powershell
python scripts/probe_batch.py --config configs/tuan-1_3b.json --seq-len 4096
```

它做的事：二分搜索最大可容纳 micro-batch、读 `torch.cuda.max_memory_allocated`
报告**真实**峰值、测 fwd / fwd+bwd 时延与 tokens/s、反算达到目标 token 预算所需的
梯度累积步数。

把输出的三项记下来：

| 记什么 | 用来干什么 |
|---|---|
| 最大 micro-batch | 填 `--micro-batch`（留 10% 余量，别用满） |
| 峰值显存 MB | 判断能否开 `--grad-checkpoint` 换更大 batch |
| tokens/s | 反推 `--max-steps` 与训练时长 |

SimplCore 显存的瞬时项随序列长度**线性**增长（FFT 路径的中间量），
所以 seq_len 从 4096 翻到 8192 时，账不能按"注意力二次"估，但也**不能**按常数估。

---

## 3. 空跑自检（不碰数据，30 秒）

```powershell
python -m simplcore.core.train --preset tuan-150m --dry-run --max-steps 20
```

`--dry-run` 用随机 token 走完 fwd/bwd/step/写 checkpoint 全链路，用来验证：
CUDA 可用、dtype 选择正确、checkpoint 目录可写、四层闸门没有形状错误。

---

## 4. 正式训练

```powershell
python -m simplcore.core.train --config configs/tuan-1_3b.json --seq-len 4096 `
  --micro-batch 4 --global-batch-tokens 2000000 `
  --lr 3e-4 --lr-min 3e-5 --warmup 200 --max-steps 100000 `
  --grad-clip 1.0 --eval-every 250 --save-every 500 `
  --out-dir runs/tuan-1_3b
```

### 4.1 训练增强（`--enhance`，opt-in、默认关闭、零回归）

六个控制器（accel / nonlinear / reasoning / envlearn / smm / dream）已通过
`simplcore.core.training_integration` 接入训练闭环。**不开 `--enhance` 时训练逐位不变。**

```powershell
# 把"思考算力"训进模型（仅 ponder；其余默认关）
python -m simplcore.core.train --preset tuan-150m --enhance ponder --dry-run

# 评估期叠加加速比 / 记忆 / 推理（不改主 val_loss；ponder 与 envlearn 互斥，后者覆盖前者）
python -m simplcore.core.train --preset tuan-1b --enhance accel,smm,reasoning

# reasoning 自定义提示：文件（每行一条）或单条字符串
python -m simplcore.core.train --preset tuan-1b --enhance reasoning `
  --enhance-prompts prompts/reasoning.txt
```

- `ponder` / `envlearn`：替换训练前向，直接改变训练信号（前者把 latent 思考 + ponder 正则
  训进模型；后者冻结基座、只训 `LearnedAdapter`，`WeightBudget` 守 3B 上限）。
- `accel` / `smm` / `reasoning`：评估 hook，分别报结构性加速比、把采样窗口写入情景记忆库、
  跑迭代式深度思考——不改变主 val_loss。
- `dream`：训练后 post-hook，跑 `ReflectionLoop` 自进化巩固（no-regression gate 逐位回滚；
  权重写回需独立 held-out 评测器，否则 gate 默认拒绝写回）。
- `smm`/`dream`/`reasoning` 需要编解码器：未显式传入时自动用 `CharTokenizer(limit=vocab_size)`
  兜底。单元验证见 `simplcore/core/tests/test_training_integration.py`（CPU 可跑）。

---

关键参数为什么这么设：

- **`--global-batch-tokens`**：按 token 数给预算，`grad_accum` 由它自动反算。
  用 token 而不是 batch size 描述预算，换 seq_len 时不必重算。
- **`--grad-clip 1.0`**：振荡器参数（`raw_rho`/`raw_theta`）在早期容易一步跳飞，
  裁剪是必需的，不是保险。
- **weight decay 只作用于权重矩阵**：`raw_rho`、`raw_theta`、`feed`、`out_scale`、
  所有 norm 与 embedding 都不衰减（`train.py: build_param_groups`）。
  对衰减率参数做 weight decay 等于直接掐死长程记忆。
- **`--compile`**：先跑 `--max-steps 50` 对比有/无的 tokens/s，有收益再常开。
  首个 step 的编译开销约几十秒，短跑里看不出来。
- **`--grad-checkpoint`**：显存吃紧时用它换 batch。FFT 路径重算成本不低，
  实测收益要看 `probe_batch.py` 的数，不要盲开。

### MI250 专用提速（已在 `train.py` 落地）

仓库的训练循环已针对 MI250 / ROCm 做了真实优化，**默认只在探测到 ROCm 时自动生效**，
不碰 CPU/默认路径：

- **`torch.compile --compile-mode max-autotune`**：ROCm 上 inductor 选 **Composable
  Kernel(CK)** 内核，matmul/conv 显著快于朴素 HIP 内核。第一个 step 编译开销几十秒，
  短跑看不出来——先 `--max-steps 50` 对比有/无的 tokens/s 再常开。
- **ROCm 分配器调优**：自动设 `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
  （128GB HBM 减碎片），并开 `cudnn.benchmark=True`（固定形状下让 MIOpen 选最快卷积）。
- **去掉每步 `cuda.synchronize()`**：改事件计时，CPU 不每步 stall，数据加载与下一
  step 的 CPU 侧工作能与 GPU 重叠。
- **HIP Graph（`--cuda-graph`，实验性）**：捕获单 micro-batch 的 fwd+bwd，消除每步
  kernel 启动开销（AMD 上启动开销显著）。**单卡 + 未开 grad-checkpoint + 未开 compile**
  时才尝试，**失败自动回退 eager**。先不动它，验证 MFU 后再开。
- **早停 + 最佳检查点回滚**（与 `sft.py` 对齐）：`--early-stop-patience N`
  `--early-stop-min-delta` `--restore-best`，记忆场景下末态过拟合，`restore_best` 让
  最终 `last.pt` 即最佳 val 检查点。

> MI250 是**双 GCD 封装**，ROCm 暴露为 2 张卡；`torchrun --nproc_per_node=2` 用 DDP
> 跨 2 个 GCD 训练（1B/16GB 优化器状态塞进单 GCD 64GB 绰绰有余）。不要用单设备 128GB 的
> 思路配 batch。

**一行启动**（参数含义见脚本头部注释，`probe_batch` 先定 micro-batch）：

```bash
bash scripts/train_tuan_mi250.sh --preset tuan-1b --micro-batch 4 \
  --global-batch-tokens 2000000 --max-steps 100000 \
  --data-dir data/processed/tuan_zh
```

脚本已预设 ROCm 环境（`ROCR_VISIBLE_DEVICES=0,1`、`HSA_FORCE_FINE_GRAIN_PCIE=1`、
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`、`OMP_NUM_THREADS=8`）与编译/早停开关；
`--no-compile` / `--cuda-graph` / `--no-early-stop` 等可微调。

### 断点与续跑

```powershell
python -m simplcore.core.train --config configs/tuan-1_3b.json --resume runs/tuan-1_3b/step_005000.pt
```

`--time-budget-min 480` 到点会写完 checkpoint 再退出，适合抢占式机器/租用实例。

---

## 5. 评测

```powershell
# 流式一致性（分段前向 vs 整段前向，逐 token 对齐）
python -m pytest simplcore/core/tests/test_stream_checkpoint.py simplcore/core/tests/test_eval.py -q

# 吞吐与显存（实机 MFU）
python -m simplcore.core.eval throughput --preset tuan-150m

# O(1) 流式状态（状态字节数必须与序列长度无关）
python -m simplcore.core.eval streaming --preset tuan-150m
```

`eval` 的子命令：`evaluate` / `throughput` / `streaming` / `probe` / `compare` / `self-test`。
结果写 `bench_*.json`（与仓库内已有的同结构，可直接对比，不要改字段名）。

---

## 6. 排障

| 症状 | 原因 | 处置 |
|---|---|---|
| `FileNotFoundError: index.json` | 语料没建或路径不对 | 回到 §1；或用 `--data-dir` 显式指定 |
| 启动即报分片 SHA-256 不匹配 | 分片被改/传输损坏 | 重建语料；临时用 `--skip-hash-check` 定位哪片坏了 |
| step 3000 左右 OOM | micro-batch 按算术估的，没实测 | 回到 §2 重新测；降 micro-batch 或开 `--grad-checkpoint` |
| loss 抖动后发散 | 未裁梯度 / lr 过高 | 确认 `--grad-clip 1.0`；降 lr 到 1e-4 重跑 |
| loss 长期平坦不降 | 闸门退化（历史 bug：`evidence_proj` 零初始化被覆盖） | 检查 `simplcore/core/arch/abstain.py` 的闸门参数是否可学习；跑 `test_gates.py` |
| tokens/s 只有预期一半 | `--num-workers 2` 供不上 | 提到 8；确认语料在本地 SSD 不在网络盘 |
| ROCm 上 `torch.compile` 报 unicode 错 | Windows 临时目录中文用户名 | 属已知问题，`--compile` 关掉即可（eager 结果一致） |
| ROCm 上 FFT 明显慢于 NVIDIA | cuFFT / rocFFT 实现差异 | 以 `eval throughput` 的实测 MFU 为准，按 `TRAINING_BUDGET_MI250.md` 下调 token 预算 |

---

## 7. 产物与空间

| 路径 | 内容 | 能否删 |
|---|---|---|
| `runs/<name>/step_*.pt` | checkpoint（含 optim 状态） | 保留最近 2 个即可 |
| `runs/<name>/metrics.jsonl` | 逐步 loss / lr / tokens/s | 保留（画曲线用） |
| `data/processed/` | 打包后的分片 + manifest | 可删，按 §1 重建 |
| `data/raw/` | 原始 parquet | 可删，按 §1 重新下载 |

单次 1.3B / 100k step 的 checkpoint 约 2 × 2.7 GB（权重 + Adam 状态），
按 `--save-every 500` 会攒下 200 份 ≈ 1 TB —— **务必定期清旧 checkpoint**，
或改用 `--save-every 2000`。

---

## 8. 纪律

三件在本仓库反复踩过的事，写在这里：

1. **不要凭算术设 batch。** 先测。
2. **不要在没有 checkpoint 的情况下跑超过 1 小时的任务。** `--time-budget-min` 常开。
3. **不要把"待测"写成"已实现"。** 没跑过的数字不留空口结论，
   写进 `docs/TRAINING_BUDGET_MI250.md` 的实测栏与 issue 待办里。
