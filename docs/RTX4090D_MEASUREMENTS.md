# 24h·4090D 实测口径模板（Phase 0/1 回填表）

> **用途**：24h×4090D 战役的实机实测数字回填口径。所有字段在实机填充前为
> `null`——**禁止把解析估算当实测值填入**；解析估算（estimate）与实测
> （measured）并存时自动计算偏差，>20% 必须写原因。
> **机器可读**：`artifacts/4090d/measurement_template.json`（每字段带
> `caliber` 口径说明）
> **自动回填**：`python scripts/fill_measurements.py --run-dir runs/tuan-2b-4090d
> --out artifacts/4090d/phase0_measurements.json`（从 `run.json` +
> `metrics.jsonl` 读取；缺列保持 null，绝不编造）
> **守护测试**：`simplcore/core/tests/test_measurements.py`（结构完整性 /
> 口径强制 / 诚实 null / 回填算术 / REVIEW 规则）

## 1. Phase 0：栈验收回填表（≤1 天硬停）

| 字段 | 口径（caliber） | 解析估算 | 实机实测 |
|------|----------------|----------|----------|
| `oom_free` | dry-run 100 步全程无 CUDA OOM | 预期 true | ☐ |
| `mem_peak_gib` | `torch.cuda.max_memory_allocated()/2^30`，micro=2·chunk=4096 | 22.9 | ☐ |
| `mem_reserved_gib` | `max_memory_reserved()/2^30`（差值=碎片开销） | — | ☐ |
| `wall_sec_per_step` | metrics.jsonl step 时间中位数（预热 10 步后） | — | ☐ |
| `compile_warmup_sec` | 首步 − 稳态步时间差（torch.compile 编译开销） | — | ☐ |
| `tokens_per_sec` | (micro×seq×world)/step 秒，稳态中位数，≥200 步 | — | ☐ |
| `mfu` | tokens/s × F_tok / 157e12（F_tok = scaling 解析式） | 0.30（预设） | ☐ |
| 三态对照 | eager / compile-max-autotune / cuda-graph 各 ≥50 步 | — | ☐ ☐ ☐ |
| chunk 扫描 | chunk ∈ {2048, 4096, 8192}：tokens/s + mem_peak + oom | — | ☐ |
| `manifest_sha256` | data/packed/manifest.json 整体哈希（冻结） | — | ☐ |

**Phase 0 出口判据**：`oom_free = true` ∧ `mem_peak_gib ≤ 23`；MFU 实测值
回填 `python -m simplcore.core.campaign --mfu <实测>` 修正 24h token 预算。

## 2. Phase 1：24h 训练回填表（硬停）

| 字段 | 口径 | 计划/估算 | 实机实测 |
|------|------|-----------|----------|
| `tokens_total` | metrics.jsonl 末行 tokens（全局口径） | ~2e8 | ☐ |
| `tokens_per_param` | tokens_total / n_params —— **NT42 定理要求必报** | ≈0.10 | ☐ |
| `token_budget_vs_plan` | 实测 / MFU 修正后预算；≥0.9 达标（软闸门） | — | ☐ |
| `mem_peak_gib`（全程） | 含课程换挡后最长 seq 的分配峰值 | 22.9 | ☐ |
| `oom_events` | OOM 次数（>0 必须记录降档动作） | 0 | ☐ |
| `diverged` | val_loss 连续 3 次上升即 true 并停（硬闸门） | false | ☐ |
| checkpoints | 每 15 分钟：{wall_min, step, tokens, val_loss} | — | ☐ |
| eval_points | 每 30 分钟：{wall_min, val_loss, val_ppl} | — | ☐ |

## 3. 复盘回填表（立即，不占训练窗口）

| 字段 | 口径 | 实机实测 |
|------|------|----------|
| `final_val_ppl` | 最优 checkpoint held-out ppl | ☐ |
| `equal_token_baseline_ratio` | vs 等参数 Transformer 等 token 基线（X-equal-budget） | ☐ |
| `longctx_retrieval` | chunked 流式长上下文检索准确率 + 基线比值 | ☐ |
| `stream_consistency` | 流式 vs 整段最大偏差（应 ≈0） | ☐ |
| `undertrained_reported` | tokens/param 是否随结果一起报告（**NT42 义务**） | ☐ |

## 4. 偏差纪律（诚实规则）

* `measured=null` 且 `estimate!=null` 是**合法状态**（待实机）；
* 两者并存时自动计算 `deviation_pct`；**>20% 必须**把 `cause` 从
  `"REVIEW"` 改写为具体原因（如规格峰值与实测卡的差异、MFU 预设偏差、
  chunk 降档等）后才能作为汇报数字引用；
* 精度栈理论契约（NT39–NT44）给出的都是**界**而非点估计——实测显著低于
  上界属预期（随机舍入 ≈√t 增长远慢于最坏情况），不算偏差事故。

## 5. 与理论账本的对应

| 实测字段 | 对应定理 | 定理保证 |
|----------|----------|----------|
| `mem_peak_gib` | 战役显存账（campaign.py ADAFACTOR 口径） | 2B 全参 ≤23 GiB |
| `mfu` | NT42 | tokens/param ≤ 0.58 @ MFU=1（欠训档定理） |
| `tokens_per_param` | NT42 | 必报义务的来源 |
| loss 稳定性 | NT39/NT40/NT41/NT44 | 量化栈舍入有界、方向保持 |
| `stream_consistency` | NT34/NT36 | 流式==批量 |
