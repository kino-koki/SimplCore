# RTX 4090D-24GB：1B × 24h 战役（目标：同 token 对照完全超越 Mamba2）

> **用户口径**：24h 训 1B，直到完全超越 Mamba2。启动器
> `scripts/train_tuan_1b_4090d.sh`；对照协议 = `X-equal-budget`
> （等 token/等 FLOPs，真实 held-out ppl + 长上下文检索 + 解码吞吐）。
> **现状基线**（4M token 前哨，D=128/4 层，单种子，char 级中文语料）：
> sgc 8.13 vs Mamba2 7.07（**1.15×**）；sgc-bind 10.04（绑定通道拖累
> 纯 LM +21%）。1B/24h 档是这一前哨的放大验证，不是同一实验。

## 1. 24h 预算（NT42/NT46 口径）

| 项 | 值 | 来源 |
|----|----|------|
| 模型 | tuan-1b：1.013B（dim 2048 / 15 层 / FFN 4.0 / vocab 32000） | config 预设 |
| F_tok @4096 | 5.73e9 | scaling 解析式 |
| 24h token 预算 | **0.46B @ MFU 0.30**（0.77B @ 0.5，1.54B @ 1.0） | effective_flops |
| tokens/param | 0.46（@0.30）——欠训档，NT42 同款必报 | NT42 |
| 显存 | Adafactor 7.5 GiB + 激活 ~1.9 GiB（micro=8）≈ **9.5 GiB** < 23 | 账本 |
| max-steps（NT46） | **14063** = 0.46e9 / 32768（micro=8·seq4096） | 启动器已内置 |

比 2B 战役宽裕：micro_batch 8（vs 2）、AdamW 也可放下（15.1 GiB）——
为与 2B 同栈可比默认仍 Adafactor，`OPTIMIZER=adamw` 可切换做 A/B。

## 2. 差距解剖：1.15× 从哪来（按证据强度排序）

前哨证据（1M/4M 真实语料、等 token）指出的三个可攻击点：

1. **绑定通道拖累纯 LM（+21%）**：`lm_tuning.json` 根因定位——sgc-bind
   在纯 LM 上比 sgc 差 21%。24h 1B 主训**默认关绑定通道**
   （`mixer=sgc`），把 sgc-bind 留给检索/长上下文阶段（若需要）。
   这一条直接拿回 21% 里的大部分，是**零风险最大单项**。
2. **MFU 缺口**：Mamba2 的 selective-scan 是 fused CUDA 核；我们的
   FFT 路径走 cuFFT + Python 侧编排。Phase 0 三态对照
   （eager / compile / cuda-graph）+ chunk 扫描是**实测动作**，不是
   猜测——吞吐每多 10%，token 预算多 10%。
3. **容量/结构残差**：同 token 下 Mamba2 的选择性（输入相关衰减）对
   真实文本的建模优势是文献口径；我们的对应物是 sgc_path_gate /
   srf（输入相关 Δ）——**1B×24h 档先不动架构**（保持与 4M 前哨同构，
   差距归因才干净），架构变体作为第二战役。

## 3. 执行计划

### Phase 0（≤1 天硬停）
1. `bash scripts/train_tuan_1b_4090d.sh --dry-run`：OOM-free + 显存峰值 ≤23 GiB；
2. 真实数据 ≥200 步：实测 tokens/s、MFU → 修正 `MAX_STEPS`（NT46）；
3. 三态吞吐对照（eager/compile/cuda-graph）+ chunk {2048,4096,8192} 扫描，锁定配置；
4. 语料冻结（SHA-256）。

### Phase 1（24h 硬停）
1. `bash scripts/train_tuan_1b_4090d.sh`（max-steps 14063，NT46 校准）；
2. **同 token 对照臂**：同数据、同 0.46B tokens 训一个等参 Transformer
   基线（或引用 Mamba2 同 token 实测），产出 X-equal-budget 对照；
3. 每 15 分钟 checkpoint、每 30 分钟 eval；loss 曲线硬闸门。

### 复盘（立即）
- held-out ppl vs Mamba2/Transformer 同 token 比值；**超越判据 =
  ratio < 1.0**（ppl 意义），长上下文检索 ≥ 基线，解码吞吐 ≥ 基线
  （KV cache vs O(1) 流式状态的结构优势项）；
- `fill_measurements.py` 回填全部口径；tokens/param 必报（NT42）。

## 4. 诚实边界

* 24h×1B 的 token 预算（~0.46B）下，"超越 Mamba2"是**同 token 等预算
  对照**口径，不是绝对能力 SOTA；tokens/param ≈ 0.45 的模型与充分训练的
  Mamba2 不可比，报告中必须携带 tokens/param（NT42 义务）；
* 4M 前哨的 1.15× 是 char 级小模型结论；1B/word-level/真实语料上的
  差距可能不同（更大也可能更小）——这正是本战役要实测的；
* 本机（无卡）只能交付：启动器、账本、协议、模板与守护测试；实机
  数字全部按 `docs/RTX4090D_MEASUREMENTS.md` 口径回填。
