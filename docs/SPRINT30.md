# Sprint-30：1 个月内真实确保完成训练（单卡 MI250）

> 模块：`simplcore/core/sprint30.py` · CLI：`python -m simplcore.core.sprint30`
> 与 90 天 v2 战役（`campaign.py`）并存：v2 管"1B→3B 战役"，本文管
> **"1 月必须完全做好训练"的硬保证**。

## 1. 为什么 v2 的"30 天"不等于"真实确保"

campaign v2 的 token 预算建立在 **25% MFU** 假设上；自研 HIP 后端尚未
验收，rocFFT 融合路径的现实 MFU 下限在 **0.12** 一带。用纸面 MFU 定预算，
超支时只能延期——这与"1 月必须完全做好训练"矛盾。

## 2. 三根支柱

1. **测后锁预算**（D2）：512K 上下文 8192-chunk 分块冒烟 → 实测 MFU →
   `locked_token_budget = min(20B, 有效算力 / 每 token FLOPs)`，
   此后训练中段不再改预算。
2. **保守下限闭合**：全部账本在 MFU=0.12 闭合（`closes_at_floor` 硬断言）。
3. **降档阶梯**：实测后 1B 档预算 < 15B → 自动切 `tuan-870m`（0.866B，
   每 token FLOPs 5.735G→4.91G）最大化 token 预算。
   MFU≥0.15 保 1B（≥15B）；MFU<~0.14 时两档均 <15B，账本照实呈现——
   **交付出一个训完的模型，好过一个没训完的 1B**。

## 3. 解除 Phase 0 阻塞

训练主路径走 **PyTorch + torch.compile → ROCm CK**（README 运行时口径），
自研 HIP 后端移出关键路径、训练期并行推进。**第 1 天即开训**。

## 4. 两种模式

| 模式 | 分配 | 基座 token 预算（MFU 0.15 / 0.25） |
|------|------|-----------------------------------|
| `backbone30`（默认） | 基座 30 天；2B 完整学习按 campaign v2 续期 | 16.0B / 20B（Chinchilla 达标） |
| `allin30`（一个月全要） | 基座 22d + 复盘 1d + 2B 蒸馏 7d | 1B: 11.7B / 15.6B；低 MFU 自动降档 |

全部场景 `closes=True`（预算 × FLOPs/token ≤ 有效算力，30 天硬停 D30）。

## 5. MFU 情景表（backbone30，tuan-1b）

| MFU | 30 天有效 FLOPs | tokens | ≥20B | ≥15B |
|-----|----------------|--------|------|------|
| 0.10 | 4.47e19 | 7.8B | ✗ | ✗ → 降档 12.4B |
| 0.12 | 5.37e19 | 9.4B | ✗ | ✗ → 降档 14.9B |
| 0.15 | 6.71e19 | 16.0B | ✗ | ✓ |
| 0.20 | 8.95e19 | 20.0B | ✓ | ✓ |
| 0.25 | 1.12e20 | 20.0B | ✓ | ✓ |

## 6. 日程与闸门（backbone30）

D1 冻结 → D2 实测锁预算 → D3–6（1M）→ D7–12（2M）→ D13–26（4M 主训）
→ D27–28 评测+失败轨迹 → D29–30 2B 分支起步蒸馏 → **D30 硬停**。
翻倍 HBM 增量 ≤1 GiB 与"流式==并行逐位一致"作为窗口闸门随行。

## 7. 先跑 Pilot：A1/A2 最小证伪实验（`pilot.py`）

30 天战役押在最大的未验证主张上——**自主改进闭环真的能持续学习且不退步**。
回答它不需要 1B×30 天：`simplcore/core/pilot.py` 用 dim=32 小模型 +
真实代码沙箱 + 若干轮闭环（交互→总结→蒸馏→no-regression gate→回滚），
CPU 几分钟出信号：

```bash
python -m simplcore.core.pilot --rounds 2 --out runs/pilot        # CPU 冒烟
python -m simplcore.core.pilot --rounds 8 --dim 256 --layers 6 \
    --consolidate-epochs 4 --out runs/pilot-gpu                   # 实机档位
```

判定口径：A1 = 闭环前后真实执行 pass_rate（学习信号与生成质量**分开记账**）；
A2 = 每轮 no-regression gate 的提交/回滚计数。建议排期：**Sprint-30 D2
实测 MFU 的同一窗口顺手跑 pilot**；pilot 闭环信号不成立 → Phase 2 的
2B 自主学习预算改投 token 预算（降档阶梯的一部分）。

## 8. 诚实边界

本计划保证**时间账本闭合 + 预算锁定机制**，不保证"1B 打 3B Transformer"；
后者仍以 campaign 协议 X-* 的实测输出验收。MFU 是全部差异的来源：
D2 实测值决定最终交付形态（1B@16–20B tokens，或 870m@更低 MFU）。
