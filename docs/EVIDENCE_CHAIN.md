# SimplCore 证据链汇总（sgc-bind · 0.1-α CPU 侧）

> **用途**：把本仓库围绕轴 D 绑定缺陷修复（`sgc-bind` mixer）产生的全部
> 实验账本收敛为**一份可逐项验收的清单**。每项均指向
> `artifacts/binding_probe/` 中的原始 JSON 账本与守护测试。
> **口径**：≤1M 参数前哨，合成任务 + 真实词分布 + 真实语料字符级 LM；
> **不构成任何规模上的 SOTA 主张**（README 能力边界条款继续有效）。

## 九项验收清单

| # | 验收项 | 结果 | 账本 / 证据 |
|---|--------|------|-------------|
| 1 | **机制正确性**：FFT↔IIR 双形式 max\|Δ\|≈6e-8、严格因果、O(1) 定容状态、purity 三层守卫、`analytic_params` 逐位一致、断点恢复含槽状态 | 12 项守护测试全过 | `simplcore/core/tests/test_bsgc.py` |
| 2 | **MQAR**（chance 0.125）3 seeds 胜 attention，CI 不重叠，参数 31.8K < TS 35.8K；**G=16**（chance 1/16，3 seeds）sgc-bind **1.000×3** vs ts 0.035–0.075 vs sgc 0.03–0.10；**srf（Mamba 机制 mixer，2 长度）** 0.12–0.14（chance）——修复来自绑定记忆而非任意 SSM | sgc-bind **0.785** vs ts 0.325 vs sgc 0.158 | `mqar_final.json` `mqar_multiseed.json` `sweep3b.json` `sweep2.json` |
| 3 | **扩规模** D=128/4 层，3 seeds 胜 attention | sgc-bind **0.895** vs ts 0.300 vs sgc 0.137 | `mqar_scale.json` + `multiseed2.json` |
| 4 | **长上下文**联想召回：**长度扫描 L=96→2048 全程满分**（attention L≥256 跌破 chance；L≥1024 需为其扩位置表才能跑）；L=256 MQAR 格式 3 seeds 全 1.0 vs ts 0.235–0.28 | sgc-bind **1.000** @ L=96/128/256/512/1024/2048 | `mqar_longctx.json` + `mqar_sweep.json` + `mqar_sweep2.json` + `sweep3.json` + `overnight_seeds.json` |
| 4' | **零样本长度外推**（真实语料，seq=128 训练 → 1024 评测）：sgc/sgc-bind 近乎平坦（+0.9%/+1.3%），ts 漂移 +4.3%（位置嵌入未训练段） | sgc 1.79 / bind 1.85 / ts 2.49 @1024 | `len_extrap.json` |
| 4'' | **MQAR 零样本长度外推**（训 L=128 → 评 256/512）：2× 零样本满分（S=256 槽无覆盖）；4× 跌至 0.155——**与容量预算预言精确一致**（511 写入 > 256 槽） | sgc-bind 1.0/1.0/0.155 vs ts 0.195/0.235/0.31 | `mqar_zs_len.json` |
| 5 | **等预算 vs Mamba2**（HF 参考实现，等 token，逐窗配对 bootstrap CI，**3 seeds**） | PPL 比 **0.685 / 0.717 / 0.668**，**96/96 窗全胜**，三闸门全过；vs Transformer 0.968/0.940/0.948（80/96 窗） | `equal_budget_bind.json` + `equal_budget_multiseed.json` |
| 5' | **更大规模等预算**（D=256/8 层/seq 256/128K tokens，3 seeds，32 步欠训练） | sgc-bind vs TS ratio 1.031/1.024/1.036（均在 ≤1.05 容差内，ts 略优；PPL ~4000 远未收敛）——**诚实记录欠训练档** | `equal_budget_small_ms.json` |
| 5'' | **真实语料等预算**（中文 cn 语料打包，char 级，smoke 规模，3 seeds） | **vs Mamba2 直接对照：sgc 1.545 / srf 1.537（0/10 窗，负）**——差距约 1.5 倍而非初测的 2 倍（初测 2.01/2.03 混入了绑定通道拖累）。**根因定位（`lm_tuning.json`）：绑定通道拖累纯 LM**（sgc-bind 288.9 vs sgc 238.3）；**纯 sgc 0.836 / srf 0.831 vs TS（10/10 窗全胜）** | `equal_budget_real.json` + `lm_tuning.json` + `lm_vs_mamba2.json` |
| 6 | **真实词分布召回迁移**（Zipf 噪声，3 seeds） | sgc-bind **1.000** vs ts 0.332 vs sgc 0.132 | `recall_transfer.json` + `multiseed2.json` |
| 7 | **诚实负结果**：list-ops（ts 0.485 > sgc 0.335 > sgc-bind 0.265）、fuzzy-jump（三臂 chance）、多 needle（三臂饱和无区分度） | 已入档 | `lra_probe.json` `lra_probe2.json` |
| 8 | **真实语料 LM 成本**（corpus.txt 字符级）：绑定通道 +5.7% PPL（检索能力专用增强的代价） | sgc 5.93 / sgc-bind 6.27 / ts 11.05 | `reallm.json` |
| 9 | **MI250 战役就绪**：数据管线（SHA-256 manifest）、闸门清单、`tuan-1b` dry-run 启动预检、3 步实跑——四步全过；修复 2 个上卡首日必爆缺陷 | 预演四步全 ✅ | `mi250_rehearsal.json` + `campaign_gates.txt` |

## 可执行验收

验收清单已固化为可执行检查器（逐项加载账本断言判据，全部通过才退出 0；
含长度扫描曲线与 Mamba2 三种子等预算两项扩展，共 **11 项**）：

```bash
python research/binding/acceptance_check.py   # → 11/11 PASS
```

机器可读结果：`artifacts/binding_probe/acceptance_check.json`。

## 复现命令

```bash
python -m pytest simplcore/core/tests/test_bsgc.py -q      # 验收项 1
python research/binding/final_compare.py                   # 验收项 2
python research/binding/multiseed.py                       # 验收项 2（多种子）
python research/binding/scale_probe.py                     # 验收项 3
python research/binding/longctx_mqar.py                    # 验收项 4
python research/binding/equal_budget_run.py                # 验收项 5
python research/binding/recall_transfer.py                 # 验收项 6
python research/binding/lra_probe.py                       # 验收项 7
python research/binding/lra_probe2.py                      # 验收项 7
python research/binding/reallm_probe.py                    # 验收项 8
python research/binding/mi250_rehearsal.py                 # 验收项 9
```

## 边界（引用时必须携带）

* 全部为 **CPU、≤1M 参数、合成任务/字符级语料** 的前哨证据；MQAR/LRA
  探针无显著性检验（MQAR 为 3 seeds）；
* `sgc-bind` 的收益限于**联想召回/绑定类能力**；list-ops 与纯 LM 上为
  中性偏负（见 #7/#8）；
* 1B→3B 战役实测（`scripts/train_tuan_mi250.sh`，chunked BPTT 已默认）
  **依赖 MI250 硬件**，就绪状态见 `docs/MI250_READINESS.md`。
