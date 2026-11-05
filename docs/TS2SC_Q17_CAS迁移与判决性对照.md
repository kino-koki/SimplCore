# CAS 接进 1.7B + 「修脚本还是换架构」判决性对照

日期：2026-11-05　仓库：kino-koki/SimplCore
本轮脚本：`scripts/migration/q17_cas_{collect,build,gen,ood}.py`、`scripts/migration/distill_probe.py`
原始证据：`artifacts/TS2SC_Q17_cas_*.json`、`artifacts/distill_probe_{raw,state}.json`

---

## 0. 一句话结论

**把 CAS 接进 Qwen3-1.7B 的 28 层做完了，holdout 指标全绿，但生成崩了。**
随后用同一套「修好的迁移脚本」做了四臂判决性对照，结论是：

> **不是脚本的问题，是静态核的函数类问题。** 同一套修好的脚本，SGC 三臂全在
> 0.23~0.25 纹丝不动；换成 CAS（少一条纯粹性）直接到 **0.3659**，与 teacher
> 的 0.3633 持平，配对 bootstrap +0.1315，CI95 [+0.0743, +0.1640] 显著。

---

## 1. 本轮「迁」做了什么（零训练、全解析）

两轴互补，全部保持 O(1) 状态、无 N×N、无 KV cache：

| 轴 | 来源 | 状态 |
|---|---|---|
| 周期 / 连续信号 | `ck4` SGC（`linear` 网格、逐 KV 头 coef），逐字节不动 | 8,960 B/层 |
| key→value 内容寻址 | `ContentAddressedState`（φ=1+elu，F=128，V=256 低秩） | **131 KiB/层** |

三步：

1. **collect**：6 段语料 × 28 层，逐层存 `h`（attn 输入）、`a`（teacher attn 输出）、
   `sgc_inj = ck4(x) − x`、`Y`（CAS prefix-cumsum 原始读出）。
   CAS 的 q/k/v 投影用 teacher `q_proj/k_proj/v_proj` 的**行空间主成分**（SVD 低秩）
   解析初始化，行 L2 归一化；零随机参数。
2. **build**：每层解闭式 ridge
   `W = (YᵀY + αI)⁻¹ Yᵀ(a − sgc_inj)`，
   λ∈{0,.25,.5,.75,1} 与 α 网格由 **holdout 段（si=5）余弦** 决定；
   **λ=0 恒在候选里**，保证不劣于 ck4。λ 乘进 `read_out.weight`。
3. **gen**：teacher / ck4 / ckcas 三路同口径（Qwen3-Instruct chat template +
   no-think 预填、argmax、无采样无后处理），判据**只看生成**。

## 2. 指标与生成再次背离（这是本轮最重要的负面证据）

**holdout（拟合分布）**：28/28 层联合 cos 全部提升，平均 **0.5952 → 0.6474（+0.0522）**，
最大 L12 +0.1200。

**chat 分布外诊断**（`q17_cas_ood.py`，两个问句逐层重算）：
25/28 层仍提升，平均 **+0.0575 / +0.0530**，注入幅度比 0.34~0.37 —— 
**不是分布外泛化失败**，CAS 在 chat prompt 上确实把注入方向对得更准。

**但生成**（NNEW=4，原文一字不改）：

```
teacher : 华为技术有限公司（Huawei Technologies Co., Ltd
ck4     : \nOkay发现\n
ckcas   : وع\nOkay\ufffd
```

| | rep3 | 中文占比 | distinct | 事实 |
|---|---:|---:|---:|---|
| teacher | 0.0000 | 0.216 | 10 | ✅ 正确开头（8~10 token 截断） |
| ck4 | 0.0000 | 0.250 | 3 | ❌ 零事实、中英混词堆砌 |
| **ckcas** | 0.0000 | **0.000** | 4 | ❌ **更差**：阿拉伯语碎片 + 非法字节 |

**归因**：闭式读出拟合的是「该层注入量在方向上像不像 attention」，
不是「整条残差流经 28 层后能不能生成正常汉语」。方向余弦是局部代理指标，
逐层 +0.05 累积后反而把 logits 推到词表另一侧的罕见 token。
这和 WSP 那篇「6 倍性能其实是修死循环」、以及我们之前的
「余弦 1.0000 被 FFN 刷出来」「holdout CE −10.4% 但 rep3 +36%」
是**同一类坑：指标被测量方式污染**。所以这次「迁」**判定失败，ckcas 不并入任何权重包**。

## 3. 判决性对照：修脚本 vs 换架构

用户判断：「SimplCore 原生小参数模型本身没有问题，是迁移脚本的问题」。
把这句话拆成两个可证伪命题，用同一 teacher、同步数、同 seed、同评估集判决。

**修了什么**（对照 v1~v5 与 181 步自监督那次，五处全改）：

| # | v1~v5 / 181 步 | 本轮 |
|---|---|---|
| 1 | 逼 SGC 内部隐状态 / 注入量像 attention | **只对齐 teacher 冻结输出分布**（黑盒 logits） |
| 2 | profile LS 拟合当最终权重 | 只作**暖启动**（emb/head 逐位复制） |
| 3 | 冻结 emb/FFN/head，只训 mixer | **全参数可训** |
| 4 | 单末位 CE / T=1 / 无 teacher | **每位置 KL(T=4) + 末位 CE，α 0.7→0.3 退火** |
| 5 | 主指标 = 残差流余弦 | **任务 acc / 收敛曲线** |

MQAR G=4，8 类，chance 0.125，L=96，600 步 / 3 seeds，teacher 1000 步：

| 臂 | acc（3 seeds） | mean | 参数量 | 状态 |
|---|---|---:|---:|---:|
| teacher（2 层 attention） | — | **0.3633** | 34,760 | KV 随 L |
| `sgc_scratch` 从零训 | 0.2031 / 0.2773 / 0.2227 | 0.2344 | 15,370 | 8,960 B |
| `sgc_warm` 暖启动 + CE | 0.2383 / 0.2930 / 0.2188 | 0.2500 | 15,370 | 8,960 B |
| **`sgc_kd` 修好的脚本 + 重训** | 0.2578 / 0.2070 / 0.2227 | **0.2292** | 15,370 | 8,960 B |
| **`cas_kd` 同一套脚本 + CAS** | 0.3594 / 0.3516 / 0.3867 | **0.3659** | 18,120 | 8,448 B |

配对 bootstrap（2000 次）：

```
sgc_warm − scratch   +0.0157  CI95 [−0.0039, +0.0352]  ❌ 不显著
sgc_kd   − scratch   −0.0052  CI95 [−0.0703, +0.0547]  ❌ 不显著（甚至略负）
cas_kd   − scratch   +0.1315  CI95 [+0.0743, +0.1640]  ✅ 显著，且 ≈ teacher
```

**五处脚本修正全部落地后，SGC 一丝提升都没有**（暖启动 +0.0157 不显著，
KD 反而 −0.0052）。而**同一套脚本**只把 mixer 换成 CAS，直接达到 teacher 水平。
所以 H1（脚本问题）被证伪，H2（函数类上限）成立。

这与之前所有静态核证据一致：精确核 oracle 中层 cos ≈ 0、v4 L27 0.2484 ≈
oracle 0.2476、MQAR 0.1133、加基/门控/MoK/A1~A5 适配全部无效。
**静态、输入无关的谱核做不了 `q_i·k_j` 内容匹配，是信息论边界，不是调参能绕的。**

诚实边界：D=32 / 2 层 / 合成 MQAR / 3 seeds / 600 步的小规模前哨；
teacher 未调参、参数量不等（逐臂记录，不做等参断言）。
本仓从未从零训过 1.7B 语言模型，因此「原生模型在**真实汉语**上没问题」
既未证实也未证伪——这里判的是「修脚本能否越过静态核函数类上限」这一机制问题。

## 4. 怎么办（回答本轮的问题）

**别再修脚本了，也别再在静态核上加步数/加核/加适配。** 三条已全部证伪：
解析直搬（v1~v5）、自监督微调（181 步，CE 降 10.4% 而 rep3 +36%）、
修好的全参数分布蒸馏（本轮 sgc_kd −0.0052 不显著）。

**唯一有跨任务证据支撑的方向是 CAS（少一条纯粹性）**，而且它现在有两份独立证据：

| 证据 | 结果 |
|---|---|
| `cas_probe`（纯训练，无 teacher） | +0.1055 vs 静态核，CI 显著；半参数量边缘超 TS |
| `distill_probe`（修好的迁移脚本） | +0.1315 vs scratch，CI 显著；**追平 teacher 0.3633** |

具体下一步，按顺序：

1. **放弃「解析闭式读出」这条 CAS 接入方式**——本轮已证明它优化的是代理指标。
   改用 `distill_probe` 里那套**每位置 KL 分布蒸馏**训练 CAS 的 q/k/v/read_out
   （ teacher 冻结、只对齐输出、全参数可训），而不是解 ridge。
2. 在 1.7B 上先做**单模块替换实验**：只换 1 层为 CAS，其余保持 teacher 原
   attention，验证生成不崩，再逐层放开——避免 28 层同时改造成误差累积。
3. 生成判据**只看生成**（成句 / 不循环 / 事实正确 / 中文占比），
   残差流余弦与 holdout cos 一律降为辅助记录。
4. 长上下文（8K）与双形式漂移在生成过关后再测，不提前庆祝。

## 5. 复现

```bash
# 判决性对照（按 (arm,seed) 原子续跑，抗沙箱 502）
python3 scripts/migration/distill_probe.py --steps 600 --seeds 0 1 2 \
    --n-eval 256 --batch 8 --out artifacts/distill_probe_raw.json
ONLY=cas_kd SEED=2 python3 scripts/migration/distill_probe.py --steps 600 ...

# CAS 接进 1.7B（三步，均支持断点续跑）
python3 scripts/migration/q17_cas_collect.py      # BATCH / MAXSEC 控制批次
python3 scripts/migration/q17_cas_build.py        # 产出 ckcas/ 28 层
ONLY=华为是什么公司 WHO=ckcas NNEW=8 python3 scripts/migration/q17_cas_gen.py
python3 scripts/migration/q17_cas_ood.py          # chat 分布外逐层诊断
```

## 6. 当前仓库状态

- **远端 HEAD `ea5733de`** = CAS 模块 + 8 项守护 + `cas_probe` 收益报告
- 本轮新增：4 个 1.7B 迁移脚本、`distill_probe.py`、全部原始证据与本报告
- **不并入任何权重包**：`ck4` 仍是 1.7B 解析迁移的最佳版本（本身也不成句），
  `ckcas` 生成更差，仅作负面证据保留
