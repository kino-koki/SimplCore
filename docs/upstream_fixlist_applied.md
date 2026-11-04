# 上游待修清单落地报告（2026-11-04）

核对对象：`docs/upstream_fixlist.md`（另一 Agent 对 `kino-koki/SimplCore` 的
静态逐行核对 + 按源码公式独立复算）。本报告逐条记录**实际改了什么、跑了什么、
哪些没做到**，不把「计划」写成「完成」。

## 总览

| 项 | 状态 | 交付 |
|---|---|---|
| P0-1 `linear` 主路径不可用 | ✅ 已修 + 回归守护 | `config.py` 白名单 / legacy 预警 / `test_config_linear_grid.py` |
| P0-2 P05 只测小配置 | ✅ 已修 + 跑通 | `mathproof.py` 四档扫描，P05 `passed=True` |
| P0-3 默认 logfreq 静默放行塌缩字典 | ✅ 已修 + 回归守护 | `sgc.py` warning / §7 第四条理由 / 测试 |
| P0-4 两套闸门默认值 | ✅ 已修 + 测试适配 | `sgc.py` / `selective.py` 默认 False；`test_gates.py` 显式声明意图 |
| P1-1 双形式「逐位一致」过度声称 | ✅ 已修 | 论文摘要 / Abstract / 定理 1 / §5.1–5.3 |
| P1-2 状态公式 `+1` 口径 | ✅ 已修 | 论文定理 3 口径注 |
| P1-3 MQAR 不在 `TASK_SPECS` | ✅ 已修 | `axis_probe.py` 新增 `mqar_g4` + §5.5 / 附录 B 复现口径注 |
| P1-4 osc 收益是仓库外孤证 | ⚠️ **脚本已落地，收益未复现** | `eval/osc_probe.py` + `artifacts/osc_probe_raw.json`；§1/§9 的「非本仓复现」标注**不撤** |
| P1-5 「最优解」口径限定 | ✅ 已修 | §9 限定 + 非对称网格纳入 `osc_probe` 扫描 |
| P2-4 论文两处内部不一致 | ✅ 已修 | §3.3 交叉引用指向 §5.5；§5.12 参数量口径注 |
| P2-1/2/3 白盒与源码算子差异 | ⏸ 不改源码 | 属口径说明，见下 |

另附带修一处**既有失败**：`test_axis_probe.py` 的 purity 扫描被
`migration/dc_bias.py` 的 `attn_out` 标识符命中（该文件是解析迁移辅助件，
把 teacher attention 输出当外部张量读入，不实现注意力算子）。已按
`axis_probe.py` TS 对照臂的先例加入 `purity.py` 定义级豁免，并写明理由。

## P0：代码改动与验证

### P0-1
`SimplCoreConfig.validate()` 白名单加入 `linear`，`getattr` 默认值由
`"legacy"` 统一为字段默认 `"logfreq"`；`legacy` 在 validate 阶段提前
`RuntimeWarning`（说明构造期会因代数公共零点被拒），避免用户看到算子层
的间接报错。新增 5 项回归测试，其中
`test_linear_reaches_model_mixer_via_config` 锁住
`SimplCoreConfig(sgc_init_mode="linear") → SimplCoreModel → mixer.init_mode=="linear"`
这条主路径。

### P0-2
`mathproof._proof_bibo_stability` 改为按
`decay_max ∈ (64, 8192, 16384, 4e6)` 逐档构造 mixer、按各档
`theta_clamp_floor(dm)` 断言下界，并在 `residual` 里如实打印
`1/|sin floor|`。实跑 `P05 passed=True`，四档放大倍数：

```
dm=64      floor=1.000e-03   1000.0×
dm=8192    floor=3.835e-04   2607.6×     ← 默认 / tuan-150m
dm=16384   floor=1.917e-04   5215.2×
dm=4e6     floor=7.854e-07   1273239.5×  ← tuan-2b
```

论文命题 5 段同步写明：上界绝对、下界随 `decay_max` 让步、
**不提供固定 1e-3 保证**，撤回旧口径。

### P0-3
`_init_oscillator_grid` 的 `logfreq` 分支不再静默放行：按
`_grid_rank_condition(theta, L=max(512, 4M))` 复算并 `RuntimeWarning`。
实测默认 M=32 / decay=8192 触发 `rank=24/32, cond=1.92e+16`，与清单一致。
`linear` / `hybrid` 仍走硬拒绝质量门，既有 checkpoint 可复现性不受影响。
`docs/frequency_grid_fix.md` §7 补第四条理由（默认网格仅以 warning 放行）。

### P0-4
`sgc.py:281-282` 与 `selective.py:101-102, 379` 的
`evidence_gate / uncertainty_gate` 默认值由 `True` 统一为 `False`，
与 `SimplCoreConfig` 默认值、技术报告 §3.1、§5.14 的 402432 B 口径一致。
`test_gates.py::_layer` 改用 `setdefault` **显式声明**「本文件测的是带闸门
行为」，调用方显式传值优先——测试意图不再依赖隐式默认。

## P1：口径与复现

### P1-1 / P1-2 / P1-5 / P2-4（论文与文档表述）
逐处替换，全部带 `assert` 精确匹配后写入：

- 摘要第 1 条、英文 Abstract、定理 1、§5.1–5.3：撤回「fp32 无条件逐位一致」，
  改为**分路径口径**（流式逐位 / 并行 fp32 容差内、偏差随 L 增长）。
- 定理 3：`+1` 注明为 L2 证据门开启时的每序列标量，**不是结构常数**；
  默认 OFF 时实际为 `4(2MD+(K−1)D)`。
- §5.5 / 附录 B：补 `mqar_g4` 复现口径注，明确论文旧 MQAR 数字来自外部
  白盒、与新增前哨任务定义不同，**不得直接比较**。
- §9：「最优解」限定为「本文扫过的组合与本文 Gram 口径」，纳入非对称
  θ 网格、`linear_floor` 连续扫描、含 ρ 口径三项后续搜索方向。
- §3.3「MQAR 对照（§7.2）」→（§5.5）；§5.12 补注 1.888 GiB 对应约
  10.1 亿参数的更大预设，与 §5.14 的 1.085 亿参数不是同一配置。

### P1-3
`axis_probe.py` 新增 `gen_mqar`（4 对 key→value、8 类、chance 0.125、
key/value disjoint id 空间、末位 query），加入 `TASK_SPECS` 为 `mqar_g4`。
附录 B 的 `--seeds 0 1 2 3 4` 现在确实能跑出一条 MQAR 行。

### P1-4 ⚠️ 未达成目标，如实记录
`eval/osc_probe.py` 已落地：周期信号生成器、7 种网格扫描（含打破 π/2
镜像的非对称网格）、解析 chance 与 `copy_last` 平凡基线、配对 percentile
bootstrap、raw json 落盘。

但**本仓小预算前哨没有复现出 linear 收益**：

| 版本 | 结果 |
|---|---|
| 第一版单锯齿周期任务，200 步 | 7 网格**全部 1.0000** —— 天花板效应，区分不出网格，已废弃 |
| 改为「每次随机 2 个周期分量」，250 步 / 3 seeds | 7 网格 **0.1094~0.1354**，全部**低于** `copy_last=0.1575` 平凡基线；配对差 `linear − logfreq` 全为负或 CI 跨 0 |

配置：`D=32 / 1 层 / M=16 / L=96 / ε=0.15 / V=16 / 250 步 / 3 seeds`。
结论：模型连「复制上一个 token」都没学会，训练预算不足；这组数字
**既不支持也不否定** linear 收益，只证明脚本能在本仓重跑。

因此 `docs/frequency_grid_fix.md` §1 / §9 的「**非本仓复现**」标注
**保持不撤**，只更新为「已有可重跑骨架 + 小预算 raw json，收益未复现」。
要撤标注需按白盒口径把预算提到 1500/3000 步、配好 TS 对照臂与任务难度后重跑。

## 测试

```
pytest -q  test_config_linear_grid.py  test_linear_grid.py
           test_gates.py  test_gates_selective.py
           test_axis_probe.py  test_architecture_purity.py
→ 74 passed
mathproof._proof_bibo_stability() → P05 passed=True（四档）
osc_probe → artifacts/osc_probe_raw.json（负面结果，见 P1-4）
```

改动文件：`config.py` / `arch/sgc.py` / `arch/selective.py` /
`arch/purity.py` / `eval/mathproof.py` / `eval/axis_probe.py` /
`eval/osc_probe.py`（新）/ `tests/test_config_linear_grid.py`（新）/
`tests/test_gates.py` / `SimplCore_Paper.md` /
`docs/frequency_grid_fix.md` / `docs/upstream_fixlist_applied.md`（本文件）。

## P2 为什么不改

P2-1（冲激响应差 `1/sinθ` 缩放）、P2-2（网格参数三处不同）、P2-3
（srf selective 死参数史）是**白盒与源码的算子口径差异**，不是源码 bug。
改源码会破坏既有 checkpoint 与已发表数字的可比性；正确做法是在引用白盒
数字时标注口径，本报告不越界。
