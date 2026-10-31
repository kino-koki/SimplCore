# SimplUnity 三个月：2B 主干 + 自进化 + 后台模块训练

> 目标（用户口径）：2B 主干训 1 个月；随后 2 个月跑自进化学习，同时后台训练多个
> 增强模块；最终验证"模块可拆分/拼接、主干只学基础能力"的可扩展性。
>
> 代码现状：`UnityModel`（冻结基座 + 路由 + 专家）、`UnitySelfEvolution`
> （多 worker + 无回归门）、`ModuleBundle`（模块级拆分/拼接）、
> `ExperienceLoader`（经验→next-token batch）。本文给**资源账与阶段验收**，
> 不把"并行"说成免费。

## 1. 事实先摆：单卡不是无限并行

MI250 是 **2×64 GB GCD**，不是一块 128 GB 的 GPU。真正可并行的方式只有两种：

| 方式 | 做法 | 代价 |
|---|---|---|
| **GCD 分工**（推荐） | GCD0 跑主干推理/自进化（生成经验、SMM/Dream 巩固）；GCD1 跑 Unity 专家训练 | 每个 GCD 只有半卡算力；主干侧 MFU 仍约 25% |
| **同 GCD 时间片** | 主干推理与专家训练交错，共享一个 GCD | 显存可共存，算力串行；适合只有单 GCD 可用时 |

所以"后台训练多个模块"的真实含义是：**在 GCD1 上并行/分时训练多个小专家**，
不是在主干训练的同时再塞一份 2B 训练。M2/M3 的模块训练预算必须按 **半卡算力**
（≈45 TFLOPS 有效）来算。

## 2. 三个月节奏

### M1（30 天）：2B 主干基础能力

见 `docs/TRAINING_2B_MI250_30D.md`：`tuan-2b`（2024M）× **~13B tokens**，
4M 课程末段 10%，`--train-chunk 4096` + 2 GCD DDP。

**门槛（不通过不进入 M2）**：

- `val loss` 与 token 预算对得上；`grad_norm` 无爆炸；
- `GATE-2`（1M needle）**不是 below_chance**；at_chance 允许（轴 D 预期）；
- checkpoint 能加载、流式 4M 能跑通、`test_rho_logspace` 的 4M 极点可达。

### M2（30 天）：自进化 phase 1 + 4 个域专家

**GCD 分工**：GCD0 主干自进化；GCD1 训练专家。

| 域（示例） | 训练数据 | 专家 | token 预算 |
|---|---|---|---|
| code / repo | 真实代码 + 沙箱成功轨迹 | 1 × 50M | 2–5B |
| cli / tool | CLI 会话、错误-修复对 | 1 × 50M | 2–5B |
| docs / longctx | 长文档问答、引用 | 1 × 50M | 2–5B |
| dialogue | 对话/记忆召回 | 1 × 50M | 2–5B |

算力账：单专家 50M，6ND × 5B tokens ≈ 1.5e18 FLOPs；半卡有效 ≈45 TFLOPS
→ **~9.3 小时/专家**；4 个 ≈ **1.5 天纯算力**，30 天足够做多轮（含数据/评估/回滚）。

**流程（已可实现）**：

1. 主干在 GCD0 上与环境交互，`SimplModelMemory.experience()` / `Dream` 落经验；
2. 每域导出文本 → `ExperienceLoader` → `(x,y)` batch；
3. `UnitySelfEvolution(num_workers=2)` 在 GCD1 上训练候选专家；
4. **GATE-3 无回归门**：候选在域 eval 上不得比"无专家基线"差超过阈值，否则拒绝发布；
5. 通过 → `publish_expert` 原子换入；失败 → 丢弃，活动模型不变；
6. 每轮结束 `save_learned` + `ModuleBundle` 导出（可审计来源/训练量/val loss）。

**M2 验收**：

- 4 个域专家都通过 GATE-3；主干在通用 val 上**不退化**（≤0.5%）；
- 每个专家可导出 bundle、拼接到另一个同构主干（`import_bundle`）；
- 路由不 collapse（各域激活分布可审计）。

### M3（30 天）：扩展 + 组合 + 热插拔

- 专家从 4 → 8–16；`top_k` 从 2 → 2/3 做消融；
- **热插拔**：训练中新增/替换专家，主线推理不中断（bundle 导入）；
- **组合验证**：单域、双域、全量组合的 CE / 吞吐 / 显存三本账；
- **扩展性曲线**：专家数 vs 学得参数 / 推理延迟 / 路由熵 / 域增益；
- 最终验收：GATE-2（长上下文）+ GATE-3（无回归）+ 组合矩阵 + 跨模型 bundle 拼接。

## 3. 关键约束（不做就会变成"看起来并行"）

1. **域数据必须不同**。同一份数据训练多个专家 = 重复参数，只会让路由退化。
2. **主干必须先有基础能力**。M1 不达标时，专家只是在小数据上过拟合；不要用
   "模块化"掩盖主干欠训。
3. **GATE-3 默认拒绝发布**。后台训练不能直接改活动模型；候选不优于基线就不发布。
4. **预算守卫**：`WeightBudget` 的 2B 可学 / 3B 总上限是硬约束；`import_bundle`
   也必须经过 `spawn_expert` 的预算检查。
5. **线程 ≠ 进程**。当前 `UnitySelfEvolution` 是线程版（ROCm 上 GIL/流竞争需实测）；
   M2 上卡若发现吞吐异常，切到进程级 worker（队列 + 共享存储）是下一版工作。

## 4. 交付物（3 个月结束必须能拿出来）

- `tuan-2b` 主干 checkpoint + 训练账（tokens/MFU/val）。
- 4–8 个域的 `ModuleBundle`（含 `base_fingerprint`、训练量、val loss）+ 路由报告。
- `gate2.json` / `gate3.json` / 组合评测矩阵。
- 一次**跨主干拼接**演示：把专家从 one run 搬到另一个同构 2B 主干并通过 GATE-3。
- 诚实标注：2B × 13B tokens 是欠训交付；"并行"是 GCD 分工，不是免费算力。

## 5. 当前代码对应关系

| 能力 | 代码 |
|---|---|
| 冻结主干 + 路由 + 专家 | `simplcore/unity/model.py` |
| 多 worker 后台训练 + 无回归门 | `simplcore/unity/evolution.py` |
| 模块级导出/拼接/指纹校验 | `simplcore/unity/bundle.py` |
| 经验 → next-token loader | `simplcore/unity/experience.py` |
| 2B/1B 的 Unity 预算预设 | `simplcore/unity/config.py`（`UNITY_PRESETS`） |
| 可执行入口 | `scripts/unity_evolve_2b.py` |
| 测试 | `test_simplunity_modular.py` |