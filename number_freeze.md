# 05 · number_freeze.md — 冻结数字表（单一真值源）

> 技能包 §6：每个数字必须能追到 配置文件 / seed 数 / 精度 / 样本量 / 生成脚本 / 结果文件。
> 禁止从正文 A 复制到正文 B；所有出现处引用同一冻结值。
> 机器复核：`theory_work/results/governance_number_freeze.json`（16 条，`unique: true`）。

## 1. 冻结表

| 配置 ID | 设定 | 冻结值 | seeds | dtype | 样本量 | 来源文件 | 出现位置 |
|---|---|---:|---|---|---|---|---|
| C01 | chance（本任务） | 0.0313 (=1/32) | — | — | 定义 | `1/V_K`, V_K=32 | §4, Table 2/5/10, Appendix H/I |
| C02 | H=2, 2 层, 1500 步 | 1.0000 | 5 | fp64 | 512 held-out/seed | `convergence_matrix.json` | §4.12, Table 15, §8 |
| C03 | H=2, 1 层, 1500 步 | 0.1809 | 5 | fp64 | 512 held-out/seed | `convergence_matrix.json` | §4.12, Table 15 |
| C04 | 配对差（2层−1层） | +0.8191, CI 含于 Table 15 | 5 | fp64 | 10k bootstrap | `convergence_matrix.json` | §4.12 |
| C05 | 首次 ≥0.99 的步 | 150 | 5 | fp64 | 每 150 步评估 | `convergence_curve.json` | §4.12, Appendix F |
| C06 | SE2 解析 E[T_cover] | 129.87 | — | 解析 | n=32, β≈0 | `fusion_guard.json` | §4.13, Appendix A.8, Table 30 |
| C07 | SE2 实测均值首次全覆盖 | 130.00 | 5 | fp64 | 320 轮 | `fusion_guard.json` | §4.13, Appendix A.8, Table 30 |
| C08 | SE2 相对偏差 | 0.099% | — | 解析/实测 | — | `fusion_guard.json` | §4.13 |
| C09 | SE3 精确 E[acc] | 0.709375 | — | 解析 | p=0.30, V_K=32 | `fusion_theory.json` | Appendix A.9, Table 30 |
| C10 | SE3 枚举均值 | 0.709102 | — | fp64 | 20000×32 key trials | `fusion_theory.json` | Appendix A.9, Table 30 |
| C11 | SE3 Hoeffding 4σ 区间 | [0.7071, 0.7116] | — | 解析 | N=640000 | `fusion_theory.json` | Appendix A.9 |
| C12 | SE4 消耗 / CT7 上界 | 500 / 500 bit | — | 解析 | T=100, w=1, b=5 | `fusion_theory.json` | Appendix A.10 |
| C13 | SE5 K\* @ρ=0.999, ε=1e−3 | 6904 | — | 解析 | — | `fusion_theory.json` | §4.13, Appendix A.11 |
| C14 | fused_gated 最终 acc | 1.0000 ± 0.0000 | 5 | fp64 | 320 轮 | `fusion_evolution.json` | §4.13, Table 16/18/24/34 |
| C15 | ungated_corrupt | 0.6438 ± 0.0781 | 5 | fp64 | 320 轮, p=0.30 | `fusion_evolution.json` | §4.13, Table 16 |
| C16 | break_alignment | 0.0250 ± 0.0364 | 5 | fp64 | 320 轮 | `fusion_evolution.json` | §4.13, Table 16 |
| C17 | rho_fast | 0.1313 ± 0.0234 | 5 | fp64 | 320 轮, ρ=0.90 | `fusion_evolution.json` | §4.13, Table 16 |
| C18 | 核心状态字节 | 32768 | 全部 | fp64 | d=64, 64×64 | 代码结构量 | §4.13, Table 18/20/34 |
| C19 | parity gapped（5-seed） | 0.3438 | 5 | fp64 | 512 held-out/seed | `stress_five_seed.json` | Table 2, Table 10 |
| C20 | duplicate keys（5-seed trained） | 0.9328 | 5 | fp64 | 512 held-out/seed | `stress_duplicate.json` | Table 2, Table 10 |
| C21 | FFT/IIR 对偶误差（最终构造） | ≤ 3e−16（3 探针最大值） | — | fp64 | L=1025 | Table 4 + `fusion_guard.json` | 摘要只写 "machine precision" |

所有值由生成器在构建时从 JSON 注入（`build_simplcore_pdf_v7.py`），正文中不手抄。
Appendix I 的 Table 30 是逐值溯源表，Table 31 是 witness 明细。

## 2. v6 已知冲突对 —— 逐对消解（技能包 §6.3）

| 项 | 冲突值 | 判定 | 消解动作 |
|---|---|---|---|
| chance | `0.0313` vs `[0.0332, 0.3125]` 的 "near the four-pair chance level" | **不同任务** | 本任务 chance 唯一 `1/V_K=0.0313`；`0.25` 是历史四对任务的 chance，只在 Table 10 以「DIFFERENT historical task」出现；附录里所有 "near chance" 措辞改成精确区间 `0.21–0.29` + 「far from this task's 1/32 chance」 |
| +parity gapped | `0.2969` (§4.4/4.5) / `0.2711` (§4.6/4.9) | 单 seed、不同 probe | 两者均列为 superseded；冻结值 `0.3438`（5-seed mean, Table 2）。Table 28 的 `0.2969` 标注为「SINGLE-SEED, alternating layout, position-prior carrier, DIFFERENT configuration」 |
| duplicate keys | `0.9316 / 0.9414 / 0.9328` | 单 seed analytic vs 5-seed trained | 前两者标 superseded；冻结 `0.9328`（5-seed trained, Table 2） |
| FFT/IIR 误差 | `2.27e−16 / 2.89e−16 / 1.14e−16 / 1.45e−16 / 2.43e−14 / 3.16e−16 / 3.27e−16` | 不同构造/路径/精度 | Table 4 逐探针列清（analytic stack 2.27e−16 / codebook stack 2.89e−16 / learned interface 2.89e−16）；Table 10 冻结「≤3e−16，3 探针最大值」；`2.43e−14` 是 `bitwise=False` 的旧撤回实例，标 W1；摘要只写 "machine precision" |
| bitwise 主张 | 摘要「bit-identical outputs (deviation 0.0e+00)」 vs W1 撤回 fp32 逐位一致 | **两个不同对象** | Table 11/27 的 0.0e+00 是「同一 side-channel fibre 内跨 compatible role assignments 输出相同」，合法；W1 撤回的是「FFT/IIR 跨执行顺序逐位一致」。§3.4 P1 与附录 B W1 均写明这一区分 |
| 5×5 表 overshoot | `0.0645/0.0684/0.0879/0.0918` vs 正文「at most chance」 | 实测高于 chance | 正文统一写 `≤0.10`，并解释 overshoot 是末层读出产物、不等于正确；Table 5/29 caption 自带该说明 |
| 参数 / 内存 | `2Vr+1 / 4194304 / 549.8 GB / 512 B/layer / 1024·H_L / 56320 B / 32768 B` | 不同构造的不同量 | `549.8 GB` 明写「若物化稠密码本」的假想值（V=2^18），不是实测占用；`32768 B` 是本轮融合对象的核心状态；`56320 B` 是历史张量探针状态。Appendix I 逐项标注 V、r、P、H_L 取值 |
| W12 撤回残留 | `p=0.4998 / 0.9987 / no significant difference` | 已撤回 | 只出现在 §4.6（带 WITHDRAWN 标签）、Table 10（标 withdrawn）、Appendix B；**摘要 / 结论 / §1.2 / Figure caption 均为 0** |

## 3. 判定

| 闸门 | 结果 |
|---|---|
| 同配置数字唯一 | ✅ 16 条冻结值，`unique: true` |
| 所有表格 caption 自包含（不看正文可懂） | ✅ 每张表标 statistic / seeds / source |
| chance 只有一个定义 | ✅ `1/32 = 0.0313`，本任务专用 |
| 摘要中每个数字都能在冻结表里查到 | ✅ 摘要只含 `129.87 / 130.00`（C06/C07）与 "machine precision"（C21 的定性表述） |
| 旧值不作为结论陈述 | ✅ Table 10 列 superseded，Table 28 标不同配置 |

**Phase 4 判定：放行。**
