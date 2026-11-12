# 上游源码待修清单
## —— 白盒复现里成立 / 做到了，但 `kino-koki/SimplCore` 源码里不行或对不上的项

> **核对对象**：`https://github.com/kino-koki/SimplCore`，`main` 分支，
> zip 目录名 `kino-koki-SimplCore-e00b550`，`pushed_at 2026-11-04T13:57:07Z`，
> Apache-2.0，860 条目 / 731 文件 / 8.6 MiB 解压后，默认语言 Java（历史归档）。
> 仓库只有这一个 SimplCore，同用户另有 `TS2SC` / `Tuan-CLI` / `Tuan-Nursery-1B` / `Yama-Engine`。
>
> **核对方法**：API 取真实包 → 完整文件树 → 选择性解压源码 → 静态逐行核对 +
> **按源码公式独立复算**（`verify_source_grid.py`）。
> ⚠️ **未运行仓库的 pytest**（需 torch + 完整依赖，本沙盒 2 GB / 无 GPU）。
> 因此下面凡是写「源码守护/文档声称」都是**读到的文本**，不是我跑出来的测试结果。
>
> **重要前提**：`SimplCore_Paper.md` 与用户提供的 `/data/inputs/SimplCore_Paper.md`
> **逐字节一致**（285 行 / 17168 字节，`diff` 无输出）——所以上一轮的引用核对结论不变。

---

## 1. P0：明确 bug，可直接改

### P0-1 ⭐ `linear` 网格在主路径根本启用不了

| | |
|---|---|
| **白盒做到** | 把 θ 改成线性均匀后 cond 1.35e36 → 3.97e3，osc 上 1500 步 0.9844 vs TS 3000 步 0.8535 |
| **源码现状** | `simplcore/core/arch/sgc.py:330` 已支持 `init_mode="linear"`；`docs/frequency_grid_fix.md` §7 教用户 `SpectralGlobalConv(..., init_mode="linear")`；`test_linear_grid.py` 10 项守护齐备。但 **`simplcore/core/model/config.py:319-323` 的 `validate()` 白名单只有 `("legacy","logfreq","hybrid")`**，`registry.py:96/118/140/162/188` 把 `cfg.sgc_init_mode` 传下去。走 `SimplCoreConfig → SimplCoreModel` 时传 `"linear"` 直接抛 `ValueError: sgc_init_mode must be 'legacy' \| 'logfreq' \| 'hybrid'`。**整个模型主路径用不了已实现的修复**，只有绕过 config 直接构造 mixer 才行。 |

```diff
# simplcore/core/model/config.py:319-323
-        if getattr(self, "sgc_init_mode", "legacy") not in ("legacy", "logfreq",
-                                                            "hybrid"):
-            raise ValueError(
-                f"sgc_init_mode must be 'legacy' | 'logfreq' | 'hybrid', "
-                f"got {self.sgc_init_mode!r}")
+        if getattr(self, "sgc_init_mode", "logfreq") not in (
+                "legacy", "logfreq", "hybrid", "linear"):
+            raise ValueError(
+                f"sgc_init_mode must be 'legacy' | 'logfreq' | 'hybrid' | 'linear', "
+                f"got {self.sgc_init_mode!r}")
```

顺带：`:319` 的 `getattr` 默认值 `"legacy"` 与 `:191` 的字段默认 `"logfreq"` 不一致；且 `legacy` 在 `sgc.py:1462-1470` 会因代数公共零点**显式抛 ValueError**——config 放行、算子拒绝。建议要么把 `legacy` 从白名单删掉，要么在 validate 里提前给一句「legacy 网格有不可修时间盲点，构造会被拒绝」。

补一条回归测试：`SimplCoreConfig(sgc_init_mode="linear")` → `SimplCoreModel(cfg)` 能构造且 `mixer.init_mode == "linear"`。

---

### P0-2 ⭐ θ 钳制卫兵（P1）对默认与大预设形同虚设，而数学证明只测了小配置

| | |
|---|---|
| **白盒做到** | 由 θ_min=π/8192 实算 `1/\|sinθ\| = 2607.5947`；ρ_max 是双形式偏差第一杠杆（105× vs 18×） |
| **源码现状** | `sgc.py:30-38` `theta_clamp_floor(decay_max) = min(1e-3, π/decay_max)`；`sgc.py:551` `theta = ...clamp(self._theta_floor, π−1e-3)`。于是 decay_max=8192（默认 / tuan-150m）→ floor **3.83e-4**；`configs/tuan-2b.json:9` decay_max=**4_000_000** → floor **7.85e-7**（`1/\|sinθ\|≈1.27e6`）。源码注释自己写明「logfreq 配对设计不被钳」——**卫兵恰好豁免了它最该管的那个初值**。<br>更关键：`mathproof.py:274-275` P05 断言「θ ∈ [1e-3, π−1e-3]，\|sinθ\| ≥ sin(1e-3)」，`:306-309` 硬断言 `theta_lo >= 1e-3`；但 `:291` 构造的是 `decay_max=64`（floor=1e-3），**P05 只在这个不代表生产配置的小值上成立**。默认配置的误差界保证被高估约 2.6 倍，tuan-2b 高约 1270 倍。 |

修法二选一：

- **保守**：P05 改为按 `decay_max` 扫描 `(64, 8192, 16384, 4e6)`，把断言改成 `theta_lo >= theta_clamp_floor(decay_max)`，并在 `ProofResult.detail` 里如实打印每个档位的 `1/|sin floor|`；同时把 `docs/SGC_CORE.md` / 论文 §3.1 里「P1 卫兵」的保证口径改成「上界绝对钳制有效；下界随 decay_max 让步，不提供固定 1e-3 保证」。
- **彻底**：把 θ_floor 与 decay_max 解耦（固定 `1e-3`，logfreq 最慢基改为初始化后由 `coef` 缩放补偿），并把 P1 的「下界跟随 decay_max」从注释和 README 里撤掉。

---

### P0-3 ⭐ 默认 `logfreq` 被构造期数值秩守护**显式豁免**

| | |
|---|---|
| **白盒做到** | 默认网格 Gram cond 1.35e36、基间余弦 0.9965，32 维状态塌成近似 1 维 |
| **源码现状** | `sgc.py:1528-1531`：<br>`_assert_no_common_zero(theta0, num_bases)` 对所有模式跑；<br>`if mode in ("hybrid","linear"): _assert_numerical_rank(theta0, num_bases)`——**只对 hybrid/linear 跑**。`_assert_numerical_rank` 门槛 `max_cond=1e6`（`:1550`），而默认 logfreq 按源码公式复算 cond(Ψ)=**7.96e16**、数值秩 **24/32**（`verify_source_grid.py`，与 `docs/frequency_grid_fix.md` §3 的 1.92e16 / 24/32 同量级）。注释甚至直说「logfreq 的秩缺失是它们存在的理由」，等于承认默认路径不设防。 |

```diff
# simplcore/core/arch/sgc.py:1528-1531
     _assert_no_common_zero(theta0, num_bases)
-    if mode in ("hybrid", "linear"):
-        # hybrid / linear 必须过数值质量门（logfreq 的秩缺失是它们存在的理由）
-        _assert_numerical_rank(theta0, num_bases)
+    if mode in ("hybrid", "linear"):
+        _assert_numerical_rank(theta0, num_bases)          # 硬拒绝
+    else:
+        # logfreq 是历史默认：不破坏既有 checkpoint，但必须留下可审计的警告，
+        # 不能静默放行一个 cond~1e16 / rank 24/32 的字典。
+        rank, cond = _grid_rank_condition(theta0)
+        if rank < num_bases or cond > 1e6:
+            warnings.warn(
+                f"init_mode={mode!r}: damped-oscillator grid is rank-deficient "
+                f"(rank={rank}/{num_bases}, cond={cond:.2e}); "
+                f"consider init_mode='linear'.", RuntimeWarning, stacklevel=2)
```

并在 `docs/frequency_grid_fix.md` §7「不改默认的三条理由」里补第四条：**默认网格未过构造期数值质量门，仅以 warning 放行**。

---

### P0-4 同一算子两套闸门默认值

| | |
|---|---|
| **白盒做到** | 状态严格 `4(2MD+(K−1)D+1)` 字节、四长度逐字节一致（402,436 B） |
| **源码现状** | `sgc.py:275` `evidence_gate: bool = True, uncertainty_gate: bool = True`；<br>`config.py:139-140` `SimplCoreConfig` 默认 **False / False**；<br>技术报告 §3.1 写「evidence/uncertainty 闸门：**默认 OFF**」；<br>`docs/ARCHITECTURE_PIVOT.md:264` 自记 R9「**闸门默认开启改变了模型行为**，既有模型权重与新默认不等价，旧实验复现须显式 `evidence_gate=False, uncertainty_gate=False`」。<br>主路径经 `registry.py` 传 cfg 所以是 False，但**直接构造 `SpectralGlobalConv` 是 True**——同一算子两条路径行为不同，且 `sgc.py:1003-1017` 的 `streaming_state_bytes` 会在 True 时多算 `batch*4`。 |

```diff
# simplcore/core/arch/sgc.py:275-276
-        evidence_gate: bool = True,
-        uncertainty_gate: bool = True,
+        evidence_gate: bool = False,      # 与 SimplCoreConfig / 技术报告 §3.1 对齐
+        uncertainty_gate: bool = False,
```

同步改 `selective.py:101 / 379` 的同名默认值。`test_gates.py` / `test_gates_selective.py` 若依赖旧默认，显式传 `True`（测试意图不该依赖隐式默认）。

---

## 2. P1：口径 / 文档不一致（不改代码逻辑，但论文表述必须改）

| # | 白盒证据 | 源码里的矛盾 | 改哪里 |
|---|---|---|---|
| **P1-1** | fp32 下 L=64 就 `bitwise=False`（rel_err 1.15e-5）；显式 fp64 才 2.43e−14 → **数学等价，fp32 不逐位** | `test_audit_guards.py:282,305` 自己写「定理 2 的分路径口径：流式逐位精确因果；**FFT 并行 fp32 噪声 < 1e-5**」；`test_bio_stream.py` / `test_sgc_enhancements.py:51` 普遍用 `atol=1e-4~1e-5` 而非 `torch.equal`。但 `SimplCore_Paper.md:16` 摘要第 1 条与 §4 定理 1 仍写**无条件**「fp32 下逐位一致（守护测试锁定）」 | 摘要/定理 1 改为「流式路径逐位；整段路径 fp32 容差内一致，偏差随 L 增长（§5.4）」；§5.1–5.3 的「8 项数学核心全过」标注口径 |
| **P1-2** | 状态 402,436 B（含 `+1`），四长度逐字节命中公式 | `sgc.py:1003-1017`：`+1` 实际是 `if evidence_gate: extra += batch*4`，**是可选闸门标量，不是结构常数**。`config.py` 默认 False → 实际 402,432 B/层；§5.14 的 `4,829,184 B = 402432×12` 正是不含 `+1` 的口径。定理 3 把 `+1` 写成无条件公式 | 定理 3 标注「`+1` 仅在 L2 证据门开启时常驻」；或公式去掉 `+1`，另起一句说明 |
| **P1-3** | 我按 chance 0.125 构造 MQAR G=4，得 TS 0.3945 vs SGC 0.3066 | `axis_probe.py` 的 `TASK_SPECS` **只有 4 个任务**：`precise_pointer / aggregate_count / counting / markov`，**没有 MQAR 生成器**；`counting` 是 9 类、chance 1/9，不是 §5.5 的 0.5 奇偶计数。但论文 §5.5 表格 7 行里 2 行是 MQAR，附录 B 让读者跑 `python -m ...axis_probe --seeds 0 1 2 3 4` 复现——**该命令复现不了 MQAR 行** | 把 MQAR G=4 生成器搬进 `TASK_SPECS`（`eval/probes.py` 已有 MQAR 相关实现可复用），或把 §5.5 / 附录 B 的协议指向正确文件 |
| **P1-4** | osc 主结果 0.9844 / diff +0.1309 CI [+0.0996,+0.1621] / 收敛曲线 / 随机基消融 1.0000 | `docs/frequency_grid_fix.md` §1 明确标「**来源：用户的白盒复现报告第 15 节，非本仓复现**」，§9 又写「训练侧 acc / bootstrap CI / MQAR 数字来自用户报告，**本仓未重跑**」。全仓搜 `osc0` 只在该 md 出现 2 次，**没有任何 osc 任务脚本或 raw 结果**。`init_mode="linear"` 的收益目前只有一份仓库外孤证 | 落一个 `eval/osc_probe.py`（任务生成器 + 网格扫描 + 配对 bootstrap + 平凡基线）和一份 raw json；`docs/frequency_grid_fix.md` §1/§9 的「非本仓复现」才能撤掉 |
| **P1-5** | 我报「linear30 是最优解」 | `docs/frequency_grid_fix.md` §3 末尾自己写：**含 ρ 的口径 B 下 linear cond 9.53e6 并不优于被拒绝的 legacy 均匀 2.19e6**，原因是 θ∈[0.3,π−0.3] 关于 π/2 对称引入 `sin((τ+1)(π−θ))=(−1)^τ sin((τ+1)θ)` 冗余对；结论是「**纯线性均匀未必是最优终点，最优 θ 网格仍可继续扫**」。我按源码公式复算口径 B：logfreq 2.95e10 / hybrid 9.66e8 / linear 1.03e7，与文档一致 | 把非对称 θ 网格（打破 π/2 镜像）、`linear_floor` 连续扫描、含 ρ 口径纳入搜索；论文里「最优解」限定为「在本文扫过的组合与本文 Gram 口径下」 |

---

## 3. P2：我的白盒与源码**不是同一个算子**（不是源码 bug，但结论迁移前必须知道）

### P2-1 冲激响应差一个 `1/sinθ` 缩放 ⚠️ 最影响数字可比性

```
源码  sgc.py:954-957   c = sin(theta)
                       y_new = a·y1 + b·y2 + sin(θ)·u
      sgc.py:567-570   psi = ρ^τ · sin((τ+1)θ)          ← 不含 1/sinθ
      → 信号冲激响应 = ρ^τ sin((τ+1)θ)

白盒  sgc_core.py:78-102  h = ρ^t sin((t+1)θ) / sin(θ)  ← 含 1/sinθ
      iir_scan 分子 [1]（+u_t，无 sinθ 注入）
      → 信号冲激响应 = ρ^t sin((t+1)θ)/sinθ
```

两者**函数类等价**（可学习 `coef` 能吸收每基 `1/sinθ` 缩放），但：

- 训练动态 / 梯度尺度差 `1/sinθ`，θ_min 处达 **2607×**；
- 数值误差特性不同：源码把 `sinθ` 放在输入侧，信号核在 θ→0 时幅度趋 0；我的白盒让 θ→0 的信号被放大到 `(t+1)ρ^t`；
- 我的命题 5 实测、双形式偏差、Gram cond、osc 训练结果**全部在这个不同归一化下得到**。

> 顺带说明论文 §4 命题 5「舍入误差注入系数为 1（非 sinθ），误差响应为 ψ/|sinθ|」在源码里是**成立**的：舍入注入进 `y_new` 的加法结果与输入缩放 `c=sinθ` 无关，误差传播增益仍由递推齐次解决定、含 `1/|sinθ|`。这一条源码与论文自洽，不用修。

### P2-2 网格参数三处不同（我按源码公式独立复算，见 `verify_source_grid.py`）

| 项 | 我的白盒 | 源码 `sgc.py` | 影响 |
|---|---|---|---|
| logfreq θ 上界 | `π−1e-3 ≈ 3.1406` | `0.9·π/2 ≈ 1.4137`（源码注释：内缩避免 Nyquist 对称退化） | 我的默认网格多了 π/2 镜像冗余，**比源码更病态** |
| ρ 网格 | 线性 `0.9 → 0.999` | `lam=exp(linspace(log1,log8192,M)); ρ=exp(−1/lam)` → `0.368 ~ 0.99988` | 记忆窗口分布完全不同 |
| linear 网格 | `linspace(0.3, π−0.3, 32)` **含端点** → `[0.300, 2.842]` | `(m+0.5)/M` 栅格 → `[0.3397, 2.8019]`（源码注释：避开端点代数结构） | 我的 linear30 ≠ 源码 `init_mode="linear"` |

**三种条件数口径对齐结果**（`results_source_grid_check.json`）：

| 口径 | logfreq | hybrid | linear | 出处 |
|---|---:|---:|---:|---|
| A：`Ψ=sin((τ+1)θ)`, L=512, SVD, 不含 ρ | **7.96e16**（秩 24/32） | 6.52 | **1.04** | 源码 `docs/frequency_grid_fix.md` §3 报 1.92e16 → 1.04；**linear 的 1.04 精确命中** |
| B：含 ρ, L=1024, 列归一化 Gram | 2.95e10 | 9.66e8 | 1.03e7 | 源码报 2.95e10 → 9.53e6，同量级 |
| 本文报告：含 ρ, L=128, **fp32 未归一化 Gram** | 9.70e31（源码公式）/ **1.35e36**（白盒网格） | 6.79e31 | 2.15e8（源码公式）/ **3.97e3**（白盒网格） | 我的 1.35e36 是**fp32 数值条件数**（最小特征值被舍入钳在 ~1e−30），不是数学条件数；换 fp64 会变 1e300 量级 |

> 结论：**方向与机制完全一致**（logfreq 低频塌缩、抬 θ_min 后秩满且良条件），但绝对值不是同一口径。上一版报告里「cond 改善 8~9 个数量级」应改写为「在本文 Gram 定义下改善约 32 个数量级；源码 SVD 口径下改善约 16 个数量级」，并注明口径。

### P2-3 `srf` 的 selective 门曾有死参数史

`test_selective.py:119-124` 记录：「`gate_gamma`/`gate_beta` 权重零初始化 + `nn.Linear(bias=False)` 无偏置 → 门恒等于 `sigmoid(0)=0.5`，**『逐位置逐通道调制写入强度』名存实亡（selective 路径实际是死参数）**」。我白盒的 selective 门控实测 **0.2871**（比 LTI 基线 0.3066 更差，多花 9408 参数），与「只给写不写、不给写到哪」的诊断一致。测试现已修，但建议补一条端到端守护：默认预设下 `srf` 的 γ/β 门确实随输入变化，且开启后在 MQAR 类任务上不劣于 SGC。

### P2-4 论文内部两处（上一轮已列，源码侧确认）

- `SimplCore_Paper.md:117` §3.3 写「MQAR 对照（**§7.2**）」，但 §7 下无 7.2 小节 → 应指 §5.5。
- §5.12 给 1.948 GiB（权重 1.888 GiB）却未给参数量/层数；按 bf16 2 B/参数反推约 **10.1 亿参数**，与 §5.14 明确写的 **1.085 亿参数（dim=768/12 层/M=64）** 不一致，源码 `configs/tuan-150m.json` 对应后者。建议补一句「§5.12 为另一更大预设」。

---

## 4. 这些**不用修**，源码已经落地（别重复劳动）

| 我白盒里的「发现」 | 源码 |
|---|---|
| 改线性频率网格是最优解 | `sgc.py` 已有 `init_mode="linear"` / `hybrid` / `logfreq` / `legacy` 四模式、`linear_floor=0.3`、`docs/frequency_grid_fix.md` 整篇、`tests/test_linear_grid.py` 10 项守护（含 `test_linear_costs_nothing`、`test_linear_parallel_matches_streaming`、`test_linear_is_orthogonal_even_with_rho_decay`） |
| 「反超原因是正交性不是振荡先验」 | `sgc.py:1445-1450` 的 `_init_oscillator_grid` docstring 已写「★ 归因修正：换成纯随机正交基同样能赢……起作用的是基的正交性/可分性」；`docs/frequency_grid_fix.md` §4 整节 |
| 「正交网格救不了 MQAR 绑定轴」 | 同上 docstring「★ 边界：该改动只在周期/连续信号轴兑现；key→value 绑定轴无收益甚至回退」；`docs/frequency_grid_fix.md` §5 |
| R1 定容绑定记忆 / L0–L2 三层引用 | `memory/binding_memory.py`、`memory/sidecar.py`、`memory/sidecar_exact_attn.py` 均在，`tests/test_binding_memory.py` / `test_sidecar.py` 在；`REPRODUCTION_REPORT.md` 归档注记标 R1/R2/R3 ✅ 已落地 |
| 13 项机器可验证证明 | `eval/mathproof.py` P01–P13 齐全（P01 闭式↔IIR、P04 FFT 无混叠、P06 流式==并行 fp64、P10 O(1) 状态充分性、P12 HBM 账本） |
| 双形式 / 因果 / 守护 | `test_mixer_math.py`、`test_audit_guards.py`、`test_architecture_purity.py`、`arch/purity.py` 齐全；`backend/` 有 16-op 契约与 fp32 CPU 参考后端 |

> ⚠️ 这也意味着**投稿定位要改**：`init_mode="linear"` 与正交性归因不是「本文独立发现」，上游 `docs/frequency_grid_fix.md` 已明确把训练收益归给「用户的白盒复现报告第 15 节」。写作上应改成「本文独立实测定位了字典塌缩并给出可复现数据；上游据此落地了 `init_mode=linear`」，或按上一版 §16.5 方案 A 把两手稿合并。

---

## 5. 建议的修的顺序

1. **P0-1**（一行白名单）→ 立刻让 `linear` 在主路径可用，并补一条 config 回归测试。
2. **P0-3**（logfreq 加 warning）→ 默认路径不再静默放行塌缩字典。
3. **P0-2**（P05 按 decay_max 扫描）→ 修正卫兵保证的口径，避免数学证明高估。
4. **P0-4**（统一闸门默认）→ 消掉 R9 的两套行为。
5. **P1-4**（补 osc probe + raw json）→ 让 `linear` 的收益在仓库内可复现，撤掉「非本仓复现」。
6. **P1-3**（MQAR 进 `TASK_SPECS`）→ 让附录 B 的复现命令真能复现 §5.5。
7. **P1-1 / P1-2 / P1-5** 改论文表述与网格扫描范围。
8. **P2** 不用改源码，但投稿前把白盒数字按源码 psi 归一化重跑一遍，或在文稿里标清口径差异。

---

## 附：本轮产出的核对脚本

| 文件 | 用途 |
|---|---|
| `verify_source_grid.py` | 按源码 `_init_oscillator_grid` 公式独立复算三种条件数口径，可重跑 |
| `results_source_grid_check.json` | 复算结果（口径 A/B/本文 Gram） |
| `SimplCore_upstream_e00b550.zip` | 上游完整真实包（2.98 MB，860 条目），已一并交付 |
| `SimplCore_src/` | 选择性解压的关键源码/文档（26+17 个文件），供逐行核对 |
