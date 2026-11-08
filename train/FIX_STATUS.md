# 0.1-β 修复状态（2026-11-05）

> 依据 `docs/UPSTREAM_FIXES_20261105` 一类的独立静态核对清单。本次提交只做
> **可在无 torch 环境下静态验证、且不改变既有 checkpoint 行为**的修复。
> 沙盒为 2 核 CPU / 2 GB，**未运行本仓 pytest**；所有「守护通过」的说法
> 指的是读到的源码文本，不是本轮跑出的测试结果。

## 已修（本次提交）

| # | 文件 | 改动 | 风险 |
|---|---|---|---|
| P0-1 | `simplcore/core/model/config.py` | `sgc_init_mode` 白名单加入 `"linear"`；`getattr` 默认值 `"legacy"` → `"logfreq"`（与字段默认一致）；错误信息补充 `legacy` 会在算子构造期被拒 | 低：纯放宽 + 默认值对齐 |
| P0-3 | `simplcore/core/arch/sgc.py` | `import warnings`；`init_mode="logfreq"` 不再静默绕过数值秩门——计算 `_grid_rank_condition` 后发 `RuntimeWarning`（不硬拒绝，保既有 checkpoint） | 低：不改构造结果，仅新增告警 |
| P0-2（口径） | `simplcore/core/eval/mathproof.py` P05 | 文档写明 θ 钳制下界随 `decay_max` 让步；`ProofResult.detail` 打印 64/8192/16384/4e6 四档实际下界与 `1/|sin(floor)|`。**`ok` 判定逻辑未改** | 低：纯信息 |
| P1-1 | `SimplCore_Paper.md` 摘要第 1 条、定理 1 | 把无条件「fp32 逐位一致」改成分路径口径（fp64 数学等价 / 流式 fp32 逐位 / 整段 FFT fp32 容差 <1e-5），与 `test_audit_guards.py::test_causality_by_path` 已有口径一致 | 文档 |
| P1-2 | `SimplCore_Paper.md` 定理 3 | 标注 $+1$ 是 L2 证据门的 `batch·4` B 常驻标量，默认 OFF 时公式为 $4(2MD+(K{-}1)D)$；§5.14 的 4,829,184 B 按后者 | 文档 |
| P1-5 | `docs/frequency_grid_fix.md` §10 | 三种条件数口径对齐表、fp32 数值条件数说明、「linear 最优」的限定、构造期守护记录 | 文档 |
| 引用 | `SimplCore_Paper.md` §3.3 | 悬空 `§7.2` → `§5.5` | 文档 |

## 明确未修（附理由，不隐瞒）

### P0-4：闸门默认值两套口径 —— **故意不改**

- `simplcore/core/arch/sgc.py:275-276`：`evidence_gate=True, uncertainty_gate=True`
- `simplcore/core/model/config.py:139-140` 与技术报告 §3.1：默认 `False`
- `docs/ARCHITECTURE_PIVOT.md` R9 自记「闸门默认开启改变了模型行为，既有模型权重与新默认不等价」

**为什么不改**：把算子默认统一到 `False` 会改变所有直接构造 mixer 的既有
行为与 checkpoint 等价性；而验证它需要跑完整 pytest（86 个测试文件），
本沙盒无 torch / 无 GPU，跑不了。在没有回归验证的前提下改默认值，
比留着这个不一致更危险。

**建议的改法**（在有测试环境的机器上做）：

```diff
# simplcore/core/arch/sgc.py:275-276
-        evidence_gate: bool = True,
-        uncertainty_gate: bool = True,
+        evidence_gate: bool = False,
+        uncertainty_gate: bool = False,
```

同步改 `simplcore/core/arch/selective.py:101, :379`；
`test_gates.py` / `test_gates_selective.py` / `test_long_sequence_error.py` 等
目前多数已显式传 `True`，改动后逐个跑一遍确认。同时把
`ARCHITECTURE_PIVOT.md` R9 标为 resolved。

### P1-3：`axis_probe.py` 复现不了论文 §5.5 的 MQAR 行

`TASK_SPECS` 只有 `precise_pointer / aggregate_count / counting / markov`，
无 MQAR 生成器，`counting` 是 9 类 chance 1/9；但附录 B 让读者跑
`axis_probe --seeds 0 1 2 3 4` 复现 7 行表格（含 2 行 MQAR）。
需把 `eval/probes.py` 已有的 MQAR 实现搬进 `TASK_SPECS`，或修正附录 B 的
协议指向。**本次未动**，因为涉及评估协议语义，需先确认论文表格的任务定义。

### P1-4：`init_mode="linear"` 的训练收益仍是仓库外孤证

`docs/frequency_grid_fix.md` §1/§9 自己标注「来源：用户的白盒复现报告
第 15 节，非本仓复现」。全仓无 osc 任务脚本、无 raw 结果。
本次新增的 `tools/transfer_distill/` 是**字符级语言建模**前哨，
**不是** osc 周期轴实验，不能替代这个缺口。需补 `eval/osc_probe.py`
（任务生成器 + 网格扫描 + 配对 bootstrap + 平凡基线）与 raw json，
才能撤掉那句「非本仓复现」。

### P2：白盒与源码不是同一个算子（不是 bug，但数字不可互换）

- 上游 `build_kernel`：`ψ[τ,m] = ρ^τ sin((τ+1)θ)`（**不含** `1/sinθ`；
  `step()` 输入注入 `c=sinθ`，恰好抵消递推的 `1/sinθ`）
- 更早白盒 `h = ρ^t sin((t+1)θ)/sinθ`（含 `1/sinθ`）
- 网格参数三处不同：θ 上界 `0.9π/2` vs `π−1e-3`；ρ 网格
  `exp(−1/lam)` 0.368~0.99988 vs 线性 0.9~0.999；linear 用
  `(m+0.5)/M` 栅格 `[0.3397, 2.8019]` vs 含端点 `linspace(0.3, π−0.3)`

函数类等价（`coef` 可吸收缩放），但梯度尺度差 `1/sinθ`（θ_min 处 2607×），
条件数/偏差/训练数字**不是同一口径**。见 `docs/frequency_grid_fix.md` §10。

## 本次未触碰

- 任何 `init_mode` 默认值（仍是 `logfreq`，§7 的三条理由保留）
- 任何既有 checkpoint / 权重文件
- 任何注意力相关代码（仓库内仍无 attention 模块）
- `tools/transfer_distill/` 是独立白盒实现，不 import 本仓 `simplcore`，
  不改主产品线任何行为
