# 二阶 IIR 长卷积的数值稳定性（原创点 A）

> 对应 `SimplCore-原创贡献清单.md` 的 **A**：误差界 / 基字典排布 / 重同步准则。
> 代码：`simplcore/core/numerical_stability.py`；验证：`test_numerical_stability.py`、
> `test_long_sequence_error.py`；测量脚本：`scripts/measure_long_sequence_error.py`。

## 1. 结论（先看这里）

| 判据 | 状态 | 证据 |
|---|---|---|
| 误差界能预测实测（同一数量级） | **部分成立**：单基界可覆盖实测；复合界需传入实测 `state_scale` | `test_bound_does_not_underestimate_single_basis` |
| 误差感知网格相对对数网格 ≥2× | **L=2048 成立（2.2×）；L=8192 只有 1.8×** | 下表 |
| 重同步周期由误差预算反解 | **成立**（闭式初解 + 精确二分） | `test_resync_period_respects_budget` |

**没有做成的地方**：复合误差界在默认 `state_scale=1` 时会低估长基（实测可低 10×）。
界本身没错，但 `state_scale` 必须由调用方按实际状态幅度给；这是使用契约，不是自动估计。

## 2. 误差模型

单基信号：`y_t = a y_{t-1} + b y_{t-2} + c u_t`，`a=2ρcosθ, b=−ρ², c=sinθ`。
fp32 下每步舍入 `δ_t` 的传播（v1.1.2 修正，与论文命题 5 同步）：

```
e_t = a e_{t-1} + b e_{t-2} + δ_t
|e_t| ≤ max|δ| · Σ_{k≤t} |ψ_k| / |sinθ|,   ψ_k = ρ^k sin((k+1)θ)
max|δ| ≤ ε · (|a|·|y| + |b|·|y| + |c|·|u|),  ε = 2^-24
```

关键在注入系数：舍入误差 `δ` **直接加在 y 上（注入系数 1）**，而信号注入
系数是 `c = sinθ`——因此误差冲激响应不是 `ψ` 而是 `ψ/sinθ`（同一齐次模态、
注入端归一）。`single_basis_error_bound` 的增益已按此修正。

随 `ρ→1` 与 `θ→0` 双重变差：

- `Σ|ψ|/|sinθ|` 中 `Σ|ψ| ≈ 1/(1−ρ)`（P05 的 ℓ1 界），`1/|sinθ|` 是**长基
  相位**常数（v1.1.2 起显式进入 P05，不再是"隐含"）；
- `max|δ|` 里的 `|a|,|b| → 2,1`；
- `θ→0` 或 `π`（特征根重合/翻转）时界退化——网格构造（logfreq 默认 +
  `_assert_no_common_zero`）保证初始化避开。

**P05 是核的 ℓ1 界，不是数值误差界。** 拿它判 fp32 长序列绝对偏差是口径错配。

## 3. 实测（M=64, D=128, decay_max=8192）

| L | legacy | logfreq | error_aware | error_aware / logfreq |
|---|---|---|---|---|
| 2048 | 5.40e-4 | 9.41e-5 | 4.25e-5 | **0.45（2.2×）** |
| 4096 | 1.17e-3 | 1.40e-4 | 1.00e-4 | 0.71（1.4×） |
| 8192 | 2.21e-3 | 2.07e-4 | 1.15e-4 | 0.56（1.8×） |

（`full` vs 逐 token `step`，fp32，含 content gate；`local_kernel=4`。）

结论：
- **短阈值不能外推**：`<4e-5` 只在 `L≲512` 成立（`docs/SGC_CORE.md §3.2` 已改）。
- `logfreq` 相对 legacy 降约 10×；`error_aware` 再降 1.4–2.2×。
- `error_aware` 在 L=8192 未达 2×，剩余误差已不是长基数量主导（把长基从 8 降到 4
  误差只从 1.15e-4 到 1.14e-4），而是短基/局部卷积/FFT 侧的本底误差。

## 4. 基字典排布（`error_aware_grid`）

两个正交手段：

1. **长基稀疏**：`λ∈[decay_max/16, decay_max]` 的基从几何网格的约一半压到 1/8；
   最长基保留（`ρ_max` 不变），长程**秩**下降。
2. **长基相位抬升**：`ρ>0.99` 的 `θ` 抬到 `π/3`；`1/sinθ` 从 `1e3~1e6` 量级降到 ~1.15。

`error_aware` 不是免费午餐：长基不再密集覆盖低频，适合长程检索/复制，
不适合极慢平滑结构。采用前必须在目标任务上 A/B；默认网格为 `logfreq`
（v1.2 起；legacy 均匀网格因公共零点被构造期守护拒绝），`error_aware`
保持 opt-in。

## 5. 重同步周期（`resync_period`）

给定单基误差预算，返回最大 N：

```
N = max{ n : max|δ| · Σ_{k<n}|ψ_k| ≤ budget }
```

实现用闭式上界 `Σ|ψ| ≤ (1−ρ^n)/(1−ρ)` 给初解，再倍增 + 二分校正。
示例（`θ=π/3`, `state_scale=1`, `ε=2^-24`）：

| ρ | budget 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|
| 0.99 | 10 | ≥4e6 | ≥4e6 | ≥4e6 |
| 0.999 | 9 | 106 | ≥4e6 | ≥4e6 |
| 0.9999 | 9 | 100 | 1068 | ≥4e6 |
| 0.999999 | 9 | 100 | 1014 | 10189 |

`0` 表示"该基在 fp32 下连 N=1 都超预算"（必须 fp64 / 换参数化）；
`≥4e6` 表示预算内 N 超过 `max_period` 上限。

注意：**分块本身不自动降误差**。实测 `chunk=512` 的 `forward_chunk` 在 L=8192
是 2.89e-3（比逐 token 的 1.26e-3 更差）；"分块重同步"只有在块边界**用高精度重算
状态**时才成立，单纯切块不是解。

## 6. 4M 的 ρ 参数化（已实现 opt-in）

旧实现有两道夹子：

- `sgc._logit(p, eps=1e-4)` 初始化把 `ρ` 夹到 `0.9999` —— 所以
  `decay_max=4e6` 预设**实际最长衰减只有 1e4**（实测 `rho_max=0.9999`）；
- `sgc.oscillator_params` 的 `clamp(max=1−1e−6)` 让训练也只能到 `0.999999`，
  与 4M 差 400 倍。

已落地：`sgc_rho_param="logspace"`（opt-in，`tuan-2b` 已启用）：

```
rho = exp(-softplus(raw_rho_log))
raw_init = log(expm1(1/λ))
```

- 无 sigmoid 饱和；上限 `1−2^-23` 是 fp32 在 1 附近的分辨率（时间常数 ≈8.4M），
  4M 所需的 `0.99999975` 可达（`test_rho_logspace` 断言 `rho_max ≥ 0.9999997`）；
- 初始化 `decay_max` 不再被 `_logit` 截断；`tuan-2b` 用 `decay_max=4e6` +
  `init_mode="logfreq"`；
- `_load_from_state_dict` 在 `raw_rho ↔ raw_rho_log` 之间自动迁移旧 checkpoint；
- 默认仍是 `sigmoid`，旧模型逐位不变。

**这是 4M 的前置，不是优化项。** 剩余限制是 fp32 极点分辨率：若以后要 16M+
时间常数，需要 fp64 状态或换状态表示。

## 7. 复现

```bash
python -m pytest simplcore/core/tests/test_numerical_stability.py -q
python -m pytest simplcore/core/tests/test_long_sequence_error.py -q
python scripts/measure_long_sequence_error.py --lengths 512,1024,2048,4096,8192
```