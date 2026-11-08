# 上游仓库核对包 · 索引（2026-11-04）

核对对象：`github.com/kino-koki/SimplCore`，main @ `kino-koki-SimplCore-e00b550`
（pushed 2026-11-04T13:57:07Z，Apache-2.0，860 条目 / 731 文件 / 8.6 MiB）。

## 交付物

| 文件 | 内容 |
|---|---|
| **`upstream_fixlist.md`**（=《源码待修清单.md》） | **主交付**：白盒成立但源码不行/对不上的 14 项，分 P0/P1/P2，每条给 `文件:行` + 可直接改的 diff |
| `SimplCore_upstream_e00b550.zip` | 上游**完整真实包**（2.98 MB，未删改） |
| `verify_source_grid.py` | 按源码 `_init_oscillator_grid` 公式复算三种条件数口径 |
| `results_source_grid_check.json` | 复算结果 |
| `SimplCore_src/` | 选择性解压的 43 个关键源码/文档，供逐行核对 |

## 一句话结论

仓库里 `init_mode="linear"`、hybrid 网格、正交性归因、R1 绑定记忆、三层 sidecar、
13 项 mathproof **都已经落地**；真正要修的是四个 P0：

1. `config.py:319` 白名单漏了 `"linear"` → **主路径启用不了已实现的修复**
2. `theta_clamp_floor` 随 decay_max 让步，而 `mathproof.py` P05 只在 `decay_max=64` 上证明「θ≥1e-3」→ 默认 8192 / tuan-2b 4e6 档的误差界保证被高估
3. `_assert_numerical_rank` 只跑 hybrid/linear → **默认 logfreq 的 cond~1e16 / 秩 24/32 被静默豁免**
4. `sgc.py` 闸门默认 True vs `config.py` / 论文默认 False → 同一算子两套行为

外加 `axis_probe.py` 没有 MQAR 生成器（复现不了 §5.5）、仓库没有 osc 任务脚本
（`docs/frequency_grid_fix.md` 自己标「非本仓复现」）。

## 核对边界

- `SimplCore_Paper.md` 与用户提供的 `/data/inputs/SimplCore_Paper.md` **逐字节一致**（diff 无输出）。
- 本轮是**静态逐行核对 + 按源码公式独立复算**，**未运行仓库 pytest**（需 torch/完整依赖，
  本沙盒 2 GB 无 GPU）。凡写「源码守护声称」均为读到的文本，不是我跑出的测试结果。
- 未改动上游任何文件；`SimplCore_upstream_e00b550.zip` 是原样 zipball。
