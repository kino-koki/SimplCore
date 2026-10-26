# 可复现性指南（SimplCore-V 0.1-ALPHA）

**目标**：任何人在干净环境按本文档操作，核心链路验证通过率 ≥ 90%。

---

## 1. 环境锁定

| 项 | 要求 | 实测环境 |
|----|------|---------|
| Python | 3.11.x | 3.11.15 |
| PyTorch | 2.4.1+cpu（CPU 验证） | 2.4.1+cpu（`.venv311`） |
| 操作系统 | Windows / Linux / macOS | Windows 11（AMD 680M） |
| GPU | 可选；训练需 CUDA/ROCm | 本机无可用 GPU 训练 |

安装：
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

> 注意：早期仓库有 4 套 venv（`.venv`/`.venv_sc`/`.venv21`/`.venv311`）。
> 单一推荐 `.venv311`（torch 2.4.1）。其余为历史残留，不影响主线。

---

## 2. 单一入口

| 操作 | 命令 | 产出 |
|------|------|------|
| 核心链路冒烟 | `python smoke_test.py` | `smoke_result.json`，通过率 |
| 集成自测 (MoE/流式常数显存/原生多模态) | `python -m simplcore_pg.test_integration_smoke` | `integration_smoke_result.json`，3/3 |
| 流式一致性验证 | `python test_stream_consistency.py` | 流式 vs 整段前向 max_err |
| 长序列延迟基准 | `python bench_rfp_vs_attention.py --max-seq 2048` | `bench_rfp_vs_attention.json` |
| 复杂度阶数验证 (RFP·Local 线性 vs FullAttn 二次) | `python -m simplcore_pg.bench_throughput` | `bench_throughput_result.json` |
| 长序列精度对比（RFP vs 注意力） | `python bench_longctx_accuracy.py --train-steps 80` | `bench_longctx_accuracy.json` |
| **全局路由复制（局部 vs 全局消融）** | `python bench_multihop.py --seq-len 48 --S 16` | `bench_multihop.json` |
| 训练闭环（CPU 冒烟） | `python -m simplcore_pg.training.train --smoke --max-steps 30` | loss 下降日志 |
| 可用性核查（诚实版） | `python demo_simplcore_availability.py` | `SimplCore_availability_report.json` |
| 长上下文显存公式 | `python benchmark_longctx_memory.py` | `longctx_memory_benchmark.json` |
| 真实中文语料困惑度对比 | `python bench_corpus_ppl.py` | `bench_corpus_ppl.json`（SimplCore vs Transformer 基线） |
| **组件消融（真实语料逐组件归因）** | `python bench_ablation.py --steps 200 --hidden 96 --layers 2` | `bench_ablation.json`（full/no_rfp/no_dmc/no_recurrence/no_conv 五变体 val_ppl） |
| **规模扩展（真实语料困惑度随规模下降）** | `python bench_scaling.py --steps 200 --hidden-list 64,96,128 --layers-list 2,2,3 --rfp-freeze` | `bench_scaling.json`（H=64/96/128 三规模点 val_ppl；`--rfp-freeze` 为 Windows/CPU 下 FFT 反向泄漏的文档化回退） |
| **论文级图表（从 bench JSON 实时生成 SVG）** | `python make_paper_figures.py` | `docs/figures/fig_scaling_curve.svg` / `fig_ablation.svg` / `fig_rfp_latency.svg` |
| 权重安全保存/加载管线 | `python -m pytest tests_simplcore_pg/test_checkpointing.py` | 明文/加密往返、口令错误拒绝、篡改指纹检测、RBAC 拒绝、用后即焚，10/10 |

> **Windows/CPU 训练步数护栏**：FFT 反向在 Windows/CPU 的 torch 构建下有 workspace
> 累积泄漏。RFP 多步演化已默认经 `torch.utils.checkpoint` 包裹以缓解；CPU 下建议训练
> 步数 ≤150、batch≤8（已验证稳定）。若仍触发，可设 `rfp_freeze_evolution=True` 回退为
> 固定全局混合器。长训练用 GPU/ROCm。

**`smoke_test.py` 设计原则（关键）**：
- 每个用例独立判定 `PASS/FAIL`；
- **任何真实异常记 FAIL，绝不降级为 SKIP 后报成功**；
- 退出码：通过率 ≥90% 返回 0，否则 1；
- 物理证明用例显式断言 `all_passed is False`（防止"已学会"类不实结论回潮）；
- 包含"无注意力配置可构造/前向/反向"用例，守护 v2.0 架构不回退到注意力。

---

## 3. 复现验证结果（2026-08-24 实测）

```text
✅ 无注意力架构 构造/前向/反向 (LocalInteraction + RFP + DMC + MoE，源码无注意力)
✅ 长上下文流式 前向/反向/状态传递 (stream_init/stream_step/generate_stream 均验证)
✅ ResonanceField 频谱演化 可训练梯度可达 (omega/sigma/beta + 投影)
✅ DynamicMemoryChannels 记忆读写 梯度可达
✅ RFP vs 注意力 长序列延迟基准 (n=2048 时 RFP 快 3.0×，交叉点 n≈700)
✅ 物理规律诚实证明 加载权重实测 (rel_err spring=12.2%)
✅ torch.compile 安全回退
✅ 无注意力架构 源码审计 (无 Q/K/V/RoPE/SDPA，含原生多模态 ViT 与推理核)
通过率: 8/8 = 100.0%
```

> 注：v2.0 起 smoke 用例已覆盖无注意力配置（含对源码的注意力泄漏断言）；
> `bench_multihop.py` / `bench_throughput.py` / `bench_longctx_accuracy.py` / `test_integration_smoke.py`
> 为有独立 JSON 产出的消融/集成基准，不计入 smoke 通过率但纳入整体可复现证据链。
> 当前核心链路 **全部绿灯**：smoke 8/8、integration 3/3、流式一致性 max_err=2.4e-7、
> 长序列精度两者均 acc=1.0。

---

## 4. 已知不可复现项（诚实列出）

1. **GPU 显存实测**：本机 AMD 680M + Windows 下 PyTorch 不识别 AMD 训练；
   真实峰值显存需在 WSL2+ROCm 或 Linux+ROCm 重测。当前 `longctx_memory_benchmark.json`
   为公式 + CPU 小尺度验证 + 固定状态数学证明。
2. **CUDA 内核编译**：`simplcore_pg/cuda/simplcore_kernels.cu` 源码存在但本机无 nvcc，
   未编译/加载验证。
3. **56B 全量实例化**：CPU 内存不足，仅用小尺度配置验证组件接入与梯度正确性。
4. **Java 并行实现**：独立代码线，本文档不保证其可复现性。

---

## 5. 如何新增可复现用例

在 `smoke_test.py` 中加一个 `@case("名称")` 函数，返回字符串描述；抛异常即 FAIL。
不要为"环境缺失"的用例用 SKIP——要么真实 FAIL，要么明确环境前提并在文档 §4 列出。
