# SimplCore-V 0.1-ALPHA

以 PyTorch 为主线、**无注意力 (attention-free)** 的可训练 AI 架构仓库。
当前**主推架构是 SGC**（`simplcore/sgc/`）：内容门控的因果谱全局卷积。

> ### ⚠️ 2026-10 重大更正（请先读这一段）
>
> 一次真实执行（而非阅读式）的独立性审计发现：本仓库此前的**主线** `simplcore_pg`
> 中承担全局混合的 RFP（ResonanceField）**是非因果算子**——它用实值（零相位）带通窗
> 做谱滤波，而实值频率响应对应的冲激响应必然关于 t=0 偶对称，因此必然看到未来 token。
> 实测：只扰动最后一个 token，前 511/511 个位置的输出全部改变。
>
> **后果**：用旧 RFP 训练出的语言模型，其交叉熵是被"偷看答案"压低的，
> 因此本文档与 `docs/HONEST_STATUS.md` 中所有基于旧 RFP 的**语言建模结论（困惑度、
> 消融、规模扩展）一律作废**。
>
> **状态**：该缺陷**已修复**——`ResonanceField` 现在默认构造因果 FIR 并做严格因果卷积
> （`causal=True`），并新增 `tests_simplcore_pg/test_causality.py`（12 项）守护；
> 修复后 `tests_simplcore_pg` 全绿（134 项）。**但受影响的实验结果必须重跑**，
> 在重跑完成前不得引用。
>
> 另有两条路径的同类问题：`simplcore/mix` 存在三处序列维全局归约破坏因果性，
> 且它其实是自研 NumPy 引擎而非 PyTorch。详见 `docs/HONEST_STATUS.md` 第零节。
>
> **因此：新工作请使用 SGC（`simplcore/sgc/`），它对因果性有实测守护。**

---

## 项目结构

| 路径 | 定位 | 状态 |
|------|------|------|
| **`simplcore/sgc/`** | **SGC —— 主推架构**（因果谱全局卷积 + 内容门控） | **可用，测试全绿** |
| `simplcore_pg/` | 早期主线（LocalInteraction + RFP + DMC + MoE） | RFP 因果性已修，**历史结果作废待重跑** |
| `simplcore/mix/` | 纯文本路线 | **非 PyTorch（NumPy 引擎）；三处非因果已于 2026-11 修复**（因果前缀统计 + 因果掩码，守护测试 `tests_simplcore_mix/` 全绿），可用于自回归 LM 训练 |
| `verify_sgc/` | SGC 原型实验脚本 | 历史实验，语料与显存测量已被证伪，仅存档 |
| `.archive/` | 历史/失真文档 | 未删除，可回溯 |

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/SGC_SPEC.md` | **SGC 规范**：设计依据、对偶形式、与 v2 的逐条对照、诚实边界 |
| `docs/SGC_A100_RUNBOOK.md` | **1×A100 80GB 运行手册**：语料 → 标定 → 训练 → 评测 → 排障 |
| `docs/HONEST_STATUS.md` | 诚实状态总表（**第零节为 2026-10 重大更正**） |
| `docs/PAPER_SPEC.md` | 早期论文级规范（基于 RFP，数字待重跑） |
| `docs/LONG_CONTEXT_STREAMING.md` | 长上下文流式设计 |
| `docs/DREAMING_CMT.md` | Dreaming 自进化反思 + CMT 连续机器学习（残差流加性记忆） |
| `docs/ARCHITECTURE_PIVOT.md` | 架构转向记录：去 PyTorch / 四层闸门 / 待办清单 |
| `docs/CLEANUP_2026-10-26.md` | 仓库清理审计：删了什么、为什么、怎么重建 |
| `docs/REPRODUCIBILITY.md` | 环境锁定与可复现性 |
| `README.md` | 本文件 |

---

## SGC 是什么

一个**因果全局卷积**序列混合器：每个位置看到全部前文，复杂度 `O(L log L)`，
推理时以 **O(1) 状态**流式运行。

核由 **M 个阻尼振荡基**给出：`ψ_m[τ] = ρ_m^τ · sin((τ+1)θ_m)`，`K[τ,d] = Σ_m C[m,d]·ψ_m[τ]`。
该核有一个**对偶形式**，这是 SGC 的核心：

* **训练（并行）**：显式构造核，零填充到 `2L` 用 FFT 做卷积 —— `O(L log L)`；
* **推理（流式）**：同一算子的二阶 IIR 递推 —— 每个基仅 2 个状态，
  **常驻状态与序列长度无关**。

两者数值一致（实测 fp32 下 `max|Δ| < 4e-5`）。

它修掉了 SGC 原型（v2）的三个已证实缺陷：

1. **感受野被静默截断**：v2 在 `L > max_len` 时不报错，静默退化为固定窗卷积
   （实测精确等于 16-tap 截断核）。SGC 的核由闭式/递推定义，**没有 `max_len`**。
2. **时间基秩瓶颈**：v2 的核 `K = low@up` 秩 ≤ r（默认 8）——整条长 4096 的卷积核
   只有 8 个独立时间基函数，这是它无法按内容路由的结构性原因。SGC 改用带衰减
   先验的 M 维字典（M 默认 64/128）。
3. **无流式路径**：v2 只有整段前向，"常数显存流式"无对应实现。

另加**内容门控**（`u = W_in·h ⊙ sigmoid(W_g·h)`），使有效核随内容变化。
**无位置嵌入表**，因此不受训练长度限制。

## 快速开始

```powershell
# 环境（已锁定）
.venv311\Scripts\python.exe -c "import torch;print(torch.__version__)"   # 2.4.1+cpu

# SGC 数学正确性 + 双形式等价 + 因果性 + 梯度可达（25 项）
python -m pytest simplcore/sgc/tests/test_mixer_math.py -q

# 主线因果性守护（12 项）
python -m pytest tests_simplcore_pg/test_causality.py -q

# 最小示例
python -c @"
from simplcore.sgc import get_preset, SGCModel
import torch
cfg = get_preset('debug'); m = SGCModel(cfg)
x = torch.randint(0, cfg.vocab_size, (1, 64))
logits, loss = m(x, x); loss.backward()
print('forward/backward OK', tuple(logits.shape), float(loss))
st = m.init_stream(1)
_, st = m.stream_step(x[:, 0], st)          # O(1) 状态流式
print('stream state bytes:', m.stream_bytes(1))
"@
```

## 真实语料与训练

```powershell
# 下载真实简体中文语料（走 hf-mirror 镜像）
python scripts/download_corpus.py --dataset opencsg/chinese-fineweb-edu --out data/raw/fineweb_zh

# 训练分词器 + 清洗 + 打包为定长窗口分片（文档级无泄漏切分）
python scripts/build_corpus.py --parquet-dir data/raw/fineweb_zh --out data/processed/sgc_zh `
  --vocab-size 32000 --seq-len 4096

# 先标定显存与吞吐（别凭算术猜）
python scripts/probe_batch.py --config configs/sgc-1_3b.json --seq-len 4096

# 训练（A100 上把 micro-batch/accum 换成实测值）
python -m simplcore.sgc.train --config configs/sgc-1_3b.json --data-dir data/processed/sgc_zh `
  --out-dir runs/sgc_1_3b --seq-len 4096 --micro-batch 4 --global-batch-tokens 2000000 `
  --grad-checkpoint --max-steps 100000
```

规模配置（解析计算值）：`sgc-150m` 131.4M · `sgc-350m` 348.8M ·
`sgc-1_3b` 1.329B · `sgc-3b` 3.239B，并有同宽同深的注意力对照基线。

## 当前真实能力边界

- ✅ SGC 核心：数学正确、严格因果、双形式等价、梯度可达（25 项测试）
- ✅ **SGC 混合架构**（`simplcore/sgc/hybrid.py`）：全局谱卷积 O(L log L) + 局部滑窗因果注意力 O(L·W) 交替堆叠（LongNet/Jamba 路线），`attn_every`/`attn_window` 可调；继承严格因果 + 有界流式状态 + 内容门控 + 无位置嵌入表；参数量 bit-exact（12 项守护测试）
- ✅ **SGC Hybrid 对齐当前最强注意力体系**（`attention.py` + `attention_yarn.py` + `selective.py` + `rope_scales.py`）：① GQA 分组查询滑窗注意力（`num_kv_heads` 省 KV 缓存 H/G 倍）；② GQA+YaRN 组合层（`GQASlidingWindowAttentionYaRN`，KV 按 G 头存 + NTK 位置缩放外推，Llama3 标配组合，路由自动走 GQA 布局而非 MHA）；③ 门控增强多头谱基 mixer（GLA/Mamba2 式 γ/β 双门 + 每 head 独立谱基系数组，O(1) 状态字节数不变）；④ YaRN/NTK 位置缩放外推（none/linear/yarn 三策略）；`SGCModel` 经 `mixer="hybrid"` + 新 config 字段（`num_kv_heads`/`gate_ab`/`rope_scale_strategy`/`rope_scale_factor`）自动路由；8 种组合参数量 bit-exact + 流式==整段逐位一致（31 项守护测试）
- ✅ **SGC Hybrid 无词表模式 + GQA HF 权重迁移**（`tokenization.py` + `hf_bridge.attention_gqa_warmstart`）：① 可插拔编解码器 `CharTokenizer`/`ByteTokenizer`/`BPETokenizer`（纯 Python 零第三方，BPE 标准 byte-pair + GPT-2 字节约定），`SGCConfig.from_codec` 自动填 `vocab_size`，`SGCModel.forward` 接受文本字符串（经 codec 编码），`generate_text` 端到端文本生成；同一套模型权重可切换字符/字节/BPE 序列，不重新初始化词表参数；② Llama GQA 风格 q_proj/k_proj/v_proj → GQA 合并 qkv 权重迁移（逐层搬运 + MHA→GQA 截断 + 形状错配一律 skip 不污染），`init_sgc_from_hf` 的 `cfg_override` 路径对 GQA/YaRN 配置自动触发（21 项守护测试：6 GQA 桥接 + 15 无词表）
- ✅ **长上下文外推 + 衰减消融工具**（`simplcore/sgc/extrapolation.py`）：`extrapolation_probe`（固定 L0 训练 + 1/2/4/8× 零微调评测 + O(1) 状态字节守护）+ `decay_ablation`（扫描 decay_max 曲线）；复用 eval 探针任务与可解性协议（6 项守护测试；真实质量曲线待 A100）
- ✅ 主线 `simplcore_pg` 因果性缺陷**已修复**并有 12 项守护测试
- ✅ `simplcore/mix` 三处非因果**已修复**（因果前缀统计 + 因果掩码），5 项守护测试全绿
- ✅ 训练栈可复现性守护：同 seed bit-exact、检查点 RNG 恢复、seed 生效验证（`simplcore/sgc/tests/test_reproducibility.py` 4 项）
- ✅ **Transformer/HF 互操作桥接**（`simplcore/sgc/hf_bridge.py`）：保守 1:1 搬运 embedding/FFN/norms/head + 注意力→SGC 振荡基谱投影暖启动 + HF 评测桥（伪 KV 适配），零第三方依赖；迁移后 O(L log L) / O(1) 状态流式 / 因果性**全部原样继承**（10 项守护测试）
- ✅ 真实中文语料流水线（下载 → BPE → 清洗 → 无损打包 → 文档级切分）
- ✅ 可复现训练栈（bf16/TF32、梯度累积、warmup+cosine、断点续训、JSONL 指标）
- ✅ 显存/吞吐标定脚本（实测分配器计数，而非 RSS 猜测）
- ⚠️ **1B 以上规模从未实际训练过**：配置与链路已打通，A100 曲线待测
- ⚠️ 未与 S4 / Mamba / Hyena / GLA 在 LRA 等公认基准对比
- ⚠️ 长上下文外推只有"输出有限"的证据，没有质量证据
- ❌ 不得据此宣称任何 SOTA

详见 `docs/HONEST_STATUS.md` 与 `docs/SGC_SPEC.md` 第 8 节。
