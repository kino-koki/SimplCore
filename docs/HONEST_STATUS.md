# 诚实状态总表（SimplCore-V 0.1-ALPHA）

2026-08-24 整理。本文档是项目真实状态的"单一事实来源"，替代此前十余份互相矛盾的
评估报告/审计/证明文档（已归档至 `.archive/`）。

---

## 零、2026-10 重大更正（独立性审计发现，**优先于以下一切内容**）

一次系统性的独立性审计（真实执行、非阅读性审查）发现了三个**足以推翻既有结论**的缺陷。
凡与之冲突的历史结论一律作废，本节的优先级高于本文档其余全部内容。

> 审计全文与全部原始日志（40 个文件、含逐条命令输出）已归档至
> [`docs/audits/2026-10-sgc-audit/AUDIT_REPORT.md`](audits/2026-10-sgc-audit/AUDIT_REPORT.md)。
> 本节所有"实测"结论均可在该目录的对应日志中找到原始输出。

### 0.1 〔致命·已修复〕主线 RFP 是非因果算子 → 全部语言模型结论作废

`simplcore_pg/resonance.py` 的旧实现用 `rfft → 乘实值带通窗 → irfft` 做"谱滤波"。
**一个实值（零相位）频率响应对应的冲激响应必然关于 t=0 偶对称，因此该算子必然看到未来。**
实测：只扰动最后一个 token，前 511/511 个位置的输出全部改变（`max|Δ| = 2.74`）；
关闭 RFP（`use_rfp=False`）后同一测试的泄漏恰好为 `0.0` —— 即 RFP 是唯一成因。

**后果**：用旧 RFP 训练的模型，其交叉熵是被"偷看答案"压低的。因此本文档
"一、已实现并验证"表中以下条目**作为语言建模证据全部作废**，必须用修复后的因果 RFP 重跑：
`语言建模训练闭环可学习`、`真实中文语料困惑度对比（val_ppl 1.28 vs 2.05）`、
`组件消融（bench_ablation）`、`规模扩展（bench_scaling）`、`长序列精度对比`、
`流式推理正确性`（其一致性验证的是同一个非因果算子）。

**修复**：`band_to_causal_kernel()` 用经典窗函数法把目标幅频响应转成因果 FIR
（逆 FFT → 取因果半边 → 补回增益），再用 2L 零填充 FFT 做严格因果卷积；
`ResonanceField(causal=True)` 为默认。新增 `tests_simplcore_pg/test_causality.py`（12 项）
守护：只扰动未来、断言过去逐位不变。**修复后 `tests_simplcore_pg` 全绿（134 项）。**
旧的零相位行为仍可用 `causal=False` 复现，仅供对照，禁止用于训练或生成。

### 0.2 〔致命·已修复〕`simplcore/mix` 有三处非因果且并非 PyTorch

`simplcore/mix` 被 README 称为"唯一活跃模型基座"，但实际是**自研 NumPy 自动微分引擎**
（`import simplcore.tensor as torch`；全包 `import torch` 匹配数为 0），AMP 被硬编码关闭。
三处**序列维全局归约**破坏因果性（均以"只扰动最后一个 token"实测定位）：
`cin.py` 的 `softmax(q@kᵀ)` 无因果掩码（训练与推理均生效，默认开启；
关闭后泄漏由 `1.83e-3` 降到 `0.0`）、`memory.py` 的 `x.mean(dim=1)` 全序列均值、
`blocks.py` 的 Hebbian 全局均值。

**修复（2026-11）**：三处全部改为因果前缀统计（cumsum/cumcount 的前缀和形式 +
有限大负数因果掩码，与 DCGD triu_mask 同策略）。新增
[`tests_simplcore_mix/test_mix_causality.py`](../tests_simplcore_mix/test_mix_causality.py)
（5 项）守护：组件级"扰动末位 token 断言前 L-1 行不变" + 端到端 Mix 因果性 +
KV 缓存生成确定性。**修复后该路径已可用于自回归 LM 训练**，但仍是 NumPy 引擎、
AMP 关闭，性能与可训练性边界以 `simplcore/mix/README` 为准。

> 注意：`mix` 的非因果修复**不改变**它"非 PyTorch、AMP 关闭"的定位；
> 主线仍为 `simplcore/sgc/`（PyTorch、因果、双形式对偶）。

### 0.3 〔严重〕其他已证实缺陷（按影响排序，详见各文件）

| 缺陷 | 证据 | 状态 |
|------|------|------|
| `verify_sgc` 报告的"真实中文语料"实为 83% 仓库自身日志与 markdown | `build_corpus.py` 采集 24 文件 115 KB，其中仅 12.7 KB 是语料，~62 KB 是构建日志 | 已改用 FineWeb-Edu-Chinese 真实语料 |
| `verify_sgc` 的显存实测全部为 0 | `GetProcessMemoryInfo` 在本机返回 0（GetLastError 122），`stream_infer.py`/`_memcap.py` 打印 "峰值RSS 0.0MB"，`memcap.json` 的 attn_delta 全为 0 | 报告 §11.4 的"O(1) 显存实测"无有效测量 |
| 报告 §11.2/§11.3 的机理探针数字无 JSON 支撑 | 数值只作为字符串字面量存在于 `_append_hm_docx.py:74,89,90,92` | 已重跑复现，但须补 JSON |
| `GatedAssocMixer` 非因果且非确定 | `sgc_v3.py:190` 每次前向用 `torch.randn` 重置状态 | 已弃用，由 SGC 取代 |
| `simplcore_pg` 无任何随机种子 | 全包 `manual_seed` 匹配数为 0，CLI 无 `--seed`；连续两次构造权重不同 | 待修 |
| `simplcore_pg` 打包器静默丢 token | `training/data.py:212-215` 截断超长残余缓冲，实测每遍丢 602/195,226 token | 待修 |
| `simplcore_pg` 训练检查点无法被自己的加载器读取 | `train.py:178-182` 写 `torch.save({step,config,state})`；`checkpointing.py` 要求 `config.json`+`model.bin` 目录 | 待修 |
| 两条路径都**没有因果性测试** | 122 + 53 项测试全绿却漏掉上述致命缺陷 | 已补 `test_causality.py` |

### 0.4 〔新增〕SGC —— 重建后的主线架构

针对"位置相对核无法按内容路由"与上述工程缺陷，`simplcore/sgc/` 提供重写的正式实现：
阻尼振荡基字典（对偶：训练走 FFT `O(L log n)`，推理走二阶 IIR **O(1) 状态**）、
内容门控（selective）、有界局部深度卷积，**无位置嵌入表、无 `max_len` 截断**、
严格因果（`test_mixer_math.py` 25 项全绿）。
详见 `docs/SGC_SPEC.md`。

### 0.5 〔严重〕SGC v2 报告中最具分量的负面证据**没有来源**

`simplcore/docs/SGC_v2_诚实版报告.docx` 的 §3.4 / §4.1 / §5 / §6 反复引用
"独立测试集 test_acc：SGC 2.67% / Gated-SGC 1.50% / Attention 12.89%"，
并注引 `simplcore/docs/SGC_v2_门控消融_诚实报告.md`。全仓库检索结果：

* 该 markdown **不存在**；
* 字符串 `2.67%` / `12.89%` / `1.50%` 在**任何**文件中都不出现。

即该组数字**无任何可复现来源**。它是"SGC 在更严格独立测试协议下崩坏至随机水平"
这一论断的唯一依据，而 §3.4、§4.1、§5、§6 都建立其上。
**处置**：已在报告追加"十二、勘误（2026-10 独立性审计）"，撤回该组数字，
并逐条列出与原始 JSON 不符的引用（`gated_compare.json` 实为 `steps=5`、且缺
`attn`/`sgc_nogate` 两条对照臂）、§9.2 端到端效率主张不成立（ca 991 ms/step vs
attn 499 ms/step）、以及经审计确认可信、予以保留的结论清单。

**结论**：撤回后，SGC v2 报告**不再包含任何"SGC 优于或持平注意力"的正面泛化主张**，
应定位为内部技术备忘，不构成可发表的实证主张。

### 0.6 〔已修复〕本轮修复清单

| 缺陷 | 修复 | 守护测试 |
|------|------|----------|
| RFP 非因果（主线致命） | 窗函数法构造因果 FIR + 2L 零填充因果卷积，`causal=True` 默认 | `tests_simplcore_pg/test_causality.py`（12 项） |
| `simplcore/mix` 三处非因果（cin 自注意力无掩码 + memory/Hebbian 全局均值） | 因果前缀统计（cumsum/cumcount 前缀和 + 有限大负数因果掩码）；gate/decay 改为逐位置 `[B,T,C]` | `tests_simplcore_mix/test_mix_causality.py`（5 项：组件级 + 端到端 + 生成确定性） |
| SGC 与 Transformer/HF 生态无互通路径 | 新增 `simplcore/sgc/hf_bridge.py`：保守 1:1 搬运（embedding/FFN/norms/head，形状错配一律跳过）+ 注意力→SGC 振荡基谱投影暖启动 + HF 评测桥（`sgc_to_hf_eval_bridge` / 伪 KV 适配）+ CLI；零第三方依赖（不 import transformers） | `simplcore/sgc/tests/test_hf_bridge.py`（10 项：1:1 搬运、错配跳过不污染、因果继承、流式状态不变、暖启动、评测桥、伪 KV、报告审计） |
| **SGC 混合架构缺失**：谱全局卷积单独无法表达尖锐的局部 token 级路由（检索/指代消解） | 新增 `simplcore/sgc/hybrid.py`：全局 SGC（O(L log L)）+ 局部滑窗因果注意力（O(L·W)）交替堆叠（LongNet/Jamba 路线），`attn_every` 控制密度、`attn_window` 控制滑窗；继承严格因果 + 有界流式状态（SGC 层 O(1) + 滑窗 O(W·H·D)，与 L 无关）+ 内容门控 + 无位置嵌入表；`SGCModel` 自动路由，参数量 bit-exact 解析 | `simplcore/sgc/tests/test_hybrid.py`（12 项：布局、因果继承、流式==整段逐位、状态长度无关、参数量 bit-exact、滑窗外不可见/窗口内因果、窗口≥L 退化为全因果注意力、生成路径） |
| **SGC Hybrid 对齐当前最强注意力体系**：固定 RoPE 无长程外推保证、滑窗注意力无 GQA 省 KV、SGC 层内容门控未达 GLA/Mamba2 双门水平、GQA 与 YaRN 不能同时启用 | 新增 4 个模块 + 接入 `SGCModel`（`mixer="hybrid"` 新字段 `num_kv_heads`/`gate_ab`/`rope_scale_strategy`/`rope_scale_factor`）：① `attention.py` GQA 分组查询滑窗注意力（KV 缓存按 kv 头存，省 H/G 倍）；② `attention_yarn.py` GQA+YaRN 组合层（`GQASlidingWindowAttentionYaRN`，KV 按 G 头存 + NTK 位置缩放外推，Llama3 标配组合）；③ `selective.py` 门控增强多头谱基 mixer（GLA/Mamba2 式 γ/β 双门 + 每 head 独立谱基系数组，O(1) 状态字节数与单头相同）；④ `rope_scales.py` YaRN/NTK 位置缩放外推（none/linear/yarn 三策略，流式==整段逐位一致）；路由规则升级：YaRN/linear + GQA 组合自动走 `GQASlidingWindowAttentionYaRN`（而非 MHA 布局）；参数量 bit-exact 解析覆盖 8 种组合 | `simplcore/sgc/tests/test_attention.py`（8 项）+ `test_attention_yarn.py`（10 项：GQA+YaRN 构建/流式==整段/stream_bytes 按 G 头/bit-exact 参数/none 策略==固定 GQA/MHA 退化==MHA YaRN/因果性/preset 端到端/HF 桥接）+ `test_selective.py`（6 项）+ `test_rope_scales.py`（7 项）+ `test_integration_2_1.py`（4 项集成：8 种组合构建/前向/流式/参数量 bit-exact/GQA 省 KV） |
| **SGC Hybrid Transformer 兼容互操作缺口**：GQA/YaRN 注意力层无 HF 权重迁移路径；编解码器烧死在词表里（传统 Transformer tokenize 词表不可换）；O(1) 流式状态无 checkpoint/restore 持久化路径 | ① `hf_bridge.py` 新增 `attention_gqa_warmstart`：Llama 风格 q_proj/k_proj/v_proj → GQA 合并 qkv（逐层搬运，MHA→GQA 截断 + note 记录，形状错配一律 skip 不污染）；`init_sgc_from_hf` 增 `cfg_override` 参数，GQA/YaRN 配置自动触发 GQA 暖启动。② `tokenization.py` 无词表模式：`CharTokenizer`/`ByteTokenizer`/`BPETokenizer` 可插拔编解码器（纯 Python 零第三方，BPE 标准 byte-pair 算法 + GPT-2 字节约定）+ `SGCConfig.from_codec` 工厂 + `SGCModel.forward` 文本输入 + `generate_text` 端到端文本生成；同一套模型权重可切换字符/字节/BPE，不重新初始化词表参数。③ `SGCModel.checkpoint_stream` / `restore_stream`：O(1) 流式状态（SGC IIR 二阶递推 + 滑窗注意力环形 KV）序列化为 plain dict，可 `torch.save` 持久化"断点流"，restore 后继续 `stream_step` 逐位一致 | `simplcore/sgc/tests/test_hf_bridge_gqa.py`（6 项：MHA→MHA 逐位、MHA→GQA 截断、GQA→GQA 逐位、形状错配 skip、YaRN 触发、端到端）+ `test_tokenization.py`（15 项：Char/Byte/BPE roundtrip + BPE 字节无损 + from_codec 自动 vocab_size + forward 文本==ID + generate_text 端到端 + rebuild_codec）+ `test_stream_checkpoint.py`（5 项：混合/GQA/纯 SGC 流式状态 checkpoint/restore + torch.save 持久化往返 + GQA 环形 KV 字典序列化） |
| 长上下文外推 / 衰减消融**只有工具没有证据**（SGC_SPEC §9.4/§9.5） | 新增 `simplcore/sgc/extrapolation.py`：`extrapolation_probe`（固定 L0 训练 + 1/2/4/8× 零微调评测 + O(1) 状态字节守护）+ `decay_ablation`（扫描 decay_max → 困惑度/外推曲线）；复用 `eval.py` 的探针任务与可解性协议，纯 torch 零第三方 | `simplcore/sgc/tests/test_extrapolation.py`（6 项：任务形状、未知任务报错、stream_bytes 跨倍率恒定、结果字段、衰减消融管线、JSON 可序列化） |
| `simplcore_pg` 无随机种子 | 新增 `simplcore_pg/seeding.py` + `--seed`/`--deterministic` | `test_reproducibility.py`（5 项） |
| 打包器静默丢 token | 去掉 `buf = buf[-min_seq:]` 截断，残余整段带入下一批 | `test_packed_dataset_does_not_drop_tokens` |
| 去重不可复现（`hash()` 加盐） | 改用 `blake2b` 稳定摘要 | `test_dedup_is_stable_across_processes` |
| `GatedAssocMixer` 非确定（伪装成非因果） | 固定 generator 播种初值 | 实测确定性/因果性均为 0.0 |
| §11.5 实验脚本对全部位置计损失 → 恒为 chance | 只对查询位置计损失 | 已重跑，见 §0.7 |
| 显存测量工具失效（恒 0） | SGC 评测改用张量记账 + `cuda.max_memory_allocated()`，并自证（`self-test`） | `test_eval.py` |

### 0.7 §11.5 实验：已实跑完成，脚本有 bug（已修），结果**推翻报告的 SGC-HM 主张**

**第一步（审计，未修脚本）**：把 §11.5 的三行跑到终点（G=512、700 步）：
`hm` 905.0s / `ca` 535.2s / `attn` 862.3s，loss 2.787 / 2.782 / 2.780，
**全部高于 ln(16)=2.773**，recall ≈ 0（chance 0.0625）。

根因不是 mixer 而是脚本：`_train_lingap.py` 对全部 L 个位置计交叉熵，
而序列结构 `[Q, v, 干扰×G, Q]` 中除最后一位外，其余位置的"下一个 token"是
与历史无关的随机干扰，在 16 类上不可学。514/515 个不可学位置把损失顶在
ln(16) 之上——**连能精确检索任意历史 token 的注意力也同样失败**。
这些"失败"数据已存档于 `verify_sgc/results/_broken_loss_2026_10/`。

**第二步（修复脚本后重跑）**：只对查询位置计损失
（`F.cross_entropy(logits[:, -1, :], tv)`），700 步，三种 mixer：

| mixer | G=64 | G=128 | G=512 | 参数量 |
|-------|------|-------|-------|--------|
| `attn`（因果注意力） | **1.000** | **1.000** | **1.000** | 422,210 |
| `ca`（内容自适应线性混合） | **1.000** | **1.000** | **1.000** | 428,352 |
| `hm`（SGC-HM 覆盖写记忆） | 1.000 | **0.000** | **0.188** | 440,642 |

（chance = 0.0625。修复后 G=64 三者的训练 loss 均为 0.000，任务确实可学。）

**结论 —— 与报告 §11.1/§11.6 相反**：报告称 SGC-HM 的 delta 覆盖写
"不随上下文线性衰退"，而 CA 会因叠加稀释而衰退。端到端实测**恰好相反**：
CA 与注意力在 G 到 512 时均保持精确召回（1.000），
**SGC-HM 在 G≥128 就崩到 chance 附近**。

这其实与报告自己的 §11.2 面板 A 一致：该面板实测"低维随机键下 argmax 路由
P(hit) 随干扰数 T 跌到随机水平（T=512 时 0.022）"。§11.3 面板 B 之所以显示
覆盖写平坦，是因为它**假设内容已分离、信号与干扰各自路由到专用槽**；
§11.2 已指出该前提在未训练/低维下不成立。端到端训练证实：
**面板 A 的警告成立，面板 B 的前提不成立**，因此
"SGC-HM 不随上下文衰退"这一主张在该任务上**不成立**。

处置：`verify_sgc/sgc_v3.py` 的 `GatedAssocMixer` 非确定性缺陷已修
（改用固定 generator 播种；实测确定性/因果性均从"失败"变为 0.0）；
SGC-HM 的定位应从"已解决的记忆方案"降级为"机理探针中有趣、端到端未兑现"。
SGC（`simplcore/sgc`）不采用 HM 的 argmax 覆盖写路线。

---


## 一、已实现并验证（真实）

| 能力 | 证据 | 验证方式 |
|------|------|---------|
| 无注意力架构 构造/前向/反向 | `smoke_test.py` + `get_attention_free_config` | CPU torch 实跑，源码无 Q/K/V/RoPE/SDPA |
| 长上下文流式 API (`stream_init`/`stream_step`/`generate_stream`) | `smoke_test.py` 用例 2 | 分块喂入 + 生成实跑 |
| **ResonanceField 可训练**（omega/sigma/beta/out_scale 参与反向） | `smoke_test.py` 用例 3 | `in_proj/out_proj.weight.grad > 0`，多步演化经 checkpoint |
| DynamicMemoryChannels 梯度可达 | `smoke_test.py` 用例 4 | `memory_bank.grad > 0` |
| RFP vs 注意力 长序列延迟优势 | `bench_rfp_vs_attention.json` | n=2048 时 RFP 5.2ms vs 注意力 41.3ms（8.0×），n=4096 时 16.2×，交叉点 n≈400，差距随 n 单调扩大 |
| 长上下文激活显存常数增长 | `longctx_memory_benchmark.json` | 1M/2M/4M 增量 0 |
| **语言建模训练闭环可学习** | `train_smoke.log` | 30 步 loss 4.93→4.63（RFP+MoE 自研链路） |
| **长序列精度对比（RFP vs 注意力）** | `bench_longctx_accuracy.json` | 长距离检索任务两者均 acc=1.0，RFP 训练损失更低 |
| **全局可达性消融（局部递归 vs RFP 谱全局）** | `bench_multihop.json` + `.tmp_multihop_exp.json` | 全局路由复制任务：local_only 门控递归本身全局可达，500–1500 步可达 acc≈0.73–0.76；RFP 带限谱卷积对尖锐长距复制不占优（acc≈0.24–0.27）。RFP 优势在效率/流式而非尖锐路由 |
| **流式推理正确性** | `test_stream_consistency.py` | 分块流式 logits 与整段前向 max_err=2.4e-7 |
| **真实中文语料困惑度对比（SimplCore vs Transformer）** | `bench_corpus_ppl.json` | 文档级留出、参数对齐（1.250M vs 1.235M）：SimplCore val_ppl=1.28 vs Baseline 2.05（越低越好） |
| **组件消融（真实语料，逐组件归因）** | `bench_ablation.json` | 同一数据/协议下 5 变体：full=1.2374 最优；去卷积 +6.6%（局部主力）、去 RFP +0.7%、去 DMC +0.3%、去递归 +0.2%，各组件互补 |
| **规模扩展（真实语料，困惑度随规模下降）** | `bench_scaling.json` | 匹配 200 步预算：H=64→1.2637 / H=96→1.2385 / H=128→1.2599（最大模型欠训练）；H=128 延至 400 步收敛到 1.2294（低于 H=96）。全部规模点以 `rfp_freeze_evolution=True` 训练（Windows/CPU FFT 反向泄漏的文档化回退，冻结与可训练 RFP 的 val_ppl 几乎一致） |
| **安全模块（AES-256-GCM / RBAC / 内存金丝雀保护）** | `safety/crypto.py` + `safety/rbac.py` + `safety/memory_guard.py` + `safety/tests/` | 权重加密/解密往返、权限 deny-by-default、越界写金丝雀检测，全部单测通过 |
| **安全模块接入权重主链路（save_pretrained/from_pretrained）** | `simplcore_pg/checkpointing.py` + `tests_simplcore_pg/test_checkpointing.py` | 明文/加密往返、口令错误拒绝、篡改指纹检测、RBAC 未授权拒绝、解密明文用后即焚（GuardedBuffer 金丝雀），10/10 单测通过 |
| **FX 图模式追踪（全模型）** | `tests_simplcore_pg/test_model.py::test_fx_trace_and_optimize` | GatedLinearRecurrence 改为并行扫描（log 空间前缀积），ResonanceField 显式传 L，全模型可 `symbolic_trace` + 推理图优化 pass |
| **原生多模态模型与感知前端 完全无注意力** | `native_mm.py` + `multimodal.py` + `perception.py` | ViT 块改为 LocalInteraction(局部)+ResonanceField(全局)；视频时序混合与 Perceiver 重采样 (原 cross/self-attn) 改为 LocalInteraction + 跨模态加性汇聚；generate/增量推理统一走固定大小 StreamState，**无任何 Q/K/V/RoPE/SDPA/KV Cache** |
| **集成自测 (MoE / 流式常数显存 / 原生多模态生成)** | `test_integration_smoke.py` | 3/3 通过；流式峰值显存 ratios=[1.0,1.0,1.0] |
| **复杂度阶数验证 (RFP·Local 线性 vs FullAttn 二次)** | `bench_throughput_result.json` | n 增 4× 时 RFP×5.5 / Local×4.9 / FullAttn×11.6（趋近理论线性/二次） |
| torch.compile 安全回退 | `smoke_test.py` 用例 7 | 平台不支持时回退 eager |

---

## 二、已纠正的失真（历史问题 → 现状）

| 历史失真表述 | 真实情况 | 处置 |
|------------|---------|------|
| "已学会真实物理规律，泛化优秀" | spring 相对误差 12.2%、gravity 34.5%、泛化 RMSE 0.45，均 FAIL | `final_physics_proof.json` 已如实改写；用例 6 守护 |
| "零 PyTorch 依赖的纯原生引擎" | 主线 `simplcore_pg` 依赖 torch；仅 `simplcore/tensor` 子模块是 NumPy 引擎 | 文档删除该断言 |
| "流式 API 已可用（`stream_init`/`stream_step`）" | 此前 `SimplCoreStreamingModel` 无此方法 | **已真实实现**（2026-08-24） |
| "Java 为主线、GPU 为 CUDA 内核" | 主线是 Python；Java GPU 实为 OpenCL | 架构文档 v1.4 修正 |
| "512× 理论加速比" | 无对应实测 | 删除；改由 `bench_rfp_vs_attention.py` 真实数据支撑 |
| 演示把 SKIP（含真实断链）当成功报"完全可用" | Cognition 等真实导入失败 | `demo_simplcore_availability.py` 改为 FAIL 才算失败 |
| "架构依赖自注意力/Transformer" | v2.0 前 block 含 SelfAttention(RoPE+SDPA) | **已彻底移除**：自 v2.0 起所有 block = LocalInteraction→RFP→DMC→MoE，无 Q/K/V/RoPE/SDPA；原生多模态 ViT 与推理核亦改为 LocalInteraction+RFP，增量生成走固定 StreamState（无 KV Cache）。`attention.py` 仅保留为基准对比模块（被 `bench_throughput.py` 用作复杂度基准），已移出公开 API；`v2/` 子包为独立实验性 SWA 变体，非本架构组成部分 |

---

## 三、未验证 / 已知限制（不掩饰）

1. **GPU 训练与显存实测**：本机 AMD 680M + Windows 不支持；待 ROCm/Linux。
2. **CUDA 内核编译加载**：`.cu` 源码未编译验证。
3. **物理规律学习**：实测未通过，不能宣称"学会"。
4. **安全模块（AES-256/RBAC/内存保护）**：~~仅文档设计，未在主线落地~~ → **已落地**
   （`safety/crypto.py` / `safety/rbac.py` / `safety/memory_guard.py` + 单测通过）。
   ~~尚未接入模型权重保存/加载主链路~~ → **已打通**（`simplcore_pg/checkpointing.py`：
   `save_pretrained` / `from_pretrained` 支持 AES-256-GCM 加密、RBAC 鉴权、
   解密明文用后即焚、SHA-256 指纹防篡改，10/10 单测通过）。
5. **56B 全量训练**：CPU 无法实例化，仅小尺度验证组件正确性。
6. **RFP 在 Windows/CPU 的反向算子内存泄漏（已定位 + 已缓解）**：
   `torch.fft` 的反向 workspace 在 Windows/CPU 的 torch 构建下会累积不释放，
   纯前向 200 步不崩，但 `.backward()` 在长循环里会被 OS 静默杀死。
   **缓解（默认开启）**：RFP 多步演化经 `torch.utils.checkpoint` 包裹，反向时仅
   保留每步输入并重算前向，避免反向 workspace 在长循环里持续累积；CPU 下训练步数
   受控（~80–150 步 / batch≤8）可稳定跑完。若环境仍触发，可设
   `rfp_freeze_evolution=True` 回退为"仅投影可学、FFT 走 no_grad"的固定全局混合器
   （与 S4 类将核心卷积设为近似/不可导的惯例一致）。DMC 不受影响。
7. **多跳任务的规模**：当前 `bench_multihop.py` 为小规模可复现证据。诚实结论：
   LocalInteraction 中的门控线性递归本身即全局可达算子，在"全局路由复制"任务上
   不弱于 RFP；RFP 作为带限谱全局卷积，擅长平滑/频域全局结构，但对尖锐长距内容
   路由并不占优。RFP 的真实优势在**效率**（O(n log n)、n=2048 快 3.0×）与
   **常数显存流式**，已分别由 `bench_rfp_vs_attention.py` / `bench_longctx_memory` 实证。
   尚未在大规模语料上做 SOTA 对比。
8. **Java 并行实现可复现性**：不在本文档保证范围。

---

## 四、下一步真实开发优先级

1. ROCm/WSL2 环境下实跑 `longctx_memory_benchmark.json` 的 GPU 显存二次确认；
2. ~~长序列精度对比~~ / ~~多跳区分度~~ → **已完成**（见 `bench_longctx_accuracy.json`、
   `bench_multihop.json` + `.tmp_multihop_exp.json`，RFP 全局优势在小规模合成任务上已实证）；
3. ~~安全模块落地 + 测试~~ → **已完成**（`safety/` 三模块 + 单测）；
4. ~~真实语料 perplexity 对比~~ → **已完成**（`bench_corpus_ppl.json`：文档级留出、
   参数对齐，SimplCore val_ppl=1.28 vs Baseline 2.05）；
5. ~~安全模块接入模型权重保存/加载主链路~~ → **已完成**（`simplcore_pg/checkpointing.py`：
   `save_pretrained` / `from_pretrained` 打通 AES-256-GCM / RBAC / 内存金丝雀 / 指纹防篡改，
   10/10 单测通过）；
6. 持续运行 `smoke_test.py` 守护通过率 ≥90%；
7. ~~更大规模多跳任务 + 更大语料 perplexity 对比~~ → **规模扩展已完成**（`bench_scaling.json`：
   H=64/96/128 三规模点困惑度随规模下降，H=128 延至 400 步收敛到 1.2294 低于 H=96；
   因 Windows/CPU FFT 反向泄漏，规模点以 `rfp_freeze_evolution=True` 训练，冻结与可训练
   RFP 的 val_ppl 几乎一致）。更大语料 SOTA 对比仍待 GPU/ROCm。
