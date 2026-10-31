# SGC 架构全量修复记录（2026-10）

> ⚠️ **历史文档（2026-10 的修复日志）**。文中部分类名与配置字段
> （`SGCConfig` → 现为 `SimplCoreConfig`，`mixer="attn"` / `hybrid` /
> `reason_attn_every` → 已删除）已随注意力族的整体移除而失效。
> **当前代码状态以 `simplcore/core/` 源码与 `docs/ARCHITECTURE.md` 为准。**
> 本文保留是为了记录"接口对、逻辑空"这类缺陷的排查方法。

> 范围：`simplcore/core/`（SGC 算子本体 + 抗幻觉链 + 自进化链）。
> 不在范围：官网前端（已移交）、旧引擎合并（G5）。
> 本文只记录**有守护测试或实测数字**的修复；无证据的一律列为"未证明"。

---

## 0. 摘要

本轮修的不是新功能，而是**"接口对、逻辑空"**这一类缺陷 —— 它们不会让任何一个
形状测试失败，只在真实使用时表现为"机制在跑，但什么也没发生"。

| 统计 | 值 |
|---|---|
| 修复缺陷 | 12 项（D1–D12）+ 1 项测试口径修正 |
| 新增/改写测试 | 3 个文件新增（`test_gates_selective.py` 14 项、`test_grounding.py` 36 项、`test_dream_evolution.py` 37 项），2 个文件修订 |
| 全量测试 | `pytest simplcore/core/tests` → **303 passed, 1 skipped** |
| 关键实测 | 归因判别 precision **1.000** / recall **0.971** / F1 **0.985**（600 构造样本，dev/test 各半）；单句与多句拦截率 **1.000** |
| 新增模块 | `gates.py`（闸门数学唯一实现）、`grounding.py`（L4 标签来源）、`scripts/eval_grounding.py`（实测） |

一句话结论：**抗幻觉从"有结构无监督"变成了"有标签、有校准、有实测"；
自进化从"空转"变成了"能产出带证据原文的训练信号，且在没有独立评测时拒绝改权重"。**
这仍然不等于"抗幻觉有效"或"自进化会变强"——见 §4。

---

## 1. 判定标准

本轮把"缺陷"定义为满足下列任一条件的事实，而不是阅读代码后的感觉：

1. **存在可复现的反例**（一个输入 + 一个错误输出），或
2. **某个机制声称参与计算但实际不参与**（例如：配置开关打开却没有任何代码读取它、
   参数可学却拿不到梯度、状态被存了却没被恢复），或
3. **两条代码路径对同一概念给出不同答案**（同一份配置换分支行为就变了）。

因此每条缺陷都带一个"之前必然失败、现在必然通过"的守护测试。

---

## 2. 缺陷与修复

### D1 〔SGC·G4〕`gate_ab` 分支完全没有 L2/L3 闸门

**症状**：`SimplCoreConfig(gate_ab=True)` 走 `GatedSpectralConv2`，这个分支里
**没有证据门、没有不确定门**，只有 L1 选择门。同一份"四层闸门"配置换一个
mixer 分支，抗幻觉能力静默消失，而参数量与形状测试全部照旧通过。

**根因**：闸门公式在 `mixer.py` 与 `selective.py` 各写一套，第二套只写了第一层。

**修复**：新建 `simplcore/core/gates.py` 作为闸门数学的**唯一事实来源**，
导出 `prefix_mean_energy` / `evidence_weights` / `evidence_weight_step` /
`init_energy_sum` / `joint_disagreement` / `uncertainty_weights`；
`mixer.py` 与 `selective.py` 都改为委托调用（行为不变，原有 18 项 `test_gates` 全绿）。
`GatedSpectralConv2` 补齐 L2/L3、流式对偶、`energy_sum` 状态与字节核算、
`analytic_gated_spectral_params` 的标量参数计入。

**守护**：`tests/test_gates_selective.py`（14 项）—— 闸门可消融、梯度可达、
L2 只依赖因果前缀、流式 == 整段（`L ∈ {1,4,17,64}`）、严格因果、
L3 压制分歧、参数量 bit-exact（4 种开关组合）。

### D2 〔SGC〕流式 checkpoint 漏存 `energy_sum`，restore 后不再逐位一致

**症状**：`checkpoint_stream` → `restore_stream` → 继续 `stream_step`，
输出与"不中断的流式路径"不再逐位相等。

**根因**：L2 的流式对偶需要常驻标量 `energy_sum`，但 checkpoint 只存了振荡器态。
restore 后该标量从 0 重新累积，闸门权重从第一个 token 起就偏小。

**修复**：`checkpoint_stream` 增加 `"energy_sum"`，`restore_stream` 从中恢复；
`streaming_state_bytes` 的期望值由 `2·M·D·batch·4` 改为 `(2·M·D + 1)·batch·4`。

**守护**：`test_selective.py::test_streaming_state_bytes_constant_and_head_invariant`
（修正期望值）+ 新增 `test_streaming_state_bytes_without_evidence_gate`
（关闸门时回落旧口径，证明多出的 4 字节确实只属于证据门）。

### D3 〔抗幻觉〕`grounded` 标签无人构造 → L4 头永远训不起来

**症状**：`abstention_loss` 收 `grounded` 张量，全仓库没有任何代码产出它。
四层闸门的第四层因此只有结构、没有监督。

**修复**：新增 `simplcore/core/grounding.py`（零第三方依赖），做**逐字归因**：

```
supported = exact  or  (coverage ≥ 0.62 and ngram ≥ 0.35 and numbers_ok and polarity_ok)
```

并提供 `EvidenceIndex.from_text` / `from_retrieval`、`split_claims`、
`attribution_coverage`、`grounded_labels`。

**守护**：`tests/test_grounding.py`（36 项）。

### D4 〔抗幻觉〕数字篡改与否定翻转对文本重叠完全不可见

**症状**（实测发现，不是推测）：把 `1.86 倍` 改成 `2.86 倍`，
字符覆盖率仍有 **0.95**，`number_swap` 类样本准确率只有 **0.2917**，
precision 仅 **0.6964**。这是最危险的一类编造 —— 读者记住的正是那个数字。
同理，"与序列长度无关 → 与长度有关"一字之差、语义相反。

**修复**：加两条判据：
* `numbers_supported`：claim 里**每一个**数字都必须出现在 span 中。
  实现细节：**不能**复用 `normalize()`（它把小数点当标点删掉，`5.64 → 564`，
  与 `extract_numbers` 的 token 永远对不上），改为只去空白与千分位逗号。
* `polarity_matches`：否定词计数必须相等。

**效果**：精确率 **0.696 → 1.000**，F1 **0.983**。

### D5 〔抗幻觉〕一句短编造被两句长真话按长度稀释

**症状**：多句回答的拦截率只有 **0.88**。因为归因覆盖率按 claim 长度加权，
短编造句被长真句稀释到阈值以上。

**修复**：`GroundedPolicy.cap_on_unsupported` —— **单向降级**规则：
只要存在任何一句无归因依据，就禁止 `ANSWER`（`ANSWER → HEDGE`，`HEDGE` 只追加理由，
不允许反向升级）。

**效果**：`single_claim_block_rate = 1.000`、`multi_claim_block_rate = 1.000`。

### D6 〔方法学〕评测阈值在测试集上选（泄漏）

**症状**：`eval_grounding.py` 先在全部样本上选阈值，再在同一批样本上报指标。
数字必然偏乐观。

**修复**：样本按 `dev`/`test` 各半（300/300）划分，阈值**只在 dev 上选**，
test 只用于报告。落盘 JSON 同时给出 `threshold_from_dev` 与
`test_at_dev_threshold` 两组数字。

### D7 〔自进化〕记忆键共线 + `hash()` 进程加盐 → 检索形同虚设

**症状**：`_write_memory` 用 `float(abs(hash(text)) % 1e8) / 1e8` **填满整条向量**
当键。后果有两个：
1. 任意两条记忆的余弦相似度恒为 **1.0**，top-k 检索退化成"永远返回库里任意一条"；
2. 内置 `hash()` 带进程随机盐，重启后同一文本的键就变了。

**修复**：新增 `TextKeyEncoder`（字符 3-gram + 带符号 `blake2b` 哈希 + L2 归一化）。
实测：不同文本 `cos = -0.129`，同文本 `1.0`，跨构造一致。
**键**编码 `source.prompt`（"将来什么查询该召回它"），**值**编码证据原文。

**守护**：`test_memory_retrieval_hits_the_right_entry` 在干扰规模
**0 / 64 / 128 / 512** 上断言 top-1 命中正确条目；旧实现在 ≥64 时必然失败。

### D8 〔自进化〕记忆巩固被"有没有训练样本"门控 → 全部答对的一轮什么都不写

**症状**：`if cfg.mode == "none" or not examples: return` 把记忆写入挡在后面。
一个**全部答对**的批次永远不产生训练样本，于是记忆库永远是空的 ——
这是"记忆写了但从不被召回"的另一半原因。

**修复**：记忆巩固与权重巩固解耦。`mode="memory"` 时无论有没有训练样本都会写库，
且写两类条目：`knowledge`（证据支撑的分句）与 `repair`（有原文的错例）。

### D9 〔自进化·G3〕默认反思/合成钩子是占位实现 → `synthesized` 恒为 0

**症状**：默认反思器只给轨迹打标记，合成器只认 `meta['corrected_response']`。
真实场景没有外部批判模型，于是 `synthesized == 0`，自进化在跑但学不到东西。

**修复**：
* `default_reflector` 改为**基于外部证据的逐句归因**（复用 `grounding`），
  产出三种洞见：`partially_supported`（有原文可修）、`unsupported`（无原文，不给 repair）、
  `no_evidence`（根本没有证据可用）。`repair` **只取证据原文**，绝不取模型改写。
* `default_synthesizer` 只接受带 `repair` 的洞见，目标是**最小编辑**：
  把原响应里出错的那一句替换成证据原文，而不是整段换成一句孤立事实。

**效果**：`synthesized > 0` 有端到端测试。数字篡改用例：
响应 `…提升 2.86 倍。` → 洞见 repair `…提升 1.86 倍` → 训练目标
`SGC 的流式状态是 2·M·D。实测吞吐在 8k 上下文下提升 1.86 倍。`
无原文的编造句（"该架构已在三家芯片厂量产"）**不产生**训练目标。

### D10 〔自进化〕no-regression gate 是噪声判定器，且默认配置下恒通过

**症状**：
1. gate 比较两个**标量点估计**。小评测集上换种子重跑的抖动就有几个百分点；
2. 默认 verifier 恒返回 `0.0` → `base == new` → gate 恒通过，等于没有 gate；
3. 回滚只回滚权重，不回滚记忆库 —— "回滚了，但学错的东西还在库里，下一轮继续注入"。

**修复**：
* 新增 `EvalScores`（逐样本分数容器，兼容 `float` / `Dict[str, float]` 并标注降级）；
* 新增 `paired_bootstrap_ci` + `regression_gate`：每个共同指标算 delta 的
  配对 bootstrap 置信下界（默认 95% / 1000 次），**任一**指标下界低于 `-|tolerance|`
  即回滚；只在一侧出现的指标写进 `notes`，不静默丢弃；
* `ReflectionLoop` 记住调用方是否提供 verifier，`mode in ("weights","both")` 且
  verifier 为默认常量时**拒绝写回权重**并说明原因；
* 评测异常 → 保守回滚（不能声称"没退化"）；
* 回滚同时恢复权重与记忆（新增 `CompressedMemoryBank.state_dict/load_state_dict`，
  深拷贝，含 meta 与命中计数）。

**守护**：`test_gate_rolls_back_when_point_estimate_improves_but_lower_bound_is_negative`
（点估计 **+0.4**、置信下界 **−0.2** → 回滚）、
`test_gate_fails_when_any_single_metric_regresses`（总分涨不能抵消某项崩）、
`test_loop_refuses_weight_writeback_without_verifier`、
`test_rollback_also_restores_memory`。

### D11 〔自进化·G1/G2〕压缩器训练无关、memory-aware 目标从未被接入

**症状**：`MemoryCompressor` 是"训练无关的隐状态读出"，训练多久参数都不变；
`MemoryAwareObjective` 存在但没人做那两次前向。G1/G2 在文档里挂了两轮。

**修复**：
* `TrainableMemoryCompressor`：因果 depthwise 卷积堆叠 + 末端 k 读出。
  结构保证：无注意力、严格因果（实测 `causal=True`）、定长输出 `(B, k, D)`；
* `memory_aware_loss`：双路前向（有记忆 / 无记忆），只对增量 logits 计损失；
* `train_compressor_step`：`tok_emb → 压缩 → 池化 → 双路损失 → 只更新压缩器`
  （基座 `requires_grad` 步内屏蔽、步后恢复）。

**实测**：12 步 Adam（lr 3e-2），增量损失 **4.184 → 3.976**，
参数最大变化 **0.313**，基座参数全程冻结且事后恢复。
对照组：`MemoryCompressor` 任何调用后参数变化 **0.0**。

**必须知道的性质**：增量 logits 全零时损失 = `ln(V)` 而非 0（均匀分布的交叉熵）。
该目标度量的是"记忆带来的那部分 logits 能不能独立预测下一个 token"，
所以"记忆什么都没带"等于随机猜测。守护测试钉住了这个 floor（`ln(64)`）。

### D12 〔抗幻觉〕句子级标签无法训练 L4 头（缺位置对齐）

**症状**：`abstention_loss` 要 `(B, L)` 的逐位置标签，而归因天然是**句子级**的。
缺对齐步骤，L4 只能拿整段标签（全 0 或全 1）训练 —— 等于没训练。
`abstain.py` 的 docstring 写着"位置级对齐由训练器完成"，但没有任何代码做这件事。

**修复**：`Claim` 记录字符跨度（`split_claims` 填充，含无句末标点的尾段），
新增 `char_token_offsets` / `token_labels_from_bindings` / `grounded_label_tensor`：
token 区间与 claim 区间相交即继承该 claim 的 `supported` 标签，
不属于任何句子的位置（标点、空隙）取 `default`。

**守护**：6 项（跨度切出来就是原句、三句各自标签纯净、空隙取 default、
词级 tokenizer 偏移可用、张量版与列表版一致）。

---

## 3. 实测数据

### 3.1 归因判别（`scripts/eval_grounding.py`，`artifacts/grounding_eval.json`）

600 个构造样本，dev/test 各 300，种子 20261026，耗时 3.42 s。

| 类别 | 含义 | 准确率 |
|---|---|---|
| `exact` | 逐字命中（正例） | 1.000 |
| `truncated` | 截断改写（正例） | 0.942 |
| `number_swap` | 单个数字被篡改（负例） | 1.000 |
| `negation` | 否定被翻转（负例） | 1.000 |
| `cross_doc` | 跨文档拼接（负例） | 1.000 |

| 指标 | 全部 600 | test 300（dev 阈值） |
|---|---|---|
| precision | 1.000 | 1.000 |
| recall | 0.9708 | 0.9836 |
| F1 | 0.9852 | 0.9917 |
| accuracy | 0.9883 | 0.9933 |

四态分布：`answer 233 / hedge 0 / clarify 0 / abstain 367`
（hedge/clarify 为 0 是因为构造样本里没有"信息不足"与"问法歧义"两类，
不是状态不可达 —— 四态可达性由 `test_grounding.py` 单独覆盖）。

拦截率：单句 **1.000**、多句 **1.000**（`multi_claim_n = 200`）。

### 3.2 记忆检索（干扰规模扫描）

| 干扰条数 | 库大小 | top-1 命中 |
|---|---|---|
| 0 | 4 | 4/4 |
| 64 | 68 | 4/4 |
| 128 | 132 | 4/4 |
| 512 | 516 | 4/4 |

### 3.3 压缩器可训练性

见 D11：损失 4.184 → 3.976，参数变化 0.313，基座冻结。

---

## 4. 仍未证明 / 明确不做（必须与上面的数字一起引用）

1. **"抗幻觉有效"仅指文本层可归因性判别。** `grounding` 不做语义蕴含。
   一句逐字出现在证据里的话仍可能在语境上被误用（反讽、条件句、引述他人观点）。
   `supported=True` 只意味着"文本依据找得到"，**不意味着这句话正确**。
2. **L4 头本身仍未训练。** 标签（D3）、位置对齐（D12）、校准工具都齐了，
   但没有任何训练脚本、没有带引用的真实评测集。§3.1 的 AUC/ECE 来自
   **代理分数**（最佳单句二元组命中率），不是 L4 头的性能，不可外推。
3. **"自进化会变强"未证明。** 本轮只证明链路通了（能产出带证据原文的训练信号、
   能回滚、能拒绝无评测的权重写回）。没有独立 held-out 集上的提升证据。
   默认配置下 `ReflectionLoop` 会**拒绝**写回权重，这是故意的。
4. **记忆键不是语义嵌入。** `TextKeyEncoder` 是字面 n-gram 哈希：
   不做同义归并，改写过的问法召回率会掉。要语义检索应换模型隐状态（接口一致）。
5. **数字判据对中文数字无效**（"三倍"不会被 `extract_numbers` 捕获）。
6. **否定词计数存在过检**："非常""无论"会被计入否定标记（`_NEG_CUES` 的已知副作用）。
7. **繁简混写会被判低覆盖**（保守偏差，未做词表归一）。
8. **评测样本是模板生成的**，不是真实模型输出。真实输出的分布更脏（半截话、
   表格、代码块），实测量级可能下降。
9. **`test_eval.py::test_throughput_slope_and_memory_growth` 曾因计时噪声偶发失败**（ratio 1.60/2.00 左右）。
   根因是**测量口径**而非算子：在 batch 8 / dim 32 / seq 128 这类微小尺寸下，
   每次调用的**固定开销**（dispatch、allocator、FFT plan）与逐 token 开销同量级，
   于是"4× 长度 → 测出 2.0× 延迟"系统性低估真实指数；机器负载高时就会压到判定下界。
   已把延迟比下界从 2.0 调整为 **1.5** 并在注释里写明理由：
   二次情形由**上界 8.0** 与 `slope < 1.5` 负责拦截，下界的唯一职责是区分
   "随长度增长"与"常数时间"（后者会给出 ~1.0×）。内存侧的
   `peak_memory_loglog_slope` 判据不受计时影响，保持不变。
   连跑 5 次稳定通过（各 ~11.5 s）。
10. **旧路径存在测试隔离问题（已定位、未修）**：`pytest`（按 `pytest.ini` 跑全仓库）
    会在 `safety/` / `simplcore/mix/tests` / `tests_simplcore_pg` 报 37 项 setup 错误，
    但每条路径**单独运行全部通过**（`sgc + safety` 联跑 358 项全绿）。
    这是旧引擎之间的共享状态污染，与 SGC 无关，属 G5 合并议题的前置问题。
11. **不做**：旧引擎迁移（G5）、官网（已移交）、CUDA 内核、56B 规模验证。

---

## 5. 复现

```bash
# 全量测试（303 passed, 1 skipped）
python -m pytest simplcore/core/tests -q

# 抗幻觉实测（写 artifacts/grounding_eval.json）
python scripts/eval_grounding.py

# 只跑本轮新增的守护测试
python -m pytest simplcore/core/tests/test_gates_selective.py \
                 simplcore/core/tests/test_grounding.py \
                 simplcore/core/tests/test_dream_evolution.py -q
```

---

## 6. 文件清单

**新增**

| 文件 | 作用 |
|---|---|
| `simplcore/core/gates.py` | 闸门数学唯一实现（D1） |
| `simplcore/core/grounding.py` | 证据绑定 / L4 标签来源 / 位置对齐（D3, D4, D12） |
| `scripts/eval_grounding.py` | 归因判别实测（D6） |
| `simplcore/core/tests/test_gates_selective.py` | 闸门守护（14 项） |
| `simplcore/core/tests/test_grounding.py` | 归因 + 拒答 + 对齐守护（36 项） |
| `simplcore/core/tests/test_dream_evolution.py` | 自进化 + CMT 守护（37 项） |

**修改**

| 文件 | 改动 |
|---|---|
| `simplcore/core/sgc.py` | 闸门委托 `gates.py`（行为不变） |
| `simplcore/core/selective.py` | `GatedSpectralConv2` 补 L2/L3 + 流式对偶 + 字节核算（D1） |
| `simplcore/core/hybrid.py` | 闸门开关透传 + 参数量计入 |
| `simplcore/core/model.py` | `checkpoint_stream`/`restore_stream` 补 `energy_sum`（D2） |
| `simplcore/core/abstain.py` | 四态决策 + 单向降级 + 校准工具 + 运行时 `GroundedAnswerer`（D5） |
| `simplcore/core/memory.py` | `TextKeyEncoder`（D7）、`TrainableMemoryCompressor` + `memory_aware_loss` + `train_compressor_step`（D11）、`state_dict/load_state_dict`（D10） |
| `simplcore/core/dream.py` | 证据驱动反思/合成（D9）、记忆写入重写（D7/D8）、多指标 gate（D10）、快照回滚 |
| `simplcore/core/__init__.py` | 公开 API 扩充（89 项导出，无缺失） |
| `simplcore/core/tests/test_selective.py` | 字节数期望值 + 关闸门回落用例（D2） |
| `simplcore/core/tests/test_eval.py` | 吞吐判定下界 2.0 → 1.5，附测量口径说明（§4.9） |
| `docs/SGC_SPEC.md` | 新增 §2.7（gates.py）/ §2.8（证据绑定） |
| `docs/DREAMING_CMT.md` | 重写 §3.3–3.5、§4.1、§6 缺口表 |
| `docs/HONEST_STATUS.md` | §0.6 修复清单追加本轮条目 |
