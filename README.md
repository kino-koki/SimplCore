# SimplCore-0.1-α — 无注意力序列架构（实验阶段）

> ⚠️ **实验阶段警告**：目前该架构仅仅处于实验阶段，如果不是真的想试试，
> 建议还是去用 **Mamba** 进行开发、设计模型。本仓库没有经过大规模训练验证，
> 所有"优于 Transformer"的主张都是**机制层结构性质**（可运行断言），
> **不是**实测的质量结论。

**可训练、彻底非注意力**的序列建模仓库。架构叫 **SimplCore**，
主推实现位于 `simplcore/core/`：用**因果谱全局卷积**替代自注意力，
复杂度 `O(L log L)` 训练、`O(1)` 状态流式推理，**无位置嵌入表、无 `max_len` 截断、
无 KV 缓存**。

> **运行时**：训练/推理部署走**自研 CUDA·HIP/C++ 后端**（算子契约见
> `simplcore/core/backend/backend_contract.py`，`python -m simplcore.core.backend_contract`）；
> 仓库内 PyTorch 实现退役为**冻结数值规格（oracle）**——它是后端逐位对齐的数学定义，
> 不参与部署。目标硬件为 **AMD Instinct MI250（ROCm）**：融合流式核以 CUDA 写法
> 交付、经 PyTorch 以 hipcc 编译为 HIP（仅用 HIP 兼容原语），训练主路径走
> `torch.compile`→ROCm CK（见 `scripts/train_tuan_mi250.sh`）。16-op 契约已有
> **fp32 CPU 参考后端**（`csrc/sc_ref_backend.c`，验收见 `docs/REF_BACKEND.md`），
> 生产 HIP 后端待移植。战役计划（1B→3B，硬停闸门 + HBM 账本）见
> `simplcore/core/deploy/campaign.py`（`python -m simplcore.core.campaign`）。

> **SGC 是 SimplCore 内部的一个 mixer 模块**（`simplcore/core/arch/sgc.py`），
> 不是架构的名字。SimplCore 下面有多个可切换的非注意力 mixer：
> SGC / SRFN / GatedSpectralConv2。架构层面的类名一律是 `SimplCore*`。

---

## 1. 它是什么 / 不是什么

| 是 | 不是 |
|----|------|
| 一套可运行的 PyTorch 序列混合器（SGC / SRFN / GatedSpectralConv2） | 一个已证明 SOTA 的模型 |
| 用 FFT 全局卷积 + 递归替代 softmax 注意力 | Transformer 的变体或混合体（**仓库内已无任何注意力模块**） |
| 严格因果、有 O(1) 流式状态、无位置表 | 依赖 KV 缓存或位置编码的架构 |
| 在 CPU 上验证数学正确 / 因果 / 双形式等价 | 在 1B+ 规模上训练过的系统 |

**架构本体完全不属于 Transformer 框架**：全局交互由 `torch.fft` 谱卷积 +
阻尼振荡基字典完成，**没有** `Q/K/V`、`softmax(QKᵀ)`、`scaled_dot_product_attention`、
RoPE、KV 缓存、位置嵌入表。注意力族（`CausalAttention` / `ReasoningAttention` /
`SlidingWindowAttention` / GQA-YaRN / RoPE）已从产品线**整体删除**，
仅作为历史归档保留在 `.trash/attn_removed/`，早期路线原型归档在 `research/`。
两道防线守护这一点：`simplcore/core/arch/purity.py`（配置 + 实例双重断言）与
`SimplCoreConfig.validate()`（只允许非注意力 mixer）。

> ⚖️ **0.1-α 独立复现的实测定位（2026-10-31，合成小任务前哨）**：
> 逐字**精确指向**是 O(1) 状态的**结构性劣势**（2 层探针 SGC 0.398 vs
> attention 1.000 vs chance 0.125，甚至低于局部卷积 0.427）——信息论上限，
> 不是调参问题；聚合任务**未复现出通过率优势**（0.875 vs 0.938），SGC 的
> 赢面在效率轴（常驻状态 36,868 B 恒定 vs 1M KV ~120 GiB；训练缩放斜率
> 1.09 vs 2.35；流式零泄漏）。**engineering trade-off，不是"更好的模型"**。
> 架构响应已落地：精确引用走 `memory/sidecar.py` 三层旁路（见 §3），
> O(1) 状态定位为**选择性寻址器**。完整表与局限见
> `docs/THEORETICAL_EDGE.md` 实测对照节、`SimplCore_Paper.md` §7.2 轴注记、
> 复现协议 `eval/axis_probe.py`；MQAR 绑定缺陷的修复路线分级
> （A/B/C 一致性判据 + R1 定容绑定记忆已落地）见
> `docs/REPRODUCTION_REPORT.md` 与 `docs/THEORETICAL_EDGE.md` 路线分级节。

---

## 2. 快速开始

```bash
# 环境
python -c "import torch; print(torch.__version__)"

# 全量测试（在 WorkBuddy 沙箱里跑需关掉批量删除守卫，见下方说明）
python -m pytest simplcore/core/tests/ -q

# 核心数学 / 双形式等价 / 因果性 / 梯度可达
python -m pytest simplcore/core/tests/test_mixer_math.py -q

# 主线因果性守护（扰动末位 token，断言前文逐位不变）
python -m pytest simplcore/core/tests/test_architecture_purity.py -q

# 生物启发增强：三支柱 + 生成期混合计算 + 自适应课程
python -m pytest simplcore/core/tests/test_bio.py simplcore/core/tests/test_bio_stream.py \
                 simplcore/core/tests/test_adaptive_curriculum.py -q

# 最小示例
python -c "
from simplcore.core import get_preset, SimplCoreModel
import torch
cfg = get_preset('tuan-150m')
m = SimplCoreModel(cfg)
x = torch.randint(0, cfg.vocab_size, (1, 64))
logits, loss = m(x, x); loss.backward()
print('fwd/bwd OK', tuple(logits.shape), float(loss))
st = m.init_stream(1)
_, st = m.stream_step(x[:, 0], st)        # O(1) 状态流式
print('stream bytes:', m.stream_bytes(1))
"

# 理论优势的可执行验证（34 项机制层断言，全过才退出 0）
python -m simplcore.core.eval.claims_audit

# A1/A2 最小证伪实验：自主学习闭环（真实代码沙箱判分，CPU 可跑）
python -m simplcore.core.pilot --rounds 2 --out runs/pilot
```

### ⚠️ 常见陷阱（第三方使用者必读）

**1. `model(x, x)` 是恒等陷阱——loss 会静默收敛到垃圾。**
`forward(idx, targets)` 的 `targets` 与 `logits` **同位置对齐，不做任何自动
shift**。位置 t 的输入包含 `x[t]` 本身，若 `targets is x`，模型只需学会
"复制当前 token" 即可把 loss 压到 1e-4 量级——**不报错、loss 照常下降，
只有看生成文本才会暴露**。正确写法是标签右移一位：

```python
x, y = batch[:, :-1], batch[:, 1:]   # y[t] 是 x[t] 的下一个 token
logits, loss = model(x, y)
```

本 README 快速开始与仓库测试里出现的 `model(x, x)` 全部是 **API 冒烟**
（只验证前向/反传能跑通），不构成训练语义示范。数据链路（`ShardedTokenDataset`/
`SFTDataset`）内部已做正确偏移。

**2. 内置 5 题的 pass_rate 上升只能证明记忆。**
`pilot` 的内置题目**同时**是闭环的训练经验来源——闭环前后在同一组题上的
pass_rate 上升只能证明**记忆（同题重放）**，不证明泛化。要主张「越用越好」，
必须给 `PilotConfig(heldout_problems=...)` 提供与经验**不重叠**的 held-out
题集，报告会以 `a1.heldout` 为准记账；未提供时 `verdict` 只会给出
`memory-evidence-only`，不给 pass。

**3. 「不退步」≠「越用越好」。**
`regression_gate` 证明的是写回权重**没有退化**（必要条件），不是能力提升。
泛化提升的判据只能是 held-out 评测——`ReflectionConfig.heldout_eval`
提供后在旧/新权重上各跑一次并记入 `ReflectionReport.heldout_delta`；
未提供时报告明确标注「泛化主张无判据」。

> ⚠️ **WorkBuddy 沙箱里跑全量测试**：沙箱的 `sitecustomize` 装了「批量删除守卫」，
> 测试清理临时目录（≥50 个文件）时会被拦下并抛 `SystemExit(1)`，表现为
> `test_data / test_tokenization / test_claims_audit / test_paper_audit` 等
> **单独跑全过、随全量套件一起跑必挂**的假失败。关掉守卫即可：
> `CODEBUDDY_SAFE_DELETE_ENABLED=0 python -m pytest simplcore/core/tests/ -q`。
> **0.1-α 复核（2026-10-31）**：非沙箱环境默认全绿——**1110 通过 / 6 环境跳过
> / 0 失败**（504 s；跳过项 = 融合热路径扩展未编译 + 慢探针环境开关）。

---

## 3. 架构一图流

```
输入 token id ──► tok_emb ──► [ Block ]×N ──► ln_f ──► head ──► logits
                         │
        ┌────────────────┼───────────────────────────────────────┐
        ▼                ▼                                        ▼
   非注意力 mixer      L4 拒答头                              SwiGLU FFN
   (sgc / srf /        (Linear(dim→1) 标量头)                   + 残差
    gated-spectral)
        │
        ▼
   FFT 全局卷积(训练) ⇄ 二阶 IIR 递推(流式)
   阻尼振荡基字典 + 深度因果卷积 + 内容门控 + L2/L3 闸门
```

数据流的每一层都是 `mixer → FFN`，外加可选 CMT 记忆注入（残差流加性项）
与 L4 拒答头。详见 `docs/ARCHITECTURE.md`。

### 三层精确引用（L0/L1/L2）——「第几个字、哪一句话」怎么做到

要回答「第几个字」，物理上必然需要 O(N) 存储（香农，没有例外）。所以架构
**不让 O(1) 做精确检索**——分工如下（`memory/sidecar.py`）：

| 层 | 内容 | 存储 | 状态 |
|----|------|------|------|
| **L0 常驻** | gist 匹配 + 指针头（**不是原文**） | O(1)，HBM（PointerHead ≈ 2·dim² 参数，与 N 无关） | ✅（流式状态 + `PointerHead`） |
| **L1 旁路** | 原文/token **精确副本**，可寻址 | O(N)，Host RAM/SSD（4M tokens：int32 原文 16 MiB，uint16 可压至 8 MiB；块 gist 索引 256 MiB @ dim=1024） | ✅（`SidecarArchive` + `BlockGistIndex`） |
| **L2 取回** | O(1) 状态每步输出**指针分布** → top-k → 精确读取 → 注入 | 取回 O(k·B·dim)，**与 L 无关** | ✅（`PointerHead` → `read_topk` → 零初始化注入；局部精确注意力实验件在 `sidecar_exact_attn.py`） |

**定位改口（诚实成本）**：加了 L1 之后「O(1)」必须说全——

> **O(1) 常驻状态（HBM），O(N) 精确档案（主机内存/SSD）。**

§5.12 的「1M ≤ 2 GiB」讲的是 **HBM 峰值**，仍然成立：档案在主机内存，
不进 GCD。**O(1) 状态的价值不是"存储器"，是"选择性寻址器"**——它不存
原文，它决定去几百万字里的哪几处取。这个定位比"O(1) 通吃"更准确，也更
经得起审稿人问。保留的架构优势只有两条：避免 O(L²) 全量注意力（取回
O(k·B·dim) 与 L 无关）、HBM 常驻仍 O(1)；总存储不再是 O(1)，如实记账。

诚实边界：块 gist 是**无参数池化**，指针头**零初始化 → 初始均匀分布**
（未训练时检索无判别力）；"检索质量提升"未经训练验证——本模块先把
**通路与账本**立起来（可测、可审计），gist 学习是后续工作。

---

## 4. Mixer 家族（同一套 config 切换，全部非注意力）

| `mixer=` | 混合器 | 注意力 | 流式状态 | 训练 FLOPs 长程项 |
|----------|--------|--------|----------|-------------------|
| `sgc` | 谱全局卷积（默认） | 无 | O(1) IIR | `O(D·log L)` |
| `sgc-nogate` / `sgc-nolocal` | 去内容门控 / 去局部卷积 | 无 | O(1) | `O(D·log L)` |
| `sgc-r` | SGC-R 谱专家路由（M 基分 G 组 + 逐 token top-k 注入，MoE 路由非注意力；v1.2） | 无 | O(1)（与 sgc 逐字节同） | `O(G·D·log L)`（核按组即建即弃） |
| `srf` | SGC + selective SSM（输入相关 Δ） | 无 | O(1) chunked scan | `O(D·L/chunk)` |

> 曾经的 `attn` / `hybrid*` / `reason_attn_every` 取值**已删除**：
> `SimplCoreConfig.validate()` 现在只接受上面五类非注意力 mixer。

**SGC v1.2 增强**（config 开关，默认全关，开启不影响 FFT/IIR 对偶与纯洁性）：

* `sgc_path_gate` —— 长程/局部/瞬时三路的 softmax 自适应混合（`w_t = softmax(W·h_t)`，
  逐位置因果，并行/流式逐位一致）；
* `sgc_inst_nonlin` —— 瞬时项非线性 `feed·SiLU(u)`（默认 `feed·u`）；
* `sgc_init_mode="logfreq"` —— 振荡频率**对数网格**初始化（振荡周期与衰减窗口
  尺度配对；**v1.2 起为默认**——legacy 均匀网格存在 tau=2M(M+1)-1 处的公共
  零点，构造期守护 `_assert_no_common_zero` 直接拒绝）；
* `sgc_rho_param="logspace"` —— 衰减极点 logspace 参数化（4M 所需极点可达，
  `tuan-2b` 已启用）；
* `sgc_route_groups` / `sgc_route_topk` / `sgc_route_aux` —— SGC-R 谱专家路由
  （见 `docs/SGC_CORE.md` §10）。

**Agent 域 v1.2**：`jitrl.py` —— JitRL 式**免训练测试时优势调制**（非参数经验
轨迹记忆 + 检索式动作优势 + 加性 logit 调制，KL 约束策略优化的闭式更新；
JitRL arXiv:2601.18510）。`SimplCoreModel.generate(logit_bias_fn=...)` 为一等
接入钩子（默认 None 逐位恒等）。与 Reflexion（提示端）、L2 证据门（mixer 端）
构成"经验约束生成"三位置实例化。无环境实测（WebArena 判据 W4 见论文 §7.4）。

---

## 5. 包结构（0.1-α 结构化拆分）

`core/` 按依赖方向拆为八个子包（详细表见 `docs/ARCHITECTURE.md`）：

```
simplcore/core/
├── arch/       mixer 数学核心（sgc / srf / ssm / selective / gates / purity / abstain / grounding）
├── model/      配置与模型（config / model / hf_bridge / tokenization / tokenizer）
├── memory/     记忆与反思巩固（memory / dream/ / code_exec / smm）
├── dialogue/   对话域（chat / dialogue_runtime / dialogue_eval / reasoning）
├── training/   训练域（data/ / train/ / trainer / phase1_train / sft / gate2 / envlearn）
├── eval/       评测与账本（eval / probes / scaling / longctx_eval / theory / mathproof / paper_audit）
├── backend/    后端契约（backend_contract / backend_bridge / conformance_runner）
└── deploy/     战役与部署（campaign / deploy_budget / sprint30）
simplcore/lab/  实验性模块（jitrl / pilot / conditions_ledger），core 不得反向依赖
```

历史脚本与老 checkpoint 的旧平铺路径（`simplcore.core.sgc` 等）由
`core/_compat_aliases.py` 的 sys.modules 别名继续兼容；新代码一律用新路径。

### 命名纪律（0.1-α 命名降调）

为避免"伪科学式命名"，0.1-α 起对模块与公共类名做了降调，旧名全部保留为兼容别名
（pickle / 历史脚本不受影响）：

| 旧名（已弃用） | 新名 | 降调理由 |
|----------------|------|----------|
| `native_intelligence.py` / lab | `conditions_ledger.py` | "原生智能"是口号；模块实际是**五项智能条件的可执行断言 + 规模阈值账本** |
| `bio_computation.py` | `bio_enhance.py` | "生物计算主义"神秘化；模块实际是**生物启发的训练增强**（混合计算耦合 + 代谢预算） |
| `eval/theory.py` | `eval/claims_audit.py` | "理论"言过其实；模块实际是**把文档主张变成可运行断言的主张审计器** |
| `DeepReasoner` | `IterativeRefiner` | 不"深度"也不"推理"；实际是**迭代式文本精化循环 + 早停** |
| `NonlinearThinker` / `NonlinearThinkCell` / `ThinkLedger` | `LatentRefiner` / `LatentRefineCell` / `RefineLedger` | "思考者"拟人化；实际是 **latent 迭代精化 + ACT 停止头** |
| `ThinkAccelerator` | `AdaptiveDepth` | "思考加速"拟人化；实际是**逐 token 自适应计算深度** |
| `DreamingLoop` / `DreamConfig` / `DreamReport` / `DreamStage` | `ReflectionLoop` / `ReflectionConfig` / `ReflectionReport` / `ReflectionStage` | "做梦"隐喻神秘化；实际是**离线反思巩固循环**（HARVEST→REFLECT→SYNTHESIZE→CONSOLIDATE→VERIFY） |

保留项及理由：`dream/` 目录名与"Dreaming"概念词保留——它对应持续学习文献中的
offline consolidation 术语，且模块内已用**诚实边界**声明"自进化未证明"；`mathproof.py`
（13 项符号+fp64 数值证明）、`purity.py`（纯洁性守卫）命名与实际能力一致，不动。

---

## 6. 文档索引

| 文档 | 内容 |
|------|------|
| `SimplCore_Paper.md` | **架构白皮书（v1.1.2，含 v1.2 扩展）**：定理 1–4 / 命题 5–7（命题 5 误差界 v1.1.2 含 1/\|sinθ\| 修正）、§3.5 世界模型理论动机、SGC-R 谱专家路由、负面结果与撤回、13 项数学证明、部署/战役账本、§7 等效参数推测（明确标注为可证伪理论推测） |
| `docs/THEORETICAL_EDGE.md` | **逐模块的机制层主张与边界论证**：每个模块相对传统 Transformer 的复杂度/状态/机制主张与边界（可执行断言在 `eval/claims_audit.py`） |
| `docs/ARCHITECTURE.md` | **总体架构设计**：数据流、mixer 分类、流式、记忆、编解码、训练/评测 |
| `docs/SGC_CORE.md` | **SGC mixer 深度解析**：振荡基字典、FFT↔IIR 对偶、内容门控、局部路径、四层闸门、流式状态 |
| `docs/SPRINT30.md` | **30 天冲刺计划**：测后锁预算 + MFU 下限闭合 + 降档阶梯——"1 月必须完全做好训练"的硬保证（`python -m simplcore.core.sprint30`） |
| `docs/CONDITIONS_LEDGER.md` | **智能条件账本**：五项条件可执行断言 + 120B 主干/2B token 视野阈值账本（`python -m simplcore.core.conditions_ledger`） |
| `docs/MODULES.md` | 逐模块索引（每个模块一段职责说明） |
| `research/` | 早期路线原型归档（**不发布、不参与 CI**），仅供考古与对照 |

---

## 7. 当前真实能力边界

- ✅ mixer 核心：数学正确、严格因果、FFT↔IIR 双形式等价（fp32 `max|Δ|<4e-5`）、梯度可达
- ✅ 四层抗幻觉闸门（L1 选择门 / L2 证据门 / L3 不确定门 / L4 拒答头）结构完整、可消融、流式==整段
- ✅ O(1) 流式状态 + `checkpoint_stream/restore_stream` 持久化、与 L 无关
- ✅ 无词表模式（字符/字节/BPE 可插拔）、HF 权重迁移桥（只搬 embedding/FFN/norms）
- ✅ 架构纯洁性：产品线内**零注意力模块**，由 `purity.py` 的**三层守卫**
  （配置 → 实例 → 源码 AST）与 CI 共同守护；类名改名、保留 `q_proj/k_proj/v_proj`、
  或在 `forward` 里调用 `scaled_dot_product_attention` 都会被抓（后者仅允许在
  `equal_budget.py` / `throughput_eval.py` 这类 **TS 对照臂**里，且需文件头显式
  `purity: allow-attention-reference` 标记）。
- ✅ **原创点 A/B 的可执行产出**：`numerical_stability.py`（含 `ρ/θ/L` 显式依赖的
  二阶 IIR 误差界、误差感知基网格、由误差预算反解的 `resync_period`）与
  `retrieval_bound.py`（把"状态容量够"与"线性读出不可寻址"拆开：1M/30.4MiB
  容量比 ≈6.1×，读出 SNR≈2.8）。长序列 fp32 误差实测补入
  `docs/SGC_CORE.md §3.2`（`<4e-5` 只适用于 `L≲512`）；详见
  `docs/NUMERICAL_STABILITY.md`、`docs/RETRIEVAL_BOUND.md`。
- ✅ **理论优势可执行化**：34 项机制层断言（`simplcore.core.eval.claims_audit.py`），
  `docs/THEORETICAL_EDGE.md` 的每条主张都有对应检查，由 `test_claims_audit.py` 守护
- ✅ **支柱 C 闭环**：代码沙箱（`code_exec.py`）+ 独立 verifier + no-regression gate +
  **内置权重巩固器**（`default_consolidator`）→ "真实环境写代码 → 验证 → 写回权重"
  默认就是通的，且退化时逐位回滚
- ✅ **「越用越好」的判定口径已收紧（0.1-α）**：gate 的基线评测移到**巩固前**
  （旧权重）——此前 base/new 同在巩固后权重上评测，配对 delta 恒为噪声；
  「不退步」（`regression_gate`）与「泛化提升」（`ReflectionConfig.heldout_eval`
  → `heldout_delta`）分开记账，后者是与经验不重叠的 held-out 判据，
  未提供时报告明确标注"泛化主张无判据"
- ✅ **思考加速**（`accel.py`）：`AdaptiveDepth` 逐 token 按必要性选深度，
  `skip_threshold=-inf` 逐位退化全算力，`ComputeLedger` 记结构性加速比；不引入注意力
- ✅ **非线性思考**（`nonlinear.py`）：`LatentRefiner` 在 latent 空间做**可微迭代精化**
  （门控非线性残差单元 + ACT 式停止头自适应步数）；`max_iters=0` 逐位直通，
  思考零新增流式状态，CE + ponder 正则可训练；不引入注意力
- ✅ **Simpl Model Memory**（`smm.py`）：三层记忆整合架构——工作记忆（O(1) 流式状态）
  → 情景记忆（CMT 库检索注入）→ 语义记忆（蒸馏写权 + no-regression gate 回滚）；
  `experience/recall/answer/sleep` 四入口闭环，`SMMLedger` 全周期记账
- ✅ **深度推理**（`reasoning.py`）：`IterativeRefiner` 迭代式"提示→思考→答案"循环，
  `fixed_steps`/`stabilize` 早停 + `tool` 钩子，复用 O(1) 流式状态；不引入注意力
- ✅ **环境学习**（`envlearn.py`）：`WeightBudget`（1B 冻结 + ≤2B 可学、总上限 3B，
  显存只随可学权重增长）+ 分阶段蒸馏（交互→总结→蒸馏→验证→回滚），复用 Dreaming 链路
- ✅ **SimplUnity 模块化**（`simplcore/unity/`）：冻结主干 + 路由 + 多专家；
  `ModuleBundle` 支持专家级拆分/跨主干拼接（`base_fingerprint` + 预算守卫），
  `UnitySelfEvolution(num_workers=N)` 后台并行训练候选专家，`register_domain(..., eval_loader)`
  的 **GATE-3 无回归门**拒绝退化候选，`ExperienceLoader` 把域经验转 next-token batch；
  0.1-α 补齐统一生成入口 `UnityModel.generate()`（O(1) 流式 prefill + MoE 路由采样，
  与基座同温度/top-k/top-p 口径）。
  三个月路线（2B 1 月 + 自进化/模块 2 月，GCD 分工）见 `docs/SIMPLUNITY_3MONTH_PLAN.md`。
- ✅ **多模态入口（0.1-α，从 0 到 1）**（`model/multimodal.py`）：图像 patchify /
  音频帧下采样 → **软 token 拼接**进无位置表序列（模态前缀因果在前）——
  无词表/无位置编码架构对多模态天然友好，不引入注意力；`generate` 走
  `stream_step(inputs_embeds=…)` 真 O(1) 模态前缀 prefill；无模态时与纯文本
  路径**逐位一致**（测试锁定）。诚实边界：adapter 未在真实多模态数据训练验证，
  只锁接口/形状/梯度/一致性。
- ✅ **世界模型容量探针**（`eval/world_probe.py`，0.1-α）：白皮书 §3.5 的
  世界模型主张第一次变成**可证伪测量**——线性读出头（fp64 闭式岭回归）从
  O(1) 递推状态读出 t+Δ 未来 token 桶；三基线对照（行置换/多数类/样本账本），
  `lift>0` 才支持「状态携带可预测未来」的弱主张，`lift≤0` 照实记账负结果。
  **不构成「世界模型已实现」宣称**（模块头与账本 claim 字段均已声明）。
- ✅ **训练吞吐基准**（`scripts/bench_train_throughput.py`，0.1-α）：
  tokens/s 稳态记账（中位数 + p90）、fp32/bf16 对照、序列长度缩放健康检查
  （超线性即有 O(L²) 隐患混入）、显存峰值单列；同轮修复 Unity 域隔离
  warm-up 的 DataLoader 硬编码 `num_workers=0`（MI250 上实测拖低 GPU 利用率
  的 IO 瓶颈）——多域隔离训练吞吐提升。
- 🧪 **物理世界模型**（`model/physics_world.py`，0.1-α，**实验功能，opt-in
  默认关闭**）：文本/多模态抗幻觉的**第三正交信号源**——L2 证据门管
  「有没有证据」、L3 不确定门管「模型确不确定」，物理门管「说的和物理
  规律矛不矛盾」（连续性/守恒/运动学三类手工约束 + 可学习预测头，
  `hallucination_signal` 供 L4 拒答头联用，`bias_fn` 适配生成钩子——
  物理不一致时给 EOS 正偏置，零侵入）。**tuan-2b 等全部预设完全不使用，
  训练主路径零引用**（守护测试锁定）；只有显式
  `use_physics_world_model=True` 才接线，未开时零行为差异。
  诚实边界：轨迹抽取器为结构化占位（阿拉伯数字）、约束为手工先验、
  预测头未训练——不构成「模型理解物理」宣称。
- ✅ **部署账本达标**：1M 上下文完整推理 **1.948 GiB ≤ 2 GB**（bf16 权重 1.888 +
  O(1) 流式状态 **30.36 MiB**（tuan-1b，batch=1，fp32，15 层）+ 融合单缓冲 workspace；削减来自融合单遍+缓冲复用，
  非 chunk/层数/精度），1M→8M 翻倍增量**恒为 0**（≤1 GB 主张成立）
- ✅ **超长上下文：课程 + 分块训练（截断 BPTT）**：
  `simplcore/core/training/seqlen_curriculum.py` + `train.py --seq-len-curriculum`
  （或预设 `context`）可把 seq_len 从 4K 阶梯爬升到 **4M**；课程切换**不改模型权重**
  （SGC 长度无关），仅拉长数据窗口并按比例重算梯度累积；无课程时退化为固定
  `--seq-len`（零回归）。
  **训练有两条路径**：默认整段并行 FFT（物化 `(L, D)` 核，`--max-kernel-gib`
  默认 4 GiB 会在启动时拒绝 4M×2048=30.5 GiB 的窗口）；加 `--train-chunk 4096`
  后改走逐块前向：块内并行 FFT、块间用 mixer 状态 `(y1, y2)` 的齐次响应续接，
  每块**立即 backward**，激活 **`O(chunk·D)`**，块间状态按截断 BPTT detach
  （梯度窗口 = chunk，前向无损）。分块前向与整段路径逐位等价
  （`test_chunked_training`，fp32 误差 <1e-5）。
  `--train-chunk` 与 `--enhance / --compile / --cuda-graph` 及多卡 DDP 互斥，
  且目前只支持 `sgc` 系列 mixer（SRF 的 chunk 内核尚未实现，遇到会显式报错）。
- ✅ **1B→3B 战役就绪**：自研 CUDA·HIP/C++ 后端算子契约（16 内核 + 逐位一致性硬检查；
  fp32 CPU 参考后端已可执行 9 项验收，生产 HIP 后端待移植）+
  **一致性跑批器**（`BackendAdapter` 接口 + 9 项可执行验收，负向变异测试证明
  每项检查真能抓缺陷）+ **纯算子重组模型**（与原生模型三重等价，契约完备性已证）+
  **训练驱动**（分块 BPTT + 课程 + 冻结掩码优化器 + 回滚闸门，tiny 实跑
  loss 3.50→0.26）+ 战役计划（90 天总预算硬约束 + "无人可忽视"演示包 8 条）——
  运行时零依赖 PyTorch，后端过 conformance 即可开训
- ✅ **数学证明 13/13**（`mathproof.py`）：闭式核↔IIR 递推（sympy 符号 + fp64）、
  分块携带律、FFT 无混叠、BIBO 稳定（含 fp32 sigmoid 饱和的 ρ 卫兵）、
  流式==并行（fp64）、O(1) 状态充分性（马尔可夫性）、复杂度公式逐项审计、
  HBM 账本代数、冻结基座梯度隔离——**卡没到，数学先做到无误**
- ⚠️ 六大控制器（accel / nonlinear / reasoning / envlearn / smm / dream）现已通过
  `simplcore.core.training_integration` + `train.py --enhance` **接入训练闭环**
  （opt-in、默认关闭、不开任何增强时训练逐位不变、零回归），但它们**仍未训出对应的
  加速比、推理质量、代码能力**——机制已接线 ≠ 质量达标，与 `theory.py` 诚实边界一致
- 🧪 **生物启发增强增强 `bio`**（第七个、研究型）：把 Milinković & Aru (2026) 的
  三条计算原则（混合计算 / 尺度不可分 / 代谢约束）做成可训练机制，同样 opt-in、零回归；
  **这是机制实验件，绝非声称模型「有意识」**——论文本身也只是论证这些是意识的*可能*
  必要条件，且明确可被非生物系统复现。详见 `docs/BIO_ENHANCE.md`
- ⚠️ **1B 以上规模从未实际训练**；配置与链路已打通，曲线待测（目标硬件：1× AMD Instinct MI250）
- ⚠️ 未与 S4 / Mamba / Hyena / GLA 在 LRA 等公认基准做公平对比
- ⚠️ 长上下文外推只有"输出有限"证据，无质量证据
- ❌ 不得据此宣称任何 SOTA

### 训练增强集成（`--enhance`）

六个控制器不再是孤立 demo：``train.py --enhance`` 把它们接成训练流水线里的
**可选、默认关闭**的真实组件，不开任何开关时训练行为与历史版本逐位一致。

| 增强 | 训练角色 | 接线方式 |
| --- | --- | --- |
| `ponder`（nonlinear） | latent 思考进入训练前向，CE + ponder 正则可训练 | 替换训练前向（``nn.Module``，基座+思考单元同进优化器） |
| `envlearn` | 冻结基座、只训 ``LearnedAdapter``（``WeightBudget`` 守 3B 上限） | 替换训练前向 + 仅适配器进优化器（``nn.Module``） |
| `accel` | 评估期跑自适应深度，``ComputeLedger`` 报结构性加速比 | 评估 hook（不改主 val_loss） |
| `smm` | 评估期把采样窗口解码成文本、写入情景记忆库（经验采集） | 评估 hook |
| `dream` | 训练后跑 ``ReflectionLoop`` 自进化巩固（no-regression gate 逐位回滚） | 训练后 post-hook |
| `reasoning` | 评估期跑 ``IterativeRefiner`` 迭代式思考 | 评估 hook |
| `unity` | 冻结基座 + 路由门 + 多个域专属小专家（MoE 效果，训练数据按专家分域）；``--unity-domains`` 强制按域隔离 warm-up；``spawn_expert`` + ``UnitySelfEvolution`` 后台线程长新专家；专家可持久化 | 替换训练前向（``nn.Module``，仅路由+专家进优化器）；详见 ``docs/SIMPLUNITY_SGC.md`` |
| `bio` | **生物启发增强**（Milinković & Aru 2026, Neurosci. Biobehav. Rev. 181, 106524）：把论文三条核心计算原则做成可训练增强——①混合计算（常驻 O(1) 连续场 × 离散 token 脉冲相互构成）②尺度不可分（micro 路由 / meso 连续场 / macro seq-len 课程三尺度可训双向耦合）③代谢约束（能量预算 + 代谢效率损失塑形学习/稳定性/信息流）；可叠加 ``unity``（meso/macro 调节路由温度）；**生成期也跑混合计算**（``stream_step``/``generate`` 走连续场，与整段前向逐位一致）；**meso→macro 反馈**（连续场就绪度门控序列长度课程推进） | 替换训练前向（``nn.Module``）并自带流式/生成接口；旋钮：``--bio-energy-weight / --bio-metabolic-budget / --bio-field-dt / --bio-field-scale / --bio-adaptive-curriculum / --bio-readiness-gate / --bio-readiness-grace``；详见 ``docs/BIO_ENHANCE.md`` |

使用：

```bash
# 注意：tuan-1b / tuan-3b 预设自带 4M 课程，默认整段路径会被 --max-kernel-gib 4 拒绝。
# 跑 4M 课程时加 --train-chunk（例：--train-chunk 4096），激活降到 O(chunk)；
# 增强/compile/CUDA graph/多卡 DDP 与 --train-chunk 互斥。

# 仅把"思考算力"训进模型（其余增强默认关，行为逐位不变）
python -m simplcore.core.train --preset tuan-150m --enhance ponder

# MoE 粘合：4 个初始小专家、每 token 激活 2 个
python -m simplcore.core.train --preset tuan-1b --enhance unity \
  --unity-experts 4 --unity-topk 2

# 组合（同位置 wrapper 的替代关系由 ENHANCEMENT_SPECS 显式声明：
#   envlearn 替代 ponder，unity 替代 envlearn，bio 消费 unity（唯一合法叠加）；
#   被替代项会写入 em.superseded 并由 train.py 打印 WARNING，不再静默 last-wins。
#   accel/smm/dream/reasoning 可叠加；未知增强名直接报错）
python -m simplcore.core.train --preset tuan-1b --enhance accel,smm,reasoning

# 生物启发增强：混合计算 + 尺度不可分 + 代谢约束（可叠加 unity）
python -m simplcore.core.train --preset tuan-1b --enhance bio \
  --bio-metabolic-budget 1.5 --bio-field-dt 0.5

# 生物启发增强 + SimplUnity：中/高尺度连续约束离散路由
python -m simplcore.core.train --preset tuan-1b --enhance unity,bio \
  --unity-experts 4 --unity-topk 2

# 生物启发增强 + 自适应长上下文课程：连续场就绪度不足时推迟拉长 seq_len
python -m simplcore.core.train --preset tuan-1b --enhance bio \
  --seq-len-curriculum 4096:0.1,16384:0.25,65536:0.45,262144:0.65,1048576:0.85,4000000 \
  --bio-adaptive-curriculum --bio-readiness-gate 0.5 --bio-readiness-grace 200
```

`smm`/`dream`/`reasoning` 需要编解码器：未显式传入时自动用 ``CharTokenizer(limit=vocab_size)``
兜底，保证接入点永远可跑；`dream` 的权重写回需独立 held-out 评测器（否则 no-regression gate
默认拒绝写回，只跑 reflect→synthesize 与记忆写入）。单元验证见
``simplcore/core/tests/test_training_integration.py``（CPU 可跑，全部 PASS）。

---

## 8. 目录结构

```
simplcore/                       # 发布的产品线（唯一一条）
  core/                          # ★ SimplCore 架构本体（0.1-α 八子包，依赖自上而下）
    arch/                        #   mixer 数学核心
      sgc.py                     #     SGC mixer：谱全局卷积 + FFT/深度卷积 + O(1) 流式状态
      srf.py / ssm.py            #     SRFN mixer / selective 谱状态空间
      selective.py               #     GatedSpectralConv2（GLA/Mamba2 式通道门，非注意力）
      gates.py                   #     L2 证据门 / L3 不确定门（共享数学）
      registry.py / purity.py    #     mixer 注册表 + 架构纯洁性守卫（无注意力硬约束）
      abstain.py / grounding.py  #     L4 拒答头 + 证据绑定层
      accel.py / nonlinear 位于 model/
    model/                       #   配置与模型
      config.py                  #     SimplCoreConfig + 规模预设（SIMPLCORE_PRESETS）
      model.py                   #     SimplCoreModel / SimplCoreBlock / SimplCoreForCausalLM
      nonlinear.py               #     非线性思考（latent 可微迭代精化 + ACT 停止头）
      tokenization.py / tokenizer.py       # 无词表编解码器（char/byte/bpe）
      hf_bridge.py / export.py / adapter_model.py
    memory/
      memory.py                  #     CMT 压缩记忆库
      dream/                     #     自进化 Dreaming 循环（types/buffer/evaluation/hooks/loop）
      code_exec.py               #     代码执行沙箱 + pass@k 验证器
      smm.py                     #     Simpl Model Memory（三层记忆自主学习整合）
    dialogue/
      chat.py / dialogue_data.py / dialogue_runtime.py / dialogue_eval.py
      reasoning.py               #     深度推理（迭代式思考循环 + 早停 + tool 钩子）
    training/
      data/                      #     真实数据管道（clean/sources/pack/dataset/cli）
      train/                     #     训练主循环（distributed/optim/checkpoint/evaluate/cli/main）
      trainer.py                 #     Phase 1/2 训练驱动（分块 BPTT + 课程 + 回滚闸门）
      phase1_train.py / sft.py / ab.py / seqlen_curriculum.py / gate2.py
      training_integration.py    #     增强整合（nonlinear/smm/dream/envlearn/bio/unity 接线）
      envlearn.py                #     环境学习（1B 冻结 + ≤2B 可学、分阶段蒸馏进权重）
    eval/
      eval.py / probes.py / bench.py / memprobe.py / scaling.py / extrapolation.py
      equal_budget.py            #     等预算对照 harness（协议 X-equal-budget）
      longctx_eval.py            #     长上下文检索评测（X-longctx-retrieval + X-stream-consistency）
      code_passk_eval.py / throughput_eval.py / theory.py / mathproof.py
      paper_audit.py             #     论文/战役账目交叉审计（文档数字 vs 代码事实源）
    backend/
      backend_contract.py        #     自研 CUDA·HIP/C++ 后端算子契约（16 内核 + 逐位一致性）
      backend_bridge.py / conformance_runner.py
    deploy/
      campaign.py                #     1B→3B 战役计划（HBM 账本 + 硬停闸门 + 验收协议）
      deploy_budget.py / sprint30.py
    common.py / bio_enhance.py   # 共享工具 / 生物启发增强（三支柱 + 代谢预算）
    tests/                       #   守护测试（以 pytest 统计为准；可选依赖项自动跳过）
  lab/                           # 实验性模块（jitrl / pilot / conditions_ledger），不承诺稳定
  unity/                         # SimplUnity：冻结基座 + 路由门 + 多域小专家
  safety/                        # 内容安全 / 越狱检测 / RBAC / 加密
  rag/ search/                   # 外围子系统（联网检索 / RAG，可选依赖，未与 core 强耦合）
research/                        # ★ 早期路线原型归档（不发布、不进 CI、不入库）
.archive/                        # 已退役材料（含 java-high-performance Java 平行原型、QXT 遗留代码）
third_party/                     # 第三方子仓
```

> 打包发布 `simplcore` / `simplcore.core`（八子包）/ `simplcore.lab` / `simplcore.safety`
> / `simplcore.ops`（见 `pyproject.toml`）；`research/`、`.archive/` 与 `.trash/`
> 均不参与 CI，Java 平行原型已冻结于 `.archive/`，不得作为架构证据引用。
