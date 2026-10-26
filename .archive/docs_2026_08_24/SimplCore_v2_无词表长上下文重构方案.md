# SimplCore v2 架构重构方案
## 无词表连续符号接口 (GlyphStream) + 近 O(1) 上下文显存 (TriMemory)

> 文档版本：v2.0-draft
> 日期：2026-08-22
> 评价与方案基于对仓库的实际代码复核（`simplcore_pg/`、`simplcore/`、`native_mm.py`、训练脚本与既有审计报告 `SimplCore_架构评价.md`）

---

# 第一部分　现状评价

## 1.1 总评

在 2026-08-07 既有审计（综合 ≈5/10）的基础上，本次复核（2026-08-22）维持原结论并补充新发现：

**这是一个"真实可运行、代码有质量"的研究原型，主线 `simplcore_pg/` 的工程质量明显优于仓库平均水平；但工程收敛度低、文档与实现脱节、"证明"产物失真三大问题依然存在。**

| 维度 | 评分(10) | 本次复核要点 |
|------|---------|------------|
| 架构创新完整性 | 7 | SGC/DMC/CCR/MoE 组件真实实现且可微；`native_mm.py` 已有 token-free 雏形 |
| 主线代码质量 (`simplcore_pg/`) | 7.5 | 配置强校验、KV Cache + RoPE 位置偏移正确、torch.compile/FX 双路径、MoE aux/z-loss 完备 |
| 文档准确性 | 3 | `simplcore_ARCHITECTURE.md` 仍描述不存在的 Java/CUDA 主线 |
| 实验严谨性 | 3 | `final_physics_proof.json` 结论与指标矛盾的问题未修复 |
| 工程收敛度 | 4 | 根目录 10+ 个训练入口、历史产物混杂、四套并行实现（`SimplCore.py`/`simplcore/`/`simplcore_pg/`/`java-high-performance/`） |

## 1.2 本次复核的新发现（与重构直接相关）

1. **TensorFlow 实际已移除。** 全仓核心源码（`simplcore/`、`simplcore_pg/`、`src/`、`scripts/`、`tests/`）已无 `tensorflow`/`sentencepiece` 引用，仅剩历史产物 `tensorflow_backup_20260712.zip` 与 `tokenized_cache/`（自定义分词缓存）。因此"抛弃 TF 词表"的真正含义是：**抛弃残留的离散词表路线本身**——`nn.Embedding(V,H)` + `lm_head(V,H)` + 分词器缓存。
2. **token-free 路线已有雏形。** `native_mm.py` 已实现"文本→字形位图→GlyphEncoder→连续向量"与"连续输出→最近字形匹配解码"，`config.py` 已有完整原生多模态配置（`get_native_moe_config`）。**v2 方案不是从零发明，而是把这条实验线扶正为主线。**
3. **当前长上下文能力是硬瓶颈。** `SelfAttention.max_seq_len` 上限 4096，标准 KV Cache 在 1M 上下文下需 ~96GB 显存（见 §4.4 账目）；DMC 记忆库虽是常数大小但与序列位置无关、无法承担远程精确检索。
4. **断链未修。** `SimplCore.knowledge.knowledge_graph.KnowledgeReasoner` 导入失败问题仍在。

## 1.3 必须先立的三条规矩

1. **结论由度量生成**：所有报告/JSON 的结论文案必须由实测指标布尔量自动生成，禁止写死。
2. **基准先行**：任何"显著优势"声明必须附带 RULER / NIAH / LongBench 实测数据与显存实测曲线。
3. **单一主线**：归档 `SimplCore.py`、`simplcore/`（旧栈）、`java-high-performance/` 与根目录一次性脚本，只保留 `simplcore_pg/` 作为 v2 基座。

---

# 第二部分　重构目标与一个诚实的澄清

## 2.1 目标

- **G1 无词表**：删除 `nn.Embedding` / `lm_head` / 分词器，输入输出全程连续向量。
- **G2 1M 上下文起步**：推理显存满足 **M(L) = M_base + 1GB × log₂(L / 1M)**；单步解码显存与 L 完全无关。
- **G3 训练可行**：1M 上下文训练的激活显存与 L 无关（分块 + 状态传递 + 重计算）。

## 2.2 诚实的澄清：严格 O(1) 与"每翻倍 +1GB"是矛盾的

若显存严格 O(1)（翻倍零增长），则远程状态必须是**固定大小**的——信息论上这意味着对无限上下文**必然有损**（有限维状态无法无损存储任意长历史）。"每翻倍 +1GB"实际是 **O(log L)** 增长。

本方案采用如下诚实且可达的口径：

- **单步解码显存：严格 O(1)**——生成第 t 个符号所需显存不随已读上下文长度变化；
- **总上下文存储：O(log L)，每翻倍恰好 +1GB**——通过"八度分层压缩记忆"实现，每一层固定预算 1GB；
- **代价**：超远程信息是有损压缩检索，精确复制型任务（如 passkey 原值复述）需配合"重要段不压缩"策略（§4.5）。

这是本方案与市面宣传的本质区别：**我们明说有损，并用基准量化损失。**

---

# 第三部分　无词表方案：GlyphStream 连续符号接口

## 3.1 设计总览

```
原始输入 ──┬── 文本: 字符 → 字形位图(32×32) → GlyphPatch-ViT → 连续向量/字符
           ├── 图像: 像素 → ImageViT → PerceptualResampler → K 个感知向量
           ├── 视频: 帧   → VideoEncoder → Resampler → K 个感知向量
           └── 任意Unicode兜底: UTF-8 字节块 → BytePatch 编码器 → 连续向量
                                    │
                          Glyph Pooling (动态合并)
                                    ▼
                    连续"语义单元"流 (B, L, H) → TriMemory 推理核
                                    ▼
        输出: ContinuousTextHead → 最近原型匹配 + 轻量判别器 → 字符/图像
```

## 3.2 输入侧：三层通道

**通道 A　字形通道（主通道，沿用并强化 `perception.py` 已有组件）**
- 文本逐字符渲染为 32×32 灰度字形位图（字体无关渲染器，CJK 扩展区经系统字体回退，覆盖率 100%）；
- GlyphPatch-ViT（patch 8×8、6 层、512 维）输出**每字符 1 个连续向量**；
- 归纳偏置：字形相近的字向量天然相近——对中文"形声字/部首"是免费的知识注入，这是离散词表永远给不了的。

**通道 B　字节兜底通道（BLT 式动态分块）**
- 渲染失败/控制字符/罕见符号 → UTF-8 原始字节 → BytePatch 编码器（1D 卷积 + 局部注意力）；
- 熵驱动动态分块（参考 Meta BLT）：用小模型预测下一字节熵，熵峰处切分，实现"难字多分块、常见串少分块"的算力自适应。

**通道 C　视觉通道（已有）**：图像/视频走现有 `ImageEncoder`/`VideoEncoder` + `PerceptualResampler`。

## 3.3 Glyph Pooling：解决字符级序列过长的关键

字符级序列长度约为 subword 的 3–4 倍，直接进推理核会把 1M 上下文变成 3–4M。采用**两级动态池化**：

1. **熵驱动合并**：低熵连续字符（常见词"中国""人工智能"）合并为一个语义单元，目标压缩率 2.5–3.5×，使有效序列长度回到 subword 同级；
2. **可学习池化头**：cross-attention perceiver（沿用 `PerceptualResampler` 结构），可微、端到端与主模型联合训练。

合并边界由数据驱动学出，不依赖任何人工词表——**这就是"词表"的替代物：边界是学出来的连续决策，不是查出来的离散条目。**

## 3.4 输出侧：连续解码

- `ContinuousTextHead` 预测 H 维连续向量；
- **可学习字形原型库**（~6000 原型 × H 维，是参数而非查表字典）：内积检索 + temperature softmax 训练（对比学习目标），推理时最近邻匹配出字符；
- **轻量判别器消歧**：对 top-8 候选原型用 2 层 Transformer 结合上下文重排，解决同形/近形字歧义；
- 生僻字符兜底：输出字形位图本身（扩散式位图解码头，32×32 小图，仅 4 步去噪），保证 100% 字符覆盖。

## 3.5 消除的组件与收益

| 消除项 | 参数量/显存（128K 词表 × H=4096） |
|--------|--------------------------------|
| `nn.Embedding(V, H)` | 0.52B 参数 ≈ 1.0GB (bf16) |
| `lm_head`（未绑定时） | 再 0.52B ≈ 1.0GB |
| 分词器、`tokenized_cache/`、词表文件、`tensorflow_backup_*.zip` | 磁盘 + 依赖链 |
| **合计释放** | **~2GB 显存 + 整条分词依赖链** |

新增：GlyphEncoder ~50M + 原型库 6000×4096 ≈ 25M + 判别器 ~30M ≈ **0.1B**，净省约 0.9B 参数。

## 3.6 迁移策略：教师蒸馏过渡

v2 训练初期保留现有 token 路径模型作为**蒸馏教师**（KL 对齐 logits 层等价物：对齐最终隐状态与原型分布），待 GlyphStream 在内部困惑度评测上追平 ±2% 后，物理删除 token 路径。`SimplCoreConfig` 中 `use_native_multimodal=True` 成为唯一支持的模式。

---

# 第四部分　长上下文核心：TriMemory 三级记忆架构

## 4.1 总体结构

每个 Block 内不再使用全序列自注意力，而是三级记忆协同：

```
L0 精确近程: Sliding Window Attention (窗口 W=8192, GQA, RoPE)
     └─ 精确、无损，KV Cache 恒定 ~0.77GB
L1 线性中程: Gated DeltaNet 线性循环层 (chunked scan)
     └─ 状态 (dk×dv)/头，恒定 ~50MB，读写 O(1)/步
L2 压缩远程: Octave Memory 八度分层压缩记忆
     └─ 每层固定 1GB 预算，总预算 = 1GB × log₂(L/1M)
```

Block 排布：`(SWA + DeltaNet) × 3 → Octave 读取层` 循环，即每 4 层做一次跨层远程检索。

## 4.2 L0：滑动窗口注意力（SWA）

- 窗口 W = 8192，GQA（8 KV 头，head_dim 128），RoPE 仅在窗口内使用——**不存在位置外推问题**，因为 RoPE 永远只见 0–8191；
- KV Cache 大小恒定：8192 × 96KB/token ≈ **0.77GB**（FP16），与 L 无关；
- 职责：近程精确依赖（语法、局部指代、代码块）。

## 4.3 L1：Gated DeltaNet 线性循环层

采用 gated delta rule（ΔNet/Mamba-2 一系）替代 SGC 谱全局卷积承担中程记忆：

```
S_t = S_{t-1} ⊙ (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ      (状态更新, 可学习遗忘门 β)
o_t = S_t q_t                                            (读取)
```

- 每头状态 dk×dv = 128×128，16 头 × 24 层 × FP16 ≈ **50MB，恒定**；
- 每步读写 O(dk·dv)，**单步显存/算力与 t 无关——这是"单步 O(1)"的来源**；
- 训练用 chunked scan（块 64）并行化，吞吐与全注意力同量级；
- 它同时**吸收并取代**现有 SGC 谱全局卷积的角色（频域全局混合 ↔ 线性递归全局混合），旧 SGC 保留为可选研究分支，不进主线。

## 4.4 L2：Octave Memory 八度分层压缩记忆（"每翻倍 +1GB"的实现）

核心思想：远程记忆按"八度"（log₂ 尺度）分层，每层**固定槽数**，层满即向上一层再压缩——显存随 log₂L 线性增长，每翻倍恰好新增一层（+1GB）。

**写入（压缩）**
- 每 4096 token 段 → 段压缩器（2 层 perceiver）→ 32 个 memory token（**128× 压缩**）→ 写入第 0 层；
- 第 k 层槽满（4096 槽）→ 对其内容再 1/2 压缩推入第 k+1 层，第 k 层清空滚动；
- 第 k 层覆盖 2^k × 1M token 的历史区间，分辨率随距离指数衰减——近处清晰、远处保留"要旨"。

**读取**
- 每层 attention 块中，query 对所有八度层做 cross-attention（总槽数 = 4096 × 层数，1M 时仅 4096，10 亿 token 时也仅 ~4 万槽，远小于全序列）；
- 层级偏置：近层注意力偏置高，模拟记忆衰退曲线，偏置斜率可学习。

**预算账目（参考配置 QXT-Long-56B：H=4096，24 层，MoE 7×7.8B，GQA-8）**

| 上下文 L | 基线全注意力 KV (FP16) | 本方案：SWA + Δ状态 + Octave | 对比 |
|---------|----------------------|------------------------------|------|
| 1M | ~96 GB | 0.77 + 0.05 + 1.0 ≈ **1.8 GB** | 53× |
| 2M | ~192 GB | ≈ **2.8 GB** | 69× |
| 4M | ~384 GB | ≈ **3.8 GB** | 101× |
| 16M | ~1.5 TB | ≈ **5.8 GB** | 265× |
| 1B (10⁹) | ~96 TB | ≈ **11.8 GB** | ~8000× |

每翻倍增量恒为 **+1.0GB**（新八度层），满足 G2。加上 int4 权重（56B ≈ 28GB），**单张 48GB 级显卡即可推理 1M 上下文**。

**存储格式**：memory token FP8-E4M3 + per-channel scale，paged 显存管理（借鉴 vLLM PagedAttention 的块表机制，按八度层组织）。

## 4.5 有损压缩的对策：显著性保护

有损是 Octave Memory 的固有属性，用机制而非宣传来兜底：

1. **显著性不写压**：段压缩器输出显著性分数（ surprisal + 检索频率），高分段（如密钥、精确数字、用户显式标记）以 FP16 原样保留在"受保护区"（固定 128MB 预算，LRU）；
2. **训练目标显式建模**：在训练数据中合成"远程精确检索"任务（NIAH 变体），让压缩器学会保关键值；
3. **评测公开损失**：RULER 上按距离分桶报告召回率，不藏损失曲线。

## 4.6 训练显存：分块 TBPTT + 选择性重计算

1M 上下文训练不可能一次性前向，采用：

- **序列切块 C=16K**，块间传递 detached 的（Δ状态 + Octave bank + 保护区索引），即沿序列的 TBPTT；反向传播只在块内 + 跨块状态用**重计算**补梯度（每块多算一次前向，省存全部中间激活）；
- 激活显存 ≈ 24 层 × 16K × H × ~16B/token/layer ≈ **~8GB，与 L 无关**；
- 状态蒸馏：短上下文全注意力教师 → 对齐 Octave 检索分布与 ΔNet 状态轨迹，加速长上下文能力收敛；
- 课程式扩展：32K → 256K → 1M → 4M，每级跑通 NIAH ≥ 95% 才进下一级。

---

# 第五部分　内核与系统实现

| 组件 | 实现 | 说明 |
|------|------|------|
| ΔNet chunked scan | Triton kernel（参考 FLA 库 `chunk_gated_delta_rule`） | 训练并行扫描 |
| SWA | FlashAttention-3 varlen + sliding window | 直接复用 |
| Octave 压缩器 | 复用现有 PerceptualResampler 结构 | 代码资产复用 |
| FP8 存储/读取 | CUDA kernel + Paged 块表 | per-channel scale |
| Glyph 渲染 | 离线预渲染缓存 + 在线 FreeType 回退 | 渲染不进热路径 |
| 图模式 | 维持 `torch.compile`/`torch.export` 兼容约束 | 沿用 `model.py` 现有符号形状纪律 |

现有代码的 `_rope_table` 符号形状处理、`MemoryState` 运行时记忆模式与训练态完全可微的分离设计，都是可以直接继承的好基础。

---

# 第六部分　迁移路线与里程碑

**P0　清理与基线（2 周）**
- 归档旧栈（`SimplCore.py`、`simplcore/`、`java-high-performance/`、根目录一次性脚本、`.pt/.zip/.log` 历史产物）；
- 修复 `KnowledgeReasoner` 断链；报告结论一律改为度量自动生成；
- 建立 eval harness：RULER / NIAH / LongBench + 显存实测脚本（`torch.cuda.max_memory_allocated` 曲线）。

**P1　GlyphStream 替代词表（4–6 周）**
- 完成 GlyphEncoder/Pooling/原型解码头；token 教师蒸馏；
- 验收：内部困惑度差 ≤2%，OOV=0，删除 `nn.Embedding`/`lm_head`/`tokenized_cache`。

**P2　TriMemory 核心（6–8 周）**
- SWA + ΔNet 先上，32K 全量训练；Octave Memory 随后；
- 验收：256K NIAH ≥ 95%，实测显存曲线符合 log 公式（实测点与预测误差 <10%）。

**P3　1M 端到端与公开基准（4 周）**
- 课程扩展至 1M；RULER-1M / LongBench 全量；
- 验收：显存公式 M(L)=1.8GB+1GB·log₂(L/1M) 实测成立；基准数据随发布公开。

---

# 第七部分　风险与诚实声明

1. **远程精确召回有理论上限**：固定预算的压缩记忆无法保证任意远程 token 的无损召回。本方案用"显著性保护区 + 分桶召回率公开"来量化并缓解，而不是宣称"无限无损记忆"。
2. **GlyphStream 的已知难点**：同形字消歧、极长英文/代码序列的池化效率、渲染字体差异——均有兜底通道，但 P1 阶段必须用数据证明困惑度追平，否则回退教师蒸馏继续迭代，不硬上。
3. **"O(1)"的准确表述**：严格 O(1) 的是**单步解码显存**；总上下文存储是 O(log L)。对外文档必须使用此准确表述。
4. **训练成本现实**：56B MoE 的 1M 课程训练是多卡工程（ZeRO-3 + 序列并行），单机 4090 只能跑 P0/P1 的小尺度验证；方案文档不再出现"33B/4090 单卡训练"式的不实表述。

---

# 附录 A　新旧组件映射

| 旧组件 | 去向 |
|--------|------|
| `nn.Embedding` + `lm_head` + 分词器 | **删除**，由 GlyphStream（A/B/C 通道 + 原型解码）取代 |
| `tokenized_cache/`、`tensorflow_backup_*.zip` | **物理删除**（P1 验收后） |
| `SelfAttention`（全序列） | 改造为 SWA（窗口 8192），RoPE 逻辑保留 |
| `SpectralGlobalConv` (SGC) | 主线由 ΔNet 取代；保留为可选研究分支 |
| `DynamicMemoryChannels` (DMC) | 概念并入 Octave Memory（可微记忆库 ↔ 压缩记忆槽） |
| `PerceptualResampler` | 复用为段压缩器与 Glyph Pooling |
| `TopologyAwareRouter` (CCR) | 保留，用于 MoE 路由与 Octave 层选择偏置 |
| `MemoryState` 运行时/训练态分离 | 保留，扩展为（Δ状态, Octave bank, 保护区）三元组 |

# 附录 B　关键公式速查

- 推理总显存：`M(L) = 0.77GB (SWA) + 0.05GB (Δ) + 1.0GB × log₂(L / 1M) (Octave) + 保护区 0.13GB`
- 单步解码显存：`O(W·d_kv + d_k·d_v·heads + slots·d_mem)`，**与 L 无关**
- 训练激活显存：`≈ 24层 × C(16K) × H × 16B`，**与 L 无关**
