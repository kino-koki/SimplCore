# Dreaming 模式与 CMT 连续知识学习 —— SGC 架构层设计

> 配套阅读：`docs/SGC_SPEC.md`（算子规格）、`docs/ARCHITECTURE_PIVOT.md`（总路线）、
> `.learnings/2026-10-26-architecture-pivot.md`（决策理由）。
> 本文所有"已实现"条目均有守护测试；"待验证"条目明确标注。

---

## 0. 现状体检：机制早就有，缺的是与 SGC 的连接

指令要求"加入 Dreaming 模式（模型自进化反思）+ CMT 精华 + 完善 loop 循环"。
勘察结果与前两次指令一致 —— **机制已存在，但长在旧引擎上，没和 SGC 打通**：

| 已有资产 | 位置 | 状态 |
|---|---|---|
| 持续学习引擎（增量学习/知识蒸馏/遗忘抑制） | `simplcore/learning/continuous_learning.py` | 跑在旧 NumPy 引擎 |
| 反思模块 | `simplcore/cognition/reflexion.py` | 同上 |
| 自改进模块 | `simplcore/cognition/self_improvement.py` | 同上 |
| 元认知 | `simplcore/cognition/metacognition.py` | 同上 |
| 知识迭代 | `simplcore/cognition/knowledge_iteration.py` | 同上 |
| 自进化器 | `simplcore/unity/self_evolver.py` | 同上 |
| **缺失** | — | **SGC 架构层没有自进化循环**；记忆机制与 SGC 无关 |

**本次交付**：把 Dreaming 循环与 CMT 记忆机制**实现为 SGC 的一等架构能力**
（`simplcore/sgc/dream.py` + `simplcore/sgc/memory.py`），
而不是再造第三套机制。

---

## 1. Dreaming 与 Agent 的本质区别

| | Agent 反思 | **Dreaming** |
|---|---|---|
| 时机 | 推理期（这次回答里多想一步） | **离线**（"睡眠"阶段） |
| 作用对象 | 当前这一条回答 | **模型自身**（参数 / 记忆库） |
| 产物 | 一段更好的回答 | **训练信号**（合成语料 / 偏好对 / 课程） |
| 风险 | 低（不影响基座） | **高**（改权重会遗忘） |

Dreaming 因此必须是**带护栏的离线系统**，而不是一个 prompt 技巧。

---

## 2. 循环结构

```
        在线 (serving)                              离线 (dreaming)
        ─────────────                              ────────────────
        推理 → 轨迹入流                             HARVEST     采样高价值经验
             ↓                                    REFLECT     模型自评 → 洞见
        CMT 压缩记忆库                              SYNTHESIZE  洞见 → 训练信号
        (self-match → top-agg → 注入)               CONSOLIDATE 巩固（记忆 / 权重）
             ↑                                    VERIFY      held-out 评测
        权重 / 记忆库  ←─── COMMIT                 ROLLBACK    退化则回滚
```

### 2.1 四个阶段的具体契约

**HARVEST —— 优先级经验回放**
从经验池按价值采样（`ExperienceBuffer`）。优先级 = 事后评分；无评分时用
`(1 - confidence) + (1 - grounded)` —— **模型自己越不确信、越没有证据支撑的轨迹，
越值得拿去反思**。这正好与四层闸门的 L4 拒答门信号对接。

**REFLECT —— 反思**
可插拔钩子 `(model, trajectories) -> List[Insight]`。
生产实现应是 **Constitutional AI 式批判-修订**：模型对照一组原则批判自己的输出，
再产出修订。默认实现是**结构性占位**（只标记待反思项，不调用模型批判）。

**SYNTHESIZE —— 合成**
`(model, insights) -> List[TrainExample]`。默认实现**只在轨迹带有
`meta["corrected_response"]` 时产出样本**（通常来自真实批判模型），
否则如实返回空列表并在报告中写明 `synthesized == 0`。
**不制造假信号**是这里的硬要求。

**CONSOLIDATE —— 巩固**
四种模式：

| mode | 行为 | 遗忘风险 | 适用 |
|---|---|---|---|
| `memory` | 只更新 CMT 压缩记忆库，**基座权重冻结** | **零** | CMT 默认，生产首选 |
| `weights` | 自蒸馏写回权重 | 高 | 真正的自进化 |
| `both` | 先写记忆，再试写权重（权重部分受 gate 保护） | 中 | 过渡期 |
| `none` | 只反思不巩固 | 零 | 观察模式 / 消融 |

**VERIFY —— no-regression gate**
写回权重后必须在**独立 held-out 集**上重测；相对基线退化超过容差 → **回滚**
（权重快照恢复 + 本轮不提交）。这是自进化唯一的硬护栏。

### 2.2 硬性约束（全部有守护测试）

1. **no-regression gate**：退化即回滚（`test_dreaming_regression_triggers_rollback`）
2. **可恢复**：模型支持 `snapshot()`/`restore()` 即可回滚；不支持时权重模式自动降级为不安全并记录
3. **有预算**：`batch_size` / `max_insights` / `max_examples` / `max_rounds` 均为硬上限
4. **可观测**：每轮产出 `DreamReport`（各阶段产出计数、分数变化、耗时、notes）
5. **异常隔离**：反思/合成/巩固任一阶段抛异常 → 该轮标记 `FAILED` 并中止 run，不污染历史

---

## 3. CMT（Compression Memory Training, AAAI'25）的 SGC 原生落地

### 3.1 原文内核

CMT 的三阶段：**编码(压缩) → 存储 → 检索与聚合**。核心取舍是**基座权重冻结**，
新知识进外部记忆库，因此天然没有灾难性遗忘。三项技术：
**memory-aware objective** / **self-matching** / **top-aggregation**。

### 3.2 关键适配：注入点从 KV 换成残差流

CMT 原文把压缩记忆检索出来后，**映射成 KV 对注入 LLM 的注意力层**。
**SGC 没有注意力，也没有 KV 缓存**（流式状态是 `2·M·D` 的 O(1) 振荡器态）。
强行引入 KV 就等于把 Transformer 拽回来。

**SGC 原生的等价注入点是残差流上的加性项**：

```
x ← x + memory_in(retrieved_memory)     # 投影到模型宽度，逐层注入
```

这一路径：
* 不引入任何 softmax 注意力（守住 attention-free 主张）
* 每层成本 O(D)，不随序列长度增长
* 零初始化的 `memory_in` → **默认是恒等，基座行为完全不变**（有守护测试）

### 3.3 三项技术的落地

| CMT 技术 | SGC 落地 | 位置 |
|---|---|---|
| self-matching | 查询向量与记忆键归一化点积（余弦） | `self_match_scores` |
| top-aggregation | 取 top-k 后按相似度 softmax 加权求和 | `top_aggregate` |
| memory-aware objective | 只对"有记忆 vs 无记忆"的**增量 logits** 计损失 | `MemoryAwareObjective` |

### 3.4 与 CMT 的差异（诚实声明）

| 项 | CMT 原文 | 本实现 | 原因 |
|---|---|---|---|
| 压缩器 | 可训练的 causal Transformer decoder | 训练无关的隐状态读出 | 保持零依赖与 attention-free |
| 记忆库 | 无上限 | 定容 + LRU 驱逐 | 线上内存占用可预测 |
| 注入 | 注意力层 KV | 残差流加性项 | SGC 无 KV 缓存 |
| 记忆写入 | 外部文档流 | Dreaming 反思产出的洞见 | 本项目的自进化目标 |

**压缩器是可替换的**：`MemoryCompressor` 是独立类，换成可训练版本即可对齐 CMT 效果。

---

## 4. 可观测性与生产指标

Dreaming 是会自己改模型的系统，**必须能观测**。每轮 `DreamReport` 至少覆盖：

| 指标 | 含义 | 告警阈值（建议） |
|---|---|---|
| `harvested` | 采样到多少经验 | < batch_size/2 → 经验池枯竭 |
| `insights` | 产出多少洞见 | 连续多轮为 0 → 反思钩子失效 |
| `synthesized` | 产出多少训练样本 | **连续为 0 → 无有效学习信号，必须人工介入** |
| `score_delta` | held-out 分数变化 | 连续为负 → 暂停自进化 |
| `rolled_back` | 是否发生回滚 | 频繁回滚 → 反思策略有问题 |
| `seconds` | 单轮耗时 | 超出离线窗口 → 降频 |

---

## 5. 安全与伦理约束

1. **`grounded` 必须来自外部证据**（引文、检索命中、工具返回值、人工标注）。
   用模型自身生成的"我认为有依据"标签会退化成置信度估计的平方，四层闸门与
   Dreaming 一起失效。
2. **自进化不得自我加冕**。没有独立评测集上的显著提升，就不能说"模型变好了"。
3. **写回权重必须可回滚**，且回滚要在同一个离线窗口内完成。
4. **基座冻结是默认**（`mode="memory"`）。任何把"改权重"设为默认的改动都需要
   显式的风险评估与独立评测。
5. **反思钩子是外部输入点**。接入真实批判模型时，批判模型本身可能带偏 →
   它的输出只能作为**候选信号**，最终由 no-regression gate 与 held-out 评测裁决。

---

## 6. 已知缺口

| # | 缺口 | 影响 | 计划 |
|---|---|---|---|
| G1 | 压缩器是训练无关读出，非可训练模块 | 压缩质量弱于 CMT 原文 | 替换 `MemoryCompressor` |
| G2 | 记忆注入是**残差流加性项**，未做"记忆 vs 无记忆"的双路前向 | memory-aware objective 需要两次前向，成本 2× | `MemoryAwareObjective` 已支持，接入训练器 |
| G3 | 反思/合成默认钩子是占位实现 | `synthesized` 长期为 0 | 接入真实批判模型 |
| G4 | `gate_ab` 分支（`GatedSpectralConv2`）仍无 L2/L3 闸门 | 闸门能力在两条分支间不一致 | 与 P2 一并补齐 |
| G5 | 旧引擎侧（`simplcore/cognition/*`、`simplcore/unity/*`）未迁移 | 两套自进化机制并存 | P1 引擎合并后统一 |

---

## 7. 验收标准

| 阶段 | 验收 |
|---|---|
| **D0（本次已完成）** | Dreaming 循环与 CMT 记忆库实现为 SGC 一等能力；27 项守护测试；全量 538 项测试全绿；模型默认行为零变化 |
| D1 | 接入真实批判模型 → `synthesized > 0`；在带引用的评测集上验证反思质量 |
| D2 | 记忆注入走通端到端：文档 → 压缩 → 入库 → 检索 → 注入 → 回答质量提升（需独立评测集） |
| D3 | `mode="weights"` 在 held-out 上产生**统计显著**的提升（多轮、多 seed），且回滚率 < 5% |
| D4 | 旧引擎侧自进化机制迁移/下线，二者合一 |

---

## 8. 用法速查

```python
from simplcore.sgc import (
    CompressedMemoryBank, DreamConfig, DreamingLoop, ExperienceBuffer, Trajectory,
)

# 在线：轨迹入流 + 记忆检索
buffer.add(Trajectory(prompt=q, response=a, confidence=0.3, grounded=False))

bank = CompressedMemoryBank(dim=cfg.dim, capacity=8192)
bank.add(keys, values)                      # 压缩记忆入库
memory, scores = bank.aggregate(query_vec)  # self-match + top-agg

# 离线：跑一轮 Dreaming
loop = DreamingLoop(
    model=model, buffer=buffer, memory_bank=bank,
    config=DreamConfig(mode="memory"),       # 零遗忘风险的默认模式
    reflector=my_constitutional_critic,      # 真实批判模型
    synthesizer=my_synthesizer,
    verifier=my_heldout_eval,                # 独立评测，越高越好
)
report = loop.step()
print(report.as_dict())
```
