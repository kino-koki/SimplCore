# 青小团多模态语言大模型 — 世界模型 (World Model) 架构设计

> 本文深化「世界模型」概念，并将其与青小团 (Qingxiaotuan) 主干架构的工程实现对齐。已实现于 `simplcore/architecture/simplcore_world_model.py`、`world_model.py`、`rollout_engine.py`。

---

## 1. 什么是世界模型，以及为什么需要它

标准自回归语言模型只做一件事：给定上文，预测下一个 token。它不显式建模「世界如何变化」。世界模型 (World Model) 把模型从「下一个词预测器」升级为「环境模拟器」：

- **状态 (State)** 显式表示「此刻世界是什么」
- **转移 (Transition)** 显式建模「动作/输入如何让状态演化」
- **预测 (Prediction)** 在潜空间推演未来，而非只在 token 空间采样

这与人类「在脑中预演」的能力同源：你看到乌云（状态），想到「要下雨」（转移），预判「地面会湿」（预测），再决定「带伞」（动作）。

---

## 2. 状态表示与环境建模

青小团世界模型的状态是一个 **潜空间向量**，而非 raw token。它由三部分耦合：

| 维度 | 来源 | 作用 |
|------|------|------|
| 语义状态 zₛ | 文本/多模态编码 | 语言与概念 |
| 物理状态 zₚ | 动力学模型隐变量 | 可推演的「世界规律」 |
| 记忆状态 m | 持久记忆 (O(1)) | 跨步/跨会话不丢失的长期上下文 |

状态不随序列长度增长（潜空间固定维度），因此 1B token 上下文只需 O(1) 记忆（见 MI250 文档 §3）。

---

## 3. 预测与规划嵌入主干

青小团的做法不是「先语言、后规划」两段式，而是把预测/规划 **嵌进主干**：

```
输入 (文本/图像/视频)
      │
      ▼
  [多模态路由 ModalityRouter]
      │  分派到文本 / 视觉 / 动作路径
      ▼
  [SimplCoreModel 主干] ──► 语义潜状态 zₛ
      │
      ▼
  [RolloutEngine 潜空间推演] ── 用 DynamicsModel 作转移函数, 多步精炼 z
      │
      ├─► WorldModelHead → logits (语言)
      ├─► ActionHead     → 动作 a (规划 / 决策)
      └─► VideoPredictionHead → 下一帧画面

  DynamicsModel: (zₜ, aₜ) ──► zₜ₊₁   (状态转移, 物理规律建模)
```

- **RolloutEngine** 在潜空间反复调用 SimplCoreBlock，参数共享地精炼状态（类「潜空间思维链」）。支持固定步数推演与自适应提前停止（潜状态变化幅度 < 阈值即停），并按 `detach_each_step` 截断 BPTT 节省显存。
- **DynamicsModel** 显式学习 `f(z, a) → z'`，支持确定性 / 概率性（均值+方差）/ 多模态（多个可能未来）三种预测。

---

## 4. 与语言 / 多模态的协同

- **语言**：主干仍是 SimplCoreModel（SGC→DMC→CCR→FFN），世界模型在其上组合「环境建模」分支。
- **多模态**：ModalityRouter 把图像/视频经 VisionEncoder 投影进同一潜空间，使「看」与「想」共享表征。
- **记忆**：持久记忆 (`simplcore.memory.PersistentMemory`) 提供长期状态锚点，使推演可跨回合延续，不被上下文窗口截断。

---

## 5. 训练目标与数据方案

### 训练目标（多目标联合）
1. **语言建模损失** `L_LM`：标准下一 token 交叉熵（主干输出）。
2. **动力学一致性损失** `L_dyn`：预测状态 `ẑₜ₊₁` 与真实下一状态 `zₜ₊₁` 的 MSE/对比损失。
3. **奖励/连续性预测** `* L_head`：ActionHead 的 reward、VideoPredictionHead 的 continuity 监督（若有环境反馈）。
4. **对比正则** `L_contrast`：improved_world_model 中引入的自监督对比学习，抑制过拟合。

总损失：`L = L_LM + λ₁·L_dyn + λ₂·L_head + λ₃·L_contrast`。

### 数据方案
- **文本**：大规模语料 + 自监督下一句/状态预测。
- **多模态对齐**：图文/视频-文本对，用于 ModalityRouter 与 VideoPredictionHead。
- **交互/轨迹**：(状态, 动作, 下一状态) 三元组，用于 DynamicsModel 的监督或强化学习。
- **无限自迭代**：improved_world_model 支持自举式数据生成与再训练（课程式扩张）。

---

## 6. 可量化评估指标

| 维度 | 指标 | 说明 |
|------|------|------|
| 语言 | perplexity / 下游任务准确率 | 主干能力不退化 |
| 动力学 | 状态预测 MSE / 多步误差累积 | 转移函数准确性 |
| 规划 | 任务成功率（goal-conditioned） | ActionHead 有效性 |
| 视频 | FVD / PSNR | VideoPredictionHead 质量 |
| 记忆 | 跨会话召回率 / 利用率 | PersistentMemory 长期保持 |
| 效率 | 推理延迟、显存占用 | 与标准 LLM 对比 |

---

## 7. 结论

青小团世界模型把「状态—转移—预测—动作」四个要素统一进一个主干，并用 **O(1) 持久记忆** 解决长程依赖。它与自研张量引擎、去框架化分布式、MI250 优化方案共同构成一个可落地、可验证、不依赖外部深度学习框架的完整系统。
