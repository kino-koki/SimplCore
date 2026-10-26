# SimplCore 56B 中文 MoE 训练蓝图

> 目标：用 SimplCore 架构训练一个 56B 参数、纯中文语料、7 专家 (每专家角色 ~7.8B、top-2 激活) 的 MoE 大模型，
> 在中文能力上对标并力争超越当前前沿模型 (注：公开资料中并无 "Claude Mythos5" 这一型号，本文以
> C-Eval / CMMLU / CLUE / CSDial 等中文基准作为可量化对照线)。

---

## 0. 现实边界 (先说清楚)

- 从零训练 56B 需要 **数千卡 GPU × 数月**（数据、算力、工程三线并行）。本仓库交付的是
  **可直接在集群 `torchrun` 跑、且在 CPU 小尺度已验证正确的完整代码与配置**，而非已训好的权重。
- 代码已在 **torch 2.1.2+cpu (Windows)** 上验证：56B 参数账命中、MoE 前向/反向/aux loss/生成/FX 追踪
  全部通过，80 个单元测试全绿，训练与评估冒烟脚本可运行。
- "超越" 必须靠**数据 + 配方 + 后训练 + 评估闭环**持续迭代，单靠架构不足以断言超越。

---

## 1. 架构缩放 (已落地在 `simplcore_pg/`)

### 1.1 参数账 (精确，见 `config.get_56b_moe_config`)

| 组件 | 参数量 |
|---|---|
| 嵌入 (vocab=128000, H=4096, 权重绑定) | 0.524B |
| 共享注意力 (14 层 × 4H²+H) | 0.940B |
| 专家 FFN (7 专家 × 14 层 × 3·H·45344) | 54.604B |
| 路由 (7×H×14) | 0.0004B |
| **合计** | **56.07B** |

- 每专家角色跨 14 层累计 **7.80B**（与 "7 个专家、每个 7.8B" 对齐）。
- 每 token 激活 2/7 专家 → 激活专家参数 ~15.6B，推理算力省 **~3.5×**。
- **生产级默认启用 SimplCore 自研架构**：每个 MoE block 在注意力之后、MoE 之前
  插入 **共振场处理 RFP（谱全局卷积, O(n log n), 可训练）** 与 **动态记忆通道 DMC
  （内容寻址外部记忆, 可训练）**；二者参数量已计入 56.07B 总账（见
  `config.estimate_parameters` MoE 口径）。主干同时复用 SimplCore 的
  RMSNorm / RoPE / SDPA / SwiGLU。

### 1.2 核心文件

- `simplcore_pg/moe.py`：`MoELayer`(容量调度 + 负载均衡 loss + router z-loss)、`ExpertFFN`、`SharedExpert`。
- `simplcore_pg/model_moe.py`：`SimplCoreMoEModel`(注意力 + MoE Block、aux 聚合、generate、FX trace)。
- `simplcore_pg/config.py`：MoE 字段、`get_56b_moe_config()`、`estimate_parameters()` 精确账。

---

## 2. 纯中文语料策略 (决定中文上限的关键杠杆)

1. **中文优化分词器**：vocab 含大量汉字与中文词 (参考训练好的 BPE / sentencepiece)。
   中文 token 膨胀率每降 30%，同算力下可多学 ~30% 中文上下文——这是性价比第一杠杆。
   (`training/data.py` 的 `SimpleChineseTokenizer` 是可替换接口，生产应接入真实中文 BPE。)
2. **语料配方 (纯中文为主)**：
   - 中文百科 / 书籍 / 高质量网页 (严格质量过滤 + 全局去重)；
   - 中文代码与数学 (抬升推理)；
   - 少量平行语料做对齐 (不影响"纯中文"主体)。
3. **课程式训练**：通用语料 → 长上下文 (4K→32K+ 渐进) → 指令阶段。
4. **质量管线**：`IterableCorpus` 提供 `quality_filter` / `dedup` 可插拔钩子。

---

## 3. 训练配方 (56B 必须分布式)

- **分片**：专家并行 (EP) + 张量并行 (TP) + FSDP/ZeRO-3 (`distributed.py` 提供规划与 wrap 策略)。
  56B 在 8×80GB GPU 上经 EP+FSDP 可分片容纳。
- **数值**：bf16 混合精度 + 梯度检查点 (`model_moe.gradient_checkpointing`) + 梯度裁剪。
- **稳定性**：负载均衡 loss (系数 0.01) + router z-loss (1e-4) 防路由归一化。
- **调度**：warmup + cosine，峰值 LR ~3e-4。
- 启动：`torchrun --nproc_per_node=8 simplcore_pg/training/train.py --preset 56b-moe ...`
  (`distributed.launch_torchrun_note()` 给出模板)。

---

## 4. 后训练对齐 (中文能力超车段)

- 中文 SFT (指令、对话、长文写作、古文、诗词、方言理解)；
- 中文偏好数据 DPO / RLHF，专项抬升中文推理与安全性；
- 继续在 C-Eval/CMMLU 风格基准上做错误分析驱动的数据补强。

---

## 5. 评估对照 (`eval/eval_chinese.py`)

- few-shot 选择题评测，支持加载真实 C-Eval/CMMLU JSONL，输出**总体与各科目准确率**；
- 内置小样本集用于离线冒烟；加载真实基准后可对照前沿模型分数线；
- "超越" 的判据：在同等 few-shot 设置下，中文综合分高于对照模型且在分科目上无显著短板。

---

## 6. 验证状态 (本机 torch 2.1.2+cpu)

- ✅ 56B 参数量估算 56.07B，每专家角色 7.80B；
- ✅ MoE 前向 / 反向 / aux loss>0 / 生成 / FX 追踪；
- ✅ 容量调度与无丢 token 时"按权重聚合"数值等价 (`test_moe_dispatch_matches_dense_when_no_drop`)；
- ✅ 80 个单元测试全通过；
- ✅ 训练冒烟 (loss 下降、aux 正常)、评估冒烟 (harness 跑通) 可运行；
- ⏳ 真机 56B 训练 / 真实中文基准分数线：需在 GPU 集群执行，不在本机范围。

---

## 7. 后续批次 (对齐原仓库其余 476 文件)

本交付是**核心架构的生产级标杆 + 56B MoE 扩展标准**。原仓库的认知/知识库/分布式训练脚本/Java 层等
外围模块，建议按此标准逐批对齐（统一配置、异常体系、测试与文档规范），而非单轮重写。
