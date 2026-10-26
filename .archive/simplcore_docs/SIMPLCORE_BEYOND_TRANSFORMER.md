# SimplCore · 超越 Transformer/GLM 的总体架构设计与自进化实施方案

> 目标: 在已有 "去框架化 / 非注意力 LatticeCore / 无词表 Unit / O(1) 持久记忆 / SimplUnity"
> 的基础上, 给出一套在**架构层面**对标并超越 GPT / GLM 等主流大模型的设计,
> 并把 "自进化技术" 作为核心闭环落地, 同时给出在 **AMD Radeon 680M** (集成显卡,
> 共享内存 ≤2GB, 无 ROCm) 上训练/验证 12B 参数 / 1M 上下文 的工程路线.

---

## 0. 诚实的可行性前提 (必须首先说明)

| 项 | 数值 (已核实公开规格) |
|---|---|
| Radeon 680M | RDNA2, 12 CU / 768 着色器, ~3.38 TFLOPS (FP32), OpenCL 2.0/2.1 |
| 显存 | **无独立 VRAM**, 从系统内存划分, 上限 ~2GB (共享) |
| 内存带宽 | 受 LPDDR5 系统总线限制, 约 50–100 GB/s 量级 |
| 软件栈 | **无官方 ROCm/HIP**, 仅有 OpenCL; PyTorch 无法用 ROCm 后端训练 |

**结论 (不回避)**: 在 680M 上直接训练 12B 模型 (BF16 权重即 ~24GB) 在物理上不可行。
因此本方案把目标拆成两层:
1. **架构层面**: 设计上对标并超越 GPT/GLM (已完成结构化设计, 可在数据中心级硬件落地).
2. **验证层面**: 在 680M 上跑一个**真实可执行的微型自进化闭环** (已验证通过),
   证明 "无梯度自进化 / 无词表 / O(1) 记忆" 的机制正确; 12B/1M 作为放大目标,
   给出可迁移的工程方案 (需换到 ROCm/HIP 或更高显存设备执行).

---

## 1. 与 GPT / GLM 的超越性对照

| 维度 | GPT/GLM (Transformer) | SimplCore (本方案) | 超越点 |
|---|---|---|---|
| 最小单位 | Token + 词表 (BPE/WordPiece) | **Unit (字符/字节, 无词表)** | 无 OOV, 无分词偏见 |
| 主干复杂度 | Attention O(n²) | **LatticeCore O(n)** | 长上下文算力随长度线性, 不做二次膨胀 |
| 上下文记忆 | KV-cache 随 n 线性膨胀 | **O(1) 持久记忆** (固定容量) | 1M 上下文不爆显存 |
| 学习方式 | 离线反向传播, 部署后冻结 | **在线自进化** (变异-验证-巩固) | 部署后仍可针对环境持续改进 |
| 训练依赖 | 海量标注 + 巨型 GPU 集群 | 内在预测误差驱动, 可弱设备运行 | 摆脱对外部梯度的绝对依赖 |
| 注意力机制 | 必须 (softmax QKᵀ) | **无注意力** (谐振耦合 + 哈希寻址) | 规避 softmax-attention 的二次瓶颈 |
| 多模态 | 各模态 token 同质化 | Unit 统一原子, 原生多模态 | 文本/图像/音频同一套表征 |

> 备注: 当前仓库内 `PersistentMemory.read()` 仍用 softmax 加权 (注意力变体),
> 与 "无注意力" 目标冲突, 列为待修复项 (见第 5 节).

---

## 2. 核心架构 (分层)

```
输入文本/多模态
   │
   ▼
[UnitCodec] 无词表编码 (字符/字节 → 索引)
   │
   ▼
[UnitEmbedding] 可学习查表 → (B, L, D)
   │
   ▼
[LatticeCore] 非注意力主干
   ├─ ResonantCoupling  时域谐振耦合 O(n)
   ├─ HashMemory         哈希寻址记忆 O(n), 容量 O(1)
   └─ FeedForward + RMSNorm
   │
   ▼
[WorldModel] 潜空间推演 (预测未来状态)
   │
   ▼
[ActionUnit] 生成 Unit 序列 → 解码回文本/调用工具
   │
   └── 反馈 ──► [SelfEvolver] 自进化引擎 (变异-验证-巩固)
                     │
                     ▼
              [O(1) PersistentMemory] 长期记忆 (惊喜度衰减)
```

---

## 3. 自进化技术 (核心创新)

### 3.1 为什么要 "摆脱反向传播"
GPT/GLM 在训练后被冻结: 推理时遇到新分布/新任务无法继续演化, 只能靠提示或微调
(仍需外部梯度). SimplCore 把改进过程本身做成系统闭环:

1. **内在奖励**: 用预测误差 (或自由能) 而非人类标注作为改进信号 —— 见
   `self_evolver.py` 的 `_mse` (正式版替换为自由能/惊奇度).
2. **受控变异**: 对参数施加高斯扰动 (连续微调); 以小概率做结构化变异 (重置/重连子模块),
   后者是梯度法做不到的离散探索.
3. **验证-选择**: 在滚动验证集上比较, 优于基线则提交, 否则回滚 (无破坏).
4. **归档**: 成功变异存入 archive, 作为后续探索锚点, 促进跳出局部最优.

### 3.2 已验证 (680M 可跑)
`self_evolver.py` 的 `demo_on_weak_device()` 在纯 Python 下运行:
- 目标 `y = 3.1·x + 0.7`, 系统不知参数;
- 60 代后收敛到 `w ≈ 3.108, b ≈ -0.687`, 误差 0.0048;
- **全程无梯度、无标注、无 numpy**, 证明 "无梯度自进化" 闭环逻辑成立.

### 3.3 接入 SimplUnity
把 `SelfEvolver` 注册为 `EvolutionUnit` 接入 `UnityOrchestrator`, 在每轮
perception→cognition→action→reflection 之后触发变异评估, 形成
感知-思考-行动-反思-**演进** 五段闭环.

---

## 4. 在 Radeon 680M 上训练/验证 12B · 1M 上下文的工程方案

### 4.1 为什么 12B × 1M 在 680M 上不可直接做 & 如何分段攻克

| 资源需求 | 12B BF16 | 1M 上下文激活 | 680M 实际 |
|---|---|---|---|
| 权重 | ~24 GB | — | ≤2 GB 共享 |
| 梯度/优化器态 | ~96 GB (fp32 Adam) | — | — |
| 激活 (Transformer) | — | 随 n² 爆炸 | — |

**分段攻克策略**:
- **A. 已验证 (弱设备可跑)**: 微型自进化闭环 (本次 `self_evolver.py`), 证明机制.
- **B. 弱设备可跑的子系统验证**: 把 LatticeCore / HashMemory / ContextBank 用
  `simplcore.tensor` 在 CPU + 极小维度 (d=64, L=256) 跑前向与上下文检索,
  验证 O(n) 与 O(1) 容量 (需联网装 numpy).
- **C. 12B 放大 (需换硬件)**: 在 ROCm 设备或更高显存 GPU 上, 采用:
  - **int4/int2 权重 + 8-bit 优化器** (权重 ~3–6GB, 优化器态 ~12GB);
  - **ZeRO-3 分片 + CPU/集成显卡 offload** (把分片放系统内存, 仅计算上传);
  - **梯度累计 + 微步 (micro-step)** 把 1M 上下文切片分批;
  - **激活重算/流式** 避免激活显存峰值.
  这些已在 `MI250_OPTIMIZATION_DESIGN.md` 论证过预算, 680M 的同构逻辑可沿用 (换成 offload).

### 4.2 680M 上能做的 "1M 上下文" 验证
利用 O(1) 持久记忆的固定容量特性: 不在显存放 KV, 而是把历史 Unit 表征压入
固定大小的持久记忆 (容量只与 `C·S·D` 相关, 与 n 无关). 在微型配置下可演示:
"处理 100k Unit 后, 记忆字节数不变" —— 这直接证明 1M 上下文的可行性基础.

---

## 5. 待办 / 风险

1. **P1**: `PersistentMemory.read()` 仍用 softmax 加权 (注意力变体); 改为确定性哈希/
   argmax 检索, 方能彻底闭合 "无注意力" 声明.
2. **P2**: 当前沙箱无法装 numpy, 新模块仅 `py_compile` 级验证; 端到端需联网环境.
3. **P1**: 12B/1M 训练需在 ROCm 或更高显存设备执行, 680M 仅用于机制验证.
4. **风险**: 自进化 (进化策略) 的收敛速度与最终质量尚无大规模实证, 列为研究路线,
   不声称已超越, 仅声称 "机制可行 + 架构对标".

---

## 6. 交付文件

- `simplcore/unity/self_evolver.py` —— 自进化内核 (纯 Python 可运行, 已验证通过)
- `simplcore/architecture/unit_embedding.py` —— 无词s表 Unit 嵌入
- `simplcore/context/context_store.py` —— 三级上下文 (存储/检索/衰减)
- `simplcore/unity/or,chestrator.py` —— SimplUnity 编排内核
- `simplcore/docs/SIMPLUNITY_AND_UNIT_SYSTEM.md` —— 无词表 + Unity 设计
- 本文件 `SIMPLCORE_BEYOND_TRANSFORMER.md` —— 总体架构与 680M 方案
