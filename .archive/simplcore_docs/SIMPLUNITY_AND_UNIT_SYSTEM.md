# 无词表系统 · 单元(Unit)体系 与 SimplUnity 技术体系设计

> 在 SimplCore 去框架化、非注意力架构 (LatticeCore) 与 O(1) 持久记忆之上,
> 进一步剥离对 "Token 词表" 的依赖, 并给出 SimplUnity 自主协作框架的整体设计.
> 本文档与既有 AGI_ARCHITECTURE_DESIGN.md / LATTICE_VS_TRANSFORMER.md / FORMULA_VERIFICATION.md 互补.

---

## 1. 命名重构: Token → Unit (单元)

### 1.1 为什么要改
原架构以 `vocab_size` + `Embedding(vocab_size, d)` + `lm_head` 把文本离散成
"Token id", 再映射嵌入. 这不但是 "词表" 概念, 还带来:
- **OOV / 未登录词** 问题 (词表外无法表示);
- **分词偏见** (BPE/WordPiece 合并规则影响语义切分);
- 与多模态 (图像块、音频帧) 的 "离散 token" 同质化, 失去 "无词表" 的纯粹性.

因此我们**把最小处理单位从 "Token (词表 id)" 重命名为 "Unit (单元)"**:
- 文本模态: Unit = Unicode 字符 (字素), 或下探到 UTF-8 字节;
- 多模态: Unit = 各模态的原生原子 (图像块 / 音频帧 / 传感器采样);
- 不做任何统计合并, 不维护词表.

### 1.2 术语对照表

| 旧术语 | 新术语 | 说明 |
|---|---|---|
| Token | **Unit (单元)** | 最小离散处理单位, 无词表 |
| vocab_size | `unit_table_size` | 仅字符/字节码空间规模, 非词表 |
| tokenizer.encode | `UnitCodec.encode_text` | 字符/字节确定性映射, 无训练 |
| `Embedding(vocab, d)` | `UnitEmbedding(d, mode)` | 字符/字节可学习查表 |
| `lm_head(logits→vocab)` | `UnitDecoder` | 输出字符/字节分布 |

> 注: 旧 `config.py` 的 `vocab_size / pad_token_id / eos_token_id` 保留为
> **兼容字段** (默认退化为字符级). 后续迁移策略见第 5 节.

---

## 2. 无词表架构 (Unit-centric)

### 2.1 编码入口 —— UnitCodec
字符串 → 非负整数序列, 不走任何分词器:
- `char` 模式: `ord(ch)` 直接作为索引 (覆盖 Unicode 码空间 0x110000);
- `byte` 模式: `text.encode("utf-8")` 逐字节 (规模 256).

这种方式对中文尤其友好: 每个汉字都是独立 Unit, 无 "词" 切分争议,
不存在 OOV.

### 2.2 嵌入 —— UnitEmbedding
- 可学习查表 `weight[unit_table_size, D]`, `unit_table_size` 为字符/字节规模,
  不再是统计词表.
- 前向: `weight[idx.clamp(0, size-1)]`, 支持越界容错.
- 输出 `(B, L, D)`, 与 LatticeCore / LatticeBlock 完全兼容, 可直接替换原 Embedding.

### 2.3 解码 —— UnitDecoder (规划中)
输出层从 "vocab 分布" 改为 "字符/字节分布", 用 argmax 或采样生成下一个 Unit,
再经 `UnitCodec` 反向拼回字符串. 因 Unit 是字符级, 生成单位长度更长但完全
无词表, 且与 "世界模型 / 潜空间推演" 的连续表征天然对齐.

---

## 合规性检查 (与既有约束一致)
- 不使用 self-attention / softmax 注意力 / positional encoding / FFT / PyTorch.
- 底层仍是 `simplcore.tensor` 自研后端 (零真实 torch 导入).
- 与 LatticeCore (O(n) 谐振耦合 + O(n) 哈希记忆) 协同, 不引入 n² 注意力.

---

## 3. 上下文机制: 存储 / 检索 / 衰减

### 3.1 三级记忆架构

| 层级 | 介质 | 容量 | 生命周期 | 复杂度 |
|---|---|---|---|---|
| 瞬时 (sensory) | 当前 batch 张量 | 单次前向 | 即时 | O(1) |
| 工作 (working) | 循环缓冲 `_working` | 固定 K (默认 2048) | 会话内 | O(K) |
| 长期 (long-term) | PersistentMemory (mmap) | 固定 (C,S,D) | 跨会话 | **空间 O(1) wrt n** |

长期记忆复用既有 `PersistentMemory`: 形状 `(C, S, D)` 与序列长度 n **无关**,
因此无论处理 1 个还是 1B 个 Unit, 驻留显存/内存恒定 (已在 FORMULA_VERIFICATION.md 验证).

### 3.2 检索 —— 确定性内容寻址 (无 softmax)
`ContextBank.retrieve()` 调用 `PersistentMemory.read()`, 但需注意: 现有 read
内部仍用 softmax 加权 (attention 变体). **为彻底脱离注意力**, 建议把 read 改为
argmax/gather 的确定性定位 (与 HashMemory 一致), 已在 FORMULA_VERIFICATION.md 标记为待办. 检索返回通道聚合读出, 不依赖概率打分.

### 3.3 衰减 / 巩固策略 (surprise-driven)
引入 **惊喜度 (surprise)** 信号 (= 预测误差, 来自世界模型 free-energy 项):
- 写入时: 高惊喜 (>=0.3) 的 Unit 优先进入长期记忆; 低惊喜溢出工作区时直接丢弃.
- 衰减时: 长期记忆按 `value = access_count / (1 + decay_rate · age)`
  综合价值做温和衰减 (不直接删除), 高频/高惊喜内容因反复写入而 "非遗忘".
- 反思阶段 (`ReflectionUnit`) 每轮触发一次 `decay_step()`.

---

## 4. SimplUnity 技术体系

### 4.1 定位
SimplUnity 是 SimplCore 的 **编排神经系统**, 而非底层模型. 它把已有的
LatticeCore (语言/多模态理解)、WorldModel (世界模型)、PersistentMemory (长期记忆)、
ContextBank (上下文)、metacognition / self_model (待建) 作为可插拔的 "能力单元"
(Capability Unit) 编排成闭环.

### 4.2 核心架构

```
        ┌─────────────┐
  env   │ Perception  │  (UnitCodec: 文本/多模态 -> Unit 表征)
        └──────┬──────┘
               │ percept
        ┌──────▼──────┐
        │  MessageBus │  (发布-订阅, target 路由 + 广播)
        └──────┬──────┘
               │ cognition
        ┌──────▼──────┐
        │  Cognition  │  (LatticeCore / WorldModel 前向与潜空间推演)
        └──────┬──────┘
               │ action
        ┌──────▼──────┐
        │   Action    │  (UnitDecoder 生成 / 工具调用 / 环境交互)
        └──────┬──────┘
               │ feedback
        ┌──────▼──────┐
        │ Reflection  │  (更新 SelfModel + 触发记忆衰减)
        └──────┬──────┘
               │ 循环回到 Perception 的下一轮
```

### 4.3 模块协作方式
- **MessageBus**: 所有载荷都是 Unit 表征或标量信号 (无词表), 按 `target` 路由;
  `target=None` 时为广播. 注册新能力只需 `orchestrator.register(Unit)`.
- **闭环优先**: 每次交互都走 感知→推理→行动→反思, 而不是单向生成; 与
  AGI_ARCHITECTURE_DESIGN.md 的 "主体性是闭环涌现" 一致.
- **自我模型**: `UnityOrchestrator.self_state` 预留与待建 `self_model.py` 对接,
  承载价值/目标/对自身能力的表征.

### 4.4 与其他组件集成
| 组件 | 接入点 | 角色 |
|---|---|---|
| UnitEmbedding | PerceptionUnit | 感知编码 |
| LatticeCore / LatticeLM | CognitionUnit | 理解与生成内核 |
| WorldModel + RolloutEngine | CognitionUnit | 潜空间推演 / 规划 |
| PersistentMemory / ContextBank | ReflectionUnit | 长期与上下文记忆 |
| metacognition | ReflectionUnit | 元认知监控 |
| Expert Center (100+ 专家) | MessageBus | 垂直专业能力 (可选挂载) |

### 4.5 可扩展性
- **新增能力**: 写一个继承 `ExpertUnit` 的类并 `register`, 不改核心.
- **多智能体**: `MessageBus` 天然支持多订阅者, 可扩展为团队 (Team) 协作.
- **插件/专家**: 左侧 "专家" 中心已有的 100+ 专家可作为 `ExpertUnit` 接入总线.

---

## 5. 迁移策略与未决风险

### 5.1 迁移路径 (避免破坏既有 60+ 文件)
1. 新增 `unit_table_size` 字段, `vocab_size` 保留为兼容默认值 (字符级).
2. 训练/推理入口逐步从 `Embedding(vocab_size,d)` 切到 `UnitEmbedding(d)`.
3. `lm_head` 改为 UnitDecoder (字符分布输出).
4. 文本管线移除 tokenizer 依赖, 默认走 `UnitCodec`.
5. 全局检索环节的 softmax-read 改为确定性哈希 (闭环 "无注意力").

### 5.2 风险
- **P1**: 字符级序列更长 → 训练/推理成本上升; 但配合 LatticeCore 的 O(n) 不放大为 n².
- **P1**: `PersistentMemory.read` 仍含 softmax, 需在联网环境补测并改造 (见 FORMULA_VERIFICATION.md).
- **P2**: 当前 sandbox 无法装 numpy, 新增模块仅完成 py_compile 级验证; 集成测试需联网.

### 5.3 交付文件
- `simplcore/architecture/ * unit_embedding.py` —— 无词表单元嵌入与编码.
- `simplcore/context/context_store.py` —— 上下文存储/检索/衰减 (ContextBank).
- `simplcore/unity/orchestrator.py` —— SimplUnity 编排内核 (skeleton).
- 本文档 —— 命名、架构、上下文、SimplUnity 的完整设计.

下一步: 把 `PersistentMemory.read` 改为非 softmax 确定性检索, 并接入
UnityOrchestrator 做端到端冒烟测试 (需联网装 numpy).
