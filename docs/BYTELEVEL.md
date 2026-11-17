# Byte-Level 主路径（工业级 tokenizer-free 模块）

> **目标**（用户指令）：抛弃 BPE 分词器这一「最丑陋」的设计，让模型**直接
> 理解原始字节**；性能不降、语言能力反升；本模块开发到工业级，并与
> SimplCore 全部模块完美兼容。
> **状态**：模块 + 守护测试已落地（`simplcore/core/model/bytelevel.py`、
> `simplcore/core/tests/test_bytelevel.py`）。**未跑训练**（用户指令）。

## 1. 研究依据（tokenizer-free 文献，2023–2025）

> ⚠️ 本节为**设计依据的知识综述**；本会话网络工具不可用，未能逐篇在线
> 核对，正式投稿前应对照原文复核。

* **MegaByte**（Meta，arXiv:2305.07185）：字节流按**定长 patch** 分组，
  local encoder / 全局 Transformer / local decoder；子二次复杂度。
  局限：定长 patch 常跨词。
* **BLT / Byte Latent Transformer**（Meta，arXiv:2412.04771）：按**字节熵**
  动态划 patch（小熵模型决定断点），local encoder + 大全局模型 + local
  decoder；同算力下匹配 Llama-3 级模型，对噪声/长尾更稳健。
* **SpaceByte**（arXiv:2502.06803）：在**空格/词边界**断 patch，修正
  MegaByte 定长 patch 跨词的问题。
* **bGPT**（arXiv:2306.08254）：字节级 + 定长 patch，「万物皆字节」。
* **ByT5 / CHAR-S**：无 patch 的字符/字节级——算力需求更高但更稳健、
  对噪声不敏感。

**共同结论**：字节级输入去掉分词器后，用「动态分块 + 局部/全局结构」
可以追平甚至超过 BPE 词表模型，且对噪声、长尾、多语言更稳健。

## 2. 本模块的设计

### 2.1 ByteLevelCodec —— 工业级字节编解码

* 256 字节值 + 可选 BOS/EOS；**严格 UTF-8 往返**（任意 Unicode，
  含 4 字节码点 / emoji / CJK / 控制字符）。
* **流式编码**：`feed()` 分片输入、不完整码点的尾部字节进位；
  `feed_flush()` 收尾。
* **解码容错**：非法字节 → U+FFFD 替换（保留位置），不抛异常。
* `save` / `load`（JSON）。

### 2.2 EntropyPatcher —— 动态分块（三模式）

| 模式 | 机制 | 出处 |
|------|------|------|
| `entropy`（默认） | 在线一阶马尔可夫替代熵（Laplace 平滑、因果、零参数），累计熵超阈值断开 | BLT |
| `space` | 非字母数字字符（空格/标点）后断开 | SpaceByte |
| `fixed` | 定长 `max_len` | MegaByte |

共同性质：边界**单调**、**因果**（只依赖已见字节）、**流式 = 批量**
逐边界一致、**确定性**、`max_len` 防无限增长。

两个入口共享同一份转移逻辑：

* `boundaries(byte_ids)` —— **纯批量**：不读不改任何实例状态，同一输入
  重复调用逐项相同（可安全用于前向图 / 数据管线 / 生成重分块）；
* `feed()` / `feed_flush()` —— **有状态流式**：进位保留未决字节，
  返回**全流绝对下标**；`feed(全部) + feed_flush()` 与
  `boundaries(全部)` 逐项一致（三模式均有守护测试锁定）。

### 2.3 BytePatchBridge —— 与任意 SimplCore mixer 桥接

字节 embedding（256）按 patch **均值池化** → patch 级 latent token 序列
→ 任意 SimplCore mixer（SGC / sgc-bind / SRF…）。mixer 的 O(1) 状态、
严格因果、流式推理不变——patcher 只是**因果、可流式、纯函数式**的字节
分段器，不引入注意力 / KV cache / 学习参数。

## 3. 工业级保证（守护测试 `test_bytelevel.py`，21 项）

* **对抗 Unicode 往返**：emoji、CJK、4 字节码点、控制字符、混合文本
  encode → decode 恒等；
* **流式 = 批量**：任意分片 `feed` + `flush` 与一次 `encode` 逐字节一致；
  patcher 三模式的 `feed`（**全流绝对下标**）+ `flush` 与批量
  `boundaries` 逐边界一致（含收尾 patch）；
* **解码容错**：截断/非法字节 → U+FFFD，不抛异常；
* **patcher 三模式**：确定性、边界单调、`boundaries` **纯函数**
  （重复调用 + 先流式消费一批均不改结果）、`max_len` 有界、
  `reset()` 清态；
* **mixer 兼容**：sgc / sgc-bind / srf 三种 mixer 的前向 + 反向
  （梯度可达）shape/finite 检查；
* **入口性质**：`byte_level_logits_ids` 纯字节空间（非法 UTF-8 上下文
  不被替换字符改写、无跨调用状态泄漏）；`byte_level_logits` 不 detach，
  梯度可达字节嵌入表；生成 greedy 确定性；
* **空输入 / 单字节**等边界情形。

## 3.5 工业级入口（字节/文本 → patch → 前向/生成）

```python
from simplcore.core.model.bytelevel import (
    byte_level_logits_ids, byte_level_logits, byte_level_generate)

logits = byte_level_logits_ids(model, byte_ids)  # 纯字节空间原语
logits = byte_level_logits(model, "任意文本")     # (n_patches, 256) 字节词表
out = byte_level_generate(model, "prompt", max_new_patches=8, greedy=True)
```

* 兼容任意 SimplCore mixer（SGC / sgc-bind / SRF…）——patch 级 latent
  序列直接过 mixer 栈；O(1) 状态/流式不变；
* **生成不改写上下文**：`byte_level_generate` 全程在字节 ID 空间拼接
  上下文，绝不 decode→encode 往返——即使采到非法 UTF-8 字节（孤立续
  字节等）也不会被替换字符悄悄改写历史；
* 守护测试：入口与手动路径（bridge + blocks）逐值一致、纯字节空间
  幂等、嵌入梯度可达、生成 smoke。

## 4. 与 SimplCore 全部模块的兼容

| 模块 | 兼容方式 |
|------|----------|
| mixer（sgc / sgc-bind / srf / …） | patch 级 latent 序列直接进 mixer；O(1) 状态/流式不变 |
| 训练数据管线 | `pack_documents` 接受任意 codec——char/byte 打包即 tokenizer-free 语料 |
| checkpoint | 模型权重 + codec JSON 一起保存即完整恢复 |
| dialogue / chat | `forward(text=..., codec=...)` 文本入口直接可用 |
| unity / memory / eval | 消费模型输出，与输入粒度无关 |

## 5. 诚实边界

* **未跑训练**（用户指令）：「性能不降、语言能力反升」的最终证据需要
  训练对比；本模块提供的是**结构 + 正确性 + 兼容**，文献（BLT 等）
  支持该设计方向。
* 熵 patcher 用**一阶马尔可夫替代熵**（零参数、因果），不是 BLT 的
  学习熵模型；更贴近 BLT 需接一个小熵模型（后续工作）。
* 本会话网络工具不可用，文献综述基于既有知识，**正式引用前需对照
  arXiv 原文复核**。
