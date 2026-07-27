# SimplCore Mix 纯文本参数堆叠训练设计

**日期**：2026-07-27  
**适用底座**：`simplcore/mix`  
**目标模型**：先构建纯文本 7B 参数模型，再扩展到 14B 参数模型  
**硬件现实约束**：当前本机为 AMD Radeon Graphics 680M，适合代码整理、数据清洗、小模型冒烟训练和参数堆叠流程验证，不适合直接完成 7B/14B 全量预训练。

## 1. 目标

本设计将 `simplcore/mix` 确认为当前唯一训练底座。旧的 `qingxiaotuan` 包路线、`simplcore/architecture` 中残缺或冲突的模型入口、以及多模态相关训练入口不再作为本轮训练主线。

本轮模型是纯文本因果语言模型。训练、参数堆叠、评测和导出都只面向文本 token 序列，不接入视觉、语音、相机、多模态融合或世界模型模块。

最终目标分两阶段：

1. 训练多个约 150M 参数的小型 SimplCore Mix 文本模型。
2. 通过自研“参数堆叠技术”把小模型参数迁移、拼接、重标定为一个约 7B 参数模型。
3. 在 7B 模型完成堆叠和继续训练校准后，复用同一技术路线扩展到 14B。

## 2. 非目标

本轮不做以下内容：

- 不接入 SimplCore 多模态模块。
- 不训练视觉模型、语音模型、图文模型或世界模型。
- 不生成合成预训练数据作为主要语料。
- 不把不可导入的旧 `qingxiaotuan` 包作为训练依赖。
- 不声称 AMD Radeon Graphics 680M 可以完成 7B/14B 全量训练。
- 不使用来源不明、授权不清或无法合法获取的数据。

## 3. 数据策略

训练数据只选择简体中文文本。优先接入公开、可合法获取、官方或权威发布、已清洗或可复核清洗流程的数据源。

第一优先级数据源为 CLUECorpus2020。该数据集说明为 100G 中文预训练语料，来源于 Common Crawl 中文部分并经过清洗，适合作为简体中文预训练语料起点。

数据管线需要在 `simplcore/mix/data` 或 `simplcore/mix/data_pipeline` 下形成统一入口，负责：

- 下载或导入公开中文语料。
- 校验授权说明和数据来源记录。
- 解压、分片和流式读取。
- 只保留简体中文文本。
- 过滤乱码、繁体比例过高、英文比例过高、过短、重复、低质量和敏感文本。
- 输出统一 JSONL 或二进制 token shard。
- 记录每批数据的来源、清洗规则、样本数量、token 数和哈希。

训练前必须先跑数据冒烟检查：随机抽样不少于 100 条，确认文本为简体中文、无明显乱码、字段格式稳定。

## 4. 模型底座

`simplcore/mix` 是唯一底座。核心模块包括：

- `SimplCoreMixConfig`
- `SimplCoreMix`
- `PreludeBlock`
- `RecurrentBlock`
- `CodaBlock`
- `RFPAttention` 或 `RFPMLAttention`
- `TARMoE`
- `DynamicMemoryChannel`
- `LTIInjection`
- `ACTHalting`
- `LoRAAdapter`
- `BPETokenizer` 或后续统一 tokenizer
- 文本数据加载、训练、评估、推理和导出模块

多模态、视觉、语音、相机、世界模型和认知演示模块不进入本轮训练主链。

## 5. 参数堆叠技术

参数堆叠不是简单权重平均，也不是简单模型 ensemble。它是一套把多个小模型的参数迁移到更大目标模型中的结构化扩容技术。

### 5.1 小模型约束

每个约 150M 参数小模型需要满足：

- tokenizer 完全一致。
- vocab size 完全一致。
- hidden size、attention head 维度、RoPE 设置和核心归一化方式与目标大模型兼容。
- 模块命名和参数 shape 可被映射器识别。
- 训练数据来源一致，但每个小模型可使用不同数据 shard 或不同训练阶段。
- checkpoint 包含 config、tokenizer 元信息、训练步数、数据 shard 记录和参数哈希。

### 5.2 7B 堆叠方式

7B 目标模型采用“深度段 + 专家段 + 记忆段”的复合堆叠。

小模型参数会被映射到目标模型的不同区域：

- `Prelude` 参数映射到目标模型前部层。
- `RecurrentBlock` 参数映射到目标模型中部共享循环模块或循环深度组。
- `Coda` 参数映射到目标模型后部层。
- `TARMoE` 专家参数映射为目标模型更大的专家池。
- `DynamicMemoryChannel` 参数映射为更多记忆通道。
- embedding 和 output projection 采用共享 tokenizer 下的受控初始化、复制、平均或重标定策略。

堆叠完成后，必须执行参数校验：

- 参数总量接近目标规模。
- 所有目标参数都有来源或明确初始化规则。
- shape 不兼容时必须报错，不允许静默跳过。
- 每个来源 checkpoint 的贡献位置写入 manifest。
- 前向传播必须通过随机 token 冒烟测试。

### 5.3 堆叠后校准

7B 模型堆叠完成后不能直接视为可用模型，必须继续训练校准。

校准阶段目标：

- 对齐残差尺度。
- 对齐不同小模型之间的激活分布。
- 稳定 MoE 路由。
- 稳定 ACT 停止概率。
- 稳定 DynamicMemoryChannel 的门控和衰减。
- 降低初始 loss 抖动。

最低要求是先跑小数据校准冒烟训练；如果硬件不足，则生成可在远端 GPU 环境继续运行的配置和 checkpoint。

## 6. 14B 扩展

14B 不作为第一阶段直接训练目标。它复用 7B 参数堆叠技术，待 7B 路线验证后扩展。

14B 扩展可以采用两种方式：

1. 从更多 150M 小模型堆叠到 14B。
2. 以已校准的 7B 为母体，再追加深度段、专家段和记忆段扩容到 14B。

推荐第二种方式，因为它能复用 7B 的已校准表示，训练稳定性更高。

## 7. 硬件策略

AMD Radeon Graphics 680M 不适合直接完成 7B/14B 全量预训练。当前本机执行范围限定为：

- 代码整理。
- 文档更新。
- 数据下载、清洗和分片。
- tokenizer 训练或加载。
- 极小配置模型语法与导入检查。
- 150M 配置的短步数冒烟训练，视内存情况可降配。
- 参数堆叠脚本 dry-run。
- 小尺寸目标模型模拟堆叠测试。
- 生成 7B/14B 的正式训练配置、manifest 和恢复脚本。

真实 7B/14B 训练建议在具备足够显存的独立 GPU 或多卡服务器上执行。脚本必须支持断点续训，避免长任务中断后从零开始。

## 8. 文件整理

有用文件应向 `simplcore/mix` 收敛。

建议目标结构：

```text
simplcore/mix/
  config.py
  model.py
  blocks.py
  attention.py
  moe.py
  memory.py
  tokenizer.py
  data_loader.py
  data_pipeline/
    sources.py
    clean_zh.py
    dedup.py
    shard_writer.py
    prepare_cluecorpus.py
  training/
    train_small.py
    train_stacked.py
    deepspeed_config.py
    checkpoint.py
  stacking/
    stack_config.py
    mapper.py
    manifest.py
    validate.py
    build_7b.py
    build_14b.py
  evaluation/
    smoke_eval.py
    loss_eval.py
    zh_text_eval.py
  inference/
    generate.py
  export/
    export_pytorch.py
    export_manifest.py
  tests/
    test_imports.py
    test_tiny_forward.py
    test_data_cleaning.py
    test_stacking_dry_run.py
```

旧文件处理规则：

- 可复用的文本训练、数据、评测、推理、导出逻辑迁移到 `simplcore/mix`。
- 多模态、视觉、语音、世界模型代码保留在原位置，但不进入本轮训练主链。
- 损坏测试文件修复或重写。
- 残缺模型入口改为兼容导入或明确废弃。
- 根目录旧训练脚本不再作为主入口。

## 9. 已知问题修复

第一轮必须修复：

- `qingxiaotuan` 包缺失导致训练和测试失败。
- `simplcore/mix/test_fullstack.py` 语法损坏。
- `simplcore/architecture/model.py` 和 `simplcore/architecture/simplcore_model.py` 内容残缺且与当前底座冲突。
- 文档声称 Java/SimplCore 与实际 Python/PyTorch 训练路线不一致。
- `setup.py` 引用缺失的 `README.md`。
- 数据管线未限定简体中文。
- 训练入口分散。
- 测试不能验证最小训练闭环。

修复原则是优先让 `simplcore/mix` 形成可导入、可测试、可冒烟训练的闭环，不做无关重构。

## 10. 训练流程

阶段一：整理与验证

- 更新技术文档。
- 整理有用文件到 `simplcore/mix`。
- 修复导入、语法、测试和入口问题。
- 跑 `simplcore/mix` 最小 forward/backward 测试。

阶段二：数据准备

- 接入公开简体中文语料。
- 清洗、去重、分片。
- 训练或确认 tokenizer。
- 生成训练 manifest。

阶段三：150M 小模型训练

- 生成 150M 配置。
- 启动短步数冒烟训练。
- 保存 checkpoint。
- 验证 loss、参数哈希、恢复训练。
- 批量训练多个小模型分片。

阶段四：7B 参数堆叠

- 生成 7B 目标配置。
- 加载多个 150M checkpoint。
- 执行参数映射。
- 输出 7B checkpoint 和 stacking manifest。
- 执行随机输入 forward 冒烟测试。

阶段五：7B 继续训练校准

- 使用简体中文语料继续训练。
- 监控 loss、梯度、路由分布、ACT 停止概率。
- 支持中断恢复。

阶段六：14B 扩展

- 基于 7B 已校准模型扩展。
- 重复参数映射、校验、继续训练流程。

## 11. 验收标准

进入正式训练前必须满足：

- `simplcore/mix` 可以被干净导入。
- 损坏测试文件已修复。
- 最小模型 forward/backward 通过。
- 数据清洗脚本只输出简体中文样本。
- 参数堆叠 dry-run 通过。
- 7B 配置可以生成。
- 训练脚本支持 checkpoint 保存和恢复。
- 文档与代码入口一致。

7B 阶段验收：

- 生成完整 7B config。
- 生成 7B stacking manifest。
- 参数总量接近 7B。
- 随机 token forward 通过。
- 继续训练可以启动并保存 checkpoint。

14B 阶段验收：

- 生成完整 14B config。
- 可以从 7B checkpoint 扩展。
- 14B stacking dry-run 通过。
- 继续训练配置可用。

## 12. 风险

主要风险：

- 本机硬件无法完成真实 7B/14B 训练。
- 公开数据下载体积大，可能受网络和磁盘限制。
- 多个小模型堆叠后激活分布不一致，必须继续训练校准。
- 150M 到 7B 的结构映射如果设计不稳，可能导致 loss 初始很高。
- CLUECorpus2020 等数据源需要复核下载入口、授权和文件格式。

应对方式：

- 先用 tiny 配置和 150M 配置验证流程。
- 所有训练任务支持断点续训。
- 所有堆叠操作生成 manifest，便于追溯。
- 全量训练脚本和本机冒烟脚本分离。
- 数据源接入前记录来源和授权说明。
