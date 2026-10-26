# SimplCore Mix 7B 堆叠训练准备设计

## 背景

本机使用 AMD 680M 集成显卡，不适合完整训练 7B 模型。当前路线是训练多个约 150M 参数的 `SimplCore Mix` 子模型，再通过参数堆叠构造 7B 初始权重，并进行受控校准。

## 语料位置

- 公开语料：`E:\qingxiaotuan_text_data\raw\Fineweb-Edu-Chinese-V2.2\2_3`
- 用户授权 `.docx` 语料：`E:\qingxiaotuan_text_data\processed\user_docx\user_docx_corpus.jsonl`
- 用户授权 `.docx` manifest：`E:\qingxiaotuan_text_data\processed\user_docx\user_docx_corpus.manifest.json`

最近检查结果：公开语料目录包含 `6840` 个 Parquet 分片，连续范围为 `000000.parquet` 到 `006839.parquet`，范围内未发现缺号，总大小约 `12825972317` 字节。

## 训练门禁

- 开发和训练目标解释器为 Python 3.11。
- 当前可发现命令行 Python 仍是 3.10.11，不能作为真实 7B 准备运行环境。
- 真实训练前必须通过 `check_python311()`。
- 本机只承载 150M 训练、堆叠 dry-run、manifest 校验和轻量校准准备。

## 损失约束

- 目标损失：`loss < 1.5`。
- 最高损失：`loss <= 3.9`。
- 超过 `3.9` 时必须停止训练并回滚。
- 位于 `1.5` 到 `3.9` 时允许继续，但必须触发学习率衰减、数据批次检查或 checkpoint 回退策略。

## 新增能力

- `gradient_checkpointing`：训练时启用激活重计算，降低显存压力。
- `deep_thinking_enabled`：为深度思考模式提供配置开关，并允许增加循环推理上限。
- `web_search_enabled`：为联网搜索工具集成提供配置开关；该开关表示运行时允许接入搜索工具，不表示基础权重自身具备真实网络访问能力。
- `LossGuard`：统一执行损失门禁。
- `copy_or_expand_tensors`：支持 150M 到 7B 的同形复制和 tile 扩展。
- `build_calibration_plan`：生成 7B 堆叠后的校准步骤与损失守卫策略。
- `build_7b_readiness_report`：生成 Python、语料路径和 loss 门禁准备报告。

## 后续实现

下一阶段应补齐完整 150M checkpoint registry、按角色分配的 stacking manifest、分块写出 7B checkpoint、抽样 forward 校验和 LoRA/分层校准入口。
