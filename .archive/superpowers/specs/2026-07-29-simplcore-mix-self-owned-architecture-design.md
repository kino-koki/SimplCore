# SimplCore Mix 自研外观与深度融合设计

**日期**：2026-07-29  
**架构名称**：`SimplCore Mix`  
**开发语言策略**：以后统一使用 Python 3.11 开发；Java 高性能侧统一使用 Java26。  

## 目标

本轮目标不是更名，而是让 `SimplCore Mix` 在工程表层和架构语义上都更像自研架构。公开模型主干不再直接暴露 `torch.nn.Module`、`nn.Linear`、`nn.Parameter`、`nn.Dropout` 等第三方框架命名，而统一使用 SimplCore 自有 `SC*` 抽象。

底层训练能力保持稳定。当前默认后端命名为 `simplcore_torch`，它只是过渡后端，不作为架构身份。后续可继续替换为 `simplcore_native` 或 Java 高性能后端。

## 自研外观层

新增 `simplcore/mix/core/`，提供以下公开接口：

- `SCBackend`
- `SCModule`
- `SCParameter`
- `SCLinear`
- `SCEmbedding`
- `SCDropout`
- `SCModuleList`
- `SCSequential`
- `SCReLU`
- `SCSigmoid`
- `SCRMSNorm`
- `SCFunctional`

`SimplCoreMix` 主模型继承 `SCModule`。核心模型文件通过测试约束，不再出现公开 PyTorch 层命名。

## 深度融合组件

新增 `simplcore/mix/fusion.py`，承接 Java SimplCore 主线中的四类组件语义：

- `SimplCoreCausalInterventionNetwork`：CIN 因果干预网络，用于循环状态校正。
- `SimplCoreInformationBottleneck`：IB 信息瓶颈，用于压缩和过滤循环噪声。
- `SimplCoreMetaRepresentationDistiller`：MRD 元表示蒸馏，用于低维元表示重构。
- `SimplCorePlasticityUpdateModule`：PUM 可塑性更新模块，用于稳定的可塑性通道。

这些组件接入 `RecurrentBlock`，位于 DMC 动态记忆之后、SPM 自发可塑性之前，使 Java SimplCore 的组件语义进入 `SimplCore Mix` 主计算路径。

## 测试标准

新增 `test_self_owned_core.py`，验证：

- `SC*` 自研核心接口可导入。
- `SimplCoreMix` 是 `SCModule` 实例。
- 架构信息声明 `backend=simplcore_torch` 和 `self_owned_surface=True`。
- `attention.py`、`moe.py`、`memory.py`、`blocks.py`、`model.py` 不再出现公开 PyTorch 层命名。
- `RecurrentBlock` 包含 CIN、IB、MRD、PUM 四类深度融合组件。

继续保留原有 tiny 前向/反向测试，确保新增组件不破坏梯度计算。

## 后续阶段

下一阶段应继续推进：

- 将 `utils.py` 的 `RMSNorm`、RoPE、mask 等工具迁移到 `SCFunctional` 或 `core/ops.py`。
- 将训练脚本中的 `torch.optim` 包装为 `SCOptimizer`。
- 将数据加载中的 `torch.utils.data` 包装为 `SCDataLoader`。
- 逐步让 `simplcore/engine` 的 `SimpTensor` 成为可选后端。
- 为 Java 高性能后端补充权重导入、推理等价性和小规模训练一致性测试。
