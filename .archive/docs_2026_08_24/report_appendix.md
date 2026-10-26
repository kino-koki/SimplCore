## 附录 A 代码与产物证据索引（Evidence Index）

本章列出本报告各论断对应的可直接复核的代码路径、文件与产物字段，便于独立验证。

### A.1 架构与分层证据

| 论断 | 证据路径 / 字段 |
|------|----------------|
| 文档描述 Java + CUDA | `simplcore_ARCHITECTURE.md` 第 4.2 节、`resources/cuda/SimplCore_gpu.cu`（文档声称，仓库不存在） |
| 真实 GPU 为 OpenCL/JOCL | `java-high-performance/src/main/resources/opencl/simplcore_kernels.cl`、`gpu/OpenCLAccelerator.java`、`gpu/GPUMatrixOperations.java` |
| Java 包结构 30+ 子包 | `java-high-performance/src/main/java/com/qingxiaotuan/simpcore/` 遍历结果 |
| Python 平铺子包 27 个 | `simplcore/` 目录遍历（agi/cognition/emotion/speech/vision/rag/search/tensor/engine/mix/...） |
| 抗幻觉 9 门控（文档未载） | `java-high-performance/.../antihallucination/`：InputValidationGate/OutputValidationGate/FactVerificationGate/ConsistencyGate/BoundaryConstraintGate/KnowledgeModuleGate 等 |

### A.2 模型层证据

| 论断 | 证据路径 |
|------|---------|
| 7 组件被集成进模型 | `java-high-performance/.../model/SimpCoreModel.java` 的 `initModel()` 方法（mhaLayers/ablLayers/dmcLayers/tarLayers/cinLayers/ibLayers/mrdLayers/pumLayers 每层实例化） |
| 组件 backward 返回 GradientResult | 各组件类 `GradientResult backward(...)`（架构文档第 3 章已更正） |
| Python 核心模型基于 torch | `simplcore/architecture/simplcore_model.py:137 class SimplCoreModel(nn.Module)`、`simplcore_block.py:430 class SimplCoreBlock` |
| pg 路线配置化改善 | `simplcore_pg/config.py`（6 档预设 + use_attention 开关）、`simplcore_pg/model.py` |

### A.3 数据 / 服务 / 接口层证据

| 论断 | 证据路径 |
|------|---------|
| Java 数据管道 | `java-high-performance/.../data/DataPipeline.java`、内置 7 个 Processor |
| Python 数据清洗/分片 | `simplcore/mix/data_pipeline/`（clean_zh/docx_corpus/shard_writer）、`simplcore/data/` |
| 训练数据路径缺失风险 | `run_train.py` 引用 `data/distill/sft/*.jsonl`（根目录未检索到对应文件） |
| Java 接口层 | `java-high-performance/.../api/APIServer.java`（ServerSocket + 16 线程池 + 8 个 /api/v1/* 端点 + WebSocket） |
| Python 服务层声明但未落地 | `requirements.txt`（fastapi/uvicorn/gradio），`simplcore/` 主包未检索到对应 app |

### A.4 技术选型 / 可扩展性证据

| 论断 | 证据路径 |
|------|---------|
| 双语言并存 | `java-high-performance/`（237 .java）、`simplcore/`（~82k 行 py）、`simplcore_pg/`（13 模块） |
| 分布式自研 Socket 通信 | `java-high-performance/.../distributed/CommunicationLayer.java`（SocketChannel allReduce/broadcast）、`ClusterManager`/`DataParallelStrategy`/`ModelParallelStrategy` |
| Python 分布式 | `simplcore/tensor/distributed.py`（DistributedConfig gloo）、`simplcore/training/deepspeed_integration.py` |
| 规模预设 | `simplcore_pg/config.py`（nano~72b）、`configs/train_72b.yaml`、`SimplCore_56B_MoE_Plan.md` |
| 环境依赖未隔离 | 根目录 `libs/`（site-packages 快照：filelock/fsspec/functorch 等 38 项） |
| 历史产物未归档 | 根目录 `tensorflow_backup_20260712.zip`（30MB）、各类 `*.log` |

### A.5 性能 / 安全证据

| 论断 | 证据路径 |
|------|---------|
| C++ 扩展算子 | `simplcore/engine/csimplcore.cpp`（matmul/softmax/conv2d/relu/gelu 等 11 算子 + OpenMP） |
| CuPy GPU 后端 | `simplcore/engine/gpu.py`（`import cupy`，回退 NumPy） |
| 对象池内存优化 | `java-high-performance/.../memory/MemoryOptimizer.java`（`ScopedFloatArray`） |
| 混合精度 | `simplcore/mix/amp.py`、`TrainingLoop.enableMixedPrecision` |
| 安全模块真实 | `java-high-performance/.../security/`（ModelEncryption/AES、RBACSystem、SecurityFilter） |
| 性能基准仅为算子级 | `benchmark/PerformanceBenchmark.java`、`simplcore/benchmarks/`（无端到端长序列对比） |

### A.6 训练/推理闭环证据

| 论断 | 证据路径 |
|------|---------|
| Java 训练闭环 | `SimpCoreDemo` 示例（trainStep→trainEpoch→generate→evaluate） |
| mix 路线三件套 | `simplcore/mix/scripts/{train,infer,eval}.sh` |
| 第四套 qxt 模型 | `run_train.py`（`from qxt.model.modeling import QXTForCausalLM`） |
| pg 路线无 train 主循环 | `simplcore_pg/model.py`（仅 configure_optimizers/compile/generate） |

### A.7 监控 / 文档 / 规范证据

| 论断 | 证据路径 / 字段 |
|------|----------------|
| 训练监控 14 项指标 | `java-high-performance/.../monitoring/TrainingMonitor.java`（`MetricSnapshot`：loss/gradNorm/LR/stepTime/GPU 利用率/显存/CPU/内存/吞吐等） |
| 告警通道单一 | `TrainingMonitor.MonitorConfig`（仅 console/file 输出，无 Webhook/Prometheus） |
| 可用性演示 SKIP 美化 | `simplcore_availability_report.json`：`Cognition: SKIP（KnowledgeReasoner 导入失败）`，但 demo 判定"可用" |
| 物理证明失真 | `final_physics_proof.json`：`spring/gravity/generalization` 三项 `passed:0`、相对误差均 30.10%、`final_conclusion` 写"成功学习" |
| 结论写死 | `prove_physics_learning.py`（布尔量按真实指标算，但 statement 字段写死"成功"） |
| 日期造假 | `DOCUMENT_AUDIT_REPORT.md`（将 2026 文件改为 2024-01-15） |
| pg 路线规范标杆 | `simplcore_pg/`（docstring + 类型标注 + 异常体系 + 71 pytest 用例 + README） |
| 全局规范缺位 | 无 `.github/workflows`、无统一 Makefile、lint 仅依赖声明 |

### A.8 方法局限说明

本报告为**静态核查**，未在实际 GPU 集群上运行大规模训练以验证端到端性能（评估环境 GPU 不可用，CuPy/JOCL 回退 CPU）。因此"性能基准缺失""跨语言数值一致性"等论断基于代码结构与产物字段推断，建议后续补充动态基准测试以进一步印证。此外，部分根目录脚本（如 `qxt` 依赖的 `data/distill/` 目录）在静态遍历中未命中，可能存在但未纳入核查范围，已在对应论断中标注"风险/待确认"。

---

## 附录 B 术语与缩写对照

| 缩写 | 全称 | 说明 |
|------|------|------|
| RFP | Resonance Field Processor / AdaptiveBasisLayer | 自适应谐振场（频域建模） |
| DMC | Dynamic Memory Channel | 动态记忆通道（外部记忆） |
| TAR | Topology-Aware Router | 拓扑感知动态路由（条件计算） |
| CIN | Causal Intervention Network | 因果干预网络 |
| IB | Information Bottleneck | 信息瓶颈自动剪枝 |
| MRD | Meta Representation Distiller | 元表示蒸馏 |
| SPM | Spontaneous Plasticity Engine / PlasticityUpdateModule | 自发塑性（推理期自适应） |
| MoE | Mixture of Experts | 混合专家（与 TAR 近似） |
| FFT | Fast Fourier Transform | 快速傅里叶变换（谐振场核心） |
| RBAC | Role-Based Access Control | 基于角色的访问控制 |
| AES | Advanced Encryption Standard | 高级加密标准（AES-256-GCM） |
| allReduce | All Reduction | 分布式梯度归约原语 |
| FX | PyTorch FX | 图模式优化框架 |
| OOM | Out Of Memory | 显存溢出 |

---

## 附录 C 推荐阅读顺序（给接手工程师）

1. `SimplCore_PG_Delivery_Report.md` + `simplcore_pg/README.md` —— 先建立"生产级标准"心智模型。
2. `simplcore_pg/model.py`、`block.py`、`attention.py`、`resonance.py` —— 理解核心架构组件的最小可运行实现。
3. `simplcore_ARCHITECTURE.md`（注意其中 CUDA/Java 失真） —— 理解设计理念与七大组件命名。
4. `java-high-performance/.../model/SimpCoreModel.java` —— 对照 Java 侧完整集成。
5. `java-high-performance/.../monitoring/TrainingMonitor.java` + `api/APIServer.java` —— 理解服务与监控骨架。
6. 本报告第 9、10 章 —— 直接定位短板与改进路线。

---

*（全文完。本报告约 30,000 字，含 11 章正文与 3 个附录，覆盖架构合理性、模块完备度与耦合、技术选型与可扩展性、性能与安全、训练/推理闭环、监控运维、文档规范九大维度，所有论断附代码或产物证据。）*
