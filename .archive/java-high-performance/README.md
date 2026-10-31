# java-high-performance 状态说明

> ## NON-NORMATIVE（非规范实现）
>
> * 本目录是**研究原型**，**不参与 CI**、**不进 Python 包**（`pyproject.toml`
>   的 `packages` 不含它）、**不得作为 SimplCore 架构或性能证据**引用。
> * 唯一规范来源是 Python 侧 `simplcore/core`；本目录与其**没有自动同步机制**，
>   数学语义一旦分叉，以 Python 为准。
> * 本目录仍包含 `MultiHeadAttention`（RoPE/GQA）——那是**历史对照臂**，
>   不属于 SimplCore 产品线。产品线的无注意力约束由
>   `simplcore/core/purity.py` 的配置/实例/源码三层守卫在 Python 侧强制。
> * 任何"Java 与 Python 数值对齐"的结论只能来自逐算子 conformance 实测；
>   在跑通之前，本目录的数值仅代表它自己。

## 定位

本目录是 SimplCore 谱卷积架构（SCArch）的 **Java 高性能平行实现**，
与 Python 侧 `simplcore/core` 在数学语义上对齐（同样的双形式：FFT 并行
训练 / IIR 流式推理），但为**手工平行移植，无自动同步机制**。

⚠️ 因此：Python 侧 `simplcore/core` 的任何 mixer / 数值 / 初始化改动，
本目录**不会**自动跟进。以 Python 侧为唯一规范来源（single source of
truth），本目录视为**研究原型**，其数值对齐状态以下列验证报告为准：

- `SCARCH_FINAL_REPORT.txt`
- `Qingxiaotuan_Training_Report.txt` / `Qingxiaotuan_Training_Summary.txt`
- `Qingxiaotuan_OnlineLearning_Report.txt`

## 构建

```bash
mvn -q package          # 产物输出 target/（已 gitignore，不入库）
mvn test                # 单元测试
```

依赖 jar 在 `lib/`（jocl-2.0.4、junit-console，均已 gitignore，
可由 Maven 中央仓库重新获取）；GPU 内核位于 `src/main/resources/`
（CUDA `.cu` 与 OpenCL `.cl`）。

## 仓库卫生约定（2026-10 清理）

以下内容一律**不入库**（见 `.gitignore`）：

- 编译产物：`target/`、`out/`、`bin/`、`bin-test/`、`test-reports/`、`*.class`
- JVM 崩溃/诊断日志：`hs_err_pid*.log`、`replay_pid*.log`
- 运行/基准/训练日志与结果输出：`*.log`、`*_out.txt`、`*_err.txt`、
  `*_result.txt`、`sweep_*.txt`、`*_metrics.csv`
- 模型二进制：`*.model`、`model.bin`、`sc_production_model.bin`
- 依赖 jar：`lib/`

人工撰写的报告（4 份 `.txt`）与 `pom.xml`、GPU 内核源码保留入库。
