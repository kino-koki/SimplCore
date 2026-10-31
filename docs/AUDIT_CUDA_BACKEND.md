# CUDA·HIP / C++ 后端算子契约审计（Track 2 · 目标硬件：AMD Instinct MI250 / ROCm）

日期：2026-10-29 · 审计对象：`simplcore/core/backend_contract.py`、
`conformance_runner.py`、`backend_bridge.py`、`adapter_model.py`、
`simplcore/csrc/mixer_stream*.{h,cu,cpp}`、`simplcore/ops/mixer_stream.py`

> 结论先行：**契约/桥接/跑批器基础设施扎实且自洽，但「后端已就绪」的说法需要分层修正。**
> 仓库里实际存在**一个**真实融合算子源码（`mixer_stream_cuda.cu`，CUDA 写法），但
> **从未在任何目标机编译运行**。关键：该源码只用了 HIP 兼容原语（`__shfl_down_sync`、
> `<<<>>>` 启动、`torch::Tensor`），**在 MI250/ROCm 上 PyTorch 会以 hipcc 直接编译 `.cu`**，
> 移植工作量极小——故问题不是「放弃 CUDA」，而是「把它重新定位为 HIP 的移植起点」。
> 契约要求的 **16-op `sc_*` ABI 后端则完全没有实现**，Phase 0 验收目前只对着
> torch 参考数学自证，任何真实后端都过不了 `load_backend` 的符号校验。

---

## 1. 事实厘清：到底有什么

| 组件 | 状态 | 说明 |
|------|------|------|
| `csrc/mixer_stream_cuda.cu` + `mixer_stream.h` | **真实源码，未编译** | 融合单遍流式 mixer（RMSNorm→in_proj→内容门→IIR→局部卷积→L2/L3 门→out_proj→残差）。CUDA 写法但仅用 HIP 兼容原语；**MI250/ROCm 下 PyTorch 以 hipcc 编译 `.cu` 即可用**，移植成本极低。文件内「本机无 nvcc 无法实测」仅指 NVIDIA 构建，对 ROCm 不适用。 |
| `csrc/test_mixer_stream.cpp` | 存在 | CPU 参考 + GPU 驱动的测试：NVIDIA 需 nvcc 构建；**ROCm/MI250 下 PyTorch 以 hipcc 编译 `.cu` 路径，同套源码可测**。 |
| `simplcore/ops/mixer_stream.py` | 存在 | Python 绑定，`sgc.step()` 在 `use_mixer_stream=True` 时下沉；**默认 `False`**。 |
| `backend_contract.py`（16-op `sc_*` ABI） | **契约 + fp32 CPU 参考后端** | `csrc/sc_ref_backend.c`（2026-10-29 起提供，见 `docs/REF_BACKEND.md`）：纯 C99 无依赖、直接时域卷积，`load_backend` 已可加载真实编译库；生产 HIP 后端仍缺。 |
| `conformance_runner.py` | 自洽 | 9 项检查默认跑 torch 参考自检；现在可对参考后端执行真实验收（dlopen/符号校验/编组全走真路径）。 |

**关键差异（修正评估）**：之前「CUDA 内核不存在」的判断**不准确**——`mixer_stream`
融合算子是真写的；但它与契约的 16-op `sc_*` ABI **是两套不同的东西**：
前者是可选推理热路径，后者是战役运行时的完整算子契约。战役文档把两者混称为
「后端已就绪」，易误导。

---

## 2. 逐位一致性风险点（按严重度）

### [高] R1 — 16-op 契约后端缺位，Phase 0 闸门实质未达成
> **2026-10-29 部分缓解**：新增 `csrc/sc_ref_backend.c`（fp32 CPU 参考后端），
> `load_backend` 首次可加载真实编译库并跑全部 9 项一致性检查（构建/验收
> 步骤见 `docs/REF_BACKEND.md`）。生产 HIP 后端（bf16/FFT/MI250）仍缺。
所有 `conformance_checks()` 硬项（P0-backend-conformance 等）此前只验证了
`TorchReferenceAdapter` 暴露的算子语义。真实后端必须经 `load_backend` 加载并
过 9 项检查。README §6「自研 CUDA·HIP/C++ 后端算子契约（16 内核 +
逐位一致性硬检查）」当前**部分成立**：契约+参考实现已可执行，生产后端未到位。

### [高] R2 — 融合算子从未在目标机实测，存在未暴露的 bug 风险
`.cu` 文件记录了 v2 修复的并发竞争（写 `hb` 与读 `hb[j]` 跨线程竞争，因本机
无编译工具链（NVIDIA: nvcc / ROCm: hipcc）导致 pytest 一直 skip，bug 从未暴露）。**该修复本身未经执行验证**。
并行归约（warp-shuffle）改变了求和顺序，与串行 CPU 参考有 ≤1e-6 量级差，
文档声称「远在 1e-4 守护阈值内」，但**这一声明未被任何实测支撑**。

### [中] R3 — 一致性跑批只在 fp32 下执行，bf16 路径从未被验收
`conformance_runner` 的 `_mixer_weights` 与全部检查均为 fp32；契约的精度策略
（bf16 计算 / fp32 累加、递推状态 fp32）与 `P0-bf16-accum` 只验了
gemm/rmsnorm/ce_loss，**从未验 mixer 的谱卷积 / IIR 扫描在 bf16 I/O 下的数值**。
而 `sgc.step()` 的核心诚实性正是「bf16 计算、fp32 递推状态」——这条路径在
跑批器里是**真空白**。

### [中] R4 — 长上下文累积漂移未测
命题 5 的 fp32 误差界在最坏参数下急剧保守（v1.1.2 修正后的界含
`1/((1−ρ)|sinθ|)` 联合因子：`ρ→1` 贡献 ≈1.6e4（默认 decay_max），`logfreq`
慢基 `θ→0` 再贡献 `decay_max/π`（4M 配置 ≈1.3e6）——联合后对长序列几乎无
约束力）。
`C-dual-form` 的全口径 L=4M 测试**只建议「在后端机上跑」**，CI 用短序列；
L≥10⁵ 的逐 token 递推漂移从未实测。这是 §6 已自承的缺口，但无缓解计划。

### [中] R5 — `weight_decay_exempt` 未验证接入训练器
契约要求对 ρ/θ 基参数豁免 weight decay（防止长程记忆被掐死），`TorchReferenceAdapter`
和 `FakeCLib` 都实现了该逻辑，但**没有任何检查证明 `trainer.py` 真的调用了它**。
若训练器对基参数施加 decay，长程记忆会被静默破坏——且与纯度/数值无关，曲线上看不见。

### [低] R6 — 2×GCD 切分仅做 fp32 顺序加和，无真实分布式一致性测试
`shard_allreduce` 检查只把两个张量相加比对；「2 GCD 切分下训练数值与单卡一致」
（all-reduce fp32）是声明，无多卡/多分片数值复现测试。

### [低] R7 — 契约与融合算子的 fp32 状态语义不一致
`backend_contract.py` 的 `stream_state_init` 标注「O(1) 常驻状态，与 L 无关」；
而 `mixer_stream.h` 的 `MixerStreamWeights` 也持有 `energy[]`（L2 运行和）。
两者一致，但 `conformance_runner` 的 `_mixer_streaming` 用 python dict 持有
`energy_sum`，**未校验它在真实后端 ABI（`sc_chunked_iir_scan` 不带 energy 出参）
下如何随状态 serialize/restore**——跨块携带的 energy 一致性是 C-chunked-carry
的未覆盖分支。

---

## 3. 缺测项汇总（对照契约 bitexact_requirements）

| 契约要求 | 是否被任何测试覆盖 | 缺口 |
|----------|-------------------|------|
| C-dual-form (fp32, L=4M) | 仅短序列 CI | 全口径未跑 |
| C-chunked-carry | runner 内 python 验证 | 真实后端未验；energy 跨块未验 |
| C-rollback | runner 内 | 真实后端未验 |
| C-stream-serialize | runner 内 | 真实后端未验 |
| P0-bf16-accum | gemm/rmsnorm/ce_loss | **mixer 谱卷积/IIR 未验** |
| P0-shard-2gcd | 顺序加和 | 无分布式 |
| P0-no-attention | 结构审查 | 真实库符号未审查（库不存在） |
| `weight_decay_exempt` 接入 | 适配器方法 | **训练器接入未验** |

---

## 4. 测试补全清单（可执行，按优先级）

1. **[阻断] 构建并实测融合算子**：在 MI250/ROCm 目标机以 PyTorch 自带的 hipcc 编译 `.cu`（`python -c "from simplcore.ops import mixer_stream_step; print(mixer_stream_step.is_available())"`）；
   （或 `test_mixer_stream.py`），断言 `mixer_stream_step` 与 `sgc.SpectralGlobalConv.step`
   在 fp32 下 max\|Δ\|<4e-5。给 v2 并发修复一个**真实执行**的回归测试。
2. **[阻断] 明确「后端就绪」口径**：在 README §6 区分「融合流式热路径（源码，未编译）」
   与「16-op 契约后端（未实现）」，删除「自研 CUDA/C++ 后端…已就绪」的笼统表述。
3. **bf16 路径验收**：扩展 `conformance_runner` 增加 `P0-bf16-mixer` 检查——
   bf16 I/O + fp32 状态跑 `mixer_stream` vs fp32 真值，验证递推状态 fp32 不变量。
4. **长上下文漂移**：在后端机跑 `C-dual-form` L=4M / 1M / 8M，并在 L=10⁵、10⁶ 对
   `step()` 逐 token 递推做 fp64 对照，给出实测误差曲线（补命题 5 的 ρ→1 边界）。
5. **energy 跨块一致性**：在 `C-chunked-carry` 中显式断言 `energy_sum` 跨块携带与
   整段一致（当前 runner 未断言 energy）。
6. **训练器豁免接入**：在 `trainer.py` 加单测，断言 ρ/θ 基参数不在 weight-decay 组，
   且冻结掩码对基参数生效（直接卡 R5）。
7. **契约后端最小实现**：若战役真要走 16-op ABI，至少先实现 `TorchReferenceAdapter`
   到 `sc_*` 符号的桥接 `.so`（`backend_bridge.load_backend` 现在对任何库都会因缺符号失败），
   让 `load_backend` 路径先跑通一版参考实现。

---

## 5. 本审计不改代码
本报告为只读审计。代码改动（默认值/闸门可控化）见 `../docs/ARCH_ENHANCEMENT.md`
及 `config.py` 的 Track 4 提交。R1/R2 的修复需在目标机（MI250 + ROCm/hipcc）执行，
不在本机范围。
