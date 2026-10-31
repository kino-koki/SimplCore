# MIXER_STREAM：SimplCore mixer 的融合流式推理热路径

> 命名说明：本模块融合的是 **整个 mixer step**（不止卷积），因此叫 `mixer_stream`
> 而不是 sgc。SGC 专指注意力替代机制（`SpectralGlobalConv` 的振荡全局卷积），
> 它只是 mixer 内部的一个分量。命名纪律详见 `docs/SGC_CORE.md` 与 Agent Skill
> `simplcore-architecture`。
>
> 本热路径有**两个平行模块**：`mixer_stream`（SGC，time-invariant IIR）与
> `srf_stream`（SRF，input-dependent selective SSM）。二者数学不同、各自守护测试，
> 详见 §3.2。

## 1. 是什么

把 `SpectralGlobalConv.step()` 的十个 PyTorch 小算子：

```
RMSNorm → in_proj → 内容门控(L1) → IIR 振荡递推 → 局部卷积
        → 证据门(L2) → 不确定门(L3) → out_proj → 输出门 → 残差
```

融合成**单遍 C++/CUDA 算子** `mixer_stream_step`，O(1) 状态原地更新。
训练走 FFT（`fft_causal_conv`），推理走本热路径——两者与纯 PyTorch `step()`
三方在 fp32 下逐位近似（仓库守护阈值 **4e-5**）。

## 2. 文件地图

| 文件 | 角色 |
|---|---|
| `simplcore/csrc/mixer_stream.h` | 零依赖核心（只 include `<cmath>/<cstring>`）：`MixerStreamWeights`、`resonator_conv_and_local`、`mixer_stream_step_host`（CPU 参考） |
| `simplcore/csrc/mixer_stream_ext.cpp` | torch C++ 扩展入口，pybind 注册 `mixer_stream_step`，设备分发 |
| `simplcore/csrc/mixer_stream_cuda.cu` | CUDA kernel（`mixer_kernel`，block=256/batch；并行归约 + 无跨线程竞争，v2） |
| `simplcore/csrc/test_mixer_stream.cpp` | 独立数值核验（与头文件不同代码路径的 reference 交叉验证） |
| `simplcore/ops/mixer_stream.py` | JIT 加载 + `mixer_stream_step()` / `is_available()` / `self_test()` |
| `simplcore/core/tests/test_mixer_stream.py` | pytest 守护：等价性 / 严格因果 / 末位扰动不影响前文 |

## 3. 接入方式

```python
cfg = SimplCoreConfig(..., use_mixer_stream=True)   # 默认 False
# 或对单个 mixer：
mixer.use_mixer_stream = True
```

`step()` 在满足 **fp32/bf16/fp16 + 3D 输入 (B,1,D) + `ops.is_available()`** 时下沉到
C++/CUDA；任一条件不满足自动回退纯 PyTorch。(B,D) 直调、扩展未构建回退。

### 3.1 低精度（bf16 / fp16）策略

MI250 主用 bf16，而融合内核的 IIR 振荡递推对数值最敏感，必须跑在 fp32 才能保证
与纯 PyTorch step 的 4e-5 等价守护。因此 `ops.mixer_stream.mixer_stream_step` 在 Python
层做归一化：

1. 非 fp32 输入（bf16/fp16）先把 `x`、全部权重张量（`norm/in/gate/out/outgate/coef/
   rho/theta/local_w/feed`）与状态张量（`y1/y2/conv_buf/energy_sum`）**统一 cast 到 fp32**；
2. 把 fp32 张量下传 C++/CUDA，数值敏感的递推恒在 fp32 执行；
3. 输出 `y_t` 再 cast 回原输入精度，fp32 状态副本 `.copy_` 回原精度状态张量。

这样带来两个好处（诚实工程）：

* **MI250 用得上融合算子**：bf16 推理/服务不再静默回退纯 PyTorch；
* **消除潜伏 bug**：此前若强行放开 fp32 守卫，权重/状态会以 bf16 字节被当成 fp32
  读取（位 reinterpret），结果是错的；归一化方案从根上杜绝。

对 bf16 而言，融合路径（内部 fp32）比纯 PyTorch 的 bf16 step（全程 bf16）**更精确**。
fp32 输入走零拷贝快路径，与 `test_mixer_stream.py` 的守护逐位一致，不受影响。

> **与全局策略一致**：同样的"递推恒 fp32"规则现已贯穿到**所有 mixer 的纯 PyTorch
> 流式路径**（SGC / SRF / GatedSpectralConv2），由 `simplcore.core.sgc.recurrent_dtype`
> 作为单一事实源——`init_state` 把递推状态按 `recurrent_dtype(compute_dtype)` 分配
> （bf16/fp16 时退化为 fp32），`step` 在状态 dtype 中递推、输出再 cast 回计算精度。
> 因此 MI250 上即使融合扩展未构建（回退纯 PyTorch），流式递推仍然是数值可信的 fp32；
> 融合算子只是把同一策略搬进 C++/CUDA 以获得速度。详见 `SKILL.md` §2 低精度数值策略。

### 3.2 SRF 融合（平行模块，`srf_stream`）

SRF（`SpectralResonanceFlow`）与 SGC 是不同 mixer 家族——前者是 **input-dependent
selective SSM**（复数递推、步长 `Δ=softplus(dt_proj(u))` 由输入预测），后者是
time-invariant IIR。二者数学不同，故融合也分模块、分守护测试，**不共用一套 kernel**：

| 文件 | 角色 |
|---|---|
| `simplcore/csrc/srf_stream.h` | SRF 融合零依赖核心：`SrfStreamWeights`、`srf_recurrence_and_local`（复数 SSM 递推+局部卷积）、`srf_stream_step_host`（CPU 参考） |
| `simplcore/csrc/srf_stream_ext.cpp` | SRF 融合 torch 扩展入口，pybind 注册 `srf_stream_step`，CPU/CUDA 分发 |
| `simplcore/csrc/srf_stream_cuda.cu` | SRF 融合 CUDA kernel `srf_kernel`（与 SGC 同款并行归约；多一个 `delta` 的块内点积归约相位） |
| `simplcore/ops/srf_stream.py` | SRF 融合 JIT 加载 + `srf_stream_step()` / `srf_is_available()` / `srf_self_test()` |
| `simplcore/core/tests/test_srf_stream.py` | pytest 守护：fp32 融合 vs 纯 PyTorch（≤1e-3）、bf16 可运行性（扩展未构建时 skip） |

接入方式与 SGC 对称：`SpectralResonanceFlow(use_mixer_stream=True)`（或经
`registry._build_srf` 由 `config.use_mixer_stream` 传入）→ `step()` 在同样条件下下沉
`srf_stream_step`。本机（无 torch）由 `pytest` 自动 skip；fp32 等价守护阈值 1e-3（SRF
算子链更长，留足并行归约的求和顺序差异余量）。CUDA 路径的实跑与 fp32 等价性同样**必须
在含 torch + nvcc 的目标机验证**后方可启用。

## 4. 构建与验证

| 层级 | 命令 | 环境 | 期望 |
|---|---|---|---|
| 独立核验 | `g++ -std=c++17 -O2 -I. simplcore/csrc/test_mixer_stream.cpp -o t && ./t` | 任意 C++17 编译器（无需 torch） | `worst\|Δ\| = 0.000e+00` PASS |
| pytest 守护 | `pytest simplcore/core/tests/test_mixer_stream.py` | torch（首次自动 JIT 构建扩展；CUDA 另需 nvcc） | 3 项通过或（未构建时）skip |
| pytest 守护（SRF） | `pytest simplcore/core/tests/test_srf_stream.py` | torch（首次自动 JIT 构建 `srf_stream` 扩展） | fp32 等价 + bf16 可运行，未构建时 skip |
| 运行时自检 | `python -c "from simplcore.ops import self_test; print(self_test())"` | torch | `True` |
| 运行时自检（SRF） | `python -c "from simplcore.ops import srf_self_test; print(srf_self_test())"` | torch | `True` |

状态更新契约：`y1/y2/conv_buf/energy_sum` 由算子**原地**更新（传入张量共享存储），
`length` 由 Python 侧推进。

## 5. 诚实边界

- **CPU 路径数学已证明**：独立 reference（先算 long 再统一推进 IIR，与融合实现的
  原地推进是两条不同代码路径）交叉验证，fp32 零误差（`worst|Δ|=0`）。
- **CUDA kernel 未在本机实测**（开发机无 nvcc）：源码以正确性 + 效率并重编写，
  **v2 修复了一处真实的跨线程竞争**——初版的 phase-1 在"写 `hb[d]` 的同时读
  `hb[j]`（j 属其它线程）"，而 `__syncthreads()` 在循环之后，`hb` 又是未初始化的
  全局 scratch，跨线程会读到垃圾，CUDA 路径会产出错误结果（纯 PyTorch / CPU 串行
  参考无此问题，且本机无 nvcc 导致 pytest 一直 skip，故此前从未暴露）。现把 h 的
  写与矩阵读拆成两趟 + 同步，消除竞争；三个 per-b 归约（RMSNorm 平方和 / 证据能量
  和 / L3 长-短分歧）由 `thread0` 串行改为 warp-shuffle + 共享内存的并行归约，conv_buf
  移位也逐 d 并行。并行归约改变了求和顺序，与串行 CPU 参考会有 ≤1e-6 量级的相对差，
  远在守护阈值 1e-4 之内。
- **上生产前必须在目标机实测**：含 torch+CUDA+nvcc 的目标机由
  `pytest simplcore/core/tests/test_mixer_stream.py` 守护验证 fp32 等价性
  （阈值 1e-4）后方可启用 CUDA 路径；任一失败 `is_available()` 仍返回 False 回退。
- **绝不悄悄出错**：torch 缺失 / 编译失败 → `is_available()` 返回 False → 回退
  纯 PyTorch；失败原因可通过 `build_error()` 诊断。
- 无 autograd：这是推理热路径，训练始终走 FFT 并行形式。
