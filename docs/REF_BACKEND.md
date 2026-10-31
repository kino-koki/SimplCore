# 参考后端（fp32 CPU, 纯 C99）—— `load_backend` 的第一个真实共享库

日期：2026-10-29 · 组件：`simplcore/csrc/sc_ref_backend.c`、
`scripts/build_ref_backend.sh`、`scripts/test_ref_backend.py`

## 定位（诚实边界）

这是**参考实现**，不是 MI250 生产后端：

| 维度 | 参考后端（本文件） | 生产后端（HIP，待移植） |
|------|------|------|
| 语言/平台 | 纯 C99，CPU，无依赖 | HIP（由 `mixer_stream_cuda.cu` 起） |
| 精度 | fp32 单一契约 | bf16 计算 / fp32 累加（MI250） |
| 谱卷积 | 直接时域因果卷积 O(T²·D)（契约允许的等效形式） | FFT O(L log L)（rocFFT） |
| 适用规模 | CI profile（L≈4096）一致性验收 | full profile（L=4M）+ 训练/服务 |
| 作用 | 让 `load_backend` 路径首次跑通**真实编译库** | 战役运行时本体 |

它解决的问题：此前 `backend_bridge.load_backend` 对任何真实库都会因缺
`sc_*` 符号失败，Phase 0 验收只能靠 `FakeCLib`（Python 替身）自证编组层——
dlopen / 符号校验 / 版本协商 / ctypes 编组的完整链路从未被真实锻炼。

## 构建与验证

```bash
bash scripts/build_ref_backend.sh      # → csrc/build/libsc_ref_backend.so / .dll
python scripts/test_ref_backend.py     # 16 项本地数值自测（纯 stdlib+ctypes）
```

自测内容：逐内核对照测试内嵌的 Python 参考数学（逐式镜像
`TorchReferenceAdapter` / `gates.py` / `sgc.py`），最差相对误差 ~1e-7；
外加两项关键检查——**chunked vs 整段 IIR 扫描字节级一致**（P0-chunked-carry
同构）与 evidence 并行/分块口径一致。全部通过（2026-10-29，MinGW gcc/ucrt64）。

## 在装 torch 的目标机上跑 Phase 0 一致性验收

```bash
python -c "
from simplcore.core.backend_bridge import load_backend
from simplcore.core.conformance_runner import run_conformance
rep = run_conformance(load_backend('simplcore/csrc/build/libsc_ref_backend.so'))
print(rep.format())
raise SystemExit(0 if rep.all_passed else 1)
"
```

这会执行全部 9 项检查（7 项契约 + C-rollback + C-stream-serialize），
`exit 0` 即 P0-backend-conformance 闸门达成（参考后端口径）。

## 同步修复的桥接层缺陷（R8）

**ABI v1 不传 dtype**，而 P0-bf16-accum 检查会给 `gemm/rmsnorm/ce_loss` 喂
bf16——旧 `CtypesBackendAdapter` 把 bf16 原始字节直传内核，`FakeCLib` 则按
fp32 解读，**必然产生垃圾值**。修复方式与仓库既有哲学一致
（`ops/mixer_stream.py`「内核保持 fp32 单一契约」）：

* `CtypesBackendAdapter` 的 `gemm/rmsnorm/swiglu/residual_add/
  causal_depthwise_conv/content_gate/ce_loss` 现在在桥接层把非 fp32 输入
  升为 fp32，输出按原 dtype 回写——数学上与 `TorchReferenceAdapter`
  （`(x.float() op).to(orig)`）逐式相同；
* 真实内核（含本参考后端与未来 HIP 后端）从此**永远只见 fp32**，
  dtype 分派收口在桥接层一处。

## 已知边界

1. `spectral_causal_conv` 为 O(B·T²·D) 直接卷积：CI profile（L=4096,
   D=16, B=2）秒级；**不要**用参考后端跑 `--full`（L=4M 需 FFT 后端）。
2. `ABI_SPEC` 中 `chunked_iir_scan` 把 `use_coef` 列在 `scalars`，实际
   桥接层按 `iargs`（int64）传递——文档性偏差，以桥接层行为为准。
3. dual-form 检查（torch FFT vs 本后端直接卷积，阈值 4e-5）在 L=4096
   的 fp32 舍入风险已由同型 torch 检查覆盖过；若目标机 borderline，
   优先核对 kernel 构造的 `psi` 精度而非放宽阈值。
