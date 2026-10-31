#!/bin/bash
# ========================================================================
# SimplCore · MI250/ROCm 融合热路径冒烟测试（一键）
# ========================================================================
# 目的：验证 CUDA 写法的融合核（csrc/mixer_stream_cuda.cu / srf_stream_cuda.cu）
#       能否在 MI250 上经 PyTorch 以 hipcc 编译为 HIP 并通过等价守护。
#
# 用法（在 MI250 目标机上，仓库根目录）：
#   bash scripts/smoke_hip_mi250.sh
#
# 通过标准：全部 [OK]；任何 [FAIL] 的修复指引见 docs/AUDIT_CUDA_BACKEND.md R2。
# ========================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

pass=0; fail=0
ok()   { echo " [OK]   $1"; pass=$((pass+1)); }
bad()  { echo " [FAIL] $1"; fail=$((fail+1)); }
head() { echo; echo "=== $1 ==="; }

head "0. 环境"
python - <<'PY' || bad "torch 环境探测（见上方输出）"
import torch, os
av = torch.cuda.is_available()
hip = getattr(torch.version, "hip", None)
print(f"  torch={torch.__version__} cuda_avail={av} HIP={hip}")
print(f"  device={torch.cuda.get_device_name(0) if av else 'N/A'}")
print(f"  ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES','?')}")
if not av: raise SystemExit(1)
if hip: print("  判定：ROCm/HIP 环境（.cu 将以 hipcc 编译）")
else:   print("  判定：NVIDIA/CUDA 环境（非 MI250 目标，但同样可实测）")
PY
[[ $? -eq 0 ]] && ok "torch + GPU 可用" || true

command -v hipcc >/dev/null 2>&1 && ok "hipcc 在 PATH: $(hipcc --version | tail -1)" \
                                 || echo " [NOTE] hipcc 不在 PATH（PyTorch 自带 ROCm 时正常，走 torch 的 hipcc）"

head "1. 融合扩展构建（关键步骤：hipcc 编译 .cu）"
python - <<'PY'
import sys
sys.path.insert(0, ".")
from simplcore.ops import mixer_stream as ms
from simplcore.ops import srf_stream as ss
for name, mod in (("mixer_stream", ms), ("srf_stream", ss)):
    if mod.is_available():
        print(f"  {name}: 构建成功")
    else:
        err = mod.build_error()
        print(f"  {name}: 构建失败 → {type(err).__name__}")
        print(str(err)[-2000:])
        sys.exit(1)
PY
[[ $? -eq 0 ]] && ok "mixer_stream + srf_stream 扩展均构建成功" \
               || bad "构建失败：最常见为 __shfl_down_sync mask 变体/未设 PYTORCH_ROCM_ARCH=gfx90a"

head "2. 等价守护（fp32 4e-5 + bf16 精度路径）"
if python -c "from simplcore.ops import is_available" 2>/dev/null \
   && python -c "import simplcore.ops as o; sys.exit(0 if o.mixer_stream.is_available() else 1)" 2>/dev/null \
   || python -c "from simplcore.ops.mixer_stream import is_available; import sys; sys.exit(0 if is_available() else 1)"; then
    python -m pytest simplcore/core/tests/test_mixer_stream.py \
                     simplcore/core/tests/test_srf_stream.py \
                     simplcore/core/tests/test_stream_precision.py -x -q \
        && ok "流式等价/精度测试全部通过" \
        || bad "等价守护失败：并行归约求和序差异超阈值或存在真实数值 bug"
else
    bad "扩展不可用，跳过等价守护（先解决步骤 1）"
fi

head "3. 需人工确认的点（不阻塞，但必须过目）"
echo "  a) __shfl_down_sync(0xffffffff,...)：HIP 支持该变体；若编译报错，"
echo "     改为 HIP 原生 __shfl_down 或 shfl_down_sync 宏分支。"
echo "  b) 编译 flag：确认无 -ffp-contract=fast（IIR 递推禁 FMA 收缩，"
echo "     extra_cuda_cflags 建议加 '-ffp-contract=off'）。"
echo "  c) bf16 I/O + fp32 状态路径已由 test_stream_precision.py 覆盖——"
echo "     MI250 主用 bf16，该项必须真跑过而非 skip。"
echo "  d) 并行归约（warp-shuffle）与 CPU 串行参考的 ≤1e-6 相对差："
echo "     对照守护阈值 1e-4 是否成立（.cu 文件头的声明至今无实测支撑）。"

head "结果"
echo "  OK=$pass FAIL=$fail"
[[ $fail -eq 0 ]] && echo "  冒烟通过 ✓ —— 融合热路径在 MI250 上可用" \
                  || echo "  存在失败项 —— 修复指引见 docs/AUDIT_CUDA_BACKEND.md"
exit $fail
