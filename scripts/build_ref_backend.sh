#!/bin/bash
# ========================================================================
# SimplCore · 参考后端（fp32 CPU, 纯 C99）构建脚本
# ========================================================================
# 产出：csrc/build/libsc_ref_backend.so  (Linux/ROCm)
#       csrc/build/sc_ref_backend.dll    (Windows/MinGW)
#
# 用途：让 backend_bridge.load_backend 首次跑通真实共享库路径，
#       conformance_runner 9 项检查可在任意装 torch 的机器上对它执行：
#         python -c "from simplcore.core.backend_bridge import load_backend; \
#                    from simplcore.core.conformance_runner import run_conformance; \
#                    rep = run_conformance(load_backend('simplcore/csrc/build/libsc_ref_backend.so')); \
#                    print(rep.format())"
#
# 数值纪律：-ffp-contract=off 禁止 FMA 收缩（P0-chunked-carry 逐位一致的前提）。
# ========================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p simplcore/csrc/build

SRC=simplcore/csrc/sc_ref_backend.c
CFLAGS="-std=c99 -O2 -ffp-contract=off -fvisibility=hidden -Wall -Wextra"

if [[ "${OS:-}" == "Windows_NT" ]]; then
    OUT=simplcore/csrc/build/sc_ref_backend.dll
    gcc $CFLAGS -shared -o "$OUT" "$SRC" -lm
else
    OUT=simplcore/csrc/build/libsc_ref_backend.so
    gcc $CFLAGS -shared -fPIC -o "$OUT" "$SRC" -lm
fi

echo "[ref-backend] built: $OUT"
echo "[ref-backend] 符号自检："
if [[ "${OS:-}" == "Windows_NT" ]]; then
    objdump -p "$OUT" | grep -c "sc_" || true
else
    nm -D --defined-only "$OUT" | grep " sc_" | awk '{print "  " $3}'
fi
