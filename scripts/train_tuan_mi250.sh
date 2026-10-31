#!/bin/bash
# ========================================================================
# 青小团 / Tuan · AMD Instinct MI250 训练启动器（SimplCore 架构，SGC mixer）
# ========================================================================
# 适用：1× MI250（双 GCD，ROCm 暴露为 2 张卡）。目标模型 tuan-1b（~1B，
# 20–35B tokens，见 docs/TRAINING_BUDGET_MI250.md）。
#
# 用法：
#   bash scripts/train_tuan_mi250.sh --preset tuan-1b --micro-batch 4 \
#       --global-batch-tokens 2000000 --max-steps 100000 \
#       --data-dir data/processed/tuan_zh
#
# 首次上卡前必做（见 docs/TRAINING_RUNBOOK.md §2）：
#   python scripts/probe_batch.py --preset tuan-1b --seq-len 4096
# 用它的输出填 --micro-batch（留 10% 余量）。
# ========================================================================
set -euo pipefail

# ---------- 默认参数 ----------
PRESET="tuan-1b"
NUM_GCD=2                       # 1× MI250 = 2 个 GCD；单 GCD 卡改成 1
SEQ_LEN=4096
MICRO_BATCH=4
GLOBAL_BATCH_TOKENS=2000000
MAX_STEPS=100000
LR=3e-4
LR_MIN=3e-5
WARMUP=200
GRAD_CLIP=1.0
EVAL_EVERY=250
SAVE_EVERY=1000
TIME_BUDGET_MIN=0               # >0 时到点写完 checkpoint 退出（抢占式/租用实例）
OUT_DIR="runs/tuan-1b"
DATA_DIR=""
RESUME=""
COMPILE=1                       # 1=开 torch.compile(max-autotune)→ROCm 走 CK 内核
CUDA_GRAPH=0                    # 1=开 HIP Graph（实验性，失败自动回退 eager）
EARLY_STOP_PATIENCE=8           # 连续 N 次 eval 无 val 改善即停（0=关）
EARLY_STOP_MIN_DELTA=1e-3
RESTORE_BEST=1                  # 1=结束回滚到最佳 val 检查点（抗记忆过拟合）

# ---------- ROCm / MI250 环境（必须在进程启动前设置）---------------------
# 双 GCD 都用上；多 GCD 跨卡通信偶尔需要细粒度 PCIe 映射。
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
# 展开式显存段：128GB HBM 上减少碎片（train.py 也会 setdefault，这里显式化）。
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
# 每 GCD 一个训练进程，限制 OpenMP 线程避免 CPU  oversubscription。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---------- 解析覆盖参数 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)            PRESET="$2"; shift 2;;
        --num-gcd)           NUM_GCD="$2"; shift 2;;
        --seq-len)           SEQ_LEN="$2"; shift 2;;
        --micro-batch)       MICRO_BATCH="$2"; shift 2;;
        --global-batch-tokens) GLOBAL_BATCH_TOKENS="$2"; shift 2;;
        --max-steps)         MAX_STEPS="$2"; shift 2;;
        --lr)                LR="$2"; shift 2;;
        --lr-min)            LR_MIN="$2"; shift 2;;
        --warmup)            WARMUP="$2"; shift 2;;
        --grad-clip)         GRAD_CLIP="$2"; shift 2;;
        --eval-every)        EVAL_EVERY="$2"; shift 2;;
        --save-every)        SAVE_EVERY="$2"; shift 2;;
        --time-budget-min)   TIME_BUDGET_MIN="$2"; shift 2;;
        --out-dir)           OUT_DIR="$2"; shift 2;;
        --data-dir)          DATA_DIR="$2"; shift 2;;
        --resume)            RESUME="$2"; shift 2;;
        --no-compile)        COMPILE=0; shift;;
        --cuda-graph)        CUDA_GRAPH=1; shift;;
        --no-early-stop)     EARLY_STOP_PATIENCE=0; RESTORE_BEST=0; shift;;
        --no-restore-best)   RESTORE_BEST=0; shift;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

# ---------- 启动前自检 ----------
python -c "import torch, sys; \
av=torch.cuda.is_available(); \
print('torch', torch.__version__, '| cuda_avail', av, \
'| device', (torch.cuda.get_device_name(0) if av else 'CPU')); \
import os; print('HIP' if getattr(torch.version,'hip',None) else 'CUDA/NONE', \
'(ROCR_VISIBLE_DEVICES='+os.environ.get('ROCR_VISIBLE_DEVICES','?')+')'); \
sys.exit(0 if av else 1)" \
    || { echo "[FATAL] torch.cuda 不可用或未探测到 MI250，先解决驱动/ROCm" >&2; exit 1; }

mkdir -p "$OUT_DIR"

# ---------- 组装 train.py 参数 ----------
ARGS=(
    --preset "$PRESET"
    --seq-len "$SEQ_LEN"
    --micro-batch "$MICRO_BATCH"
    --global-batch-tokens "$GLOBAL_BATCH_TOKENS"
    --max-steps "$MAX_STEPS"
    --lr "$LR" --lr-min "$LR_MIN" --warmup "$WARMUP"
    --grad-clip "$GRAD_CLIP"
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY"
    --out-dir "$OUT_DIR"
)
[[ -n "$DATA_DIR" ]] && ARGS+=(--data-dir "$DATA_DIR")
[[ -n "$RESUME" ]]   && ARGS+=(--resume "$RESUME")
[[ "$TIME_BUDGET_MIN" -gt 0 ]] && ARGS+=(--time-budget-min "$TIME_BUDGET_MIN")

# 编译 / 图 / 早停（与 simplcore.core.train 的开关一一对应）
[[ "$COMPILE" -eq 1 ]] && ARGS+=(--compile --compile-mode max-autotune)
[[ "$CUDA_GRAPH" -eq 1 ]] && ARGS+=(--cuda-graph)
[[ "$EARLY_STOP_PATIENCE" -gt 0 ]] && ARGS+=(
    --early-stop-patience "$EARLY_STOP_PATIENCE"
    --early-stop-min-delta "$EARLY_STOP_MIN_DELTA"
)
[[ "$RESTORE_BEST" -eq 1 ]] && ARGS+=(--restore-best)

# ---------- 启动（torchrun DDP，nproc=GCD 数）--------------------------
echo "[MI250] launching torchrun --nproc_per_node=$NUM_GCD -m simplcore.core.train"
echo "[MI250] ARGS: ${ARGS[*]}"
echo "[MI250] env: ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF OMP_NUM_THREADS=$OMP_NUM_THREADS"

torchrun --nproc_per_node="$NUM_GCD" -m simplcore.core.train "${ARGS[@]}" \
    || { echo "[FATAL] training exited non-zero; 见 $OUT_DIR/metrics.jsonl 与 run.json" >&2; exit 1; }

echo "[MI250] done. 最终模型: $OUT_DIR/last.pt（RESTORE_BEST=1 时即最佳 val 检查点）"
