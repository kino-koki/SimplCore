#!/bin/bash
# ========================================================================
# Tuan-2B · 单 MI250（2×64 GB GCD）· 30 天 · 4M 上下文阶段课程
# ========================================================================
# 预算口径（docs/TRAINING_2B_MI250_30D.md）：
#   30d × 86400 × 0.65 util × 362.1 TFLOPS × 25% MFU = 1.525e20 FLOPs
#   tuan-2b（2024M，非嵌入 1959M）≈ 11.83e9 FLOPs/token
#   → 上限 ≈ 12.9B tokens。本脚本按 12.4B tokens 排课，硬停 30 天。
#
# 这不是 Chinchilla-optimal（20× = 40B tokens，需 ~92 天）。"完全训练"在
# 本脚本里指：跑完 4M 课程、产出可用的 2B 主干 + 4M 可运行/基础检索能力。
#
# 用法（上卡第一件事：probe_batch 定 micro-batch/chunk，见 RUNBOOK §2）：
#   bash scripts/train_tuan_2b_mi250.sh \
#       --data-dir data/processed/tuan_zh --out-dir runs/tuan-2b
#
# 分段课程（每段 resume 上一段 last.pt；step 累计）：
#   seq_len   micro  tokens/step  steps  tokens   估时(25% MFU)
#   4K        2      1.0M         2300   2.30B    ~3.5d
#   16K       2      1.02M        2300   2.34B    ~3.6d
#   64K       2      1.05M        2300   2.41B    ~3.7d
#   256K      1      1.05M        1950   2.04B    ~4.4d
#   1M        1      1.05M        1950   2.04B    ~4.6d
#   4M        1      4.0M         325    1.30B    ~5.0d
#   合计 ~12.4B tokens / ~25d 纯算力；+评估/重启/数据 → 30d 硬停
# ========================================================================
set -euo pipefail

PRESET="tuan-2b"
NUM_GCD=2
CHUNK=4096                  # 梯度窗口；micro×chunk 控制在 ~8192 token 以内
MICRO_OVERRIDE=""           # 非空则覆盖所有阶段的 micro-batch
GLOBAL_OVERRIDE=""          # 非空则覆盖所有阶段的 global-batch-tokens
SEQ_OVERRIDE=""             # 非空则只跑单阶段（调试）
OUT_DIR="runs/tuan-2b"
DATA_DIR=""
LR=2.4e-4
LR_MIN=2.4e-5
WARMUP=2000
GRAD_CLIP=1.0
EVAL_EVERY=500
EVAL_BATCHES=2
SAVE_EVERY=1000
MAX_MINUTES=0               # >0：总时长硬停（分钟），到点写完 checkpoint 退出
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)      PRESET="$2"; shift 2;;
        --num-gcd)     NUM_GCD="$2"; shift 2;;
        --chunk)       CHUNK="$2"; shift 2;;
        --micro-batch) MICRO_OVERRIDE="$2"; shift 2;;
        --global-batch-tokens) GLOBAL_OVERRIDE="$2"; shift 2;;
        --seq-len)     SEQ_OVERRIDE="$2"; shift 2;;
        --out-dir)     OUT_DIR="$2"; shift 2;;
        --data-dir)    DATA_DIR="$2"; shift 2;;
        --lr)          LR="$2"; shift 2;;
        --lr-min)      LR_MIN="$2"; shift 2;;
        --warmup)      WARMUP="$2"; shift 2;;
        --eval-every)  EVAL_EVERY="$2"; shift 2;;
        --save-every)  SAVE_EVERY="$2"; shift 2;;
        --max-minutes) MAX_MINUTES="$2"; shift 2;;
        --dry-run)     DRY_RUN=1; shift;;
        -h|--help)     sed -n '1,40p' "$0"; exit 0;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

# ---------- ROCm / MI250 环境（必须在进程启动前设置）---------------------
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# 长上下文分块训练更依赖 host 侧数据搬运：让 DataLoader 线程尽量重叠。
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "$DATA_DIR" ]]; then
    echo "[FATAL] 必须指定 --data-dir（packed 语料，manifest 需含 train/val）" >&2
    exit 1
fi
if [[ ! -f "$DATA_DIR/index.json" ]]; then
    echo "[FATAL] $DATA_DIR/index.json 不存在；先跑 python -m simplcore.core.data --help" >&2
    exit 1
fi

echo "[MI250-2B] preset=$PRESET chunk=$CHUNK gcd=$NUM_GCD out=$OUT_DIR"
echo "[MI250-2B] data=$DATA_DIR"
echo "[MI250-2B] env: ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE"
echo "[MI250-2B] PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF OMP_NUM_THREADS=$OMP_NUM_THREADS"

mkdir -p "$OUT_DIR"
STEP_END=0
TOTAL_MINUTES=0

# seq_len  micro  global_batch_tokens  steps
STAGES=(
    "4096      2      1000000            2300"
    "16384     2      1000000            2300"
    "65536     2      1000000            2300"
    "262144    1      1000000            1950"
    "1048576   1      1000000            1950"
    "4000000   1      4000000            325"
)

run_stage() {
    local seq="$1" micro="$2" gbt="$3" steps="$4"
    [[ -n "$MICRO_OVERRIDE" ]] && micro="$MICRO_OVERRIDE"
    [[ -n "$GLOBAL_OVERRIDE" ]] && gbt="$GLOBAL_OVERRIDE"
    STEP_END=$((STEP_END + steps))

    local args=(
        --preset "$PRESET"
        --seq-len "$seq"
        --micro-batch "$micro"
        --global-batch-tokens "$gbt"
        --max-steps "$STEP_END"
        --train-chunk "$CHUNK"
        --lr "$LR" --lr-min "$LR_MIN" --warmup "$WARMUP"
        --grad-clip "$GRAD_CLIP"
        --eval-every "$EVAL_EVERY" --eval-batches "$EVAL_BATCHES"
        --save-every "$SAVE_EVERY"
        --out-dir "$OUT_DIR"
        --data-dir "$DATA_DIR"
        --seed 1337
    )
    if [[ "$seq" -ge 1048576 ]]; then
        # GATE-2：课程首次进入 1M 阶段时测一次 needle 召回（只记录，不阻塞；
        # 4M 验收用 `python -m simplcore.core.gate2` 离线跑）。
        args+=(--gate2 --gate2-lengths "262144,1048576" --gate2-chunk "$CHUNK")
    fi
    if [[ -f "$OUT_DIR/last.pt" ]]; then
        args+=(--resume "$OUT_DIR/last.pt")
    fi
    if [[ "$MAX_MINUTES" -gt 0 ]]; then
        args+=(--time-budget-min "$MAX_MINUTES")
    fi

    echo "----------------------------------------------------------------"
    echo "[MI250-2B] stage seq=$seq micro=$micro gbt=$gbt steps=$steps -> max_steps=$STEP_END"
    echo "[MI250-2B] ARGS: ${args[*]}"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return 0
    fi
    torchrun --nproc_per_node="$NUM_GCD" -m simplcore.core.train "${args[@]}" \
        || { echo "[FATAL] stage seq=$seq 失败；见 $OUT_DIR/metrics.jsonl" >&2; exit 1; }
}

if [[ -n "$SEQ_OVERRIDE" ]]; then
    # 单阶段调试：按给定 seq 选择最近的阶段参数
    for row in "${STAGES[@]}"; do
        read -r seq micro gbt steps <<< "$row"
        if [[ "$seq" == "$SEQ_OVERRIDE" ]]; then
            run_stage "$seq" "$micro" "$gbt" "$steps"
            exit 0
        fi
    done
    echo "[FATAL] --seq-len $SEQ_OVERRIDE 不在课程表内" >&2
    exit 1
fi

for row in "${STAGES[@]}"; do
    read -r seq micro gbt steps <<< "$row"
    run_stage "$seq" "$micro" "$gbt" "$steps"
done

echo "[MI250-2B] 完成：$OUT_DIR/last.pt（分阶段课程按 step 累计）"
echo "[MI250-2B] 审计：$OUT_DIR/metrics.jsonl / run.json / best.pt"