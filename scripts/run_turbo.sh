#!/bin/bash
# ========================================================================
# Qingxiaotuan LLM Turbo Distiller - Training Launcher (Linux Bash)
# ========================================================================
# Usage: bash scripts/run_turbo.sh [options]
# ========================================================================

set -euo pipefail

# ---------- Trap ----------
SCRIPT_PID=$$
trap 'rc=$?; if [ $rc -ne 0 ]; then echo -e "\n${RED}[FATAL]${NC} Script aborted with exit code $rc (line $LINENO)"; exit $rc; fi' EXIT

# ---------- Colors ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ---------- Helpers ----------
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
info()  { echo -e "${GREEN}[INFO  ]${NC} $(timestamp) | $*"; }
warn()  { echo -e "${YELLOW}[WARN  ]${NC} $(timestamp) | $*"; }
error() { echo -e "${RED}[ERROR ]${NC} $(timestamp) | $*"; }
fatal() { echo -e "${RED}[FATAL ]${NC} $(timestamp) | $*"; exit 1; }

step_start() {
    local n=$1; shift
    echo -e "\n${BOLD}${BLUE}========================================================================${NC}"
    echo -e "${BOLD}${BLUE}  Step $n: $* (start: $(timestamp))${NC}"
    echo -e "${BOLD}${BLUE}========================================================================${NC}"
}
step_done() {
    local n=$1; shift
    echo -e "\n${BOLD}${GREEN}------------------------------------------------------------------------${NC}"
    echo -e "${BOLD}${GREEN}  Step $n done (end: $(timestamp))${NC}"
    echo -e "${BOLD}${GREEN}------------------------------------------------------------------------${NC}"
}
require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        fatal "Required command '$1' not found in PATH"
    fi
}

# ---------- Argument Parsing ----------
PRESET=""
NUM_GPUS=8
HOURS=24
DATA_FILES=()
OUTPUT_DIR="./output"
SEQ_LEN=2048
MICRO_BATCH=4
GRAD_ACC=8
FP8="true"
CPU_OFFLOAD="false"
RESUME_CKPT=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)         PRESET="$2"; shift 2;;
        --num-gpus)       NUM_GPUS="$2"; shift 2;;
        --hours)          HOURS="$2"; shift 2;;
        --data)           DATA_FILES+=("$2"); shift 2;;
        --output)         OUTPUT_DIR="$2"; shift 2;;
        --seq-len)        SEQ_LEN="$2"; shift 2;;
        --micro-batch)    MICRO_BATCH="$2"; shift 2;;
        --grad-acc)       GRAD_ACC="$2"; shift 2;;
        --fp8)            FP8="true"; shift;;
        --no-fp8)         FP8="false"; shift;;
        --cpu-offload)    CPU_OFFLOAD="true"; shift;;
        --no-cpu-offload) CPU_OFFLOAD="false"; shift;;
        --resume)         RESUME_CKPT="$2"; shift 2;;
        --dry-run)        DRY_RUN="true"; shift;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [options]

Training preset:
  --preset NAME          12B_dense | 30B_dense | 60B_moe | 100B_moe | 600B_moe (required)

Resources:
  --num-gpus N           Number of GPUs (default: 8)
  --hours H              Max training hours (default: 24)

Data & output:
  --data PATH            Training JSONL (repeatable, required)
  --output PATH          Output directory (default: ./output)

Training hyperparameters:
  --seq-len N            Sequence length (default: 2048)
  --micro-batch N        Micro batch size per GPU (default: 4)
  --grad-acc N           Gradient accumulation steps (default: 8)

Optimizations:
  --fp8 | --no-fp8       FP8 mixed precision (default: on)
  --cpu-offload          ZeRO CPU offload (default: off)

Resume & debug:
  --resume PATH          Resume from checkpoint directory
  --dry-run              Run only steps 0-2, do not launch training
EOF
            exit 0
            ;;
        *) fatal "Unknown argument: $1";;
    esac
done

# ---------- Validate ----------
[[ -n "$PRESET" ]] || fatal "--preset is required"
[[ ${#DATA_FILES[@]} -gt 0 ]] || fatal "--data is required (at least one JSONL)"
for f in "${DATA_FILES[@]}"; do
    [[ -f "$f" ]] || fatal "Data file not found: $f"
done
if [[ -n "$RESUME_CKPT" ]]; then
    [[ -d "$RESUME_CKPT" ]] || fatal "Resume checkpoint dir not found: $RESUME_CKPT"
fi

# ---------- Directories & Logs ----------
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/configs"
mkdir -p "$OUTPUT_DIR/checkpoints"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$OUTPUT_DIR/logs/run_${TIMESTAMP}.log"
PID_FILE="$OUTPUT_DIR/run_${TIMESTAMP}.pid"
DS_CONFIG="$OUTPUT_DIR/configs/ds_config_${TIMESTAMP}.json"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "$$" > "$PID_FILE"

# ========================================================================
# Step 0: Environment Check
# ========================================================================
step_start 0 "环境检查 (Python / CUDA / GPU 数量)"

require_cmd python3
PYTHON_VER=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "N/A")
info "Python version: $PYTHON_VER"

if command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE_GPUS=${CUDA_VISIBLE_DEVICES:-"all"}
    ACTUAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    info "Visible GPUs (CUDA_VISIBLE_DEVICES): $VISIBLE_GPUS"
    info "Physical GPUs detected:   $ACTUAL_GPUS"
    info "Requested num-gpus:       $NUM_GPUS"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null | head -n "$((NUM_GPUS+1))" || true
else
    warn "nvidia-smi not found - GPU training may not work"
    ACTUAL_GPUS=0
fi

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VER=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -n1)
    info "CUDA toolchain: $CUDA_VER"
else
    warn "nvcc not found - proceeding anyway"
fi

python3 - <<'PY' 2>&1 || fatal "Python / PyTorch sanity check failed"
import sys, json
try:
    import torch
    has_torch = True
    ver = torch.__version__
    cuda = torch.version.cuda
    devc = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception as e:
    has_torch, ver, cuda, devc = False, "N/A", "N/A", 0
try:
    import deepspeed
    ds = deepspeed.__version__
except Exception:
    ds = "N/A"
try:
    import transformer_engine
    te = transformer_engine.__version__
except Exception:
    te = "N/A"
print(f"  torch            = {ver}")
print(f"  torch.cuda       = {cuda}")
print(f"  cuda.is_available= {torch.cuda.is_available() if has_torch else 'N/A'}")
print(f"  cuda.device_count= {devc}")
print(f"  deepspeed        = {ds}")
print(f"  transformer_engine = {te}")
PY

step_done 0

# ========================================================================
# Step 1: Budget Analysis
# ========================================================================
step_start 1 "训练预算分析 (budget_scheduler --dry-run)"

DATA_ARGS=()
for f in "${DATA_FILES[@]}"; do
    DATA_ARGS+=(--data "$f")
done

python3 -m qxt.training.budget_scheduler \
    --dry-run \
    --preset "$PRESET" \
    --num-gpus "$NUM_GPUS" \
    --hours "$HOURS" \
    --seq-len "$SEQ_LEN" \
    --micro-batch "$MICRO_BATCH" \
    --grad-acc "$GRAD_ACC" \
    --fp8 "$FP8" \
    --cpu-offload "$CPU_OFFLOAD" \
    --output "$OUTPUT_DIR" \
    "${DATA_ARGS[@]}" \
    || fatal "budget_scheduler failed"

step_done 1

# ========================================================================
# Step 2: DeepSpeed Config
# ========================================================================
step_start 2 "生成 DeepSpeed 配置 (deepspeed_config.py)"

python3 -m qxt.training.deepspeed_config \
    --preset "$PRESET" \
    --num-gpus "$NUM_GPUS" \
    --fp8 "$FP8" \
    --cpu-offload "$CPU_OFFLOAD" \
    --output "$DS_CONFIG" \
    || fatal "deepspeed_config generation failed"

info "DeepSpeed config written to: $DS_CONFIG"
step_done 2

# ========================================================================
# Step 3: Launch Training
# ========================================================================
if [[ "$DRY_RUN" == "true" ]]; then
    warn "--dry-run: skipping Step 3 (training launch) and Step 4 (report)"
    warn "Review budget analysis + DS config, then rerun without --dry-run."
    echo ""
    echo -e "${GREEN}  _______________________________"
    echo -e "${GREEN} /     DRY-RUN  COMPLETED        \\"
    echo -e "${GREEN} \\  ready to train when you are  /"
    echo -e "${GREEN}  -------------------------------"
    echo -e "${GREEN}         \   ^__^"
    echo -e "${GREEN}          \  (oo)\_______"
    echo -e "${GREEN}             (__)\       )\\/\\ /\\  "
    echo -e "${GREEN}                 ||----w |"
    echo -e "${GREEN}                 ||     ||${NC}"
    echo ""
    exit 0
fi

step_start 3 "启动 DeepSpeed 分布式训练 (turbo_distiller.py)"

if command -v deepspeed >/dev/null 2>&1; then
    info "Using 'deepspeed' launcher"
    LAUNCHER="deepspeed --num_gpus=$NUM_GPUS"
    ENTRY="$(python3 -c "import qxt.training.turbo_distiller, os; print(os.path.abspath(qxt.training.turbo_distiller.__file__))")"
else
    warn "'deepspeed' launcher not found, falling back to torchrun DDP"
    LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS"
    ENTRY="-m qxt.training.turbo_distiller"
fi

RESUME_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
    RESUME_ARGS+=(--resume "$RESUME_CKPT")
fi

set -x
eval "$LAUNCHER $ENTRY \
    --preset \"$PRESET\" \
    --seq-len \"$SEQ_LEN\" \
    --micro-batch \"$MICRO_BATCH\" \
    --grad-acc \"$GRAD_ACC\" \
    --fp8 \"$FP8\" \
    --cpu-offload \"$CPU_OFFLOAD\" \
    --hours \"$HOURS\" \
    --deepspeed-config \"$DS_CONFIG\" \
    --output \"$OUTPUT_DIR\" \
    ${RESUME_ARGS[*]+"${RESUME_ARGS[@]}"} \
    ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} \
    || fatal "Training process exited with non-zero status"
set +x

step_done 3

# ========================================================================
# Step 4: Summary Report
# ========================================================================
step_start 4 "训练结束，自动汇总报告 (monitor.py)"

python3 -m qxt.training.monitor \
    --mode summary \
    --output "$OUTPUT_DIR" \
    || warn "monitor.py summary step failed - check logs manually"

info "PID file:        $PID_FILE"
info "Log file:        $LOG_FILE"
info "Config file:     $DS_CONFIG"
info "Checkpoints:     $OUTPUT_DIR/checkpoints"
info "TensorBoard:     $OUTPUT_DIR/tb"

echo ""
echo -e "${GREEN}  ___________________________________________________________"
echo -e "${GREEN} /                                                           \\"
echo -e "${GREEN} |   QINGXIAOTUAN LLM TRAINING COMPLETED SUCCESSFULLY         |"
echo -e "${GREEN} |                                                           |"
echo -e "${GREEN} |   preset      : ${PRESET}"
echo -e "${GREEN} |   num-gpus    : ${NUM_GPUS}"
echo -e "${GREEN} |   seq-len     : ${SEQ_LEN}"
echo -e "${GREEN} |   micro-batch : ${MICRO_BATCH}"
echo -e "${GREEN} |   grad-acc    : ${GRAD_ACC}"
echo -e "${GREEN} |   fp8         : ${FP8}"
echo -e "${GREEN} |   cpu-offload : ${CPU_OFFLOAD}"
echo -e "${GREEN} |   output      : ${OUTPUT_DIR}"
echo -e "${GREEN} |   log         : ${LOG_FILE}"
echo -e "${GREEN} \\___________________________________________________________/"
echo -e "${GREEN}      \   ^__^"
echo -e "${GREEN}       \  (oo)\_______"
echo -e "${GREEN}          (__)\       )\\/\\ /\\  "
echo -e "${GREEN}              ||----w |"
echo -e "${GREEN}              ||     ||${NC}"
echo ""

step_done 4
