#!/bin/bash
# SimplCore Mix 训练启动脚本

set -e

# 默认配置
CONFIG="${1:-1b}"
OUTPUT_DIR="${2:-./output/mix}"
DATA_PATH="${3:-./data/train.txt}"

# 训练超参
BATCH_SIZE=4
GRAD_ACCUM=8
MAX_STEPS=100000
WARMUP_STEPS=2000
LR=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# 硬件
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "Detected $NUM_GPUS GPUs"
else
    NUM_GPUS=1
    echo "No GPUs detected, using CPU"
fi

echo "========================================"
echo "SimplCore Mix Training"
echo "========================================"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Data: $DATA_PATH"
echo "GPUs: $NUM_GPUS"
echo "========================================"

# 单卡训练
if [ "$NUM_GPUS" -eq 1 ]; then
    python -m SimplCore.mix.training \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --data_path "$DATA_PATH" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --max_steps "$MAX_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --lr "$LR" \
        --min_lr "$MIN_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --grad_clip "$GRAD_CLIP" \
        --mixed_precision
else
    # 分布式训练
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        -m SimplCore.mix.training \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --data_path "$DATA_PATH" \
        --batch_size "$BATCH_SIZE" \
        --grad_accum "$GRAD_ACCUM" \
        --max_steps "$MAX_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --lr "$LR" \
        --min_lr "$MIN_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --grad_clip "$GRAD_CLIP" \
        --mixed_precision
fi

echo "========================================"
echo "Training complete!"
echo "Output: $OUTPUT_DIR"
echo "========================================"
