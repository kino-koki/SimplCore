#!/bin/bash
# 青小团 SFT 训练启动脚本 - 青奈科技
# 使用 DeepSpeed 分布式训练

set -x
set -e

# 分布式训练配置
NUM_GPUS=8
MASTER_PORT=29500
DEEPSPEED_CONFIG="ds_config_zero2.json"

# 训练参数
MODEL_NAME_OR_PATH="/path/to/qingxiaotuan-checkpoint"
DATA_PATH="/path/to/sft_data.jsonl"
OUTPUT_DIR="./output_sft"
MAX_LENGTH=2048
NUM_EPOCHS=3
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=10
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3

# 创建DeepSpeed配置文件（如果不存在）
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    cat > "$DEEPSPEED_CONFIG" << 'EOF'
{
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOF
    echo "已创建DeepSpeed配置文件: $DEEPSPEED_CONFIG"
fi

# 启动分布式训练
deepspeed \
    --num_gpus $NUM_GPUS \
    --master_port $MASTER_PORT \
    train_sft.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --max_length $MAX_LENGTH \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --deepspeed $DEEPSPEED_CONFIG

echo "SFT训练完成！模型已保存至: $OUTPUT_DIR"
