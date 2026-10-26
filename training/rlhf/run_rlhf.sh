#!/bin/bash
# 青小团 RLHF (PPO) 训练启动脚本 - 青奈科技
# 使用 DeepSpeed 分布式训练

set -x
set -e

# 分布式训练配置
NUM_GPUS=8
MASTER_PORT=29501
DEEPSPEED_CONFIG="ds_config_zero2.json"

# 模型路径
SFT_MODEL_PATH="/path/to/sft-checkpoint"
REWARD_MODEL_PATH="/path/to/reward-model-checkpoint"

# 训练数据
DATA_PATH="/path/to/ppo_data.jsonl"

# 输出目录
OUTPUT_DIR="./output_rlhf"

# PPO超参数
MAX_LENGTH=2048
MAX_PROMPT_LENGTH=1024
LEARNING_RATE=1e-6
BATCH_SIZE=32
MINI_BATCH_SIZE=4
KL_PENALTY=0.1
NUM_TRAIN_EPOCHS=3
NUM_PPO_EPOCHS=4
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

# 启动分布式PPO训练
deepspeed \
    --num_gpus $NUM_GPUS \
    --master_port $MASTER_PORT \
    train_ppo.py \
    --sft_model_path $SFT_MODEL_PATH \
    --reward_model_path $REWARD_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --max_length $MAX_LENGTH \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --mini_batch_size $MINI_BATCH_SIZE \
    --kl_penalty $KL_PENALTY \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --num_ppo_epochs $NUM_PPO_EPOCHS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --deepspeed $DEEPSPEED_CONFIG

echo "RLHF (PPO) 训练完成！模型已保存至: $OUTPUT_DIR"
