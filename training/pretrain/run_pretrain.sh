#!/bin/bash
# 青小团 Qingxiaotuan 分布式预训练启动脚本
# 512K长序列 + 12步梯度累积
# 5节点 × 8卡 = 40 GPU (推荐配置)

set -x

DATA_PATH="/path/to/tokenized_data"
OUTPUT_DIR="./checkpoints_qingxiaotuan_512K"
MODEL_NAME_OR_PATH=""  # 留空则从头训练
TOKENIZER_PATH="qingxiaotuan"

# ========== 512K长序列训练超参数 ==========
PER_DEVICE_BATCH_SIZE=1       # 512K下每卡1条
GRADIENT_ACCUMULATION_STEPS=12 # 12步梯度累积
LEARNING_RATE=3e-4
MIN_LR=3e-5
WEIGHT_DECAY=0.1
WARMUP_STEPS=2000
MAX_TRAIN_STEPS=100000
SAVE_STEPS=1000
LOGGING_STEPS=10
MAX_LENGTH=524288              # 512K 序列长度

# ========== 长上下文特性开关 ==========
FLASH_ATTENTION=1              # 启用Flash Attention
SEQUENCE_PARALLEL=1            # 启用序列并行
PACK_SEQUENCES=1               # 启用动态序列打包
GRADIENT_CHECKPOINTING=1       # 启用梯度检查点
ZERO_STAGE=3                   # ZeRO-3 (模型参数分片)

# ========== 节点配置 ==========
MASTER_ADDR="node-0"
MASTER_PORT=29500
NNODES=5
NPROC_PER_NODE=8               # 推荐每节点8卡 (A100 80GB)

# ========== 日志配置 ==========
LOG_DIR="./logs_512K"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

deepspeed \
    --hostfile hostfile \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --num_nodes ${NNODES} \
    --num_gpus ${NPROC_PER_NODE} \
    train.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --max_length ${MAX_LENGTH} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --min_learning_rate ${MIN_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_steps ${WARMUP_STEPS} \
    --max_train_steps ${MAX_TRAIN_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --seed 42 \
    --zero_stage ${ZERO_STAGE} \
    $( [[ ${FLASH_ATTENTION} -eq 1 ]] && echo "--flash_attention" ) \
    $( [[ ${SEQUENCE_PARALLEL} -eq 1 ]] && echo "--sequence_parallel" ) \
    $( [[ ${PACK_SEQUENCES} -eq 1 ]] && echo "--pack_sequences" ) \
    $( [[ ${GRADIENT_CHECKPOINTING} -eq 1 ]] && echo "--gradient_checkpointing" ) \
    2>&1 | tee ${LOG_DIR}/pretrain_512K_${TIMESTAMP}.log
