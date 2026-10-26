#!/bin/bash
# SimplCore Mix 推理启动脚本

MODEL="${1:-./output/mix/final_model.pt}"
TOKENIZER="${2:-./output/mix/tokenizer.json}"
CONFIG="${3:-1b}"

echo "========================================"
echo "SimplCore Mix Interactive Inference"
echo "========================================"
echo "Model: $MODEL"
echo "Config: $CONFIG"
echo "========================================"

python -m SimplCore.mix.inference \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --config "$CONFIG"
