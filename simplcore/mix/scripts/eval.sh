#!/bin/bash
# SimplCore Mix 评估启动脚本

set -e

MODEL="${1:-./output/mix/final_model.pt}"
TOKENIZER="${2:-./output/mix/tokenizer.json}"
CONFIG="${3:-1b}"
OUTPUT="${4:-./output/mix/eval_results.json}"

echo "========================================"
echo "SimplCore Mix Evaluation"
echo "========================================"
echo "Model: $MODEL"
echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo "========================================"

python -m SimplCore.mix.inference \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --config "$CONFIG" \
    --prompt "Hello world" \
    --max_tokens 50

echo "========================================"
echo "Evaluation complete!"
echo "========================================"
