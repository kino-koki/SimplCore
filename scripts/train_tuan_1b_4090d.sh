#!/usr/bin/env bash
# =============================================================================
# train_tuan_1b_4090d.sh — 1B 全参 × 24h × 单张 RTX 4090D-24GB 启动器。
#
# 目标（用户口径）：24h 训 1B，**完全超越 Mamba2**（真实语料同 token 对照，
# 验收协议 X-equal-budget）。
#
# 预算与闸门：python -m simplcore.core.campaign --preset tuan-1b
# 文档：      docs/RTX4090D_TRAINING.md + docs/RTX4090D_1B_VS_MAMBA2.md
#
# 24GB 上的可行性（比 2B 宽裕得多）：
#   * 1B 全参 Adafactor ≈ 7.5 GiB（8 B/参数）+ chunk 激活 ~1.9 GiB
#     → ~9.5 GiB，**micro_batch 可以开到 8**（吞吐优先）；
#   * AdamW 16 B/参数 = 15.1 GiB 理论上也放得下，但为与 2B 战役同栈
#     （对照可比、状态可迁移），默认仍用 Adafactor；--optimizer adamw 可选。
#
# NT46 校准：24h 解析 token 预算 ~0.46B（MFU 0.30）；micro=8、seq=4096 →
# 32K tokens/step → N_plan = 0.46e9/32768 ≈ 14063 步（until_frac 阶梯
# 在窗口内展开，课程不卡死）。
# =============================================================================
set -euo pipefail

MODE="${1:-train}"
PRESET="tuan-1b"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_CHUNK="${TRAIN_CHUNK:-4096}"
MICRO_BATCH="${MICRO_BATCH:-8}"
GLOBAL_BATCH_TOKENS="${GLOBAL_BATCH_TOKENS:-32768}"   # 8×4096 = 32K tokens/step
# NT46 校准：N_plan = ceil(24h token 预算 / 每步 token) = 0.46e9/32768 ≈ 14063。
# Phase 0 实测 MFU 后按比例修正（MAX_STEPS 覆盖）。
MAX_STEPS="${MAX_STEPS:-14063}"
DATA_DIR="${DATA_DIR:-data/packed}"
OUT_DIR="${OUT_DIR:-runs/tuan-1b-4090d}"
OPTIMIZER="${OPTIMIZER:-adafactor}"

if [ "$MODE" = "--dry-run" ] || [ "$MODE" = "dry-run" ]; then
    echo "[launch] Phase 0 栈验收：dry-run（随机 token，100 步）"
    exec python -m simplcore.core.training.train.main \
        --preset "$PRESET" \
        --seq-len "$SEQ_LEN" \
        --train-chunk "$TRAIN_CHUNK" \
        --micro-batch "$MICRO_BATCH" \
        --optimizer "$OPTIMIZER" \
        --dtype bf16 \
        --compile \
        --max-steps 100 \
        --log-every 10 \
        --out-dir "$OUT_DIR/dryrun" \
        --dry-run
fi

echo "[launch] Phase 1：1B 全参 24h 训练（硬停 1440 分钟）"
exec python -m simplcore.core.training.train.main \
    --preset "$PRESET" \
    --data-dir "$DATA_DIR" \
    --seq-len "$SEQ_LEN" \
    --train-chunk "$TRAIN_CHUNK" \
    --micro-batch "$MICRO_BATCH" \
    --global-batch-tokens "$GLOBAL_BATCH_TOKENS" \
    --optimizer "$OPTIMIZER" \
    --lr 3e-4 --lr-min 3e-5 --weight-decay 0.1 \
    --warmup 200 \
    --max-steps "$MAX_STEPS" \
    --grad-clip 1.0 \
    --eval-every 100 --eval-batches 20 \
    --log-every 10 \
    --save-every 60 \
    --dtype bf16 \
    --compile \
    --time-budget-min 1440 \
    --resume "$OUT_DIR/last.pt" \
    --out-dir "$OUT_DIR"
