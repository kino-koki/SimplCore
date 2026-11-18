#!/usr/bin/env bash
# =============================================================================
# train_tuan_2b_4090d.sh — 2B 全参 × 24h × 单张 RTX 4090D-24GB 启动器。
#
# 预算与闸门：python -m simplcore.core.campaign（解析账本，不占显存）
# 文档：      docs/RTX4090D_TRAINING.md（显存账 / token 预算 / 排障）
#
# 关键决策（24GB 上 2B 全参的唯一可行组合，见 campaign.py 硬闸门）：
#   * --optimizer adafactor   16 B/参数的 AdamW 在 2B 时 = 32 GiB 直接爆；
#                             Adafactor 因子化二阶矩 ≈ 8 B/参数 → ~19 GiB；
#   * --train-chunk 4096      分块流式训练：激活 O(chunk)，与总上下文 L 无关
#                             （OOM 时降到 2048，勿降 batch）；
#   * --dtype bf16            bf16 autocast 前向 + fp32 参数/梯度（4090D TF32
#                             tensor core + bf16 低精度归约已在 apply_cuda_tuning 开启）；
#   * --time-budget-min 1440  24h 硬停（soft stop，落盘 checkpoint 后退出）；
#   * --compile               torch.compile max-autotune（首个 step 编译开销
#                             几十秒，24h 长跑可忽略）。
#
# Phase 0（栈验收，≤1 天硬停）：
#   bash scripts/train_tuan_2b_4090d.sh --dry-run   # 随机 token 100 步，显存峰值 ≤ 23 GiB
# =============================================================================
set -euo pipefail

MODE="${1:-train}"
PRESET="tuan-2b"
SEQ_LEN="${SEQ_LEN:-4096}"
TRAIN_CHUNK="${TRAIN_CHUNK:-4096}"
MICRO_BATCH="${MICRO_BATCH:-2}"
GLOBAL_BATCH_TOKENS="${GLOBAL_BATCH_TOKENS:-65536}"   # 2×4096×8 ≈ 64K tokens/step
# NT46 校准（硬停闸门的课程口径）：N_plan 必须 = token 预算 / 每步 token。
# 24h 解析预算 ~2.2e8 tokens（MFU 0.30，Phase 0 实测后修正）→
# N_plan ≈ 3358 步。若按惯例给 1e6，until_frac 阶梯（0.18×N_plan=1.8e5 起）
# 在 24h 内一步都走不到——课程整个卡在 stage 0（seq 4096），4M 末段
# 需要 0.9×1e6×65536 ≈ 5.9e10 tokens，是预算的 267 倍。
MAX_STEPS="${MAX_STEPS:-3358}"
DATA_DIR="${DATA_DIR:-data/packed}"
OUT_DIR="${OUT_DIR:-runs/tuan-2b-4090d}"

if [ "$MODE" = "--dry-run" ] || [ "$MODE" = "dry-run" ]; then
    echo "[launch] Phase 0 栈验收：dry-run（随机 token，100 步）"
    exec python -m simplcore.core.training.train.main \
        --preset "$PRESET" \
        --seq-len "$SEQ_LEN" \
        --train-chunk "$TRAIN_CHUNK" \
        --micro-batch "$MICRO_BATCH" \
        --optimizer adafactor \
        --dtype bf16 \
        --compile \
        --max-steps 100 \
        --log-every 10 \
        --out-dir "$OUT_DIR/dryrun" \
        --dry-run
fi

echo "[launch] Phase 1：2B 全参 24h 训练（硬停 1440 分钟）"
exec python -m simplcore.core.training.train.main \
    --preset "$PRESET" \
    --data-dir "$DATA_DIR" \
    --seq-len "$SEQ_LEN" \
    --train-chunk "$TRAIN_CHUNK" \
    --micro-batch "$MICRO_BATCH" \
    --global-batch-tokens "$GLOBAL_BATCH_TOKENS" \
    --optimizer adafactor \
    --lr 3e-4 --lr-min 3e-5 --weight-decay 0.1 \
    --warmup 200 \
    --max-steps "$MAX_STEPS" \
    --grad-clip 1.0 \
    --eval-every 60 --eval-batches 20 \
    --log-every 10 \
    --save-every 30 \
    --dtype bf16 \
    --compile \
    --time-budget-min 1440 \
    --resume "$OUT_DIR/last.pt" \
    --out-dir "$OUT_DIR"
