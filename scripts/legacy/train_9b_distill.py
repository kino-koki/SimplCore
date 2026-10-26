# -*- coding: utf-8 -*-
"""9B 参数模型蒸馏训练启动脚本。

使用 DeepSeek V4 Flash / NVIDIA NIM 双教师模型的蒸馏数据，
训练 9B 参数的学生模型。

使用方法:
    # 查看帮助
    python train_9b_distill.py --help

    # 单机单卡（需要 A100-80G 或 2xRTX-4090）
    python train_9b_distill.py --data data/distill/sft/distill_dataset_final.jsonl

    # 梯度检查点节省显存（推荐）
    python train_9b_distill.py --grad-checkpoint --batch-size 8 --grad-accum 4

    # 多机多卡 (DDP)
    torchrun --nproc_per_node=8 train_9b_distill.py --distributed
"""
from __future__ import annotations
import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(description="青小团 9B 模型蒸馏训练")
    
    ap.add_argument("--data", nargs="*", default=None,
                    help="蒸馏 JSONL 数据文件列表")
    ap.add_argument("--output", default="checkpoints/distill_9b",
                    help="模型输出目录")
    ap.add_argument("--vocab-size", type=int, default=32000,
                    help="词表大小")
    ap.add_argument("--max-len", type=int, default=4096,
                    help="最大序列长度")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="每个 GPU 的 batch size")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="梯度累积步数 (有效 batch = batch_size * grad_accum * num_gpus)")
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="学习率 (9B模型用较小的lr)")
    ap.add_argument("--epochs", type=int, default=2,
                    help="训练轮数")
    ap.add_argument("--warmup-ratio", type=float, default=0.03,
                    help="warmup 比例")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="最大步数 (覆盖 epochs)")
    ap.add_argument("--log-every", type=int, default=10,
                    help="每N步打印日志")
    ap.add_argument("--save-every", type=int, default=500,
                    help="每N步保存checkpoint")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="启用梯度检查点 (节省显存, 慢约30%)")
    ap.add_argument("--mixed-precision", default="bf16",
                    choices=["bf16", "fp16", "none"],
                    help="混合精度")
    ap.add_argument("--distributed", action="store_true",
                    help="分布式训练 (DDP)")
    ap.add_argument("--limit", type=int, default=0,
                    help="限制数据条数 (调试用)")
    
    args = ap.parse_args()
    
    print("=" * 60)
    print("  青小团 9B 模型蒸馏训练")
    print("=" * 60)
    print()
    
    print("[配置]")
    print(f"  模型规模: 9B 参数 (hidden=4800, layers=40, heads=40)")
    print(f"  词表大小: {args.vocab_size}")
    print(f"  最大长度: {args.max_len}")
    print(f"  Batch Size: {args.batch_size} (单卡)")
    print(f"  梯度累积: {args.grad_accum}")
    print(f"  学习率: {args.lr}")
    print(f"  混合精度: {args.mixed_precision}")
    print(f"  梯度检查点: {args.grad_checkpoint}")
    print(f"  输出目录: {args.output}")
    print()
    
    print("[硬件要求]")
    print(f"  最低配置: 1x A100-80G (fp16 + 梯度检查点)")
    print(f"  推荐配置: 2-8x A100-80G (DDP 分布式训练)")
    print(f"  显存估算: ~48GB (fp16 权重+梯度+优化器状态)")
    print()
    
    print("[说明]")
    print("  本脚本调用 qxt.distill.run_distill_sft 模块执行训练")
    print("  使用 --preset 9b 配置")
    print()
    
    default_data = [
        "data/distill/sft/distill_dataset_final.jsonl",
        "data/distill/sft/distill_dataset.jsonl",
        "data/distill/sft/deepseek_v4_distill.jsonl",
        "data/distill/sft/deepseek_v4_batch2.jsonl",
    ]
    
    data_files = args.data or [os.path.join(_PROJECT_ROOT, d) for d in default_data]
    data_files = [d for d in data_files if os.path.exists(d)]
    
    if not data_files:
        print("[错误] 未找到任何数据文件！")
        print("  请先生成蒸馏数据:")
        print("    python -m qxt.distill.generate_distill_data --api-key YOUR_KEY --num 1000")
        sys.exit(1)
    
    print(f"[数据] 找到 {len(data_files)} 个数据文件:")
    for d in data_files:
        print(f"  - {d}")
    print()
    
    print("启动训练...")
    print("-" * 60)
    
    from qxt.distill.run_distill_sft import main as train_main
    
    sys.argv = [
        "run_distill_sft.py",
        "--preset", "9b",
        "--output", args.output,
        "--vocab-size", str(args.vocab_size),
        "--max-len", str(args.max_len),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--warmup-ratio", str(args.warmup_ratio),
        "--log-every", str(args.log_every),
        "--save-every", str(args.save_every),
        "--seed", str(args.seed),
        "--grad-accum", str(args.grad_accum),
        "--mixed-precision", args.mixed_precision,
    ]
    
    if args.max_steps > 0:
        sys.argv.extend(["--max-steps", str(args.max_steps)])
    if args.grad_checkpoint:
        sys.argv.append("--grad-checkpoint")
    if args.limit > 0:
        sys.argv.extend(["--limit", str(args.limit)])
    if args.data:
        sys.argv.extend(["--data"] + args.data)
    
    train_main()


if __name__ == "__main__":
    main()
