# -*- coding: utf-8 -*-
"""SimplCore 1.5B模型训练脚本 - 专为两张RTX 4090 (48GB)优化.

**真实显存需求分析 (1.5B参数模型):**
- 参数 (bf16): ~3GB
- AdamW状态 (fp32的m+v): ~12GB
- 梯度 (bf16): ~3GB
- 激活 (梯度检查点): ~5-8GB
- **总计: ~23-26GB/卡**

**硬件要求:**
- 2×RTX 4090 (48GB) → 完全可行!
- 2×RTX 3090/4080 (24GB) → 需要开启CPU Offload

**启动命令:**
    torchrun --nproc_per_node=2 run_train_1.5b_4090.py

**训练时间:** 10天 (240小时)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.SimplCore_model import SimplCoreModel
from SimplCore.training.trainer import TrainingConfig, SimplCoreTrainer


# ======================================================================
# 训练配置
# ======================================================================
TRAIN_CONFIG = TrainingConfig(
    # 批次配置
    batch_size=64,
    micro_batch_size=2,
    gradient_accumulation_steps=16,
    
    # 学习率
    learning_rate=3e-4,
    min_learning_rate=3e-5,
    warmup_steps=1000,
    total_steps=86400,  # 10天训练
    
    # 优化器
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    
    # 梯度
    grad_clip=1.0,
    grad_clip_mode="norm",
    
    # Checkpoint
    checkpoint_every=500,
    keep_checkpoint_max=5,
    save_best=True,
    
    # 日志 & 评估
    log_every=10,
    eval_every=500,
    eval_steps=50,
    
    # 精度
    precision="bf16",
    
    # 稳定性
    seed=42,
    loss_spike_threshold=3.0,
    loss_spike_patience=5,
    lr_reduce_factor=0.5,
    lr_reduce_min_lr=1e-6,
    
    # 分布式
    distributed=True,
    find_unused_parameters=False,
    
    # 显存优化
    gradient_checkpointing=True,
    use_deepspeed=False,  # 1.5B不需要DeepSpeed
    use_8bit_optimizer=False,  # 1.5B不需要8-bit
    
    # 路径
    output_dir="./output/1.5b_4090",
    resume_from_checkpoint=True,
)


# ======================================================================
# 1.5B模型配置
# ======================================================================
MODEL_CONFIG = {
    "vocab_size": 32000,
    "d_model": 2048,
    "n_layers": 24,
    "ffn_dim": 6144,
    "n_heads": 16,
    "max_seq_len": 2048,
    "dropout": 0.0,
}


# ======================================================================
# 模拟数据集（实际使用时替换为真实数据）
# ======================================================================
def create_synthetic_dataset(samples=500000, seq_len=512, vocab_size=32000):
    """创建模拟训练数据."""
    input_ids = torch.randint(0, vocab_size, (samples, seq_len))
    labels = input_ids.clone()
    return TensorDataset(input_ids, labels)


# ======================================================================
# 主训练函数
# ======================================================================
def run_training(rank, world_size):
    """执行分布式训练."""
    print(f"启动进程 {rank}/{world_size}")
    
    # 初始化分布式
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # 设置随机种子
    torch.manual_seed(TRAIN_CONFIG.seed + rank)
    
    # 创建模型配置
    model_config = SimplCoreConfig(
        vocab_size=MODEL_CONFIG["vocab_size"],
        d_model=MODEL_CONFIG["d_model"],
        n_layers=MODEL_CONFIG["n_layers"],
        ffn_dim=MODEL_CONFIG["ffn_dim"],
        n_heads=MODEL_CONFIG["n_heads"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        dropout=MODEL_CONFIG["dropout"],
    )
    
    # 创建模型
    print(f"[Rank {rank}] 创建 SimplCore 模型...")
    model = SimplCoreModel(model_config)
    
    # 打印模型信息
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数: {total_params / 1e9:.2f} B")
        print(f"d_model: {MODEL_CONFIG['d_model']}")
        print(f"n_layers: {MODEL_CONFIG['n_layers']}")
        print(f"n_heads: {MODEL_CONFIG['n_heads']}")
        print(f"ffn_dim: {MODEL_CONFIG['ffn_dim']}")
        print(f"\n显存估算 (bf16):")
        print(f"  参数: ~{total_params * 2 / 1e9:.1f} GB")
        print(f"  AdamW状态: ~{total_params * 4 / 1e9:.1f} GB")
        print(f"  梯度: ~{total_params * 2 / 1e9:.1f} GB")
        print(f"  激活 (梯度检查点): ~5-8 GB")
        print(f"  总计: ~{total_params * 8 / 1e9 + 6:.1f} GB/卡")
        print(f"\n训练配置:")
        print(f"  微批大小: {TRAIN_CONFIG.micro_batch_size}")
        print(f"  梯度累积: {TRAIN_CONFIG.gradient_accumulation_steps}")
        print(f"  全局批大小: {TRAIN_CONFIG.batch_size}")
        print(f"  训练步数: {TRAIN_CONFIG.total_steps}")
        print(f"  精度: {TRAIN_CONFIG.precision}")
        print(f"  梯度检查点: {TRAIN_CONFIG.gradient_checkpointing}")
        print(f"\n两张4090 (48GB×2) 完全可以容纳!\n")
    
    # 移动到GPU
    model = model.to(rank)
    
    # 创建数据
    dataset = create_synthetic_dataset(
        samples=500000,
        seq_len=512,
        vocab_size=model_config.vocab_size,
    )
    
    # 分布式采样
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG.micro_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    eval_loader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG.micro_batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    
    # 创建训练器
    trainer = SimplCoreTrainer(
        config=TRAIN_CONFIG,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    
    # 开始训练
    print(f"[Rank {rank}] 开始训练 ({trainer.config.total_steps} 步)...")
    start_time = time.time()
    
    try:
        summary = trainer.train()
    except KeyboardInterrupt:
        print(f"[Rank {rank}] 训练被中断")
        summary = trainer.get_monitoring_report()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 打印训练总结
    if rank == 0:
        print("\n" + "=" * 70)
        print("  训练完成")
        print("=" * 70)
        print(f"  总步数: {summary['training_summary']['global_step']}")
        print(f"  总耗时: {elapsed/3600:.2f} 小时")
        print(f"  平均步长: {elapsed/max(1, summary['training_summary']['global_step']):.2f} 秒")
        print(f"  最终损失: {summary['loss'].get('recent_mean', 'N/A')}")
        print(f"  最佳评估损失: {summary['best_model']['best_eval_loss']}")
        print(f"  跳过的batch数: {summary['performance']['skipped_batches']}")
        print("=" * 70)
    
    # 清理
    dist.destroy_process_group()


# ======================================================================
# 入口
# ======================================================================
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU")
    
    # 检查GPU显存
    gpu_info = []
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        gpu_info.append(f"GPU{i}: {props.name}, {props.total_mem/1e9:.1f}GB")
        print(f"  {gpu_info[-1]}")
    
    # 检查是否有足够显存
    min_mem = min(torch.cuda.get_device_properties(i).total_mem for i in range(world_size))
    if min_mem < 24 * 1e9:
        print("\n警告: 显存不足24GB, 建议开启梯度检查点或降低模型大小")
    else:
        print("\n✓ 显存充足, 可以训练1.5B模型")
    
    mp.spawn(
        run_training,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

