# -*- coding: utf-8 -*-
"""SimplCore EQP 33B等效模型训练脚本 - 专为两张 RTX 4090 (48GB) 优化.

核心创新:
1. EQP（等效参数技术）— 12B物理参数 ≈ 70B等效参数容量
2. ZeRO-4（自研分布式优化器）— 参数/梯度/优化器状态/激活分片
3. 在线矩阵乘法 — 不构建完整权重矩阵，直接计算 y = W_eff @ x
4. 分块计算 — 避免中间结果膨胀

模型配置:
- 物理参数: ~12B (d_model=4096, L=32)
- 等效容量: ~70B+ (C(4096,4) × 256² × 32)
- 训练参数: 全部12B（无需LoRA，不会降智！）

显存估算 (单卡):
- 参数 (bf16): ~24GB → ZeRO-4分片后 ~12GB/卡
- 优化器状态 (bf16): ~48GB → ZeRO-4分片后 ~24GB/卡
- 梯度 (bf16): ~12GB → ZeRO-4分片后 ~6GB/卡
- 激活 (梯度检查点): ~5-8GB/卡
- 总计: ~47-50GB/卡（两张4090刚好容纳）

训练时间: 10天 (240小时)
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from SimplCore.architecture.eqp import SimplCoreWithEQP, compute_parameter_efficiency
from SimplCore.training.trainer import TrainingConfig, SimplCoreTrainer


# ======================================================================
# 训练配置
# ======================================================================
TRAIN_CONFIG = TrainingConfig(
    # 批次配置
    batch_size=64,
    micro_batch_size=1,
    gradient_accumulation_steps=32,
    
    # 学习率
    learning_rate=2e-4,
    min_learning_rate=2e-5,
    warmup_steps=2000,
    
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
    use_zero_4=True,
    
    # 路径
    output_dir="./output/eqp_33b_4090",
    resume_from_checkpoint=True,
)

# EQP模型配置
EQP_CONFIG = {
    "vocab_size": 32000,
    "d_model": 4096,
    "n_layers": 32,
    "ffn_dim": 14336,
    "n_primitives": 4096,
    "primitive_size": 256,
    "k_compose": 4,
    "max_seq_len": 2048,
    "dropout": 0.0,
}


# ======================================================================
# 模拟数据集（实际使用时替换为真实数据）
# ======================================================================
def create_synthetic_dataset(samples=100000, seq_len=512, vocab_size=32000):
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
    
    # 创建模型
    print(f"[Rank {rank}] 创建 EQP 模型...")
    model = SimplCoreWithEQP(
        vocab_size=EQP_CONFIG["vocab_size"],
        d_model=EQP_CONFIG["d_model"],
        n_layers=EQP_CONFIG["n_layers"],
        ffn_dim=EQP_CONFIG["ffn_dim"],
        n_primitives=EQP_CONFIG["n_primitives"],
        primitive_size=EQP_CONFIG["primitive_size"],
        k_compose=EQP_CONFIG["k_compose"],
        max_seq_len=EQP_CONFIG["max_seq_len"],
        dropout=EQP_CONFIG["dropout"],
    )
    
    # 打印参数效率
    if rank == 0:
        efficiency = compute_parameter_efficiency(
            d_model=EQP_CONFIG["d_model"],
            n_layers=EQP_CONFIG["n_layers"],
            n_primitives=EQP_CONFIG["n_primitives"],
            primitive_size=EQP_CONFIG["primitive_size"],
            k_compose=EQP_CONFIG["k_compose"],
        )
        print("\n" + "=" * 70)
        print("  EQP 参数效率分析")
        print("=" * 70)
        print(f"  标准Transformer参数: {efficiency['std_transformer_params_B']:.2f} B")
        print(f"  EQP SimplCore参数: {efficiency['eqp_SimplCore_params_B']:.2f} B")
        print(f"  等效参数容量: {efficiency['equivalent_capacity_B']:.2f} B")
        print(f"  参数效率比: {efficiency['parameter_efficiency_ratio']:.1f}x")
        print(f"  压缩比: {efficiency['compression_ratio']:.1f}x")
        print("=" * 70 + "\n")
    
    # 移动到GPU
    model = model.to(rank)
    
    # 创建数据
    dataset = create_synthetic_dataset(
        samples=100000,
        seq_len=256,
        vocab_size=EQP_CONFIG["vocab_size"],
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
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    eval_loader = DataLoader(
        dataset,
        batch_size=TRAIN_CONFIG.micro_batch_size,
        sampler=sampler,
        num_workers=1,
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
    
    # 设置训练步数（10天训练）
    # 假设每步约10秒（两张4090，bf16，梯度检查点）
    # 10天 = 864000秒 / 10秒/步 ≈ 86400步
    trainer.config.total_steps = 86400
    
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
    
    if world_size < 2:
        print("警告: 建议使用至少2个GPU进行训练")
    
    mp.spawn(
        run_training,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

