# -*- coding: utf-8 -*-
"""SimplCore 33B模型训练脚本 - 使用DeepSpeed ZeRO-3 + 8-bit优化器.

**真实显存需求分析 (33B参数模型):**
- 参数 (bf16): 66GB
- AdamW状态 (fp32的m+v): 264GB → 8-bit后: 66GB
- 梯度 (bf16): 66GB
- 激活 (梯度检查点): ~30GB
- **总计 (标准AdamW): ~426GB**
- **总计 (8-bit): ~228GB**

**硬件要求:**
- 2×96GB GPU: 每卡约114GB → 勉强可行
- 4×48GB GPU: 每卡约69GB → 可行
- 8×48GB GPU: 每卡约34GB → 最佳

**技术栈:**
1. DeepSpeed ZeRO-3: 参数/梯度/优化器状态分片
2. bitsandbytes 8-bit AdamW: 优化器状态降至1/4
3. 梯度检查点: 激活内存减少50%
4. bf16混合精度: 参数存储减半
5. CPU Offload: 可选, 将部分数据卸载到CPU内存

**重要警告:**
- 闲鱼上的96GB魔改4090极大概率是骗局
- 验证方法: 运行 cuda-memtest --size=96G 写入测试
- 虚假显存写入超过物理容量会静默损坏数据

启动命令:
    # 使用torchrun (推荐)
    torchrun --nproc_per_node=2 run_train_33b_deepspeed.py

    # 或使用deepspeed
    deepspeed --num_gpus=2 run_train_33b_deepspeed.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

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
    micro_batch_size=1,
    gradient_accumulation_steps=32,
    
    # 学习率
    learning_rate=2e-4,
    min_learning_rate=2e-5,
    warmup_steps=2000,
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
    use_deepspeed=True,
    use_8bit_optimizer=True,
    
    # 路径
    output_dir="./output/33b_deepspeed",
    resume_from_checkpoint=True,
)


# ======================================================================
# DeepSpeed配置文件
# ======================================================================
DEEPSPEED_CONFIG = {
    "train_batch_size": TRAIN_CONFIG.batch_size,
    "train_micro_batch_size_per_gpu": TRAIN_CONFIG.micro_batch_size,
    "gradient_accumulation_steps": TRAIN_CONFIG.gradient_accumulation_steps,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": TRAIN_CONFIG.learning_rate,
            "betas": [TRAIN_CONFIG.adam_beta1, TRAIN_CONFIG.adam_beta2],
            "eps": TRAIN_CONFIG.adam_epsilon,
            "weight_decay": TRAIN_CONFIG.weight_decay,
        },
    },
    
    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": TRAIN_CONFIG.total_steps,
            "warmup_num_steps": TRAIN_CONFIG.warmup_steps,
            "min_lr": TRAIN_CONFIG.min_learning_rate,
        },
    },
    
    "bf16": {
        "enabled": TRAIN_CONFIG.precision == "bf16",
    },
    
    "fp16": {
        "enabled": TRAIN_CONFIG.precision == "fp16",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True,
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True,
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 50000000,
        "stage3_prefetch_bucket_size": 50000000,
        "stage3_param_persistence_threshold": 10000000,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "sub_group_size": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    
    "gradient_clipping": TRAIN_CONFIG.grad_clip,
    
    "logging": {
        "steps_per_print": TRAIN_CONFIG.log_every,
        "log_level": "INFO",
    },
    
    "checkpoint": {
        "load_universal": False,
        "save_interval": TRAIN_CONFIG.checkpoint_every,
        "ckpt_path": os.path.join(TRAIN_CONFIG.output_dir, "checkpoints"),
    },
    
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,
        "cpu_checkpointing": False,
    },
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
    
    # 创建33B模型配置
    print(f"[Rank {rank}] 创建 33B 模型配置...")
    model_config = SimplCoreConfig.get_33b_config()
    
    # 创建模型
    print(f"[Rank {rank}] 创建 SimplCore 33B 模型...")
    model = SimplCoreModel(model_config)
    
    # 打印模型信息
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数: {total_params / 1e9:.2f} B")
        print(f"d_model: {model_config.d_model}")
        print(f"n_layers: {model_config.n_layers}")
        print(f"vocab_size: {model_config.vocab_size}")
        print(f"\n显存估算:")
        print(f"  参数 (bf16): ~{total_params * 2 / 1e9:.1f} GB")
        print(f"  AdamW状态 (8-bit): ~{total_params / 1e9:.1f} GB")
        print(f"  梯度 (bf16): ~{total_params * 2 / 1e9:.1f} GB")
        print(f"  总计 (8-bit + ZeRO-3): ~{(total_params * 5) / (1e9 * world_size):.1f} GB/卡")
        print(f"  需要 {world_size} 张 {math.ceil((total_params * 5) / (1e9 * world_size))}GB+ GPU\n")
    
    # 移动到GPU
    model = model.to(rank)
    
    # 创建数据
    dataset = create_synthetic_dataset(
        samples=100000,
        seq_len=256,
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
    import math
    
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU")
    
    # 检查GPU显存
    gpu_info = []
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        gpu_info.append(f"GPU{i}: {props.name}, {props.total_mem/1e9:.1f}GB")
        print(f"  {gpu_info[-1]}")
    
    # 显存检查
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        if props.total_mem < 40 * 1e9:
            print(f"\n警告: GPU{i} 显存不足40GB, 可能无法训练33B模型")
    
    # 确认用户知道风险
    print("\n" + "=" * 70)
    print("  重要提示")
    print("=" * 70)
    print("  1. 33B模型从零训练需要大量GPU显存")
    print("  2. 建议使用8张40GB+ GPU或租用云GPU")
    print("  3. 闲鱼魔改4090极大概率是骗局")
    print("  4. 验证方法: cuda-memtest --size=96G")
    print("=" * 70 + "\n")
    
    mp.spawn(
        run_training,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

