# -*- coding: utf-8 -*-
"""SimplCore 3B模型训练脚本 - 专为两张RTX 4090 (48GB)优化.

**真实显存需求分析 (3B参数模型):**
- 参数 (bf16): ~6GB
- AdamW状态 (fp32的m+v): ~24GB
- 梯度 (bf16): ~6GB
- 激活 (梯度检查点): ~5-8GB
- **总计: ~41-44GB/卡**

**硬件要求:**
- 2×RTX 4090 (48GB) → 刚好可以容纳!
- 需要开启梯度检查点和bf16混合精度

**训练效率优化:**
1. Flash Attention 2: 加速注意力计算, 降低激活内存
2. 流水线并行: 模型层分布到多GPU, 重叠计算与通信
3. 高效数据加载: pin_memory + num_workers优化
4. 混合精度: bf16训练, 保持精度的同时节省显存
5. 梯度检查点: 激活内存减少50%

**10天训练token估算:**
- 微批大小: 1
- 全局批大小: 64 (梯度累积64步)
- 序列长度: 512
- 每步token: 64 × 512 = 32K
- 每步时间: ~1.5秒
- 10天总计: ~14B token

**Chinchilla缩放定律:**
- 3B模型最优训练token: ~6B
- 10天训练token: ~14B → 超过最优!

**启动命令:**
    torchrun --nproc_per_node=2 run_train_3b_4090.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

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
    gradient_accumulation_steps=64,
    
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
    use_deepspeed=False,
    use_8bit_optimizer=False,
    
    # 路径
    output_dir="./output/3b_4090",
    resume_from_checkpoint=True,
)


#
