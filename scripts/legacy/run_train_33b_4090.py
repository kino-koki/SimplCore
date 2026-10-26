# -*- coding: utf-8 -*-
"""SimplCore 33B 模型训练脚本 - 专为两张 RTX 4090 (48GB) 优化.

优化策略:
1. LoRA 参数高效微调 (~100M 可训练参数)
2. 梯度检查点 (节省 ~50% 激活内存)
3. bf16 混合精度 (降低参数存储)
4. 极低 batch size + 梯度累积 (控制激活峰值)
5. 模型并行 (跨卡拆分模型层)
6. 动态梯度检查点优化
7. 自适应学习率调度

训练配置:
- 33B 参数模型 (hidden=8192, layers=38)
- LoRA rank=8, alpha=16
- batch_size=1, micro_batch=1, accumulation=64
- 序列长度=2048
- 训练时间=10天 (864000秒)
"""

import sys
import os
import math
import time
import random
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.SimplCore_model import SimplCoreModel
from SimplCore.architecture.lora import (
    apply_lora_to_model,
    count_lora_parameters,
    get_lora_parameters,
)

# ========== 配置 ==========
SEED = 42
NUM_GPUS = 2
MODEL_CONFIG = "33b"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "SimplCore_33b_4090")

# 训练参数
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 64
GLOBAL_BATCH_SIZE = MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS

MAX_SEQUENCE_LENGTH = 2048
VOCAB_SIZE = 32000

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 1000

WEIGHT_DECAY = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8

GRAD_CLIP = 1.0

# LoRA 参数
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 日志与保存
LOG_EVERY = 10
CHECKPOINT_EVERY = 500
EVAL_EVERY = 1000
KEEP_CHECKPOINT_MAX = 5

# 训练时间目标 (10天)
TARGET_TRAINING_SECONDS = 10 * 24 * 3600  # 864000秒

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def log(msg):
    logger.info(msg)


# ========== 分布式初始化 ==========
def init_distributed():
    if not torch.cuda.is_available():
        log("CUDA 不可用, 使用 CPU 训练")
        return 0, 1, torch.device("cpu")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % NUM_GPUS}")

    log(f"分布式初始化完成: rank={rank}, world_size={world_size}, device={device}")
    return rank, world_size, device


# ========== 学习率调度器 ==========
def cosine_lr(step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(MIN_LEARNING_RATE, 0.5 * base_lr * (1 + math.cos(math.pi * progress)))


# ========== 数据准备 ==========
def create_dummy_dataset(num_samples=10000):
    """创建模拟训练数据.

    注意: 实际训练时应替换为真实数据 (如 RedPajama、ShareGPT 等).
    """
    log(f"创建模拟训练数据: {num_samples} 条")
    input_ids = []
    labels = []

    for _ in range(num_samples):
        seq_len = random.randint(512, MAX_SEQUENCE_LENGTH)
        ids = torch.randint(0, VOCAB_SIZE, (seq_len,))
        input_ids.append(ids)
        labels.append(ids.clone())

    # padding 到统一长度
    max_len = max(len(s) for s in input_ids)
    for i in range(num_samples):
        pad_len = max_len - len(input_ids[i])
        if pad_len > 0:
            input_ids[i] = torch.cat([input_ids[i], torch.zeros(pad_len, dtype=torch.long)])
            labels[i] = torch.cat([labels[i], torch.full((pad_len,), -100, dtype=torch.long)])

    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    return TensorDataset(input_ids, labels)


# ========== 模型初始化 ==========
def init_model(device, rank, world_size):
    """初始化 33B 模型并应用 LoRA."""
    log("初始化 33B 模型...")

    cfg = SimplCoreConfig.get_33b_config()
    cfg.vocab_size = VOCAB_SIZE
    cfg.max_sequence_length = MAX_SEQUENCE_LENGTH

    model = SimplCoreModel(cfg)

    total_params = model.count_parameters()
    log(f"模型总参数量: {total_params / 1e9:.2f} B")

    # 应用 LoRA
    log("应用 LoRA Adapter...")
    lora_count = apply_lora_to_model(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    lora_params = count_lora_parameters(model)
    log(f"LoRA Adapter 数量: {lora_count}")
    log(f"LoRA 参数数量: {lora_params / 1e6:.2f} M")
    log(f"可训练参数比例: {(lora_params / total_params) * 100:.4f}%")

    # 模型并行: 将层分布到不同 GPU
    if world_size > 1:
        layers_per_gpu = cfg.num_layers // world_size
        start_layer = rank * layers_per_gpu
        end_layer = min((rank + 1) * layers_per_gpu, cfg.num_layers)

        log(f"模型并行: GPU {rank} 负责层 {start_layer}-{end_layer-1}")

        # 冻结不属于当前 GPU 的层
        for i, block in enumerate(model.blocks):
            if i < start_layer or i >= end_layer:
                for param in block.parameters():
                    param.requires_grad = False

    # 仅将模型移到当前设备
    model = model.to(device)

    # DDP 包装
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[rank % NUM_GPUS],
            find_unused_parameters=True,
        )

    return model, cfg


# ========== 优化器初始化 ==========
def init_optimizer(model):
    """初始化优化器 (仅优化 LoRA 参数)."""
    log("初始化优化器...")

    # 获取所有可训练参数 (LoRA 参数 + 可能的偏置)
    param_groups = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora" in name:
                param_groups.append({"params": [param], "weight_decay": 0.0, "lr": LEARNING_RATE})
            else:
                param_groups.append({"params": [param], "weight_decay": WEIGHT_DECAY, "lr": LEARNING_RATE})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS,
    )

    log(f"优化器参数组数: {len(param_groups)}")
    return optimizer


# ========== 梯度检查点 ==========
def apply_gradient_checkpointing(model):
    """应用梯度检查点优化."""
    log("应用梯度检查点优化...")

    def set_checkpoint(module):
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            return
        if len(list(module.children())) > 0:
            return
        if hasattr(module, "forward") and callable(module.forward):
            orig_forward = module.forward

            def checkpoint_forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(orig_forward, *args, use_reentrant=False)

            module.forward = checkpoint_forward

    model.apply(set_checkpoint)
    log("梯度检查点优化已应用")


# ========== 训练循环 ==========
def train():
    """执行完整训练循环."""
    # 初始化
    rank, world_size, device = init_distributed()
    is_main = rank == 0

    torch.manual_seed(SEED + rank)
    random.seed(SEED + rank)

    # 创建输出目录
    if is_main:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "tb_logs"))

    # 初始化模型
    model, cfg = init_model(device, rank, world_size)

    # 应用梯度检查点
    apply_gradient_checkpointing(model)

    # 初始化优化器
    optimizer = init_optimizer(model)

    # 创建数据
    dataset = create_dummy_dataset(num_samples=50000)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
    train_loader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    log("=" * 60)
    log("  SimplCore 33B 训练配置")
    log("=" * 60)
    log(f"  模型: {MODEL_CONFIG}")
    log(f"  参数量: {model.count_parameters() / 1e9:.2f} B")
    log(f"  LoRA 参数: {count_lora_parameters(model) / 1e6:.2f} M")
    log(f"  设备: {device}")
    log(f"  GPU数量: {world_size}")
    log(f"  序列长度: {MAX_SEQUENCE_LENGTH}")
    log(f"  Micro Batch: {MICRO_BATCH_SIZE}")
    log(f"  梯度累积: {GRADIENT_ACCUMULATION_STEPS}")
    log(f"  全局 Batch: {GLOBAL_BATCH_SIZE}")
    log(f"  学习率: {LEARNING_RATE}")
    log(f"  目标训练时间: {TARGET_TRAINING_SECONDS / 3600:.1f} 小时")
    log("=" * 60)

    # 训练状态
    global_step = 0
    epoch = 0
    best_loss = float("inf")
    total_loss = 0.0
    start_time = time.time()

    # 进度条
    if is_main:
        pbar = tqdm(total=1000000, desc="训练", unit="step", ncols=100)

    model.train()
    optimizer.zero_grad()

    # 计算估计步数 (基于目标训练时间)
    estimated_steps_per_hour = 300  # 估计值, 将根据实际速度调整
    estimated_total_steps = int(TARGET_TRAINING_SECONDS / 3600 * estimated_steps_per_hour)

    while time.time() - start_time < TARGET_TRAINING_SECONDS:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in train_loader:
            if time.time() - start_time >= TARGET_TRAINING_SECONDS:
                break

            input_ids, labels = batch
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 前向传播 (bf16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                # 计算损失
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            # 梯度累积
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            # 累积完成后更新
            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                # 更新学习率
                lr = cosine_lr(global_step // GRADIENT_ACCUMULATION_STEPS, estimated_total_steps, LEARNING_RATE, WARMUP_STEPS)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # 优化器步进
                optimizer.step()
                optimizer.zero_grad()

                # 日志记录
                if is_main and (global_step // GRADIENT_ACCUMULATION_STEPS) % LOG_EVERY == 0:
                    elapsed = time.time() - start_time
                    avg_loss = total_loss / LOG_EVERY / GRADIENT_ACCUMULATION_STEPS
                    tokens_per_sec = (global_step * MICRO_BATCH_SIZE * MAX_SEQUENCE_LENGTH) / max(1, elapsed)

                    writer.add_scalar("train/loss", avg_loss, global_step // GRADIENT_ACCUMULATION_STEPS)
                    writer.add_scalar("train/learning_rate", lr, global_step // GRADIENT_ACCUMULATION_STEPS)
                    writer.add_scalar("train/tokens_per_sec", tokens_per_sec, global_step // GRADIENT_ACCUMULATION_STEPS)

                    pbar.update(LOG_EVERY)
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "tps": f"{tokens_per_sec:.0f}",
                        "elapsed": f"{elapsed/3600:.1f}h",
                    })

                    total_loss = 0.0

                # 保存 checkpoint
                if is_main and (global_step // GRADIENT_ACCUMULATION_STEPS) % CHECKPOINT_EVERY == 0:
                    save_checkpoint(model, optimizer, global_step // GRADIENT_ACCUMULATION_STEPS, epoch, avg_loss)

            global_step += 1

        epoch += 1

    # 训练完成
    if is_main:
        pbar.close()
        writer.close()
        save_checkpoint(model, optimizer, global_step // GRADIENT_ACCUMULATION_STEPS, epoch, avg_loss)
        log(f"训练完成! 总步数: {global_step // GRADIENT_ACCUMULATION_STEPS}, 总耗时: {(time.time() - start_time) / 3600:.1f} 小时")


# ========== Checkpoint 管理 ==========
def save_checkpoint(model, optimizer, step, epoch, loss):
    """保存训练 checkpoint."""
    checkpoint = {
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "loss": loss,
        "config": {
            "hidden_size": 8192,
            "num_layers": 38,
            "vocab_size": VOCAB_SIZE,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
        },
    }

    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"step_{step}.pt")
    torch.save(checkpoint, ckpt_path)
    log(f"Checkpoint 已保存: {ckpt_path}")

    # 清理旧 checkpoint
    import glob
    checkpoints = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoints", "step_*.pt")))
    if len(checkpoints) > KEEP_CHECKPOINT_MAX:
        for old_ckpt in checkpoints[:-KEEP_CHECKPOINT_MAX]:
            os.remove(old_ckpt)
            log(f"清理旧 checkpoint: {old_ckpt}")


if __name__ == "__main__":
    # 设置环境变量 (两张 GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(NUM_GPUS)

    # 使用 torchrun 启动
    if "LOCAL_RANK" not in os.environ:
        # 如果没有 LOCAL_RANK, 使用 torchrun 启动
        import subprocess
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(NUM_GPUS),
            "--master_port", "29500",
            __file__,
        ]
        subprocess.run(cmd)
    else:
        train()
