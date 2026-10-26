import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_from_disk
import deepspeed

from qingxiaotuan import QingxiaotuanConfig, QingxiaotuanForCausalLM, QingxiaotuanTokenizer
from dataset import QingxiaotuanPretrainDataset, collate_fn_for_packing


def parse_args():
    parser = argparse.ArgumentParser(description="青小团分布式预训练脚本 - 支持512K长序列")
    parser.add_argument("--data_path", type=str, required=True, help="tokenized数据集路径")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="模型保存路径")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="预训练模型路径，None则从头训练")
    parser.add_argument("--max_length", type=int, default=524288, help="最大序列长度 (默认512K)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每设备训练batch size (512K下每卡1条)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=12, help="梯度累积步数 (默认12)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="峰值学习率")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="最小学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="warmup步数")
    parser.add_argument("--max_train_steps", type=int, default=100000, help="总训练步数")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存间隔步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志打印间隔步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地rank，deepspeed自动传入")
    parser.add_argument("--sequence_parallel", action="store_true", default=False, help="启用序列并行 (sequence parallel)")
    parser.add_argument("--flash_attention", action="store_true", default=True, help="启用Flash Attention")
    parser.add_argument("--pack_sequences", action="store_true", default=True, help="启用动态序列打包")
    parser.add_argument("--resume", type=str, default=None, help="从checkpoint恢复训练的路径")
    parser.add_argument("--zero_stage", type=int, default=3, choices=[1, 2, 3], help="ZeRO优化阶段 (默认3)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="启用梯度检查点")
    parser.add_argument("--tokenizer_path", type=str, default="qingxiaotuan", help="tokenizer路径")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_tokens_per_step(args, world_size):
    """计算每步处理的token数

    公式：per_device_batch_size * world_size * gradient_accumulation_steps * max_length
    """
    tokens_per_step = (args.per_device_train_batch_size *
                       world_size *
                       args.gradient_accumulation_steps *
                       args.max_length)
    return tokens_per_step


def build_deepspeed_config(args, world_size):
    """构建DeepSpeed配置，针对512K长序列优化

    - ZeRO-3：将模型参数、梯度、优化器状态分片到所有GPU
    - 高效的通信设置：减少all-gather和reduce-scatter的频率
    - 针对梯度累积12步优化通信
    """
    zero_optimization = {
        "stage": args.zero_stage,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
    }

    ds_config = {
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": args.weight_decay,
            },
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.warmup_steps,
                "total_num_steps": args.max_train_steps,
            },
        },
        "fp16": {
            "enabled": False,
        },
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": zero_optimization,
        "gradient_clipping": 1.0,
        "steps_per_print": args.logging_steps,
        "wall_clock_breakdown": False,
        "comms_logger": {
            "enabled": False,
        },
        "communication_data_type": "bf16",
        "gradient_accumulation_dtype": "bf16",
    }

    if args.flash_attention:
        ds_config["flashattention"] = {
            "enabled": True,
        }

    if args.sequence_parallel:
        ds_config["sequence_parallel"] = {
            "enabled": True,
            "sequence_parallel_group_size": world_size,
            "gather_output": True,
        }

    return ds_config


def main():
    args = parse_args()

    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print("青小团 Qingxiaotuan 分布式预训练启动")
        print(f"版本: 512K长序列 + {args.gradient_accumulation_steps}步梯度累积")
        print("=" * 60)
        print(f"世界大小 (world_size): {world_size}")
        print(f"总训练步数: {args.max_train_steps}")
        print(f"Batch size (每设备): {args.per_device_train_batch_size}")
        print(f"梯度累积步数: {args.gradient_accumulation_steps}")
        print(f"有效batch size: {args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"最大序列长度: {args.max_length} ({args.max_length // 1024}K)")
        tokens_per_step = calculate_tokens_per_step(args, world_size)
        print(f"每步处理token数: {tokens_per_step / 1e6:.2f}M ({tokens_per_step / 1e9:.2f}B)")
        print(f"ZeRO阶段: {args.zero_stage}")
        print(f"序列并行: {args.sequence_parallel}")
        print(f"Flash Attention: {args.flash_attention}")
        print(f"序列打包: {args.pack_sequences}")
        print(f"梯度检查点: {args.gradient_checkpointing}")
        print("=" * 60)

    set_seed(args.seed)

    tokenizer = QingxiaotuanTokenizer.from_pretrained(args.tokenizer_path)

    if args.model_name_or_path is not None:
        if rank == 0:
            print(f"从预训练路径加载模型: {args.model_name_or_path}")
        model = QingxiaotuanForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        if rank == 0:
            print("从头初始化青小团模型 (QingxiaotuanForCausalLM)")
        config = QingxiaotuanConfig()
        config.max_position_embeddings = args.max_length
        config.use_flash_attention = args.flash_attention
        config.sequence_parallel = args.sequence_parallel
        model = QingxiaotuanForCausalLM(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if rank == 0:
            print("梯度检查点已启用")

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数量: {num_params / 1e9:.2f}B")
        print(f"可训练参数量: {num_trainable_params / 1e9:.2f}B")

    if rank == 0:
        print(f"加载数据集: {args.data_path}")

    dataset = QingxiaotuanPretrainDataset(
        data_files=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pack_sequences=args.pack_sequences,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collate_fn_for_packing if args.pack_sequences else None,
        num_workers=2,
        pin_memory=True,
    )

    ds_config = build_deepspeed_config(args, world_size)

    resume_path = args.resume
    if resume_path is not None and rank == 0:
        print(f"从checkpoint恢复训练: {resume_path}")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
    )

    start_step = 0
    if resume_path is not None:
        load_path, client_state = model_engine.load_checkpoint(resume_path)
        if load_path is not None:
            start_step = client_state.get("global_step", 0)
            if rank == 0:
                print(f"成功从 {load_path} 恢复训练，起始步数: {start_step}")
        else:
            if rank == 0:
                print(f"警告: 从 {resume_path} 恢复失败，从头开始训练")

    global_step = start_step
    loss_sum = 0.0
    log_loss_sum = 0.0
    tokens_accumulated = 0

    model_engine.train()

    data_iter = iter(dataloader)

    if rank == 0:
        print("开始训练...")
        print("=" * 60)

    while global_step < args.max_train_steps:
        for batch in data_iter:
            if global_step >= args.max_train_steps:
                break

            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            loss_weights = batch.get("loss_weights", None)
            attention_mask = batch.get("attention_mask", None)

            if loss_weights is not None:
                loss_weights = loss_weights.to(model_engine.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_engine.device)

            batch_tokens = input_ids.numel()
            tokens_accumulated += batch_tokens

            model_kwargs = {"input_ids": input_ids, "labels": labels}
            if attention_mask is not None:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model_engine(**model_kwargs)
            loss = outputs.loss

            if loss_weights is not None:
                loss = (loss * loss_weights).mean()

            model_engine.backward(loss)
            model_engine.step()

            global_step += 1
            loss_val = loss.item()
            loss_sum += loss_val
            log_loss_sum += loss_val

            if global_step % args.logging_steps == 0:
                avg_loss = log_loss_sum / args.logging_steps
                lr = model_engine.get_lr()[0]
                avg_tokens_per_step = tokens_accumulated / args.logging_steps
                tokens_per_second = tokens_accumulated / (args.logging_steps * args.gradient_accumulation_steps)
                if rank == 0:
                    print(
                        f"[青小团] 步 {global_step}/{args.max_train_steps} | "
                        f"loss: {avg_loss:.4f} | "
                        f"lr: {lr:.2e} | "
                        f"tokens/步: {avg_tokens_per_step / 1e6:.2f}M | "
                        f"tokens/s/GPU: {tokens_per_second / 1e3:.1f}K"
                    )
                log_loss_sum = 0.0
                tokens_accumulated = 0

            if global_step % args.save_steps == 0:
                if rank == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    print(f"保存checkpoint到: {save_dir}")
                model_engine.save_checkpoint(
                    save_dir,
                    client_state={"global_step": global_step},
                )

    if rank == 0:
        final_save_dir = os.path.join(args.output_dir, "final")
        print(f"训练完成，保存最终模型到: {final_save_dir}")
        model_engine.save_checkpoint(
            final_save_dir,
            client_state={"global_step": global_step},
        )
        print("青小团预训练完成！")


if __name__ == "__main__":
    main()
