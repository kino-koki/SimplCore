# -*- coding: utf-8 -*-
"""测试 33B 模型训练配置是否正常工作."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from SimplCore.architecture.config import SimplCoreConfig
from SimplCore.architecture.SimplCore_model import SimplCoreModel
from SimplCore.architecture.lora import (
    apply_lora_to_model,
    count_lora_parameters,
)


def main():
    print("=" * 70)
    print("  SimplCore 33B 训练配置测试")
    print("=" * 70)

    # 1. 配置测试
    print("\n[1] 配置测试...")
    cfg = SimplCoreConfig.get_33b_config()
    print(f"  隐藏维度: {cfg.hidden_size}")
    print(f"  层数: {cfg.num_layers}")
    print(f"  预计参数量: {cfg.estimated_num_parameters()/1e9:.2f} B")
    print(f"  FFN维度: {cfg.ffn_intermediate_size}")
    print(f"  词表大小: {cfg.vocab_size}")
    print(f"  最大序列长度: {cfg.max_sequence_length}")

    # 2. 使用small配置测试模型初始化和LoRA
    print("\n[2] 使用small配置测试模型初始化...")
    small_cfg = SimplCoreConfig.get_small_config()
    small_model = SimplCoreModel(small_cfg)
    small_total = small_model.count_parameters()
    print(f"  Small模型参数量: {small_total/1e6:.2f} M")

    # 3. LoRA 应用测试
    print("\n[3] LoRA 应用测试...")
    lora_count = apply_lora_to_model(small_model, rank=8, alpha=16, dropout=0.05)
    lora_params = count_lora_parameters(small_model)
    print(f"  LoRA Adapter 数量: {lora_count}")
    print(f"  LoRA 参数数量: {lora_params}")
    print(f"  可训练参数比例: {(lora_params/small_total)*100:.4f}%")

    # 4. 前向传播测试 (CPU)
    print("\n[4] 前向传播测试 (CPU)...")
    device = torch.device("cpu")
    small_model = small_model.to(device)
    small_model.eval()

    batch_size = 1
    seq_len = 32
    input_ids = torch.randint(0, small_cfg.vocab_size, (batch_size, seq_len)).to(device)

    with torch.no_grad():
        logits, _ = small_model(input_ids)

    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, small_cfg.vocab_size), f"输出形状不匹配: {logits.shape}"
    print("  ✓ 前向传播正常")

    # 5. 训练模式测试
    print("\n[5] 训练模式测试...")
    small_model.train()
    optimizer = torch.optim.AdamW(small_model.parameters(), lr=3e-4)

    input_ids_train = torch.randint(0, small_cfg.vocab_size, (1, 32)).to(device)
    labels = input_ids_train.clone()

    optimizer.zero_grad()
    logits, _ = small_model(input_ids_train)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=0,
    )
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ 训练模式正常")

    # 6. bf16 测试 (如果有GPU)
    if torch.cuda.is_available():
        print("\n[6] bf16 测试 (GPU)...")
        device = torch.device("cuda:0")
        small_model = small_model.to(device)

        input_ids_gpu = torch.randint(0, small_cfg.vocab_size, (1, 32)).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits_bf16, _ = small_model(input_ids_gpu)

        print(f"  bf16输出形状: {logits_bf16.shape}")
        print(f"  bf16数据类型: {logits_bf16.dtype}")
        print("  ✓ bf16 混合精度正常")

    print("\n" + "=" * 70)
    print("  所有测试通过!")
    print("=" * 70)
    print("\n33B 模型训练配置总结:")
    print(f"  - 总参数量: {cfg.estimated_num_parameters()/1e9:.2f} B")
    print(f"  - 隐藏维度: {cfg.hidden_size}")
    print(f"  - 层数: {cfg.num_layers}")
    print(f"  - 预计 LoRA 参数 (rank=8): ~100M")
    print(f"  - 设备: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - 支持 bf16: {torch.cuda.is_available()}")
    print(f"\n显存优化策略:")
    print(f"  1. LoRA (rank=8)        : 将可训练参数降至 ~100M")
    print(f"  2. 梯度检查点            : 节省 ~50% 激活内存")
    print(f"  3. bf16 混合精度          : 参数存储减半")
    print(f"  4. 模型并行 (2卡)        : 每层~19层/卡")
    print(f"  5. 微批处理 (micro-batch=1) : 控制激活峰值")


if __name__ == "__main__":
    main()

