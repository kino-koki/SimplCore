# -*- coding: utf-8 -*-
"""测试EQP模型实现."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from SimplCore.architecture.eqp import SimplCoreWithEQP, compute_parameter_efficiency


def main():
    print("=" * 70)
    print("  EQP 模型测试")
    print("=" * 70)

    # 创建小型EQP模型
    print("\n[1] 创建小型EQP模型...")
    model = SimplCoreWithEQP(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_primitives=64,
        primitive_size=32,
        k_compose=2,
        max_seq_len=32,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params/1e6:.2f} M")

    # 前向传播测试
    print("\n[2] 前向传播测试...")
    input_ids = torch.randint(0, 256, (2, 16))
    out = model(input_ids)
    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {out['logits'].shape}")
    print(f"  Loss: {out['loss']}")

    # 训练模式测试
    print("\n[3] 训练模式测试...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    out = model(input_ids, labels=input_ids)
    loss = out["loss"]
    loss.backward()
    optimizer.step()

    print(f"  Loss after step: {loss.item():.4f}")
    print("  ✓ 训练模式正常")

    # 参数效率分析
    print("\n[4] 参数效率分析...")
    efficiency = compute_parameter_efficiency(
        d_model=4096,
        n_layers=32,
        n_primitives=4096,
        primitive_size=256,
        k_compose=4,
    )
    print(f"  标准Transformer: {efficiency['std_transformer_params_B']:.2f} B")
    print(f"  EQP SimplCore: {efficiency['eqp_SimplCore_params_B']:.2f} B")
    print(f"  等效容量: {efficiency['equivalent_capacity_B']:.2f} B")
    print(f"  参数效率比: {efficiency['parameter_efficiency_ratio']:.1f}x")
    print(f"  压缩比: {efficiency['compression_ratio']:.1f}x")

    print("\n" + "=" * 70)
    print("  所有测试通过!")
    print("=" * 70)


if __name__ == "__main__":
    main()

