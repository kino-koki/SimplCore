# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from SimplCore.architecture.eqp import PrimitivePool, CompositionRouter, EQPLinear, SimplCoreWithEQP

print("✅ 导入成功")

print("\n--- Test 1: PrimitivePool ---")
pool = PrimitivePool(n_primitives=256, primitive_size=64, d_model=256)
print(f"基元池形状: {pool.primitives.shape}")

print("\n--- Test 2: CompositionRouter ---")
router = CompositionRouter(d_model=256, n_primitives=256, k_compose=4, n_heads=8)
x = torch.randn(2, 16, 256)
indices, weights, aux_loss = router(x)
print(f"索引形状: {indices.shape}")
print(f"权重形状: {weights.shape}")
print(f"辅助损失: {aux_loss.item():.4f}")

print("\n--- Test 3: EQPLinear ---")
eqp_linear = EQPLinear(256, 256, pool, k_compose=4)
y = eqp_linear(x)
print(f"输出形状: {y.shape}")

print("\n--- Test 4: SimplCoreWithEQP ---")
model = SimplCoreWithEQP(
    vocab_size=4096,
    d_model=128,
    n_layers=4,
    ffn_dim=512,
    n_primitives=128,
    primitive_size=32,
    k_compose=3,
    max_seq_len=128,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数: {total_params/1e6:.2f}M")

input_ids = torch.randint(0, 4096, (2, 64))
out = model(input_ids)
print(f"logits形状: {out['logits'].shape}")

print("\n 所有测试通过!")
