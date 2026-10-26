# -*- coding: utf-8 -*-
"""
SimplCore vs PyTorch 性能对比基准测试
测试核心张量操作和神经网络层的性能
"""
import sys
import time
import numpy as np

print("=== 正在导入 SimplCore... ===")
from SimplCore.engine.tensor import randn, ones, zeros, softmax, layer_norm
from SimplCore.engine.nn import Linear, LayerNorm, Conv2d

print("=== 正在导入 PyTorch... ===")
import torch

def benchmark(name, func, iterations=10, warmup=5):
    """性能基准测试"""
    for _ in range(warmup):
        func()
    
    start = time.time()
    for _ in range(iterations):
        func()
    elapsed = time.time() - start
    
    print(f"{name}: {elapsed/iterations*1000:.2f} ms/iter")
    return elapsed / iterations

print("\n=== 张量操作性能对比 ===")

# 1. 矩阵乘法
print("\n--- 矩阵乘法 ---")
simp_a = randn(256, 256, requires_grad=True)
simp_b = randn(256, 256, requires_grad=True)
torch_a = torch.randn(256, 256, requires_grad=True)
torch_b = torch.randn(256, 256, requires_grad=True)

simp_time = benchmark("SimplCore matmul", lambda: (simp_a.matmul(simp_b) ** 2).sum().backward())
torch_time = benchmark("PyTorch matmul", lambda: (torch.matmul(torch_a, torch_b) ** 2).sum().backward())
print(f"加速比: {torch_time/simp_time:.2f}x")

# 2. Softmax
print("\n--- Softmax ---")
simp_x = randn(64, 1024, requires_grad=True)
torch_x = torch.randn(64, 1024, requires_grad=True)

simp_time = benchmark("SimplCore softmax", lambda: softmax(simp_x, dim=-1).sum().backward())
torch_time = benchmark("PyTorch softmax", lambda: torch.softmax(torch_x, dim=-1).sum().backward())
print(f"加速比: {torch_time/simp_time:.2f}x")

# 3. LayerNorm
print("\n--- LayerNorm ---")
simp_norm = LayerNorm(512)
simp_x = randn(64, 128, 512, requires_grad=True)
torch_norm = torch.nn.LayerNorm(512)
torch_x = torch.randn(64, 128, 512, requires_grad=True)

simp_time = benchmark("SimplCore LayerNorm", lambda: simp_norm(simp_x).sum().backward())
torch_time = benchmark("PyTorch LayerNorm", lambda: torch_norm(torch_x).sum().backward())
print(f"加速比: {torch_time/simp_time:.2f}x")

# 4. Linear层
print("\n--- Linear层 ---")
simp_lin = Linear(512, 1024)
simp_x = randn(64, 128, 512, requires_grad=True)
torch_lin = torch.nn.Linear(512, 1024)
torch_x = torch.randn(64, 128, 512, requires_grad=True)

simp_time = benchmark("SimplCore Linear", lambda: simp_lin(simp_x).sum().backward())
torch_time = benchmark("PyTorch Linear", lambda: torch_lin(torch_x).sum().backward())
print(f"加速比: {torch_time/simp_time:.2f}x")

# 5. Conv2d
print("\n--- Conv2d ---")
simp_conv = Conv2d(3, 64, kernel_size=3, padding=1)
simp_x = randn(16, 3, 64, 64, requires_grad=True)
torch_conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
torch_x = torch.randn(16, 3, 64, 64, requires_grad=True)

simp_time = benchmark("SimplCore Conv2d", lambda: simp_conv(simp_x).sum().backward())
torch_time = benchmark("PyTorch Conv2d", lambda: torch_conv(torch_x).sum().backward())
print(f"加速比: {torch_time/simp_time:.2f}x")

print("\n=== 架构模块性能对比 ===")

from SimplCore.engine.nn import ResonanceFieldLayer, DynamicMemoryChannel, TopologyRouter

# 6. ResonanceFieldLayer
print("\n--- ResonanceFieldLayer ---")
simp_rfp = ResonanceFieldLayer(dim=128, num_waves=8, damping=0.1, num_steps=3)
simp_x = randn(2, 64, 128, requires_grad=True)

try:
    simp_time = benchmark("SimplCore RFP", lambda: simp_rfp(simp_x).sum().backward())
    print(f"SimplCore: {simp_time*1000:.2f} ms/iter")
except Exception as e:
    print(f"SimplCore RFP failed: {e}")

# 7. TopologyRouter
print("\n--- TopologyRouter ---")
simp_tar = TopologyRouter(dim=128, num_paths=4, min_depth=2, max_depth=8, topk=2)
simp_x = randn(2, 64, 128, requires_grad=True)

try:
    simp_time = benchmark("SimplCore TAR", lambda: simp_tar(simp_x).sum().backward())
    print(f"SimplCore: {simp_time*1000:.2f} ms/iter")
except Exception as e:
    print(f"SimplCore TAR failed: {e}")

print("\n=== 测试完成 ===")

