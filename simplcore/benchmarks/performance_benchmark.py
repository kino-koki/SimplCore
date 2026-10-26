# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Performance Benchmark
性能基准测试，对比C++扩展和纯Python实现的性能差异
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import numpy as np
import time
from SimplCore.engine.perf_backend import matmul, softmax, layer_norm, relu, gelu, sigmoid, tanh

def benchmark_matmul():
    print("=" * 70)
    print("矩阵乘法性能测试")
    print("=" * 70)
    
    test_cases = [
        (100, 100, 100),
        (200, 200, 200),
        (512, 512, 512),
        (1024, 1024, 1024),
        (128, 256, 512),
    ]
    
    for m, k, n in test_cases:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        
        start = time.time()
        for _ in range(10):
            result = matmul(a, b)
        elapsed = (time.time() - start) / 10
        
        gflops = (2 * m * k * n) / (elapsed * 1e9)
        print(f"  {m}x{k} x {k}x{n}: {elapsed*1000:.2f} ms, {gflops:.2f} GFLOPS")

def benchmark_softmax():
    print("\n" + "=" * 70)
    print("Softmax性能测试")
    print("=" * 70)
    
    test_cases = [
        (32, 1024),
        (64, 2048),
        (128, 4096),
        (256, 8192),
    ]
    
    for batch, dim in test_cases:
        x = np.random.randn(batch, dim).astype(np.float32)
        
        start = time.time()
        for _ in range(100):
            result = softmax(x, dim=-1)
        elapsed = (time.time() - start) / 100
        
        print(f"  ({batch}, {dim}): {elapsed*1000:.2f} ms")

def benchmark_layer_norm():
    print("\n" + "=" * 70)
    print("LayerNorm性能测试")
    print("=" * 70)
    
    test_cases = [
        (32, 10, 1024),
        (64, 20, 2048),
        (128, 32, 4096),
    ]
    
    for batch, seq_len, dim in test_cases:
        x = np.random.randn(batch, seq_len, dim).astype(np.float32)
        w = np.random.randn(dim).astype(np.float32)
        b = np.random.randn(dim).astype(np.float32)
        
        start = time.time()
        for _ in range(50):
            result = layer_norm(x, w, b)
        elapsed = (time.time() - start) / 50
        
        print(f"  ({batch}, {seq_len}, {dim}): {elapsed*1000:.2f} ms")

def benchmark_activations():
    print("\n" + "=" * 70)
    print("激活函数性能测试")
    print("=" * 70)
    
    test_cases = [
        (1000, 1000),
        (2000, 2000),
        (4000, 4000),
    ]
    
    for rows, cols in test_cases:
        x = np.random.randn(rows, cols).astype(np.float32)
        
        for fn_name, fn in [('ReLU', relu), ('GELU', gelu), ('Sigmoid', sigmoid), ('Tanh', tanh)]:
            start = time.time()
            for _ in range(100):
                result = fn(x)
            elapsed = (time.time() - start) / 100
            
            print(f"  {fn_name} ({rows}x{cols}): {elapsed*1000:.2f} ms")

def main():
    print("=" * 70)
    print("SimplCore 性能基准测试")
    print("=" * 70)
    print(f"NumPy版本: {np.__version__}")
    
    benchmark_matmul()
    benchmark_softmax()
    benchmark_layer_norm()
    benchmark_activations()
    
    print("\n" + "=" * 70)
    print("性能测试完成!")
    print("=" * 70)

if __name__ == '__main__':
    main()

