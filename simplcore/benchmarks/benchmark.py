# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 基准测试工具
====================
测试SimplCore架构的计算效率和内存使用。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict

from SimplCore.architecture.theory import SimplCoreConfig
from SimplCore.architecture.integrated_model import SimplCoreModel


@dataclass
class BenchmarkResult:
    """基准测试结果。"""
    model_size: str
    params_millions: float
    forward_time_ms: float
    memory_mb: float
    throughput_tokens_per_sec: float
    peak_memory_mb: float


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark_model(
    config: SimplCoreConfig,
    batch_size: int = 2,
    seq_len: int = 128,
    n_warmup: int = 3,
    n_iters: int = 10,
    device: str = "cpu",
) -> BenchmarkResult:
    """对一个模型配置运行基准测试。"""
    model = SimplCoreModel(config).to(device)
    model.eval()

    total_params = count_parameters(model)
    input_ids = torch.randint(1, 100, (batch_size, seq_len)).to(device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids=input_ids)

    # 计时
    torch.cuda.synchronize() if device != "cpu" else None
    start_time = time.time()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = model(input_ids=input_ids)
    torch.cuda.synchronize() if device != "cpu" else None
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / n_iters * 1000

    # 吞吐量
    tokens_per_sec = batch_size * seq_len / (avg_time_ms / 1000)

    # 内存估算
    param_bytes = total_params * 4  # float32
    activation_bytes = batch_size * seq_len * config.d_model * 4 * config.n_layers
    total_mb = (param_bytes + activation_bytes) / (1024 * 1024)

    return BenchmarkResult(
        model_size=f"{config.n_layers}L{config.d_model}D",
        params_millions=total_params / 1e6,
        forward_time_ms=avg_time_ms,
        memory_mb=total_mb,
        throughput_tokens_per_sec=tokens_per_sec,
        peak_memory_mb=total_mb * 1.2,  # 估算峰值
    )


def run_scale_benchmarks():
    """运行不同规模的基准测试。"""
    scales = [
        ("nano", SimplCoreConfig(d_model=64, n_layers=2, n_bases=16, n_heads=2,
                                 n_memory_channels=4, memory_dim=64,
                                 n_paths=2, top_k=1, ffn_dim=128,
                                 vocab_size_text=1000, max_seq_len=128)),
        ("tiny", SimplCoreConfig(d_model=128, n_layers=4, n_bases=32, n_heads=4,
                                 n_memory_channels=8, memory_dim=128,
                                 n_paths=4, top_k=2, ffn_dim=256,
                                 vocab_size_text=1000, max_seq_len=256)),
        ("small", SimplCoreConfig(d_model=256, n_layers=6, n_bases=64, n_heads=8,
                                  n_memory_channels=16, memory_dim=256,
                                  n_paths=8, top_k=2, ffn_dim=512,
                                  vocab_size_text=2000, max_seq_len=512)),
        ("base", SimplCoreConfig(d_model=512, n_layers=8, n_bases=128, n_heads=8,
                                 n_memory_channels=32, memory_dim=512,
                                 n_paths=8, top_k=2, ffn_dim=1024,
                                 vocab_size_text=4000, max_seq_len=512)),
        ("7B", SimplCoreConfig.from_preset("7B")),
    ]

    results = []
    print("\n" + "=" * 80)
    print("SimplCore 架构基准测试报告")
    print("=" * 80)
    print(f"{'Scale':<10} {'Params(M)':<12} {'Forward(ms)':<14} {'Memory(MB)':<14} {'Tokens/s':<14}")
    print("-" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    for name, config in scales:
        result = benchmark_model(config, device=device)
        results.append(result)
        print(f"{name:<10} {result.params_millions:<12.2f} {result.forward_time_ms:<14.2f} "
              f"{result.memory_mb:<14.2f} {result.throughput_tokens_per_sec:<14.2f}")

    print("=" * 80)
    return results


def compare_with_transformer():
    """理论对比SimplCore vs Transformer的复杂度。"""
    print("\n" + "=" * 80)
    print("复杂度对比：SimplCore vs Transformer")
    print("=" * 80)

    seq_lens = [256, 512, 1024, 2048, 4096, 8192]
    d_model = 4096

    print(f"{'SeqLen':<10} {'Transformer O(n²)':<20} {'SimplCore O(n log n)':<22} {'加速比':<10}")
    print("-" * 80)

    for n in seq_lens:
        transformer_flops = n * n * d_model * 4  # QKV + output
        SimplCore_flops = n * int(d_model * 0.5) * d_model * 2  # SGC bases
        speedup = transformer_flops / SimplCore_flops
        print(f"{n:<10} {transformer_flops:<20,} {SimplCore_flops:<22,} {speedup:<10.2f}x")

    print("=" * 80)


def test_self_evolution():
    """测试自进化引擎的效果。"""
    print("\n" + "=" * 80)
    print("自进化引擎效果测试")
    print("=" * 80)

    config = SimplCoreConfig(d_model=64, n_layers=2, n_bases=16, n_heads=2,
                             n_memory_channels=4, memory_dim=64,
                             n_paths=2, top_k=1, ffn_dim=128,
                             vocab_size_text=1000, max_seq_len=128)
    model = SimplCoreModel(config)
    model.eval()

    input_ids = torch.randint(1, 100, (1, 16))

    # 记录初始参数
    initial_weight = model.layers[0].spm.plastic_weight.clone()

    # 进行多次推理
    n_steps = 100
    for i in range(n_steps):
        _ = model(input_ids=input_ids, use_evolution=True)

    final_weight = model.layers[0].spm.plastic_weight
    weight_change = (final_weight - initial_weight).norm().item()

    report = model.evolution_engine.get_evolution_report()

    print(f"自进化步数: {report['evolution_steps']}")
    print(f"塑性权重变化 (L2): {weight_change:.6f}")
    print(f"自我改进率: {report['self_improvement_rate']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimplCore 基准测试工具")
    parser.add_argument("--all", action="store_true", help="运行全部测试")
    parser.add_argument("--scales", action="store_true", help="运行规模基准测试")
    parser.add_argument("--compare", action="store_true", help="运行复杂度对比")
    parser.add_argument("--evolution", action="store_true", help="运行自进化测试")
    args = parser.parse_args()

    if args.all or len(sys.argv) == 1:
        run_scale_benchmarks()
        compare_with_transformer()
        test_self_evolution()
    else:
        if args.scales:
            run_scale_benchmarks()
        if args.compare:
            compare_with_transformer()
        if args.evolution:
            test_self_evolution()


