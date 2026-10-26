# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Benchmark - 性能基准测试模块
与PyTorch对比验证性能
"""
from __future__ import annotations

import numpy as np
import time
import json
from typing import Dict, Any, List, Tuple

try:
    import simplcore.tensor as torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .tensor import SimpTensor, randn, softmax, relu, gelu, layer_norm
from .nn import Linear, Conv2d, MaxPool2d, BatchNorm2d, Sequential, GELU


class BenchmarkResult:
    """基准测试结果"""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def add_result(self, name: str, SimplCore_time: float, torch_time: float = None, speedup: float = None):
        """添加测试结果"""
        result = {
            "name": name,
            "SimplCore_time": SimplCore_time,
            "torch_time": torch_time,
            "speedup": speedup,
        }
        self.results.append(result)
    
    def summary(self) -> str:
        """生成摘要"""
        lines = []
        lines.append("=" * 70)
        lines.append("SimplCore vs PyTorch 性能基准测试")
        lines.append("=" * 70)
        lines.append(f"{'测试项':<30} {'SimplCore':<12} {'PyTorch':<12} {'Speedup':<10}")
        lines.append("-" * 70)
        
        for result in self.results:
            name = result["name"]
            SimplCore_time = f"{result['SimplCore_time']:.6f}s"
            torch_time = f"{result['torch_time']:.6f}s" if result["torch_time"] is not None else "N/A"
            speedup = f"{result['speedup']:.2f}x" if result["speedup"] is not None else "N/A"
            lines.append(f"{name:<30} {SimplCore_time:<12} {torch_time:<12} {speedup:<10}")
        
        lines.append("-" * 70)
        
        if TORCH_AVAILABLE:
            valid_results = [r for r in self.results if r["speedup"] is not None]
            if valid_results:
                avg_speedup = np.mean([r["speedup"] for r in valid_results])
                lines.append(f"平均加速比: {avg_speedup:.2f}x")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.results, indent=2)


class TensorBenchmark:
    """张量操作基准测试"""
    
    def __init__(self, sizes: List[Tuple[int, ...]] = None):
        if sizes is None:
            self.sizes = [
                (100, 100),
                (500, 500),
                (1000, 1000),
                (2000, 2000),
            ]
        else:
            self.sizes = sizes
    
    def run(self) -> BenchmarkResult:
        """运行测试"""
        result = BenchmarkResult()
        
        for size in self.sizes:
            result.add_result(
                f"Matmul {size}",
                *self._benchmark_matmul(size)
            )
            
            result.add_result(
                f"Softmax {size}",
                *self._benchmark_softmax(size)
            )
            
            result.add_result(
                f"ReLU {size}",
                *self._benchmark_relu(size)
            )
            
            result.add_result(
                f"GELU {size}",
                *self._benchmark_gelu(size)
            )
            
            result.add_result(
                f"LayerNorm {size}",
                *self._benchmark_layer_norm(size)
            )
        
        return result
    
    def _benchmark_matmul(self, size: Tuple[int, ...]) -> Tuple[float, float, float]:
        """矩阵乘法基准测试"""
        a = randn(*size)
        b = randn(*size)
        
        start = time.time()
        c = a.matmul(b)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            ta = torch.randn(*size)
            tb = torch.randn(*size)
            
            start = time.time()
            tc = torch.matmul(ta, tb)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup
    
    def _benchmark_softmax(self, size: Tuple[int, ...]) -> Tuple[float, float, float]:
        """Softmax基准测试"""
        x = randn(*size)
        
        start = time.time()
        y = softmax(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            
            start = time.time()
            ty = torch.softmax(tx, dim=-1)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup
    
    def _benchmark_relu(self, size: Tuple[int, ...]) -> Tuple[float, float, float]:
        """ReLU基准测试"""
        x = randn(*size)
        
        start = time.time()
        y = relu(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            
            start = time.time()
            ty = torch.relu(tx)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup
    
    def _benchmark_gelu(self, size: Tuple[int, ...]) -> Tuple[float, float, float]:
        """GELU基准测试"""
        x = randn(*size)
        
        start = time.time()
        y = gelu(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            
            start = time.time()
            ty = torch.nn.functional.gelu(tx)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup
    
    def _benchmark_layer_norm(self, size: Tuple[int, ...]) -> Tuple[float, float, float]:
        """LayerNorm基准测试"""
        x = randn(*size)
        
        start = time.time()
        y = layer_norm(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            
            start = time.time()
            ty = torch.nn.functional.layer_norm(tx, tx.shape[-1:])
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup


class CNNBenchmark:
    """CNN基准测试"""
    
    def __init__(self, input_sizes: List[Tuple[int, int, int, int]] = None):
        if input_sizes is None:
            self.input_sizes = [
                (16, 3, 32, 32),
                (32, 3, 64, 64),
                (16, 3, 128, 128),
            ]
        else:
            self.input_sizes = input_sizes
    
    def run(self) -> BenchmarkResult:
        """运行测试"""
        result = BenchmarkResult()
        
        for size in self.input_sizes:
            result.add_result(
                f"Conv2d {size}",
                *self._benchmark_conv2d(size)
            )
            
            result.add_result(
                f"MaxPool2d {size}",
                *self._benchmark_maxpool2d(size)
            )
            
            result.add_result(
                f"BatchNorm2d {size}",
                *self._benchmark_batchnorm2d(size)
            )
            
            result.add_result(
                f"CNN Model {size}",
                *self._benchmark_cnn_model(size)
            )
        
        return result
    
    def _benchmark_conv2d(self, size: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """Conv2d基准测试"""
        x = randn(*size)
        conv = Conv2d(size[1], 64, kernel_size=3, padding=1)
        
        start = time.time()
        y = conv(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            tconv = torch.nn.Conv2d(size[1], 64, kernel_size=3, padding=1)
            
            start = time.time()
            ty = tconv(tx)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup
    
    def _benchmark_maxpool2d(self, size: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """MaxPool2d基准测试"""
        x = randn(*size)
        pool = MaxPool2d(kernel_size=2)
        
        start = time.time()
        y = pool(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            tpool = torch.nn.MaxPool2d(kernel_size=2)
            
            start = time.time()
            ty = tpool(tx)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup
    
    def _benchmark_batchnorm2d(self, size: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """BatchNorm2d基准测试"""
        x = randn(*size)
        bn = BatchNorm2d(size[1])
        
        start = time.time()
        y = bn(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            tbn = torch.nn.BatchNorm2d(size[1])
            
            start = time.time()
            ty = tbn(tx)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup
    
    def _benchmark_cnn_model(self, size: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """CNN模型基准测试"""
        x = randn(*size)
        
        model = Sequential(
            Conv2d(size[1], 64, kernel_size=3, padding=1),
            GELU(),
            MaxPool2d(2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            GELU(),
            MaxPool2d(2),
        )
        
        start = time.time()
        y = model(x)
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            tx = torch.randn(*size)
            
            tmodel = torch.nn.Sequential(
                torch.nn.Conv2d(size[1], 64, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.MaxPool2d(2),
            )
            
            start = time.time()
            ty = tmodel(tx)
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup


class AutoGradBenchmark:
    """自动微分基准测试"""
    
    def __init__(self, sizes: List[Tuple[int, ...]] = None):
        if sizes is None:
            self.sizes = [
                (100, 100),
                (500, 500),
                (1000, 1000),
            ]
        else:
            self.sizes = sizes
    
    def run(self) -> BenchmarkResult:
        """运行测试"""
        result = BenchmarkResult()
        
        for size in self.sizes:
            result.add_result(
                f"Backward {size}",
                *self._benchmark_backward(size)
            )
        
        return result
    
    def _benchmark_backward(self, size: Tuple[int, int]) -> Tuple[float, float, float]:
        """反向传播基准测试"""
        a = randn(*size, requires_grad=True)
        b = randn(*size, requires_grad=True)
        
        start = time.time()
        c = a.matmul(b)
        d = c.sum()
        d.backward()
        SimplCore_time = time.time() - start
        
        torch_time = None
        speedup = None
        
        if TORCH_AVAILABLE:
            ta = torch.randn(*size, requires_grad=True)
            tb = torch.randn(*size, requires_grad=True)
            
            start = time.time()
            tc = torch.matmul(ta, tb)
            td = tc.sum()
            td.backward()
            torch_time = time.time() - start
            
            speedup = torch_time / SimplCore_time if SimplCore_time > 0 else 0
        
        return SimplCore_time, torch_time, speedup


def run_all_benchmarks() -> BenchmarkResult:
    """运行所有基准测试"""
    print("=" * 70)
    print("SimplCore 性能基准测试")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping comparison")
    
    combined_result = BenchmarkResult()
    
    print("\n--- 张量操作基准测试 ---")
    tensor_benchmark = TensorBenchmark()
    tensor_result = tensor_benchmark.run()
    print(tensor_result.summary())
    combined_result.results.extend(tensor_result.results)
    
    print("\n--- CNN基准测试 ---")
    cnn_benchmark = CNNBenchmark()
    cnn_result = cnn_benchmark.run()
    print(cnn_result.summary())
    combined_result.results.extend(cnn_result.results)
    
    print("\n--- 自动微分基准测试 ---")
    grad_benchmark = AutoGradBenchmark()
    grad_result = grad_benchmark.run()
    print(grad_result.summary())
    combined_result.results.extend(grad_result.results)
    
    print("\n" + "=" * 70)
    print("综合基准测试结果")
    print("=" * 70)
    print(combined_result.summary())
    
    return combined_result


if __name__ == "__main__":
    run_all_benchmarks()

