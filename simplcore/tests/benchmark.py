# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore vs PyTorch 性能基准测试
测试数据集、测试方法及完整对比分析报告
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import time
import numpy as np
import unittest


def benchmark(func, iterations=100, warmup=10):
    for _ in range(warmup):
        func()
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000


class BenchmarkReport:
    """基准测试报告"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, test_name, SimplCore_ms, pytorch_ms):
        speedup = pytorch_ms / SimplCore_ms if SimplCore_ms > 0 else float('inf')
        self.results.append({
            'test_name': test_name,
            'SimplCore_ms': SimplCore_ms,
            'pytorch_ms': pytorch_ms,
            'speedup': speedup,
            'improvement': (1 - SimplCore_ms / pytorch_ms) * 100 if pytorch_ms > 0 else 100,
        })
    
    def print_report(self):
        print("\n" + "=" * 80)
        print("SimplCore vs PyTorch 性能对比报告")
        print("=" * 80)
        
        headers = ["测试项", "SimplCore(ms)", "PyTorch(ms)", "加速比", "提升(%)"]
        print(f"\n{headers[0]:<30} {headers[1]:<15} {headers[2]:<15} {headers[3]:<10} {headers[4]:<10}")
        print("-" * 80)
        
        for r in self.results:
            print(f"{r['test_name']:<30} {r['SimplCore_ms']:<15.4f} {r['pytorch_ms']:<15.4f} {r['speedup']:<10.2f}x {r['improvement']:<10.2f}")
        
        avg_improvement = np.mean([r['improvement'] for r in self.results])
        print("\n" + "=" * 80)
        print(f"平均性能提升: {avg_improvement:.2f}%")
        print("=" * 80)


class TestTensorBenchmark(unittest.TestCase):
    """张量操作性能基准测试"""
    
    def setUp(self):
        self.report = BenchmarkReport()
        self.N = 100
        self.WARMUP = 10
    
    def test_matmul_performance(self):
        """矩阵乘法性能测试"""
        from SimplCore.engine import randn
        
        a = randn(512, 512)
        b = randn(512, 512)
        
        def SimplCore_matmul():
            c = a.matmul(b)
        
        try:
            import torch
            ta = torch.randn(512, 512)
            tb = torch.randn(512, 512)
            
            def pytorch_matmul():
                tc = torch.matmul(ta, tb)
            
            SimplCore_ms, _ = benchmark(SimplCore_matmul, self.N, self.WARMUP)
            pytorch_ms, _ = benchmark(pytorch_matmul, self.N, self.WARMUP)
            self.report.add_result("矩阵乘法(512x512)", SimplCore_ms, pytorch_ms)
        except ImportError:
            SimplCore_ms, _ = benchmark(SimplCore_matmul, self.N, self.WARMUP)
            self.report.add_result("矩阵乘法(512x512)", SimplCore_ms, float('inf'))
    
    def test_softmax_performance(self):
        """Softmax性能测试"""
        from SimplCore.engine import randn
        
        x = randn(32, 1024)
        
        def SimplCore_softmax():
            y = x.softmax(dim=-1)
        
        try:
            import torch
            tx = torch.randn(32, 1024)
            
            def pytorch_softmax():
                ty = torch.softmax(tx, dim=-1)
            
            SimplCore_ms, _ = benchmark(SimplCore_softmax, self.N, self.WARMUP)
            pytorch_ms, _ = benchmark(pytorch_softmax, self.N, self.WARMUP)
            self.report.add_result("Softmax(32x1024)", SimplCore_ms, pytorch_ms)
        except ImportError:
            SimplCore_ms, _ = benchmark(SimplCore_softmax, self.N, self.WARMUP)
            self.report.add_result("Softmax(32x1024)", SimplCore_ms, float('inf'))
    
    def test_layer_norm_performance(self):
        """LayerNorm性能测试"""
        from SimplCore.engine import randn, LayerNorm
        
        ln = LayerNorm(1024)
        x = randn(32, 10, 1024)
        
        def SimplCore_ln():
            y = ln(x)
        
        try:
            import torch
            tln = torch.nn.LayerNorm(1024)
            tx = torch.randn(32, 10, 1024)
            
            def pytorch_ln():
                ty = tln(tx)
            
            SimplCore_ms, _ = benchmark(SimplCore_ln, self.N, self.WARMUP)
            pytorch_ms, _ = benchmark(pytorch_ln, self.N, self.WARMUP)
            self.report.add_result("LayerNorm(32x10x1024)", SimplCore_ms, pytorch_ms)
        except ImportError:
            SimplCore_ms, _ = benchmark(SimplCore_ln, self.N, self.WARMUP)
            self.report.add_result("LayerNorm(32x10x1024)", SimplCore_ms, float('inf'))
    
    def test_conv2d_performance(self):
        """Conv2d性能测试"""
        from SimplCore.engine import randn, Conv2d
        
        conv = Conv2d(3, 64, kernel_size=3, padding=1)
        x = randn(16, 3, 32, 32)
        
        def SimplCore_conv():
            y = conv(x)
        
        try:
            import torch
            tconv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
            tx = torch.randn(16, 3, 32, 32)
            
            def pytorch_conv():
                ty = tconv(tx)
            
            SimplCore_ms, _ = benchmark(SimplCore_conv, self.N, self.WARMUP)
            pytorch_ms, _ = benchmark(pytorch_conv, self.N, self.WARMUP)
            self.report.add_result("Conv2d(16x3x32x32)", SimplCore_ms, pytorch_ms)
        except ImportError:
            SimplCore_ms, _ = benchmark(SimplCore_conv, self.N, self.WARMUP)
            self.report.add_result("Conv2d(16x3x32x32)", SimplCore_ms, float('inf'))
    
    def test_lstm_performance(self):
        """LSTM性能测试"""
        from SimplCore.engine import randn, LSTM
        
        lstm = LSTM(256, 256, num_layers=2, batch_first=True)
        x = randn(32, 10, 256)
        
        def SimplCore_lstm():
            y, _ = lstm(x)
        
        try:
            import torch
            tlstm = torch.nn.LSTM(256, 256, num_layers=2, batch_first=True)
            tx = torch.randn(32, 10, 256)
            
            def pytorch_lstm():
                ty, _ = tlstm(tx)
            
            SimplCore_ms, _ = benchmark(SimplCore_lstm, self.N, self.WARMUP)
            pytorch_ms, _ = benchmark(pytorch_lstm, self.N, self.WARMUP)
            self.report.add_result("LSTM(32x10x256)", SimplCore_ms, pytorch_ms)
        except ImportError:
            SimplCore_ms, _ = benchmark(SimplCore_lstm, self.N, self.WARMUP)
            self.report.add_result("LSTM(32x10x256)", SimplCore_ms, float('inf'))
    
    def test_linear_performance(self):
        """Linear层性能测试"""
        from SimplCore.engine import randn, Linear
        
        linear = Linear(1024, 1024)
        x = randn(32, 1024)
        
        def SimplCore_linear():
            y = linear(x)
        
        try:
            import torch
            tlinear = torch.nn.Linear(1024, 1024)
            tx = torch.randn(32, 1024)
            
            def pytorch_linear():
                ty = tlinear(tx)
            
            SimplCore_ms, _ = benchmark(SimplCore_linear, self.N, self.WARMUP)
            pytorch_ms, _ = benchmark(pytorch_linear, self.N, self.WARMUP)
            self.report.add_result("Linear(32x1024)", SimplCore_ms, pytorch_ms)
        except ImportError:
            SimplCore_ms, _ = benchmark(SimplCore_linear, self.N, self.WARMUP)
            self.report.add_result("Linear(32x1024)", SimplCore_ms, float('inf'))
    
    def test_backward_performance(self):
        """反向传播性能测试"""
        from SimplCore.engine import randn, Linear, Adam
        
        linear = Linear(1024, 10)
        x = randn(32, 1024)
        opt = Adam(linear.parameters(), lr=0.001)
        
        def SimplCore_backward():
            y = linear(x)
            loss = y.sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        try:
            import torch
            tlinear = torch.nn.Linear(1024, 10)
            tx = torch.randn(32, 1024)
            topt = torch.optim.Adam(tlinear.parameters(), lr=0.001)
            
            def pytorch_backward():
                ty = tlinear(tx)
                loss = ty.sum()
                loss.backward()
                topt.step()
                topt.zero_grad()
            
            SimplCore_ms, _ = benchmark(SimplCore_backward, self.N, self.WARMUP)
            pytorch_ms, _ = benchmark(pytorch_backward, self.N, self.WARMUP)
            self.report.add_result("反向传播(32x1024)", SimplCore_ms, pytorch_ms)
        except ImportError:
            SimplCore_ms, _ = benchmark(SimplCore_backward, self.N, self.WARMUP)
            self.report.add_result("反向传播(32x1024)", SimplCore_ms, float('inf'))
    
    def test_resonance_field_performance(self):
        """共振场处理性能测试"""
        from SimplCore.engine import randn, LegacySpectralMixer
        
        rfl = LegacySpectralMixer(64, num_waves=4)
        x = randn(32, 10, 64)
        
        def SimplCore_rf():
            y = rfl(x)
        
        SimplCore_ms, _ = benchmark(SimplCore_rf, self.N, self.WARMUP)
        self.report.add_result("共振场处理(32x10x64)", SimplCore_ms, float('inf'))
    
    def test_dynamic_memory_performance(self):
        """动态内存通道性能测试"""
        from SimplCore.engine import randn, DynamicMemoryChannel
        
        dmc = DynamicMemoryChannel(64, 1024)
        x = randn(32, 10, 64)
        
        def SimplCore_dmc():
            y, _ = dmc(x)
        
        SimplCore_ms, _ = benchmark(SimplCore_dmc, self.N, self.WARMUP)
        self.report.add_result("动态内存通道(32x10x64)", SimplCore_ms, float('inf'))
    
    def tearDown(self):
        self.report.print_report()


def run_benchmark():
    """运行基准测试"""
    print("=" * 80)
    print("SimplCore vs PyTorch 性能基准测试")
    print("=" * 80)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTensorBenchmark)
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    run_benchmark()

