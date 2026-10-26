# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location("tensor", os.path.join(os.path.dirname(__file__), "SimplCore", "engine", "tensor.py"))
tensor = importlib.util.module_from_spec(spec)
sys.modules['SimplCore.engine.tensor'] = tensor
spec.loader.exec_module(tensor)

spec_nn = importlib.util.spec_from_file_location("nn", os.path.join(os.path.dirname(__file__), "SimplCore", "engine", "nn.py"))
nn = importlib.util.module_from_spec(spec_nn)
sys.modules['SimplCore.engine.nn'] = nn
spec_nn.loader.exec_module(nn)

import time

print('=== 测试优化后的CNN性能 ===')

x = tensor.randn(16, 3, 32, 32)
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

start = time.time()
for _ in range(10):
    out = conv(x)
conv_time = (time.time() - start) / 10
print(f'Conv2d (16,3,32,32): {conv_time:.6f}s per call')

pool = nn.MaxPool2d(kernel_size=2)
start = time.time()
for _ in range(100):
    out = pool(x)
pool_time = (time.time() - start) / 100
print(f'MaxPool2d (16,3,32,32): {pool_time:.6f}s per call')

avgpool = nn.AvgPool2d(kernel_size=2)
start = time.time()
for _ in range(100):
    out = avgpool(x)
avgpool_time = (time.time() - start) / 100
print(f'AvgPool2d (16,3,32,32): {avgpool_time:.6f}s per call')

bn = nn.BatchNorm2d(3)
start = time.time()
for _ in range(1000):
    out = bn(x)
bn_time = (time.time() - start) / 1000
print(f'BatchNorm2d (16,3,32,32): {bn_time:.6f}s per call')

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.GELU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.GELU(),
    nn.MaxPool2d(2),
)

start = time.time()
for _ in range(10):
    out = model(x)
model_time = (time.time() - start) / 10
print(f'CNN Model (16,3,32,32): {model_time:.6f}s per call')

print(f'Output shape: {out.shape}')

print('\n=== 更大尺寸测试 ===')
x2 = tensor.randn(32, 3, 64, 64)
conv2 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
start = time.time()
for _ in range(5):
    out2 = conv2(x2)
conv2_time = (time.time() - start) / 5
print(f'Conv2d (32,3,64,64): {conv2_time:.6f}s per call')

x3 = tensor.randn(16, 3, 128, 128)
conv3 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
start = time.time()
out3 = conv3(x3)
conv3_time = time.time() - start
print(f'Conv2d (16,3,128,128): {conv3_time:.6f}s per call')

print('\n=== 自动微分测试 ===')
x_grad = tensor.randn(2, 3, 8, 8, requires_grad=True)
w_grad = tensor.randn(8, 3, 3, 3, requires_grad=True)
b_grad = tensor.randn(8, requires_grad=True)

out_grad = tensor.conv2d(x_grad, w_grad, b_grad, padding=1)
loss = out_grad.sum()
loss.backward()

print(f'Input grad shape: {x_grad.grad.shape}')
print(f'Weight grad shape: {w_grad.grad.shape}')
print(f'Bias grad shape: {b_grad.grad.shape}')
print('自动微分测试通过!')

print('\n=== 所有测试通过 ===')

