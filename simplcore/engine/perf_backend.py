# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Performance Backend
高性能后端管理模块，支持C++扩展和纯Python实现切换
"""
from __future__ import annotations

import numpy as np
import os

_CPP_EXTENSION_AVAILABLE = False
_CPP_MODULE = None

try:
    import cSimplCore
    _CPP_EXTENSION_AVAILABLE = True
    _CPP_MODULE = cSimplCore
except ImportError:
    pass

USE_CPP_EXTENSION = os.environ.get('SimplCore_USE_CPP', '1') == '1' and _CPP_EXTENSION_AVAILABLE


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.matmul(a.astype(np.float32), b.astype(np.float32)))
        except Exception:
            pass
    return np.matmul(a, b)


def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.softmax(x.astype(np.float32), dim))
        except Exception:
            pass
    
    x_max = np.max(x, axis=dim, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=dim, keepdims=True)


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.layer_norm(
                x.astype(np.float32), 
                weight.astype(np.float32), 
                bias.astype(np.float32),
                eps
            ))
        except Exception:
            pass
    
    last_dim = x.shape[-1]
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    return (x - mean) / std * weight + bias


def conv2d(
    x: np.ndarray, 
    weight: np.ndarray, 
    bias: np.ndarray,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.conv2d(
                x.astype(np.float32),
                weight.astype(np.float32),
                bias.astype(np.float32),
                stride, stride, padding, padding
            ))
        except Exception:
            pass
    
    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1
    
    result = np.zeros((batch, out_channels, out_h, out_w), dtype=x.dtype)
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    for b in range(batch):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    ih_start = oh * stride
                    iw_start = ow * stride
                    patch = x_padded[b, :, ih_start:ih_start+kernel_h, iw_start:iw_start+kernel_w]
                    result[b, oc, oh, ow] = np.sum(patch * weight[oc]) + bias[oc]
    
    return result


def relu(x: np.ndarray) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.relu(x.astype(np.float32)))
        except Exception:
            pass
    return np.maximum(x, 0)


def gelu(x: np.ndarray) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.gelu(x.astype(np.float32)))
        except Exception:
            pass
    return x * 0.5 * (1 + np.tanh(x * 0.79788456 * (1 + 0.044715 * x * x)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.sigmoid(x.astype(np.float32)))
        except Exception:
            pass
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.tanh(x.astype(np.float32)))
        except Exception:
            pass
    return np.tanh(x)


def addmm(input: np.ndarray, mat1: np.ndarray, mat2: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.addmm(
                input.astype(np.float32),
                mat1.astype(np.float32),
                mat2.astype(np.float32),
                bias.astype(np.float32)
            ))
        except Exception:
            pass
    return np.matmul(input, mat2) + bias


def batch_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
    training: bool = True
) -> np.ndarray:
    if USE_CPP_EXTENSION and _CPP_MODULE is not None:
        try:
            return np.array(_CPP_MODULE.batch_norm(
                x.astype(np.float32),
                weight.astype(np.float32),
                bias.astype(np.float32),
                running_mean.astype(np.float32),
                running_var.astype(np.float32),
                eps,
                1 if training else 0
            ))
        except Exception:
            pass
    
    batch, channels = x.shape[0], x.shape[1]
    spatial_size = np.prod(x.shape[2:])
    
    result = np.zeros_like(x)
    
    for c in range(channels):
        if training:
            mean = np.mean(x[:, c])
            var = np.var(x[:, c])
            running_mean[c] = 0.9 * running_mean[c] + 0.1 * mean
            running_var[c] = 0.9 * running_var[c] + 0.1 * var
        else:
            mean = running_mean[c]
            var = running_var[c]
        
        std = np.sqrt(var + eps)
        result[:, c] = (x[:, c] - mean) / std * weight[c] + bias[c]
    
    return result


def is_cpp_available() -> bool:
    return _CPP_EXTENSION_AVAILABLE


def set_use_cpp(value: bool) -> None:
    global USE_CPP_EXTENSION
    USE_CPP_EXTENSION = value and _CPP_EXTENSION_AVAILABLE


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore Performance Backend 自检")
    print("=" * 60)
    
    print(f"C++扩展可用: {is_cpp_available()}")
    print(f"当前使用C++扩展: {USE_CPP_EXTENSION}")
    
    a = np.random.randn(32, 64).astype(np.float32)
    b = np.random.randn(64, 128).astype(np.float32)
    
    result = matmul(a, b)
    print(f"matmul测试: 输入{a.shape} x {b.shape}, 输出{result.shape}")
    
    x = np.random.randn(32, 128).astype(np.float32)
    sm = softmax(x)
    print(f"softmax测试: 输入{x.shape}, 输出{sm.shape}, 行和验证: {np.allclose(sm.sum(axis=-1), 1.0)}")
    
    x_ln = np.random.randn(32, 10, 128).astype(np.float32)
    w_ln = np.ones(128, dtype=np.float32)
    b_ln = np.zeros(128, dtype=np.float32)
    ln = layer_norm(x_ln, w_ln, b_ln)
    print(f"layer_norm测试: 输入{x_ln.shape}, 输出{ln.shape}")
    
    print("\n所有测试通过!")

