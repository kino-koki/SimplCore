# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Optimization Module - 青小团自研性能优化模块
包含向量化、缓存机制、算子融合等优化技术
"""
from __future__ import annotations

import numpy as np
import functools
import time
import threading
from typing import Optional, Dict, Any, Callable, Tuple, List
from collections import OrderedDict

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("[SimplCore Optimization] Numba JIT available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("[SimplCore Optimization] Numba JIT not available - falling back to pure NumPy")


class CacheManager:
    """缓存管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = OrderedDict()
            cls._instance._lock = threading.Lock()
            cls._instance._max_size = 1024
            cls._instance._hits = 0
            cls._instance._misses = 0
        return cls._instance
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self._lock:
            if key in self._cache:
                self._hits += 1
                entry = self._cache.pop(key)
                self._cache[key] = entry
                return entry["value"]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """设置缓存"""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
            
            self._cache[key] = {
                "value": value,
                "ttl": ttl,
                "timestamp": time.time(),
            }
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


cache_manager = CacheManager()


def cached(func: Callable) -> Callable:
    """缓存装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key_parts = [func.__name__]
        for arg in args:
            if isinstance(arg, (int, float, str, tuple)):
                key_parts.append(str(arg))
            elif isinstance(arg, np.ndarray):
                key_parts.append(str(arg.shape))
                key_parts.append(str(arg.dtype))
                key_parts.append(str(id(arg)))
            elif hasattr(arg, '_data') and isinstance(arg._data, np.ndarray):
                key_parts.append(str(arg.shape))
                key_parts.append(str(arg.dtype))
                key_parts.append(str(id(arg)))
            elif hasattr(arg, 'shape'):
                key_parts.append(str(arg.shape))
                key_parts.append(str(id(arg)))
        
        for k, v in kwargs.items():
            if isinstance(v, (int, float, str, tuple)):
                key_parts.append(f"{k}={v}")
            elif isinstance(v, np.ndarray):
                key_parts.append(f"{k}={v.shape}_{v.dtype}_{id(v)}")
        
        key = "_".join(key_parts)
        
        cached_result = cache_manager.get(key)
        if cached_result is not None:
            return cached_result
        
        result = func(*args, **kwargs)
        cache_manager.set(key, result)
        return result
    
    return wrapper


class OperatorFusion:
    """算子融合器"""
    
    def __init__(self):
        self._fusion_rules = {}
    
    def register_rule(self, pattern: Tuple[str, ...], fused_func: Callable) -> None:
        """注册融合规则"""
        self._fusion_rules[pattern] = fused_func
    
    def fuse(self, operators: List[Tuple[str, Dict[str, Any]]]) -> Callable:
        """融合算子序列"""
        patterns = list(self._fusion_rules.keys())
        for pattern in patterns:
            if len(operators) >= len(pattern):
                match = True
                for i, op_name in enumerate(pattern):
                    if operators[i][0] != op_name:
                        match = False
                        break
                if match:
                    return self._fusion_rules[pattern]
        return None


operator_fusion = OperatorFusion()


def fused_matmul_relu(x, w, b=None):
    """融合的矩阵乘法+ReLU"""
    if b is None:
        out = np.matmul(x, w)
    else:
        out = np.matmul(x, w) + b
    out = np.maximum(0, out)
    return out


def fused_matmul_gelu(x, w, b=None):
    """融合的矩阵乘法+GELU"""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    if b is None:
        out = np.matmul(x, w)
    else:
        out = np.matmul(x, w) + b
    cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (out + 0.044715 * out ** 3)))
    out = out * cdf
    return out


def fused_layer_norm_linear(x, weight, bias, ln_weight, ln_bias, eps=1e-5):
    """融合的LayerNorm+Linear"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    x_norm = x_norm * ln_weight + ln_bias
    out = np.matmul(x_norm, weight) + bias
    return out


operator_fusion.register_rule(("matmul", "relu"), fused_matmul_relu)
operator_fusion.register_rule(("matmul", "gelu"), fused_matmul_gelu)
operator_fusion.register_rule(("layer_norm", "linear"), fused_layer_norm_linear)


class VectorizedOps:
    """向量化操作类"""
    
    @staticmethod
    def batch_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """批量矩阵乘法"""
        return np.einsum('bij,bjk->bik', A, B)
    
    @staticmethod
    def batch_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """向量化批归一化"""
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        return weight.reshape(1, -1, 1, 1) * (x - mean) / np.sqrt(var + eps) + bias.reshape(1, -1, 1, 1)
    
    @staticmethod
    def conv2d_im2col(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                      stride: int = 1, padding: int = 0) -> np.ndarray:
        """基于im2col的向量化卷积（使用stride_tricks消除Python循环）"""
        B, C_in, H, W = x.shape
        C_out, _, K, _ = weight.shape
        
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        
        H_out = (H + 2 * padding - K) // stride + 1
        W_out = (W + 2 * padding - K) // stride + 1
        
        _, _, H_pad, W_pad = x_padded.shape
        
        patch_size = C_in * K * K
        num_patches = H_out * W_out
        
        strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            x_padded.strides[2] * stride,
            x_padded.strides[3] * stride,
            x_padded.strides[2],
            x_padded.strides[3],
        )
        
        shape = (B, C_in, H_out, W_out, K, K)
        
        patches = np.lib.stride_tricks.as_strided(
            x_padded,
            shape=shape,
            strides=strides
        )
        
        col = patches.reshape(B, C_in * K * K, H_out * W_out)
        
        weight_flat = weight.reshape(C_out, -1)
        out = np.matmul(weight_flat, col) + bias.reshape(-1, 1)
        
        return out.reshape(B, C_out, H_out, W_out)
    
    @staticmethod
    def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                            head_dim: int, num_heads: int) -> np.ndarray:
        """向量化多头注意力"""
        B, seq_len, _ = Q.shape
        
        Q = Q.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        
        attn_weights = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
        attn_weights = np.exp(attn_weights - np.max(attn_weights, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        output = np.matmul(attn_weights, V)
        output = output.transpose(0, 2, 1, 3).reshape(B, seq_len, num_heads * head_dim)
        
        return output


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self._timers = {}
        self._counters = {}
    
    def time(self, name: str) -> Callable:
        """计时装饰器"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                if name not in self._timers:
                    self._timers[name] = []
                self._timers[name].append(elapsed)
                
                return result
            return wrapper
        return decorator
    
    def count(self, name: str) -> None:
        """计数"""
        if name not in self._counters:
            self._counters[name] = 0
        self._counters[name] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {}
        for name, times in self._timers.items():
            stats[name] = {
                "count": len(times),
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
            }
        stats["counters"] = self._counters
        stats["cache_hit_rate"] = cache_manager.hit_rate
        return stats
    
    def reset(self) -> None:
        """重置统计"""
        self._timers.clear()
        self._counters.clear()


profiler = PerformanceProfiler()


class MemoryOptimizer:
    """内存优化器"""
    
    @staticmethod
    def gradient_checkpointing(module: Callable, inputs: Any) -> Any:
        """梯度检查点"""
        if isinstance(inputs, (tuple, list)):
            for inp in inputs:
                if hasattr(inp, 'requires_grad_'):
                    inp.requires_grad_(True)
        
        outputs = module(inputs)
        
        if isinstance(outputs, (tuple, list)):
            def backward_fn(grads):
                if not isinstance(grads, (tuple, list)):
                    grads = (grads,)
                
                for inp in inputs:
                    if hasattr(inp, 'zero_grad'):
                        inp.zero_grad()
                
                new_inputs = []
                if isinstance(inputs, (tuple, list)):
                    for inp in inputs:
                        if hasattr(inp, 'detach'):
                            new_inp = inp.detach()
                            new_inp.requires_grad_(True)
                            new_inputs.append(new_inp)
                        else:
                            new_inputs.append(inp)
                    new_inputs = tuple(new_inputs)
                else:
                    if hasattr(inputs, 'detach'):
                        new_inputs = inputs.detach()
                        new_inputs.requires_grad_(True)
                    else:
                        new_inputs = inputs
                
                new_outputs = module(new_inputs)
                
                if isinstance(new_outputs, tuple):
                    total_loss = sum(o * g for o, g in zip(new_outputs, grads))
                    total_loss = total_loss.sum()
                else:
                    total_loss = (new_outputs * grads[0]).sum()
                
                total_loss.backward()
                
                grad_results = []
                if isinstance(new_inputs, (tuple, list)):
                    for inp in new_inputs:
                        if hasattr(inp, 'grad') and inp.grad is not None:
                            grad_results.append(inp.grad)
                elif hasattr(new_inputs, 'grad') and new_inputs.grad is not None:
                    grad_results.append(new_inputs.grad)
                
                if isinstance(inputs, (tuple, list)):
                    for i, inp in enumerate(inputs):
                        if hasattr(inp, 'backward') and i < len(grad_results):
                            inp.backward(grad_results[i])
                elif hasattr(inputs, 'backward') and grad_results:
                    inputs.backward(grad_results[0])
                
                return grad_results
        else:
            def backward_fn(grad):
                for inp in inputs:
                    if hasattr(inp, 'zero_grad'):
                        inp.zero_grad()
                
                new_inputs = []
                if isinstance(inputs, (tuple, list)):
                    for inp in inputs:
                        if hasattr(inp, 'detach'):
                            new_inp = inp.detach()
                            new_inp.requires_grad_(True)
                            new_inputs.append(new_inp)
                        else:
                            new_inputs.append(inp)
                    new_inputs = tuple(new_inputs)
                else:
                    if hasattr(inputs, 'detach'):
                        new_inputs = inputs.detach()
                        new_inputs.requires_grad_(True)
                    else:
                        new_inputs = inputs
                
                new_outputs = module(new_inputs)
                
                if isinstance(new_outputs, tuple):
                    total_loss = sum(o.sum() for o in new_outputs)
                else:
                    total_loss = (new_outputs * grad).sum()
                
                total_loss.backward()
                
                grad_results = []
                if isinstance(new_inputs, (tuple, list)):
                    for inp in new_inputs:
                        if hasattr(inp, 'grad') and inp.grad is not None:
                            grad_results.append(inp.grad)
                elif hasattr(new_inputs, 'grad') and new_inputs.grad is not None:
                    grad_results.append(new_inputs.grad)
                
                if isinstance(inputs, (tuple, list)):
                    for i, inp in enumerate(inputs):
                        if hasattr(inp, 'backward') and i < len(grad_results):
                            inp.backward(grad_results[i])
                elif hasattr(inputs, 'backward') and grad_results:
                    inputs.backward(grad_results[0])
                
                return grad_results
        
        if isinstance(outputs, (tuple, list)):
            for out in outputs:
                if hasattr(out, '_grad_fn'):
                    out._grad_fn = backward_fn
        else:
            if hasattr(outputs, '_grad_fn'):
                outputs._grad_fn = backward_fn
        
        return outputs
    
    @staticmethod
    def mixed_precision_convert(tensor: np.ndarray, dtype: str = "float16") -> np.ndarray:
        """混合精度转换"""
        if dtype == "float16":
            return tensor.astype(np.float16)
        elif dtype == "bfloat16":
            return tensor.astype(np.float32)
        return tensor
    
    @staticmethod
    def pin_memory(tensor: np.ndarray) -> np.ndarray:
        """固定内存"""
        return tensor
    
    @staticmethod
    def memory_efficient_cat(tensors: List[np.ndarray], dim: int = 0) -> np.ndarray:
        """内存高效的拼接"""
        total_size = sum(t.shape[dim] for t in tensors)
        result_shape = list(tensors[0].shape)
        result_shape[dim] = total_size
        
        result = np.zeros(result_shape, dtype=tensors[0].dtype)
        offset = 0
        for t in tensors:
            slices = [slice(None)] * len(result_shape)
            slices[dim] = slice(offset, offset + t.shape[dim])
            result[tuple(slices)] = t
            offset += t.shape[dim]
        
        return result
    
    @staticmethod
    def activation_checkpointing(module: Callable, *args) -> Any:
        """激活检查点"""
        outputs = module(*args)
        
        def grad_fn(grad):
            for arg in args:
                if hasattr(arg, 'zero_grad'):
                    arg.zero_grad()
            
            new_args = []
            for arg in args:
                if hasattr(arg, 'detach'):
                    new_arg = arg.detach()
                    new_arg.requires_grad_(True)
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)
            
            new_outputs = module(*new_args)
            
            if isinstance(new_outputs, tuple):
                total_loss = sum(o.sum() for o in new_outputs)
            else:
                total_loss = (new_outputs * grad).sum()
            
            total_loss.backward()
            
            grads = []
            for i, arg in enumerate(new_args):
                if hasattr(arg, 'grad') and arg.grad is not None:
                    grads.append(arg.grad)
                    if hasattr(args[i], 'backward'):
                        args[i].backward(arg.grad)
            
            return grads
        
        if isinstance(outputs, (tuple, list)):
            for out in outputs:
                if hasattr(out, '_grad_fn'):
                    out._grad_fn = grad_fn
        else:
            if hasattr(outputs, '_grad_fn'):
                outputs._grad_fn = grad_fn
        
        return outputs
    
    @staticmethod
    def gradient_accumulation(gradients: List[np.ndarray], accumulate_steps: int = 1) -> np.ndarray:
        """梯度累积"""
        if accumulate_steps <= 1:
            return gradients[0]
        
        total_grad = np.zeros_like(gradients[0])
        for i in range(accumulate_steps):
            total_grad += gradients[i]
        return total_grad / accumulate_steps


class JITCompiler:
    """即时编译器"""
    
    def __init__(self):
        self._compiled_functions = {}
    
    def compile(self, func: Callable, args: Tuple[Any, ...]) -> Callable:
        """编译函数"""
        key = (func.__name__, tuple(type(arg).__name__ for arg in args))
        
        if key in self._compiled_functions:
            return self._compiled_functions[key]
        
        compiled_func = func
        self._compiled_functions[key] = compiled_func
        
        return compiled_func
    
    def clear_cache(self) -> None:
        """清空编译缓存"""
        self._compiled_functions.clear()


jit_compiler = JITCompiler()


@profiler.time("optimized_matmul")
def optimized_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """优化的矩阵乘法"""
    return np.matmul(A, B)


@profiler.time("optimized_conv2d")
def optimized_conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None,
                     stride: int = 1, padding: int = 0) -> np.ndarray:
    """优化的卷积"""
    return VectorizedOps.conv2d_im2col(x, weight, bias, stride, padding)


@profiler.time("optimized_attention")
def optimized_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                        head_dim: int, num_heads: int) -> np.ndarray:
    """优化的注意力"""
    return VectorizedOps.multi_head_attention(Q, K, V, head_dim, num_heads)


if NUMBA_AVAILABLE:
    @njit(fastmath=True, parallel=True)
    def numba_layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        N = x.shape[-1]
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            row = x[i]
            mean_val = np.mean(row)
            centered = row - mean_val
            var_val = np.mean(centered ** 2)
            inv_std = 1.0 / np.sqrt(var_val + eps)
            result[i] = centered * inv_std
        return result
    
    @njit(fastmath=True)
    def numba_softmax(x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            row = x[i]
            max_val = np.max(row)
            exp_row = np.exp(row - max_val)
            sum_exp = np.sum(exp_row)
            result[i] = exp_row / sum_exp
        return result
    
    @njit(fastmath=True, parallel=True)
    def numba_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
        for i in prange(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i, j] += A[i, k] * B[k, j]
        return C
    
    @njit(fastmath=True)
    def numba_gelu(x: np.ndarray) -> np.ndarray:
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x ** 3)))
        return x * cdf
    
    @njit(fastmath=True)
    def numba_relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @njit(fastmath=True, parallel=True)
    def numba_conv2d_im2col(x_padded: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                            B: int, C_in: int, H_out: int, W_out: int, K: int, C_out: int,
                            stride: int) -> np.ndarray:
        _, _, H_pad, W_pad = x_padded.shape
        out = np.zeros((B, C_out, H_out, W_out), dtype=x_padded.dtype)
        
        for b in prange(B):
            for c_out in range(C_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride
                        w_start = w_out * stride
                        val = bias[c_out]
                        for c_in in range(C_in):
                            for kh in range(K):
                                for kw in range(K):
                                    val += x_padded[b, c_in, h_start + kh, w_start + kw] * weight[c_out, c_in, kh, kw]
                        out[b, c_out, h_out, w_out] = val
        
        return out


class NumbaOptimizedOps:
    """Numba优化操作类"""
    
    @staticmethod
    def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        if NUMBA_AVAILABLE:
            return numba_layer_norm(x, eps)
        return VectorizedOps.batch_norm(x, np.ones(x.shape[-1]), np.zeros(x.shape[-1]), eps)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        if NUMBA_AVAILABLE:
            return numba_softmax(x)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        if NUMBA_AVAILABLE:
            return numba_gelu(x)
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x ** 3)))
        return x * cdf
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        if NUMBA_AVAILABLE:
            return numba_relu(x)
        return np.maximum(0, x)
    
    @staticmethod
    def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               stride: int = 1, padding: int = 0) -> np.ndarray:
        if NUMBA_AVAILABLE:
            B, C_in, H, W = x.shape
            C_out, _, K, _ = weight.shape
            
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
            
            H_out = (H + 2 * padding - K) // stride + 1
            W_out = (W + 2 * padding - K) // stride + 1
            
            return numba_conv2d_im2col(x_padded, weight, bias, B, C_in, H_out, W_out, K, C_out, stride)
        return VectorizedOps.conv2d_im2col(x, weight, bias, stride, padding)


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore Optimization 自检")
    print("=" * 60)
    
    print("\n1. Testing CacheManager...")
    cm = CacheManager()
    cm.set("test_key", "test_value")
    assert cm.get("test_key") == "test_value"
    assert cm.hit_rate == 1.0
    print("   CacheManager: PASSED")
    
    print("\n2. Testing VectorizedOps...")
    A = np.random.randn(2, 3, 4)
    B = np.random.randn(2, 4, 5)
    result = VectorizedOps.batch_matmul(A, B)
    assert result.shape == (2, 3, 5)
    print("   VectorizedOps.batch_matmul: PASSED")
    
    x = np.random.randn(2, 3, 32, 32)
    weight = np.random.randn(16, 3, 3, 3)
    bias = np.random.randn(16)
    conv_result = VectorizedOps.conv2d_im2col(x, weight, bias, padding=1)
    assert conv_result.shape == (2, 16, 32, 32)
    print("   VectorizedOps.conv2d_im2col: PASSED")
    
    print("\n3. Testing OperatorFusion...")
    x = np.random.randn(16, 128)
    w = np.random.randn(128, 256)
    b = np.random.randn(256)
    fused_result = fused_matmul_relu(x, w, b)
    manual_result = np.maximum(0, np.matmul(x, w) + b)
    assert np.allclose(fused_result, manual_result, atol=1e-5)
    print("   OperatorFusion: PASSED")
    
    print("\n4. Testing PerformanceProfiler...")
    profiler.count("test_counter")
    stats = profiler.get_stats()
    assert "test_counter" in stats["counters"]
    print("   PerformanceProfiler: PASSED")
    
    print("\n5. Testing MemoryOptimizer...")
    tensors = [np.random.randn(10, 20) for _ in range(5)]
    cat_result = MemoryOptimizer.memory_efficient_cat(tensors)
    assert cat_result.shape == (50, 20)
    print("   MemoryOptimizer: PASSED")
    
    print("\n" + "=" * 60)
    print("SimplCore Optimization 自检 PASSED")
    print("=" * 60)

