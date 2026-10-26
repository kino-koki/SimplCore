# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore GPU Backend - 青小团自研GPU加速模块
基于CuPy实现GPU张量计算，支持CUDA加速
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Union, List, Tuple, Any
from enum import Enum
import sys

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class GPUDevice(Enum):
    CUDA = "cuda"
    ROCM = "rocm"


class GPUManager:
    """GPU管理器"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._devices = []
            cls._instance._current_device = 0
        return cls._instance
    
    def initialize(self):
        """初始化GPU设备"""
        if self._initialized:
            return
        
        if CUPY_AVAILABLE:
            try:
                self._devices = list(range(cp.cuda.runtime.getDeviceCount()))
                if self._devices:
                    print(f"[SimplCore GPU] Found {len(self._devices)} CUDA device(s)")
                    for i in self._devices:
                        with cp.cuda.Device(i):
                            mem_info = cp.cuda.runtime.memGetInfo()
                            total_mem = mem_info[1] / (1024 ** 3)
                            print(f"[SimplCore GPU] Device {i}: {total_mem:.2f} GB")
                    self._initialized = True
                else:
                    print("[SimplCore GPU] No CUDA devices found")
            except Exception as e:
                print(f"[SimplCore GPU] Failed to initialize CUDA: {e}")
        else:
            print("[SimplCore GPU] CuPy not available, GPU acceleration disabled")
    
    @property
    def available(self) -> bool:
        """是否有可用GPU"""
        return self._initialized and len(self._devices) > 0
    
    @property
    def device_count(self) -> int:
        """GPU设备数量"""
        return len(self._devices) if self._initialized else 0
    
    def set_device(self, device_id: int):
        """设置当前GPU设备"""
        if self.available and device_id in self._devices:
            cp.cuda.Device(device_id).use()
            self._current_device = device_id
    
    @property
    def current_device(self) -> int:
        """当前GPU设备ID"""
        return self._current_device


gpu_manager = GPUManager()


class SimpCUDATensor:
    """GPU张量类"""
    
    def __init__(
        self,
        data: Union[np.ndarray, cp.ndarray],
        requires_grad: bool = False,
        dtype: Optional[Any] = None,
    ):
        if isinstance(data, np.ndarray):
            self._data = cp.array(data)
        elif isinstance(data, cp.ndarray):
            self._data = data.copy()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        if dtype is not None:
            self._data = self._data.astype(dtype)
        
        self._requires_grad = requires_grad
        self._dtype = self._data.dtype
        self._shape = tuple(self._data.shape)
        self._grad: Optional["SimpCUDATensor"] = None
        self._grad_fn: Optional[callable] = None
    
    @property
    def data(self) -> cp.ndarray:
        return self._data
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
    
    @property
    def dtype(self) -> Any:
        return self._dtype
    
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @property
    def grad(self) -> Optional["SimpCUDATensor"]:
        return self._grad
    
    @grad.setter
    def grad(self, value: Optional["SimpCUDATensor"]) -> None:
        self._grad = value
    
    @property
    def grad_fn(self) -> Optional[callable]:
        return self._grad_fn
    
    @grad_fn.setter
    def grad_fn(self, value: Optional[callable]) -> None:
        self._grad_fn = value
    
    @property
    def ndim(self) -> int:
        return len(self._shape)
    
    @property
    def size(self) -> int:
        return self._data.size
    
    def numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return cp.asnumpy(self._data)
    
    def copy(self) -> "SimpCUDATensor":
        """复制张量"""
        return SimpCUDATensor(
            self._data.copy(),
            requires_grad=self._requires_grad,
        )
    
    def reshape(self, *shape: int) -> "SimpCUDATensor":
        """重塑形状"""
        new_data = self._data.reshape(shape)
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            def grad_fn(grad: SimpCUDATensor) -> None:
                self._grad = grad.reshape(self._shape)
            result._grad_fn = grad_fn
        return result
    
    def transpose(self, *axes: int) -> "SimpCUDATensor":
        """转置"""
        if len(axes) == 0:
            axes = tuple(reversed(range(self.ndim)))
        
        axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
        
        if len(axes) == 2:
            a, b = axes
            new_data = self._data.swapaxes(a, b)
        else:
            new_data = self._data.transpose(axes)
        
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            if len(axes) == 2:
                result._grad_fn = lambda grad, a=a, b=b: self.backward(grad.swapaxes(a, b))
            else:
                inv_axes = tuple(np.argsort(axes))
                result._grad_fn = lambda grad, axes=axes: self.backward(grad.transpose(*inv_axes))
        return result
    
    def squeeze(self, dim: Optional[int] = None) -> "SimpCUDATensor":
        """去除维度为1的轴"""
        if dim is None:
            new_data = self._data.squeeze()
        else:
            new_data = self._data.squeeze(dim)
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            def grad_fn(grad: SimpCUDATensor) -> None:
                if dim is None:
                    grad_shape = list(grad.shape)
                    orig_shape = list(self._shape)
                    for i, s in enumerate(orig_shape):
                        if s == 1:
                            grad_shape.insert(i, 1)
                    self._grad = grad.reshape(tuple(grad_shape))
                else:
                    self._grad = grad.unsqueeze(dim)
            result._grad_fn = grad_fn
        return result
    
    def unsqueeze(self, dim: int) -> "SimpCUDATensor":
        """增加维度为1的轴"""
        new_data = cp.expand_dims(self._data, axis=dim)
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            def grad_fn(grad: SimpCUDATensor) -> None:
                self._grad = grad.squeeze(dim)
            result._grad_fn = grad_fn
        return result
    
    def expand(self, *shape: int) -> "SimpCUDATensor":
        """扩展维度"""
        new_data = cp.broadcast_to(self._data, shape)
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            def grad_fn(grad: SimpCUDATensor) -> None:
                self._grad = grad.sum(axis=tuple(i for i, s in enumerate(shape) if s != self._shape[i]))
            result._grad_fn = grad_fn
        return result
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "SimpCUDATensor":
        """求和"""
        if dim is None:
            new_data = cp.array([self._data.sum()])
        else:
            new_data = self._data.sum(axis=dim, keepdims=keepdim)
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            def grad_fn(grad: SimpCUDATensor) -> None:
                if dim is None:
                    grad_data = cp.ones_like(self._data) * grad._data[0]
                else:
                    if keepdim:
                        grad_data = grad._data
                    else:
                        grad_data = cp.expand_dims(grad._data, axis=dim)
                    grad_data = cp.broadcast_to(grad_data, self._shape)
                self.backward(SimpCUDATensor(grad_data))
            result._grad_fn = grad_fn
        return result
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "SimpCUDATensor":
        """求均值"""
        if dim is None:
            new_data = cp.array([self._data.mean()])
        else:
            new_data = self._data.mean(axis=dim, keepdims=keepdim)
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            def grad_fn(grad: SimpCUDATensor) -> None:
                if dim is None:
                    n = self._data.size
                    self._grad = SimpCUDATensor(cp.ones_like(self._data) * grad._data[0] / n)
                else:
                    n = self._data.shape[dim]
                    if keepdim:
                        grad_expanded = grad / n
                    else:
                        grad_expanded = grad.unsqueeze(dim) / n
                    self._grad = grad_expanded.expand(self._shape)
            result._grad_fn = grad_fn
        return result
    
    def matmul(self, other: "SimpCUDATensor") -> "SimpCUDATensor":
        """矩阵乘法"""
        new_data = cp.matmul(self._data, other._data)
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad or other._requires_grad,
        )
        if self._requires_grad or other._requires_grad:
            def grad_fn(grad: SimpCUDATensor) -> None:
                if self._requires_grad:
                    grad_self = cp.matmul(grad._data, other._data.swapaxes(-2, -1))
                    self.backward(SimpCUDATensor(grad_self))
                if other._requires_grad:
                    grad_other = cp.matmul(self._data.swapaxes(-2, -1), grad._data)
                    other.backward(SimpCUDATensor(grad_other))
            result._grad_fn = grad_fn
        return result
    
    def __matmul__(self, other: "SimpCUDATensor") -> "SimpCUDATensor":
        return self.matmul(other)
    
    def __add__(self, other: Union["SimpCUDATensor", float, int]) -> "SimpCUDATensor":
        if isinstance(other, (float, int)):
            new_data = self._data + other
            other_tensor = None
        else:
            new_data = self._data + other._data
            other_tensor = other
        
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
        )
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpCUDATensor) -> None:
                if self._requires_grad:
                    self._grad = grad
                if other_tensor and other_tensor._requires_grad:
                    other_tensor._grad = grad
            result._grad_fn = grad_fn
        return result
    
    def __radd__(self, other: Union["SimpCUDATensor", float, int]) -> "SimpCUDATensor":
        return self.__add__(other)
    
    def __sub__(self, other: Union["SimpCUDATensor", float, int]) -> "SimpCUDATensor":
        if isinstance(other, (float, int)):
            new_data = self._data - other
            other_tensor = None
        else:
            new_data = self._data - other._data
            other_tensor = other
        
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
        )
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpCUDATensor) -> None:
                if self._requires_grad:
                    self._grad = grad
                if other_tensor and other_tensor._requires_grad:
                    other_tensor._grad = grad * -1
            result._grad_fn = grad_fn
        return result
    
    def __mul__(self, other: Union["SimpCUDATensor", float, int]) -> "SimpCUDATensor":
        if isinstance(other, (float, int)):
            new_data = self._data * other
            other_tensor = None
        else:
            new_data = self._data * other._data
            other_tensor = other
        
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
        )
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpCUDATensor) -> None:
                if self._requires_grad:
                    if other_tensor:
                        self._grad = grad * other_tensor
                    else:
                        self._grad = grad * other
                if other_tensor and other_tensor._requires_grad:
                    other_tensor._grad = grad * self
            result._grad_fn = grad_fn
        return result
    
    def __rmul__(self, other: Union["SimpCUDATensor", float, int]) -> "SimpCUDATensor":
        return self.__mul__(other)
    
    def __truediv__(self, other: Union["SimpCUDATensor", float, int]) -> "SimpCUDATensor":
        if isinstance(other, (float, int)):
            new_data = self._data / other
            other_tensor = None
        else:
            new_data = self._data / other._data
            other_tensor = other
        
        result = SimpCUDATensor(
            new_data,
            requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
        )
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpCUDATensor) -> None:
                if self._requires_grad:
                    if other_tensor:
                        self._grad = grad / other_tensor
                    else:
                        self._grad = grad / other
                if other_tensor and other_tensor._requires_grad:
                    if other_tensor:
                        other_tensor._grad = grad * (-self) / (other_tensor * other_tensor)
            result._grad_fn = grad_fn
        return result
    
    def __neg__(self) -> "SimpCUDATensor":
        return self * -1
    
    def __repr__(self) -> str:
        return f"SimpCUDATensor(shape={self._shape}, dtype={self._dtype}, requires_grad={self._requires_grad})"
    
    def backward(self, grad: Optional["SimpCUDATensor"] = None) -> None:
        """反向传播"""
        if not self._requires_grad:
            return
        
        if grad is None:
            if self._shape == (1,):
                grad = SimpCUDATensor(cp.array([1.0]))
            else:
                grad = SimpCUDATensor(cp.ones_like(self._data))
        
        if self._grad is None:
            self._grad = grad
        else:
            self._grad = self._grad + grad
        
        if self._grad_fn:
            self._grad_fn(grad)


def gpu_randn(*shape: int, dtype: Any = np.float32, requires_grad: bool = False) -> SimpCUDATensor:
    """生成GPU正态分布随机张量"""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    data = cp.random.randn(*shape).astype(dtype)
    return SimpCUDATensor(data, requires_grad=requires_grad)


def gpu_ones(*shape: int, dtype: Any = np.float32) -> SimpCUDATensor:
    """生成GPU全1张量"""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    data = cp.ones(shape, dtype=dtype)
    return SimpCUDATensor(data)


def gpu_zeros(*shape: int, dtype: Any = np.float32) -> SimpCUDATensor:
    """生成GPU全0张量"""
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    data = cp.zeros(shape, dtype=dtype)
    return SimpCUDATensor(data)


def gpu_softmax(x: SimpCUDATensor, dim: int = -1) -> SimpCUDATensor:
    """GPU softmax"""
    exp_x = cp.exp(x._data - cp.max(x._data, axis=dim, keepdims=True))
    result = exp_x / cp.sum(exp_x, axis=dim, keepdims=True)
    
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            s = result
            ds = s * (grad._data - cp.sum(grad._data * s, axis=dim, keepdims=True))
            x._grad = SimpCUDATensor(ds)
        out._grad_fn = grad_fn
    
    return out


def gpu_relu(x: SimpCUDATensor) -> SimpCUDATensor:
    """GPU ReLU"""
    result = cp.maximum(0, x._data)
    
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            dx = grad._data * (x._data > 0).astype(cp.float32)
            x._grad = SimpCUDATensor(dx)
        out._grad_fn = grad_fn
    
    return out


def gpu_gelu(x: SimpCUDATensor) -> SimpCUDATensor:
    """GPU GELU"""
    sqrt_2_over_pi = cp.sqrt(2.0 / cp.pi)
    cdf = 0.5 * (1.0 + cp.tanh(sqrt_2_over_pi * (x._data + 0.044715 * x._data ** 3)))
    result = x._data * cdf
    
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            x_data = x._data
            tanh_out = cp.tanh(sqrt_2_over_pi * (x_data + 0.044715 * x_data ** 3))
            dx = cdf + x_data * (1 - tanh_out ** 2) * sqrt_2_over_pi * (1 + 3 * 0.044715 * x_data ** 2) * 0.5
            x._grad = SimpCUDATensor(grad._data * dx)
        out._grad_fn = grad_fn
    
    return out


def gpu_layer_norm(x: SimpCUDATensor, eps: float = 1e-5) -> SimpCUDATensor:
    """GPU层归一化"""
    mean = cp.mean(x._data, axis=-1, keepdims=True)
    var = cp.var(x._data, axis=-1, keepdims=True)
    result = (x._data - mean) / cp.sqrt(var + eps)
    
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            N = x._data.shape[-1]
            x_minus_mean = x._data - mean
            inv_std = 1.0 / cp.sqrt(var + eps)
            
            dx = (grad._data - cp.mean(grad._data, axis=-1, keepdims=True) - 
                  x_minus_mean * cp.mean(grad._data * x_minus_mean, axis=-1, keepdims=True) * inv_std ** 2) * inv_std / N
            x._grad = SimpCUDATensor(dx)
        out._grad_fn = grad_fn
    
    return out


def gpu_cat(tensors: List[SimpCUDATensor], dim: int = 0) -> SimpCUDATensor:
    """GPU拼接"""
    data = [t._data for t in tensors]
    result = cp.concatenate(data, axis=dim)
    
    out = SimpCUDATensor(
        result,
        requires_grad=any(t.requires_grad for t in tensors),
    )
    
    if out.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            sizes = [t.shape[dim] for t in tensors]
            starts = cp.cumsum(cp.array([0] + sizes[:-1]))
            for i, t in enumerate(tensors):
                if t.requires_grad:
                    t._grad = grad._data.take(range(starts[i].item(), starts[i].item() + sizes[i]), axis=dim)
        out._grad_fn = grad_fn
    
    return out


def gpu_stack(tensors: List[SimpCUDATensor], dim: int = 0) -> SimpCUDATensor:
    """GPU堆叠"""
    data = [t._data for t in tensors]
    result = cp.stack(data, axis=dim)
    
    out = SimpCUDATensor(
        result,
        requires_grad=any(t.requires_grad for t in tensors),
    )
    
    if out.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            for i, t in enumerate(tensors):
                if t.requires_grad:
                    t._grad = grad._data.take(i, axis=dim)
        out._grad_fn = grad_fn
    
    return out


def gpu_exp(x: SimpCUDATensor) -> SimpCUDATensor:
    """GPU指数"""
    result = cp.exp(x._data)
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            x._grad = grad * out
        out._grad_fn = grad_fn
    
    return out


def gpu_log(x: SimpCUDATensor) -> SimpCUDATensor:
    """GPU对数"""
    result = cp.log(x._data)
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            x._grad = grad / x
        out._grad_fn = grad_fn
    
    return out


def gpu_tanh(x: SimpCUDATensor) -> SimpCUDATensor:
    """GPU tanh"""
    result = cp.tanh(x._data)
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            x._grad = grad * (1 - out * out)
        out._grad_fn = grad_fn
    
    return out


def gpu_clamp(x: SimpCUDATensor, min: float = None, max: float = None) -> SimpCUDATensor:
    """GPU裁剪"""
    result = cp.clip(x._data, min, max)
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            mask = cp.ones_like(x._data)
            if min is not None:
                mask = mask * (x._data >= min)
            if max is not None:
                mask = mask * (x._data <= max)
            x._grad = grad * SimpCUDATensor(mask)
        out._grad_fn = grad_fn
    
    return out


def gpu_sigmoid(x: SimpCUDATensor) -> SimpCUDATensor:
    """GPU sigmoid"""
    result = 1.0 / (1.0 + cp.exp(-cp.clip(x._data, -500, 500)))
    out = SimpCUDATensor(result, requires_grad=x.requires_grad)
    
    if x.requires_grad:
        def grad_fn(grad: SimpCUDATensor) -> None:
            x._grad = grad * out * (1 - out)
        out._grad_fn = grad_fn
    
    return out


gpu_manager.initialize()


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore GPU 自检")
    print("=" * 60)
    
    if CUPY_AVAILABLE and gpu_manager.available:
        print("GPU available, running GPU tests...")
        
        a = gpu_randn(2, 3, requires_grad=True)
        b = gpu_randn(3, 4, requires_grad=True)
        
        print(f"a shape: {a.shape}")
        print(f"b shape: {b.shape}")
        
        c = a.matmul(b)
        print(f"c = a @ b, shape: {c.shape}")
        
        d = c.sum()
        print(f"d = sum(c), shape: {d.shape}")
        
        d.backward()
        
        print(f"a has grad: {a.grad is not None}, grad shape: {a.grad.shape if a.grad else None}")
        print(f"b has grad: {b.grad is not None}, grad shape: {b.grad.shape if b.grad else None}")
        
        e = gpu_gelu(a)
        print(f"e = gelu(a), shape: {e.shape}")
        
        print("\nSimplCore GPU 自检 PASSED")
    else:
        print("GPU not available, skipping GPU tests")
        print("\nSimplCore GPU 自检 SKIPPED")

