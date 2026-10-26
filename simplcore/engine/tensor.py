# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor Module - Optimized Version
高性能张量库优化版
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Callable
from enum import Enum

from .perf_backend import matmul, softmax as perf_softmax, layer_norm, relu, gelu, sigmoid, tanh

np.set_printoptions(precision=4, suppress=True)


class Dtype(Enum):
    FLOAT32 = np.float32
    FLOAT64 = np.float64
    INT32 = np.int32
    INT64 = np.int64
    BOOL = np.bool_


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "cuda"


class SimpTensor:
    """高性能张量类"""

    def __init__(self, data: Union[np.ndarray, float, int], 
                 requires_grad: bool = False,
                 device: DeviceType = DeviceType.CPU,
                 dtype: Dtype = Dtype.FLOAT32):
        if isinstance(data, (float, int)):
            self._data = np.array([data]).astype(dtype.value)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(dtype.value)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        self._shape = tuple(self._data.shape)
        self._dtype = dtype
        self._device = device
        self._requires_grad = requires_grad
        self._grad: Optional[SimpTensor] = None
        self._grad_fn: Optional[Callable] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> Dtype:
        return self._dtype

    @property
    def device(self) -> DeviceType:
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    @property
    def grad(self) -> Optional["SimpTensor"]:
        return self._grad
    
    @grad.setter
    def grad(self, value: Optional["SimpTensor"]) -> None:
        self._grad = value

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def real(self) -> "SimpTensor":
        return SimpTensor(np.real(self._data), device=self._device, dtype=self._dtype, requires_grad=self._requires_grad)

    def item(self) -> float:
        return float(self._data.item())

    def reshape(self, *shape: int) -> "SimpTensor":
        new_shape = shape
        if len(new_shape) == 1 and isinstance(new_shape[0], tuple):
            new_shape = new_shape[0]
        new_data = self._data.reshape(new_shape)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                self.backward(grad.reshape(self._shape))
            result._grad_fn = grad_fn
        return result

    def transpose(self, *dims: int) -> "SimpTensor":
        if len(dims) == 0:
            dims = tuple(reversed(range(self.ndim)))
        dims = tuple(d if d >= 0 else d + self.ndim for d in dims)
        new_data = np.swapaxes(self._data, *dims) if len(dims) == 2 else self._data.transpose(*dims)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            forward_perm = list(range(self.ndim))
            for i, d in enumerate(dims):
                forward_perm[d] = i
            reverse_dims = tuple(forward_perm)
            def grad_fn(grad: SimpTensor) -> None:
                self.backward(grad.transpose(*reverse_dims))
            result._grad_fn = grad_fn
        return result
    
    @property
    def T(self) -> "SimpTensor":
        return self.transpose(*reversed(range(self.ndim)))

    def unsqueeze(self, dim: int) -> "SimpTensor":
        new_data = np.expand_dims(self._data, axis=dim)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                self.backward(grad.squeeze(dim))
            result._grad_fn = grad_fn
        return result

    def squeeze(self, dim: int = None) -> "SimpTensor":
        new_data = np.squeeze(self._data, axis=dim)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                if dim is None:
                    grad_data = grad._data
                    for i in range(self.ndim):
                        if self._shape[i] == 1:
                            grad_data = np.expand_dims(grad_data, axis=i)
                else:
                    grad_data = np.expand_dims(grad._data, axis=dim)
                self.backward(SimpTensor(grad_data))
            result._grad_fn = grad_fn
        return result

    def expand(self, *shape: int) -> "SimpTensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        shape_list = list(shape)
        for i, s in enumerate(shape_list):
            if s == -1:
                shape_list[i] = self._shape[i]
        new_data = np.broadcast_to(self._data, tuple(shape_list))
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                axes = tuple(i for i, s in enumerate(shape_list) if s != self._shape[i])
                self.backward(grad.sum(dim=axes))
            result._grad_fn = grad_fn
        return result

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "SimpTensor":
        if dim is None:
            new_data = np.array([self._data.sum()])
        else:
            new_data = self._data.sum(axis=dim, keepdims=keepdim)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                if dim is None:
                    grad_data = np.ones_like(self._data) * grad._data[0]
                else:
                    grad_expanded = np.expand_dims(grad._data, axis=dim) if not keepdim else grad._data
                    grad_data = np.broadcast_to(grad_expanded, self._data.shape)
                self.backward(SimpTensor(grad_data))
            result._grad_fn = grad_fn
        return result
    
    def max(self, dim: Optional[int] = None, keepdim: bool = False) -> "SimpTensor":
        if dim is None:
            new_data = np.array([self._data.max()])
        else:
            new_data = self._data.max(axis=dim, keepdims=keepdim)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                grad_data = np.zeros_like(self._data)
                mask = (self._data == new_data)
                if dim is None:
                    grad_data[mask] = grad._data[0]
                else:
                    grad_expanded = grad._data if keepdim else np.expand_dims(grad._data, axis=dim)
                    grad_data[mask] = grad_expanded[mask]
                self.backward(SimpTensor(grad_data))
            result._grad_fn = grad_fn
        return result

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "SimpTensor":
        if dim is None:
            new_data = np.array([self._data.mean()])
        else:
            new_data = self._data.mean(axis=dim, keepdims=keepdim)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                if dim is None:
                    n = self._data.size
                    grad_data = np.ones_like(self._data) * grad._data[0] / n
                else:
                    n = self._data.shape[dim]
                    grad_expanded = grad._data / n
                    if not keepdim:
                        grad_expanded = np.expand_dims(grad_expanded, axis=dim)
                    grad_data = np.broadcast_to(grad_expanded, self._data.shape)
                self.backward(SimpTensor(grad_data))
            result._grad_fn = grad_fn
        return result

    def matmul(self, other: "SimpTensor") -> "SimpTensor":
        new_data = matmul(self._data, other._data)
        result = SimpTensor(new_data, requires_grad=self._requires_grad or other._requires_grad, 
                           device=self._device, dtype=self._dtype)
        if self._requires_grad or other._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                if self._requires_grad:
                    grad_a = np.matmul(grad._data, other._data.transpose())
                    self.backward(SimpTensor(grad_a))
                if other._requires_grad:
                    grad_b = np.matmul(self._data.transpose(), grad._data)
                    other.backward(SimpTensor(grad_b))
            result._grad_fn = grad_fn
        return result

    def __add__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        if isinstance(other, (float, int)):
            new_data = self._data + other
            other_tensor = None
        else:
            new_data = self._data + other._data
            other_tensor = other
        result = SimpTensor(new_data, requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
                           device=self._device, dtype=self._dtype)
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpTensor) -> None:
                if self._requires_grad:
                    self.backward(grad)
                if other_tensor and other_tensor._requires_grad:
                    other_tensor.backward(grad)
            result._grad_fn = grad_fn
        return result

    def __radd__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        return self.__add__(other)

    def __sub__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        if isinstance(other, (float, int)):
            new_data = self._data - other
            other_tensor = None
        else:
            new_data = self._data - other._data
            other_tensor = other
        result = SimpTensor(new_data, requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
                           device=self._device, dtype=self._dtype)
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpTensor) -> None:
                if self._requires_grad:
                    self.backward(grad)
                if other_tensor and other_tensor._requires_grad:
                    other_tensor.backward(-grad)
            result._grad_fn = grad_fn
        return result

    def __rsub__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        if isinstance(other, (float, int)):
            new_data = other - self._data
        else:
            new_data = other._data - self._data
        result = SimpTensor(new_data, requires_grad=self._requires_grad or (isinstance(other, SimpTensor) and other._requires_grad),
                           device=self._device, dtype=self._dtype)
        if self._requires_grad or (isinstance(other, SimpTensor) and other._requires_grad):
            def grad_fn(grad: SimpTensor) -> None:
                self.backward(-grad)
                if isinstance(other, SimpTensor) and other._requires_grad:
                    other.backward(grad)
            result._grad_fn = grad_fn
        return result

    def __mul__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        if isinstance(other, (float, int)):
            new_data = self._data * other
            other_tensor = None
        else:
            new_data = self._data * other._data
            other_tensor = other
        result = SimpTensor(new_data, requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
                           device=self._device, dtype=self._dtype)
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpTensor) -> None:
                if self._requires_grad:
                    if other_tensor:
                        self.backward(grad * other_tensor)
                    else:
                        self.backward(SimpTensor(grad._data * other))
                if other_tensor and other_tensor._requires_grad:
                    other_tensor.backward(grad * self)
            result._grad_fn = grad_fn
        return result

    def __rmul__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        if isinstance(other, (float, int)):
            new_data = self._data / other
            other_tensor = None
        else:
            new_data = self._data / other._data
            other_tensor = other
        result = SimpTensor(new_data, requires_grad=self._requires_grad or (other_tensor and other_tensor._requires_grad),
                           device=self._device, dtype=self._dtype)
        if self._requires_grad or (other_tensor and other_tensor._requires_grad):
            def grad_fn(grad: SimpTensor) -> None:
                if self._requires_grad:
                    if other_tensor:
                        self.backward(grad / other_tensor)
                    else:
                        self.backward(SimpTensor(grad._data / other))
                if other_tensor and other_tensor._requires_grad:
                    other_tensor.backward(grad * (-self) / (other_tensor * other_tensor))
            result._grad_fn = grad_fn
        return result

    def __rtruediv__(self, other: Union["SimpTensor", float, int]) -> "SimpTensor":
        if isinstance(other, (float, int)):
            new_data = other / self._data
        else:
            new_data = other._data / self._data
        result = SimpTensor(new_data, requires_grad=self._requires_grad or (isinstance(other, SimpTensor) and other._requires_grad),
                           device=self._device, dtype=self._dtype)
        if self._requires_grad or (isinstance(other, SimpTensor) and other._requires_grad):
            def grad_fn(grad: SimpTensor) -> None:
                self.backward(-grad * other / (self * self))
                if isinstance(other, SimpTensor) and other._requires_grad:
                    other.backward(grad / self)
            result._grad_fn = grad_fn
        return result

    def __pow__(self, power: float) -> "SimpTensor":
        new_data = np.power(self._data, power)
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                grad_data = power * np.power(self._data, power - 1) * grad._data
                self.backward(SimpTensor(grad_data))
            result._grad_fn = grad_fn
        return result

    def __neg__(self) -> "SimpTensor":
        return self * -1

    def __getitem__(self, idx) -> "SimpTensor":
        new_data = self._data[idx]
        result = SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
        if self._requires_grad:
            def grad_fn(grad: SimpTensor) -> None:
                grad_data = np.zeros_like(self._data)
                grad_data[idx] = grad._data
                self.backward(SimpTensor(grad_data))
            result._grad_fn = grad_fn
        return result

    def detach(self) -> "SimpTensor":
        return SimpTensor(self._data.copy(), requires_grad=False, device=self._device, dtype=self._dtype)

    def requires_grad_(self, requires_grad: bool = True) -> "SimpTensor":
        self._requires_grad = requires_grad
        return self

    def zero_grad(self) -> None:
        self._grad = None

    def contiguous(self) -> "SimpTensor":
        if self._data.flags['C_CONTIGUOUS']:
            return self
        return SimpTensor(np.ascontiguousarray(self._data), requires_grad=self._requires_grad, 
                         device=self._device, dtype=self._dtype)

    def repeat(self, *repeats: int) -> "SimpTensor":
        new_data = np.repeat(self._data, repeats)
        return SimpTensor(new_data, requires_grad=self._requires_grad, device=self._device, dtype=self._dtype)
    
    def softmax(self, dim: int = -1) -> "SimpTensor":
        return softmax(self, dim)

    def __repr__(self) -> str:
        return f"SimpTensor(shape={self._shape}, dtype={self._dtype.name}, device={self._device.value}, requires_grad={self._requires_grad})"

    def backward(self, grad: Optional["SimpTensor"] = None) -> None:
        if not self._requires_grad:
            return
        if grad is None:
            if self._shape == (1,):
                grad = SimpTensor(np.array([1.0]))
            else:
                grad = SimpTensor(np.ones_like(self._data))
        if self._grad is None:
            self._grad = grad
        else:
            self._grad = self._grad + grad
        if self._grad_fn:
            self._grad_fn(grad)


def randn(*shape: int, device: DeviceType = DeviceType.CPU, 
          dtype: Dtype = Dtype.FLOAT32, requires_grad: bool = False) -> SimpTensor:
    data = np.random.randn(*shape).astype(dtype.value)
    return SimpTensor(data, device=device, dtype=dtype, requires_grad=requires_grad)


def rand(*shape: int, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    data = np.random.rand(*shape).astype(dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def ones(*shape: int, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    data = np.ones(shape, dtype=dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def zeros(*shape: int, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    data = np.zeros(shape, dtype=dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def eye(n: int, m: Optional[int] = None, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    m = m or n
    data = np.eye(n, m, dtype=dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def arange(start: int, stop: int = None, step: int = 1, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    if stop is None:
        start, stop = 0, start
    data = np.arange(start, stop, step, dtype=dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def softmax(x: SimpTensor, dim: int = -1) -> SimpTensor:
    exp_x = np.exp(x._data - np.max(x._data, axis=dim, keepdims=True))
    result = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            s = result
            ds = s * (grad._data - np.sum(grad._data * s, axis=dim, keepdims=True))
            x.backward(SimpTensor(ds))
        out._grad_fn = grad_fn
    return out


def layer_norm(x: SimpTensor, eps: float = 1e-5) -> SimpTensor:
    x_data = x._data
    last_dim = x_data.shape[-1]
    
    if x_data.ndim == 2:
        mean = np.mean(x_data, axis=1, keepdims=True)
        centered = x_data - mean
        var = np.mean(centered ** 2, axis=1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        result = centered * inv_std
    else:
        reshaped = x_data.reshape(-1, last_dim)
        mean = np.mean(reshaped, axis=1, keepdims=True)
        centered = reshaped - mean
        var = np.mean(centered ** 2, axis=1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        result = centered * inv_std
        result = result.reshape(x_data.shape)
        mean = mean.reshape(x_data.shape[:-1] + (1,))
        centered = centered.reshape(x_data.shape)
        inv_std = inv_std.reshape(x_data.shape[:-1] + (1,))
    
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_data = grad._data
            N = last_dim
            
            if grad_data.ndim == 2:
                grad_sum = np.sum(grad_data, axis=1, keepdims=True)
                grad_x_mu_sum = np.sum(grad_data * centered, axis=1, keepdims=True)
                dx = (grad_data * N - grad_sum - centered * grad_x_mu_sum * inv_std ** 2) * inv_std / N
            else:
                grad_reshaped = grad_data.reshape(-1, last_dim)
                grad_sum = np.sum(grad_reshaped, axis=1, keepdims=True)
                grad_x_mu_sum = np.sum(grad_reshaped * centered.reshape(-1, last_dim), axis=1, keepdims=True)
                dx = (grad_reshaped * N - grad_sum - centered.reshape(-1, last_dim) * grad_x_mu_sum * inv_std.reshape(-1, 1) ** 2) * inv_std.reshape(-1, 1) / N
                dx = dx.reshape(grad_data.shape)
            
            x.backward(SimpTensor(dx))
        out._grad_fn = grad_fn
    
    return out


def gelu(x: SimpTensor) -> SimpTensor:
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    result = 0.5 * x._data * (1.0 + np.tanh(sqrt_2_over_pi * (x._data + 0.044715 * x._data ** 3)))
    
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            tanh_out = np.tanh(sqrt_2_over_pi * (x._data + 0.044715 * x._data ** 3))
            cdf = 0.5 * (1.0 + tanh_out)
            dx = cdf + x._data * (1.0 - tanh_out ** 2) * sqrt_2_over_pi * (1.0 + 0.134145 * x._data ** 2)
            x.backward(SimpTensor(grad._data * dx))
        out._grad_fn = grad_fn
    
    return out


def silu(x: SimpTensor) -> SimpTensor:
    result = x._data / (1.0 + np.exp(-x._data))
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            exp_neg = np.exp(-x._data)
            dx = (1.0 + exp_neg + x._data * exp_neg) / (1.0 + exp_neg) ** 2
            x.backward(SimpTensor(grad._data * dx))
        out._grad_fn = grad_fn
    return out


def relu(x: SimpTensor) -> SimpTensor:
    result = np.maximum(0, x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            mask = (x._data > 0).astype(np.float32)
            x.backward(SimpTensor(grad._data * mask))
        out._grad_fn = grad_fn
    return out


def dropout(x: SimpTensor, p: float = 0.5, training: bool = True) -> SimpTensor:
    if not training or p == 0:
        return x
    mask = np.random.rand(*x.shape) > p
    result = x._data * mask / (1 - p)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            x.backward(SimpTensor(grad._data * mask / (1 - p)))
        out._grad_fn = grad_fn
    return out


def sigmoid_tensor(x: SimpTensor) -> SimpTensor:
    result = 1.0 / (1.0 + np.exp(-x._data))
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            dx = result * (1.0 - result)
            x.backward(SimpTensor(grad._data * dx))
        out._grad_fn = grad_fn
    return out


def softplus(x: SimpTensor) -> SimpTensor:
    result = np.log(1.0 + np.exp(x._data))
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            dx = 1.0 / (1.0 + np.exp(-x._data))
            x.backward(SimpTensor(grad._data * dx))
        out._grad_fn = grad_fn
    return out


def tanh(x: SimpTensor) -> SimpTensor:
    result = np.tanh(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            dx = 1.0 - result ** 2
            x.backward(SimpTensor(grad._data * dx))
        out._grad_fn = grad_fn
    return out


def clamp(x: SimpTensor, min: float = None, max: float = None) -> SimpTensor:
    result = np.clip(x._data, min, max)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            mask = np.ones_like(x._data)
            if min is not None:
                mask = mask * (x._data >= min)
            if max is not None:
                mask = mask * (x._data <= max)
            x.backward(SimpTensor(grad._data * mask))
        out._grad_fn = grad_fn
    return out


def abs(x: SimpTensor) -> SimpTensor:
    result = np.abs(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            sign = np.sign(x._data)
            sign[sign == 0] = 0
            x.backward(SimpTensor(grad._data * sign))
        out._grad_fn = grad_fn
    return out


def sqrt(x: SimpTensor) -> SimpTensor:
    result = np.sqrt(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            x.backward(grad * (0.5 / SimpTensor(result)))
        out._grad_fn = grad_fn
    return out


def exp(x: SimpTensor) -> SimpTensor:
    result = np.exp(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            x.backward(SimpTensor(grad._data * result))
        out._grad_fn = grad_fn
    return out


def log(x: SimpTensor) -> SimpTensor:
    result = np.log(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            x.backward(SimpTensor(grad._data / x._data))
        out._grad_fn = grad_fn
    return out


def stack(tensors: List[SimpTensor], dim: int = 0) -> SimpTensor:
    if not tensors:
        raise ValueError("Cannot stack empty list")
    arrays = [t._data for t in tensors]
    result = np.stack(arrays, axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = SimpTensor(result, requires_grad=requires_grad)
    if requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            for i, t in enumerate(tensors):
                if t.requires_grad:
                    grad_i = grad[i]
                    t.backward(grad_i)
        out._grad_fn = grad_fn
    return out


def cat(tensors: List[SimpTensor], dim: int = 0) -> SimpTensor:
    if not tensors:
        raise ValueError("Cannot concatenate empty list")
    arrays = [t._data for t in tensors]
    result = np.concatenate(arrays, axis=dim)
    requires_grad = any(t.requires_grad for t in tensors)
    out = SimpTensor(result, requires_grad=requires_grad)
    if requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            offset = 0
            for t in tensors:
                if t.requires_grad:
                    grad_i = grad[offset:offset + t.shape[dim]]
                    t.backward(grad_i)
                offset += t.shape[dim]
        out._grad_fn = grad_fn
    return out


def einsum(equation: str, *operands: SimpTensor) -> SimpTensor:
    arrays = [op._data for op in operands]
    result = np.einsum(equation, *arrays)
    requires_grad = any(op.requires_grad for op in operands)
    out = SimpTensor(result, requires_grad=requires_grad)
    if requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_arr = grad._data
            for i, op in enumerate(operands):
                if op.requires_grad:
                    subscripts = equation.split('->')
                    if len(subscripts) == 1:
                        subscripts = [subscripts[0], subscripts[0]]
                    else:
                        subscripts = [subscripts[0], subscripts[1]]
                    input_subscripts = subscripts[0].split(',')
                    output_subscript = subscripts[1]
                    other_subscripts = [input_subscripts[j] for j in range(len(input_subscripts)) if j != i]
                    try:
                        grad_result = np.einsum(
                            f"{','.join(other_subscripts)},{output_subscript}->{input_subscripts[i]}",
                            *[arrays[j] for j in range(len(operands)) if j != i],
                            grad_arr
                        )
                        op.backward(SimpTensor(grad_result))
                    except:
                        pass
        out._grad_fn = grad_fn
    return out


def topk(x: SimpTensor, k: int, dim: int = -1, largest: bool = True) -> Tuple[SimpTensor, SimpTensor]:
    if largest:
        indices = np.argsort(x._data, axis=dim)[:, :, -k:]
        values = np.take_along_axis(x._data, indices, axis=dim)
    else:
        indices = np.argsort(x._data, axis=dim)[:, :, :k]
        values = np.take_along_axis(x._data, indices, axis=dim)
    result = SimpTensor(values, requires_grad=x.requires_grad)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_data = np.zeros_like(x._data)
            np.put_along_axis(grad_data, indices, grad._data, axis=dim)
            x.backward(SimpTensor(grad_data))
        result._grad_fn = grad_fn
    return result, SimpTensor(indices)


def gather(input: SimpTensor, dim: int, index: SimpTensor) -> SimpTensor:
    result = np.take_along_axis(input._data, index._data.astype(np.int64), axis=dim)
    out = SimpTensor(result, requires_grad=input.requires_grad)
    if input.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_data = np.zeros_like(input._data)
            np.add.at(grad_data, tuple([slice(None)] * dim + [index._data.astype(np.int64)] + [slice(None)] * (input._data.ndim - dim - 1)), grad._data)
            input.backward(SimpTensor(grad_data))
        out._grad_fn = grad_fn
    return out


def manual_seed(seed: int) -> None:
    np.random.seed(seed)


def allclose(a: SimpTensor, b: SimpTensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return np.allclose(a._data, b._data, rtol=rtol, atol=atol)


def mean(x: SimpTensor, dim: int = None, keepdim: bool = False) -> SimpTensor:
    return x.mean(dim=dim, keepdim=keepdim)


def var(x: SimpTensor, dim: int = None, keepdim: bool = False) -> SimpTensor:
    if dim is None:
        new_data = np.array([np.var(x._data)])
    else:
        new_data = np.var(x._data, axis=dim, keepdims=keepdim)
    result = SimpTensor(new_data, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            n = x._data.size if dim is None else x._data.shape[dim]
            grad_expanded = grad._data / n
            if dim is not None and not keepdim:
                grad_expanded = np.expand_dims(grad_expanded, axis=dim)
            grad_data = np.broadcast_to(grad_expanded, x._data.shape) * (x._data - np.mean(x._data, axis=dim, keepdims=True)) * 2
            x.backward(SimpTensor(grad_data))
        result._grad_fn = grad_fn
    return result


def std(x: SimpTensor, dim: int = None, keepdim: bool = False) -> SimpTensor:
    if dim is None:
        new_data = np.array([np.std(x._data)])
    else:
        new_data = np.std(x._data, axis=dim, keepdims=keepdim)
    result = SimpTensor(new_data, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            n = x._data.size if dim is None else x._data.shape[dim]
            std_val = np.std(x._data, axis=dim, keepdims=True)
            grad_expanded = grad._data / (2 * std_val * n)
            if dim is not None and not keepdim:
                grad_expanded = np.expand_dims(grad_expanded, axis=dim)
            grad_data = np.broadcast_to(grad_expanded, x._data.shape) * (x._data - np.mean(x._data, axis=dim, keepdims=True)) * 2
            x.backward(SimpTensor(grad_data))
        result._grad_fn = grad_fn
    return result


def log_softmax(x: SimpTensor, dim: int = -1) -> SimpTensor:
    max_vals = np.max(x._data, axis=dim, keepdims=True)
    exp_x = np.exp(x._data - max_vals)
    log_sum = np.log(np.sum(exp_x, axis=dim, keepdims=True))
    result = x._data - max_vals - log_sum
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            softmax_val = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
            dx = grad._data - softmax_val * np.sum(grad._data, axis=dim, keepdims=True)
            x.backward(SimpTensor(dx))
        out._grad_fn = grad_fn
    return out


def pad(x: SimpTensor, pad_width, mode: str = 'constant', constant_values=0) -> SimpTensor:
    new_data = np.pad(x._data, pad_width, mode=mode, constant_values=constant_values)
    result = SimpTensor(new_data, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            slices = []
            for pw in pad_width:
                slices.append(slice(pw[0], grad._data.shape[len(slices)] - pw[1]))
            grad_x = grad._data[tuple(slices)]
            x.backward(SimpTensor(grad_x))
        result._grad_fn = grad_fn
    return result


def linspace(start: float, end: float, steps: int, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    data = np.linspace(start, end, steps, dtype=dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def meshgrid(*tensors: SimpTensor, indexing: str = 'xy') -> List[SimpTensor]:
    arrays = [t._data for t in tensors]
    grids = np.meshgrid(*arrays, indexing=indexing)
    return [SimpTensor(g, requires_grad=any(t.requires_grad for t in tensors)) for g in grids]


def where(condition: SimpTensor, x: SimpTensor, y: SimpTensor) -> SimpTensor:
    result = np.where(condition._data, x._data, y._data)
    requires_grad = condition.requires_grad or x.requires_grad or y.requires_grad
    out = SimpTensor(result, requires_grad=requires_grad)
    if requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            mask = condition._data
            if x.requires_grad:
                x.backward(SimpTensor(grad._data * mask))
            if y.requires_grad:
                y.backward(SimpTensor(grad._data * (1 - mask)))
        out._grad_fn = grad_fn
    return out


def eye(n: int, m: int = None) -> SimpTensor:
    m = m or n
    return SimpTensor(np.eye(n, m))


def arange(start: int = 0, end: int = None, step: int = 1) -> SimpTensor:
    if end is None:
        end = start
        start = 0
    return SimpTensor(np.arange(start, end, step))


def split(x: SimpTensor, split_size_or_sections, dim: int = 0) -> List[SimpTensor]:
    if isinstance(split_size_or_sections, int):
        sections = [split_size_or_sections] * (x._data.shape[dim] // split_size_or_sections)
        if x._data.shape[dim] % split_size_or_sections != 0:
            sections[-1] += x._data.shape[dim] % split_size_or_sections
    else:
        sections = split_size_or_sections
    arrays = np.split(x._data, sections, axis=dim)
    return [SimpTensor(arr, requires_grad=x.requires_grad) for arr in arrays]


def permute(x: SimpTensor, dims: Tuple[int, ...]) -> SimpTensor:
    return SimpTensor(np.transpose(x._data, axes=dims), requires_grad=x.requires_grad)


def sin(x: SimpTensor) -> SimpTensor:
    return SimpTensor(np.sin(x._data), requires_grad=x.requires_grad)


def cos(x: SimpTensor) -> SimpTensor:
    return SimpTensor(np.cos(x._data), requires_grad=x.requires_grad)


def tan(x: SimpTensor) -> SimpTensor:
    return SimpTensor(np.tan(x._data), requires_grad=x.requires_grad)


def silu(x: SimpTensor) -> SimpTensor:
    return SimpTensor(x._data / (1 + np.exp(-x._data)), requires_grad=x.requires_grad)


def abs(x: SimpTensor) -> SimpTensor:
    return SimpTensor(np.abs(x._data), requires_grad=x.requires_grad)


def argmax(x: SimpTensor, dim: int = None) -> SimpTensor:
    if dim is None:
        return SimpTensor(np.array([np.argmax(x._data)]))
    return SimpTensor(np.argmax(x._data, axis=dim))


def argmin(x: SimpTensor, dim: int = None) -> SimpTensor:
    if dim is None:
        return SimpTensor(np.array([np.argmin(x._data)]))
    return SimpTensor(np.argmin(x._data, axis=dim))


def clamp_min(x: SimpTensor, min: float) -> SimpTensor:
    return SimpTensor(np.maximum(x._data, min), requires_grad=x.requires_grad)


def clamp_max(x: SimpTensor, max: float) -> SimpTensor:
    return SimpTensor(np.minimum(x._data, max), requires_grad=x.requires_grad)


def flatten(x: SimpTensor, start_dim: int = 0, end_dim: int = -1) -> SimpTensor:
    if end_dim == -1:
        end_dim = x.ndim - 1
    new_shape = x.shape[:start_dim] + (np.prod(x.shape[start_dim:end_dim + 1]),) + x.shape[end_dim + 1:]
    return SimpTensor(x._data.reshape(new_shape), requires_grad=x.requires_grad)


def repeat(x: SimpTensor, repeats, dim: int = None) -> SimpTensor:
    if dim is None:
        return SimpTensor(np.repeat(x._data.flatten(), repeats).reshape(x.shape))
    return SimpTensor(np.repeat(x._data, repeats, axis=dim), requires_grad=x.requires_grad)


def cross_entropy(input: SimpTensor, target: SimpTensor) -> SimpTensor:
    log_probs = -np.log(np.clip(input._data, 1e-9, 1.0))
    target_data = target._data.astype(np.int64)
    loss = log_probs[np.arange(len(log_probs)), target_data].mean()
    return SimpTensor(np.array([loss]))


def nll_loss(input: SimpTensor, target: SimpTensor) -> SimpTensor:
    target_data = target._data.astype(np.int64)
    loss = input._data[np.arange(len(input._data)), target_data].mean()
    return SimpTensor(np.array([loss]))


def conv2d(x: SimpTensor, weight: SimpTensor, bias: Optional[SimpTensor] = None,
           stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> SimpTensor:
    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape
    
    x_padded = np.pad(x._data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    
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
    
    weight_flat = weight._data.reshape(C_out, -1)
    out = np.matmul(weight_flat, col)
    
    if bias is not None:
        out = out + bias._data.reshape(-1, 1)
    
    result = SimpTensor(out.reshape(B, C_out, H_out, W_out), 
                       requires_grad=x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad),
                       device=x.device, dtype=x.dtype)
    
    if result.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_flat = grad._data.reshape(B, C_out, -1)
            
            if weight.requires_grad:
                grad_weight = np.zeros((C_out, C_in * K * K))
                for b in range(B):
                    grad_weight += np.matmul(grad_flat[b], col[b].transpose())
                grad_weight = grad_weight.reshape(C_out, C_in // groups, K, K)
                weight.backward(SimpTensor(grad_weight))
            
            if bias is not None and bias.requires_grad:
                grad_bias = grad._data.sum(axis=(0, 2, 3))
                bias.backward(SimpTensor(grad_bias))
            
            if x.requires_grad:
                grad_col = np.zeros((B, C_in * K * K, H_out * W_out))
                for b in range(B):
                    grad_col[b] = np.matmul(weight_flat.transpose(), grad_flat[b])
                grad_col = grad_col.reshape(B, C_in, K, K, H_out, W_out)
                
                grad_x = np.zeros_like(x_padded)
                
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * stride
                        w_start = w_out * stride
                        grad_x[:, :, h_start:h_start+K, w_start:w_start+K] += grad_col[:, :, :, :, h_out, w_out]
                
                if padding > 0:
                    grad_x = grad_x[:, :, padding:-padding, padding:-padding]
                x.backward(SimpTensor(grad_x))
        
        result._grad_fn = grad_fn
    
    return result


def max_pool2d(x: SimpTensor, kernel_size: int, stride: Optional[int] = None, 
               padding: int = 0) -> SimpTensor:
    if stride is None:
        stride = kernel_size
    
    B, C, H, W = x.shape
    
    x_padded = np.pad(x._data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    result = np.zeros((B, C, H_out, W_out))
    
    for h_out in range(H_out):
        for w_out in range(W_out):
            h_start = h_out * stride
            w_start = w_out * stride
            region = x_padded[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
            result[:, :, h_out, w_out] = np.max(region, axis=(2, 3))
    
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_x = np.zeros_like(x_padded)
            
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    w_start = w_out * stride
                    region = x_padded[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
                    max_vals = result[:, :, h_out, w_out].reshape(B, C, 1, 1)
                    mask = (region == max_vals).astype(np.float32)
                    mask = mask / np.sum(mask, axis=(2, 3), keepdims=True)
                    grad_x[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size] += \
                        grad._data[:, :, h_out, w_out].reshape(B, C, 1, 1) * mask
            
            if padding > 0:
                grad_x = grad_x[:, :, padding:-padding, padding:-padding]
            x.backward(SimpTensor(grad_x))
        out._grad_fn = grad_fn
    
    return out


def avg_pool2d(x: SimpTensor, kernel_size: int, stride: Optional[int] = None, 
               padding: int = 0) -> SimpTensor:
    if stride is None:
        stride = kernel_size
    
    B, C, H, W = x.shape
    
    x_padded = np.pad(x._data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    
    result = np.zeros((B, C, H_out, W_out))
    
    for h_out in range(H_out):
        for w_out in range(W_out):
            h_start = h_out * stride
            w_start = w_out * stride
            region = x_padded[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size]
            result[:, :, h_out, w_out] = np.mean(region, axis=(2, 3))
    
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_x = np.zeros_like(x_padded)
            scale = 1.0 / (kernel_size * kernel_size)
            
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    w_start = w_out * stride
                    grad_x[:, :, h_start:h_start+kernel_size, w_start:w_start+kernel_size] += \
                        grad._data[:, :, h_out, w_out].reshape(B, C, 1, 1) * scale
            
            if padding > 0:
                grad_x = grad_x[:, :, padding:-padding, padding:-padding]
            x.backward(SimpTensor(grad_x))
        out._grad_fn = grad_fn
    
    return out


def batch_norm2d(x: SimpTensor, weight: SimpTensor, bias: SimpTensor,
                 running_mean: Optional[SimpTensor] = None, running_var: Optional[SimpTensor] = None,
                 training: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> SimpTensor:
    B, C, H, W = x.shape
    
    if training:
        mean = np.mean(x._data, axis=(0, 2, 3), keepdims=True)
        var = np.var(x._data, axis=(0, 2, 3), keepdims=True)
    else:
        mean = running_mean._data.reshape(1, C, 1, 1)
        var = running_var._data.reshape(1, C, 1, 1)
    
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm = (x._data - mean) * inv_std
    result = weight._data.reshape(1, C, 1, 1) * x_norm + bias._data.reshape(1, C, 1, 1)
    
    out = SimpTensor(result, requires_grad=x.requires_grad or weight.requires_grad or bias.requires_grad,
                    device=x.device, dtype=x.dtype)
    
    if out.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            N = B * H * W
            
            if weight.requires_grad:
                grad_weight = np.sum(grad._data * x_norm, axis=(0, 2, 3))
                weight.backward(SimpTensor(grad_weight))
            
            if bias.requires_grad:
                grad_bias = np.sum(grad._data, axis=(0, 2, 3))
                bias.backward(SimpTensor(grad_bias))
            
            if x.requires_grad:
                grad_x_norm = grad._data * weight._data.reshape(1, C, 1, 1)
                grad_var = np.sum(grad_x_norm * (x._data - mean) * (-0.5) * inv_std ** 3, axis=(0, 2, 3), keepdims=True)
                grad_mean = np.sum(grad_x_norm * (-inv_std), axis=(0, 2, 3), keepdims=True) + \
                           grad_var * np.mean(-2 * (x._data - mean), axis=(0, 2, 3), keepdims=True)
                
                grad_x = grad_x_norm * inv_std + grad_var * 2 * (x._data - mean) / N + grad_mean / N
                x.backward(SimpTensor(grad_x))
        out._grad_fn = grad_fn
    
    return out


def interpolate(x: SimpTensor, size: Optional[Tuple[int, int]] = None, 
                scale_factor: Optional[float] = None, mode: str = 'bilinear') -> SimpTensor:
    B, C, H, W = x.shape
    
    if scale_factor is not None:
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    elif size is not None:
        new_H, new_W = size
    else:
        raise ValueError("Either size or scale_factor must be provided")
    
    if mode == 'bilinear':
        y = np.arange(new_H) / (new_H - 1) * (H - 1)
        x = np.arange(new_W) / (new_W - 1) * (W - 1)
        
        y0 = np.floor(y).astype(np.int64)
        y1 = np.minimum(y0 + 1, H - 1)
        x0 = np.floor(x).astype(np.int64)
        x1 = np.minimum(x0 + 1, W - 1)
        
        wy = y - y0
        wx = x - x0
        
        result = np.zeros((B, C, new_H, new_W))
        
        for b in range(B):
            for c in range(C):
                for h in range(new_H):
                    for w in range(new_W):
                        result[b, c, h, w] = (
                            x._data[b, c, y0[h], x0[w]] * (1 - wy[h]) * (1 - wx[w]) +
                            x._data[b, c, y0[h], x1[w]] * (1 - wy[h]) * wx[w] +
                            x._data[b, c, y1[h], x0[w]] * wy[h] * (1 - wx[w]) +
                            x._data[b, c, y1[h], x1[w]] * wy[h] * wx[w]
                        )
    else:
        result = np.repeat(np.repeat(x._data, scale_factor, axis=2), scale_factor, axis=3)
    
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_x = np.zeros_like(x._data)
            
            if mode == 'bilinear':
                for b in range(B):
                    for c in range(C):
                        for h in range(new_H):
                            for w in range(new_W):
                                g = grad._data[b, c, h, w]
                                grad_x[b, c, y0[h], x0[w]] += g * (1 - wy[h]) * (1 - wx[w])
                                grad_x[b, c, y0[h], x1[w]] += g * (1 - wy[h]) * wx[w]
                                grad_x[b, c, y1[h], x0[w]] += g * wy[h] * (1 - wx[w])
                                grad_x[b, c, y1[h], x1[w]] += g * wy[h] * wx[w]
            else:
                grad_x = grad._data[:, :, ::scale_factor, ::scale_factor]
            
            x.backward(SimpTensor(grad_x))
        out._grad_fn = grad_fn
    
    return out


def eye(n: int, m: int = None, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    if m is None:
        m = n
    data = np.eye(n, m, dtype=dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def arange(start: int = 0, end: int = None, step: int = 1, device: DeviceType = DeviceType.CPU, dtype: Dtype = Dtype.FLOAT32) -> SimpTensor:
    if end is None:
        end = start
        start = 0
    data = np.arange(start, end, step, dtype=dtype.value)
    return SimpTensor(data, device=device, dtype=dtype)


def silu(x: SimpTensor) -> SimpTensor:
    result = x._data * (1.0 / (1.0 + np.exp(-np.clip(x._data, -500, 500))))
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            sigmoid_x = 1.0 / (1.0 + np.exp(-np.clip(x._data, -500, 500)))
            dx = sigmoid_x * (1.0 + x._data * (1.0 - sigmoid_x))
            x.backward(SimpTensor(grad._data * dx))
        out._grad_fn = grad_fn
    return out


def split(x: SimpTensor, split_size_or_sections, dim: int = 0) -> List[SimpTensor]:
    if isinstance(split_size_or_sections, int):
        sections = [split_size_or_sections] * (x.shape[dim] // split_size_or_sections)
        if x.shape[dim] % split_size_or_sections != 0:
            sections[-1] += x.shape[dim] % split_size_or_sections
    else:
        sections = split_size_or_sections
    
    indices = np.cumsum([0] + list(sections))
    result = []
    for i in range(len(sections)):
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(indices[i], indices[i+1])
        result.append(SimpTensor(x._data[tuple(slices)], requires_grad=x.requires_grad, device=x.device, dtype=x.dtype))
    
    return result


def permute(x: SimpTensor, *dims: int) -> SimpTensor:
    return x.transpose(*dims)


def argmax(x: SimpTensor, dim: int = -1) -> SimpTensor:
    result = np.argmax(x._data, axis=dim)
    return SimpTensor(result, requires_grad=False, device=x.device, dtype=Dtype.INT64)


def argmin(x: SimpTensor, dim: int = -1) -> SimpTensor:
    result = np.argmin(x._data, axis=dim)
    return SimpTensor(result, requires_grad=False, device=x.device, dtype=Dtype.INT64)


def clamp_min(x: SimpTensor, min: float) -> SimpTensor:
    return clamp(x, min=min)


def clamp_max(x: SimpTensor, max: float) -> SimpTensor:
    return clamp(x, max=max)


def flatten(x: SimpTensor, start_dim: int = 0, end_dim: int = -1) -> SimpTensor:
    if end_dim < 0:
        end_dim = x.ndim + end_dim
    new_shape = x.shape[:start_dim] + (np.prod(x.shape[start_dim:end_dim+1]),) + x.shape[end_dim+1:]
    return x.reshape(*new_shape)


def std(x: SimpTensor, dim: int = None, keepdim: bool = False) -> SimpTensor:
    if dim is None:
        new_data = np.array([np.std(x._data)])
    else:
        new_data = np.std(x._data, axis=dim, keepdims=keepdim)
    result = SimpTensor(new_data, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            n = x._data.size if dim is None else x._data.shape[dim]
            mean = np.mean(x._data, axis=dim, keepdims=True)
            std_val = np.std(x._data, axis=dim, keepdims=True)
            grad_expanded = grad._data / (n * std_val) if dim is not None else grad._data / (n * std_val)
            if dim is not None and not keepdim:
                grad_expanded = np.expand_dims(grad_expanded, axis=dim)
            grad_data = np.broadcast_to(grad_expanded, x._data.shape) * (x._data - mean)
            x.backward(SimpTensor(grad_data))
        result._grad_fn = grad_fn
    return result


def sin(x: SimpTensor) -> SimpTensor:
    result = np.sin(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            x.backward(SimpTensor(grad._data * np.cos(x._data)))
        out._grad_fn = grad_fn
    return out


def cos(x: SimpTensor) -> SimpTensor:
    result = np.cos(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            x.backward(SimpTensor(grad._data * -np.sin(x._data)))
        out._grad_fn = grad_fn
    return out


def tan(x: SimpTensor) -> SimpTensor:
    result = np.tan(x._data)
    out = SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)
    if x.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            x.backward(SimpTensor(grad._data / (np.cos(x._data) ** 2)))
        out._grad_fn = grad_fn
    return out


def cross_entropy(input: SimpTensor, target: SimpTensor, weight: Optional[SimpTensor] = None, reduction: str = 'mean') -> SimpTensor:
    log_probs = -np.log(np.clip(input._data, 1e-10, 1.0))
    if target._data.ndim == 1:
        target_indices = target._data.astype(np.int64)
        loss = log_probs[np.arange(len(target_indices)), target_indices]
    else:
        loss = np.sum(log_probs * target._data, axis=-1)
    
    if weight is not None:
        loss = loss * weight._data[target._data.astype(np.int64)]
    
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    out = SimpTensor(np.array([loss]))
    if input.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_data = -target._data / np.clip(input._data, 1e-10, 1.0)
            if reduction == 'mean':
                grad_data = grad_data / input._data.shape[0]
            input.backward(SimpTensor(grad._data[0] * grad_data))
        out._grad_fn = grad_fn
    return out


def nll_loss(input: SimpTensor, target: SimpTensor, weight: Optional[SimpTensor] = None, reduction: str = 'mean') -> SimpTensor:
    if target._data.ndim == 1:
        target_indices = target._data.astype(np.int64)
        loss = input._data[np.arange(len(target_indices)), target_indices]
    else:
        loss = np.sum(input._data * target._data, axis=-1)
    
    if weight is not None:
        loss = loss * weight._data[target._data.astype(np.int64)]
    
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    out = SimpTensor(np.array([loss]))
    if input.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_data = np.zeros_like(input._data)
            if target._data.ndim == 1:
                grad_data[np.arange(len(target._data)), target._data.astype(np.int64)] = 1
            else:
                grad_data = target._data
            if reduction == 'mean':
                grad_data = grad_data / input._data.shape[0]
            input.backward(SimpTensor(grad._data[0] * grad_data))
        out._grad_fn = grad_fn
    return out


def repeat_interleave(x: SimpTensor, repeats, dim: int = None) -> SimpTensor:
    if dim is None:
        result = np.repeat(x._data.flatten(), repeats)
    else:
        result = np.repeat(x._data, repeats, axis=dim)
    return SimpTensor(result, requires_grad=x.requires_grad, device=x.device, dtype=x.dtype)


def index_select(input: SimpTensor, dim: int, index: SimpTensor) -> SimpTensor:
    result = np.take(input._data, index._data.astype(np.int64), axis=dim)
    out = SimpTensor(result, requires_grad=input.requires_grad)
    if input.requires_grad:
        def grad_fn(grad: SimpTensor) -> None:
            grad_data = np.zeros_like(input._data)
            np.put_along_axis(grad_data, index._data.astype(np.int64), grad._data, axis=dim)
            input.backward(SimpTensor(grad_data))
        out._grad_fn = grad_fn
    return out


if __name__ == "__main__":
    print("=" * 60)
    print("SimplCore Tensor 自检")
    print("=" * 60)
    
    print("\n1. Testing basic operations...")
    a = randn(3, 4, requires_grad=True)
    b = randn(4, 5, requires_grad=True)
    c = a.matmul(b)
    assert c.shape == (3, 5)
    print("   matmul: PASSED")
    
    d = c.sum()
    assert d.shape == (1,)
    print("   sum: PASSED")
    
    print("\n2. Testing gradients...")
    a = randn(2, 3, requires_grad=True)
    b = randn(3, 4, requires_grad=True)
    c = a.matmul(b)
    loss = (c ** 2).mean()
    loss.backward()
    assert a.grad is not None
    assert b.grad is not None
    print("   gradients: PASSED")
    
    print("\n3. Testing softmax...")
    x = randn(2, 5, requires_grad=True)
    y = softmax(x, dim=-1)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("   softmax: PASSED")
    
    print("\n4. Testing layer_norm...")
    x = randn(2, 3, 4, requires_grad=True)
    y = layer_norm(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("   layer_norm: PASSED")
    
    print("\n5. Testing gelu...")
    x = randn(2, 3, requires_grad=True)
    y = gelu(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print("   gelu: PASSED")
    
    print("\n6. Testing einsum...")
    a = randn(2, 3, requires_grad=True)
    b = randn(3, 4, requires_grad=True)
    c = einsum("ij,jk->ik", a, b)
    loss = c.sum()
    loss.backward()
    assert a.grad is not None
    print("   einsum: PASSED")
    
    print("\n7. Testing stack...")
    tensors = [randn(2, 3) for _ in range(4)]
    stacked = stack(tensors, dim=1)
    assert stacked.shape == (2, 4, 3)
    print("   stack: PASSED")
    
    print("\n" + "=" * 60)
    print("SimplCore Tensor 自检 PASSED")
    print("=" * 60)


