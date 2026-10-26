# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Functional API - 青小团自研函数式API
类似 torch.nn.functional 的函数式接口
"""
from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Union

from .tensor import SimpTensor


def relu(input: SimpTensor) -> SimpTensor:
    """ReLU激活函数"""
    from .tensor import relu
    return relu(input)


def gelu(input: SimpTensor) -> SimpTensor:
    """GELU激活函数"""
    from .tensor import gelu
    return gelu(input)


def silu(input: SimpTensor) -> SimpTensor:
    """SiLU激活函数"""
    from .tensor import silu
    return silu(input)


def sigmoid(input: SimpTensor) -> SimpTensor:
    """Sigmoid激活函数"""
    from .tensor import sigmoid_tensor
    return sigmoid_tensor(input)


def tanh(input: SimpTensor) -> SimpTensor:
    """Tanh激活函数"""
    from .tensor import tanh
    return tanh(input)


def softmax(input: SimpTensor, dim: int = -1) -> SimpTensor:
    """Softmax函数"""
    from .tensor import softmax
    return softmax(input, dim=dim)


def log_softmax(input: SimpTensor, dim: int = -1) -> SimpTensor:
    """Log Softmax函数"""
    from .tensor import log_softmax
    return log_softmax(input, dim=dim)


def dropout(input: SimpTensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> SimpTensor:
    """Dropout"""
    from .tensor import dropout
    return dropout(input, p=p, training=training)


def linear(input: SimpTensor, weight: SimpTensor, bias: Optional[SimpTensor] = None) -> SimpTensor:
    """线性变换"""
    out = input.matmul(weight.T())
    if bias is not None:
        out = out + bias
    return out


def conv2d(input: SimpTensor, weight: SimpTensor, bias: Optional[SimpTensor] = None,
           stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> SimpTensor:
    """二维卷积"""
    from .tensor import conv2d
    return conv2d(input, weight, bias, stride=stride, padding=padding, groups=groups)


def conv1d(input: SimpTensor, weight: SimpTensor, bias: Optional[SimpTensor] = None,
           stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1) -> SimpTensor:
    """一维卷积"""
    from .tensor import conv1d
    return conv1d(input, weight, bias, stride=stride, padding=padding, groups=groups)


def max_pool2d(input: SimpTensor, kernel_size: int = 2, stride: Optional[int] = None,
               padding: int = 0, dilation: int = 1, return_indices: bool = False,
               ceil_mode: bool = False) -> SimpTensor:
    """二维最大池化"""
    from .tensor import max_pool2d
    return max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding)


def avg_pool2d(input: SimpTensor, kernel_size: int = 2, stride: Optional[int] = None,
               padding: int = 0, ceil_mode: bool = False,
               count_include_pad: bool = True, divisor_override: Optional[int] = None) -> SimpTensor:
    """二维平均池化"""
    from .tensor import avg_pool2d
    return avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding,
                      count_include_pad=count_include_pad)


def adaptive_avg_pool2d(input: SimpTensor, output_size: Tuple[int, int]) -> SimpTensor:
    """自适应平均池化"""
    B, C, H, W = input.shape
    H_out, W_out = output_size
    
    stride_h = H // H_out
    stride_w = W // W_out
    kernel_h = H - (H_out - 1) * stride_h
    kernel_w = W - (W_out - 1) * stride_w
    
    from .tensor import avg_pool2d
    return avg_pool2d(input, kernel_size=(kernel_h, kernel_w), 
                      stride=(stride_h, stride_w), padding=0)


def adaptive_max_pool2d(input: SimpTensor, output_size: Tuple[int, int]) -> SimpTensor:
    """自适应最大池化"""
    B, C, H, W = input.shape
    H_out, W_out = output_size
    
    stride_h = H // H_out
    stride_w = W // W_out
    kernel_h = H - (H_out - 1) * stride_h
    kernel_w = W - (W_out - 1) * stride_w
    
    from .tensor import max_pool2d
    return max_pool2d(input, kernel_size=(kernel_h, kernel_w),
                      stride=(stride_h, stride_w), padding=0)


def batch_norm(input: SimpTensor, running_mean: Optional[SimpTensor], running_var: Optional[SimpTensor],
               weight: Optional[SimpTensor] = None, bias: Optional[SimpTensor] = None,
               training: bool = True, momentum: float = 0.1, eps: float = 1e-5) -> SimpTensor:
    """批归一化"""
    if input.ndim == 4:
        from .tensor import batch_norm2d
        return batch_norm2d(input, weight, bias, running_mean, running_var,
                           training=training, momentum=momentum, eps=eps)
    else:
        from .tensor import batch_norm1d
        return batch_norm1d(input, weight, bias, running_mean, running_var,
                           training=training, momentum=momentum, eps=eps)


def layer_norm(input: SimpTensor, normalized_shape: Union[int, List[int], Tuple[int, ...]],
               weight: Optional[SimpTensor] = None, bias: Optional[SimpTensor] = None,
               eps: float = 1e-5) -> SimpTensor:
    """层归一化"""
    from .tensor import layer_norm
    out = layer_norm(input, eps=eps)
    
    if weight is not None and bias is not None:
        out = out * weight + bias
    elif weight is not None:
        out = out * weight
    elif bias is not None:
        out = out + bias
    
    return out


def cross_entropy(input: SimpTensor, target: SimpTensor, weight: Optional[SimpTensor] = None,
                  size_average: bool = True, ignore_index: int = -100,
                  reduce: bool = True, reduction: str = 'mean') -> SimpTensor:
    """交叉熵损失"""
    from .tensor import cross_entropy
    return cross_entropy(input, target)


def nll_loss(input: SimpTensor, target: SimpTensor, weight: Optional[SimpTensor] = None,
             size_average: bool = True, ignore_index: int = -100,
             reduce: bool = True, reduction: str = 'mean') -> SimpTensor:
    """负对数似然损失"""
    from .tensor import nll_loss
    return nll_loss(input, target)


def mse_loss(input: SimpTensor, target: SimpTensor, size_average: bool = True,
             reduce: bool = True, reduction: str = 'mean') -> SimpTensor:
    """均方误差损失"""
    diff = input - target
    loss = diff * diff
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def l1_loss(input: SimpTensor, target: SimpTensor, size_average: bool = True,
            reduce: bool = True, reduction: str = 'mean') -> SimpTensor:
    """L1损失"""
    from .tensor import abs as tensor_abs
    diff = input - target
    loss = tensor_abs(diff)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def binary_cross_entropy_with_logits(input: SimpTensor, target: SimpTensor,
                                     weight: Optional[SimpTensor] = None,
                                     size_average: bool = True, reduce: bool = True,
                                     reduction: str = 'mean', pos_weight: Optional[SimpTensor] = None) -> SimpTensor:
    """带logits的二分类交叉熵"""
    max_val = (-input._data).clip(min=0)
    loss = max_val + input._data * target._data
    loss = loss + np.log1p(np.exp(-np.abs(input._data)))
    
    if pos_weight is not None:
        log_weight = 1 + (pos_weight._data - 1) * target._data
        loss = loss * log_weight
    
    out = SimpTensor(loss)
    
    if reduction == 'mean':
        return out.mean()
    elif reduction == 'sum':
        return out.sum()
    return out


def interpolate(input: SimpTensor, size: Optional[Tuple[int, int]] = None,
                scale_factor: Optional[float] = None, mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None) -> SimpTensor:
    """上采样/插值"""
    from .tensor import interpolate
    return interpolate(input, size=size, scale_factor=scale_factor, mode=mode,
                       align_corners=align_corners if align_corners is not None else False)


def upsample(input: SimpTensor, size: Optional[Tuple[int, int]] = None,
             scale_factor: Optional[float] = None, mode: str = 'nearest',
             align_corners: Optional[bool] = None) -> SimpTensor:
    """上采样"""
    return interpolate(input, size=size, scale_factor=scale_factor, mode=mode,
                       align_corners=align_corners)


def pad(input: SimpTensor, pad: Tuple[int, ...], mode: str = 'constant', value: float = 0) -> SimpTensor:
    """填充"""
    ndim = input.ndim
    pad_width = [(0, 0)] * (ndim - len(pad) // 2) + [(pad[2*i], pad[2*i+1]) for i in range(len(pad) // 2 - 1, -1, -1)]
    
    if mode == 'constant':
        result = np.pad(input._data, pad_width, mode='constant', constant_values=value)
    elif mode == 'reflect':
        result = np.pad(input._data, pad_width, mode='reflect')
    elif mode == 'replicate':
        result = np.pad(input._data, pad_width, mode='edge')
    else:
        result = np.pad(input._data, pad_width, mode=mode)
    
    return SimpTensor(result, requires_grad=input.requires_grad)


def flatten(input: SimpTensor, start_dim: int = 0, end_dim: int = -1) -> SimpTensor:
    """展平"""
    from .tensor import flatten
    return flatten(input, start_dim=start_dim, end_dim=end_dim)


def one_hot(tensor: SimpTensor, num_classes: int = -1) -> SimpTensor:
    """One-hot编码"""
    data = tensor._data.astype(np.int64)
    
    if num_classes == -1:
        num_classes = int(data.max()) + 1
    
    one_hot_data = np.eye(num_classes, dtype=np.float32)[data]
    
    return SimpTensor(one_hot_data)


__all__ = [
    "relu",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "softmax",
    "log_softmax",
    "dropout",
    "linear",
    "conv2d",
    "conv1d",
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "batch_norm",
    "layer_norm",
    "cross_entropy",
    "nll_loss",
    "mse_loss",
    "l1_loss",
    "binary_cross_entropy_with_logits",
    "interpolate",
    "upsample",
    "pad",
    "flatten",
    "one_hot",
]


