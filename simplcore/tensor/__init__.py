# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor — 青小团张量计算引擎（自研原生实现，零第三方框架依赖）
=========================================================================

本模块是青小团大模型的自研张量计算引擎，提供：

* 完整的张量代数（加/减/乘/除/矩阵乘/广播/索引等）
* 反向模式自动微分（拓扑排序 + 梯度累积），无需 PyTorch / TensorFlow
* 神经网络基础模块（Linear / Embedding / RMSNorm / Dropout …）
* 原生优化器（AdamW / SGD / Adam）与学习率调度器
* 训练后端（梯度检查点）、序列化、集体通信原语

**项目已彻底移除对 PyTorch / TensorFlow / DeepSpeed 等第三方深度学习框架的依赖：**
所有前向传播、反向传播、梯度更新与优化器逻辑均由本引擎原生实现。
"""
from __future__ import annotations

import contextlib
from .backend import (
    get_backend,
    set_backend,
    is_torch_backend,
    is_numpy_backend,
    is_gpu_backend,
    get_device,
    gpu_memory_info,
    gpu_name,
    gpu_total_gb,
    can_fit_gb,
    _CUPY,
)

# ============================================================
# 自研 NumPy 后端（唯一后端）
# ============================================================
from .tensor import (
    Tensor, no_grad, enable_grad, is_grad_enabled,
    tensor, zeros, ones, randn, rand, eye, arange,
    linspace, full, full_like, zeros_like, ones_like,
    cat, stack, where, manual_seed, allclose,
    equal, isfinite, frombuffer, from_numpy, randn_like,
    relu, gelu, gelu_tanh, sigmoid, tanh, silu, leaky_relu,
    softmax, log_softmax, dropout, layer_norm, batch_norm,
    cross_entropy, nll_loss, mse_loss, l1_loss,
    binary_cross_entropy,
    matmul, bmm, randint, erf,
    topk, flip, triu, tril, outer,
    cumsum, sort, exp, log, sqrt,
    cos, sin, abs, multinomial,
    randint_like, empty, einsum, diag,
)

# 注册子模块
from . import nn
from .nn import functional
from . import utils
from . import optim as _optim_mod
from . import device as _device_mod
from . import io as _io_mod

# NN Module 系统
from .module import (
    Parameter, Module, ModuleList, Sequential,
    Linear, Embedding, Dropout, RMSNorm,
    ReLU, Sigmoid, LayerNorm, Functional, init,
)

# 优化器
from .optim import (
    Optimizer, AdamW, SGD, Adam,
    CosineAnnealingLR, LinearLR, LambdaLR, SequentialLR,
)

from .device import device as _device_cls
device = _device_cls

# ============================================================
# 自研分布式通信层（零 torch.distributed 依赖）
# ============================================================
from . import distributed

import numpy as np


def is_available(device_type: str = "cpu") -> bool:
    """CPU 后端恒可用。"""
    return device_type.lower() == "cpu"


# ── argmax / argmin 顶层封装（torch.argmax/argmin 兼容）──────────────
def argmax(input, dim=None, keepdim: bool = False):
    return input.argmax(dim=dim, keepdim=keepdim)


def argmin(input, dim=None, keepdim: bool = False):
    return input.argmin(dim=dim, keepdim=keepdim)


# CUDA 兼容层（无 GPU 时全部降级为空操作，保证接口兼容）
class _CudaCompat:
    @staticmethod
    def is_available() -> bool:
        return is_gpu_backend()

    @staticmethod
    def empty_cache() -> None:
        # 尝试通过后端释放 GPU 缓存（cuPy 等）
        if _CUPY is not None:
            with contextlib.suppress(Exception):
                _CUPY.get_default_memory_pool().free_all_blocks()
        pass

    @staticmethod
    def memory_allocated(device=None) -> int:
        free, total = gpu_memory_info()
        return max(0, total - free)

    @staticmethod
    def memory_reserved(device=None) -> int:
        free, total = gpu_memory_info()
        return max(0, total - free)

    @staticmethod
    def device_count() -> int:
        return 1 if is_gpu_backend() else 0

    @staticmethod
    def manual_seed_all(seed: int) -> None:
        """兼容 torch.cuda.manual_seed_all：同步所有设备随机种子。"""
        manual_seed(seed)

    @staticmethod
    def synchronize() -> None:
        """同步点：GPU 后端等待计算完成（cuPy 同步）。"""
        if _CUPY is not None:
            with contextlib.suppress(Exception):
                _CUPY.cuda.Device(0).synchronize()
        pass

    @staticmethod
    def utilization(device: int = 0) -> int:
        return 0

    @staticmethod
    def mem_get_info(device: int = 0):
        return gpu_memory_info()

    @staticmethod
    def get_device_name(device: int = 0) -> str:
        return gpu_name() or "cpu"

    @staticmethod
    def temperature(device: int = 0) -> int:
        return 0


cuda = _CudaCompat

float32 = np.float32
float64 = np.float64
float16 = np.float16
bfloat16 = np.float16  # NumPy 无 bfloat16, 降级为 float16
int32 = np.int32
int64 = np.int64
int16 = np.int16
int8 = np.int8
uint8 = np.uint8
bool = np.bool_

# ── dtype 别名（兼容 torch.* 用法）──────────────────────────
float = np.float32
long = np.int64
int = np.int64

# channels_last / contiguous_format 等内存格式标记（原生引擎忽略，仅占位）
channels_last = "channels_last"
contiguous_format = "contiguous_format"


# normalize 函数 (numpy 后端)
def normalize(x, p=2.0, dim=1, eps=1e-12):
    from .tensor import Tensor
    if isinstance(x, Tensor):
        abs_x = x.abs()
        norm = abs_x.pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)
        norm = norm.clamp(min=eps)
        return x / norm
    return x


# gelu_tanh 变体（显式暴露到顶层）
def gelu_tanh(x, *args, **kwargs):
    from .tensor import Tensor
    if isinstance(x, Tensor):
        data = x.data
        out = 0.5 * data * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (data + 0.044715 * data ** 3)
        ))
        return Tensor(out, requires_grad=x.requires_grad)
    return x


from .io import save, load

# 确保 nn.functional 暴露自定义算子
functional = nn.functional
functional.gelu_tanh = gelu_tanh
functional.normalize = normalize
if not hasattr(functional, "rms_norm"):
    from .nn.functional import rms_norm as _rms_norm_fn
    functional.rms_norm = _rms_norm_fn

# ============================================================
# 梯度裁剪工具（torch.nn.utils 兼容）
# ============================================================
from .nn.utils import clip_grad_norm_, clip_grad_value_

# ============================================================
# 自动混合精度 (AMP) 兼容层（原生 float32 直通，无 GPU 时为 no-op）
# ============================================================
from .amp import GradScaler, autocast
from . import amp

# ============================================================
# autograd 兼容层（仅异常检测标记，原生引擎忽略）
# ============================================================
from . import autograd

# ============================================================
# 版本信息（兼容 torch.version.hip 检测）
# ============================================================
from .version import version

# ============================================================
# torch.compile 兼容（原生引擎直接返回模型，不做 JIT 编译）
# ============================================================
def compile(model, *args, **kwargs):
    """兼容 torch.compile：原生引擎直接返回原模型（无 JIT 融合）。"""
    return model


# ============================================================
# 统一导出
# ============================================================
__all__ = [
    # 核心
    "Tensor", "no_grad", "enable_grad", "is_grad_enabled",
    # 工厂
    "tensor", "zeros", "ones", "randn", "rand", "eye", "arange",
    "linspace", "full", "full_like", "zeros_like", "ones_like",
    "empty", "randint", "randint_like",
    # 拼接 / 条件
    "cat", "stack", "where", "manual_seed", "allclose",
    "equal", "isfinite", "frombuffer",
    # 数学
    "matmul", "bmm", "topk", "flip", "triu", "tril", "outer",
    "cumsum", "sort", "exp", "log", "sqrt", "cos", "sin", "abs",
    "multinomial", "erf", "einsum", "diag",
    "argmax", "argmin",
    # 激活 / 归一化 / 损失
    "relu", "gelu", "gelu_tanh", "sigmoid", "tanh", "silu", "leaky_relu",
    "softmax", "log_softmax", "dropout", "layer_norm", "batch_norm",
    "cross_entropy", "nll_loss", "mse_loss", "l1_loss",
    "binary_cross_entropy", "normalize",
    # dtype
    "float32", "float64", "float16", "bfloat16",
    "int32", "int64", "int16", "int8", "uint8", "bool",
    "float", "long", "int",
    # 设备
    "cuda", "is_available", "get_device", "device",
    # 序列化
    "save", "load",
    # NN
    "nn", "functional", "Parameter", "Module", "ModuleList", "Sequential",
    "Linear", "Embedding", "Dropout", "RMSNorm",
    "ReLU", "Sigmoid", "LayerNorm", "Functional", "init",
    # 优化器
    "optim", "Optimizer", "AdamW", "SGD", "Adam",
    "CosineAnnealingLR", "LinearLR", "LambdaLR", "SequentialLR",
    # utils
    "utils",
    # 后端
    "get_backend", "set_backend", "is_torch_backend", "is_numpy_backend",
    "is_gpu_backend", "gpu_memory_info", "gpu_name", "gpu_total_gb", "can_fit_gb",
    # 分布式 / 并行
    "distributed", "compile", "version",
    "autograd", "amp",
]
