# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore NN Module — 青小团自研模块系统
================================================
基于 ``simplcore.tensor.Tensor`` 实现的神经网络模块层，提供与
``torch.nn.Module`` 等价的核心能力，可作为 PyTorch 的轻量自研替代。

设计要点
--------
* ``Parameter`` 继承自 ``Tensor``，自动 ``requires_grad=True``。
* ``Module`` 通过 ``__setattr__`` / ``__getattr__`` 拦截对 ``Parameter``、
  子 ``Module`` 与 buffer 的赋值/读取，使得参数能被父模块递归收集。
* ``__call__`` 动态派发到 ``forward``（保证子类多态正确）。
* 所有数值计算均走自研 ``Tensor`` 的自动微分，梯度可正确回传。

使用示例
--------
    from simplcore.tensor import Tensor
    from simplcore.tensor.module import (
        Linear, Sequential, ReLU, ModuleList, RMSNorm, Embedding, init,
    )

    model = Sequential(
        Linear(8, 16), ReLU(), Linear(16, 4),
    )
    x = Tensor([[0.1] * 8], requires_grad=True)
    y = model(x)
    y.sum().backward()
    for p in model.parameters():
        print(p.grad.shape)
"""
from __future__ import annotations

import math
import numpy as np
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, Iterable,
)

from .tensor import (
    Tensor,
    is_grad_enabled,
    # 激活 / 归一化 / dropout
    relu,
    gelu,
    sigmoid,
    tanh,
    silu,
    softmax,
    log_softmax,
    dropout,
    layer_norm,
    # 损失
    cross_entropy,
    nll_loss,
    mse_loss,
    l1_loss,
)

__all__ = [
    "Parameter",
    "Module",
    "ModuleList",
    "Sequential",
    "Linear",
    "Embedding",
    "Dropout",
    "RMSNorm",
    "ReLU",
    "Sigmoid",
    "LayerNorm",
    "Functional",
    "init",
]


# ============================================================
# Parameter —— 参数张量
# ============================================================
class Parameter(Tensor):
    """参数张量，作为 ``Module`` 的可学习权重。

    继承自 :class:`Tensor`，构造时自动设置 ``requires_grad=True``，
    从而被 ``Module.parameters()`` 收集并参与反向传播。
    """

    def __init__(self, data: Any, requires_grad: bool = True,
                 dtype: Any = None, device: str = "cpu") -> None:
        # 参数恒为需要梯度（与 PyTorch 语义一致）
        super().__init__(data, requires_grad=True, dtype=dtype, device=device)

    def __repr__(self) -> str:
        return (f"Parameter(shape={tuple(self.data.shape)}, "
                f"dtype={self.data.dtype}, requires_grad={self.requires_grad})")


# ============================================================
# 内部工具
# ============================================================
def _validate_module(module: Any) -> "Module":
    """断言传入对象是 Module 实例，否则抛出 TypeError。"""
    if not isinstance(module, Module):
        raise TypeError(
            f"期望 Module 实例，得到 {type(module).__name__}")
    return module


# ============================================================
# Module —— 基类
# ============================================================
class Module:
    """所有神经网络模块的基类，替代 ``torch.nn.Module``。

    子类只需：
    1. 在 ``__init__`` 中调用 ``super().__init__()``；
    2. 将可学习权重声明为 ``self.xxx = Parameter(...)``；
    3. 将子模块声明为 ``self.xxx = SomeModule(...)``；
    4. 实现 ``forward(self, *args, **kwargs)``。

    之后即可通过 ``model(x)`` 调用、``model.parameters()`` 收集参数、
    ``model.state_dict()`` 序列化等。
    """

    def __init__(self) -> None:
        # 必须在赋值任意 Parameter / 子 Module 之前完成；
        # 直接写入 __dict__，绕过自定义 __setattr__
        self.__dict__["_parameters"]: Dict[str, Optional[Parameter]] = {}
        self.__dict__["_modules"]: Dict[str, Optional["Module"]] = {}
        self.__dict__["_buffers"]: Dict[str, Optional[Tensor]] = {}
        self.__dict__["_training"]: bool = True

    # ------------------------------------------------------------
    # 属性拦截：自动登记 Parameter / 子 Module / buffer
    # ------------------------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            params = self.__dict__.get("_parameters")
            if params is None:
                raise AttributeError(
                    "无法在调用 Module.__init__() 之前赋值 Parameter")
            # 从其它容器中移除同名项，避免残留
            self.__dict__.pop(name, None)
            self._modules.pop(name, None)
            self._buffers.pop(name, None)
            params[name] = value
        elif isinstance(value, Module):
            modules = self.__dict__.get("_modules")
            if modules is None:
                raise AttributeError(
                    "无法在调用 Module.__init__() 之前赋值子 Module")
            self.__dict__.pop(name, None)
            self._parameters.pop(name, None)
            self._buffers.pop(name, None)
            modules[name] = value
        else:
            # buffer 重赋值：保持登记在 _buffers 中
            buffers = self.__dict__.get("_buffers")
            if buffers is not None and name in buffers:
                buffers[name] = value
            else:
                # 普通属性；同时清理可能的历史登记
                self._parameters.pop(name, None)
                self._modules.pop(name, None)
                object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # 仅在常规查找失败时触发；访问内部容器用 __dict__ 避免递归
        if "_parameters" in self.__dict__:
            params = self.__dict__["_parameters"]
            if name in params:
                return params[name]
        if "_buffers" in self.__dict__:
            buffers = self.__dict__["_buffers"]
            if name in buffers:
                return buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' 对象没有属性 '{name}'")

    # ------------------------------------------------------------
    # 训练 / 推理模式
    # ------------------------------------------------------------
    @property
    def training(self) -> bool:
        return self.__dict__["_training"]

    def train(self, mode: bool = True) -> "Module":
        self.__dict__["_training"] = bool(mode)
        for m in self._modules.values():
            if m is not None:
                m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # ------------------------------------------------------------
    # 前向调用（动态派发到子类 forward）
    # ------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Module[{type(self).__name__}] 未实现 forward()")

    # ------------------------------------------------------------
    # 注册接口
    # ------------------------------------------------------------
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "无法在调用 Module.__init__() 之前注册 parameter")
        self.__dict__.pop(name, None)
        self._modules.pop(name, None)
        self._buffers.pop(name, None)
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"register_parameter 期望 Parameter，得到 "
                            f"{type(param).__name__}")
        self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Any]) -> None:
        if "_buffers" not in self.__dict__:
            raise AttributeError(
                "无法在调用 Module.__init__() 之前注册 buffer")
        self.__dict__.pop(name, None)
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        if tensor is None:
            self._buffers[name] = None
        else:
            if not isinstance(tensor, Tensor):
                tensor = Tensor(tensor)
            self._buffers[name] = tensor

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        if "_modules" not in self.__dict__:
            raise AttributeError(
                "无法在调用 Module.__init__() 之前添加子 Module")
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"add_module 期望 Module，得到 "
                            f"{type(module).__name__}")
        self.__dict__.pop(name, None)
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        self._modules[name] = module

    # ------------------------------------------------------------
    # 参数 / buffer / 子模块遍历
    # ------------------------------------------------------------
    def named_parameters(self, prefix: str = "",
                         recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            if p is None:
                continue
            yield (f"{prefix}.{name}" if prefix else name), p
        if recurse:
            for mname, m in self._modules.items():
                if m is None:
                    continue
                sub_prefix = f"{prefix}.{mname}" if prefix else mname
                yield from m.named_parameters(sub_prefix, recurse=True)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_buffers(self, prefix: str = "",
                      recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        for name, b in self._buffers.items():
            if b is None:
                continue
            yield (f"{prefix}.{name}" if prefix else name), b
        if recurse:
            for mname, m in self._modules.items():
                if m is None:
                    continue
                sub_prefix = f"{prefix}.{mname}" if prefix else mname
                yield from m.named_buffers(sub_prefix, recurse=True)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        for name, m in self._modules.items():
            if m is not None:
                yield name, m

    def children(self) -> Iterator["Module"]:
        for _, m in self.named_children():
            yield m

    def named_modules(self, memo: Optional[set] = None,
                      prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            yield prefix, self
            for name, m in self._modules.items():
                if m is None or id(m) in memo:
                    continue
                sub_prefix = f"{prefix}.{name}" if prefix else name
                yield from m.named_modules(memo, sub_prefix)

    def modules(self) -> Iterator["Module"]:
        for _, m in self.named_modules():
            yield m

    # ------------------------------------------------------------
    # 设备 / dtype 迁移
    # ------------------------------------------------------------
    def to(self, device: Any = None, dtype: Any = None) -> "Module":
        if device is not None and str(device).lower() not in ("cpu",):
            raise RuntimeError(
                f"SimplCore 仅支持 'cpu' 设备，收到 {device!r}")
        target = None if dtype is None else np.dtype(dtype)

        def _move(t: Tensor) -> Tensor:
            if target is not None and t.data.dtype != target:
                t.data = t.data.astype(target)
            return t

        for p in self._parameters.values():
            if p is not None:
                _move(p)
        for b in self._buffers.values():
            if isinstance(b, Tensor):
                _move(b)
        for m in self._modules.values():
            if m is not None:
                m.to(device=device, dtype=dtype)
        return self

    def cpu(self) -> "Module":
        return self.to(device="cpu")

    def float(self) -> "Module":
        return self.to(dtype=np.float32)

    # ------------------------------------------------------------
    # 梯度清零
    # ------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters():
            if set_to_none:
                p.grad = None
            else:
                if p.grad is not None:
                    p.grad = np.zeros_like(p.data)

    # ------------------------------------------------------------
    # apply：递归应用函数到所有子模块（先子后己）
    # ------------------------------------------------------------
    def apply(self, fn: Callable[["Module"], None]) -> "Module":
        for m in self._modules.values():
            if m is not None:
                m.apply(fn)
        fn(self)
        return self

    # ------------------------------------------------------------
    # 状态字典：序列化 / 反序列化
    # ------------------------------------------------------------
    def state_dict(self) -> Dict[str, np.ndarray]:
        sd: Dict[str, np.ndarray] = {}
        for name, p in self.named_parameters():
            sd[name] = p.data.copy()
        for name, b in self.named_buffers():
            if isinstance(b, Tensor):
                sd[name] = b.data.copy()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any],
                        strict: bool = True) -> Tuple[List[str], List[str]]:
        own_params: Dict[str, Parameter] = dict(self.named_parameters())
        own_buffers: Dict[str, Tensor] = {
            n: b for n, b in self.named_buffers() if isinstance(b, Tensor)}
        own_keys = set(own_params.keys()) | set(own_buffers.keys())
        loaded_keys = set(state_dict.keys())

        missing = list(own_keys - loaded_keys)
        unexpected = list(loaded_keys - own_keys)

        if strict:
            errors = []
            if missing:
                errors.append(f"缺失的键: {sorted(missing)}")
            if unexpected:
                errors.append(f"多余的键: {sorted(unexpected)}")
            if errors:
                raise RuntimeError(
                    "load_state_dict 出错（strict=True）: " + "; ".join(errors))

        for name, p in own_params.items():
            if name in state_dict:
                val = state_dict[name]
                arr = val.data if isinstance(val, Tensor) else np.asarray(val)
                p.data[...] = arr.astype(p.data.dtype, copy=False)
        for name, b in own_buffers.items():
            if name in state_dict:
                val = state_dict[name]
                arr = val.data if isinstance(val, Tensor) else np.asarray(val)
                b.data[...] = arr.astype(b.data.dtype, copy=False)
        return missing, unexpected

    # ------------------------------------------------------------
    # 表示
    # ------------------------------------------------------------
    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        child_lines: List[str] = []
        for name, m in self._modules.items():
            mod_str = repr(m).replace("\n", "\n  ")
            child_lines.append(f"  ({name}): {mod_str}")
        for name, p in self._parameters.items():
            child_lines.append(f"  ({name}): {p!r}")
        extra = self.extra_repr()
        if extra:
            child_lines.append(f"  {extra}")
        if not child_lines:
            return f"{type(self).__name__}()"
        return f"{type(self).__name__}(\n" + ",\n".join(child_lines) + "\n)"


# ============================================================
# ModuleList —— 模块列表容器
# ============================================================
class ModuleList(Module):
    """像 ``list`` 一样使用的子模块容器，参数能被父模块收集。"""

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "ModuleList"]:
        if isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
            return ModuleList([self._modules[str(i)] for i in indices])
        if idx < 0:
            idx += len(self)
        return self._modules[str(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        if idx < 0:
            idx += len(self)
        _validate_module(module)
        self._modules[str(idx)] = module

    def __delitem__(self, idx: int) -> None:
        if idx < 0:
            idx += len(self)
        items = list(self._modules.values())
        del items[idx]
        self._modules.clear()
        for i, m in enumerate(items):
            self._modules[str(i)] = m

    def append(self, module: Module) -> "ModuleList":
        _validate_module(module)
        self._modules[str(len(self))] = module
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        for m in modules:
            self.append(m)
        return self

    def insert(self, idx: int, module: Module) -> "ModuleList":
        _validate_module(module)
        if idx < 0:
            idx += len(self)
        items = list(self._modules.values())
        items.insert(idx, module)
        self._modules.clear()
        for i, m in enumerate(items):
            self._modules[str(i)] = m
        return self

    def forward(self, x: Any) -> Any:
        # 默认行为：依次把输入喂给每个子模块
        for m in self._modules.values():
            x = m(x)
        return x


# ============================================================
# Sequential —— 顺序容器
# ============================================================
class Sequential(Module):
    """按顺序执行子模块的容器。

    支持 ``Sequential(a, b, c)`` 或 ``Sequential([a, b, c])``。
    调用时将上一模块输出作为下一模块输入；若输出为 tuple 则自动解包。
    """

    def __init__(self, *args: Any) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        for i, m in enumerate(args):
            _validate_module(m)
            self._modules[str(i)] = m

    def forward(self, *args: Any) -> Any:
        for m in self._modules.values():
            out = m(*args)
            if isinstance(out, tuple):
                args = out
            else:
                args = (out,)
        if len(args) == 1:
            return args[0]
        return args


# ============================================================
# Linear —— 线性层
# ============================================================
class Linear(Module):
    """全连接层：``y = x @ weight.T + bias``。

    Parameters
    ----------
    in_features : int
    out_features : int
    bias : bool
        是否使用偏置。
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(
            np.empty((self.out_features, self.in_features), dtype=np.float32))
        if bias:
            self.bias = Parameter(np.empty(self.out_features, dtype=np.float32))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 与 PyTorch 一致：Kaiming uniform(a=sqrt(5))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ============================================================
# Embedding —— 嵌入层
# ============================================================
class Embedding(Module):
    """查表式嵌入层。

    Parameters
    ----------
    num_embeddings : int
        词表大小。
    embedding_dim : int
        嵌入维度。
    padding_idx : int, optional
        该索引对应的嵌入向量恒为零，且反向时不接收梯度。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx
        self.weight = Parameter(
            np.empty((self.num_embeddings, self.embedding_dim), dtype=np.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx] = 0.0

    def forward(self, indices: Any) -> Tensor:
        w = self.weight
        if isinstance(indices, Tensor):
            idx = indices.data
        else:
            idx = np.asarray(indices)
        idx = np.asarray(idx).astype(np.int64)
        out_data = w.data[idx]

        req = w.requires_grad and is_grad_enabled()
        out = Tensor(out_data, requires_grad=req)
        out._op = "embedding"
        if req:
            out._prev = {w}
            pad = self.padding_idx

            def _backward() -> None:
                if out.grad is None:
                    return
                g_w = np.zeros_like(w.data, dtype=w.data.dtype)
                # 重复索引梯度正确累加
                np.add.at(g_w, idx, out.grad)
                if pad is not None:
                    g_w[pad] = 0.0
                # 累加进 w.grad（与引擎 _add_grad 语义一致）
                if w.grad is None:
                    w.grad = np.array(g_w, dtype=w.data.dtype)
                else:
                    w.grad = w.grad + g_w
            out._backward = _backward
        return out

    def extra_repr(self) -> str:
        return (f"num_embeddings={self.num_embeddings}, "
                f"embedding_dim={self.embedding_dim}, "
                f"padding_idx={self.padding_idx}")


# ============================================================
# Dropout —— 随机失活
# ============================================================
class Dropout(Module):
    """训练时按概率 ``p`` 随机置零并缩放，推理时不操作。"""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout 概率 p 须在 [0, 1] 内，得到 {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        return dropout(x, p=self.p, training=self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


# ============================================================
# RMSNorm —— RMS 归一化
# ============================================================
class RMSNorm(Module):
    """Root Mean Square 归一化。

    ``forward(x) = x * rsqrt(mean(x^2, -1) + eps) * weight``
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = Parameter(np.ones(self.dim, dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        # 沿最后一维计算均方（避免 .mean keepdim 在 fp16 下的潜在问题）
        D = x.shape[-1]
        sq = x * x
        ms = sq.sum(dim=-1, keepdim=True) / D
        # rsqrt = (ms + eps)^(-1/2)
        inv = (ms + self.eps).pow(-0.5)
        # 保证 weight 沿最后一维广播 (兼容任意前导维度)
        w = self.weight
        if w.data.ndim == 1:
            if x.data.ndim == 3:
                w = w.reshape(1, 1, -1)
            elif x.data.ndim == 2:
                w = w.reshape(1, -1)
        return x * inv * w

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


# ============================================================
# 激活函数模块
# ============================================================
class ReLU(Module):
    """逐元素 ReLU。"""

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Sigmoid(Module):
    """逐元素 Sigmoid。"""

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


# ============================================================
# LayerNorm —— 层归一化
# ============================================================
class LayerNorm(Module):
    """层归一化，对最后 ``len(normalized_shape)`` 个维度做标准化。"""

    def __init__(self, normalized_shape: Union[int, Iterable[int]],
                 eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: Tuple[int, ...] = tuple(int(s) for s in normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
            self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.normalized_shape,
                          self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
                f"elementwise_affine={self.elementwise_affine}")


# ============================================================
# Functional —— 函数式命名空间
# ============================================================
class Functional:
    """函数式接口命名空间，全部转发到 ``simplcore.tensor`` 中的实现。"""

    @staticmethod
    def relu(x: Tensor) -> Tensor:
        return relu(x)

    @staticmethod
    def gelu(x: Tensor) -> Tensor:
        return gelu(x)

    @staticmethod
    def silu(x: Tensor) -> Tensor:
        return silu(x)

    @staticmethod
    def sigmoid(x: Tensor) -> Tensor:
        return sigmoid(x)

    @staticmethod
    def tanh(x: Tensor) -> Tensor:
        return tanh(x)

    @staticmethod
    def softmax(x: Tensor, dim: int = -1) -> Tensor:
        return softmax(x, dim=dim)

    @staticmethod
    def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
        return log_softmax(x, dim=dim)

    @staticmethod
    def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
        return dropout(x, p=p, training=training)

    @staticmethod
    def layer_norm(x: Tensor, normalized_shape, weight: Optional[Tensor] = None,
                   bias: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
        return layer_norm(x, normalized_shape, weight, bias, eps)

    @staticmethod
    def cross_entropy(input: Tensor, target: Any,
                      reduction: str = "mean") -> Tensor:
        return cross_entropy(input, target, reduction=reduction)

    @staticmethod
    def nll_loss(input: Tensor, target: Any,
                 reduction: str = "mean") -> Tensor:
        return nll_loss(input, target, reduction=reduction)

    @staticmethod
    def mse_loss(input: Tensor, target: Any,
                 reduction: str = "mean") -> Tensor:
        return mse_loss(input, target, reduction=reduction)

    @staticmethod
    def l1_loss(input: Tensor, target: Any,
                reduction: str = "mean") -> Tensor:
        return l1_loss(input, target, reduction=reduction)

    @staticmethod
    def normalize(x: Tensor, p: float = 2.0, dim: int = 1,
                  eps: float = 1e-12) -> Tensor:
        """沿 ``dim`` 做 L_p 归一化：``x / max(||x||_p, eps)``。"""
        abs_x = x.abs()
        norm = abs_x.pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)
        norm = norm.clamp(min=eps)
        return x / norm


# ============================================================
# init —— 参数初始化命名空间（原地操作 .data）
# ============================================================
def _calculate_fan_in_and_fan_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """仿照 PyTorch 计算 fan_in / fan_out。"""
    if len(shape) < 2:
        n = int(shape[0]) if len(shape) == 1 else 1
        return n, n
    fan_in = int(shape[1])
    fan_out = int(shape[0])
    if len(shape) > 2:
        receptive_field = int(np.prod(shape[2:]))
        fan_in *= receptive_field
        fan_out *= receptive_field
    return fan_in, fan_out


class init:
    """原地参数初始化命名空间，操作 ``Tensor.data``（不影响自动微分图）。"""

    @staticmethod
    def normal_(tensor: Any, mean: float = 0.0, std: float = 1.0) -> Any:
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        t[...] = np.random.normal(mean, std, size=t.shape).astype(t.dtype)
        return tensor

    @staticmethod
    def zeros_(tensor: Any) -> Any:
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        t[...] = 0
        return tensor

    @staticmethod
    def ones_(tensor: Any) -> Any:
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        t[...] = 1
        return tensor

    @staticmethod
    def uniform_(tensor: Any, a: float = 0.0, b: float = 1.0) -> Any:
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        t[...] = np.random.uniform(a, b, size=t.shape).astype(t.dtype)
        return tensor

    @staticmethod
    def constant_(tensor: Any, val: float) -> Any:
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        t[...] = val
        return tensor

    @staticmethod
    def kaiming_uniform_(tensor: Any, a: float = 0.0,
                         mode: str = "fan_in") -> Any:
        """Kaiming 均匀初始化。

        Parameters
        ----------
        a : float
            该层后面接的激活函数的负半轴斜率（ReLU 用 0）。
        mode : str
            ``'fan_in'``（默认，保持前向方差）或 ``'fan_out'``。
        """
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        shape = tuple(t.shape)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        else:
            raise ValueError(f"mode 须为 'fan_in' 或 'fan_out'，得到 {mode!r}")
        gain = math.sqrt(2.0 / (1.0 + a * a))
        bound = gain / math.sqrt(fan) if fan > 0 else 0.0
        t[...] = np.random.uniform(-bound, bound, size=t.shape).astype(t.dtype)
        return tensor

    @staticmethod
    def kaiming_normal_(tensor: Any, a: float = 0.0,
                        mode: str = "fan_in") -> Any:
        """Kaiming 正态初始化。"""
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        shape = tuple(t.shape)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        if mode == "fan_in":
            fan = fan_in
        elif mode == "fan_out":
            fan = fan_out
        else:
            raise ValueError(f"mode 须为 'fan_in' 或 'fan_out'，得到 {mode!r}")
        gain = math.sqrt(2.0 / (1.0 + a * a))
        std = gain / math.sqrt(fan) if fan > 0 else 0.0
        t[...] = np.random.normal(0.0, std, size=t.shape).astype(t.dtype)
        return tensor

    @staticmethod
    def xavier_uniform_(tensor: Any, gain: float = 1.0) -> Any:
        """Glorot/Xavier 均匀初始化。"""
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        shape = tuple(t.shape)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        denom = fan_in + fan_out
        std = gain * math.sqrt(2.0 / denom) if denom > 0 else 0.0
        a = math.sqrt(3.0) * std
        t[...] = np.random.uniform(-a, a, size=t.shape).astype(t.dtype)
        return tensor

    @staticmethod
    def xavier_normal_(tensor: Any, gain: float = 1.0) -> Any:
        """Glorot/Xavier 正态初始化。"""
        t = tensor.data if isinstance(tensor, Tensor) else tensor
        shape = tuple(t.shape)
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        denom = fan_in + fan_out
        std = gain * math.sqrt(2.0 / denom) if denom > 0 else 0.0
        t[...] = np.random.normal(0.0, std, size=t.shape).astype(t.dtype)
        return tensor
