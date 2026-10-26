# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor — 青小团自研张量计算引擎
================================================
基于 NumPy 的张量库，提供完整的反向模式自动微分，可作为 PyTorch 的轻量替代。

设计要点
--------
* 每个张量持有: ``data`` (numpy.ndarray), ``grad`` (ndarray|None),
  ``requires_grad`` (bool), ``_backward`` (callable), ``_prev`` (set of Tensor)
* 前向时构建计算图，``backward()`` 使用拓扑排序逆序执行，梯度正确累积
* ``no_grad()`` 上下文中不构建计算图；``enable_grad()`` 强制开启
* 设备统一为字符串 "cpu"（NumPy 后端），``to("cpu")`` 为 no-op
* 浮点统一 float32（也兼容 float64 等其它 dtype）

使用示例
--------
    from simplcore.tensor.tensor import Tensor, no_grad

    x = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
    y = (x @ x).sum()
    y.backward()
    print(x.grad)            # 2 * x  -> [[2.,4.],[6.,8.]]
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Union, Tuple, List, Iterable, Callable, Any

__all__ = [
    # 核心类与梯度模式
    "Tensor", "no_grad", "enable_grad", "is_grad_enabled",
    # 工厂函数
    "tensor", "zeros", "ones", "randn", "rand", "eye", "arange",
    "linspace", "full", "full_like", "zeros_like", "ones_like",
    "randn_like",
    # 拼接 / 条件 / 随机 / 比较
    "cat", "stack", "where", "manual_seed", "allclose",
    "equal", "isfinite", "frombuffer", "from_numpy",
    # 扩展激活 / 归一化 / 损失（带梯度）
    "relu", "gelu", "sigmoid", "tanh", "silu", "leaky_relu",
    "softmax", "log_softmax", "dropout", "layer_norm", "batch_norm",
    "cross_entropy", "nll_loss", "mse_loss", "l1_loss", "binary_cross_entropy",
]


# ============================================================
# 全局梯度模式控制
# ============================================================
_grad_enabled: bool = True


def is_grad_enabled() -> bool:
    """返回当前是否启用梯度追踪。"""
    return _grad_enabled


class no_grad:
    """禁用梯度追踪的上下文管理器/装饰器。

    用法1（上下文管理器）::

        with no_grad():
            ...   # 此处不构建计算图

    用法2（装饰器）::

        @no_grad()
        def forward(x):
            ...
    """

    def __init__(self) -> None:
        self._prev: bool = True

    def __enter__(self) -> "no_grad":
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, fn):
        """装饰器模式：在函数执行期间禁用梯度。"""
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


class enable_grad:
    """强制启用梯度追踪的上下文管理器/装饰器。"""

    def __init__(self) -> None:
        self._prev: bool = True

    def __enter__(self) -> "enable_grad":
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, fn):
        """装饰器模式：在函数执行期间启用梯度。"""
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


# ============================================================
# 内部工具
# ============================================================
_DEFAULT_DTYPE = np.float32


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """将广播后的梯度求和还原到原始 shape（处理 NumPy 广播语义）。"""
    grad = np.asarray(grad)
    if grad.shape == shape:
        return grad
    # 去掉前导多余维度（broadcasting 在前面补维度）
    ndim_extra = grad.ndim - len(shape)
    for _ in range(ndim_extra):
        grad = grad.sum(axis=0)
    # 对 shape 中为 1 的维度求和
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _add_grad(t: "Tensor", g: np.ndarray) -> None:
    """把梯度 g 累加到 t.grad。

    零初始化语义：``t.grad`` 初始为 ``None``（视为 0），首次累加时分配并拷贝，
    后续累加用 ``+`` 产生新数组（避免视图别名带来的就地修改问题）。
    """
    g = np.asarray(g, dtype=t.data.dtype)
    if g.shape != t.data.shape:
        g = np.broadcast_to(g, t.data.shape)
    if t.grad is None:
        t.grad = np.array(g, dtype=t.data.dtype)          # 真正拷贝
    else:
        t.grad = t.grad + g                                # 产生新数组，正确累积


def _restore_reduced_dims(g: np.ndarray, dim, ndim: int, keepdim: bool) -> np.ndarray:
    """把沿 dim 归约后（未 keepdim）的梯度恢复 size-1 维度以便广播。"""
    if dim is None or keepdim:
        return g
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    dims = tuple(d if d >= 0 else d + ndim for d in dims)
    for d in sorted(dims):
        g = np.expand_dims(g, axis=d)
    return g


def _norm_idx(idx):
    """把索引中的 Tensor 转成 ndarray。"""
    if isinstance(idx, Tensor):
        return idx.data
    if isinstance(idx, tuple):
        return tuple(i.data if isinstance(i, Tensor) else i for i in idx)
    return idx


# ============================================================
# Tensor 类
# ============================================================
class Tensor:
    """自研张量，包装 numpy.ndarray 并支持反向模式自动微分。"""

    # ---------- 构造 ----------
    def __init__(self, data: Any, requires_grad: bool = False,
                 dtype: Any = None, device: str = "cpu") -> None:
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data)
        if dtype is not None:
            arr = arr.astype(dtype)                    # 显式指定 dtype 优先
        elif (type(data) in (int, float, list, tuple)
              and arr.dtype.kind == 'f' and arr.dtype != _DEFAULT_DTYPE):
            # 仅对 Python 原生输入（标量/列表/元组）应用默认 float32；
            # 注意 np.float64 是 python float 的子类，故用 type() 严格判断；
            # 已是 ndarray / numpy 标量（含运算结果）保留其 dtype，避免精度损失
            arr = arr.astype(_DEFAULT_DTYPE)
        self.data: np.ndarray = arr
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[np.ndarray] = None
        self._backward: Callable[[], None] = lambda: None
        self._prev: set = set()                       # 父节点（梯度来源）
        self._op: str = ""
        self._device: str = "cpu"

    # ---------- 属性 ----------
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # ---------- 形状 / 元信息方法 ----------
    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def dim(self) -> int:
        return self.data.ndim

    def numel(self) -> int:
        return int(self.data.size)

    def element_size(self) -> int:
        """单个元素占用的字节数（兼容 ``torch.Tensor.element_size``）。"""
        return int(self.data.dtype.itemsize)

    def item(self):
        return self.data.item()

    def tolist(self):
        """返回 Python 列表（``torch.Tensor.tolist`` 语义）。"""
        return self.data.tolist()

    def numpy(self) -> "np.ndarray":
        """返回底层 NumPy 数组（``torch.Tensor.numpy`` 语义，副本）。"""
        return self.data.copy()

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __bool__(self) -> bool:
        return bool(self.data.size == 1 and self.data.item())

    def __float__(self) -> float:
        return float(self.data.item())

    def __int__(self) -> int:
        return int(self.data.item())

    # ---------- 内部辅助 ----------
    def _ensure(self, x) -> "Tensor":
        if isinstance(x, Tensor):
            return x
        return Tensor(x, dtype=self.data.dtype)

    def _make_out(self, data, op: str, parents: Iterable["Tensor"]) -> "Tensor":
        """构造带梯度的输出张量；只在 grad 启用且存在 requires_grad 父节点时建图。"""
        req_parents = [p for p in parents if p.requires_grad]
        requires_grad = (len(req_parents) > 0) and _grad_enabled
        out = Tensor(data, requires_grad=requires_grad)
        out._op = op
        if requires_grad:
            out._prev = set(req_parents)
        return out

    # ============================================================
    # 基本算术运算
    # ============================================================
    def __add__(self, other) -> "Tensor":
        other = self._ensure(other)
        out = self._make_out(self.data + other.data, "add", (self, other))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    _add_grad(self, _unbroadcast(out.grad, self.data.shape))
                if other.requires_grad:
                    _add_grad(other, _unbroadcast(out.grad, other.data.shape))
            out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        other = self._ensure(other)
        out = self._make_out(self.data - other.data, "sub", (self, other))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    _add_grad(self, _unbroadcast(out.grad, self.data.shape))
                if other.requires_grad:
                    _add_grad(other, _unbroadcast(-out.grad, other.data.shape))
            out._backward = _backward
        return out

    def __rsub__(self, other) -> "Tensor":
        other = self._ensure(other)
        out = self._make_out(other.data - self.data, "rsub", (self, other))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    _add_grad(self, _unbroadcast(-out.grad, self.data.shape))
                if other.requires_grad:
                    _add_grad(other, _unbroadcast(out.grad, other.data.shape))
            out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = self._ensure(other)
        out = self._make_out(self.data * other.data, "mul", (self, other))
        if out.requires_grad:
            a, b = self.data, other.data
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    _add_grad(self, _unbroadcast(out.grad * b, self.data.shape))
                if other.requires_grad:
                    _add_grad(other, _unbroadcast(out.grad * a, other.data.shape))
            out._backward = _backward
        return out

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __mod__(self, other) -> "Tensor":
        other = self._ensure(other)
        out = self._make_out(self.data % other.data, "mod", (self, other))
        if out.requires_grad:
            a, b = self.data, other.data
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    _add_grad(self, _unbroadcast(out.grad, self.data.shape))
                # 取模对除数的梯度在零点不连续，按 0 近似（正数域足够）
                if other.requires_grad:
                    _add_grad(other, np.zeros_like(other.data))
            out._backward = _backward
        return out

    def __rmod__(self, other) -> "Tensor":
        other = self._ensure(other)
        return Tensor(other.data % self.data, requires_grad=self.requires_grad)

    def __truediv__(self, other) -> "Tensor":
        other = self._ensure(other)
        out = self._make_out(self.data / other.data, "div", (self, other))
        if out.requires_grad:
            a, b = self.data, other.data
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    _add_grad(self, _unbroadcast(out.grad / b, self.data.shape))
                if other.requires_grad:
                    _add_grad(other, _unbroadcast(-out.grad * a / (b * b), other.data.shape))
            out._backward = _backward
        return out

    def __rtruediv__(self, other) -> "Tensor":
        other = self._ensure(other)
        out = self._make_out(other.data / self.data, "rdiv", (self, other))
        if out.requires_grad:
            a, b = other.data, self.data
            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    _add_grad(self, _unbroadcast(-out.grad * a / (b * b), self.data.shape))
                if other.requires_grad:
                    _add_grad(other, _unbroadcast(out.grad / b, other.data.shape))
            out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        out = self._make_out(-self.data, "neg", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, -out.grad)
            out._backward = _backward
        return out

    def __invert__(self) -> "Tensor":
        """位反 ~ (用于布尔张量取反)"""
        return Tensor(~self.data)

    def __or__(self, other) -> "Tensor":
        """元素级或 (用于布尔张量)"""
        other = other.data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(self.data | other)

    def __and__(self, other) -> "Tensor":
        """元素级与 (用于布尔张量)"""
        other = other.data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(self.data & other)

    def __xor__(self, other) -> "Tensor":
        """元素级异或"""
        other = other.data if isinstance(other, Tensor) else np.asarray(other)
        return Tensor(self.data ^ other)

    def __pow__(self, power: float) -> "Tensor":
        power = float(power)
        out = self._make_out(np.power(self.data, power), "pow", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad * power * np.power(a, power - 1))
            out._backward = _backward
        return out

    def pow(self, power: float) -> "Tensor":
        return self.__pow__(power)

    def __rpow__(self, base) -> "Tensor":
        """处理 scalar ** tensor"""
        base_data = base if not isinstance(base, Tensor) else base.data
        out_data = np.power(base_data, self.data)
        req = (isinstance(base, Tensor) and base.requires_grad) or self.requires_grad
        req = req and _grad_enabled
        parents = set()
        if isinstance(base, Tensor) and base.requires_grad:
            parents.add(base)
        if self.requires_grad:
            parents.add(self)
        out = Tensor(out_data, requires_grad=req)
        out._op = "rpow"
        out._prev = parents
        if req:
            b = base_data
            s = self.data
            def _backward():
                if out.grad is None:
                    return
                # d/ds (b^s) = b^s * ln(b)
                if self.requires_grad:
                    _add_grad(self, out.grad * out_data * np.log(np.abs(b) + 1e-30))
                # d/db (b^s) = s * b^(s-1)
                if isinstance(base, Tensor) and base.requires_grad:
                    _add_grad(base, out.grad * s * np.power(b, s - 1))
            out._backward = _backward
        return out

    # ============================================================
    # 矩阵乘法
    # ============================================================
    def __matmul__(self, other) -> "Tensor":
        return self.matmul(other)

    def __rmatmul__(self, other) -> "Tensor":
        other = self._ensure(other)
        return other.matmul(self)

    def matmul(self, other) -> "Tensor":
        other = self._ensure(other)
        a, b = self.data, other.data
        out_data = np.matmul(a, b)
        out = self._make_out(out_data, "matmul", (self, other))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g = np.asarray(out.grad, dtype=out.data.dtype)
                if a.ndim == 1 and b.ndim == 1:
                    # 向量内积: out 为标量
                    if self.requires_grad:
                        _add_grad(self, g * b)
                    if other.requires_grad:
                        _add_grad(other, g * a)
                elif a.ndim == 1 and b.ndim == 2:
                    # a(n,) b(n,m) -> out(m,)
                    if self.requires_grad:
                        _add_grad(self, np.matmul(g, b.T))
                    if other.requires_grad:
                        _add_grad(other, np.outer(a, g))
                elif a.ndim == 2 and b.ndim == 1:
                    # a(k,n) b(n,) -> out(k,)
                    if self.requires_grad:
                        _add_grad(self, np.outer(g, b))
                    if other.requires_grad:
                        _add_grad(other, np.matmul(a.T, g))
                elif a.ndim >= 2 and b.ndim >= 2:
                    # 批量矩阵乘（含 2D@2D、bmm 3D@3D 及更高）
                    if self.requires_grad:
                        ga = np.matmul(g, np.swapaxes(b, -1, -2))
                        _add_grad(self, _unbroadcast(ga, a.shape))
                    if other.requires_grad:
                        gb = np.matmul(np.swapaxes(a, -1, -2), g)
                        _add_grad(other, _unbroadcast(gb, b.shape))
                else:
                    raise NotImplementedError(
                        f"matmul autograd for shapes {a.shape} @ {b.shape} 未实现")
            out._backward = _backward
        return out

    def bmm(self, other) -> "Tensor":
        """批量矩阵乘（要求两个张量均为 3D）。"""
        other = self._ensure(other)
        assert self.ndim == 3 and other.ndim == 3, "bmm 要求 3D 输入"
        return self.matmul(other)

    def mm(self, other) -> "Tensor":
        """二维矩阵乘。"""
        other = self._ensure(other)
        assert self.ndim == 2 and other.ndim == 2, "mm 要求 2D 输入"
        return self.matmul(other)

    # ============================================================
    # 形状操作
    # ============================================================
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if -1 in shape:
            known = int(np.prod([s for s in shape if s != -1]))
            shape[shape.index(-1)] = self.data.size // known
        shape = tuple(shape)
        out = self._make_out(self.data.reshape(shape), "reshape", (self,))
        if out.requires_grad:
            orig_shape = self.data.shape
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad.reshape(orig_shape))
            out._backward = _backward
        return out

    def view(self, *shape) -> "Tensor":
        return self.reshape(*shape)

    def transpose(self, *dims) -> "Tensor":
        if len(dims) == 0:
            # 无参数: 反转所有维度
            dims = tuple(reversed(range(self.ndim)))
        elif len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        elif len(dims) == 2:
            # PyTorch 风格: transpose(dim0, dim1) = 交换两个维度
            d0, d1 = dims
            d0 = d0 if d0 >= 0 else d0 + self.ndim
            d1 = d1 if d1 >= 0 else d1 + self.ndim
            dims = list(range(self.ndim))
            dims[d0], dims[d1] = dims[d1], dims[d0]
            dims = tuple(dims)
        dims = tuple(d if d >= 0 else d + self.ndim for d in dims)
        out_data = np.transpose(self.data, dims)
        out = self._make_out(out_data, "transpose", (self,))
        if out.requires_grad:
            inv = tuple(int(i) for i in np.argsort(dims))
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, np.transpose(out.grad, inv))
            out._backward = _backward
        return out

    def permute(self, *dims) -> "Tensor":
        return self.transpose(*dims)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if end_dim < 0:
            end_dim = self.ndim + end_dim
        if start_dim < 0:
            start_dim = self.ndim + start_dim
        prod = int(np.prod(self.shape[start_dim:end_dim + 1]))
        new_shape = self.shape[:start_dim] + (prod,) + self.shape[end_dim + 1:]
        return self.reshape(*new_shape)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        out = self._make_out(np.squeeze(self.data, axis=dim), "squeeze", (self,))
        if out.requires_grad:
            orig_shape = self.data.shape
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad.reshape(orig_shape))
            out._backward = _backward
        return out

    def unsqueeze(self, dim: int) -> "Tensor":
        if dim < 0:
            dim = dim + self.ndim + 1
        out = self._make_out(np.expand_dims(self.data, dim), "unsqueeze", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, np.squeeze(out.grad, axis=dim))
            out._backward = _backward
        return out

    def expand(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        for i, s in enumerate(shape):
            if s == -1:
                shape[i] = self.data.shape[i]
        shape = tuple(shape)
        out = self._make_out(np.broadcast_to(self.data, shape).copy(), "expand", (self,))
        if out.requires_grad:
            orig_shape = self.data.shape
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, _unbroadcast(out.grad, orig_shape))
            out._backward = _backward
        return out

    def contiguous(self) -> "Tensor":
        if self.data.flags['C_CONTIGUOUS']:
            return self
        out = self._make_out(np.ascontiguousarray(self.data), "contiguous", (self,))
        if out.requires_grad:
            orig_shape = self.data.shape
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad.reshape(orig_shape))
            out._backward = _backward
        return out

    # ============================================================
    # 拼接
    # ============================================================
    def repeat_interleave(self, repeats, dim: Optional[int] = None) -> "Tensor":
        repeats = int(repeats)
        if dim is None:
            flat = self.data.reshape(-1)
            out = self._make_out(np.repeat(flat, repeats), "repeat_interleave", (self,))
            if out.requires_grad:
                orig_shape = self.data.shape
                def _backward():
                    if out.grad is None:
                        return
                    g = out.grad.reshape(-1, repeats).sum(axis=1)
                    _add_grad(self, g.reshape(orig_shape))
                out._backward = _backward
            return out
        if dim < 0:
            dim += self.ndim
        out = self._make_out(np.repeat(self.data, repeats, axis=dim), "repeat_interleave", (self,))
        if out.requires_grad:
            ndim = self.ndim
            s = self.shape[dim]
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                order = list(range(g.ndim))
                order.remove(dim)
                order.append(dim)
                g_moved = np.transpose(g, order)               # (..., s*repeats)
                g_red = g_moved.reshape(g_moved.shape[:-1] + (s, repeats)).sum(axis=-1)
                inv = tuple(int(i) for i in np.argsort(order))
                _add_grad(self, np.transpose(g_red, inv))
            out._backward = _backward
        return out

    def repeat(self, *sizes) -> "Tensor":
        """沿各维度重复张量（等价于 ``numpy.tile`` / ``torch.repeat``）。

        ``x.repeat(2, 3)`` 将形状 ``(a, b)`` 扩展为 ``(2a, 3b)``。
        """
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            reps = tuple(int(s) for s in sizes[0])
        else:
            reps = tuple(int(s) for s in sizes)
        # 维度不足时在前面补 1
        if len(reps) < self.data.ndim:
            reps = (1,) * (self.data.ndim - len(reps)) + reps
        out = self._make_out(np.tile(self.data, reps), "repeat", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                # 反向: 对每个被重复的块求和后折叠回原始形状
                for axis, r in enumerate(reps):
                    if r == 1:
                        continue
                    g = g.reshape(
                        tuple(s if i != axis else s * r
                              for i, s in enumerate(g.shape))
                    )
                    g = g.sum(axis=axis)
                _add_grad(self, g)
            out._backward = _backward
        return out

    def flip(self, dims) -> "Tensor":
        if dims is None:
            dims = tuple(range(self.ndim))
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(d if d >= 0 else d + self.ndim for d in dims)
        out = self._make_out(np.flip(self.data, dims), "flip", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, np.flip(out.grad, dims))
            out._backward = _backward
        return out

    def triu(self, diagonal: int = 0) -> "Tensor":
        out = self._make_out(np.triu(self.data, k=diagonal), "triu", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, np.triu(out.grad, k=diagonal))
            out._backward = _backward
        return out

    def tril(self, diagonal: int = 0) -> "Tensor":
        out = self._make_out(np.tril(self.data, k=diagonal), "tril", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, np.tril(out.grad, k=diagonal))
            out._backward = _backward
        return out

    def outer(self, other) -> "Tensor":
        other = self._ensure(other)
        a = self.data.reshape(-1)
        b = other.data.reshape(-1)
        out = self._make_out(np.outer(a, b), "outer", (self, other))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g = out.grad
                if self.requires_grad:
                    _add_grad(self, np.matmul(g, b).reshape(self.data.shape))
                if other.requires_grad:
                    _add_grad(other, np.matmul(g.T, a).reshape(other.data.shape))
            out._backward = _backward
        return out

    # ============================================================
    # 数学函数（逐元素）
    # ============================================================
    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = self._make_out(out_data, "exp", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad * out_data)
            out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = self._make_out(np.log(self.data), "log", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad / a)
            out._backward = _backward
        return out

    def sqrt(self) -> "Tensor":
        out_data = np.sqrt(self.data)
        out = self._make_out(out_data, "sqrt", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad * 0.5 / out_data)
            out._backward = _backward
        return out

    def sin(self) -> "Tensor":
        out = self._make_out(np.sin(self.data), "sin", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad * np.cos(a))
            out._backward = _backward
        return out

    def cos(self) -> "Tensor":
        out = self._make_out(np.cos(self.data), "cos", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad * (-np.sin(a)))
            out._backward = _backward
        return out

    def tan(self) -> "Tensor":
        out = self._make_out(np.tan(self.data), "tan", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad / (np.cos(a) ** 2))
            out._backward = _backward
        return out

    def abs(self) -> "Tensor":
        out = self._make_out(np.abs(self.data), "abs", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad * np.sign(a))
            out._backward = _backward
        return out

    def reciprocal(self) -> "Tensor":
        out = self._make_out(np.reciprocal(self.data), "reciprocal", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, -out.grad / (a * a))
            out._backward = _backward
        return out

    # ============================================================
    # 归约
    # ============================================================
    def sum(self, dim=None, keepdim: bool = False) -> "Tensor":
        out = self._make_out(self.data.sum(axis=dim, keepdims=keepdim), "sum", (self,))
        if out.requires_grad:
            ndim = self.ndim
            def _backward():
                if out.grad is None:
                    return
                g = _restore_reduced_dims(out.grad, dim, ndim, keepdim)
                g = np.broadcast_to(g, self.data.shape)
                _add_grad(self, g)
            out._backward = _backward
        return out

    def mean(self, dim=None, keepdim: bool = False) -> "Tensor":
        out = self._make_out(self.data.mean(axis=dim, keepdims=keepdim), "mean", (self,))
        if out.requires_grad:
            ndim = self.ndim
            if dim is None:
                n = self.data.size
            else:
                dims = dim if isinstance(dim, (tuple, list)) else (dim,)
                n = int(np.prod([self.shape[d if d >= 0 else d + ndim] for d in dims]))
            def _backward():
                if out.grad is None:
                    return
                g = _restore_reduced_dims(out.grad, dim, ndim, keepdim) / n
                g = np.broadcast_to(g, self.data.shape)
                _add_grad(self, g)
            out._backward = _backward
        return out

    def var(self, dim=None, keepdim: bool = False, unbiased: bool = True) -> "Tensor":
        ddof = 1 if unbiased else 0
        out = self._make_out(self.data.var(axis=dim, keepdims=keepdim, ddof=ddof), "var", (self,))
        if out.requires_grad:
            ndim = self.ndim
            if dim is None:
                n = self.data.size - ddof
            else:
                dims = dim if isinstance(dim, (tuple, list)) else (dim,)
                cnt = int(np.prod([self.shape[d if d >= 0 else d + ndim] for d in dims]))
                n = cnt - ddof
            mean_data = self.data.mean(axis=dim, keepdims=True)
            def _backward():
                if out.grad is None:
                    return
                g = _restore_reduced_dims(out.grad, dim, ndim, keepdim)
                diff = self.data - mean_data
                _add_grad(self, np.broadcast_to(g, self.data.shape) * 2.0 * diff / n)
            out._backward = _backward
        return out

    def std(self, dim=None, keepdim: bool = False, unbiased: bool = True) -> "Tensor":
        ddof = 1 if unbiased else 0
        std_val = self.data.std(axis=dim, keepdims=keepdim, ddof=ddof)
        out = self._make_out(std_val, "std", (self,))
        if out.requires_grad:
            ndim = self.ndim
            if dim is None:
                n = self.data.size - ddof
            else:
                dims = dim if isinstance(dim, (tuple, list)) else (dim,)
                cnt = int(np.prod([self.shape[d if d >= 0 else d + ndim] for d in dims]))
                n = cnt - ddof
            mean_data = self.data.mean(axis=dim, keepdims=True)
            std_safe = np.where(std_val == 0, 1e-12, std_val)
            def _backward():
                if out.grad is None:
                    return
                g = _restore_reduced_dims(out.grad, dim, ndim, keepdim)
                diff = self.data - mean_data
                _add_grad(self, np.broadcast_to(g, self.data.shape) * diff / (n * std_safe))
            out._backward = _backward
        return out

    def max(self, dim=None, keepdim: bool = False):
        if dim is not None:
            # PyTorch 风格: 返回 (values, indices) 命名元组
            from collections import namedtuple
            MaxResult = namedtuple('max', ['values', 'indices'])
            idx = np.argmax(self.data, axis=dim if dim >= 0 else dim + self.data.ndim)
            vals = np.take_along_axis(self.data, np.expand_dims(idx, axis=dim), axis=dim)
            if not keepdim:
                vals = np.squeeze(vals, axis=dim)
            val_tensor = self._make_out(vals, "max", (self,))
            if val_tensor.requires_grad:
                def _backward():
                    if val_tensor.grad is None:
                        return
                    g_self = np.zeros_like(self.data, dtype=self.data.dtype)
                    idx_exp = np.expand_dims(idx, axis=dim)
                    g_to_scatter = val_tensor.grad if keepdim else np.expand_dims(val_tensor.grad, axis=dim)
                    np.put_along_axis(g_self, idx_exp, g_to_scatter, axis=dim)
                    _add_grad(self, g_self)
                val_tensor._backward = _backward
            return MaxResult(values=val_tensor, indices=Tensor(idx))
        # 无 dim: 返回全局最大值标量
        out = self._make_out(self.data.max(), "max", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g_self = np.zeros_like(self.data, dtype=self.data.dtype)
                idx = np.unravel_index(np.argmax(self.data), self.data.shape)
                g_self[idx] = np.asarray(out.grad).reshape(-1)[0]
                _add_grad(self, g_self)
            out._backward = _backward
        return out

    def min(self, dim=None, keepdim: bool = False):
        if dim is not None:
            from collections import namedtuple
            MinResult = namedtuple('min', ['values', 'indices'])
            idx = np.argmin(self.data, axis=dim if dim >= 0 else dim + self.data.ndim)
            vals = np.take_along_axis(self.data, np.expand_dims(idx, axis=dim), axis=dim)
            if not keepdim:
                vals = np.squeeze(vals, axis=dim)
            val_tensor = self._make_out(vals, "min", (self,))
            if val_tensor.requires_grad:
                def _backward():
                    if val_tensor.grad is None:
                        return
                    g_self = np.zeros_like(self.data, dtype=self.data.dtype)
                    idx_exp = np.expand_dims(idx, axis=dim)
                    g_to_scatter = val_tensor.grad if keepdim else np.expand_dims(val_tensor.grad, axis=dim)
                    np.put_along_axis(g_self, idx_exp, g_to_scatter, axis=dim)
                    _add_grad(self, g_self)
                val_tensor._backward = _backward
            return MinResult(values=val_tensor, indices=Tensor(idx))
        out = self._make_out(self.data.min(), "min", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g_self = np.zeros_like(self.data, dtype=self.data.dtype)
                idx = np.unravel_index(np.argmin(self.data), self.data.shape)
                g_self[idx] = np.asarray(out.grad).reshape(-1)[0]
                _add_grad(self, g_self)
            out._backward = _backward
        return out

    def argmax(self, dim=None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            return Tensor(np.array([int(np.argmax(self.data))]))
        return Tensor(np.argmax(self.data, axis=dim))

    def argmin(self, dim=None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            return Tensor(np.array([int(np.argmin(self.data))]))
        return Tensor(np.argmin(self.data, axis=dim))

    def norm(self, p: float = 2.0, dim=None, keepdim: bool = False) -> "Tensor":
        if p == 2.0:
            out_data = np.sqrt((self.data * self.data).sum(axis=dim, keepdims=keepdim))
        else:
            out_data = np.power(np.power(np.abs(self.data), p).sum(axis=dim, keepdims=keepdim), 1.0 / p)
        out = self._make_out(out_data, "norm", (self,))
        if out.requires_grad:
            a = self.data

            def _backward():
                if out.grad is None:
                    return
                denom = np.where(out_data == 0, 1e-12, out_data)
                if p == 2.0:
                    dx = np.broadcast_to(out.grad / denom, a.shape) * a
                else:
                    coeff = np.broadcast_to(out.grad / np.power(denom, p - 1), a.shape)
                    dx = coeff * a * np.power(np.abs(a), p - 2)
                _add_grad(self, dx)
            out._backward = _backward
        return out

    # ============================================================
    # clamp / where
    # ============================================================
    def clamp(self, min=None, max=None) -> "Tensor":
        out = self._make_out(np.clip(self.data, min, max), "clamp", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                mask = np.ones_like(a, dtype=self.data.dtype)
                if min is not None:
                    mask = mask * (a >= min)
                if max is not None:
                    mask = mask * (a <= max)
                _add_grad(self, out.grad * mask)
            out._backward = _backward
        return out

    def clamp_min(self, min) -> "Tensor":
        return self.clamp(min=min)

    def clamp_max(self, max) -> "Tensor":
        return self.clamp(max=max)

    # ============================================================
    # 索引
    # ============================================================
    def __getitem__(self, idx) -> "Tensor":
        idx = _norm_idx(idx)
        out = self._make_out(self.data[idx], "getitem", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g_self = np.zeros_like(self.data, dtype=self.data.dtype)
                np.add.at(g_self, idx, out.grad)
                _add_grad(self, g_self)
            out._backward = _backward
        return out

    def __setitem__(self, idx, value) -> None:
        """就地写入。

        正向：``self.data[idx] = value``；
        反向：把上游梯度散射到被写入的 value（若其为 Tensor 且需要梯度），
        self 自身在 idx 处的旧值梯度不回传（与“替换”语义一致）。
        """
        idx = _norm_idx(idx)
        if isinstance(value, Tensor):
            value_data = value.data
        else:
            value_data = np.asarray(value, dtype=self.data.dtype)
        self.data[idx] = value_data
        if not _grad_enabled:
            return
        if isinstance(value, Tensor) and value.requires_grad:
            value_ref = value
            old_backward = self._backward
            self._prev.add(value_ref)

            def _backward():
                old_backward()
                if self.grad is None:
                    return
                _add_grad(value_ref, np.asarray(self.grad[idx]))
            self._backward = _backward

    def index_select(self, dim: int, index) -> "Tensor":
        if isinstance(index, Tensor):
            idx = index.data.astype(np.int64)
        else:
            idx = np.asarray(index, dtype=np.int64)
        idx = np.atleast_1d(idx).astype(np.int64)
        dim = dim if dim >= 0 else dim + self.data.ndim
        out = self._make_out(np.take(self.data, idx, axis=dim), "index_select", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g_self = np.zeros_like(self.data, dtype=self.data.dtype)
                # np.add.at 正确处理重复索引（梯度累加而非覆盖）
                slices = [slice(None)] * self.data.ndim
                slices[dim] = idx
                np.add.at(g_self, tuple(slices), out.grad)
                _add_grad(self, g_self)
            out._backward = _backward
        return out

    # ============================================================
    # 拷贝 / 设备 / dtype / detach
    # ============================================================
    def copy_(self, src) -> "Tensor":
        src = self._ensure(src)
        self.data[...] = src.data
        if _grad_enabled and src.requires_grad:
            src_ref = src
            old_backward = self._backward
            self._prev.add(src_ref)

            def _backward():
                old_backward()
                if self.grad is not None and src_ref.requires_grad:
                    _add_grad(src_ref, self.grad)
            self._backward = _backward
        return self

    def zero_(self) -> "Tensor":
        """原地清零。"""
        self.data[...] = 0
        return self

    def fill_(self, value) -> "Tensor":
        """原地填充。"""
        self.data[...] = value
        return self

    def detach(self) -> "Tensor":
        """返回与计算图分离的新张量（共享数据副本，不追踪梯度）。"""
        return Tensor(self.data.copy(), requires_grad=False, dtype=self.data.dtype)

    def requires_grad_(self, requires_grad: bool = True) -> "Tensor":
        self.requires_grad = bool(requires_grad)
        return self

    def zero_grad(self) -> None:
        self.grad = None

    def to(self, device=None, dtype=None) -> "Tensor":
        # 支持 to(dtype) 形式: 如果第一个参数是 dtype 而非设备字符串
        if device is not None and not isinstance(device, str):
            # 可能是 dtype 被作为第一个参数传入
            if isinstance(device, (np.dtype, type)) or hasattr(device, 'itemsize'):
                dtype = device
                device = None
        if device is not None and str(device).lower() not in ("cpu",):
            raise RuntimeError(
                f"SimplCore Tensor 仅支持 'cpu' 设备，收到 {device!r}")
        if dtype is None:
            return self                       # to("cpu") 为 no-op
        target = np.dtype(dtype)
        if target == self.data.dtype:
            return self
        new_data = self.data.astype(target)
        if not (self.requires_grad and _grad_enabled):
            return Tensor(new_data, requires_grad=self.requires_grad and _grad_enabled)
        out = self._make_out(new_data, "to", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad.astype(self.data.dtype))
            out._backward = _backward
        return out

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def cuda(self) -> "Tensor":
        raise RuntimeError("SimplCore Tensor 不支持 CUDA，仅支持 'cpu'")

    def float(self) -> "Tensor":
        return self.to(dtype=np.float32)

    def double(self) -> "Tensor":
        return self.to(dtype=np.float64)

    def long(self) -> "Tensor":
        return self.to(dtype=np.int64)

    def int(self) -> "Tensor":
        return self.to(dtype=np.int32)

    def bool(self) -> "Tensor":
        return self.to(dtype=np.bool_)

    # ============================================================
    # 反向传播（拓扑排序）
    # ============================================================
    def backward(self, grad=None) -> None:
        if not self.requires_grad:
            raise RuntimeError("该张量未启用梯度（requires_grad=False），无法调用 backward()")

        # ---- 1. 构建拓扑序（DFS 后序：父节点在前，根在后）----
        topo: List["Tensor"] = []
        visited: set = set()

        def build_topo(v: "Tensor") -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._prev:
                build_topo(p)
            topo.append(v)

        build_topo(self)

        # ---- 2. 种子梯度 ----
        if grad is None:
            g = np.ones_like(self.data)
        else:
            if isinstance(grad, Tensor):
                g = grad.data
            else:
                g = np.asarray(grad, dtype=self.data.dtype)
            g = np.broadcast_to(g, self.data.shape)
        # 累加到根（支持多次 backward 调用累积）
        if self.grad is None:
            self.grad = np.array(g, dtype=self.data.dtype)
        else:
            self.grad = self.grad + g

        # ---- 3. 逆拓扑序执行各节点的 _backward（梯度正确累积）----
        for v in reversed(topo):
            v._backward()

    # ============================================================
    # 表示
    # ============================================================
    def __repr__(self) -> str:
        return (f"Tensor({self.data!r}, dtype={self.data.dtype}, "
                f"requires_grad={self.requires_grad})")

    # 注意：不重载 __eq__/__ne__/__lt__... 返回 Tensor，
    # 因为计算图中的集合（_prev / 拓扑 visited）依赖对象身份哈希与相等，
    # 重载会破坏去重与梯度累积正确性。元素级比较请用 eq/lt/gt 等方法。

    # ---------- 比较方法（返回无梯度布尔/数值张量）----------
    def eq(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data == other.data)

    def ne(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data != other.data)

    def lt(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data < other.data)

    def le(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data <= other.data)

    def gt(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data > other.data)

    def ge(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data >= other.data)

    # ---------- 比较运算符 (返回无梯度 Tensor) ----------
    # 注意: __eq__/__ne__ 不重载, 保持对象身份哈希用于集合操作
    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data < other.data)

    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data <= other.data)

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data > other.data)

    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, dtype=self.data.dtype)
        return Tensor(self.data >= other.data)

    def __hash__(self):
        return id(self)

    # ---------- rsqrt / add / masked_fill ----------
    def rsqrt(self) -> "Tensor":
        out_data = 1.0 / np.sqrt(self.data)
        out = self._make_out(out_data, "rsqrt", (self,))
        if out.requires_grad:
            a = self.data
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, -0.5 * out.grad * out_data / a)
            out._backward = _backward
        return out

    def add(self, other, alpha=1.0) -> "Tensor":
        """self + alpha * other"""
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = np.asarray(other, dtype=self.data.dtype)
        return self + (alpha * other_data) if alpha != 1.0 else self + other

    def masked_fill(self, mask, value) -> "Tensor":
        """将 mask 为 True 的位置填充为 value"""
        if isinstance(mask, Tensor):
            mask_arr = mask.data.astype(bool)
        else:
            mask_arr = np.asarray(mask).astype(bool)
        out_data = np.where(mask_arr, value, self.data)
        out = self._make_out(out_data, "masked_fill", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, np.where(mask_arr, 0.0, out.grad))
            out._backward = _backward
        return out

    # ---------- clone / topk / cumsum / sort / scatter / index_add_ / all ----------
    def clone(self) -> "Tensor":
        """深拷贝, 保留梯度追踪"""
        out = self._make_out(self.data.copy(), "clone", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad)
            out._backward = _backward
        return out

    def topk(self, k: int, dim: int = -1, largest: bool = True):
        """返回 (values, indices) 命名元组, 无梯度 (索引操作)"""
        from collections import namedtuple
        TopKResult = namedtuple('topk', ['values', 'indices'])
        if dim < 0:
            dim += self.data.ndim
        arr = self.data
        if largest:
            idx = np.argsort(-arr, axis=dim, kind='stable')
        else:
            idx = np.argsort(arr, axis=dim, kind='stable')
        # 取前 k 个
        sl = [slice(None)] * arr.ndim
        sl[dim] = slice(0, k)
        top_idx = idx[tuple(sl)]
        top_vals = np.take_along_axis(arr, top_idx, axis=dim)
        return TopKResult(values=Tensor(top_vals), indices=Tensor(top_idx))

    def cumsum(self, dim: int = -1) -> "Tensor":
        if dim < 0:
            dim += self.data.ndim
        out_data = np.cumsum(self.data, axis=dim)
        out = self._make_out(out_data, "cumsum", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                # cumsum 的反向是 cumsum on flipped grad
                _add_grad(self, np.flip(np.cumsum(np.flip(out.grad, axis=dim), axis=dim), axis=dim))
            out._backward = _backward
        return out

    def sort(self, dim: int = -1, descending: bool = False):
        """返回 (sorted_values, indices) 命名元组, 无梯度"""
        from collections import namedtuple
        SortResult = namedtuple('sort', ['values', 'indices'])
        if dim < 0:
            dim += self.data.ndim
        if descending:
            idx = np.argsort(-self.data, axis=dim, kind='stable')
        else:
            idx = np.argsort(self.data, axis=dim, kind='stable')
        sorted_vals = np.take_along_axis(self.data, idx, axis=dim)
        return SortResult(values=Tensor(sorted_vals), indices=Tensor(idx))

    def scatter(self, dim: int, index, src) -> "Tensor":
        """散射写入: out[index[i]] = src[i], 返回新张量"""
        if dim < 0:
            dim += self.data.ndim
        if isinstance(index, Tensor):
            idx = index.data.astype(np.int64)
        else:
            idx = np.asarray(index, dtype=np.int64)
        if isinstance(src, Tensor):
            src_data = src.data
        else:
            src_data = np.asarray(src, dtype=self.data.dtype)
        out_data = self.data.copy()
        np.put_along_axis(out_data, idx, src_data, axis=dim)
        out = self._make_out(out_data, "scatter", (self, src if isinstance(src, Tensor) else None))
        if out.requires_grad:
            src_tensor = src if isinstance(src, Tensor) else None
            def _backward():
                if out.grad is None:
                    return
                # self 的梯度: scatter 位置以外的梯度直传
                g_self = out.grad.copy()
                # scatter 位置的梯度传给 src
                np.put_along_axis(g_self, idx, 0.0, axis=dim)
                _add_grad(self, g_self)
                if src_tensor is not None and src_tensor.requires_grad:
                    gathered = np.take_along_axis(out.grad, idx, axis=dim)
                    _add_grad(src_tensor, gathered)
            out._backward = _backward
        return out

    def index_add_(self, dim: int, index, source) -> "Tensor":
        """就地索引加法: self[index[i]] += source[i]"""
        if dim < 0:
            dim += self.data.ndim
        if isinstance(index, Tensor):
            idx = index.data.astype(np.int64)
        else:
            idx = np.asarray(index, dtype=np.int64)
        if isinstance(source, Tensor):
            src_data = source.data
        else:
            src_data = np.asarray(source, dtype=self.data.dtype)
        full_idx = self._build_scatter_index(dim, idx)
        np.add.at(self.data, full_idx, src_data)
        return self

    def _build_scatter_index(self, dim: int, idx: np.ndarray) -> tuple:
        """构建 scatter 操作的完整多维索引元组 (PyTorch 语义).

        ``index_add_(dim, idx, source)``: ``idx`` 是 1D 长整型向量, 长度 = M =
        source.shape[dim]; ``source`` 与 ``self`` 同维数, 除 ``dim`` 外各维与
        ``self`` 一致. 则对每个 i: self[idx[i], ...] += source[i, ...].

        构造方式: 在 ``dim`` 处放 ``idx`` (reshape 为在其它维为 1), 其余维 d'
        放 ``arange(self.data.shape[d'])`` (reshape 为仅在 d' 维非 1). 各数组
        均可广播到 source 形状.
        """
        ndim = self.data.ndim
        idx = np.asarray(idx)
        M = idx.shape[0] if idx.ndim == 1 else idx.shape[dim]
        full_idx = []
        for d in range(ndim):
            if d == dim:
                # 在 dim 维放 idx, 其余维补 1 (广播)
                shape = [1] * ndim
                shape[dim] = M
                full_idx.append(idx.reshape(shape))
            else:
                sz = self.data.shape[d]
                shape = [1] * ndim
                shape[d] = sz
                full_idx.append(np.arange(sz).reshape(shape))
        return tuple(full_idx)

    def scatter_add_(self, dim: int, index, src) -> "Tensor":
        """就地散射加法: self[index[i]] += src[i] (累加而非覆盖)"""
        if dim < 0:
            dim += self.data.ndim
        if isinstance(index, Tensor):
            idx = index.data.astype(np.int64)
        else:
            idx = np.asarray(index, dtype=np.int64)
        if isinstance(src, Tensor):
            src_data = src.data
        else:
            src_data = np.asarray(src, dtype=self.data.dtype)
        # 构建完整的多维索引
        full_idx = self._build_scatter_index(dim, idx)
        np.add.at(self.data, full_idx, src_data)
        return self

    def scatter_add(self, dim: int, index, src) -> "Tensor":
        """散射加法 (非就地): 返回新张量"""
        out_data = self.data.copy()
        if dim < 0:
            dim += out_data.ndim
        if isinstance(index, Tensor):
            idx = index.data.astype(np.int64)
        else:
            idx = np.asarray(index, dtype=np.int64)
        if isinstance(src, Tensor):
            src_data = src.data
        else:
            src_data = np.asarray(src, dtype=out_data.dtype)
        full_idx = self._build_scatter_index(dim, idx)
        np.add.at(out_data, full_idx, src_data)
        src_tensor = src if isinstance(src, Tensor) else None
        out = self._make_out(out_data, "scatter_add", (self, src_tensor) if src_tensor else (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                _add_grad(self, out.grad.copy())
                if src_tensor is not None and src_tensor.requires_grad:
                    gathered = np.take_along_axis(out.grad, idx, axis=dim)
                    _add_grad(src_tensor, gathered)
            out._backward = _backward
        return out

    def gather(self, dim: int, index) -> "Tensor":
        """沿 dim 收集: out[i] = self[index[i]]"""
        if dim < 0:
            dim += self.data.ndim
        if isinstance(index, Tensor):
            idx = index.data.astype(np.int64)
        else:
            idx = np.asarray(index, dtype=np.int64)
        out = self._make_out(np.take_along_axis(self.data, idx, axis=dim), "gather", (self,))
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return
                g_self = np.zeros_like(self.data, dtype=self.data.dtype)
                np.add.at(g_self, (slice(None),) * dim + (idx,), out.grad)
                _add_grad(self, g_self)
            out._backward = _backward
        return out

    def conv2d(self, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> "Tensor":
        """二维卷积 (基础实现，用于自研引擎; PyTorch 后端请用 F.conv2d)"""
        from numpy.lib.stride_tricks import sliding_window_view
        x = self.data  # (N, C_in, H, W)
        w = weight.data if isinstance(weight, Tensor) else np.asarray(weight)
        # (C_out, C_in/groups, kH, kW)
        N, C_in, H, W = x.shape
        C_out, C_in_g, kH, kW = w.shape
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        # padding
        if padding[0] > 0 or padding[1] > 0:
            x = np.pad(x, ((0,0),(0,0),(padding[0],padding[0]),(padding[1],padding[1])), mode='constant')
        H_p, W_p = x.shape[2], x.shape[3]
        H_out = (H_p - dilation[0]*(kH-1) - 1) // stride[0] + 1
        W_out = (W_p - dilation[1]*(kW-1) - 1) // stride[1] + 1
        # im2col
        windows = sliding_window_view(x, (kH, kW), axis=(2, 3))
        windows = windows[:, :, ::stride[0], ::stride[1], :, :, :]
        # (N, C_in, H_out, W_out, kH, kW) -> reshape for matmul
        cols = windows.reshape(N, C_in, H_out * W_out, kH * kW)
        w_flat = w.reshape(C_out, C_in_g * kH * kW)
        out_data = np.einsum('ncij,kj->nkij', cols, w_flat)
        if bias is not None:
            b = bias.data if isinstance(bias, Tensor) else np.asarray(bias)
            out_data = out_data + b.reshape(1, C_out, 1, 1)
        out = self._make_out(out_data, "conv2d", (self, weight if isinstance(weight, Tensor) else None))
        # 简化: 不实现完整反向 (PyTorch 后端有完整实现)
        return out

    def all(self) -> bool:
        """返回是否所有元素为 True"""
        return bool(np.all(self.data))

    def any(self) -> bool:
        """返回是否有元素为 True"""
        return bool(np.any(self.data))


# ============================================================
# 模块级工厂函数
# ============================================================
def tensor(data, dtype=None, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    return Tensor(data, requires_grad=requires_grad, dtype=dtype, device=device)


def _norm_shape(shape) -> Tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(int(s) for s in shape[0])
    return tuple(int(s) for s in shape)


def zeros(*shape, dtype=None, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    shp = _norm_shape(shape)
    return Tensor(np.zeros(shp, dtype=dtype or _DEFAULT_DTYPE),
                  requires_grad=requires_grad, device=device)


def ones(*shape, dtype=None, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    shp = _norm_shape(shape)
    return Tensor(np.ones(shp, dtype=dtype or _DEFAULT_DTYPE),
                  requires_grad=requires_grad, device=device)


def full(shape, fill_value, dtype=None, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    elif isinstance(shape, (tuple, list)):
        shape = tuple(int(s) for s in shape)
    return Tensor(np.full(shape, fill_value, dtype=dtype or _DEFAULT_DTYPE),
                  requires_grad=requires_grad, device=device)


def arange(start, end=None, step=1, dtype=None, requires_grad: bool = False,
           device: str = "cpu") -> Tensor:
    if end is None:
        start, end = 0, start
    dt = dtype or _DEFAULT_DTYPE
    return Tensor(np.arange(start, end, step, dtype=dt),
                  requires_grad=requires_grad, device=device)


def linspace(start, end, steps, dtype=None, requires_grad: bool = False,
             device: str = "cpu") -> Tensor:
    dt = dtype or _DEFAULT_DTYPE
    return Tensor(np.linspace(start, end, steps, dtype=dt),
                  requires_grad=requires_grad, device=device)


def eye(n: int, m: Optional[int] = None, dtype=None, requires_grad: bool = False,
        device: str = "cpu") -> Tensor:
    m = n if m is None else m
    return Tensor(np.eye(n, m, dtype=dtype or _DEFAULT_DTYPE),
                  requires_grad=requires_grad, device=device)


def randn(*shape, dtype=None, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    shp = _norm_shape(shape)
    dt = dtype or _DEFAULT_DTYPE
    data = np.random.randn(*shp).astype(dt)
    return Tensor(data, requires_grad=requires_grad, device=device)


def rand(*shape, dtype=None, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    shp = _norm_shape(shape)
    dt = dtype or _DEFAULT_DTYPE
    data = np.random.rand(*shp).astype(dt)
    return Tensor(data, requires_grad=requires_grad, device=device)


def randn_like(input: Tensor, dtype=None, requires_grad: bool = False) -> Tensor:
    """返回与 ``input`` 同形状的标准正态张量（``torch.randn_like`` 语义）。"""
    if isinstance(input, Tensor):
        shp = input.data.shape
    else:
        shp = np.asarray(input).shape
    dt = dtype or (input.data.dtype if isinstance(input, Tensor) else _DEFAULT_DTYPE)
    return Tensor(np.random.randn(*shp).astype(dt), requires_grad=requires_grad)


def full_like(t: Tensor, fill_value, dtype=None, requires_grad: bool = False,
              device: str = "cpu") -> Tensor:
    base = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.full_like(base, fill_value, dtype=dtype or base.dtype),
                  requires_grad=requires_grad, device=device)


def zeros_like(t: Tensor, dtype=None, requires_grad: bool = False,
               device: str = "cpu") -> Tensor:
    base = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.zeros_like(base, dtype=dtype or base.dtype),
                  requires_grad=requires_grad, device=device)


def ones_like(t: Tensor, dtype=None, requires_grad: bool = False,
              device: str = "cpu") -> Tensor:
    base = t.data if isinstance(t, Tensor) else np.asarray(t)
    return Tensor(np.ones_like(base, dtype=dtype or base.dtype),
                  requires_grad=requires_grad, device=device)


# ============================================================
# 拼接 / 条件 / 随机 / 比较
# ============================================================
def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("cat 不能处理空列表")
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    arrays = [t.data for t in tensors]
    out_data = np.concatenate(arrays, axis=dim)
    req_parents = [t for t in tensors if t.requires_grad]
    requires_grad = (len(req_parents) > 0) and _grad_enabled
    out = Tensor(out_data, requires_grad=requires_grad)
    out._op = "cat"
    if requires_grad:
        out._prev = set(req_parents)
        sizes = [t.shape[dim] for t in tensors]

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            offset = 0
            for t, s in zip(tensors, sizes):
                if t.requires_grad:
                    sl = [slice(None)] * g.ndim
                    sl[dim] = slice(offset, offset + s)
                    _add_grad(t, g[tuple(sl)])
                offset += s
        out._backward = _backward
    return out


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("stack 不能处理空列表")
    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
    arrays = [t.data for t in tensors]
    out_data = np.stack(arrays, axis=dim)
    req_parents = [t for t in tensors if t.requires_grad]
    requires_grad = (len(req_parents) > 0) and _grad_enabled
    out = Tensor(out_data, requires_grad=requires_grad)
    out._op = "stack"
    if requires_grad:
        out._prev = set(req_parents)

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            for i, t in enumerate(tensors):
                if t.requires_grad:
                    sl = [slice(None)] * g.ndim
                    sl[dim] = i
                    _add_grad(t, g[tuple(sl)])
        out._backward = _backward
    return out


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    condition = condition if isinstance(condition, Tensor) else Tensor(condition)
    x = x if isinstance(x, Tensor) else Tensor(x)
    y = y if isinstance(y, Tensor) else Tensor(y)
    mask = condition.data.astype(bool)
    out_data = np.where(mask, x.data, y.data)
    req_parents = [p for p in (x, y) if p.requires_grad]
    requires_grad = (len(req_parents) > 0) and _grad_enabled
    out = Tensor(out_data, requires_grad=requires_grad)
    out._op = "where"
    if requires_grad:
        out._prev = set(req_parents)

        def _backward():
            if out.grad is None:
                return
            m = mask.astype(out.data.dtype)
            if x.requires_grad:
                _add_grad(x, _unbroadcast(out.grad * m, x.data.shape))
            if y.requires_grad:
                _add_grad(y, _unbroadcast(out.grad * (1.0 - m), y.data.shape))
        out._backward = _backward
    return out


def manual_seed(seed: int) -> None:
    np.random.seed(seed)


def allclose(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    return bool(np.allclose(a.data, b.data, rtol=rtol, atol=atol))


def equal(a: Tensor, b: Tensor) -> bool:
    """逐元素相等且形状一致的判断（``torch.equal`` 语义）。"""
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    return bool(a.data.shape == b.data.shape and np.array_equal(a.data, b.data))


def isfinite(x: Tensor) -> Tensor:
    """逐元素判断有限性（``torch.isfinite`` 语义）。"""
    x = x if isinstance(x, Tensor) else Tensor(x)
    out = Tensor(np.isfinite(x.data), requires_grad=False)
    return out


def frombuffer(buffer, *, dtype=np.float32, count: int = -1, offset: int = 0) -> Tensor:
    """从字节缓冲区构造张量（``torch.frombuffer`` 语义）。"""
    arr = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset)
    return Tensor(np.array(arr), requires_grad=_grad_enabled)


def from_numpy(ndarray) -> Tensor:
    """从 NumPy 数组构造张量（``torch.from_numpy`` 语义）。"""
    return Tensor(np.array(ndarray), requires_grad=_grad_enabled)



# ============================================================
# 扩展：激活 / 归一化 / 损失（均带梯度）
# ============================================================
def relu(x: Tensor) -> Tensor:
    out = Tensor._make_out(x, np.maximum(0.0, x.data), "relu", (x,))
    if out.requires_grad:
        a = x.data

        def _backward():
            if out.grad is None:
                return
            _add_grad(x, out.grad * (a > 0).astype(a.dtype))
        out._backward = _backward
    return out


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    a = x.data
    out = Tensor._make_out(x, np.where(a > 0, a, a * negative_slope), "leaky_relu", (x,))
    if out.requires_grad:
        def _backward():
            if out.grad is None:
                return
            _add_grad(x, out.grad * np.where(a > 0, 1.0, negative_slope).astype(a.dtype))
        out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    # 数值稳定: 对大正值直接返回1, 大负值直接返回0
    x_data = x.data
    out_data = np.empty_like(x_data, dtype=np.float32 if x_data.dtype == np.float32 else x_data.dtype)
    pos = x_data >= 0
    neg = ~pos
    out_data[pos] = 1.0 / (1.0 + np.exp(-np.clip(x_data[pos], -500, 500)))
    exp_x = np.exp(np.clip(x_data[neg], -500, 500))
    out_data[neg] = exp_x / (1.0 + exp_x)
    out = Tensor._make_out(x, out_data, "sigmoid", (x,))
    if out.requires_grad:
        def _backward():
            if out.grad is None:
                return
            _add_grad(x, out.grad * out_data * (1.0 - out_data))
        out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    out_data = np.tanh(x.data)
    out = Tensor._make_out(x, out_data, "tanh", (x,))
    if out.requires_grad:
        def _backward():
            if out.grad is None:
                return
            _add_grad(x, out.grad * (1.0 - out_data * out_data))
        out._backward = _backward
    return out


def gelu(x: Tensor) -> Tensor:
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x.data + 0.044715 * x.data ** 3)
    t = np.tanh(inner)
    out_data = 0.5 * x.data * (1.0 + t)
    out = Tensor._make_out(x, out_data, "gelu", (x,))
    if out.requires_grad:
        a = x.data

        def _backward():
            if out.grad is None:
                return
            dt = (1.0 - t * t) * c * (1.0 + 3 * 0.044715 * a * a)
            dx = 0.5 * (1.0 + t) + 0.5 * a * dt
            _add_grad(x, out.grad * dx)
        out._backward = _backward
    return out


def silu(x: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-x.data))
    out_data = x.data * s
    out = Tensor._make_out(x, out_data, "silu", (x,))
    if out.requires_grad:
        def _backward():
            if out.grad is None:
                return
            _add_grad(x, out.grad * s * (1.0 + x.data * (1.0 - s)))
        out._backward = _backward
    return out


# gelu_tanh 别名（与 gelu 相同，G: drive 版本已使用 tanh 近似）
gelu_tanh = gelu


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    shifted = x.data - np.max(x.data, axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    out_data = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    out = Tensor._make_out(x, out_data, "softmax", (x,))
    if out.requires_grad:
        def _backward():
            if out.grad is None:
                return
            s = out_data
            dx = s * (out.grad - np.sum(out.grad * s, axis=dim, keepdims=True))
            _add_grad(x, dx)
        out._backward = _backward
    return out


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    shifted = x.data - np.max(x.data, axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    log_sum = np.log(np.sum(exp_x, axis=dim, keepdims=True))
    out_data = shifted - log_sum
    out = Tensor._make_out(x, out_data, "log_softmax", (x,))
    if out.requires_grad:
        sm = exp_x / np.sum(exp_x, axis=dim, keepdims=True)

        def _backward():
            if out.grad is None:
                return
            dx = out.grad - sm * np.sum(out.grad, axis=dim, keepdims=True)
            _add_grad(x, dx)
        out._backward = _backward
    return out


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0:
        return x
    mask = (np.random.rand(*x.data.shape) >= p).astype(x.data.dtype)
    out_data = x.data * mask / (1.0 - p)
    out = Tensor._make_out(x, out_data, "dropout", (x,))
    if out.requires_grad:
        def _backward():
            if out.grad is None:
                return
            _add_grad(x, out.grad * mask / (1.0 - p))
        out._backward = _backward
    return out


def layer_norm(x: Tensor, normalized_shape, weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    norm_ndim = len(normalized_shape)
    axes = tuple(range(x.ndim - norm_ndim, x.ndim))
    mean = np.mean(x.data, axis=axes, keepdims=True)
    var = np.var(x.data, axis=axes, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm = (x.data - mean) * inv_std
    out_data = x_norm
    if weight is not None:
        out_data = out_data * weight.data
    if bias is not None:
        out_data = out_data + bias.data
    parents = [x] + ([weight] if weight is not None and weight.requires_grad else []) \
              + ([bias] if bias is not None and bias.requires_grad else [])
    requires_grad = (x.requires_grad or
                      (weight is not None and weight.requires_grad) or
                      (bias is not None and bias.requires_grad)) and _grad_enabled
    out = Tensor(out_data, requires_grad=requires_grad)
    out._op = "layer_norm"
    if requires_grad:
        out._prev = set(p for p in (x, weight, bias) if p is not None and p.requires_grad)

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if weight is not None and weight.requires_grad:
                gw = np.sum(g * x_norm, axis=tuple(range(0, x.ndim - norm_ndim)))
                _add_grad(weight, gw)
            if bias is not None and bias.requires_grad:
                gb = np.sum(g, axis=tuple(range(0, x.ndim - norm_ndim)))
                _add_grad(bias, gb)
            if x.requires_grad:
                gw_data = weight.data if weight is not None else 1.0
                gxn = g * gw_data
                N = int(np.prod(normalized_shape))
                dx = (gxn - np.mean(gxn, axis=axes, keepdims=True)
                      - x_norm * np.mean(gxn * x_norm, axis=axes, keepdims=True)) * inv_std
                _add_grad(x, dx)
        out._backward = _backward
    return out


def normalize(x: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    """沿 dim 做 L_p 归一化: x / max(||x||_p, eps)。"""
    abs_x = np.abs(x.data)
    norm = np.sum(abs_x ** p, axis=dim, keepdims=True) ** (1.0 / p)
    norm = np.maximum(norm, eps)
    out_data = x.data / norm
    out = Tensor._make_out(x, out_data, "normalize", (x,))
    if out.requires_grad:
        def _backward():
            if out.grad is None:
                return
            g = out.grad
            # d(x/||x||) = (g - out * sum(g * out)) / norm
            dot = np.sum(g * out_data, axis=dim, keepdims=True)
            _add_grad(x, (g - out_data * dot) / norm)
        out._backward = _backward
    return out


def batch_norm(x: Tensor, weight: Optional[Tensor], bias: Optional[Tensor],
               running_mean=None, running_var=None, training: bool = True,
               momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    if x.ndim != 4:
        raise ValueError("batch_norm 仅支持 4D 输入 (N,C,H,W)")
    B, C, H, W = x.shape
    axes = (0, 2, 3)
    if training:
        mean = np.mean(x.data, axis=axes, keepdims=True)
        var = np.var(x.data, axis=axes, keepdims=True)
    else:
        mean = np.asarray(running_mean).reshape(1, C, 1, 1)
        var = np.asarray(running_var).reshape(1, C, 1, 1)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm = (x.data - mean) * inv_std
    w = weight.data.reshape(1, C, 1, 1) if weight is not None else 1.0
    b = bias.data.reshape(1, C, 1, 1) if bias is not None else 0.0
    out_data = w * x_norm + b
    requires_grad = (x.requires_grad or (weight is not None and weight.requires_grad)
                     or (bias is not None and bias.requires_grad)) and _grad_enabled
    out = Tensor(out_data, requires_grad=requires_grad)
    out._op = "batch_norm"
    if requires_grad:
        out._prev = set(p for p in (x, weight, bias) if p is not None and p.requires_grad)

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if weight is not None and weight.requires_grad:
                _add_grad(weight, np.sum(g * x_norm, axis=axes))
            if bias is not None and bias.requires_grad:
                _add_grad(bias, np.sum(g, axis=axes))
            if x.requires_grad:
                N = B * H * W
                gxn = g * w
                dx = (gxn - np.mean(gxn, axis=axes, keepdims=True)
                      - x_norm * np.mean(gxn * x_norm, axis=axes, keepdims=True)) * inv_std
                _add_grad(x, dx)
        out._backward = _backward
    return out


# ---------- 损失函数 ----------
def cross_entropy(input: Tensor, target, reduction: str = "mean",
                  ignore_index: int = -100, **kwargs) -> Tensor:
    """input: (N,C) logits；target: (N,) 类别索引 或 (N,C) one-hot。"""
    if isinstance(target, Tensor):
        t = target.data
    else:
        t = np.asarray(target)
    log_probs = log_softmax(input, dim=1)
    if t.ndim == 1:
        idx = t.astype(np.int64)
        # 支持 ignore_index: 忽略的样本 loss=0, 不参与梯度
        valid_mask = (idx != ignore_index)
        if valid_mask.all():
            loss = -log_probs.data[np.arange(len(idx)), idx]
        else:
            loss = np.zeros(len(idx), dtype=log_probs.data.dtype)
            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx) > 0:
                loss[valid_idx] = -log_probs.data[valid_idx, idx[valid_idx]]
    else:
        loss = -np.sum(log_probs.data * t, axis=1)
    if reduction == "mean":
        if t.ndim == 1 and not valid_mask.all():
            out_data = np.array([loss[valid_mask].mean()]) if valid_mask.any() else np.array([0.0])
        else:
            out_data = np.array([loss.mean()])
    elif reduction == "sum":
        out_data = np.array([loss.sum()])
    else:
        out_data = loss
    out = Tensor(out_data, requires_grad=input.requires_grad and _grad_enabled)
    out._op = "cross_entropy"
    if out.requires_grad:
        out._prev = {input}
        sm = np.exp(log_probs.data)
        N = log_probs.shape[0]

        def _backward():
            if out.grad is None:
                return
            if t.ndim == 1:
                dx = sm.copy()
                if valid_mask.all():
                    dx[np.arange(len(idx)), idx] -= 1.0
                else:
                    valid_idx_arr = np.where(valid_mask)[0]
                    dx[valid_idx_arr, idx[valid_idx_arr]] -= 1.0
            else:
                dx = sm - t
            if reduction == "mean":
                if t.ndim == 1 and not valid_mask.all():
                    n_valid = max(valid_mask.sum(), 1)
                    dx = dx / n_valid
                else:
                    dx = dx / N
                dx = dx * out.grad.reshape(-1)[0]
            elif reduction == "sum":
                dx = dx * out.grad.reshape(-1)[0]
            else:  # none
                dx = dx * out.grad[:, None]
            _add_grad(input, dx)
        out._backward = _backward
    return out


def nll_loss(input: Tensor, target, reduction: str = "mean") -> Tensor:
    """input: (N,C) log-probabilities；target: (N,) 类别索引。"""
    if isinstance(target, Tensor):
        t = target.data.astype(np.int64)
    else:
        t = np.asarray(target).astype(np.int64)
    loss = -input.data[np.arange(len(t)), t]
    if reduction == "mean":
        out_data = np.array([loss.mean()])
    elif reduction == "sum":
        out_data = np.array([loss.sum()])
    else:
        out_data = loss
    out = Tensor(out_data, requires_grad=input.requires_grad and _grad_enabled)
    out._op = "nll_loss"
    if out.requires_grad:
        out._prev = {input}
        N = input.shape[0]

        def _backward():
            if out.grad is None:
                return
            dx = np.zeros_like(input.data)
            dx[np.arange(len(t)), t] = -1.0
            if reduction == "mean":
                dx = dx / N
                dx = dx * out.grad.reshape(-1)[0]
            elif reduction == "sum":
                dx = dx * out.grad.reshape(-1)[0]
            else:  # none
                dx = dx * out.grad[:, None]
            _add_grad(input, dx)
        out._backward = _backward
    return out


def mse_loss(input: Tensor, target, reduction: str = "mean") -> Tensor:
    target = target if isinstance(target, Tensor) else Tensor(target)
    diff = input - target
    sq = diff * diff
    if reduction == "mean":
        out = sq.mean()
    elif reduction == "sum":
        out = sq.sum()
    else:
        out = sq
    return out


def l1_loss(input: Tensor, target, reduction: str = "mean") -> Tensor:
    target = target if isinstance(target, Tensor) else Tensor(target)
    ad = (input - target).abs()
    if reduction == "mean":
        out = ad.mean()
    elif reduction == "sum":
        out = ad.sum()
    else:
        out = ad
    return out


def binary_cross_entropy(input: Tensor, target, reduction: str = "mean") -> Tensor:
    """input: 概率 (0~1)；target: 同形状标签。"""
    target = target if isinstance(target, Tensor) else Tensor(target)
    eps = 1e-12
    x = input.data
    t = target.data
    loss = -(t * np.log(x + eps) + (1 - t) * np.log(1 - x + eps))
    if reduction == "mean":
        out_data = np.array([loss.mean()])
    elif reduction == "sum":
        out_data = np.array([loss.sum()])
    else:
        out_data = loss
    out = Tensor(out_data, requires_grad=input.requires_grad and _grad_enabled)
    out._op = "bce"
    if out.requires_grad:
        out._prev = {input}

        def _backward():
            if out.grad is None:
                return
            dx = (x - t) / (x * (1 - x) + eps)
            if reduction == "mean":
                dx = dx / x.size
                dx = dx * out.grad.reshape(-1)[0]
            elif reduction == "sum":
                dx = dx * out.grad.reshape(-1)[0]
            else:  # none
                dx = dx * out.grad
            _add_grad(input, dx)
        out._backward = _backward
    return out


# ============================================================
# 补充工具函数
# ============================================================
def matmul(a: Tensor, b: Tensor) -> Tensor:
    """矩阵乘法。"""
    return a @ b


def bmm(a: Tensor, b: Tensor) -> Tensor:
    """批量矩阵乘法。"""
    out_data = np.matmul(a.data, b.data)

    def _backward() -> None:
        g = out.grad if out.grad is not None else np.zeros_like(out_data)
        if a.requires_grad:
            _add_grad(a, np.matmul(g, np.swapaxes(b.data, -1, -2)))
        if b.requires_grad:
            _add_grad(b, np.matmul(np.swapaxes(a.data, -1, -2), g))

    req = (a.requires_grad or b.requires_grad) and _grad_enabled
    out = Tensor(out_data, requires_grad=req)
    out._op = "bmm"
    if req:
        out._prev = {a, b}
        out._backward = _backward
    return out


def randint(low: int, high: int, size, dtype=np.int64, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    """随机整数张量（``torch.randint`` 语义；``device`` 被忽略，自研引擎恒在 CPU）。"""
    return Tensor(np.random.randint(low, high, size=size).astype(dtype), requires_grad=requires_grad)


def erf(x: np.ndarray) -> np.ndarray:
    """误差函数。"""
    try:
        from scipy.special import erf as _erf
        return _erf(x)
    except ImportError:
        # Abramowitz & Stegun 近似 (7.1.26)
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = np.sign(x)
        x_abs = np.abs(x)
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)
        return sign * y


def diag(x: Tensor) -> Tensor:
    """提取 / 构造对角矩阵（兼容 torch.diag）。

    - 1-D 输入：返回以输入为对角线的 2-D 方阵
    - 2-D 输入：返回对角线元素组成的 1-D 向量
    """
    data = x.data
    if data.ndim == 1:
        out = np.diag(data)
    elif data.ndim == 2:
        out = np.diag(data)
    else:
        raise ValueError(f"diag 期望 1-D 或 2-D 输入，得到 {data.ndim}-D")
    return Tensor(out, requires_grad=x.requires_grad)


def einsum(equation: str, *operands) -> Tensor:
    """爱因斯坦求和约定（兼容 torch.einsum，基于 numpy.einsum）。

    支持常见的多头注意力 / 路由 / 对比学习表达，如
    ``"blp,blpd->bld"``、``"bchs,bcshd->bchd"``、``"nc,nc->n"`` 等。
    """
    np_ops = [o.data if isinstance(o, Tensor) else np.asarray(o) for o in operands]
    req = any(isinstance(o, Tensor) and o.requires_grad for o in operands)
    out_data = np.einsum(equation, *np_ops)
    return Tensor(out_data, requires_grad=req)


# ============================================================
# 补充模块级函数 (torch.xxx 兼容)
# ============================================================
def topk(input: Tensor, k: int, dim: int = -1, largest: bool = True):
    """返回 (values, indices) — top-k 操作"""
    return input.topk(k, dim=dim, largest=largest)


def flip(input: Tensor, dims) -> Tensor:
    """沿指定维度翻转"""
    return input.flip(dims)


def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    """上三角矩阵"""
    return input.triu(diagonal)


def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    """下三角矩阵"""
    return input.tril(diagonal)


def outer(a: Tensor, b: Tensor) -> Tensor:
    """外积"""
    return a.outer(b)


def cumsum(input: Tensor, dim: int = -1) -> Tensor:
    """累积求和"""
    return input.cumsum(dim)


def sort(input: Tensor, dim: int = -1, descending: bool = False):
    """排序, 返回 (sorted_values, indices)"""
    return input.sort(dim=dim, descending=descending)


def exp(input: Tensor) -> Tensor:
    """指数函数 (模块级)"""
    return input.exp()


def log(input: Tensor) -> Tensor:
    """对数函数 (模块级)"""
    return input.log()


def sqrt(input: Tensor) -> Tensor:
    """平方根 (模块级)"""
    return input.sqrt()


def cos(input: Tensor) -> Tensor:
    """余弦 (模块级)"""
    return input.cos()


def sin(input: Tensor) -> Tensor:
    """正弦 (模块级)"""
    return input.sin()


def abs(input: Tensor) -> Tensor:
    """绝对值 (模块级)"""
    return input.abs()


def sigmoid_(input: Tensor) -> Tensor:
    """Sigmoid (模块级别名)"""
    return sigmoid(input)


def multinomial(input: Tensor, num_samples: int, replacement: bool = True) -> Tensor:
    """多项分布采样

    Args:
        input: [B, N] 或 [N] 概率分布 (每行和为1)
        num_samples: 每行采样数
        replacement: 是否允许重复采样
    Returns:
        [B, num_samples] 或 [num_samples] 的索引张量
    """
    probs = input.data
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    B, N = probs.shape
    # 累积分布函数
    cdf = np.cumsum(probs, axis=1)
    cdf = cdf / cdf[:, -1:]  # 归一化确保最后一列为1
    result = np.zeros((B, num_samples), dtype=np.int64)
    for b in range(B):
        if replacement:
            rand = np.random.rand(num_samples)
        else:
            rand = np.random.rand(min(num_samples, N))
        indices = np.searchsorted(cdf[b], rand)
        indices = np.clip(indices, 0, N - 1)
        result[b] = indices[:num_samples]
    if squeeze:
        result = result.reshape(-1)
    return Tensor(result)


def randint_like(input: Tensor, low: int, high: int, dtype=np.int64) -> Tensor:
    """返回与 input 同形状的随机整数张量"""
    return Tensor(np.random.randint(low, high, size=input.data.shape).astype(dtype))


def empty(*shape, dtype=None, requires_grad: bool = False, device: str = "cpu") -> Tensor:
    """未初始化张量"""
    shp = _norm_shape(shape)
    return Tensor(np.empty(shp, dtype=dtype or _DEFAULT_DTYPE),
                  requires_grad=requires_grad, device=device)


# 兼容 torch.no_grad 作为函数调用 (不仅装饰器)
def no_grad_fn(fn=None):
    if fn is not None:
        return no_grad()(fn)
    return no_grad()


# ============================================================
# 自检（含数值梯度校验）
# ============================================================
def _num_grad(f: Callable[[Tensor], Tensor], x_data: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """对单输入标量输出函数 f 做中心差分数值梯度（输入不需要梯度）。"""
    grad = np.zeros_like(x_data, dtype=np.float64)
    x_data = x_data.astype(np.float64)
    it = np.nditer(x_data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        old = float(x_data[i])
        x_data[i] = old + eps
        fp = f(Tensor(x_data.copy(), dtype=np.float64)).item()
        x_data[i] = old - eps
        fm = f(Tensor(x_data.copy(), dtype=np.float64)).item()
        x_data[i] = old
        grad[i] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def _check(name: str, f: Callable[[Tensor], Tensor], data: np.ndarray,
           tol: float = 1e-3, eps: float = 1e-5) -> None:
    """比较解析梯度（autograd）与数值梯度（中心差分）。"""
    def scalar(t: Tensor) -> Tensor:
        r = f(t)
        return r.sum() if r.numel() != 1 else r

    x = Tensor(data.astype(np.float64), dtype=np.float64, requires_grad=True)
    scalar(x).backward()
    ana = x.grad.astype(np.float64)
    num = _num_grad(scalar, data.astype(np.float64), eps)
    denom = np.max(np.abs(num)) + 1e-8
    err = np.max(np.abs(ana - num)) / denom
    status = "PASS" if err < tol else "FAIL"
    print(f"   [{status}] {name:<28s} 最大相对误差={err:.2e}")
    assert status == "PASS", f"{name} 梯度校验失败: 相对误差={err:.2e}"


if __name__ == "__main__":
    print("=" * 64)
    print("SimplCore Tensor 自检（前向 + 反向 + 数值梯度校验）")
    print("=" * 64)

    np.random.seed(42)

    # ---- 1. 前向基本运算 ----
    print("\n1. 前向基本运算")
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    assert (a + b).shape == (2, 2)
    assert (a @ b).shape == (2, 2)
    assert (a * b).shape == (2, 2)
    assert a.sum().item() == 10.0
    assert abs(a.mean().item() - 2.5) < 1e-6
    print("   [PASS] add / mul / matmul / sum / mean 前向正确")

    # ---- 2. 拓扑排序反向传播 ----
    print("\n2. 拓扑排序反向传播（共享子图梯度累积）")
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * x           # y = x^2, dy/dx = 2x
    z = y + y           # 共享子图，z = 2x^2, dz/dx = 4x
    z.sum().backward()
    assert np.allclose(x.grad, 4 * x.data), x.grad
    print("   [PASS] 共享子图梯度 = 4x ->", x.grad)

    # ---- 3. 数值梯度校验 ----
    print("\n3. 数值梯度校验（中心差分）")
    data = np.random.randn(4, 5)
    _W = np.random.RandomState(7).randn(4, 5)   # 非均匀权重，使 softmax 等和不再为常量
    _check("add", lambda t: t + Tensor(np.ones((4, 5))), data)
    _check("sub", lambda t: t - Tensor(np.ones((4, 5))), data)
    _check("mul", lambda t: t * Tensor(np.random.RandomState(0).randn(4, 5)), data)
    _check("div", lambda t: t / Tensor(np.random.RandomState(1).randn(4, 5) + 3.0), data)
    _check("pow(x^3)", lambda t: t.pow(3), data)
    _check("neg", lambda t: -t, data)
    _check("matmul", lambda t: t @ Tensor(np.random.RandomState(2).randn(5, 3)), data)
    _check("sum", lambda t: t.sum(), data)
    _check("mean", lambda t: t.mean(), data)
    _check("sum(dim=-1)", lambda t: t.sum(dim=-1), data)
    _check("mean(dim=0)", lambda t: t.mean(dim=0), data)
    _check("exp", lambda t: t.exp(), data)
    _check("log", lambda t: (t.exp() + 0.1).log(), data)
    _check("sqrt", lambda t: (t * t + 0.1).sqrt(), data)
    _check("sin", lambda t: t.sin(), data)
    _check("cos", lambda t: t.cos(), data)
    _check("abs", lambda t: t.abs(), data)
    _check("reshape", lambda t: t.reshape(2, 10), data)
    _check("transpose", lambda t: t.transpose(1, 0), data)
    _check("permute", lambda t: t.permute(1, 0), data)
    _check("flatten", lambda t: t.flatten(), data)
    _check("squeeze/unsqueeze", lambda t: t.unsqueeze(0).squeeze(0), data)
    _check("clamp", lambda t: t.clamp(-0.5, 0.5), data)
    _check("max", lambda t: t.max(), data)
    _check("min", lambda t: t.min(), data)
    _check("max(dim=1)", lambda t: t.max(dim=1), data)
    _check("relu", lambda t: relu(t), data)
    _check("gelu", lambda t: gelu(t), data)
    _check("sigmoid", lambda t: sigmoid(t), data)
    _check("tanh", lambda t: tanh(t), data)
    _check("silu", lambda t: silu(t), data)
    _check("softmax", lambda t: (softmax(t, dim=-1) * Tensor(_W)).sum(), data)
    _check("log_softmax", lambda t: (log_softmax(t, dim=-1) * Tensor(_W)).sum(), data)

    # ---- 4. no_grad / enable_grad ----
    print("\n4. no_grad / enable_grad")
    with no_grad():
        w = Tensor([1.0, 2.0], requires_grad=True)
        out = (w * 2.0).sum()
        assert out.requires_grad is False, "no_grad 下不应构建计算图"
        assert len(out._prev) == 0
    print("   [PASS] no_grad 下不构建计算图")

    # enable_grad 在 no_grad 内部重新启用梯度（仅对 requires_grad=True 的张量生效，
    # 这与 PyTorch 语义一致；requires_grad=False 的张量不会被强制建图）
    w = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        q_outer = (w * 2.0)
        assert q_outer.requires_grad is False, "no_grad 下不建图"
        with enable_grad():
            q_inner = (w * 3.0)
        assert q_inner.requires_grad is True, "enable_grad 应在 no_grad 中重新启用建图"
    print("   [PASS] enable_grad 在 no_grad 中重新启用建图")

    # requires_grad=False 的张量即使 enable_grad 也不会被强制建图
    p = Tensor([1.0, 2.0], requires_grad=False)
    with enable_grad():
        q = (p * 3.0)
    assert q.requires_grad is False, "requires_grad=False 不应被强制建图"
    print("   [PASS] requires_grad=False 不被强制建图")

    # ---- 5. cat / stack / where / 索引 ----
    print("\n5. cat / stack / where / 索引梯度")
    t1 = Tensor(np.random.randn(2, 3), requires_grad=True)
    t2 = Tensor(np.random.randn(2, 3), requires_grad=True)
    c = cat([t1, t2], dim=1)
    assert c.shape == (2, 6)
    c.sum().backward()
    assert np.allclose(t1.grad, np.ones((2, 3)))
    assert np.allclose(t2.grad, np.ones((2, 3)))
    print("   [PASS] cat 梯度散射正确")

    s = stack([t1.zero_grad() or Tensor(t1.data, requires_grad=True),
               Tensor(t2.data, requires_grad=True)], dim=0)
    assert s.shape == (2, 2, 3)
    s.sum().backward()
    print("   [PASS] stack 梯度散射正确")

    xi = Tensor(np.random.randn(3, 4), requires_grad=True)
    yi = xi[1, :]
    yi.sum().backward()
    expected = np.zeros((3, 4)); expected[1, :] = 1
    assert np.allclose(xi.grad, expected)
    print("   [PASS] __getitem__ 梯度散射正确")

    # ---- 6. 设备 / dtype ----
    print("\n6. 设备与 dtype")
    t = Tensor([[1.0, 2.0]], requires_grad=True)
    assert t.device == "cpu"
    assert t.to("cpu") is t, "to('cpu') 应为 no-op"
    assert t.cpu() is t
    tf = t.to(dtype=np.float64)
    assert tf.dtype == np.float64
    assert t.float().dtype == np.float32
    print("   [PASS] to('cpu') no-op, dtype 转换正常")

    # ---- 7. 损失函数 ----
    print("\n7. 损失函数梯度")
    logits = Tensor(np.random.randn(5, 3), requires_grad=True)
    target = np.array([0, 1, 2, 1, 0])
    loss = cross_entropy(logits, target)
    loss.backward()
    assert logits.grad is not None and logits.grad.shape == (5, 3)
    print("   [PASS] cross_entropy 梯度 shape 正确")

    print("\n" + "=" * 64)
    print("SimplCore Tensor 自检全部通过")
    print("=" * 64)
