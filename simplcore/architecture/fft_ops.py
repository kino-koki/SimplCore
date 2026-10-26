# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""可微一维实数快速傅里叶变换算子 (Differentiable Real FFT).

本项目底层张量引擎 (simplcore.tensor) 基于 NumPy 自动微分, 原生未提供
复数 FFT 算子。谱全局卷积 (SGC) 依赖沿序列维度的实数 FFT/反 FFT 实现
O(n log n) 的全局卷积, 因此在此提供与 autograd 兼容的可微封装。

实数 FFT 的伴随 (反向) 推导
---------------------------
实数 FFT 是线性算子 R: R^L -> C^F (F = L//2 + 1)。
其伴随算子 R^* 满足 <R x, z> = <x, R^* z>。
对实数信号, 逆实数 FFT 的伴随正是「截取到 F 点并取共轭缩放」的实数 FFT:
    dL/dx = (1/L) * irfft( conj? ) 的实序列
等价地, 反向可通过对梯度张量做 irfft 后再线性缩放 (1/L) 得到,
这正是 NumPy irfft 配合 rfft 的伴随关系 (irfft 已含 1/L 归一化)。
因此对复梯度 grad_C (F 点), 其对应实输入梯度为 irfft(grad_C) (取实部),
仅当 grad_C 不满足共轭对称时会产生虚部, 我们通过 .real 截断以匹配实数输入。

注意: rfft 丢弃负频率的一半信息, 因此反向时仅将前半频点的梯度经 irfft
还原回实序列即可, 与正向信息一致。
"""

from __future__ import annotations

import numpy as np

from ..tensor.tensor import Tensor, _add_grad


def rfft1d(x: Tensor, dim: int = -1) -> Tensor:
    """可微一维实数 FFT.

    将实值序列沿指定维度变换到频域 (复数, 仅非负频率 F = L//2 + 1)。
    为兼容纯实数自动微分后端, 输出以「实部 / 虚部」两个实张量表示,
    返回 (cre, cim) 元组, 每个形状与 x 相同但 dim 维长度为 F。

    Args:
        x: 实值张量.
        dim: 做 FFT 的维度 (默认最后一维).

    Returns:
        (cre, cim): 频域表示的实部与虚部张量.
    """
    data = np.asarray(x.data, dtype=np.float32)
    c = np.fft.rfft(data, axis=dim).astype(np.complex64)
    cre = np.real(c).astype(np.float32)
    cim = np.imag(c).astype(np.float32)

    out_re = x._make_out(cre, "rfft_re", (x,))
    out_im = x._make_out(cim, "rfft_im", (x,))

    L = data.shape[dim]
    n = L

    def _backward():
        if out_re.grad is None and out_im.grad is None:
            return
        gre = out_re.grad if out_re.grad is not None else np.zeros_like(cre)
        gim = out_im.grad if out_im.grad is not None else np.zeros_like(cim)
        gre = np.asarray(gre, dtype=np.float32)
        gim = np.asarray(gim, dtype=np.float32)
        # 重组复梯度并做 irfft (已含 1/L 归一化)
        gc = (gre + 1j * gim).astype(np.complex64)
        grad_real = np.fft.irfft(gc, n=n, axis=dim).astype(np.float32)
        _add_grad(x, grad_real)

    out_re._backward = _backward
    out_im._backward = _backward
    return out_re, out_im


def irfft1d(cre: Tensor, cim: Tensor, n: int, dim: int = -1) -> Tensor:
    """可微一维逆实数 FFT.

    将 (cre, cim) 频域表示还原为长度为 n 的实序列。

    Args:
        cre: 频域实部张量.
        cim: 频域虚部张量 (与 cre 同形).
        n: 输出序列长度 (应等于正向输入长度 L).
        dim: 频域维度.

    Returns:
        实值重建张量, 形状与 cre 相同但 dim 维长度为 n。
    """
    re = np.asarray(cre.data, dtype=np.float32)
    im = np.asarray(cim.data, dtype=np.float32)
    c = (re + 1j * im).astype(np.complex64)
    out = np.fft.irfft(c, n=n, axis=dim).astype(np.float32)

    res = cre._make_out(out, "irfft", (cre, cim))

    def _backward():
        if res.grad is None:
            return
        g = np.asarray(res.grad, dtype=np.float32)
        # rfft 的伴随即 irfft 反向; 对实梯度做 rfft 得到复梯度
        gc = np.fft.rfft(g, axis=dim).astype(np.complex64)
        _add_grad(cre, np.real(gc).astype(np.float32))
        _add_grad(cim, np.imag(gc).astype(np.float32))

    res._backward = _backward
    return res
