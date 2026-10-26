# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor NN Functional — 函数式接口（自研 NumPy 引擎，零第三方框架依赖）
====================================================================================
提供与 ``torch.nn.functional`` 兼容的函数式算子，全部基于自研 ``Tensor`` 自动微分引擎。
"""
from __future__ import annotations

from ..tensor import (
    Tensor,
    relu, gelu, gelu_tanh, sigmoid, tanh, silu,
    softmax, log_softmax, layer_norm, dropout, normalize,
    cross_entropy, nll_loss, mse_loss, l1_loss,
    binary_cross_entropy,
    zeros, ones, zeros_like, ones_like,
    cat, stack, where,
)

import numpy as np


def linear(x: Tensor, weight: Tensor, bias=None) -> Tensor:
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def embedding(input, weight: Tensor, padding_idx=None) -> Tensor:
    if isinstance(input, Tensor):
        idx = input.data
    else:
        idx = np.asarray(input)
    idx = idx.astype(np.int64)
    return Tensor(weight.data[idx], requires_grad=weight.requires_grad)


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    ms = (x * x).mean(dim=-1, keepdim=True)
    inv = (ms + eps).pow(-0.5)
    return x * inv * weight


def scaled_dot_product_attention(q, k, v, attn_mask=None,
                                 dropout_p=0.0, is_causal=False, scale=None):
    if scale is None:
        head_dim = q.shape[-1]
        scale = 1.0 / (head_dim ** 0.5)
    scores = (q @ k.transpose(-2, -1)) * scale
    if is_causal:
        T = scores.shape[-1]
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.masked_fill(causal, -1e4)
    if attn_mask is not None:
        if isinstance(attn_mask, Tensor):
            mask = attn_mask.data
        else:
            mask = np.asarray(attn_mask)
        scores = scores.masked_fill(mask == 0, -1e4)
    attn = softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = dropout(attn, p=dropout_p)
    out = attn @ v
    return out


def pad(seq, pad_value=0, max_len=None):
    if isinstance(seq, (list, tuple)):
        arrs = [s.data if isinstance(s, Tensor) else np.asarray(s) for s in seq]
        if max_len is None:
            max_len = max(a.shape[0] for a in arrs)
        padded = []
        for a in arrs:
            if a.shape[0] < max_len:
                pad_width = ((0, max_len - a.shape[0]),) + ((0, 0),) * (a.ndim - 1)
                a = np.pad(a, pad_width, constant_values=pad_value)
            padded.append(a)
        return Tensor(np.stack(padded))
    return seq


def one_hot(x, num_classes):
    if isinstance(x, Tensor):
        idx = x.data
    else:
        idx = np.asarray(x)
    idx = idx.astype(np.int64)
    out = np.zeros((*idx.shape, num_classes), dtype=np.float32)
    np.put_along_axis(out, idx[..., np.newaxis], 1.0, axis=-1)
    return Tensor(out)


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """softplus(x) = 1/beta * log(1 + exp(beta * x)).

    数值稳定: 当 beta*x > threshold 时退化为 x (避免 exp 上溢).
    仅依赖自研 Tensor 的 exp / log / 算术运算, 不引入新算子.
    """
    if isinstance(x, Tensor):
        data = x.data.astype(np.float32)
    else:
        data = np.asarray(x, dtype=np.float32)
    z = beta * data
    out = np.empty_like(data)
    mask = z > threshold
    out[mask] = data[mask]
    out[~mask] = np.log1p(np.exp(z[~mask])) / beta
    return Tensor(out, requires_grad=x.requires_grad if isinstance(x, Tensor) else False)


__all__ = [
    "relu", "gelu", "gelu_tanh", "sigmoid", "tanh", "silu",
    "softmax", "log_softmax", "layer_norm", "dropout", "normalize",
    "cross_entropy", "nll_loss", "mse_loss", "l1_loss",
    "binary_cross_entropy",
    "linear", "embedding", "rms_norm", "scaled_dot_product_attention",
    "pad", "one_hot", "softplus",
]
