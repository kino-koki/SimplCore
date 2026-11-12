# -*- coding: utf-8 -*-
"""Minimal tape-based autodiff, pure NumPy fp64.

Only the operators used by ``simplcore_lm.py`` are implemented. Every operator
has an explicit analytic adjoint; there is no torch, no finite differences in
training, and no hidden state carried between steps.

Designed for the sandbox: 2 CPUs, 2.3 GiB RAM, no GPU, no external DL library.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

_TAPE: list = []
NO_GRAD = False


def set_no_grad(flag: bool):
    global NO_GRAD
    NO_GRAD = bool(flag)


def clear_tape():
    _TAPE.clear()


def _sum_to(g, shape):
    """Reduce a broadcast gradient back to ``shape``."""
    if g.shape == shape:
        return g
    while len(g.shape) > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_back", "shape")

    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._back = None
        self.shape = self.data.shape
        if requires_grad and not NO_GRAD:
            _TAPE.append(self)

    @property
    def size(self):
        return self.data.size

    def zero_grad(self):
        self.grad = None

    def backward(self):
        """Back-propagate from a scalar loss tensor."""
        assert self.data.size == 1, "backward() requires a scalar loss"
        self.grad = np.ones_like(self.data)
        for t in reversed(_TAPE):
            if t._back is not None:
                t._back(t.grad)
        _TAPE.clear()

    def __repr__(self):
        return "Tensor(%s, rg=%s)" % (self.shape, self.requires_grad)


def _acc(t, g):
    if t.requires_grad:
        g = _sum_to(np.asarray(g, dtype=np.float64), t.shape)
        t.grad = g if t.grad is None else t.grad + g


def node(data, inputs, back):
    """Create a tensor produced by an operator with adjoint ``back(g)``."""
    rg = any(i.requires_grad for i in inputs) and not NO_GRAD
    t = Tensor(data, requires_grad=rg)
    if rg:
        t._back = back
    return t


def param(arr):
    return Tensor(arr, requires_grad=True)


def const(arr):
    return Tensor(arr, requires_grad=False)


# ---------------------------------------------------------------------------
# elementwise / broadcast
# ---------------------------------------------------------------------------
def add(a, b):
    return node(a.data + b.data, (a, b),
                lambda g: (_acc(a, _sum_to(g, a.shape)),
                           _acc(b, _sum_to(g, b.shape))))


def mul(a, b):
    return node(a.data * b.data, (a, b),
                lambda g: (_acc(a, _sum_to(g * b.data, a.shape)),
                           _acc(b, _sum_to(g * a.data, b.shape))))


def scale(a, s):
    return mul(a, const(np.array(float(s))))


def matmul(a, b):
    out = np.matmul(a.data, b.data)
    bt = np.swapaxes(b.data, -1, -2)
    at = np.swapaxes(a.data, -1, -2)

    def back(g):
        _acc(a, _sum_to(np.matmul(g, bt), a.shape))
        _acc(b, _sum_to(np.matmul(at, g), b.shape))
    return node(out, (a, b), back)


def transpose(a, axes):
    inv = np.argsort(axes)
    return node(np.transpose(a.data, axes), (a,),
                lambda g: _acc(a, np.transpose(g, inv)))


def reshape(a, shape):
    return node(np.reshape(a.data, shape), (a,),
                lambda g: _acc(a, np.reshape(g, a.shape)))


def sum_(a, axis=None, keepdims=False):
    out = a.data.sum(axis=axis, keepdims=keepdims)

    def back(g):
        if axis is None:
            _acc(a, np.full(a.shape, g))
        elif not keepdims:
            _acc(a, np.expand_dims(g, axis=axis))
        else:
            _acc(a, g)
    return node(out, (a,), back)


def mean(a):
    return node(np.array(a.data.mean()), (a,),
                lambda g: _acc(a, np.full(a.shape, g / a.size)))


def exp(a):
    return node(np.exp(a.data), (a,), lambda g: _acc(a, g * np.exp(a.data)))


def log(a):
    return node(np.log(a.data), (a,), lambda g: _acc(a, g / a.data))


def tanh(a):
    y = np.tanh(a.data)
    return node(y, (a,), lambda g: _acc(a, g * (1.0 - y * y)))


def sigmoid(a):
    y = 1.0 / (1.0 + np.exp(-a.data))
    return node(y, (a,), lambda g: _acc(a, g * y * (1.0 - y)))


def gelu(a):
    x = a.data
    z = x / np.sqrt(2.0)
    y = 0.5 * x * (1.0 + erf(z))
    dy = 0.5 * (1.0 + erf(z)) + x * np.exp(-z * z) / np.sqrt(2.0 * np.pi)
    return node(y, (a,), lambda g: _acc(a, g * dy))


def softmax(a, axis=-1):
    z = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=axis, keepdims=True)
    return node(y, (a,),
                lambda g: _acc(a, y * (g - (g * y).sum(axis=axis, keepdims=True))))


def layernorm(x, gamma, beta, eps=1e-5):
    d = x.shape[-1]
    mu = x.data.mean(axis=-1, keepdims=True)
    xc = x.data - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = xc * inv
    out = xh * gamma.data + beta.data

    def back(g):
        dxh = g * gamma.data
        _acc(gamma, _sum_to((g * xh).sum(axis=tuple(range(g.ndim - 1))), gamma.shape))
        _acc(beta, _sum_to(g.sum(axis=tuple(range(g.ndim - 1))), beta.shape))
        dvar = (dxh * (-0.5) * xc * inv ** 3).sum(axis=-1, keepdims=True)
        dmu = -(dxh * inv).sum(axis=-1, keepdims=True)
        dx = dxh * inv + dvar * 2.0 * xc / d + dmu / d
        _acc(x, dx)
    return node(out, (x, gamma, beta), back)


def embedding(E, idx):
    out = E.data[idx]

    def back(g):
        gE = np.zeros_like(E.data)
        np.add.at(gE, idx.ravel(), g.reshape(-1, g.shape[-1]))
        _acc(E, gE)
    return node(out, (E,), back)


def cross_entropy(logits, tgt):
    """logits (B,L,V), tgt (B,L) integer ids -> scalar CE (nats/token)."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    ex = np.exp(z)
    p = ex / ex.sum(axis=-1, keepdims=True)
    B, L, V = p.shape
    pt = np.take_along_axis(p, tgt[..., None], axis=-1)[..., 0]
    loss = -np.log(np.maximum(pt, 1e-12)).mean()

    def back(g):
        dl = p.copy()
        np.put_along_axis(dl, tgt[..., None],
                          np.take_along_axis(dl, tgt[..., None], axis=-1) - 1.0,
                          axis=-1)
        _acc(logits, g * dl / (B * L))
    return node(np.array(loss), (logits,), back)


def custom(data, inputs, back):
    """Escape hatch for fused operators with hand-written adjoints."""
    return node(data, inputs, back)
