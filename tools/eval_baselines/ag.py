"""
ag.py — self-contained reverse-mode autodiff for the FAIR-BASELINE evaluation.

This file lives OUTSIDE the SimplCore architecture. It is a copy of the
project's minimal autodiff with two purely additive operators (__truediv__,
softmax) so that every compared mechanism can be trained under ONE optimizer,
ONE batch schedule and ONE numerical path. No change is made to the model code.
"""
from __future__ import annotations
import numpy as np


def free_graph(root):
    topo, seen = [], set()

    def dfs(v):
        k = id(v)
        if k in seen:
            return
        seen.add(k)
        for p in v._p:
            dfs(p)
        topo.append(v)

    if isinstance(root, (list, tuple)):
        for r in root:
            dfs(r)
    else:
        dfs(root)
    for v in topo:
        v._p = []
        v._b = None


def _unbroadcast(g, shape):
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, sz in enumerate(shape):
        if sz == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


class T:
    __slots__ = ("d", "g", "_p", "_b")

    def __init__(self, d, _p=(), _b=None):
        self.d = np.asarray(d, dtype=np.float64)
        self.g = np.zeros_like(self.d)
        self._p = list(_p)
        self._b = _b

    def backward(self):
        topo, seen = [], set()

        def dfs(v):
            k = id(v)
            if k in seen:
                return
            seen.add(k)
            for p in v._p:
                dfs(p)
            topo.append(v)

        dfs(self)
        self.g = np.ones_like(self.d)
        for v in reversed(topo):
            if v._b is not None:
                v._b()
        for v in topo:
            v._p = []
            v._b = None

    def __add__(self, o):
        return _bin(self, o, lambda a, b: a + b, lambda g, a, b: (g, g))

    def __mul__(self, o):
        return _bin(self, o, lambda a, b: a * b, lambda g, a, b: (g * b, g * a))

    def __sub__(self, o):
        return self + (-o if isinstance(o, T) else -np.asarray(o))

    def __neg__(self):
        return _un(self, lambda a: -a, lambda g, a: -g)

    def __truediv__(self, o):
        return _bin(self, o, lambda a, b: a / b,
                    lambda g, a, b: (g / b, -g * a / (b * b)))

    def __matmul__(self, o):
        return _bin(self, o, lambda a, b: a @ b,
                    lambda g, a, b: (g @ b.swapaxes(-1, -2),
                                    a.swapaxes(-1, -2) @ g))

    @property
    def T(self):
        return _un(self, lambda a: a.swapaxes(-1, -2),
                   lambda g, a: g.swapaxes(-1, -2))

    def reshape(self, *shape):
        old = self.d.shape
        return _un(self, lambda a: a.reshape(*shape),
                   lambda g, a: g.reshape(old))

    def transpose(self, *axes):
        inv = np.argsort(axes)
        return _un(self, lambda a: a.transpose(*axes),
                   lambda g, a: g.transpose(*inv))

    def sum(self, axis=None, keepdims=False):
        a = self.d

        def bw(g):
            if axis is None:
                return np.broadcast_to(g, a.shape).copy()
            g2 = g if keepdims else np.expand_dims(g, axis)
            return np.broadcast_to(g2, a.shape).copy()
        return _un(self, lambda x: x.sum(axis=axis, keepdims=keepdims),
                   lambda g, x: bw(g))

    def mean(self, axis=None):
        n = self.d.size if axis is None else self.d.shape[axis]
        return self.sum(axis=axis) * (1.0 / n)

    def tanh(self):
        y = np.tanh(self.d)
        return _un(self, lambda a: np.tanh(a), lambda g, a: g * (1 - y * y))

    def sigmoid(self):
        y = 1 / (1 + np.exp(-self.d))
        return _un(self, lambda a: 1 / (1 + np.exp(-a)),
                   lambda g, a: g * y * (1 - y))

    def relu(self):
        return _un(self, lambda a: np.maximum(a, 0), lambda g, a: g * (a > 0))

    def exp(self):
        return _un(self, lambda a: np.exp(a), lambda g, a: g * np.exp(a))

    def log(self):
        return _un(self, lambda a: np.log(np.maximum(a, 1e-300)),
                   lambda g, a: g / np.maximum(a, 1e-300))

    def softmax(self, axis=-1):
        mx = np.max(self.d, axis=axis, keepdims=True)
        e = (self - mx).exp()
        return e / e.sum(axis=axis, keepdims=True)

    def __getitem__(self, key):
        def bw(g):
            z = np.zeros_like(self.d)
            z[key] = g
            return z
        return _un(self, lambda a: a[key], lambda g, a: bw(g))


def _as(o):
    return o.d if isinstance(o, T) else np.asarray(o, dtype=np.float64)


def _bin(x, o, fw, bw):
    y = o if isinstance(o, T) else None
    a, b = _as(x), _as(o)
    out = T(fw(a, b), [x] + ([y] if y is not None else []))

    def back():
        ga, gb = bw(out.g, a, b)
        x.g += _unbroadcast(ga, x.d.shape)
        if y is not None:
            y.g += _unbroadcast(gb, y.d.shape)
    out._b = back
    return out


def _un(x, fw, bw):
    out = T(fw(x.d), [x])

    def back():
        x.g += bw(out.g, x.d)
    out._b = back
    return out


def const(a):
    return T(np.asarray(a, dtype=np.float64))


def param(a):
    return T(np.asarray(a, dtype=np.float64))


def nll_from_logits(logits, targets):
    B, C = logits.d.shape
    m = logits.d.max(axis=-1, keepdims=True)
    e = np.exp(logits.d - m)
    s = e.sum(axis=-1, keepdims=True)
    logp = logits.d - m - np.log(s)
    loss = -logp[np.arange(B), targets].mean()
    out = T(np.array(loss), [logits])

    def back():
        p = e / s
        p[np.arange(B), targets] -= 1.0
        logits.g += out.g * p / B
    out._b = back
    return out
