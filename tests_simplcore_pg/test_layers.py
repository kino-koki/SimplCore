# -*- coding: utf-8 -*-
"""基础层 (RMSNorm / FFN) 测试。

注：RoPE 相关函数（precompute_rope / apply_rotary_pos_emb / rotate_half）已随
架构"彻底放弃注意力"从 simplcore_pg 移除（见 docs/PAPER_SPEC.md v2.0），
对应测试一并删除。本文件只守护仍存在的 RMSNorm / FFN。
"""
import torch

from simplcore_pg.layers import RMSNorm, FFN, rms_norm


def test_rms_norm_shape_dtype():
    x = torch.randn(2, 4, 8)
    w = torch.ones(8)
    y = rms_norm(x, w, eps=1e-6)
    assert y.shape == x.shape
    # 正确参考：RMSNorm = x / sqrt(mean(x^2)) * w (区别于 L2 normalize)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(y, ref, atol=1e-4)


def test_rms_norm_module():
    m = RMSNorm(16)
    x = torch.randn(3, 5, 16)
    y = m(x)
    assert y.shape == x.shape and y.dtype == x.dtype


def test_ffn_shapes_and_grad():
    f = FFN(32, intermediate_ratio=4.0, dropout=0.0)
    x = torch.randn(2, 6, 32, requires_grad=True)
    y = f(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_ffn_dropout_zero_effect():
    f = FFN(32, dropout=0.0)
    x = torch.randn(2, 6, 32)
    torch.manual_seed(0)
    y1 = f(x)
    y2 = f(x)
    assert torch.allclose(y1, y2)
