# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""直流偏置迁移的契约测试（纯 tensor，不依赖任何外部模型权重）。"""
from __future__ import annotations

import torch

from simplcore.core.migration.dc_bias import (
    LayerwiseDCBias, apply_dc_bias, estimate_dc_bias,
    estimate_layerwise_bias)


def test_identity_gives_zero_bias():
    a = torch.randn(2, 7, 16)
    b = estimate_dc_bias(a, a.clone())
    assert torch.allclose(b, torch.zeros(16), atol=1e-6)


def test_shape_and_value():
    a = torch.zeros(1, 4, 8)
    a[..., 3] = 2.0
    d = torch.zeros(1, 4, 8)
    d[..., 3] = 0.5
    b = estimate_dc_bias(a, d)
    assert b.shape == (8,)
    assert abs(float(b[3]) - 1.5) < 1e-6
    assert torch.allclose(
        b, torch.tensor([0., 0., 0., 1.5, 0., 0., 0., 0.]), atol=1e-6)


def test_batch_and_position_average():
    a = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    d = torch.zeros_like(a)
    b = estimate_dc_bias(a, d)
    assert torch.allclose(b, a.float().mean(dim=(0, 1)), atol=1e-6)


def test_layerwise_stack():
    ats = [torch.randn(1, 5, 8) for _ in range(3)]
    dls = [torch.zeros_like(a) for a in ats]
    b = estimate_layerwise_bias(ats, dls)
    assert b.shape == (3, 8)
    assert torch.allclose(b[1], ats[1].mean(dim=(0, 1)), atol=1e-6)


def test_apply_is_positionwise_constant():
    y = torch.randn(1, 6, 8)
    b = torch.arange(8, dtype=torch.float32)
    out = apply_dc_bias(y, b)
    assert out.shape == y.shape
    # 每个位置的增量完全相同 → 并行与流式两条路径逐位一致
    diff = out - y
    assert torch.allclose(diff[0, 0], diff[0, 5], atol=1e-6)
    assert torch.allclose(diff[0, 3], b, atol=1e-6)


def test_beta_is_linear():
    y = torch.zeros(1, 2, 4)
    b = torch.ones(4)
    o1 = apply_dc_bias(y, b, 1.0)
    o2 = apply_dc_bias(y, b, 0.5)
    assert torch.allclose(o1, 2 * o2, atol=1e-6)


def test_module_has_no_trainable_parameters():
    m = LayerwiseDCBias(torch.randn(28, 16))
    assert m.num_layers == 28 and m.dim == 16
    assert sum(p.numel() for p in m.parameters()) == 0
    assert "bias" in m.state_dict()
    y = torch.zeros(1, 3, 16)
    assert m(y, 7).shape == y.shape
    assert torch.allclose(m(y, 7) - y, m.bias[7].view(1, 1, 16), atol=1e-6)


def test_rejects_bad_shapes():
    import pytest
    with pytest.raises(ValueError):
        estimate_dc_bias(torch.randn(1, 3, 8), torch.randn(1, 4, 8))
    with pytest.raises(ValueError):
        estimate_dc_bias(torch.randn(3, 8), torch.randn(3, 8))
    with pytest.raises(ValueError):
        apply_dc_bias(torch.randn(1, 2, 8), torch.randn(7))
    with pytest.raises(ValueError):
        LayerwiseDCBias(torch.randn(8))
