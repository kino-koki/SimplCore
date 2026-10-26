# -*- coding: utf-8 -*-
"""异常体系与张量健康工具测试。"""
import pytest
import torch

from simplcore_pg.exceptions import (
    SimplCoreError, ConfigError, ModelError, MemoryStateError, RolloutError,
    GradientCheckpointError, ModalityError, StackingError, LossGuardError,
    check_tensor_health, handle_gradient_clipping, format_exception,
)


def test_hierarchy():
    for cls in (ConfigError, ModelError, MemoryStateError, RolloutError,
                GradientCheckpointError, ModalityError, StackingError, LossGuardError):
        assert issubclass(cls, SimplCoreError)


def test_exception_details():
    e = ConfigError("bad", details={"k": 1})
    assert e.message == "bad"
    assert e.details["k"] == 1
    assert "bad" in str(e)


def test_check_tensor_health_nan():
    with pytest.raises(ModelError):
        check_tensor_health(torch.tensor([1.0, float("nan")]), name="x")


def test_check_tensor_health_inf():
    with pytest.raises(ModelError):
        check_tensor_health(torch.tensor([1.0, float("inf")]), name="x")


def test_check_tensor_health_ok():
    check_tensor_health(torch.randn(3, 4), name="x")  # 不抛异常


def test_gradient_clipping_large():
    g = torch.randn(3, 4) * 10
    c = handle_gradient_clipping(g, max_norm=1.0)
    assert c.norm().item() <= 1.0 + 1e-5


def test_gradient_clipping_small_unchanged():
    g = torch.randn(3, 4) * 0.01
    c = handle_gradient_clipping(g, max_norm=1.0)
    assert torch.allclose(g, c)


def test_gradient_clipping_nan():
    with pytest.raises(GradientCheckpointError):
        handle_gradient_clipping(torch.tensor([1.0, float("nan")]))


def test_gradient_clipping_none():
    assert handle_gradient_clipping(None) is None


def test_format_exception():
    e = ModelError("dim mismatch", details={"a": 1})
    s = format_exception(e)
    assert "[ModelError]" in s and "dim mismatch" in s
