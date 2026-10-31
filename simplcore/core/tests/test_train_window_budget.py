# -*- coding: utf-8 -*-
"""主训练路径的显存结构闸门测试。

守护一件事：主路径（整段并行 FFT）**不能**在超长窗口上假装可行。
SGC 的 O(1) 状态只属于推理/流式；训练窗口的 ``(L, D)`` 核是 ``O(L·D)``。
"""
from __future__ import annotations

import pytest

from simplcore.core.config import get_preset
from simplcore.core.train import (
    assert_train_window_fits,
    train_window_bytes,
)


def test_kernel_bytes_is_linear_in_length_and_width() -> None:
    one = train_window_bytes(4096, 2048)
    assert one == 4096 * 2048 * 4
    assert train_window_bytes(8192, 2048) == 2 * one
    assert train_window_bytes(4096, 4096) == 2 * one


def test_4m_window_is_rejected_by_default_budget() -> None:
    """4M × 2048 的 fp32 核是 32 GiB —— 默认 4 GiB 预算必须拒绝。"""
    assert train_window_bytes(4_000_000, 2048) == pytest.approx(3.2768e10, rel=0.001)
    with pytest.raises(SystemExit) as exc:
        assert_train_window_fits(4_000_000, 2048, 4.0)
    msg = str(exc.value)
    assert "simplcore.core.trainer" in msg   # 必须给出分块路径
    assert "O(L)" in msg                # 必须解释真实约束


def test_short_window_passes() -> None:
    assert_train_window_fits(4096, 2048, 4.0)


def test_zero_budget_disables_check() -> None:
    assert_train_window_fits(4_000_000, 2048, 0)


def test_preset_4m_curriculum_is_not_silently_trainable() -> None:
    """tuan-1b 自带 4M 课程；默认预算下必须显式失败，而不是训到一半 OOM。"""
    cfg = get_preset("tuan-1b")
    assert cfg.context is not None
    max_len = max(st["seq_len"] for st in cfg.context["curriculum"])
    assert max_len == 4_000_000
    with pytest.raises(SystemExit):
        assert_train_window_fits(max_len, cfg.dim, 4.0)