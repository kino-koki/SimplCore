# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor Checkpoint — 梯度检查点（自研 NumPy 引擎，零第三方框架依赖）
=================================================================================
自研引擎在 CPU 上运行，内存非瓶颈，梯度检查点采用直通模式直接执行前向计算。
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union

from ..tensor import Tensor


def checkpoint(function, *args, **kwargs):
    """梯度检查点（直通模式）。

    自研引擎在 CPU 上运行，内存不是瓶颈，直接执行前向计算；
    反向传播依赖已构建的计算图完成。
    """
    kwargs.pop('use_reentrant', None)
    return function(*args, **kwargs)


def checkpoint_sequential(functions, inputs, *args, **kwargs):
    """顺序梯度检查点（直通模式）。"""
    if callable(functions) and not isinstance(functions, (list, tuple)):
        return functions(inputs)
    for fn in functions:
        if isinstance(inputs, (tuple, list)):
            inputs = fn(*inputs)
        else:
            inputs = fn(inputs)
    return inputs


__all__ = ["checkpoint", "checkpoint_sequential"]
