# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""torch.compile 封装与后端探测工具。

集中处理 PyTorch 2.1 的 ``torch.compile`` 调用，提供安全回退与可用后端探测，
避免在生产环境因 inductor/CUDA 不可用而崩溃。
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch


def available_backends() -> list:
    """返回当前环境可用的编译后端列表。"""
    backends = ["eager"]
    try:
        from torch._inductor import config as _  # noqa: F401
        backends.append("inductor")
    except Exception:
        pass
    for cand in ("aot_eager", "aot_ts", "nvfuser", "cudagraphs"):
        try:
            torch.compile(lambda x: x, backend=cand)
            backends.append(cand)
        except Exception:
            pass
    return backends


def compile_model(
    model: torch.nn.Module,
    *,
    mode: str = "default",
    dynamic: bool = False,
    backend: str = "inductor",
    fullgraph: bool = False,
    smoke_input: Optional[tuple] = None,
) -> torch.nn.Module:
    """编译模型，失败安全回退到原模型。

    Args:
        model: 待编译模块。
        mode / dynamic / backend / fullgraph: torch.compile 参数。
        smoke_input: (可选) 用于烟雾测试的空输入形状元组，触发一次编译以提前暴露错误。

    Returns:
        编译后的模块；若编译不可用则返回原模块并给出警告。
    """
    try:
        compiled = torch.compile(
            model, mode=mode, dynamic=dynamic, backend=backend, fullgraph=fullgraph
        )
        if smoke_input is not None:
            dummy = torch.zeros(*smoke_input, dtype=torch.long)
            with torch.no_grad():
                compiled(dummy)
        return compiled
    except Exception as exc:  # pragma: no cover - 依赖运行环境
        warnings.warn(
            f"torch.compile(backend={backend}) 失败：{exc!r}；回退 eager。",
            RuntimeWarning,
            stacklevel=2,
        )
        return model
