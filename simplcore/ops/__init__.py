# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 流式推理算子包。

本包提供 SimplCore mixer 的**融合流式热路径**：把原本在 PyTorch 里拆成十几个
小算子（RMSNorm → in_proj → 内容门控 → 振荡递推 → 局部卷积 → 证据门 → 不确定门
→ out_proj → 残差）融合成单遍 C++/CUDA 算子，O(1) 状态原地更新。

真正的实现在 :mod:`simplcore.ops.mixer_stream`（文件名即语义：mixer 的流式
热路径，而非某个具体的 mixer 种类）。本文件仅做公开 API 的再导出，方便
``from simplcore.ops import is_available, mixer_stream_step`` 这种旧式写法。

命名约定（延续仓库"诚实纪律"）：
* ``SGC`` 只指 SimplCore 的**注意力替代机制**（``SpectralGlobalConv``，一种用阻尼
  振荡基参数化的全局卷积）；它只是 mixer 内部的一个分量。
* 本包融合的是**整个 mixer 的流式 step**，因此命名为 ``mixer_stream``，不叫 sgc。
* 无 torch / 无编译器 / 无 CUDA 时 ``is_available()`` 返回 ``False``，调用方自动
  回退纯 PyTorch——绝不悄悄出错。
"""
from .mixer_stream import (
    build_error,
    is_available,
    mixer_stream_step,
    self_test,
)
from .srf_stream import (
    build_error as srf_build_error,
    is_available as srf_is_available,
    srf_stream_step,
    self_test as srf_self_test,
)

__all__ = [
    "is_available",
    "mixer_stream_step",
    "build_error",
    "self_test",
    "srf_is_available",
    "srf_stream_step",
    "srf_build_error",
    "srf_self_test",
]
