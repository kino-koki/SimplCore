# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor Backend Selector (v2 — 可插拔 GPU 后端)
=======================================================

青小团自研张量引擎后端选择器。

本项目已**彻底移除对 PyTorch / TensorFlow 等第三方深度学习框架的依赖**。
但为了让「青小团」能在真实硬件上训练（而非仅 CPU 玩具），
本模块引入**可插拔后端抽象**：

- ``numpy``  : 自研 NumPy 自动微分引擎（CPU，恒可用，默认）
- ``cuda``   : NVIDIA CUDA 后端（探测 cuPy / torch-cuda / nvidia-smi）
- ``opencl`` : AMD/Intel OpenCL 后端（探测 pyopencl，适配 Radeon 680M）
- ``auto``   : 自动探测，优先 cuda > opencl > numpy

后端只负责**设备/显存查询与算子派发提示**；真正的算子仍由
``simplcore.tensor`` 的 NumPy 引擎执行（自研自动微分）。GPU 后端
通过 cuPy（若安装）做矩阵乘等重算子的真·硬件加速；未安装时
自动降级为 NumPy（仍可跑，只是慢）。

设计原则
--------
1. 不引入对 PyTorch 的硬依赖（保持零框架依赖）。
2. 后端探测失败必须**安全降级**到 numpy，绝不让 import 崩溃。
3. 所有 GPU 显存/设备查询在不可用时返回 (0, 0) / 0，绝不抛异常。
"""
from __future__ import annotations

import os
import contextlib
from typing import Tuple

_BACKEND = None  # type: str | None

# 受支持的后端标识
_NATIVE = "numpy"
_CUDA = "cuda"
_OPENCL = "opencl"

# 探测结果缓存
_GPU_INFO = None  # (backend, device_name, total_bytes, free_bytes) | None
_CUPY = None      # cuPy 模块（可选）
_PYOPENCL = None  # pyopencl 模块（可选）


# ============================================================
# GPU 探测（安全降级）
# ============================================================
def _probe_cuda() -> dict:
    """探测 NVIDIA CUDA 可用性（通过 cuPy 或 nvidia-smi）。

    返回 dict: {available, name, total, free, via}
    """
    info = dict(available=False, name="", total=0, free=0, via="")

    # 1) 优先尝试 cuPy（真·GPU 加速，无 PyTorch 依赖）
    global _CUPY
    if _CUPY is None:
        with contextlib.suppress(Exception):
            import cupy as _cp  # type: ignore
            _CUPY = _cp
    if _CUPY is not None:
        try:
            dev = _CUPY.cuda.Device(0)
            mem = dev.mem_info  # (free, total) in bytes
            name = _CUPY.cuda.runtime.getDeviceProperties(0)["name"]
            info.update(available=True, name=name,
                        total=int(mem[1]), free=int(mem[0]), via="cupy")
            return info
        except Exception:
            pass

    # 2) 退而求其次：nvidia-smi（仅能拿到名称/粗略显存）
    with contextlib.suppress(Exception):
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode == 0 and out.stdout.strip():
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info.update(
                    available=True,
                    name=parts[0],
                    total=int(float(parts[1]) * 1024 * 1024),
                    free=int(float(parts[2]) * 1024 * 1024),
                    via="nvidia-smi",
                )
    return info


def _probe_opencl() -> dict:
    """探测 OpenCL 可用性（通过 pyopencl，适配 AMD Radeon 680M）。"""
    info = dict(available=False, name="", total=0, free=0, via="")

    global _PYOPENCL
    if _PYOPENCL is None:
        with contextlib.suppress(Exception):
            import pyopencl as _cl  # type: ignore
            _PYOPENCL = _cl
    if _PYOPENCL is not None:
        try:
            platforms = _PYOPENCL.get_platforms()
            for plat in platforms:
                for dev in plat.get_devices(
                        device_type=_PYOPENCL.device_type.GPU):
                    # 取第一个 GPU 设备（680M 通常是唯一 GPU）
                    mem = dev.global_mem_size
                    info.update(available=True, name=dev.name,
                                total=int(mem),
                                free=int(mem),  # OpenCL 无精确 free 查询
                                via="pyopencl")
                    return info
        except Exception:
            pass
    return info


def _detect_backend() -> str:
    """自动探测最佳后端：cuda > opencl > numpy。"""
    if _probe_cuda().get("available"):
        return _CUDA
    if _probe_opencl().get("available"):
        return _OPENCL
    return _NATIVE


def _gpu_info() -> dict:
    """返回当前 GPU 信息（缓存），不可用返回空 dict。"""
    global _GPU_INFO
    if _GPU_INFO is not None:
        return _GPU_INFO
    if get_backend() == _CUDA:
        _GPU_INFO = _probe_cuda()
    elif get_backend() == _OPENCL:
        _GPU_INFO = _probe_opencl()
    else:
        _GPU_INFO = dict(available=False, name="", total=0, free=0, via="")
    return _GPU_INFO


# ============================================================
# 公共 API
# ============================================================
def get_backend() -> str:
    """获取当前后端名称。"""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _NATIVE  # 默认 numpy；auto 由 set_backend 触发
    return _BACKEND


def set_backend(name: str) -> None:
    """设置后端。

    接受 ``auto`` / ``numpy`` / ``native`` / ``cuda`` / ``opencl``。
    - ``auto`` : 探测最佳可用后端
    - ``cuda`` / ``opencl`` : 仅当探测可用时启用，否则安全降级到 numpy 并告警
    - 仍拒绝 ``torch`` / ``pytorch``（项目零框架依赖）
    """
    global _BACKEND
    name = name.strip().lower()
    if name in ("torch", "pytorch"):
        raise ValueError(
            "simplcore.tensor 已彻底移除对 PyTorch 的依赖，"
            "不再支持 torch 后端。"
        )
    if name in ("numpy", "np", "native"):
        _BACKEND = _NATIVE
    elif name == "auto":
        _BACKEND = _detect_backend()
    elif name == _CUDA:
        if _probe_cuda().get("available"):
            _BACKEND = _CUDA
        else:
            import warnings
            warnings.warn("CUDA 不可用，已降级到 numpy 后端。")
            _BACKEND = _NATIVE
    elif name == _OPENCL:
        if _probe_opencl().get("available"):
            _BACKEND = _OPENCL
        else:
            import warnings
            warnings.warn("OpenCL 不可用，已降级到 numpy 后端。")
            _BACKEND = _NATIVE
    else:
        raise ValueError(
            f"未知后端: {name}，可选: auto / numpy / cuda / opencl")
    # 重置 GPU 信息缓存
    global _GPU_INFO
    _GPU_INFO = None


def is_torch_backend() -> bool:
    """当前是否使用 PyTorch 后端 — 恒为 False。"""
    return False


def is_numpy_backend() -> bool:
    """当前是否纯 NumPy 后端。"""
    return get_backend() == _NATIVE


def is_gpu_backend() -> bool:
    """当前是否 GPU 后端（cuda / opencl）。"""
    return get_backend() in (_CUDA, _OPENCL)


def get_device() -> str:
    """获取默认计算设备字符串。

    - numpy   -> "cpu"
    - cuda    -> "cuda:0"
    - opencl  -> "opencl:0"
    """
    b = get_backend()
    if b == _CUDA:
        return "cuda:0"
    if b == _OPENCL:
        return "opencl:0"
    return "cpu"


def gpu_memory_info() -> Tuple[int, int]:
    """返回 (free_bytes, total_bytes)。GPU 不可用时返回 (0, 0)。

    Tuple[int, int] 在运行时才 import typing，避免顶层循环依赖。
    """
    info = _gpu_info()
    return (int(info.get("free", 0)), int(info.get("total", 0)))


def gpu_name() -> str:
    """返回 GPU 设备名称；不可用返回 ''。"""
    return _gpu_info().get("name", "")


def gpu_total_gb() -> float:
    """返回 GPU 总显存（GB，浮点）。不可用返回 0.0。"""
    total = _gpu_info().get("total", 0)
    return total / (1024 ** 3) if total else 0.0


def can_fit_gb(required_gb: float) -> bool:
    """判断当前设备能否容纳 required_gb GB 的训练态（含余量 1.2x）。

    用于训练器自动选择：单专家能否进显存。
    """
    if is_gpu_backend():
        total = gpu_total_gb()
        return total > 0 and required_gb * 1.2 <= total
    # CPU 后端：用系统可用内存粗略判断（不精确，仅提示）
    with contextlib.suppress(Exception):
        import psutil  # type: ignore
        free = psutil.virtual_memory().available / (1024 ** 3)
        return free * 1.2 >= required_gb
    return True  # 未知时乐观放行


__all__ = [
    "get_backend",
    "set_backend",
    "is_torch_backend",
    "is_numpy_backend",
    "is_gpu_backend",
    "get_device",
    "gpu_memory_info",
    "gpu_name",
    "gpu_total_gb",
    "can_fit_gb",
]
