# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Device — 青小团设备管理
==================================
提供与 ``torch.device`` 等价的设备管理接口。

当前实现仅支持 CPU（通过 NumPy 后端），
接口设计兼容 PyTorch 的 device API，方便未来扩展 GPU 支持。

使用示例
--------
    from simplcore.tensor import Tensor, device, cpu

    x = Tensor([1.0, 2.0], device="cpu")
    d = device("cpu")
    print(d.type)  # 'cpu'
"""
from __future__ import annotations

from typing import Any, Optional


# ============================================================
# device 类
# ============================================================
class device:
    """设备标识类，兼容 ``torch.device`` 接口。

    Parameters
    ----------
    value : str or int
        设备字符串（如 ``"cpu"``）或设备索引。
    """

    def __init__(self, value: Any = "cpu") -> None:
        if isinstance(value, device):
            self.type: str = value.type
            self.index: Optional[int] = value.index
        elif isinstance(value, str):
            parts = value.split(":")
            self.type = parts[0].lower()
            self.index = int(parts[1]) if len(parts) > 1 else None
        elif isinstance(value, int):
            self.type = "cpu"
            self.index = value
        else:
            self.type = str(value).lower()
            self.index = None

        # 当前仅支持 CPU
        if self.type not in ("cpu",):
            raise RuntimeError(
                f"SimplCore 当前仅支持 'cpu' 设备，收到 {value!r}")

    def __repr__(self) -> str:
        if self.index is not None:
            return f"device(type='{self.type}', index={self.index})"
        return f"device(type='{self.type}')"

    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return str(self) == other
        return False

    def __hash__(self) -> int:
        return hash((self.type, self.index))


# ============================================================
# 设备查询函数
# ============================================================
def is_available(device_type: str = "cpu") -> bool:
    """检查设备是否可用。

    当前仅 CPU 可用。
    """
    return device_type.lower() == "cpu"


def cpu() -> device:
    """返回 CPU 设备。"""
    return device("cpu")


def cuda(device: Optional[int] = None) -> "device":
    """返回 CUDA 设备（当前不支持，抛出 RuntimeError）。"""
    raise RuntimeError("SimplCore 当前不支持 CUDA 设备")


# ============================================================
# 内存管理（兼容接口，实际为空操作）
# ============================================================
def empty_cache() -> None:
    """清空缓存（CPU 模式下为空操作）。"""
    pass


def memory_allocated(device: Optional[Any] = None) -> int:
    """已分配内存（CPU 模式下返回 0）。"""
    return 0


def memory_reserved(device: Optional[Any] = None) -> int:
    """已保留内存（CPU 模式下返回 0）。"""
    return 0


def max_memory_allocated(device: Optional[Any] = None) -> int:
    """最大已分配内存（CPU 模式下返回 0）。"""
    return 0


# ============================================================
# 设备设置
# ============================================================
_current_device: device = device("cpu")


def set_device(dev: Any) -> None:
    """设置当前默认设备。"""
    global _current_device
    _current_device = device(dev) if not isinstance(dev, device) else dev


def get_device() -> device:
    """获取当前默认设备。"""
    return _current_device


def device_count() -> int:
    """返回可用设备数量。"""
    return 1  # 仅 CPU
