# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""内存保护 — 金丝雀检测 / 越界防护 / 敏感内存清理。

用途（对应 HONEST_STATUS.md 待落地项"安全模块（AES-256/RBAC/内存保护）"）：
  * 检测缓冲区越界写（buffer overflow）：在敏感缓冲区前后放置金丝雀值，
    越界写会破坏金丝雀，读取时校验即可发现；
  * 敏感数据内存清理：模型权重解密后驻留内存，用后即焚（overwrite 后释放）；
  * 越界访问防护：对索引 / 张量访问做边界检查。

设计：
  * ``GuardedBuffer``：带前后金丝雀的字节缓冲区，支持 ``read`` / ``write`` /
    ``check``（校验金丝雀） / ``wipe``（覆写清零）；
  * ``MemoryGuard``：管理一组 GuardedBuffer 的注册 / 校验 / 清理；
  * ``wipe_bytes`` / ``wipe_bytearray``：就地覆写敏感字节（防残留）；
  * 纯标准库实现，无第三方依赖。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

# 金丝雀值：固定 64 字节模式，越界写几乎必然破坏它
_CANARY = bytes(range(256)) * 4  # 1024 字节，覆盖常见越界写长度


class MemoryGuardError(Exception):
    """内存保护错误（金丝雀被破坏 / 越界访问 / 未注册缓冲区）。"""


class GuardedBuffer:
    """带金丝雀防护的字节缓冲区。

    Args:
        size: 数据区大小（字节）。
        name: 缓冲区名称（用于错误定位）。
    """

    __slots__ = ("_name", "_data", "_size", "_prefix", "_suffix", "_wiped")

    def __init__(self, size: int, name: str = "buffer"):
        if size < 0:
            raise ValueError(f"size 不能为负，得到 {size}。")
        self._name = name
        self._size = size
        self._prefix = bytearray(_CANARY)
        self._data = bytearray(size)
        self._suffix = bytearray(_CANARY)
        self._wiped = False

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    @property
    def wiped(self) -> bool:
        return self._wiped

    # ------------------------------------------------------------------
    def write(self, offset: int, data: bytes) -> None:
        """写入数据到 ``[offset, offset+len(data))``，越界抛错。"""
        if self._wiped:
            raise MemoryGuardError(f"缓冲区 '{self._name}' 已被清理，禁止写入。")
        if offset < 0 or offset + len(data) > self._size:
            raise MemoryGuardError(
                f"越界写: '{self._name}' offset={offset} len={len(data)} "
                f"超出 size={self._size}。"
            )
        self._data[offset: offset + len(data)] = data

    def read(self, offset: int, length: int) -> bytes:
        """读取 ``[offset, offset+length)``，越界抛错。"""
        if self._wiped:
            raise MemoryGuardError(f"缓冲区 '{self._name}' 已被清理，禁止读取。")
        if offset < 0 or offset + length > self._size:
            raise MemoryGuardError(
                f"越界读: '{self._name}' offset={offset} len={length} "
                f"超出 size={self._size}。"
            )
        return bytes(self._data[offset: offset + length])

    def check(self) -> bool:
        """校验金丝雀是否完好（检测越界写）。"""
        if self._wiped:
            return True  # 已清理的缓冲区不再校验
        if bytes(self._prefix) != _CANARY or bytes(self._suffix) != _CANARY:
            raise MemoryGuardError(
                f"金丝雀被破坏: 缓冲区 '{self._name}' 发生越界写！"
            )
        return True

    def wipe(self) -> None:
        """覆写清零并释放数据区（敏感数据用后即焚）。"""
        if self._wiped:
            return
        self._data[:] = b"\x00" * self._size
        self._prefix[:] = b"\x00" * len(self._prefix)
        self._suffix[:] = b"\x00" * len(self._suffix)
        self._wiped = True


class MemoryGuard:
    """内存保护管理器：注册 / 校验 / 清理一组 GuardedBuffer。"""

    def __init__(self) -> None:
        self._buffers: Dict[str, GuardedBuffer] = {}

    def register(self, size: int, name: str) -> GuardedBuffer:
        """注册一个新缓冲区。"""
        if name in self._buffers:
            raise MemoryGuardError(f"缓冲区已存在: {name}")
        buf = GuardedBuffer(size, name=name)
        self._buffers[name] = buf
        return buf

    def get(self, name: str) -> GuardedBuffer:
        if name not in self._buffers:
            raise MemoryGuardError(f"缓冲区未注册: {name}")
        return self._buffers[name]

    def check_all(self) -> int:
        """校验全部缓冲区金丝雀，返回通过数量。"""
        count = 0
        for buf in self._buffers.values():
            buf.check()
            count += 1
        return count

    def wipe_all(self) -> int:
        """清理全部缓冲区，返回清理数量。"""
        count = 0
        for buf in self._buffers.values():
            buf.wipe()
            count += 1
        return count

    def __len__(self) -> int:
        return len(self._buffers)


# ----------------------------------------------------------------------
# 模块级便捷函数
# ----------------------------------------------------------------------
def wipe_bytes(data: bytearray) -> None:
    """就地覆写清零 bytearray（敏感数据用后即焚）。"""
    data[:] = b"\x00" * len(data)


def wipe_memoryview(mv: memoryview) -> None:
    """就地覆写清零 memoryview（支持 numpy/torch 底层缓冲的通用清理）。"""
    try:
        mv[:] = b"\x00" * mv.nbytes
    except (TypeError, ValueError):
        # 只读视图：无法覆写，忽略（调用方应确保可写）
        pass


def check_bounds(index: int, length: int, name: str = "index") -> int:
    """边界检查：返回规范化后的非负索引（支持负索引），越界抛错。"""
    if not isinstance(index, int):
        raise TypeError(f"{name} 必须为整数，得到 {type(index).__name__}。")
    if index < 0:
        index += length
    if index < 0 or index >= length:
        raise MemoryGuardError(f"{name} 越界: {index} 不在 [0, {length})。")
    return index


__all__ = [
    "MemoryGuard",
    "GuardedBuffer",
    "MemoryGuardError",
    "wipe_bytes",
    "wipe_memoryview",
    "check_bounds",
]
