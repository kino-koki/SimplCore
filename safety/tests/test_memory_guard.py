# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""内存保护（金丝雀 / 越界防护 / 清理）测试。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safety.memory_guard import (
    MemoryGuard,
    GuardedBuffer,
    MemoryGuardError,
    wipe_bytes,
    wipe_memoryview,
    check_bounds,
)


def test_guarded_buffer_roundtrip():
    buf = GuardedBuffer(16, name="secret")
    buf.write(0, b"0123456789abcdef")
    assert buf.read(0, 16) == b"0123456789abcdef"
    assert buf.check() is True


def test_write_out_of_bounds():
    buf = GuardedBuffer(8, name="b")
    with pytest.raises(MemoryGuardError, match="越界写"):
        buf.write(4, b"012345")  # 4+6 > 8


def test_read_out_of_bounds():
    buf = GuardedBuffer(8, name="b")
    with pytest.raises(MemoryGuardError, match="越界读"):
        buf.read(6, 4)


def test_negative_offset():
    buf = GuardedBuffer(8, name="b")
    with pytest.raises(MemoryGuardError, match="越界写"):
        buf.write(-1, b"x")


def test_canary_detects_overflow():
    # 模拟越界写破坏前金丝雀：直接改 _prefix 内部（模拟外部指针越界）
    buf = GuardedBuffer(8, name="b")
    buf._prefix[0] ^= 0xFF
    with pytest.raises(MemoryGuardError, match="金丝雀被破坏"):
        buf.check()


def test_wipe_then_deny():
    buf = GuardedBuffer(8, name="secret")
    buf.write(0, b"SECRET!")
    buf.wipe()
    assert buf.wiped
    with pytest.raises(MemoryGuardError, match="禁止读取"):
        buf.read(0, 8)
    with pytest.raises(MemoryGuardError, match="禁止写入"):
        buf.write(0, b"x")


def test_memory_guard_register_and_check():
    mg = MemoryGuard()
    mg.register(16, "a")
    mg.register(32, "b")
    assert len(mg) == 2
    assert mg.check_all() == 2
    assert mg.wipe_all() == 2


def test_memory_guard_duplicate_name():
    mg = MemoryGuard()
    mg.register(8, "a")
    with pytest.raises(MemoryGuardError, match="已存在"):
        mg.register(8, "a")


def test_memory_guard_get_missing():
    mg = MemoryGuard()
    with pytest.raises(MemoryGuardError, match="未注册"):
        mg.get("nope")


def test_wipe_bytes():
    data = bytearray(b"super-secret")
    wipe_bytes(data)
    assert data == bytearray(len(data))


def test_wipe_memoryview():
    data = bytearray(b"secret")
    wipe_memoryview(memoryview(data))
    assert data == bytearray(len(data))


def test_check_bounds():
    assert check_bounds(0, 5) == 0
    assert check_bounds(4, 5) == 4
    assert check_bounds(-1, 5) == 4  # 负索引归一化
    with pytest.raises(MemoryGuardError, match="越界"):
        check_bounds(5, 5)
    with pytest.raises(MemoryGuardError, match="越界"):
        check_bounds(-6, 5)
    with pytest.raises(TypeError):
        check_bounds("1", 5)
