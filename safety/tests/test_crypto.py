# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""AES-256-GCM 加密引擎测试。"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safety.crypto import CryptoEngine


def test_roundtrip_bytes():
    engine, salt = CryptoEngine.from_passphrase("test-passphrase")
    data = b"SimplCore model weights \x00\x01\x02 secret"
    blob = engine.encrypt(data)
    assert blob != data
    assert engine.decrypt(blob) == data


def test_random_nonce_unique_ciphertext():
    engine, _ = CryptoEngine.from_passphrase("p")
    data = b"same plaintext"
    b1 = engine.encrypt(data)
    b2 = engine.encrypt(data)
    assert b1 != b2  # 随机 nonce → 相同明文密文不同


def test_tamper_detection():
    engine, _ = CryptoEngine.from_passphrase("p")
    blob = bytearray(engine.encrypt(b"secret payload"))
    blob[-1] ^= 0xFF  # 篡改最后一个字节（tag 区）
    with pytest.raises(ValueError, match="解密失败"):
        engine.decrypt(bytes(blob))


def test_wrong_key_fails():
    e1, _ = CryptoEngine.from_passphrase("correct")
    e2, _ = CryptoEngine.from_passphrase("wrong")
    blob = e1.encrypt(b"data")
    with pytest.raises(ValueError, match="解密失败"):
        e2.decrypt(blob)


def test_wrong_magic():
    engine, _ = CryptoEngine.from_passphrase("p")
    with pytest.raises(ValueError, match="魔数"):
        engine.decrypt(b"NOT_A_SIMPLCORE_BLOB" + b"\x00" * 64)


def test_key_length_validation():
    with pytest.raises(ValueError, match="32 字节"):
        CryptoEngine(b"short-key")


def test_torch_state_roundtrip():
    torch = pytest.importorskip("torch")
    engine, _ = CryptoEngine.from_passphrase("p")
    state = {"weight": torch.randn(4, 8), "bias": torch.zeros(8)}
    blob = engine.encrypt_torch_state(state)
    restored = engine.decrypt_torch_state(blob)
    assert set(restored.keys()) == set(state.keys())
    for k in state:
        assert torch.allclose(restored[k], state[k])


def test_file_roundtrip(tmp_path):
    engine, _ = CryptoEngine.from_passphrase("p")
    src = tmp_path / "weights.bin"
    enc = tmp_path / "weights.bin.enc"
    dec = tmp_path / "weights.bin.dec"
    src.write_bytes(b"\x01\x02\x03" * 100)
    engine.encrypt_file(str(src), str(enc))
    assert enc.read_bytes() != src.read_bytes()
    engine.decrypt_file(str(enc), str(dec))
    assert dec.read_bytes() == src.read_bytes()


def test_sha256_hex():
    engine, _ = CryptoEngine.from_passphrase("p")
    digest = engine.sha256_hex(b"abc")
    assert len(digest) == 64
    assert digest == engine.sha256_hex(b"abc")
