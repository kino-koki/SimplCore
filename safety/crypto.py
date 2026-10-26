# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""AES-256-GCM 加密引擎 — 模型权重 / 敏感数据静态加密。

用途（对应 HONEST_STATUS.md 待落地项"安全模块（AES-256/RBAC/内存保护）"）：
  * 模型权重静态加密：checkpoint 落盘前加密，防止权重被直接读取/窃取；
  * 敏感配置加密：API key / 数据路径等敏感字段；
  * 认证加密 (AEAD)：AES-256-GCM 同时保证机密性、完整性、真实性。

设计：
  * 密钥派生：PBKDF2-HMAC-SHA256（iterations=600_000，OWASP 推荐量级），
    从口令 + 随机盐派生 32 字节 AES-256 密钥；
  * 随机 nonce：每次加密生成 12 字节随机 nonce（GCM 标准），
    输出格式 ``salt(16) | nonce(12) | ciphertext | tag(16)``；
  * torch 集成：``encrypt_torch_state`` 把 state_dict 序列化为字节流后加密，
    解密还原为 state_dict（torch 可选依赖，缺失时仅支持 bytes 接口）。

依赖：``cryptography``（已加入 requirements.txt）。若未安装，构造时抛出
``ImportError`` 并给出安装指引（绝不降级为不安全的自研加密）。
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover - 环境缺库时的降级路径
    _HAS_CRYPTO = False

# 输出布局常量
_SALT_LEN = 16
_NONCE_LEN = 12
_TAG_LEN = 16
_HEADER_LEN = _SALT_LEN + _NONCE_LEN
_MAGIC = b"SCENC1"  # 格式魔数，防止误解密其它数据


class CryptoEngine:
    """AES-256-GCM 加密引擎。

    Args:
        key: 32 字节密钥（由 :meth:`from_passphrase` 或直接传入）。
    """

    def __init__(self, key: bytes):
        if not _HAS_CRYPTO:
            raise ImportError(
                "CryptoEngine 需要 'cryptography' 库。请安装: "
                "pip install cryptography"
            )
        if len(key) != 32:
            raise ValueError(f"AES-256 密钥必须为 32 字节，得到 {len(key)}。")
        self._aesgcm = AESGCM(key)

    # ------------------------------------------------------------------
    # 密钥派生
    # ------------------------------------------------------------------
    @classmethod
    def from_passphrase(cls, passphrase: str, salt: Optional[bytes] = None,
                        iterations: int = 600_000) -> "CryptoEngine":
        """从口令派生密钥（PBKDF2-HMAC-SHA256）。

        Args:
            passphrase: 用户口令（不直接作为密钥使用）。
            salt: 可选盐（16 字节）。不传则随机生成并返回，调用方需持久化。
            iterations: PBKDF2 迭代次数（默认 600k，OWASP 推荐量级）。
        Returns:
            (engine, salt) —— 若 salt 为 None，返回的 salt 需随密文保存。
        """
        if not _HAS_CRYPTO:
            raise ImportError(
                "CryptoEngine 需要 'cryptography' 库。请安装: pip install cryptography"
            )
        if salt is None:
            salt = os.urandom(_SALT_LEN)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        key = kdf.derive(passphrase.encode("utf-8"))
        return cls(key), salt

    # ------------------------------------------------------------------
    # bytes 级加解密
    # ------------------------------------------------------------------
    def encrypt(self, data: bytes, aad: Optional[bytes] = None) -> bytes:
        """加密字节流。输出: magic(6) | salt(16) | nonce(12) | ct | tag(16)。

        Args:
            data: 明文。
            aad: 附加认证数据（可选，参与认证但不加密，如版本号/文件路径）。
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"data 必须为 bytes，得到 {type(data).__name__}。")
        salt = os.urandom(_SALT_LEN)
        nonce = os.urandom(_NONCE_LEN)
        ct = self._aesgcm.encrypt(nonce, bytes(data), aad)
        return _MAGIC + salt + nonce + ct

    def decrypt(self, blob: bytes, aad: Optional[bytes] = None) -> bytes:
        """解密 :meth:`encrypt` 的输出。认证失败（篡改/错误密钥）抛 ``ValueError``。"""
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(f"blob 必须为 bytes，得到 {type(blob).__name__}。")
        blob = bytes(blob)
        if not blob.startswith(_MAGIC):
            raise ValueError("数据格式错误：缺少 SimplCore 加密魔数（非本引擎产物）。")
        if len(blob) < _HEADER_LEN + _TAG_LEN + 1:
            raise ValueError("数据长度不足，无法解密。")
        salt = blob[len(_MAGIC): len(_MAGIC) + _SALT_LEN]
        nonce = blob[len(_MAGIC) + _SALT_LEN: len(_MAGIC) + _HEADER_LEN]
        ct = blob[len(_MAGIC) + _HEADER_LEN:]
        try:
            return self._aesgcm.decrypt(nonce, ct, aad)
        except Exception as e:  # cryptography 抛 InvalidTag
            raise ValueError(
                "解密失败：认证标签不匹配（数据被篡改或密钥错误）。"
            ) from e

    # ------------------------------------------------------------------
    # torch state_dict 加解密（可选依赖 torch）
    # ------------------------------------------------------------------
    def encrypt_torch_state(self, state_dict: Dict[str, Any]) -> bytes:
        """把 torch state_dict 加密为字节流。

        序列化使用 torch.save（含张量元数据），再整体 AES-256-GCM 加密。
        返回的字节流可直接落盘；解密后经 torch.load 还原。
        """
        import io
        import torch

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return self.encrypt(buf.getvalue(), aad=b"torch_state_dict_v1")

    def decrypt_torch_state(self, blob: bytes) -> Dict[str, Any]:
        """解密 :meth:`encrypt_torch_state` 的输出并还原 state_dict。"""
        import io
        import torch

        plain = self.decrypt(blob, aad=b"torch_state_dict_v1")
        return torch.load(io.BytesIO(plain), map_location="cpu", weights_only=True)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------
    def encrypt_file(self, src_path: str, dst_path: str,
                     aad: Optional[bytes] = None) -> int:
        """加密整个文件到新路径，返回写入字节数。"""
        with open(src_path, "rb") as f:
            data = f.read()
        blob = self.encrypt(data, aad=aad)
        with open(dst_path, "wb") as f:
            f.write(blob)
        return len(blob)

    def decrypt_file(self, src_path: str, dst_path: str,
                     aad: Optional[bytes] = None) -> int:
        """解密文件到新路径，返回写入字节数。"""
        with open(src_path, "rb") as f:
            blob = f.read()
        data = self.decrypt(blob, aad=aad)
        with open(dst_path, "wb") as f:
            f.write(data)
        return len(data)

    @staticmethod
    def sha256_hex(data: bytes) -> str:
        """SHA-256 摘要（用于完整性校验 / 指纹）。"""
        return hashlib.sha256(data).hexdigest()


__all__ = ["CryptoEngine"]
