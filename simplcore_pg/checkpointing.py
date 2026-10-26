# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""模型权重安全保存/加载管线（save_pretrained / from_pretrained）。

把 ``safety/`` 安全模块接入模型权重主链路（对应 HONEST_STATUS.md 待落地项）：
  * **静态加密**：传 ``passphrase`` 时，state_dict 经 ``CryptoEngine``
    (AES-256-GCM) 加密落盘为 ``model.enc``；不传则明文 ``model.bin``；
  * **RBAC 鉴权**：传 ``rbac`` + ``user`` 时，保存要求 ``model.export``、
    加载要求 ``model.load``（deny-by-default，未授权抛 ``RBACError``）；
  * **内存保护**：解密后的明文 state_dict 驻留 ``GuardedBuffer``，加载完成
    即覆写清零（用后即焚），并校验金丝雀防越界写；
  * **完整性**：配置快照记录权重 SHA-256 指纹，加载时校验，防篡改/错配。

磁盘布局（目录）::

    <dir>/
      config.json      # SimplCoreConfig.as_dict() 快照 + 元数据（含指纹）
      model.bin        # 明文权重（passphrase=None）
      model.enc        # AES-256-GCM 加密权重（passphrase 非空）

兼容：所有 ``nn.Module`` 子类（含 SimplCoreModel / SimplCoreMoEModel /
SimplCoreStreamingModel）。``cryptography`` 为可选依赖，未安装时加密路径
抛 ``ImportError``（绝不降级为不安全的自研加密）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, Dict, Optional, Type

import torch
import torch.nn as nn

from .config import SimplCoreConfig
from .exceptions import ModelError

_CONFIG_FILENAME = "config.json"
_PLAIN_FILENAME = "model.bin"
_ENCRYPTED_FILENAME = "model.enc"
# 与 safety/crypto.py 的 encrypt_torch_state 内部 AAD 保持一致
_AAD = b"torch_state_dict_v1"


# ----------------------------------------------------------------------
# 配置序列化
# ----------------------------------------------------------------------
def _config_to_dict(config: SimplCoreConfig) -> Dict[str, Any]:
    return config.as_dict()


def _config_from_dict(data: Dict[str, Any]) -> SimplCoreConfig:
    """从 dict 重建配置（兼容旧字段缺失/多余，仅取已知字段）。"""
    return SimplCoreConfig.from_dict(data)


# ----------------------------------------------------------------------
# 权重指纹
# ----------------------------------------------------------------------
def _state_fingerprint(state_dict: Dict[str, torch.Tensor]) -> str:
    """对 state_dict 计算稳定 SHA-256 指纹（键序无关）。"""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        t = state_dict[key]
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------
# 保存
# ----------------------------------------------------------------------
def save_pretrained(
    model: nn.Module,
    directory: str,
    *,
    passphrase: Optional[str] = None,
    rbac: Optional[Any] = None,
    user: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把模型保存到目录（可选加密 / RBAC / 元数据）。

    Args:
        model: 任意 ``nn.Module``（其 ``config`` 属性须为 ``SimplCoreConfig``）。
        directory: 目标目录（不存在则创建）。
        passphrase: 非空则 AES-256-GCM 加密权重为 ``model.enc``；
            为空则明文 ``model.bin``。
        rbac: 可选 ``RBACManager``；与 ``user`` 同时给出时强制 ``model.export``。
        user: RBAC 用户标识。
        metadata: 附加元数据（写入 config.json 的 ``metadata`` 字段）。

    Returns:
        dict: 落盘摘要（路径、加密与否、指纹、参数量）。

    Raises:
        RBACError: 未授权保存。
        ImportError: 需要加密但未安装 cryptography。
    """
    if rbac is not None:
        if user is None:
            raise ValueError("启用 rbac 时必须提供 user。")
        rbac.require(user, "model.export")

    config = getattr(model, "config", None)
    if not isinstance(config, SimplCoreConfig):
        raise ModelError("模型缺少 SimplCoreConfig 类型的 config 属性。")

    os.makedirs(directory, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    fingerprint = _state_fingerprint(state)

    if passphrase:
        from safety.crypto import CryptoEngine

        engine, salt = CryptoEngine.from_passphrase(passphrase)
        blob = engine.encrypt_torch_state(state)
        enc_path = os.path.join(directory, _ENCRYPTED_FILENAME)
        with open(enc_path, "wb") as f:
            f.write(blob)
        weights_file = _ENCRYPTED_FILENAME
        encrypted = True
        # 密钥派生盐必须随 checkpoint 持久化（crypto.encrypt 的密文内嵌盐是
        # 每次加密的随机盐，与密钥派生盐无关），否则加载端无法重派生同一密钥
        salt_hex = salt.hex()
    else:
        plain_path = os.path.join(directory, _PLAIN_FILENAME)
        torch.save(state, plain_path)
        weights_file = _PLAIN_FILENAME
        encrypted = False
        salt_hex = None

    cfg_data = _config_to_dict(config)
    cfg_data["metadata"] = {
        "format": "simplcore-checkpoint-v1",
        "encrypted": encrypted,
        "weights_file": weights_file,
        "sha256": fingerprint,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "kdf_salt_hex": salt_hex,
        **(metadata or {}),
    }
    cfg_path = os.path.join(directory, _CONFIG_FILENAME)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, ensure_ascii=False, indent=2)

    return {
        "directory": directory,
        "config": cfg_path,
        "weights": os.path.join(directory, weights_file),
        "encrypted": encrypted,
        "sha256": fingerprint,
        "num_parameters": cfg_data["metadata"]["num_parameters"],
    }


# ----------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------
def from_pretrained(
    model_class: Type[nn.Module],
    directory: str,
    *,
    passphrase: Optional[str] = None,
    rbac: Optional[Any] = None,
    user: Optional[str] = None,
    map_location: str = "cpu",
) -> nn.Module:
    """从目录加载模型（自动识别加密/明文，可选 RBAC / 内存清理）。

    Args:
        model_class: 模型类（如 ``SimplCoreMoEModel``）。
        directory: 保存目录。
        passphrase: 解密口令（保存时加密则必须提供）。
        rbac: 可选 ``RBACManager``；与 ``user`` 同时给出时强制 ``model.load``。
        user: RBAC 用户标识。
        map_location: ``torch.load`` 的 map_location。

    Returns:
        nn.Module: 加载后的模型（已 ``eval()``）。

    Raises:
        RBACError: 未授权加载。
        ValueError: 口令错误 / 数据被篡改 / 指纹不匹配。
        FileNotFoundError: 目录缺少 config.json 或权重文件。
    """
    if rbac is not None:
        if user is None:
            raise ValueError("启用 rbac 时必须提供 user。")
        rbac.require(user, "model.load")

    cfg_path = os.path.join(directory, _CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"缺少 {_CONFIG_FILENAME}: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    config = _config_from_dict(cfg_data)
    meta = cfg_data.get("metadata") or {}
    encrypted = bool(meta.get("encrypted", False))
    weights_file = meta.get("weights_file") or (_ENCRYPTED_FILENAME if encrypted else _PLAIN_FILENAME)
    weights_path = os.path.join(directory, weights_file)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"缺少权重文件: {weights_path}")

    if encrypted:
        if not passphrase:
            raise ValueError("该 checkpoint 已加密，必须提供 passphrase。")
        from safety.crypto import CryptoEngine
        from safety.memory_guard import GuardedBuffer, MemoryGuardError

        # 复用保存时持久化的密钥派生盐（kdf_salt_hex），否则无法重派生同一密钥
        salt_hex = meta.get("kdf_salt_hex")
        if not salt_hex:
            raise ValueError("checkpoint 缺少 kdf_salt_hex，无法重派生密钥。")
        engine, _ = CryptoEngine.from_passphrase(passphrase, salt=bytes.fromhex(salt_hex))
        with open(weights_path, "rb") as f:
            blob = f.read()
        # 解密后的明文驻留带金丝雀的 GuardedBuffer，加载后立即覆写清零
        plain = engine.decrypt(blob, aad=_AAD)
        buf = GuardedBuffer(len(plain), name="decrypted_state")
        buf.write(0, plain)
        buf.check()
        try:
            state = torch.load(io.BytesIO(buf.read(0, len(plain))),
                               map_location=map_location, weights_only=True)
        finally:
            buf.wipe()  # 用后即焚
    else:
        state = torch.load(weights_path, map_location=map_location, weights_only=True)

    expected = meta.get("sha256")
    if expected:
        actual = _state_fingerprint(state)
        if actual != expected:
            raise ValueError(
                f"权重指纹不匹配（可能被篡改或与 config.json 不匹配）："
                f"期望 {expected[:16]}… 实际 {actual[:16]}…"
            )

    model = model_class(config)
    model.load_state_dict(state)
    model.eval()
    return model


class CheckpointMixin:
    """模型便捷方法：``save_pretrained`` / ``from_pretrained``（委托模块函数）。

    用法：模型类继承 ``(CheckpointMixin, nn.Module)`` 即获得：
      * ``model.save_pretrained(dir, passphrase=..., rbac=..., user=...)``；
      * ``ModelClass.from_pretrained(dir, passphrase=..., rbac=..., user=...)``。
    """

    def save_pretrained(
        self,
        directory: str,
        *,
        passphrase: Optional[str] = None,
        rbac: Optional[Any] = None,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """保存模型（可选 AES-256-GCM 加密 / RBAC 鉴权 / 元数据）。"""
        return save_pretrained(
            self, directory, passphrase=passphrase, rbac=rbac,
            user=user, metadata=metadata,
        )

    @classmethod
    def from_pretrained(
        cls,
        directory: str,
        *,
        passphrase: Optional[str] = None,
        rbac: Optional[Any] = None,
        user: Optional[str] = None,
        map_location: str = "cpu",
    ) -> "nn.Module":
        """从目录加载模型（自动识别加密/明文，可选 RBAC / 内存清理）。"""
        return from_pretrained(
            cls, directory, passphrase=passphrase, rbac=rbac,
            user=user, map_location=map_location,
        )


__all__ = ["save_pretrained", "from_pretrained", "CheckpointMixin",
           "_config_from_dict", "_config_to_dict"]
