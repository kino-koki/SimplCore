# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""模型权重安全保存/加载管线测试（checkpointing）。

覆盖：
  1. 明文 save/load 往返（SimplCoreModel / SimplCoreMoEModel / 流式模型）；
  2. AES-256-GCM 加密 save/load 往返（口令正确）；
  3. 口令错误 → 解密失败；
  4. 权重被篡改 → 指纹不匹配；
  5. RBAC：未授权保存/加载被拒，授权通过；
  6. SimplCoreConfig.from_dict 往返。
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model import SimplCoreModel
from simplcore_pg.model_moe import SimplCoreMoEModel
from simplcore_pg.checkpointing import save_pretrained, from_pretrained
from safety.rbac import RBACManager


def _nano_config(**overrides) -> SimplCoreConfig:
    cfg = SimplCoreConfig.get_nano_config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


def _moe_config(**overrides) -> SimplCoreConfig:
    cfg = SimplCoreConfig(
        vocab_size=128, hidden_size=64, num_layers=2, num_waves=4,
        num_channels=4, channel_size=8, dmc_num_heads=2,
        ffn_intermediate_ratio=2.0, max_sequence_length=128,
        num_experts=2, experts_per_tok=2, moe_intermediate_size=32,
        use_rfp=True, use_dmc=True, use_localmix=True, dropout=0.0,
        chunk_size=32,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


def _assert_close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> None:
    assert torch.allclose(a, b, atol=tol, rtol=tol), f"max err={float((a - b).abs().max())}"


# ----------------------------------------------------------------------
# 1. 明文往返
# ----------------------------------------------------------------------
def test_plain_roundtrip_dense():
    cfg = _nano_config()
    model = SimplCoreModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    ref = model(x)
    with tempfile.TemporaryDirectory() as d:
        info = model.save_pretrained(d)
        assert info["encrypted"] is False
        assert os.path.exists(os.path.join(d, "config.json"))
        assert os.path.exists(os.path.join(d, "model.bin"))
        loaded = SimplCoreModel.from_pretrained(d)
        _assert_close(loaded(x), ref)
        # 参数逐项一致
        for k in model.state_dict():
            _assert_close(loaded.state_dict()[k], model.state_dict()[k])


def test_plain_roundtrip_moe():
    cfg = _moe_config()
    model = SimplCoreMoEModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    ref = model(x)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        loaded = SimplCoreMoEModel.from_pretrained(d)
        _assert_close(loaded(x), ref)


def test_plain_roundtrip_streaming():
    from simplcore_pg.model_stream import SimplCoreStreamingModel
    cfg = _moe_config(chunk_size=16)
    model = SimplCoreStreamingModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    ref = model(x)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        loaded = SimplCoreStreamingModel.from_pretrained(d)
        _assert_close(loaded(x), ref)


# ----------------------------------------------------------------------
# 2. 加密往返
# ----------------------------------------------------------------------
def test_encrypted_roundtrip():
    cfg = _moe_config()
    model = SimplCoreMoEModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    ref = model(x)
    with tempfile.TemporaryDirectory() as d:
        info = model.save_pretrained(d, passphrase="s3cret-口令-测试")
        assert info["encrypted"] is True
        assert os.path.exists(os.path.join(d, "model.enc"))
        assert not os.path.exists(os.path.join(d, "model.bin"))
        # 密文不是明文 torch.save 产物（不含可读魔数）
        with open(os.path.join(d, "model.enc"), "rb") as f:
            blob = f.read()
        assert b"torch" not in blob[:64]
        loaded = SimplCoreMoEModel.from_pretrained(d, passphrase="s3cret-口令-测试")
        _assert_close(loaded(x), ref)


def test_encrypted_wrong_passphrase():
    cfg = _moe_config()
    model = SimplCoreMoEModel(cfg)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, passphrase="correct")
        with pytest.raises(ValueError):
            SimplCoreMoEModel.from_pretrained(d, passphrase="wrong")


def test_encrypted_missing_passphrase():
    cfg = _moe_config()
    model = SimplCoreMoEModel(cfg)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, passphrase="correct")
        with pytest.raises(ValueError):
            SimplCoreMoEModel.from_pretrained(d)


# ----------------------------------------------------------------------
# 3. 篡改检测（指纹）
# ----------------------------------------------------------------------
def test_tamper_detection():
    cfg = _moe_config()
    model = SimplCoreMoEModel(cfg)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        # 篡改权重文件（翻转一个字节）
        wpath = os.path.join(d, "model.bin")
        with open(wpath, "rb") as f:
            data = bytearray(f.read())
        data[len(data) // 2] ^= 0xFF
        with open(wpath, "wb") as f:
            f.write(bytes(data))
        with pytest.raises(ValueError, match="指纹不匹配"):
            SimplCoreMoEModel.from_pretrained(d)


# ----------------------------------------------------------------------
# 4. RBAC
# ----------------------------------------------------------------------
def test_rbac_gate():
    cfg = _moe_config()
    model = SimplCoreMoEModel(cfg)
    rbac = RBACManager()
    rbac.assign_role("alice", "developer")   # 有 model.export / model.load
    # mallory 无任何角色 → deny-by-default，无 model.export / model.load
    with tempfile.TemporaryDirectory() as d:
        # 未授权保存被拒
        with pytest.raises(Exception):
            model.save_pretrained(d, rbac=rbac, user="mallory")
        # 授权保存成功
        model.save_pretrained(d, rbac=rbac, user="alice")
        # 未授权加载被拒
        with pytest.raises(Exception):
            SimplCoreMoEModel.from_pretrained(d, rbac=rbac, user="mallory")
        # 授权加载成功
        loaded = SimplCoreMoEModel.from_pretrained(d, rbac=rbac, user="alice")
        assert loaded is not None


# ----------------------------------------------------------------------
# 5. 配置往返
# ----------------------------------------------------------------------
def test_config_roundtrip():
    cfg = _moe_config()
    data = cfg.as_dict()
    restored = SimplCoreConfig.from_dict(data)
    assert restored == cfg
    # 多余字段被忽略
    restored2 = SimplCoreConfig.from_dict({**data, "extra_field": 123})
    assert restored2 == cfg


def test_checkpoint_metadata():
    cfg = _moe_config()
    model = SimplCoreMoEModel(cfg)
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d, metadata={"run_id": "exp-001", "note": "论文级回归"})
        with open(os.path.join(d, "config.json"), "r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        assert cfg_data["metadata"]["run_id"] == "exp-001"
        assert cfg_data["metadata"]["format"] == "simplcore-checkpoint-v1"
        assert cfg_data["metadata"]["encrypted"] is False
        assert len(cfg_data["metadata"]["sha256"]) == 64
