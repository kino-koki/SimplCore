# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore IO — 青小团序列化/存储
==================================
提供与 ``torch.save`` / ``torch.load`` 等价的序列化接口，
基于 NumPy 的 ``.npz`` 格式和 Python ``pickle``。

支持序列化：
* Module 的 ``state_dict()``
* Optimizer 的 ``state_dict()``
* 任意可 pickle 的 Python 对象

使用示例
--------
    from simplcore.tensor import Tensor, Linear, AdamW, save, load

    model = Linear(10, 5)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 保存
    save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "checkpoint.pt")

    # 加载
    checkpoint = load("checkpoint.pt")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Any, Dict, Optional


# ============================================================
# 保存
# ============================================================
def save(obj: Any, f: str) -> None:
    """将对象序列化到文件。

    Parameters
    ----------
    obj : Any
        可序列化的对象（state_dict、优化器状态等）。
    f : str
        文件路径。
    """
    # 确保目录存在
    dir_path = os.path.dirname(f)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 将 Tensor 对象转换为 numpy 数组
    serializable = _to_serializable(obj)

    with open(f, "wb") as fh:
        pickle.dump(serializable, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _to_serializable(obj: Any) -> Any:
    """递归将 Tensor 转换为 numpy 数组，使其可被 pickle 序列化。"""
    # 延迟导入避免循环
    from .tensor import Tensor

    if isinstance(obj, Tensor):
        return {"__tensor__": True, "data": obj.data, "requires_grad": obj.requires_grad}
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_to_serializable(v) for v in obj)
    else:
        return obj


# ============================================================
# 加载
# ============================================================
def load(f: str, map_location: Any = None, weights_only: bool = False) -> Any:
    """从文件加载序列化对象。

    Parameters
    ----------
    f : str
        文件路径。
    map_location : Any
        设备映射（兼容 PyTorch 接口，当前忽略）。
    weights_only : bool
        仅加载权重（兼容 PyTorch 接口，当前忽略）。

    Returns
    -------
    Any
        反序列化后的对象。
    """
    with open(f, "rb") as fh:
        obj = pickle.load(fh)

    return _from_serializable(obj)


def _from_serializable(obj: Any) -> Any:
    """递归将序列化数据转换回 Tensor。"""
    from .tensor import Tensor

    if isinstance(obj, dict) and obj.get("__tensor__"):
        return Tensor(obj["data"], requires_grad=obj.get("requires_grad", False))
    elif isinstance(obj, dict):
        return {k: _from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_from_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_from_serializable(v) for v in obj)
    else:
        return obj


# ============================================================
# 便捷函数
# ============================================================
def save_state_dict(state_dict: Dict[str, np.ndarray], f: str) -> None:
    """专门保存 state_dict（使用 npz 格式，更高效）。"""
    dir_path = os.path.dirname(f)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # 将所有 numpy 数组保存为 npz
    arrays = {}
    for key, val in state_dict.items():
        if hasattr(val, "data"):
            arrays[key] = np.asarray(val.data)
        else:
            arrays[key] = np.asarray(val)
    np.savez(f, **arrays)


def load_state_dict(f: str) -> Dict[str, np.ndarray]:
    """从 npz 文件加载 state_dict。"""
    data = np.load(f, allow_pickle=False)
    return {key: data[key] for key in data.files}
