# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Mix AMP — 混合精度训练支持（自研，零第三方框架依赖）
===============================================================
青小团自研张量引擎在 CPU/NumPy 上运行，统一使用 float32。
本模块提供与历史双后端兼容的接口，但所有 AMP 路径均为空操作
（自研引擎不涉及 GPU 混合精度），不引入任何 torch 依赖。
"""
from __future__ import annotations

from typing import Optional, Any


class GradScaler:
    """梯度缩放器 — 自研引擎下为空操作（恒定缩放因子 1.0）。"""

    def __init__(self, enabled: bool = True, init_scale: float = 2.0 ** 16):
        # 自研引擎始终以 float32 运行，无需梯度缩放。
        self._enabled = False
        self._scale = 1.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def get_scale(self) -> float:
        return 1.0

    @property
    def enabled(self) -> bool:
        return self._enabled


class autocast:
    """自动混合精度上下文管理器 — 自研引擎下为空操作（保持 float32）。"""

    def __init__(
        self,
        enabled: bool = True,
        dtype: Any = None,
        device_type: str = None,
    ):
        self._enabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


def get_amp_config(precision: str = "fp32") -> dict:
    """根据精度字符串获取 AMP 配置（自研引擎恒为 float32 直通）。"""
    return {"enabled": False, "dtype": None, "use_scaler": False}


__all__ = ["GradScaler", "autocast", "get_amp_config"]
