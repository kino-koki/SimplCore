# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor AMP — 自动混合精度兼容层（自研，零框架依赖）
原生引擎以 float32 计算，GradScaler / autocast 均为直通 no-op，
保证与 torch.cuda.amp.GradScaler / torch.autocast 接口兼容。
"""
from __future__ import annotations

from typing import Optional


class GradScaler:
    """兼容 torch.cuda.amp.GradScaler 的直通实现。

    原生后端无 FP16/BF16 损失的数值不稳问题，因此 scaler 恒为 1.0，
    scale/unscale/step/update 均为无操作。
    """

    def __init__(self, enabled: bool = True, init_scale: float = 65536.0,
                 growth_factor: float = 2.0, backoff_factor: float = 0.5,
                 growth_interval: int = 2000) -> None:
        self._enabled = enabled
        self._scale = float(init_scale) if enabled else 1.0

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def scale(self, loss):
        """按当前缩放因子缩放 loss（原生模式直接返回）。"""
        return loss

    def unscale_(self, optimizer) -> None:
        """反缩放梯度（原生模式无操作）。"""
        return

    def step(self, optimizer) -> None:
        """以原生精度执行优化器步进。"""
        optimizer.step()

    def update(self, new_scale: Optional[float] = None) -> None:
        """更新缩放因子（原生模式无操作）。"""
        return

    def get_scale(self) -> float:
        return self._scale

    def load_state_dict(self, state_dict: dict) -> None:
        self._scale = state_dict.get("scale", 1.0)

    def state_dict(self) -> dict:
        return {"scale": self._scale, "growth_factor": 2.0,
                "backoff_factor": 0.5, "growth_interval": 2000}


class autocast:
    """兼容 torch.autocast 的上下文管理器（原生模式为 no-op）。

    用法::

        with autocast(device_type="cpu"):
            y = model(x)
    """

    def __init__(self, device_type: str = "cpu", dtype=None,
                 enabled: bool = True, cache_enabled: bool = True) -> None:
        self._device_type = device_type
        self._dtype = dtype
        self._enabled = enabled

    def __enter__(self) -> "autocast":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def __call__(self, fn):
        import functools
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return wrapper


__all__ = ["GradScaler", "autocast"]
