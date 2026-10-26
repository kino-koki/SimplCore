# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor autograd — 自动微分兼容层（自研，零框架依赖）
兼容 torch.autograd.set_detect_anomaly。原生引擎直接前向执行，
无反向异常检测机制，故全部为 no-op。
"""
from __future__ import annotations


class _DetectAnomalyCtx:
    """兼容 torch.autograd.set_detect_anomaly 的上下文管理器。"""

    def __init__(self, mode: bool = True) -> None:
        self._mode = mode

    def __enter__(self) -> "_DetectAnomalyCtx":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def set_detect_anomaly(mode: bool) -> _DetectAnomalyCtx:
    """设置是否检测反向传播异常（原生引擎忽略，恒返回 no-op 上下文）。"""
    return _DetectAnomalyCtx(mode)


def detect_anomaly():
    """兼容 torch.autograd.detect_anomaly()。"""
    return _DetectAnomalyCtx(True)


__all__ = ["set_detect_anomaly", "detect_anomaly"]
