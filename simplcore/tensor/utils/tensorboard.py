# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor TensorBoard — 原生日志兼容层（零框架依赖）
兼容 torch.utils.tensorboard.SummaryWriter，将标量/直方图写入本地
JSONL 日志文件，无需安装 tensorboard 包。在生产环境可对接 ROCm 的
rocm-smi 或 Prometheus 指标采集。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class SummaryWriter:
    """兼容 torch.utils.tensorboard.SummaryWriter 的最小实现。

    所有 add_* 方法将事件以 JSON 行追加到本地日志文件（默认 ``tb_log.jsonl``）。
    无 GPU / 无 tensorboard 环境下完全可用。
    """

    def __init__(self, log_dir: Optional[str] = None, comment: str = "",
                 flush_secs: int = 120) -> None:
        self._log_dir = log_dir or ("runs" + (f"/{comment}" if comment else ""))
        os.makedirs(self._log_dir, exist_ok=True)
        self._path = os.path.join(self._log_dir, "events.jsonl")
        self._flush_secs = flush_secs
        self._buffer: list = []
        self._step: int = 0
        self._global_count: Dict[str, int] = {}
        # 写入文件头
        self._write_raw({"type": "meta", "log_dir": self._log_dir, "t": time.time()})

    # ---- 内部 ----
    def _write_raw(self, record: Dict[str, Any]) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _emit(self, tag: str, value: float, step: Optional[int]) -> None:
        if step is None:
            step = self._global_count.get(tag, 0)
            self._global_count[tag] = step + 1
        self._write_raw({
            "type": "scalar",
            "tag": tag,
            "value": float(value),
            "step": step,
            "t": time.time(),
        })

    # ---- 公开 API ----
    def add_scalar(self, tag: str, scalar_value: float,
                   global_step: Optional[int] = None, walltime: Optional[float] = None) -> None:
        self._emit(tag, float(scalar_value), global_step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float],
                    global_step: Optional[int] = None, walltime: Optional[float] = None) -> None:
        for k, v in tag_scalar_dict.items():
            self._emit(f"{main_tag}/{k}", float(v), global_step)

    def add_histogram(self, tag: str, values, global_step: Optional[int] = None,
                      bins: str = "tensorflow", walltime: Optional[float] = None) -> None:
        import numpy as np
        arr = np.asarray(values, dtype=np.float64).ravel()
        hist = {
            "type": "histogram",
            "tag": tag,
            "step": global_step if global_step is not None else self._step,
            "min": float(arr.min()) if arr.size else 0.0,
            "max": float(arr.max()) if arr.size else 0.0,
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
            "t": time.time(),
        }
        self._write_raw(hist)

    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None) -> None:
        self._write_raw({
            "type": "text", "tag": tag,
            "text": str(text_string),
            "step": global_step, "t": time.time(),
        })

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.flush()


__all__ = ["SummaryWriter"]
