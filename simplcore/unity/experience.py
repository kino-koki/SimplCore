# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""把真实经验文本转成 Unity 后台训练 batch（域专属 loader）。

SimplUnity 的后台 worker 只要求 ``loader`` 产出 ``(x, y)``；难点不在训练，
而在"经验从哪来"。本模块给出最小、可审计的转换：

* 输入是**某域的经验文本**（例如 ``SimplModelMemory.experience`` 落下的
  prompt/response、代码沙箱的成功轨迹、环境交互日志）；
* 逐条编码成 token，按 ``seq_len`` 滑窗切 next-token 对；
* ``repeat`` 控制过采样轮数；``max_windows`` 防止一个域无限膨胀。

它不是"自进化"本身：自进化 = 经验采集 → 本 loader → 候选专家训练 →
无回归门 → 发布/回滚（``UnitySelfEvolution``）。这里只负责第一步的**数据面**。
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import torch

__all__ = ["ExperienceLoader"]


class ExperienceLoader:
    """把文本列表切成 ``(x, y)`` next-token batch。

    Args:
        codec: 具备 ``encode(text) -> Sequence[int]`` 的编码器。
        texts: 该域的经验文本（不要与其他域混用；域隔离是专家分工的前提）。
        seq_len: 训练窗口长度（token 数）。
        batch_size: 每个 batch 的窗口数。
        stride: 滑窗步长；默认 ``seq_len``（不重叠）。
        repeat: 过采样轮数。
        max_windows: 最多产出多少个窗口（跨所有文本）。
        device: 可选，把 batch 移到该设备。
    """

    def __init__(self, codec, texts: Sequence[str], *, seq_len: int = 512,
                 batch_size: int = 1, stride: Optional[int] = None,
                 repeat: int = 1, max_windows: Optional[int] = None,
                 device=None) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.codec = codec
        self.texts = [t for t in texts if isinstance(t, str) and t]
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.stride = int(stride or seq_len)
        self.repeat = max(1, int(repeat))
        self.max_windows = max_windows
        self.device = device
        self._windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._build()

    def _build(self) -> None:
        need = self.seq_len + 1
        for text in self.texts:
            ids = list(self.codec.encode(text))
            if len(ids) < need:
                continue
            for start in range(0, len(ids) - need + 1, self.stride):
                x = torch.tensor(ids[start:start + self.seq_len], dtype=torch.long)
                y = torch.tensor(ids[start + 1:start + self.seq_len + 1],
                                 dtype=torch.long)
                self._windows.append((x, y))
                if self.max_windows is not None \
                        and len(self._windows) >= self.max_windows:
                    return

    def __len__(self) -> int:
        return (len(self._windows) // self.batch_size) * self.repeat

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._windows:
            return
        for _ in range(self.repeat):
            for i in range(0, len(self._windows) - self.batch_size + 1,
                           self.batch_size):
                chunk = self._windows[i:i + self.batch_size]
                x = torch.stack([w[0] for w in chunk])
                y = torch.stack([w[1] for w in chunk])
                if self.device is not None:
                    x, y = x.to(self.device), y.to(self.device)
                yield x, y

    def stats(self) -> dict:
        return {
            "texts": len(self.texts),
            "windows": len(self._windows),
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "repeat": self.repeat,
            "batches": len(self),
        }