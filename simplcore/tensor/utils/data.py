# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Tensor Data — 数据工具兼容层
=======================================
提供与 ``torch.utils.data`` 等价的接口。

包括 ``Dataset``、``IterableDataset``、``DataLoader`` 等。
基于 Python 原生实现，不依赖 PyTorch。
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..tensor import Tensor


# ============================================================
# Dataset 基类
# ============================================================
class Dataset:
    """数据集基类，子类需实现 ``__len__`` 和 ``__getitem__``。"""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError


class IterableDataset:
    """可迭代数据集基类，子类需实现 ``__iter__``。"""

    def __iter__(self) -> Iterator:
        raise NotImplementedError


class TensorDataset(Dataset):
    """张量数据集：将多个张量打包为 (input, target) 对。"""

    def __init__(self, *tensors) -> None:
        if not all(isinstance(t, (Tensor, np.ndarray)) for t in tensors):
            raise TypeError("所有输入必须是 Tensor 或 ndarray")
        self.tensors = tensors
        self._length = len(tensors[0]) if tensors else 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        return tuple(t[index] for t in self.tensors)


class ConcatDataset(Dataset):
    """拼接多个数据集。"""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = list(datasets)
        self._lengths = [len(d) for d in self.datasets]
        self._cumulative = []
        total = 0
        for l in self._lengths:
            total += l
            self._cumulative.append(total)

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int):
        for i, cum in enumerate(self._cumulative):
            if index < cum:
                offset = self._cumulative[i - 1] if i > 0 else 0
                return self.datasets[i][index - offset]
        raise IndexError(f"索引 {index} 超出范围")


class Subset(Dataset):
    """数据集子集。"""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


# ============================================================
# DataLoader
# ============================================================
class DataLoader:
    """数据加载器。

    Parameters
    ----------
    dataset : Dataset
        数据集。
    batch_size : int
        批量大小。
    shuffle : bool
        是否每轮打乱。
    drop_last : bool
        是否丢弃最后不完整的批量。
    num_workers : int
        工作进程数（当前忽略，始终单线程）。
    collate_fn : callable, optional
        自定义批处理函数。
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        sampler=None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.collate_fn = collate_fn or _default_collate
        self.pin_memory = pin_memory
        self._indices = list(range(len(dataset)))

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        if self.shuffle:
            indices = list(self._indices)
            random.shuffle(indices)
        else:
            indices = list(self._indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)


def _default_collate(batch: List) -> Any:
    """默认批处理函数：将列表转换为批量化张量。"""
    if isinstance(batch[0], (Tensor, np.ndarray)):
        # 单个张量
        arrays = [b.data if isinstance(b, Tensor) else b for b in batch]
        return Tensor(np.stack(arrays))
    elif isinstance(batch[0], (tuple, list)):
        # 多个张量 (input, target)
        num_items = len(batch[0])
        result = []
        for i in range(num_items):
            items = [b[i] for b in batch]
            if isinstance(items[0], (Tensor, np.ndarray)):
                arrays = [t.data if isinstance(t, Tensor) else t for t in items]
                result.append(Tensor(np.stack(arrays)))
            else:
                result.append(np.array(items))
        return tuple(result)
    elif isinstance(batch[0], (int, float)):
        return np.array(batch)
    else:
        return batch


class DistributedSampler:
    """分布式采样器兼容层（当前为空操作）。"""

    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, seed=0):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.dataset) // self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        # 按rank切片
        per_replica = len(indices) // self.num_replicas
        start = self.rank * per_replica
        return iter(indices[start:start + per_replica])

    def set_epoch(self, epoch):
        self.epoch = epoch


__all__ = [
    "Dataset",
    "IterableDataset",
    "TensorDataset",
    "ConcatDataset",
    "Subset",
    "DataLoader",
    "DistributedSampler",
]
