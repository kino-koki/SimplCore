# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""训练窗口数据集：memmap LMWindows 与数据目录解析。"""
from __future__ import annotations



import os
from typing import Iterator, Optional

from torch.utils.data import DataLoader, Dataset, DistributedSampler




# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class LMWindows(Dataset):
    """Thin adapter exposing ``(inputs, targets)`` windows for the training loop.

    The corpus schema, sharding, SHA-256 verification and cross-shard windowing all
    live in :mod:`simplcore.core.data` -- this class deliberately does not
    re-implement them. It asks for windows one token longer than the training
    sequence length so that shifting by one yields exactly ``seq_len`` targets.
    """

    def __init__(self, root: str, split: str, seq_len: int,
                 verify_hashes: bool = True) -> None:
        from ..training.data import ShardedTokenDataset
        self.seq_len = int(seq_len)
        self.inner = ShardedTokenDataset(root, split, seq_len=self.seq_len + 1,
                                        verify_hashes=verify_hashes)
        self.vocab_size = int(getattr(self.inner, "vocab_size", 0) or 0)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        w = self.inner[index]
        if isinstance(w, (tuple, list)):
            w = w[0]
        w = w.long()
        return w[:-1], w[1:]


def resolve_data_dir(explicit: Optional[str]) -> Optional[str]:
    from ..training.data import MANIFEST_NAME
    if explicit:
        return explicit
    for cand in ("data/processed/tuan_zh", "data/processed/tuan_zh_smoke"):
        if os.path.exists(os.path.join(cand, MANIFEST_NAME)):
            return cand
    return None

