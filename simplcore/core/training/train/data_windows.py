# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""训练窗口数据集：memmap LMWindows 与数据目录解析。"""
from __future__ import annotations



import os
from typing import Iterator, Optional

import torch
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
        from ..data import ShardedTokenDataset
        self.seq_len = int(seq_len)
        self.inner = ShardedTokenDataset(root, split, seq_len=self.seq_len + 1,
                                        verify_hashes=verify_hashes)
        self.vocab_size = int(getattr(self.inner, "vocab_size", 0) or 0)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        w = self.inner[index]
        if isinstance(w, dict):
            tokens = w.get("tokens", w.get("input_ids"))
            if tokens is None:
                raise ValueError(
                    "token-window mapping must contain 'tokens' or 'input_ids'")
            tokens = tokens.long()
            item = {"input_ids": tokens[:-1], "targets": tokens[1:]}
            for source in ("ncm_label_targets", "ncm_group_targets"):
                values = w.get(source)
                if values is not None:
                    values = values.long()
                    if values.shape != tokens.shape:
                        raise ValueError(
                            f"{source} must match token-window shape "
                            f"{tuple(tokens.shape)}, got {tuple(values.shape)}")
                    item[source] = values[:-1]
            return item
        if isinstance(w, (tuple, list)):
            w = w[0]
        w = w.long()
        return w[:-1], w[1:]


def unpack_lm_batch(batch):
    """Return token inputs/targets and optional NCM supervision from a loader batch."""
    if isinstance(batch, dict):
        x = batch.get("input_ids")
        y = batch.get("targets")
        labels = batch.get("ncm_label_targets")
        groups = batch.get("ncm_group_targets")
    elif isinstance(batch, (tuple, list)) and len(batch) in (2, 3, 4):
        x, y = batch[:2]
        labels = batch[2] if len(batch) >= 3 else None
        groups = batch[3] if len(batch) >= 4 else None
    else:
        raise ValueError(
            "LM batch must be a mapping or a 2-4 item tuple/list")
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError("LM batch input_ids and targets must be tensors")
    if labels is not None and not torch.is_tensor(labels):
        raise TypeError("NCM label targets must be a tensor when provided")
    if groups is not None and not torch.is_tensor(groups):
        raise TypeError("NCM group targets must be a tensor when provided")
    if labels is not None and labels.shape != x.shape:
        raise ValueError("NCM label targets must have the same shape as input_ids")
    if groups is not None and groups.shape != x.shape:
        raise ValueError("NCM group targets must have the same shape as input_ids")
    return x, y, labels, groups


def resolve_data_dir(explicit: Optional[str]) -> Optional[str]:
    from ..data import MANIFEST_NAME
    if explicit:
        return explicit
    for cand in ("data/processed/tuan_zh", "data/processed/tuan_zh_smoke"):
        if os.path.exists(os.path.join(cand, MANIFEST_NAME)):
            return cand
    return None
