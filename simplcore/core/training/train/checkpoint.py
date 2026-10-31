# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""可恢复 checkpoint：model + optim + scheduler + RNG + step。"""
from __future__ import annotations



import os
import random

import numpy as np
import torch

from ...model.config import SimplCoreConfig, get_preset



# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model, optim, step: int, tokens: int, cfg: SimplCoreConfig,
                    best_val: float, elapsed: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "tokens": tokens,
        "best_val": best_val,
        "elapsed": elapsed,
        "config": cfg.to_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, optim=None, restore_rng: bool = True) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if restore_rng:
        try:
            torch.set_rng_state(ckpt["torch_rng"])
            np.random.set_state(ckpt["numpy_rng"])
            random.setstate(ckpt["python_rng"])
        except Exception:
            pass
    return ckpt

