# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""全局随机种子与确定性开关。

审计发现 ``simplcore_pg`` 全包原先**没有任何** ``manual_seed``/``random.seed`` 调用，
CLI 也没有 ``--seed``；实测连续两次构造 ``SimplCoreModel`` 得到不同权重，
因此此前所有训练结果都不可复现。本模块把播种收口到一处。
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch RNGs (CPU + all CUDA devices).

    Args:
        seed: the seed value.
        deterministic: also request deterministic kernels. This can be
            substantially slower and may raise for operators without a
            deterministic implementation, so it is off by default. It is the only
            way to get bit-identical results across runs on the same machine.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """``DataLoader(worker_init_fn=seed_worker)`` helper for reproducible loaders."""
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
