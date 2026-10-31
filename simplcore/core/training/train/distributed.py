# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""torchrun/DDP 启动契约：进程组初始化、rank 绑定。"""
from __future__ import annotations



import os

import torch
import torch.distributed as dist




# ---------------------------------------------------------------------------
# distributed setup (torchrun contract)
# ---------------------------------------------------------------------------
def setup_distributed() -> tuple:
    """按 ``torchrun`` 约定初始化进程组。

    由 ``torchrun`` 拉起时环境里有 ``WORLD_SIZE/RANK/LOCAL_RANK``：初始化
    进程组（CUDA 用 nccl，否则 gloo）并把当前进程钉到 ``cuda:LOCAL_RANK``，
    返回 ``(rank, local_rank, world_size)``。单进程直接运行时返回
    ``(0, 0, 1)``，不做任何分布式初始化——行为与历史版本完全一致。
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()

