# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""优化器与调度：参数组（振荡基免 decay）、warmup+cosine、ROCm 调优。"""
from __future__ import annotations



import math
import os

import torch




# ---------------------------------------------------------------------------
# optimiser parameter groups
# ---------------------------------------------------------------------------
def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """Decay matrices only.

    The oscillator parameters (``raw_rho``, ``raw_theta``), the instantaneous
    ``feed`` term, ``out_scale``, all norms and the embeddings are left undecayed:
    weight decay on a decay-rate parameter is a direct attack on long-range memory,
    which is the whole point of this architecture.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and not any(k in name for k in ("tok_emb", "head")):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_with_warmup(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if total <= warmup:
        return lr
    t = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


def apply_rocm_tuning(force: bool = False) -> bool:
    """MI250 / ROCm 训练调优。

    PyTorch 在 ROCm 上仍把设备暴露为 ``"cuda"``，``torch.cuda.*`` 通用；分布式
    后端 ``"nccl"`` 在 ROCm 上自动映射到 RCCL。返回是否处于 ROCm 环境。

    这些旋钮必须在**首次显存分配之前**设置（``PYTORCH_HIP_ALLOC_CONF`` 是惰性
    读取的），所以本函数在 ``main`` 里确定 device 之后、建模型之前调用。
    """
    is_hip = getattr(torch.version, "hip", None) is not None
    if not (is_hip or force):
        return False
    # 展开式显存段：128GB HBM 上大幅减少碎片，避免长训练中途 OOM。
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    # 固定输入形状（micro_batch × seq_len 全程不变）下，让 MIOpen 选最快卷积算法。
    torch.backends.cudnn.benchmark = True
    print("[train] ROCm/MI250 tuning applied: expandable_segments, cudnn.benchmark=True")
    return True

