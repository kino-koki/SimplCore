# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""优化器与调度：参数组（振荡基免 decay）、warmup+cosine、ROCm 调优。"""
from __future__ import annotations



import math
import os

import torch
from typing import Optional




# ---------------------------------------------------------------------------
# optimiser parameter groups
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Adafactor（R-edge，0.1-α）：factored 二阶矩——2GB 级机器上 1 亿参数可训
# ---------------------------------------------------------------------------
class Adafactor(torch.optim.Optimizer):
    r"""Adafactor（Shazeer & Stern 2018, 算法 5 的工程简化版）.

    动机（2026-10-31 端侧实测，`docs/EDGE_TRAINING_REPORT.md`）：AdamW 需要
    16 B/参数（fp32 参数+梯度+动量+二阶矩），1.085 亿参数直接爆 2 GB；
    Adafactor 的二阶矩只存**行/列因子**（$O(m+n)$ 而非 $O(mn)$），
    108.5M × (4 权重 + 4 梯度) ≈ 0.87 GB——**1 亿参数塞进 2 GB 的路径**。

    状态账（per 2D 参数 m×n）：~4B×(m + n) + 标量，对比 AdamW 的 8B×m·n。
    eps1 防除零、eps2 为二阶矩下限、``clip_thresh`` 做 update clipping
    （RMS 归一）；``decay_rate=-0.8`` 即 Adafactor 论文的相对衰减调度
    （$v_t = 
ho_t v_{t-1} + (1-
ho_t)\hat v_t$，$
ho_t=1-t^{-0.8}$）。
    """

    def __init__(self, params, lr: float = 1e-3, eps1: float = 1e-30,
                 eps2: float = 1e-3, clip_thresh: float = 1.0,
                 decay_rate: float = -0.8, beta1: Optional[float] = None,
                 weight_decay: float = 0.0) -> None:
        defaults = dict(lr=lr, eps1=eps1, eps2=eps2, clip_thresh=clip_thresh,
                        decay_rate=decay_rate, beta1=beta1,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_state(self) -> bool:
        return True

    @staticmethod
    def _factored_second_moment(g: torch.Tensor) -> torch.Tensor:
        r"""行/列均方的外积近似 $\hat v$（只存因子的按步重构）。"""
        q_col = g.pow(2).mean(dim=1, keepdim=True)      # 行均方 (m,1)
        q_row = g.pow(2).mean(dim=0, keepdim=True)      # 列均方 (1,n)
        return (q_col @ q_row) / g.pow(2).mean().clamp_min(1e-30)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps1, eps2 = group["eps1"], group["eps2"]
            clip = group["clip_thresh"]
            decay = group["decay_rate"]
            beta1 = group["beta1"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if beta1 is not None:
                        state["exp_avg"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                # update clipping（梯度 RMS 归一）
                g_rms = g.pow(2).mean().sqrt().clamp_min(eps1)
                g = g / (g_rms / max(clip, 1e-12)).clamp(min=1.0)
                # 二阶矩：2D 因子化；1D/标量全量
                if g.dim() >= 2 and g.numel() > 1:
                    v_hat = self._factored_second_moment(g)
                    if "v_row" not in state:
                        state["v_row"] = torch.zeros(
                            g.shape[0], dtype=g.dtype, device=g.device)
                        state["v_col"] = torch.zeros(
                            g.shape[1], dtype=g.dtype, device=g.device)
                    rho = 1.0 - step ** decay          # 相对衰减调度
                    state["v_row"].mul_(rho).add_(
                        g.pow(2).mean(dim=1), alpha=1 - rho)
                    state["v_col"].mul_(rho).add_(
                        g.pow(2).mean(dim=0), alpha=1 - rho)
                    v_hat = (state["v_row"].unsqueeze(1)
                             @ state["v_col"].unsqueeze(0))                         / state["v_col"].mean().clamp_min(eps2)
                else:
                    if "v" not in state:
                        state["v"] = torch.zeros_like(p)
                    rho = 1.0 - step ** decay
                    state["v"].mul_(rho).add_(
                        g.pow(2), alpha=1 - rho)
                    v_hat = state["v"]
                denom = v_hat.sqrt().clamp_min(eps2).add_(eps1)
                u = g / denom
                # update clipping（对最终 update 再归一一次）
                u_rms = u.pow(2).mean().sqrt().clamp_min(eps1)
                u = u / (u_rms / max(clip, 1e-12)).clamp(min=1.0)
                if beta1 is not None:
                    m = state["exp_avg"]
                    m.mul_(beta1).add_(u, alpha=1 - beta1)
                    u = m
                if wd != 0.0:
                    u = u + wd * p
                p.add_(u, alpha=-lr)
        return loss


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

