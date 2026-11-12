# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""四层闸门中 L2 / L3 的**共享数学**（单一事实来源）。

背景
----
四层闸门（见 ``docs/SGC_CORE.md`` §6）里，L2 证据门与 L3 不确定门原本只写在
``SpectralGlobalConv``（单头谱基）上，而 ``GatedSpectralConv2``（γ/β 双门 +
多头谱基，``gate_ab=True`` 分支）**完全没有闸门**。这导致同一个架构里
"闸门能力"依赖分支选择 —— 是 ``docs/DREAMING_CMT.md`` §6 的 G4 缺口。

本模块把两层的数学抽成纯函数，两条分支都调用同一份实现，从此结构上不可能再漂移。

因果前缀语义
------------
L2 的统计量是 ``c_t = (1/t)·Σ_{s≤t} ||u_s||²``，只依赖过去。因此：

* **并行路径**用 ``cumsum``；
* **流式路径**用常驻标量运行和（``energy_sum``，字节数恒定，与 L 无关）。

两者逐位一致是 FFT ↔ IIR 对偶形式得以保持的前提，
由 ``tests/test_gates.py`` 与 ``tests/test_selective.py`` 守护。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "EVIDENCE_PARAM_COUNT",
    "UNCERTAINTY_PARAM_COUNT",
    "GATE_PARAM_COUNT",
    "prefix_mean_energy",
    "evidence_weights",
    "evidence_weights_chunk",
    "evidence_weight_step",
    "init_energy_sum",
    "joint_disagreement",
    "uncertainty_weights",
]

#: L2 证据门的标量参数个数（gain / pos_gain / bias）
EVIDENCE_PARAM_COUNT = 3
#: L3 不确定门的标量参数个数（单个 logit）
UNCERTAINTY_PARAM_COUNT = 1
#: 两层闸门合计标量参数
GATE_PARAM_COUNT = EVIDENCE_PARAM_COUNT + UNCERTAINTY_PARAM_COUNT


# ---------------------------------------------------------------------------
# L2 证据门
# ---------------------------------------------------------------------------
def prefix_mean_energy(u: torch.Tensor) -> torch.Tensor:
    """因果前缀平均能量 ``c_t = (1/t)·Σ_{s≤t} ||u_s||²``。

    Args:
        u: (B, L, D) 门控后的输入。

    Returns:
        (B, L) 前缀平均能量，**严格只依赖 t 及之前**。
    """
    energy = (u.float() ** 2).sum(dim=-1)                    # (B, L)
    csum = torch.cumsum(energy, dim=1)
    t = torch.arange(1, u.shape[1] + 1, device=u.device,
                     dtype=energy.dtype).unsqueeze(0)
    return csum / t


def evidence_weights(
    u: torch.Tensor,
    gain: torch.Tensor,
    pos_gain: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """L2 证据门的逐位置权重，形状 (B, L, 1)，取值 (0, 1)。

        logit_t = gain·log1p(c_t) + pos_gain·log1p(t) + bias
        e_t     = sigmoid(logit_t)

    证据越足（前缀能量越大、位置越靠后）闸门越开；上下文几乎无信号时
    长程路径被压制，模型回退到局部证据。

    参数是**标量**而非 ``Linear``：闸门的**响应形状是结构性的**，只有强度与
    偏置可学。用全连接会让闸门在初始化时退化成常数（且被上层 ``_init_weights``
    冲掉），这是踩过的坑。
    """
    c = prefix_mean_energy(u)                                # (B, L)
    t = torch.arange(1, u.shape[1] + 1, device=u.device,
                     dtype=c.dtype).unsqueeze(0)             # (1, L)
    logit = (gain * torch.log1p(c)
             + pos_gain * torch.log1p(t.expand_as(c))
             + bias)
    return torch.sigmoid(logit).unsqueeze(-1)                # (B, L, 1)


def init_energy_sum(batch: int, device, dtype=torch.float32) -> torch.Tensor:
    """流式路径的常驻前缀能量和，形状 (B, 1)。"""
    return torch.zeros(batch, 1, device=device, dtype=dtype)


def evidence_weights_chunk(
    u: torch.Tensor,
    energy_sum: Optional[torch.Tensor],
    length: int,
    gain: torch.Tensor,
    pos_gain: torch.Tensor,
    bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """分块形式的证据门：一次处理 ``T`` 个位置，跨块携带前缀状态。

    这是 :func:`evidence_weights`（整段 cumsum）与
    :func:`evidence_weight_step`（逐 token）之间的第三条等价形式：块内用
    ``cumsum``，块间用 ``energy_sum`` / ``length`` 续接。分块训练需要它，
    否则每个 chunk 都会把证据重置为"上下文从零开始"。

    Args:
        u: (B, T, D) 本块的门控输入。
        energy_sum: (B, 1) 本块之前的前缀能量和；``None`` 视作 0。
        length: 本块之前的 token 数。
        gain / pos_gain / bias: 同 :func:`evidence_weights`。

    Returns:
        ``(weight (B, T, 1), new_energy_sum (B, 1))``。
    """
    energy = (u.float() ** 2).sum(dim=-1)                    # (B, T)
    csum = torch.cumsum(energy, dim=1)
    if energy_sum is not None:
        csum = csum + energy_sum.to(csum.dtype).expand_as(csum)
    t = torch.arange(length + 1, length + u.shape[1] + 1,
                     device=u.device, dtype=csum.dtype).unsqueeze(0)
    c = csum / t
    logit = (gain * torch.log1p(c)
             + pos_gain * torch.log1p(t.expand_as(c))
             + bias)
    return torch.sigmoid(logit).unsqueeze(-1), csum[:, -1:]


def evidence_weight_step(
    u_t: torch.Tensor,
    energy_sum: Optional[torch.Tensor],
    length: int,
    gain: torch.Tensor,
    pos_gain: torch.Tensor,
    bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`evidence_weights` 的单步（流式）对偶形式。

    Args:
        u_t: (B, 1, D) 当前位置的门控输入。
        energy_sum: (B, 1) 此前的运行和；``None`` 视作 0。
        length: 此前的 token 数（当前位置下标为 ``length``）。
        gain / pos_gain / bias: 同 :func:`evidence_weights`。

    Returns:
        ``(weight (B, 1, 1), new_energy_sum (B, 1))``。
    """
    energy = (u_t.float() ** 2).sum(dim=-1)                  # (B, 1)
    new_sum = energy if energy_sum is None else energy_sum + energy
    t = torch.tensor(float(length + 1), device=u_t.device, dtype=torch.float32)
    c = new_sum.squeeze(-1) / t                              # (B,)
    logit = (gain * torch.log1p(c)
             + pos_gain * torch.log1p(t).expand_as(c)
             + bias)
    return torch.sigmoid(logit)[:, None, None], new_sum


# ---------------------------------------------------------------------------
# L3 不确定门
# ---------------------------------------------------------------------------
def joint_disagreement(long: torch.Tensor, local: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
    """长程路径与短程路径的相对分歧，形状 (B, L, 1)，取值 [0, 1]。

        d_t = ||long_t − local_t||₁ / (||long_t||₁ + ||local_t||₁ + eps)

    两条路径是同一上下文的两个独立估计，分歧是不确定性的代理。
    """
    num = (long - local).abs().sum(dim=-1, keepdim=True)
    den = (long.abs().sum(dim=-1, keepdim=True)
           + local.abs().sum(dim=-1, keepdim=True) + eps)
    return num / den


def uncertainty_weights(long: torch.Tensor, local: torch.Tensor,
                        logit: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L3 不确定门的逐位置阻尼系数，形状 (B, L, 1)，取值 (0, 1]。

        w_t = 1 − sigmoid(λ) · d_t

    分歧越大 → 长程被压制得越多 → 模型回退到局部证据。
    ``λ`` 是可学标量：模型自己决定"多不信自己的长程路径"。
    """
    d = joint_disagreement(long, local, eps)
    return 1.0 - torch.sigmoid(logit) * d
