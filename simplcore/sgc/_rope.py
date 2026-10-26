# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Shared RoPE + attention helpers (dependency-light: no model imports).

Both :mod:`simplcore.sgc.model` (the attention baseline) and
:mod:`simplcore.sgc.hybrid` (SGC Hybrid sliding-window attention) need the same
RoPE tables / apply / scaled-softmax primitives.  Keeping them here — a module
that imports nothing from ``model`` / ``hybrid`` — avoids a circular import.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _rope_tables(head_dim: int, length: int, device, dtype, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(length, device=device).float()
    freqs = torch.outer(t, inv)                       # (L, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, hd)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.view(1, 1, x.shape[2], -1)
    s = sin.view(1, 1, x.shape[2], -1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


__all__ = ["_rope_tables", "_apply_rope"]
