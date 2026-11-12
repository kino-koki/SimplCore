# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Independent SimplCore research core: adaptive flux-ledger sequence model.

The temporal operator keeps two bounded summaries per layer: an adaptive
baseline and a baseline-relative surprise scale. It does not import the
existing SimplCore mixer/model implementation or an external model
implementation. This is an experimental SimplCore candidate, not a claim of
novelty or demonstrated quality/performance superiority.

For projected token input ``u_t``, baseline ``m_t`` and surprise scale ``v_t``:

    d_t     = u_t - m_(t-1)
    m_t     = m_(t-1) + sigmoid(a_t) * d_t
    v_t     = v_(t-1) + sigmoid(b_t) * (abs(d_t) - v_(t-1))
    r_t     = d_t / sqrt(v_t + eps)

The readout combines the updated baseline with normalized innovation. The
recurrence is causal and carries only two ``(batch, state_dim)`` tensors per
layer, regardless of the number of processed tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FluxLedgerBlock",
    "FluxLedgerMixer",
    "FluxLedgerState",
    "SimplCoreFluxLanguageModel",
]


@dataclass
class FluxLedgerState:
    """Fixed-size recurrent state for one FluxLedger mixer."""

    baseline: torch.Tensor
    surprise: torch.Tensor

    def detach(self) -> "FluxLedgerState":
        return FluxLedgerState(self.baseline.detach(), self.surprise.detach())


class FluxLedgerMixer(nn.Module):
    """Causal adaptive-baseline/surprise mixer with bounded recurrent state."""

    def __init__(self, dim: int, state_dim: Optional[int] = None,
                 eps: float = 1e-4) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        state_dim = int(state_dim if state_dim is not None else max(dim // 2, 1))
        if state_dim < 1:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.dim = int(dim)
        self.state_dim = state_dim
        self.eps = float(eps)

        self.input_projection = nn.Linear(dim, state_dim)
        self.baseline_rate = nn.Linear(dim, state_dim)
        self.surprise_rate = nn.Linear(dim, state_dim)
        self.baseline_read = nn.Linear(state_dim, dim, bias=False)
        self.innovation_read = nn.Linear(state_dim, dim, bias=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.baseline_rate.bias, -2.0)
        nn.init.constant_(self.surprise_rate.bias, -1.0)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.01)

    def init_state(self, batch_size: int, *,
                   device: Optional[torch.device] = None) -> FluxLedgerState:
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if device is None:
            device = self.input_projection.weight.device
        shape = (int(batch_size), self.state_dim)
        return FluxLedgerState(
            baseline=torch.zeros(shape, device=device, dtype=torch.float32),
            surprise=torch.ones(shape, device=device, dtype=torch.float32),
        )

    def step(self, x: torch.Tensor, state: FluxLedgerState
             ) -> Tuple[torch.Tensor, FluxLedgerState]:
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape (batch, {self.dim}), got {tuple(x.shape)}")
        expected = (x.shape[0], self.state_dim)
        if tuple(state.baseline.shape) != expected or tuple(state.surprise.shape) != expected:
            raise ValueError(f"state tensors must both have shape {expected}")

        projected = torch.tanh(self.input_projection(x)).float()
        delta = projected - state.baseline
        alpha = torch.sigmoid(self.baseline_rate(x).float())
        beta = torch.sigmoid(self.surprise_rate(x).float())
        baseline = state.baseline + alpha * delta
        surprise = state.surprise + beta * (delta.abs() - state.surprise)
        innovation = delta * torch.rsqrt(surprise + self.eps)

        compute_dtype = x.dtype
        baseline_read = self.baseline_read(baseline.to(compute_dtype))
        innovation_read = self.innovation_read(innovation.to(compute_dtype))
        context = torch.tanh(baseline_read + innovation_read)
        y = x + self.output_projection(context)
        return y, FluxLedgerState(baseline=baseline, surprise=surprise)

    def forward(self, x: torch.Tensor,
                state: Optional[FluxLedgerState] = None
                ) -> Tuple[torch.Tensor, FluxLedgerState]:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"x must have shape (batch, length, {self.dim}), got {tuple(x.shape)}")
        if x.shape[1] < 1:
            raise ValueError("sequence length must be at least one")
        if state is None:
            state = self.init_state(x.shape[0], device=x.device)
        if state.baseline.device != x.device or state.surprise.device != x.device:
            raise ValueError("input and recurrent state must be on the same device")
        outputs = []
        for t in range(x.shape[1]):
            y, state = self.step(x[:, t], state)
            outputs.append(y.unsqueeze(1))
        return torch.cat(outputs, dim=1), state


class _FluxRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class FluxLedgerBlock(nn.Module):
    """Pre-normalized ledger mixer and pointwise residual transform."""

    def __init__(self, dim: int, state_dim: Optional[int] = None,
                 expansion: int = 2) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError(f"expansion must be positive, got {expansion}")
        hidden = dim * expansion
        self.mixer_norm = _FluxRMSNorm(dim)
        self.mixer = FluxLedgerMixer(dim, state_dim)
        self.transform_norm = _FluxRMSNorm(dim)
        self.transform_in = nn.Linear(dim, hidden)
        self.transform_out = nn.Linear(hidden, dim)
        nn.init.normal_(self.transform_in.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.transform_in.bias)
        nn.init.normal_(self.transform_out.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.transform_out.bias)

    def forward(self, x: torch.Tensor,
                state: Optional[FluxLedgerState] = None
                ) -> Tuple[torch.Tensor, FluxLedgerState]:
        mixed, state = self.mixer(self.mixer_norm(x), state)
        x = mixed + self.transform_out(
            torch.tanh(self.transform_in(self.transform_norm(mixed))))
        return x, state


class SimplCoreFluxLanguageModel(nn.Module):
    """Self-contained SimplCore language-model prototype using FluxLedger blocks.

    ``targets`` are same-position labels for already-shifted input tokens. A
    caller can process long streams in chunks by passing the returned states
    into the next call; state size is independent of chunk/stream length.
    """

    def __init__(self, vocab_size: int, dim: int, num_layers: int, *,
                 state_dim: Optional[int] = None, expansion: int = 2,
                 tie_embeddings: bool = True) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be at least 2, got {vocab_size}")
        if dim < 1 or num_layers < 1:
            raise ValueError("dim and num_layers must be positive")
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.num_layers = int(num_layers)
        self.state_dim = int(state_dim if state_dim is not None else max(dim // 2, 1))
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            FluxLedgerBlock(dim, self.state_dim, expansion)
            for _ in range(num_layers)
        ])
        self.final_norm = _FluxRMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        if not tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def init_state(self, batch_size: int, *,
                   device: Optional[torch.device] = None
                   ) -> Tuple[FluxLedgerState, ...]:
        return tuple(block.mixer.init_state(batch_size, device=device)
                     for block in self.blocks)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                states: Optional[Sequence[FluxLedgerState]] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                           Tuple[FluxLedgerState, ...]]:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch, length), got {tuple(input_ids.shape)}")
        if input_ids.shape[1] < 1:
            raise ValueError("sequence length must be at least one")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if states is not None and len(states) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer states, got {len(states)}")

        x = self.token_embedding(input_ids)
        next_states = []
        for index, block in enumerate(self.blocks):
            layer_state = None if states is None else states[index]
            x, layer_state = block(x, layer_state)
            next_states.append(layer_state)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, tuple(next_states)
