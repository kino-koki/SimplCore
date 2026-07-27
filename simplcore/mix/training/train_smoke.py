"""CPU-safe smoke training loop for SimplCore Mix.

The smoke loop intentionally keeps the workload tiny while still exercising the
real model forward pass, loss, backward pass, optimizer step, and checkpoint
serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from simplcore.mix.config import SimplCoreMixConfig
from simplcore.mix.model import SimplCoreMix


def _smoke_config(seq_len: int) -> SimplCoreMixConfig:
    return SimplCoreMixConfig(
        vocab_size=64,
        dim=16,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=seq_len,
        max_loop_iters=1,
        prelude_layers=0,
        coda_layers=0,
        attn_type="rfp_gqa",
        n_bases=2,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=16,
        n_memory_channels=1,
        memory_dim=8,
        dmc_use_associative=False,
        loop_lora_enabled=False,
        spm_enabled=False,
        cognitive_enabled=False,
        anti_hallucination=False,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        lora_rank=2,
    )


def run_train_smoke(
    output_dir: str | Path,
    *,
    steps: int = 1,
    batch_size: int = 1,
    seq_len: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")

    torch.manual_seed(seed)
    device = torch.device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cfg = _smoke_config(seq_len)
    model = SimplCoreMix(cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    last_loss = None
    optimizer_steps = 0
    backward_ran = False

    for _ in range(steps):
        tokens = torch.randint(
            0,
            cfg.vocab_size,
            (batch_size, seq_len + 1),
            dtype=torch.long,
            device=device,
        )
        x = tokens[:, :-1]
        y = tokens[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, n_loops=1)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        backward_ran = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )
        optimizer.step()
        optimizer_steps += 1
        last_loss = loss.detach().cpu()

    checkpoint_path = output_path / "train_smoke.pt"
    loss_value = float(last_loss.item())
    torch.save(
        {
            "step": steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_value,
            "config": cfg.__dict__,
        },
        checkpoint_path,
    )

    return {
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "loss": loss_value,
        "optimizer_steps": optimizer_steps,
        "backward_ran": backward_ran,
        "checkpoint_path": str(checkpoint_path),
    }


run_smoke_training = run_train_smoke
