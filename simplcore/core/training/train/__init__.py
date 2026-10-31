# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore training loop.

Designed for 1 x AMD Instinct MI250 (128 GB HBM2e, bf16, sequences of 4096)
while remaining fully runnable on CPU for small-scale convergence proof.

Features that matter for a *real* training run, not a demo:

* bf16 autocast on CUDA (no GradScaler needed) with fp32 parameter/optimizer state;
* TF32 matmuls on Ampere+ (CDNA2 also honours the same matmul precision knobs);
* AdamW with decoupled weight decay applied only to matrices (norms, biases, the
  oscillator ``feed`` term and ``out_scale`` are excluded) -- this matters because
  weight decay on the oscillator parameters destroys long-range memory;
* warmup + cosine decay, gradient clipping, gradient accumulation;
* multi-GPU data parallelism via ``torchrun`` + ``DistributedDataParallel``
  (see "Distributed training" below);
* optional ``torch.compile`` and activation checkpointing, both fenced so a failure
  degrades to eager instead of crashing the run;
* resumable checkpoints (model + optimizer + scheduler + RNG + step);
* JSONL metrics plus a human-readable log, so curves can be replotted later;
* deterministic seeding and a token-budget accounting that is honest about how many
  tokens were actually consumed.

Distributed training
--------------------
The 1B/3B presets do not fit single-GPU training budgets (AdamW state alone is
~8x params). The path for them is ``torchrun`` data parallelism::

    torchrun --nproc_per_node=8 -m simplcore.core.train \\
        --preset tuan-3b --global-batch-tokens 524288 ...

Launch mechanics follow the standard PyTorch contract: when ``WORLD_SIZE > 1``
(env set by ``torchrun``) the script initialises a process group, pins each
rank to ``cuda:LOCAL_RANK``, shards the data with ``DistributedSampler``,
synchronises gradients with DDP, reduces the loss/token accounting so logged
numbers are *global*, and only rank 0 writes logs and checkpoints. Checkpoints
store the model without the DDP wrapper, so a multi-GPU run can be resumed
single-GPU and vice versa. Without ``torchrun`` everything behaves exactly as
the single-process script it always was.
"""
from __future__ import annotations


from .checkpoint import load_checkpoint, save_checkpoint
from .cli import assert_train_window_fits, build_config, parse_args, train_window_bytes
from .data_windows import LMWindows, resolve_data_dir
from .distributed import setup_distributed
from .evaluate import evaluate
from .main import main
from .optim import apply_rocm_tuning, build_param_groups, cosine_with_warmup

__all__ = [
    "LMWindows",
    "apply_rocm_tuning",
    "assert_train_window_fits",
    "build_config",
    "build_param_groups",
    "cosine_with_warmup",
    "evaluate",
    "load_checkpoint",
    "main",
    "parse_args",
    "resolve_data_dir",
    "save_checkpoint",
    "setup_distributed",
    "train_window_bytes",
]
