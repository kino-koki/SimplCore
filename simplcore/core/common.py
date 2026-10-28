# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Shared evaluation utilities split out of the former monolithic ``eval.py``:
device/dtype resolution, JSON-safe conversion, provenance helpers, and the
synthetic batch/dataset plumbing shared by LM evaluation, probes and benches."""
from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass, fields as _dc_fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects to JSON-serialisable data."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if is_dataclass(obj):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in _dc_fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _jsonable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return repr(obj)


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_dirty() -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def resolve_device(name: str) -> torch.device:
    if name in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32, "fp32": torch.float32,
        "float64": torch.float64, "fp64": torch.float64,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
    }
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(table)}")
    return table[name]


def default_config_kwargs(device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """Keep the FFT path in fp32 unless the user explicitly asks otherwise.

    An fp32 FFT is the numerically safe default (see the package docstring);
    'mixed' (compute in the requested dtype, FFT in fp32) is what the model was
    designed for on CPU.
    """
    if dtype in (torch.bfloat16, torch.float16):
        return {"fft_dtype": "float32", "compute_dtype": str(dtype).replace("torch.", "")}
    d = str(dtype).replace("torch.", "")
    return {"fft_dtype": d if d in ("float32", "float64") else "float32"}


@dataclass
class EvalBatch:
    """One evaluation batch: ``inputs`` (B, T) int64 and ``targets`` (B, T) int64."""

    inputs: torch.Tensor
    targets: torch.Tensor


class SyntheticDataset:
    """Minimal ``__len__``/``__getitem__`` dataset returning ``(seq_len,)`` LongTensors.

    A stand-in for ``simplcore.core.data`` (written concurrently by another
    agent) so the harness is testable on its own.  ``__getitem__`` materialises
    the sequence on demand, i.e. the held-out set is freshly sampled every pass.
    """

    def __init__(self, vocab_size: int, seq_len: int, length: int = 512,
                 seed: int = 0, device: str = "cpu") -> None:
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.length = int(length)
        self.seed = int(seed)
        self.device = device
        self._epoch = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed * 1000003 + (i % self.length) + self._epoch * 7919)
        return torch.randint(0, self.vocab_size, (self.seq_len,), generator=g).to(self.device)

    def reshuffle(self) -> None:
        self._epoch += 1


def _as_eval_batch(item: Any) -> EvalBatch:
    if isinstance(item, EvalBatch):
        return item
    if isinstance(item, torch.Tensor):
        return EvalBatch(item, item)
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return EvalBatch(item[0], item[1])
    raise TypeError(f"dataset item must be a Tensor or (inputs, targets); got {type(item)!r}")


def _stack_batch(items: Sequence[Any], device: torch.device) -> EvalBatch:
    batches = [_as_eval_batch(it) for it in items]
    inputs = torch.stack([b.inputs for b in batches]).to(device)
    targets = torch.stack([b.targets for b in batches]).to(device)
    return EvalBatch(inputs.long(), targets.long())


def _next_token_targets(inputs: torch.Tensor) -> torch.Tensor:
    """Standard causal-LM shift with the first position masked (``-100``)."""
    tgt = torch.full_like(inputs, -100)
    tgt[:, :-1] = inputs[:, 1:]
    return tgt


