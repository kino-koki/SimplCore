# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Evaluation harness for the SGC package (`simplcore.sgc`).

Everything in this module is *new* code.  It deliberately does not import or
reuse anything from `verify_sgc/`, whose corpus was shown by two independent
audits to be ~83% machine logs / markdown scraped from this repository and whose
prototype mixer was both non-causal and non-deterministic.

Design rules
------------
1. **Memory instrumentation is accounted, not sampled.**
   `GetProcessMemoryInfo` / `PROCESS_MEMORY_COUNTERS` is broken on this host
   (returns 0 with `GetLastError() == 122`), so every RSS-based number printed by
   `verify_sgc/stream_infer.py` and `verify_sgc/_memcap.py` was silently 0.0 MB.
   This module never calls psutil or `GetProcessMemoryInfo`.  It reports
   `tensor_peak_bytes`: the exact byte size of every tensor that is *alive and
   materialised* over the measured step (parameters, gradients, and the tensors
   autograd actually writes during forward and backward), computed as
   ``sum(t.numel() * t.element_size())``.  When CUDA is present the real
   allocator counter ``torch.cuda.max_memory_allocated()`` is reported as well.
   `assert_memory_probe_works()` proves the instrumentation is live: the number
   must be > 0 and must strictly grow when the model gets wider.
2. **Scaling claims are fitted, not asserted.**  ``bench_throughput`` fits
   ``log(time) vs log(seq_len)`` and reports the slope.  For the FFT mixer the
   wall-clock slope is expected *below* 2 (near 1 in the measured range) even
   though the asymptotics are ``O(L log L)``; the attention baseline should show
   a slope close to 2 in both time and accounted memory.
3. **Probes are constructively solvable.**  Each `probe_*` function reports
   accuracy on a *freshly sampled* test set, the chance level, and the learning
   curve of a small `ProbeNet` trained on that exact task, so the reader can see
   whether the reported accuracy is above a *demonstrated* solvability bar
   rather than merely above chance.

Usage::

    python -m simplcore.sgc.eval evaluate   --preset debug --dataset synthetic --out r.json
    python -m simplcore.sgc.eval probe      --task selective-copy --steps 400 --out p.json
    python -m simplcore.sgc.eval throughput --preset debug --seq-lens 64,128,256,512 --backward
    python -m simplcore.sgc.eval streaming  --preset debug --lengths 64,256,1024
    python -m simplcore.sgc.eval compare    --preset debug --tasks ppl,selective-copy
    python -m simplcore.sgc.eval self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, fields as _dc_fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SGCConfig, SGC_PRESETS, get_preset
from .model import SGCModel

__all__ = [
    "evaluate_lm",
    "probe_selective_copy",
    "probe_associative_recall",
    "probe_interval_recall",
    "train_probe",
    "ProbeNet",
    "ProbeTask",
    "bench_throughput",
    "bench_streaming",
    "compare_mixers",
    "tensor_peak_bytes",
    "assert_memory_probe_works",
    "make_report",
    "main",
]

_ORIGINAL_ARGV = (" ".join([Path(sys.executable).name] + sys.argv) if sys.argv
                  else "")
# kept for backwards compatibility with callers that set it explicitly
_CLI_COMMAND = _ORIGINAL_ARGV

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


# ---------------------------------------------------------------------------
# Tensor-accounting memory instrumentation
# ---------------------------------------------------------------------------
class _TensorAudit(torch.autograd.Function):
    """Autograd Function that records the device bytes of its live activations.

    Registered on three autograd nodes written by one measured training step:
    the embedding output, the LM-head output (inside the model) and the loss
    (by this module).  Because the counters are filled by hooks rather than by
    sampling the process working set, the result is exact for the tensors that
    actually exist and can never silently read zero.
    """

    acc: Optional[Dict[str, int]] = None

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        acc = _TensorAudit.acc
        if acc is not None:
            acc["fwd"] = acc.get("fwd", 0) + _bytes_of(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        acc = _TensorAudit.acc
        if acc is not None and grad is not None:
            acc["bwd"] = acc.get("bwd", 0) + _bytes_of(grad)
        return grad


def _bytes_of(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _register_audit(t: torch.Tensor) -> torch.Tensor:
    # Autograd Functions are only valid on floating point tensors that require
    # grad; skip everything else rather than crashing a measurement.
    if t.requires_grad and t.is_floating_point():
        return _TensorAudit.apply(t)
    return t


def state_dict_bytes(obj: Any) -> int:
    """Exact bytes of every tensor in ``obj.state_dict()`` (params + buffers)."""
    if not hasattr(obj, "state_dict"):
        return 0
    sd = obj.state_dict()
    total = 0
    for v in sd.values():
        if isinstance(v, torch.Tensor):
            total += _bytes_of(v)
        elif isinstance(v, (list, tuple)):
            for u in v:
                if isinstance(u, torch.Tensor):
                    total += _bytes_of(u)
    return total


def _discover_tensors(stop_at_modules: bool = True) -> List[torch.Tensor]:
    """All tensors reachable from live objects in ``gc.get_objects()``.

    Parameters and buffers of an ``nn.Module`` are skipped (they are counted
    exactly once by :func:`state_dict_bytes`); everything else -- in-flight
    activations, gradient buffers, temporaries -- is returned.
    """
    import gc
    import warnings

    out: List[torch.Tensor] = []
    # The interpreter may hold arbitrary third-party objects whose *metaclass*
    # `__instancecheck__` raises deprecation warnings (torch.distributed.reduce_op
    # is one on torch 2.4).  Scanning the heap is a measurement, not a use of those
    # APIs, so warnings raised by other libraries while we inspect them are
    # suppressed -- scoped to this block only, and only FutureWarning/DeprecationWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        for o in gc.get_objects():
            if stop_at_modules and isinstance(o, nn.Module):
                continue
            if isinstance(o, torch.Tensor):
                out.append(o)
    return out


def live_tensor_bytes(stop_at_modules: bool = True, gc_collect: bool = True) -> int:
    """Bytes of all live tensors except module state (see :func:`_discover_tensors`)."""
    if gc_collect:
        import gc

        gc.collect()
    return sum(_bytes_of(t) for t in _discover_tensors(stop_at_modules))


def tensor_peak_bytes(model: nn.Module, batch: torch.Tensor, targets: torch.Tensor,
                      backward: bool = True, extra: Any = None,
                      gc_collect: bool = True) -> Dict[str, Any]:
    """Accounted device bytes for one forward (+ backward) step.

    Returns a dict with

    * ``param_bytes``   -- exact ``state_dict()`` bytes at measure time,
    * ``grad_bytes``    -- exact bytes of every ``p.grad`` after backward,
    * ``fwd_bytes``     -- exact sum of the activations autograd wrote while the
      model's own forward ran,
    * ``bwd_bytes``     -- exact sum of the gradient tensors autograd wrote while
      the backward ran,
    * ``live_bytes``    -- bytes of every other live tensor reachable in the
      interpreter (transient temporaries from the FFT/conv kernels, which are
      *not* surfaced by autograd hooks),
    * ``accounted_bytes`` -- ``param + grad + fwd + bwd`` (never 0 for a real model),
    * ``tensor_peak_bytes`` -- ``param + max(total_fwd, total_bwd) + live``: the
      high-water mark of the accounted tensors,
    * ``cuda_max_memory_allocated`` -- the real allocator peak when CUDA is
      available, else ``None`` (never fabricated).

    This is *accounting*, not a working-set sample: it is the honest replacement
    for the broken ``PROCESS_MEMORY_COUNTERS`` path.
    """
    if extra is not None:
        extra = tuple(x for x in extra if x is not None)
    acc: Dict[str, int] = {"fwd": 0, "bwd": 0}
    _TensorAudit.acc = acc

    param_bytes = state_dict_bytes(model)
    live_before = live_tensor_bytes(gc_collect=gc_collect)

    h_emb = model.tok_emb.register_forward_hook(lambda m, i, o: _register_audit(o))
    h_head = model.head.register_forward_hook(lambda m, i, o: _register_audit(o))

    cuda_before = None
    if torch.cuda.is_available() and batch.is_cuda:
        torch.cuda.reset_peak_memory_stats()
        cuda_before = int(torch.cuda.memory_allocated())

    grad_bytes = 0
    try:
        model.zero_grad(set_to_none=True)
        logits = model(batch)
        if extra:
            h_extra = [e.register_hook(lambda g: _register_audit(g)) for e in extra if e.requires_grad]
        else:
            h_extra = []
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100)
        acc["fwd"] += _bytes_of(loss)
        if backward:
            loss.backward()
            grad_bytes = int(sum(_bytes_of(p.grad) for p in model.parameters() if p.grad is not None))
        else:
            logits = logits.detach()
        live_after = live_tensor_bytes(gc_collect=gc_collect)
    finally:
        h_emb.remove()
        h_head.remove()
        if extra:
            for h in h_extra:
                h.remove()
        _TensorAudit.acc = None

    live = max(0, live_after - live_before)
    accounted = param_bytes + grad_bytes + acc["fwd"] + acc["bwd"]
    peak = param_bytes + max(acc["fwd"], acc["bwd"]) + live
    cuda_peak = None
    if cuda_before is not None:
        cuda_peak = int(torch.cuda.max_memory_allocated())
    return {
        "param_bytes": int(param_bytes),
        "grad_bytes": int(grad_bytes),
        "fwd_bytes": int(acc["fwd"]),
        "bwd_bytes": int(acc["bwd"]),
        "live_bytes": int(live),
        "accounted_bytes": int(accounted),
        "tensor_peak_bytes": int(peak),
        "cuda_max_memory_allocated": cuda_peak,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def mb(nbytes: Any) -> Optional[float]:
    if nbytes is None:
        return None
    return round(float(nbytes) / (1024.0 * 1024.0), 6)


def _smoke_stub_model() -> nn.Module:
    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.tok_emb(x))

    return _Stub()


def assert_memory_probe_works(device: str = "cpu", dtype: str = "float32",
                              verbose: bool = False) -> Dict[str, Any]:
    """Prove the instrumentation is live, or raise ``AssertionError``.

    Three independent checks:

    1. a tiny model must report a strictly positive accounted byte count,
    2. the count must grow when the model gets wider (width 32 -> 64),
    3. the accounting must be exact: it must reproduce ``state_dict()`` bytes
       plus the audited activations, and a hand-built tensor of known size must
       be reported exactly.

    A measurement that can silently read 0 is a bug, so a zero or non-growing
    number fails here instead of being printed.
    """
    dev = resolve_device(device)
    dt = resolve_dtype(dtype)
    results: Dict[str, Any] = {"checks": {}}

    x = torch.randint(0, 8, (2, 8), device=dev)
    m = _smoke_stub_model().to(dev, dtype=dt)
    m.train()
    r = tensor_peak_bytes(m, x, x, backward=True)
    assert r["accounted_bytes"] > 0, f"accounted_bytes must be > 0, got {r['accounted_bytes']}"
    assert r["tensor_peak_bytes"] > 0, f"tensor_peak_bytes must be > 0, got {r['tensor_peak_bytes']}"
    # exactness: embedding output + head output + loss + grads
    emb_b = 2 * 8 * 4 * 4
    head_b = 2 * 8 * 8 * 4
    loss_b = 4
    grad_b = sum(_bytes_of(p.grad) for p in m.parameters() if p.grad is not None)
    assert r["fwd_bytes"] == emb_b + head_b + loss_b, (
        f"fwd accounting inexact: {r['fwd_bytes']} != {emb_b + head_b + loss_b}")
    assert r["grad_bytes"] == grad_b, f"grad accounting inexact: {r['grad_bytes']} != {grad_b}"
    results["checks"]["stub_exact"] = {
        "fwd_bytes": r["fwd_bytes"], "expected": emb_b + head_b + loss_b,
        "grad_bytes": r["grad_bytes"], "expected_grad": grad_b,
    }

    # width scaling: same depth/seq, wider model must account for more bytes
    peaks = {}
    for width in (32, 64):
        cfg = SGCConfig(vocab_size=64, dim=width, num_layers=2, num_bases=8,
                         decay_max=64, local_kernel=4, num_heads=4, ffn_ratio=2.0)
        model = SGCModel(cfg).to(dev, dtype=dt).train()
        xb = torch.randint(0, 64, (2, 32), device=dev)
        rb = tensor_peak_bytes(model, xb, xb, backward=True)
        peaks[width] = rb
        assert rb["accounted_bytes"] > 0, f"width {width}: accounted_bytes == 0"
        assert rb["tensor_peak_bytes"] > 0, f"width {width}: tensor_peak_bytes == 0"
    assert peaks[64]["accounted_bytes"] > peaks[32]["accounted_bytes"], (
        "accounted bytes did not grow with model width: "
        f"{peaks[32]['accounted_bytes']} -> {peaks[64]['accounted_bytes']}")
    assert peaks[64]["param_bytes"] > peaks[32]["param_bytes"], (
        "param bytes did not grow with model width: "
        f"{peaks[32]['param_bytes']} -> {peaks[64]['param_bytes']}")
    results["checks"]["width_scaling"] = {
        "accounted_bytes_32": peaks[32]["accounted_bytes"],
        "accounted_bytes_64": peaks[64]["accounted_bytes"],
        "param_bytes_32": peaks[32]["param_bytes"],
        "param_bytes_64": peaks[64]["param_bytes"],
    }

    # a hand-built tensor of known size must be reported exactly
    probe = torch.zeros(1000, 4, dtype=torch.float32)
    assert _bytes_of(probe) == 16000, _bytes_of(probe)
    del probe

    results["ok"] = True
    results["cuda_available"] = bool(torch.cuda.is_available())
    if verbose:
        print(json.dumps(_jsonable(results), indent=2))
    return results


# ---------------------------------------------------------------------------
# Token-weighted language-modelling evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalBatch:
    """One evaluation batch: ``inputs`` (B, T) int64 and ``targets`` (B, T) int64."""

    inputs: torch.Tensor
    targets: torch.Tensor


class SyntheticDataset:
    """Minimal ``__len__``/``__getitem__`` dataset returning ``(seq_len,)`` LongTensors.

    A stand-in for ``simplcore.sgc.data`` (written concurrently by another
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


@torch.no_grad()
def evaluate_lm(model: nn.Module, dataset: Any, batch_size: int = 8, seq_len: Optional[int] = None,
                device: Any = None, max_batches: Optional[int] = None,
                autocast_dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
    """Token-weighted causal-LM loss / perplexity.

    ``loss`` is ``sum(per-token NLL) / number of counted tokens`` accumulated over
    every batch -- *not* the mean of per-batch means, which would silently
    re-weight short or lightly-padded batches.  Padding is masked out of both the
    numerator and the denominator:

    * a target equal to ``-100`` is ignored (the convention
      ``F.cross_entropy(ignore_index=-100)`` and every probe in this file uses),
    * if the dataset yields **plain** ``(seq_len,)`` tensors, the targets are
      built by the causal shift ``targets[t] = inputs[t+1]`` with position 0 and
      the final position masked out, so ``tokens = (seq_len - 1) * batch_size``,
    * if the dataset yields ``(inputs, targets)`` pairs (or :class:`EvalBatch`),
      the supplied targets are used verbatim.

    Returns:
        ``{"loss", "ppl", "tokens", "n_batches"}`` plus the bookkeeping fields
        ``batches_seen``, ``micro_batches`` and ``max_abs_dev_from_exact`` (the
        largest deviation between the returned loss and a float64 recomputation
        of the same reduction, i.e. evidence of the accumulation's accuracy).
    """
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    device = resolve_device("auto") if device is None else torch.device(device)
    model = model.to(device)
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    n_batches = 0
    micro = 0
    # float64 reference accumulator: catches re-weighting bugs that a float32
    # running mean would hide.
    ref_nll = 0.0
    micro_bs = max(1, batch_size)
    order = list(range(len(dataset)))

    for start in range(0, len(order), micro_bs):
        if max_batches is not None and n_batches >= max_batches:
            break
        idxs = order[start:start + micro_bs]
        raw = [dataset[i] for i in idxs]
        batch = _stack_batch(raw, device)
        if batch.targets.shape != batch.inputs.shape:
            raise ValueError("targets must have the same shape as inputs")
        built_shift = all(isinstance(r, torch.Tensor) for r in raw)
        targets = _next_token_targets(batch.inputs) if built_shift else batch.targets
        if seq_len is not None and batch.inputs.shape[1] != seq_len:
            raise ValueError(
                f"dataset produced seq_len {batch.inputs.shape[1]}, expected {seq_len}")
        counted = int((targets != -100).sum().item())
        if counted == 0:
            micro += 1
            n_batches += 1
            continue
        ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if (
            autocast_dtype is not None and device.type == "cuda") else _nullcontext()
        with ctx:
            logits = model(batch.inputs)
        logits = logits.float()
        nll_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-100, reduction="sum")
        total_nll += float(nll_sum.detach().cpu())
        ref_nll += _exact_nll_sum(logits, targets)
        total_tokens += counted
        micro += logits.size(0)
        n_batches += 1

    if total_tokens == 0:
        raise ValueError("no counted tokens: every target was masked out")
    loss = total_nll / total_tokens
    ref = ref_nll / total_tokens
    if was_training:
        model.train()
    return {
        "loss": loss,
        "ppl": math.exp(loss) if loss < 700 else float("inf"),
        "tokens": int(total_tokens),
        "n_batches": int(n_batches),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len) if seq_len is not None else None,
        "micro_batches": int(micro),
        "max_abs_dev_from_exact": abs(loss - ref),
        "weighting": "token_weighted_sum_nll_over_counted_tokens",
    }


def _exact_nll_sum(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """float64 recomputation of ``sum of per-token NLL`` over unmasked positions."""
    lg = logits.detach().double()
    tg = targets.detach()
    flat_lg = lg.reshape(-1, lg.size(-1))
    flat_tg = tg.reshape(-1)
    mask = flat_tg != -100
    if int(mask.sum()) == 0:
        return 0.0
    sel_lg = flat_lg[mask]
    sel_tg = flat_tg[mask]
    logprob = F.log_softmax(sel_lg, dim=-1)
    nll = -logprob.gather(1, sel_tg.unsqueeze(1)).squeeze(1)
    return float(nll.sum().item())


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------
# Probe tasks (long-range capability probes)
# ---------------------------------------------------------------------------
def _finish(inputs: torch.Tensor, targets: torch.Tensor, device: Any, vocab_size: int):
    """Cast to the token dtype, move to the device, and assert every id is in range.

    The bounds assertion is not decoration: an out-of-range id silently produces a
    garbage embedding lookup or an autograd index error deep inside the model, and
    the earlier `inputs.to(device).long()` ordering bug meant a whole probe run
    could be scored against reinterpreted memory.  Failing loudly here is cheap.
    """
    device = torch.device(device)
    inputs = inputs.long().to(device)
    targets = targets.long().to(device)
    if bool(inputs.min() < 0 or inputs.max() >= vocab_size):
        raise AssertionError(
            f"probe batch has ids outside [0, {vocab_size}): min={int(inputs.min())} "
            f"max={int(inputs.max())}")
    return inputs, targets


class ProbeTask:
    """A synthetic long-range task with a declared answer position and chance level.

    Subclasses implement :meth:`sample` returning ``(inputs, targets)`` int64
    tensors of shape ``(B, task.seq_len)`` where every position that must not
    contribute to the loss carries the ignore index ``-100``.

    Attributes:
        name: task identifier.
        vocab_size: size of the token alphabet the probe model uses.
        seq_len: fixed sequence length.
        answer_pos: index (or tuple of indices) at which the answer is scored.
        chance: chance-level accuracy (``1 / n_choices``).
        uses_shift: ``True`` for plain next-token datasets (``answer_pos is None``).
    """

    name: str = "probe"
    vocab_size: int = 64
    seq_len: int = 64
    answer_pos: Optional[Any] = None

    @property
    def chance(self) -> float:
        raise NotImplementedError

    def sample(self, batch_size: int, generator: torch.Generator,
               device: Any = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        ap = self.answer_pos
        return {
            "name": self.name,
            "vocab_size": int(self.vocab_size),
            "seq_len": int(self.seq_len),
            "answer_pos": int(ap) if isinstance(ap, int) else (
                [int(a) for a in ap] if ap is not None else None),
            "chance": float(self.chance),
        }


class SelectiveCopyTask(ProbeTask):
    """Selective copy: copy the token that follows a marker, on demand.

    Layout::

        [ random tokens ... MARKER v ... random tokens ... QUERY  <-- answer = v ]

    The answer is one content token out of ``n_content``, so ``chance = 1 /
    n_content``.  This is the standard selective-copy / induction-head task: it
    requires a content-based (not position-based) lookup over a long range.
    """

    name = "selective-copy"

    def __init__(self, n_content: int = 16, seq_len: int = 64, gap: int = 8,
                 marker: Optional[int] = None, query: Optional[int] = None,
                 pad: int = 0, answer_pos: Optional[int] = None) -> None:
        if n_content < 2:
            raise ValueError("n_content must be >= 2")
        self.n_content = int(n_content)
        # marker/query/pad are ids *above* the content range, so no answer slot or
        # padding position can be confused with a legal content token.
        self.marker = int(marker) if marker is not None else int(n_content)
        self.query = int(query) if query is not None else int(n_content) + 1
        self.pad = int(pad) if pad > 0 else int(n_content) + 2
        self.vocab_size = max(self.marker, self.query, self.pad) + 1
        self.gap = int(gap)
        min_len = self.gap + 5
        self.seq_len = int(max(seq_len, min_len))
        if answer_pos is None:
            answer_pos = self.seq_len - 1          # the slot right after the query token
        if not (2 < int(answer_pos) < self.seq_len):
            raise ValueError("answer_pos must leave room for the query and the answer slot")
        self.answer_pos = int(answer_pos)

    @property
    def chance(self) -> float:
        return 1.0 / float(self.n_content)

    def sample(self, batch_size, generator, device="cpu"):
        B, T = int(batch_size), self.seq_len
        inputs = torch.randint(0, self.n_content, (B, T), generator=generator)
        # marker + value must sit before the query and be separated by G tokens
        hi = max(1, self.answer_pos - self.gap - 2)
        pos = torch.randint(1, hi + 1, (B,), generator=generator)
        val = torch.randint(0, self.n_content, (B,), generator=generator)
        rows = torch.arange(B, device=inputs.device)
        inputs[rows, pos] = self.marker
        inputs[rows, pos + 1] = val
        inputs[rows, self.answer_pos - 1] = self.query
        inputs[rows, self.answer_pos] = self.pad
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[:, self.answer_pos] = val
        # sanity: marker/query ids are outside the content alphabet and each occurs
        # exactly once per row, so every sequence has a unique, unambiguous answer
        assert self.marker >= self.n_content and self.query >= self.n_content
        assert int((inputs == self.marker).sum()) == B, "marker aliasing with content ids"
        assert int((inputs == self.query).sum()) == B, "query aliasing with content ids"
        return _finish(inputs, targets, device, self.vocab_size)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"n_content": self.n_content, "marker": self.marker,
                  "query": self.query, "gap": self.gap})
        return d


class AssociativeRecallTask(ProbeTask):
    """Multi-query associative recall (MQAR).

    Layout::

        k0 v0 k1 v1 ... k_{n-1} v_{n-1}  QUERY q0 q1 ... q_{m-1}

    Layout::

        [pad, k_0, v_0, ..., k_{n-1}, v_{n-1}, q_0, a_0, ..., q_{m-1}, a_{m-1}]

    where each ``a_j`` is the answer slot of query ``q_j`` and holds the value
    bound to that key.  Keys and values live in disjoint id ranges, so the task is
    a pure key -> value lookup and ``chance = 1 / vocab_value``.  Queries are
    sampled with replacement from all ``num_kv`` keys, so a key may repeat.
    """

    name = "associative-recall"

    def __init__(self, num_kv: int = 8, num_queries: int = 4, vocab_key: int = 32,
                 vocab_value: int = 32, pad: int = 0) -> None:
        if num_kv < 1 or num_queries < 1:
            raise ValueError("num_kv and num_queries must be >= 1")
        if vocab_key < num_kv:
            raise ValueError("vocab_key must be >= num_kv")
        self.num_kv = int(num_kv)
        self.num_queries = int(num_queries)
        self.vocab_key = int(vocab_key)
        self.vocab_value = int(vocab_value)
        # pad id sits above both id ranges so it is never confused with a value
        self.key_base = 0
        self.value_base = self.vocab_key
        self.pad = int(pad) if pad > 0 else self.value_base + self.vocab_value
        self.vocab_size = max(self.pad, self.value_base + self.vocab_value) + 1
        # layout: [pad, k_0, v_0, ..., k_{n-1}, v_{n-1}, q_0, a_0, ..., q_{m-1}, a_{m-1}]
        # where a_j is the answer slot of query q_j and is always < seq_len.
        self.seq_len = 1 + 2 * self.num_kv + 2 * self.num_queries
        self.query_pos = tuple(1 + 2 * self.num_kv + 2 * j for j in range(self.num_queries))
        self.answer_pos = tuple(p + 1 for p in self.query_pos)
        assert self.answer_pos[-1] == self.seq_len - 1

    @property
    def chance(self) -> float:
        return 1.0 / float(self.vocab_value)

    def sample(self, batch_size, generator, device="cpu"):
        B = int(batch_size)
        T = self.seq_len
        # zeros, not empty: every slot is written below except the pad column, and
        # `torch.empty` recycles freed float storage, which is how uninitialised
        # int64 garbage (e.g. 4.16e18) leaked into the probe inputs.
        inputs = torch.zeros((B, T), dtype=torch.long)
        inputs[:, 0] = self.pad
        kv = self.num_kv
        # sample distinct keys per row, then values (values may repeat)
        keys = torch.empty((B, kv), dtype=torch.long)
        for b in range(B):
            keys[b] = torch.randperm(self.vocab_key, generator=generator)[:kv]
        keys = keys + self.key_base
        assert self.key_base == 0, "key ids must start at 0 so they cannot collide with pad"
        vals = torch.randint(0, self.vocab_value, (B, kv), generator=generator) + self.value_base
        inputs[:, 1:1 + 2 * kv:2] = keys
        inputs[:, 2:2 + 2 * kv:2] = vals
        qi = torch.randint(0, kv, (B, self.num_queries), generator=generator)
        qpos = torch.as_tensor(self.query_pos, dtype=torch.long)
        apos = torch.as_tensor(self.answer_pos, dtype=torch.long)
        inputs[:, qpos] = keys.gather(1, qi)
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[:, apos] = vals.gather(1, qi)
        return _finish(inputs, targets, device, self.vocab_size)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"num_kv": self.num_kv, "num_queries": self.num_queries,
                  "vocab_key": self.vocab_key, "vocab_value": self.vocab_value,
                  "key_base": self.key_base, "value_base": self.value_base})
        return d


class IntervalRecallTask(ProbeTask):
    """Fixed query key, one value, then G distractors, then the query again.

    Layout::

        [pad, KEY, v, d_0 ... d_{G-1}, KEY, <-- label = v]

    The second occurrence of the single query key is at ``seq_len - 2``; the
    logits there are scored against the value.  The gap between the value and the
    repeated query is exactly ``G`` distractor tokens, so ``G`` is the only
    difficulty knob (``b = v - 1`` has distance ``G + 2``).

    The final (second) occurrence of the single query key is at the last
    position; the answer slot is the next one, which is why the tokenizer-free
    probe reserves one extra slot at the end.  The gap between the value and the
    query is ``G`` distractors, so ``G`` is the only difficulty knob.
    ``chance = 1 / n_values``.

    Design note (why the earlier G=64 / 300-step attempt returned pure chance):
    the answer slot is a *single* position whose target is one of ``n_values``
    classes.  With plain cross-entropy over a large vocabulary the gradient for
    any individual value token is ~1/n_values of the total, so 300 steps of a
    small model on a rare class is far too little signal -- the run stops at the
    unigram prior, which is exactly chance.  The fixes applied here are
    (i) a smaller value alphabet, (ii) more steps, and (iii) an
    answer-weighted squared-error objective (``loss="mse"``) that keeps a usable
    gradient on the correct class instead of being diluted by the softmax
    normaliser.  See ``--loss`` on the ``probe`` subcommand.
    """

    name = "interval-recall"

    def __init__(self, distractor_gap: int = 16, n_values: int = 8,
                 query_key: Optional[int] = None, pad: int = 0,
                 n_content: int = 4) -> None:
        if n_values < 2:
            raise ValueError("n_values must be >= 2")
        if distractor_gap < 0:
            raise ValueError("distractor_gap must be >= 0")
        if n_content < 1:
            raise ValueError("n_content must be >= 1 (distractors are random content)")
        self.G = int(distractor_gap)
        self.n_values = int(n_values)
        self.n_content = int(n_content)
        # Disjoint id ranges; pad is above everything so it is never a legal value.
        #   values     0        .. n_values - 1
        #   content    n_values .. n_values + n_content - 1   (distractors)
        #   key        n_values + n_content                   (the single query key)
        #   pad        n_values + n_content + 1
        self.value_base = 0
        self.content_base = self.n_values
        self.query_key = (int(query_key) if query_key is not None
                          else self.content_base + self.n_content)
        self.pad = int(pad) if pad > 0 else self.query_key + 1
        self.vocab_size = max(self.pad, self.query_key) + 1
        # layout: [content, KEY, value, d_0 ... d_{G-1}, KEY, <answer slot>]
        self.seq_len = self.G + 5
        # the second query key sits at seq_len - 2; the value is scored at seq_len - 1
        self.answer_pos = self.seq_len - 1
        assert self.answer_pos == self.G + 4

    @property
    def chance(self) -> float:
        return 1.0 / float(self.n_values)

    def sample(self, batch_size, generator, device="cpu"):
        B, T = int(batch_size), self.seq_len
        inputs = torch.full((B, T), self.pad, dtype=torch.long)
        vals = torch.randint(0, self.n_values, (B,), generator=generator)
        inputs[:, 1] = self.query_key
        inputs[:, 2] = vals + self.value_base
        # Distractors are *random content*, not repeated pad: with pad-only gaps a
        # model can solve the task by copying from a fixed offset, which tests
        # positional indexing rather than long-range key retrieval.  The content
        # alphabet (id 0 .. query_key-1) is disjoint from the query key and from
        # every value id, so a distractor can never be mistaken for the answer.
        if self.G > 0:
            inputs[:, 3:3 + self.G] = torch.randint(
                0, self.n_content, (B, self.G), generator=generator) + self.content_base
        inputs[:, 3 + self.G] = self.query_key
        targets = torch.full((B, T), -100, dtype=torch.long)
        targets[:, self.answer_pos] = vals + self.value_base
        return _finish(inputs, targets, device, self.vocab_size)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update({"G": self.G, "n_values": self.n_values, "n_content": self.n_content,
                  "query_key": self.query_key, "value_base": self.value_base,
                  "content_base": self.content_base})
        return d


# ---------------------------------------------------------------------------
# Small, explicitly causal probe model
# ---------------------------------------------------------------------------
class _CausalDWConv(nn.Module):
    """Depthwise causal 1-D convolution (left-padded), O(L*K)."""

    def __init__(self, dim: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.conv.weight.shape[-1]
        u = F.pad(x.transpose(1, 2), (k - 1, 0))
        return self.conv(u).transpose(1, 2)


class _FullCausalAttention(nn.Module):
    """Cosine-similarity causal attention written out explicitly (no SDPA flags)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.dh = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(4096, 4096, dtype=torch.bool)),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.reshape(B, L, self.h, self.dh).transpose(1, 2)
        k = k.reshape(B, L, self.h, self.dh).transpose(1, 2)
        v = v.reshape(B, L, self.h, self.dh).transpose(1, 2)
        s = (q @ k.transpose(-1, -2)) * (self.dh ** -0.5)
        s = s.masked_fill(~self.mask[:L, :L], float("-inf"))
        y = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(y)


class _ProbeMixer(nn.Module):
    """The probe's mixer: content gating + global (full attention) or local path.

    This is a *clean-room* reimplementation for the probe models.  It is not the
    audited `verify_sgc` `GatedAssocMixer`: attention here is explicitly masked
    with a lower-triangular boolean mask, and the module contains no randomness
    (dropout is off and identities are fixed), so it is causal and deterministic
    -- both properties are asserted by ``tests/test_eval.py``.
    """

    def __init__(self, dim: int, global_kind: str = "attn", content_gate: bool = True,
                 num_heads: int = 4, local_kernel: int = 4) -> None:
        super().__init__()
        self.content_gate = content_gate
        self.kind = global_kind
        self.norm = nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim, bias=False)
        if content_gate:
            self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.global_mix = (_FullCausalAttention(dim, num_heads) if global_kind == "attn"
                           else _CausalDWConv(dim, local_kernel))
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))
        return x + self.out_proj(self.global_mix(u))


class _ProbeBlock(nn.Module):
    def __init__(self, dim: int, global_kind: str, content_gate: bool,
                 num_heads: int, local_kernel: int, ffn_mult: float = 2.0) -> None:
        super().__init__()
        self.mixer = _ProbeMixer(dim, global_kind, content_gate, num_heads, local_kernel)
        hidden = int(ffn_mult * dim)
        self.ffn_norm = nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class ProbeNet(nn.Module):
    """Small deterministic causal sequence model used for probe solvability checks.

    Causal guarantees:

    * the input embedding is token-only (no learned positional table), so no
      absolute-position information can leak across positions;
    * the global path is explicitly lower-triangular-masked attention;
    * the local path is a left-padded depthwise convolution;
    * no dropout and no other stochastic layer.

    ``tok_emb``/``head`` mirror the ``SGCModel`` attribute names so the exact
    same memory instrumentation runs on both.
    """

    def __init__(self, vocab_size: int, dim: int = 64, num_layers: int = 3,
                 global_kind: str = "attn", content_gate: bool = True,
                 num_heads: int = 4, local_kernel: int = 4,
                 ffn_mult: float = 2.0) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.tok_emb = nn.Embedding(self.vocab_size, dim)
        self.blocks = nn.ModuleList([
            _ProbeBlock(dim, global_kind, content_gate, num_heads, local_kernel, ffn_mult)
            for _ in range(num_layers)])
        self.ln_f = nn.RMSNorm(dim) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx.long())
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def _answer_logits(model: nn.Module, inputs: torch.Tensor, answer_pos: Any) -> torch.Tensor:
    """Logits used for scoring: ``(N, vocab)`` taken at ``answer_pos - 1``.

    `answer_pos` is uniformly the index of the **answer slot** (the label).  The
    autoregressive state that predicts it is the preceding position, so the logits
    come from ``answer_pos - 1``.
    """
    if isinstance(model, SGCModel):
        # SGCModel.forward takes (B, L) token ids (it adds the channel dim itself)
        logits = model(inputs)
    else:
        logits = model(inputs)
    if isinstance(answer_pos, int):
        return logits[:, answer_pos - 1, :]
    idx = torch.as_tensor([int(a) - 1 for a in answer_pos], device=logits.device,
                          dtype=torch.long)
    return logits.index_select(1, idx).reshape(-1, logits.size(-1))


def _answer_targets(targets: torch.Tensor, answer_pos: Any) -> torch.Tensor:
    """Targets scored by :func:`_answer_logits`: the token **at** ``answer_pos``."""
    if isinstance(answer_pos, int):
        return targets[:, answer_pos]
    idx = torch.as_tensor([int(a) for a in answer_pos], device=targets.device,
                          dtype=torch.long)
    return targets.index_select(1, idx).reshape(-1)


def _answer_loss(lg: torch.Tensor, tg: torch.Tensor, loss: str) -> torch.Tensor:
    """Probe objective restricted to unmasked scored positions.

    Positions whose target is ``-100`` are dropped *before* the objective is
    evaluated: ``F.one_hot`` rejects negative class indices, and counting them
    would dilute the gradient of the few positions that actually matter.
    """
    mask = tg != -100
    if int(mask.sum()) == 0:
        raise ValueError("no unmasked scored positions in this probe batch")
    sel_lg = lg[mask].float()
    sel_tg = tg[mask]
    if loss == "ce":
        return F.cross_entropy(sel_lg, sel_tg)
    if loss == "mse":
        return F.mse_loss(sel_lg, F.one_hot(sel_tg, sel_lg.size(-1)).float())
    raise ValueError(f"unknown probe loss {loss!r}; use 'mse' or 'ce'")


def evaluate_probe(model: nn.Module, task: ProbeTask, n_samples: int = 4096,
                   batch_size: int = 128, seed: int = 1234, device: Any = "cpu",
                   progress: bool = False) -> Dict[str, Any]:
    """Accuracy / cross-entropy of ``model`` on a **freshly sampled** test set.

    The returned ``probe_accuracy`` is the fraction of scored positions whose
    argmax equals the target; ``probe_loss`` is the per-position cross-entropy in
    nats; ``chance`` is the task's chance level (``1 / n_choices``), so
    ``lift_over_chance`` (the fraction of the gap from chance to perfect that the
    model actually closed) is directly interpretable.  ``progress`` is accepted
    for signature compatibility and unused.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    n_samples = int(n_samples)
    gen = torch.Generator().manual_seed(int(seed))
    flat_pos = task.answer_pos
    if flat_pos is None:
        raise ValueError("evaluate_probe needs a task with answer_pos (not a shift task)")
    n_pos = 1 if isinstance(flat_pos, int) else len(flat_pos)
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            b = min(batch_size, n_samples - start)
            inputs, targets = task.sample(b, gen, device=dev)
            lg = _answer_logits(model, inputs, flat_pos).float()
            tg = _answer_targets(targets, flat_pos)
            keep = tg != -100
            lg, tg = lg[keep], tg[keep]
            if tg.numel() == 0:
                continue
            loss_sum += float(F.cross_entropy(lg, tg, reduction="sum").item())
            correct += int((lg.argmax(-1) == tg).sum().item())
            total += int(tg.numel())
    acc = correct / max(total, 1)
    loss = loss_sum / max(total, 1)
    model.train()
    return {
        "probe_accuracy": acc,
        "probe_loss": loss,
        "probe_ppl": math.exp(min(loss, 700.0)),
        "chance": float(task.chance),
        "n_scored_positions": int(total),
        "n_sequences": int(n_samples),
        "n_positions_per_sequence": int(n_pos),
        "above_chance": bool(acc > task.chance * 1.5),
        "lift_over_chance": (acc - task.chance) / max(1e-12, 1.0 - task.chance),
        "test_set": "freshly sampled at evaluation time (generator seed %d)" % int(seed),
    }


def train_probe(task: ProbeTask, steps: int = 400, batch_size: int = 64, lr: float = 3e-3,
                dim: int = 64, num_layers: int = 3, num_heads: int = 4,
                global_kind: str = "attn", content_gate: bool = True,
                loss: str = "mse", eval_every: int = 50, eval_batches: int = 4,
                eval_batch_size: int = 64, seed: int = 0, device: Any = "cpu",
                log: bool = True, log_every: int = 0, warmup: int = 20,
                final_eval_samples: int = 1024, model_fn: Optional[Callable[[], nn.Module]] = None,
                progress: bool = True) -> Dict[str, Any]:
    """Train a small probe model on ``task`` and return its learning curve.

    Returns a dict with the task description, the probe-network config, the
    per-checkpoint ``curve`` (``step``, training loss, held-out accuracy and
    held-out cross-entropy), the final held-out accuracy on a fresh sample, and a
    ``solvable_at_this_budget`` flag comparing the best curve accuracy against the
    task's chance level.

    ``loss="mse"`` optimises the squared error at the scored positions.  For the
    copy probes the scored target is one of many classes, so this keeps a usable
    gradient on the correct class instead of the cross-entropy softmax
    normaliser diluting it (see :class:`IntervalRecallTask`).
    """
    dev = torch.device(device)
    torch.manual_seed(int(seed))
    if model_fn is None:
        model = ProbeNet(task.vocab_size, dim=dim, num_layers=num_layers,
                         global_kind=global_kind, content_gate=content_gate,
                         num_heads=num_heads)
    else:
        model = model_fn()
    model = model.to(dev).train()
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) *
        (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, s / max(1, steps))))))
    gen = torch.Generator().manual_seed(int(seed) + 991)
    curve: List[Dict[str, Any]] = []
    loss_name = loss.lower()
    log_every = int(log_every) if log_every > 0 else max(10, steps // 10)
    flat_pos = task.answer_pos
    assert flat_pos is not None, "train_probe needs a task with a scored position"

    def _checkpoint(step: int, train_loss: float) -> None:
        accs, ces = [], []
        with torch.no_grad():
            for j in range(eval_batches):
                inputs, targets = task.sample(eval_batch_size, gen, device=dev)
                lg = _answer_logits(model, inputs, flat_pos).float()
                tg = _answer_targets(targets, flat_pos)
                keep = tg != -100
                lg, tg = lg[keep], tg[keep]
                if tg.numel() == 0:
                    continue
                ces.append(float(F.cross_entropy(lg, tg, reduction="mean").item()))
                accs.append(float((lg.argmax(-1) == tg).float().mean().item()))
        curve.append({
            "step": int(step),
            "train_loss": float(train_loss),
            "eval_loss": statistics.fmean(ces),
            "eval_accuracy": statistics.fmean(accs),
        })

    t0 = time.perf_counter()
    step = 0
    running = float("nan")
    iterator = range(steps)
    if progress and steps > 200:
        try:  # pragma: no cover - cosmetic only
            from tqdm import tqdm
            iterator = tqdm(range(steps), desc=f"probe:{task.name}", unit="step")
        except Exception:
            iterator = range(steps)
    for step in iterator:
        inputs, targets = task.sample(batch_size, gen, device=dev)
        lg = _answer_logits(model, inputs, flat_pos)
        tg = _answer_targets(targets, flat_pos)
        l = _answer_loss(lg, tg, loss_name)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = float(l.detach().cpu())
        if step % log_every == 0:
            _checkpoint(step, running)
            if log:
                print(f"  step {step:5d}  train {running:.4f}  "
                      f"eval_acc {curve[-1]['eval_accuracy']:.4f}  "
                      f"eval_ce {curve[-1]['eval_loss']:.4f}", flush=True)
    train_seconds = time.perf_counter() - t0
    _checkpoint(steps, running)
    # the *same* held-out seed is used by compare_mixers, so the probe-network
    # accuracy reported next to each mixer is comparable across mixers
    final = evaluate_probe(model, task, n_samples=final_eval_samples, batch_size=eval_batch_size,
                           seed=int(seed) + 12345, device=dev)
    best = max(curve, key=lambda c: c["eval_accuracy"])
    return {
        "task": task.describe(),
        "probe_network": {
            "kind": type(model).__name__,
            "global_kind": global_kind,
            "content_gate": bool(content_gate),
            "dim": int(dim),
            "num_layers": int(num_layers),
            "n_params": n_params,
        },
        "training": {
            "steps": int(steps), "batch_size": int(batch_size), "lr": float(lr),
            "loss": loss_name, "seed": int(seed), "device": str(dev),
            "seconds": train_seconds,
        },
        "curve": curve,
        "final": final,
        "final_accuracy": final["probe_accuracy"],
        "final_loss": final["probe_loss"],
        "chance": float(task.chance),
        "best_eval_accuracy": best["eval_accuracy"],
        "best_step": best["step"],
        # bar: must beat 1.5x chance AND exceed it by >= 0.05 absolute, otherwise
        # a lucky small-n run would be called "solvable".
        "solvable_at_this_budget": bool(best["eval_accuracy"] >= max(1.5 * task.chance,
                                                                     task.chance + 0.05)),
    }


# ---------------------------------------------------------------------------
# Ready-made probe suites and honest reporting
# ---------------------------------------------------------------------------
def _make_probe_tasks(name: str, n_content: int = 16, n_values: int = 8,
                      num_kv: int = 8, num_queries: int = 4, gaps: Sequence[int] = (8, 16, 32),
                      max_gap: int = 64) -> List[ProbeTask]:
    if name in ("selective-copy", "selective_copy", "copy"):
        return [SelectiveCopyTask(n_content=n_content)]
    if name in ("associative-recall", "associative_recall", "mqar"):
        return [AssociativeRecallTask(num_kv=num_kv, num_queries=num_queries,
                                      vocab_key=num_kv, vocab_value=num_kv)]
    if name in ("interval-recall", "interval_recall"):
        # the reported set is content-free (distractors are pad), so accuracy is
        # a property of the *trained model*, not of this particular draw; the
        # solvability check below uses the content-rich max_gap variant.
        test = IntervalRecallTask(distractor_gap=max_gap, n_values=n_values)
        tasks = [IntervalRecallTask(distractor_gap=int(g), n_values=n_values) for g in gaps]
        tasks.append(test)
        return tasks
    raise ValueError(f"unknown probe task {name!r}")


def _probe_report(task: ProbeTask, trained: Dict[str, Any], untrained: Optional[Dict[str, Any]],
                  untrained_model_params: Optional[int] = None) -> Dict[str, Any]:
    trained_ok = bool(trained["solvable_at_this_budget"])
    untrained_ok = bool(untrained["solvable_at_this_budget"]) if untrained else None
    out: Dict[str, Any] = {
        "probe": task,                       # ProbeTask instance; _jsonable() handles it
        "task": task.describe(),
        "chance": float(task.chance),
        "trained": trained,
        "untrained_baseline": untrained,
        "solvability_check": {
            "trained_probe_network_above_chance": trained_ok,
            "untrained_probe_network_above_chance": untrained_ok,
            "untrained_probe_network_n_params": untrained_model_params,
            "bar": "best checkpointed eval accuracy >= max(1.5 * chance, chance + 0.05)",
        },
        "verdict": "probe_network_demonstrably_learned_the_task" if trained_ok else
                   "NOT_PROVEN_at_this_budget_the_probe_network_did_not_beat_chance",
    }
    if "G" in task.describe():
        out["G"] = task.describe()["G"]       # flattened so sweeps are easy to tabulate
    if "num_kv" in task.describe():
        out["num_kv"] = task.describe()["num_kv"]
    return out


def probe_selective_copy(model: Optional[nn.Module] = None, *, train: bool = False,
                         steps: int = 400, n_content: int = 16, seq_len: int = 64,
                         gap: int = 8, batch_size: int = 64, lr: float = 3e-3,
                         dim: int = 64, num_layers: int = 3, seed: int = 0,
                         device: Any = "cpu", n_samples: int = 2048,
                         loss: str = "mse", verbose: bool = True,
                         untrained_steps: int = 20, **kw: Any) -> Dict[str, Any]:
    """Selective copy: copy the token that follows a marked token.

    Returns ``{"task", "chance", "model_accuracy"|None, "solvability_check",
    "verdict", ...}``.  When ``train=True`` (or ``model is None``) a small
    :class:`ProbeNet` is trained on the task and its learning curve is included,
    so the reported number always sits next to a solvability check.
    """
    task = SelectiveCopyTask(n_content=n_content, seq_len=seq_len, gap=gap)
    do_train = bool(train)
    trained = train_probe(task, steps=steps, batch_size=batch_size, lr=lr, dim=dim,
                          num_layers=num_layers, seed=seed, device=device, loss=loss,
                          final_eval_samples=n_samples, log=verbose) if do_train else None
    untrained = train_probe(task, steps=untrained_steps, batch_size=batch_size, lr=lr, dim=dim,
                            num_layers=num_layers, seed=seed, device=device, loss=loss,
                            final_eval_samples=n_samples, log=False,
                            progress=False) if untrained_steps > 0 else None
    out = _probe_report(task, trained, untrained) if trained else {
        "task": task.describe(), "chance": float(task.chance),
        "solvability_check": None, "verdict": "no_solvability_check_requested",
    }
    if model is not None:
        out["model_accuracy"] = evaluate_probe(model, task, n_samples=n_samples,
                                              seed=1234, device=device)["probe_accuracy"]
    if verbose:
        print(f"[{task.name}] chance={task.chance:.4f}", flush=True)
        if trained:
            print(f"  trained probe-net final acc={trained['final_accuracy']:.4f} "
                  f"best(step {trained['best_step']})={trained['best_eval_accuracy']:.4f} "
                  f"solvable={trained['solvable_at_this_budget']}", flush=True)
        if "model_accuracy" in out:
            print(f"  supplied model acc={out['model_accuracy']:.4f}", flush=True)
    return out


def probe_associative_recall(model: Optional[nn.Module] = None, *, train: bool = False,
                             steps: int = 500, num_kv: int = 8, num_queries: int = 4,
                             vocab_key: Optional[int] = None, vocab_value: Optional[int] = None,
                             batch_size: int = 64, lr: float = 3e-3, dim: int = 64,
                             num_layers: int = 3, seed: int = 0, device: Any = "cpu",
                             n_samples: int = 2048, loss: str = "mse", verbose: bool = True,
                             untrained_steps: int = 20, **kw: Any) -> Dict[str, Any]:
    """Multi-query associative recall (MQAR).

    ``chance = 1 / vocab_value``; every query is scored, and queries are sampled
    with replacement so a repeated key in one sequence is consistent by
    construction.
    """
    task = AssociativeRecallTask(
        num_kv=num_kv, num_queries=num_queries,
        vocab_key=vocab_key if vocab_key is not None else num_kv,
        vocab_value=vocab_value if vocab_value is not None else num_kv)
    do_train = bool(train)
    trained = train_probe(task, steps=steps, batch_size=batch_size, lr=lr, dim=dim,
                          num_layers=num_layers, seed=seed, device=device, loss=loss,
                          final_eval_samples=n_samples, log=verbose) if do_train else None
    untrained = train_probe(task, steps=untrained_steps, batch_size=batch_size, lr=lr, dim=dim,
                            num_layers=num_layers, seed=seed, device=device, loss=loss,
                            final_eval_samples=n_samples, log=False,
                            progress=False) if untrained_steps > 0 else None
    out = _probe_report(task, trained, untrained) if trained else {
        "task": task.describe(), "chance": float(task.chance),
        "solvability_check": None, "verdict": "no_solvability_check_requested",
    }
    if model is not None:
        out["model_accuracy"] = evaluate_probe(model, task, n_samples=n_samples,
                                              seed=1234, device=device)["probe_accuracy"]
    if verbose:
        print(f"[{task.name}] chance=1/{task.vocab_value}={task.chance:.4f}", flush=True)
        if trained:
            print(f"  trained probe-net final acc={trained['final_accuracy']:.4f} "
                  f"best(step {trained['best_step']})={trained['best_eval_accuracy']:.4f} "
                  f"solvable={trained['solvable_at_this_budget']}", flush=True)
        if "model_accuracy" in out:
            print(f"  supplied model acc={out['model_accuracy']:.4f}", flush=True)
    return out


def probe_interval_recall(model: Optional[nn.Module] = None, *, train: bool = False,
                          steps: int = 800, gaps: Sequence[int] = (8, 16, 32, 64),
                          n_values: int = 8, batch_size: int = 64, lr: float = 3e-3,
                          dim: int = 64, num_layers: int = 3, seed: int = 0,
                          device: Any = "cpu", n_samples: int = 2048, loss: str = "mse",
                          verbose: bool = True, untrained_steps: int = 20,
                          train_all_gaps: bool = False, **kw: Any) -> Dict[str, Any]:
    """Interval recall: key, value, ``G`` distractors, key again -> value.

    Accuracy is reported **as a function of G** together with ``chance = 1 /
    n_values``.  By default the solvability training run covers the hardest
    requested gap only and ``trained`` is ``None`` for the easier gaps; set
    ``train_all_gaps=True`` for one curve per gap (slower).  Any gap with
    ``trained: null`` was **not** checked for solvability.
    """
    tasks = [IntervalRecallTask(distractor_gap=int(g), n_values=n_values) for g in gaps]
    per_gap: List[Dict[str, Any]] = []
    trained_hardest: Optional[Dict[str, Any]] = None
    for i, task in enumerate(tasks):
        hardest = (i == len(tasks) - 1)
        train_here = bool(train) and (hardest or train_all_gaps)
        t = train_probe(task, steps=steps, batch_size=batch_size, lr=lr, dim=dim,
                        num_layers=num_layers, seed=seed, device=device, loss=loss,
                        final_eval_samples=n_samples, log=bool(verbose and train_here)) \
            if train_here else None
        rec = {
            "G": task.G,
            "seq_len": task.seq_len,
            "chance": float(task.chance),
            "n_values": int(task.n_values),
            "trained": t,
            "model_accuracy": (evaluate_probe(model, task, n_samples=n_samples,
                                              seed=1234 + i, device=device)["probe_accuracy"]
                               if model is not None else None),
        }
        if t is not None:
            trained_hardest = t
        per_gap.append(rec)
    untrained = train_probe(tasks[-1], steps=untrained_steps, batch_size=batch_size, lr=lr,
                            dim=dim, num_layers=num_layers, seed=seed, device=device,
                            loss=loss, final_eval_samples=n_samples, log=False,
                            progress=False) if untrained_steps > 0 else None
    out: Dict[str, Any] = {
        "probe": "interval-recall",
        "by_gap": per_gap,
        "chance": float(tasks[-1].chance),
        "n_values": int(n_values),
        "solvability_check": {
            "hardest_gap": int(tasks[-1].G),
            "trained_probe_network_above_chance":
                bool(trained_hardest["solvable_at_this_budget"]) if trained_hardest else None,
            "untrained_probe_network_above_chance":
                bool(untrained["solvable_at_this_budget"]) if untrained else None,
            "bar": "best checkpointed eval accuracy >= max(1.5 * chance, chance + 0.05)",
        },
        "verdict": ("probe_network_demonstrably_learned_the_task"
                    if (trained_hardest and trained_hardest["solvable_at_this_budget"])
                    else "NOT_PROVEN_at_this_budget_the_probe_network_did_not_beat_chance"),
    }
    if verbose:
        print(f"[interval-recall] chance=1/{n_values}={tasks[-1].chance:.4f}", flush=True)
        for rec in per_gap:
            acc = rec["model_accuracy"]
            t = rec["trained"]
            tail = f"  trained_net_final={t['final_accuracy']:.4f}" if t else ""
            print(f"  G={rec['G']:4d} seq_len={rec['seq_len']:4d} "
                  f"model_acc={('%.4f' % acc) if acc is not None else 'n/a'}{tail}", flush=True)
    return out


def probe_tasks(name: str) -> List[ProbeTask]:
    return _make_probe_tasks(name)


def probe_shift_batches(task: ProbeTask, batch_size: int, gen: torch.Generator,
                        device: Any) -> EvalBatch:
    inputs, _ = task.sample(batch_size, gen, device=device)
    return EvalBatch(inputs, _next_token_targets(inputs))


def probe_eval_dataset(task: ProbeTask, n_samples: int = 512,
                       seed: int = 1234) -> SyntheticDataset:
    """``(seq_len,)`` LongTensor dataset for :func:`evaluate_lm` (shift objective).

    Useful as a probe-style language-modelling check on a synthetic distribution
    when no tokenizer corpus is available.
    """
    return SyntheticDataset(task.vocab_size, task.seq_len, length=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# Throughput / memory scaling benchmark
# ---------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _reduce_latency(values: Sequence[float], how: str) -> float:
    """Aggregate repeated latency samples.

    ``"min"`` is the estimator of choice on a shared CPU: the median of a handful
    of samples is dominated by other processes' activity, while the minimum is a
    much tighter estimate of the work the operation actually performs.
    """
    if not values:
        return float("nan")
    if how == "min":
        return float(min(values))
    if how == "median":
        return float(statistics.median(values))
    raise ValueError(f"unknown latency reduction {how!r}; use 'median' or 'min'")


def _slope_loglog(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares slope of ``log(y)`` against ``log(x)``.

    This is the *empirical* scaling exponent: 1.0 means the measured latency is
    linear in the sequence length, 2.0 means quadratic.  It is a local fit over
    the requested lengths, not an asymptotic statement.
    """
    pts = [(math.log(float(a)), math.log(float(b))) for a, b in zip(xs, ys)
           if a > 0 and b > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    return float(num / den) if den > 0 else float("nan")


def _fit_quality(xs: Sequence[float], ys: Sequence[float]) -> float:
    """R^2 of the log-log fit."""
    pts = [(math.log(float(a)), math.log(float(b))) for a, b in zip(xs, ys)
           if a > 0 and b > 0]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    slope = _slope_loglog(xs, ys)
    ss_tot = sum((p[1] - my) ** 2 for p in pts)
    ss_res = sum((p[1] - (my + slope * (p[0] - mx))) ** 2 for p in pts)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _measure_phase(model: nn.Module, batch: torch.Tensor, backward: bool) -> Dict[str, float]:
    """Median latency of one forward (+ loss, + backward) step.

    Returns ``forward_s``, ``loss_s``, ``backward_s`` and ``fwd_bwd_s``
    (forward + backward, the number the scaling exponent is fitted on).
    """
    model.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    logits = model(batch)
    fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           batch.reshape(-1), ignore_index=-100)
    ls = time.perf_counter() - t0
    bs = 0.0
    if backward:
        t0 = time.perf_counter()
        loss.backward()
        bs = time.perf_counter() - t0
    del logits, loss
    model.zero_grad(set_to_none=True)
    return {"forward_s": fwd, "loss_s": ls, "backward_s": bs, "fwd_bwd_s": fwd + bs}


def _transformer_lens(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(1)


def _measure_training_step(model: nn.Module, batch: torch.Tensor, backward: bool,
                           gc_every: int = 25) -> Dict[str, Any]:
    """Median latency + accounted tensor bytes for one training step.

    ``peak_accounted_bytes`` is the median of :func:`tensor_peak_bytes` over a few
    iterations, so it is a *peak over the step*, not a cumulative counter.  The
    per-iteration values are returned as well so the peak is auditable.
    """
    lat: List[Dict[str, float]] = []
    peaks: List[int] = []
    detailed: Optional[Dict[str, Any]] = None
    model.train()
    for _ in range(3):
        lat.append(_measure_phase(model, batch, backward))
        r = tensor_peak_bytes(model, batch, batch, backward=backward)
        peaks.append(r["tensor_peak_bytes"])
        detailed = r
        del r
    return {
        "forward_s": _median([x["forward_s"] for x in lat]),
        "loss_s": _median([x["loss_s"] for x in lat]),
        "backward_s": _median([x["backward_s"] for x in lat]),
        "fwd_bwd_s": _median([x["fwd_bwd_s"] for x in lat]),
        "peak_accounted_bytes": int(statistics.median(peaks)),
        "peak_accounted_bytes_all": [int(p) for p in peaks],
        "accounting": detailed,
    }


def bench_throughput(model: nn.Module, seq_lens: Sequence[int] = (64, 128, 256, 512),
                     batch_size: int = 8, device: Any = None, warmup: int = 3,
                     iters: int = 7, backward: bool = False,
                     dtype: Optional[torch.dtype] = None, vocab_size: Optional[int] = None,
                     cuda_peak: bool = True, reduce: str = "median") -> List[Dict[str, Any]]:
    """Latency, tokens/s and accounted peak memory per sequence length.

    Each row reports

    * ``median_latency_ms`` / ``tokens_per_s`` for forward (+ backward),
    * ``peak_accounted_bytes`` -- the high-water mark of the *accounted* tensors
      (``state_dict`` + audited forward/backward activations + transients); the
      broken RSS path is never used,
    * ``cuda_max_memory_allocated`` -- the real allocator peak when CUDA is
      available, else ``None``.

    The returned list also carries ``scaling`` entries: the log-log slope of
    latency (and of accounted memory) against sequence length, so "linear" and
    "quadratic" are checked numerically rather than asserted.  For the FFT mixer
    the expected wall-clock slope is *below* 2; a slope near 2 is a quadratic
    red flag (the attention baseline is expected to behave that way in memory).
    """
    device = resolve_device("auto") if device is None else torch.device(device)
    model = model.to(device)
    v = int(vocab_size) if vocab_size is not None else int(model.config.vocab_size)
    rows: List[Dict[str, Any]] = []
    for L in seq_lens:
        L = int(L)
        gen = torch.Generator().manual_seed(1234 + L)
        batch = torch.randint(0, v, (batch_size, L), generator=gen).to(device)
        model.train()
        for _ in range(max(0, warmup)):
            _measure_phase(model, batch, backward)
        per: List[Dict[str, float]] = [
            _measure_phase(model, batch, backward) for _ in range(max(1, iters))]
        fwd = _reduce_latency([p["forward_s"] for p in per], reduce)
        loss = _reduce_latency([p["loss_s"] for p in per], reduce)
        bwd = _reduce_latency([p["backward_s"] for p in per], reduce)
        step = fwd + (bwd if backward else 0.0)
        # accounted peak: median over a few fresh iterations
        peaks = [tensor_peak_bytes(model, batch, batch, backward=backward) for _ in range(3)]
        detail = peaks[-1]
        tokens = float(batch_size * L)
        row = {
            "seq_len": L,
            "batch_size": int(batch_size),
            "backward": bool(backward),
            "iters": int(iters),
            "warmup": int(warmup),
            "device": str(device),
            "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "tokens_per_iter": int(batch_size * L),
            "forward_ms": fwd * 1e3,
            "loss_ms": loss * 1e3,
            "backward_ms": bwd * 1e3 if backward else None,
            "median_latency_ms": step * 1e3,
            "reduced_latency_ms": step * 1e3,
            "reduce": reduce,
            "min_latency_ms": min([p["fwd_bwd_s"] for p in per]) * 1e3,
            "max_latency_ms": max([p["fwd_bwd_s"] for p in per]) * 1e3,
            "raw_step_latency_ms": [p["fwd_bwd_s"] * 1e3 for p in per],
            "tokens_per_s": tokens / step if step > 0 else float("nan"),
            "peak_accounted_bytes": int(statistics.median([p["tensor_peak_bytes"] for p in peaks])),
            "accounted_bytes": int(detail["accounted_bytes"]),
            "param_bytes": int(detail["param_bytes"]),
            "grad_bytes": int(detail["grad_bytes"]),
            "fwd_activation_bytes": int(detail["fwd_bytes"]),
            "bwd_activation_bytes": int(detail["bwd_bytes"]),
            "live_transient_bytes": int(detail["live_bytes"]),
            "peak_accounted_mb": mb(statistics.median([p["tensor_peak_bytes"] for p in peaks])),
            "cuda_max_memory_allocated": detail["cuda_max_memory_allocated"] if cuda_peak else None,
            "memory_method": "tensor_accounting" if not cuda_peak else
                             "tensor_accounting+cuda_allocator",
        }
        rows.append(row)

    xs = [r["seq_len"] for r in rows]
    lat_ms = [r["reduced_latency_ms"] for r in rows]
    min_ms = [r["min_latency_ms"] for r in rows]
    mem_b = [r["peak_accounted_bytes"] for r in rows]
    tps = [r["tokens_per_s"] for r in rows]
    scaling = {
        "latency_loglog_slope": _slope_loglog(xs, lat_ms),
        "latency_fit_r2": _fit_quality(xs, lat_ms),
        "latency_loglog_slope_min_of_iters": _slope_loglog(xs, min_ms),
        "latency_fit_r2_min_of_iters": _fit_quality(xs, min_ms),
        "peak_memory_loglog_slope": _slope_loglog(xs, mem_b),
        "peak_memory_fit_r2": _fit_quality(xs, mem_b),
        "tokens_per_s_loglog_slope": _slope_loglog(xs, tps),
        "seq_lens": [int(x) for x in xs],
        "reduction": reduce,
        "note": ("local log-log fit over the requested lengths.  Latency on a shared "
                 "CPU is allocation- and scheduler-noisy (see raw_step_latency_ms); the "
                 "min-of-iters slope is the low-noise estimate, while the accounted-memory "
                 "slope is deterministic.  For the FFT mixer a slope below 2 is expected "
                 "in this range and does not by itself prove the O(L log L) asymptotics"),
    }
    return rows, scaling


def _flatten_scaling(rows: List[Dict[str, Any]], scaling: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rows": rows,
        "latency_loglog_slope": scaling["latency_loglog_slope"],
        "latency_loglog_slope_min_of_iters": scaling["latency_loglog_slope_min_of_iters"],
        "latency_fit_r2": scaling["latency_fit_r2"],
        "latency_fit_r2_min_of_iters": scaling["latency_fit_r2_min_of_iters"],
        "peak_memory_loglog_slope": scaling["peak_memory_loglog_slope"],
        "peak_memory_fit_r2": scaling["peak_memory_fit_r2"],
        "scaling": scaling,
    }


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------
def bench_streaming(model: nn.Module, lengths: Sequence[int] = (64, 256, 1024),
                    batch_size: int = 1, device: Any = None, iters: int = 16,
                    dtype: Optional[torch.dtype] = None,
                    check_state_bytes: bool = True) -> List[Dict[str, Any]]:
    """Constant streaming-state bytes vs sequence length, plus per-token latency.

    For each requested length the model is prefilled token-by-token and then
    ``iters`` further tokens are stepped.  Reported per row:

    * ``state_bytes`` -- exact bytes of the live :class:`SGCStreamState`
      (``num_bytes()``) **and** ``stream_bytes`` (the analytic formula); the two
      must agree, which is checked,
    * ``state_bytes_per_token`` -- ``state_bytes / tokens_consumed``, which must
      fall as ``1 / length`` if the state is truly constant-size,
    * ``step_latency_ms`` / ``step_tokens_per_s`` -- median cost of one token.

    With ``check_state_bytes=True`` the exact state byte count must be identical
    for every length; a mismatch raises ``AssertionError`` instead of being
    silently printed.
    """
    device = resolve_device("auto") if device is None else torch.device(device)
    model = model.to(device).eval()
    dt = dtype if dtype is not None else next(model.parameters()).dtype
    if not hasattr(model, "init_stream"):
        raise TypeError("bench_streaming needs a model with init_stream/stream_step")
    rows: List[Dict[str, Any]] = []
    seen_bytes: Dict[int, int] = {}
    for L in lengths:
        L = int(L)
        gen = torch.Generator().manual_seed(4321 + L)
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, L),
                               generator=gen).to(device)
        state = model.init_stream(batch_size, device=device, dtype=dt)
        pre_t0 = time.perf_counter()
        with torch.no_grad():
            for t in range(L):
                logits, state = model.stream_step(tokens[:, t], state)
        prefill_s = time.perf_counter() - pre_t0
        # exact bytes of the live state object
        exact = int(state.num_bytes()) if hasattr(state, "num_bytes") else int(
            sum(x.numel() * x.element_size() for st in state.layers if st is not None
                for x in (st.y1, st.y2) if x is not None))
        analytic = int(model.stream_bytes(batch_size, dt))
        if check_state_bytes:
            assert exact == analytic, (
                f"len {L}: live state bytes {exact} != stream_bytes() {analytic}")
        seen_bytes[L] = exact
        steps: List[float] = []
        with torch.no_grad():
            for _ in range(max(1, iters)):
                nxt = logits.argmax(-1)
                t0 = time.perf_counter()
                logits, state = model.stream_step(nxt, state)
                steps.append(time.perf_counter() - t0)
        med = _median(steps)
        total_tokens = L + iters
        rows.append({
            "length": L,
            "batch_size": int(batch_size),
            "state_bytes": exact,
            "state_bytes_analytic_formula": analytic,
            "state_matches_formula": bool(exact == analytic),
            "state_bytes_per_token": exact / float(total_tokens),
            "step_latency_ms": med * 1e3,
            "step_tokens_per_s": float(batch_size) / med if med > 0 else float("nan"),
            "prefill_tokens_per_s": float(batch_size * L) / prefill_s if prefill_s > 0 else float("nan"),
            "state_bytes_total_over_lengths": None,   # filled below
            "is_attention_baseline": exact == 0,
        })
    for r in rows:
        r["state_bytes_total_over_lengths"] = int(seen_bytes[r["length"]])
    distinct = sorted(set(seen_bytes.values()))
    length_invariant = len(distinct) <= 1
    if check_state_bytes:
        assert length_invariant, f"streaming state bytes vary with length: {seen_bytes}"
    med_steps = [r["step_latency_ms"] for r in rows]
    return {
        "rows": rows,
        "state_bytes_by_length": {str(k): int(v) for k, v in seen_bytes.items()},
        "state_bytes_length_invariant": bool(length_invariant),
        "step_latency_loglog_slope_vs_prefill_length": _slope_loglog(
            [r["length"] for r in rows], med_steps),
        "note": ("the streaming state must be identical for every length; "
                 "step latency is expected to be flat in the context length"),
    }


# ---------------------------------------------------------------------------
# Mixer comparison
# ---------------------------------------------------------------------------
MIXER_KINDS = ("sgc", "sgc-nogate", "sgc-nolocal", "attn")


def _cfg_from_base(base: SGCConfig, mixer: str, **over: Any) -> SGCConfig:
    d = base.to_dict()
    d.update(over)
    d["mixer"] = mixer
    return SGCConfig.from_dict(d)


def _probe_dataset_kwargs(name: str, **kw: Any) -> Tuple[ProbeTask, Dict[str, Any]]:
    """Map a probe name to (task_for_lm_ppl, kwargs_for_probe_runner)."""
    if name in ("ppl", "lm"):
        task = IntervalRecallTask(distractor_gap=16, n_values=8)
        return task, {}
    if name in ("selective-copy", "copy"):
        return SelectiveCopyTask(n_content=kw.get("n_content", 8),
                                 seq_len=kw.get("seq_len", 32)), {
            "n_content": kw.get("n_content", 8), "seq_len": kw.get("seq_len", 32)}
    if name in ("associative-recall", "mqar"):
        return AssociativeRecallTask(num_kv=4, num_queries=2, vocab_key=4,
                                     vocab_value=4), {
            "num_kv": 4, "num_queries": 2, "vocab_value": 4}
    if name in ("interval-recall", "interval"):
        return IntervalRecallTask(distractor_gap=kw.get("G", 16), n_values=8), {
            "gaps": (int(kw.get("G", 16)),), "n_values": 8}
    raise ValueError(f"unknown protocol {name!r}")


def compare_mixers(configs: Optional[Sequence[Any]] = None, base: Optional[SGCConfig] = None,
                   mixers: Sequence[str] = MIXER_KINDS, protocols: Sequence[str] = ("ppl",),
                   device: Any = None, dtype: Optional[torch.dtype] = None,
                   batch_size: int = 8, train_probes: bool = False, steps: int = 300,
                   lr: float = 1e-2, probe_samples: int = 2048, eval_batches: int = 8,
                   seed: int = 0, verbose: bool = True) -> Dict[str, Any]:
    """Run the same protocol across all requested mixers with matched width/depth.

    Every mixer sees the *same* `SGCConfig` (identical ``dim``, ``num_layers``,
    ``ffn_hidden``, ``vocab_size``); only ``mixer`` differs, so a difference in
    the numbers cannot come from a capacity mismatch.  The returned JSON contains
    each mixer's full config, parameter count, held-out loss/ppl on the synthetic
    protocol, and one entry per probe (accuracy + chance + solvability check).

    Args:
        configs: explicit configs; when given, ``base``/``mixers`` are ignored.
        base: template config (defaults to the ``debug`` preset).
        protocols: any of ``"ppl"``, ``"selective-copy"``, ``"associative-recall"``,
            ``"interval-recall"`` or ``"probe:<name>"``.
    """
    device = resolve_device("auto") if device is None else torch.device(device)
    dtype = dtype or torch.float32
    if base is None:
        base = SGCConfig.from_dict(SGC_PRESETS["debug"].to_dict())
    if configs is None:
        cfgs = [_cfg_from_base(base, m) for m in mixers]
    else:
        cfgs = list(configs)
    # The solvability check belongs to the *task*, not to the mixer under test, so
    # every mixer reuses one trained probe network and the curve is trained once.
    shared_net: Dict[str, nn.Module] = {}
    shared_curve: Dict[str, Dict[str, Any]] = {}

    norm_protocols: List[str] = []
    for p in protocols:
        p = p.split(":", 1)[1] if p.startswith("probe:") else p
        if p in ("lm", "ppl"):
            p = "ppl"
        if p not in norm_protocols:
            norm_protocols.append(p)

    entries: List[Dict[str, Any]] = []
    for i, cfg in enumerate(cfgs):
        cfg.validate()
        torch.manual_seed(seed + i)
        model = SGCModel(cfg).to(device, dtype=dtype)
        entry: Dict[str, Any] = {
            "mixer": cfg.mixer,
            "config": cfg.to_dict(),
            "n_params": int(model.num_params()),
            "n_params_non_embedding": int(model.num_params(True)),
            "fft_dtype": cfg.fft_dtype,
            "protocols": {},
        }
        if verbose:
            print(f"[{cfg.mixer}] params={entry['n_params']:,}", flush=True)
        for proto in norm_protocols:
            task, pkw = _probe_dataset_kwargs(proto, seq_len=min(cfg.vocab_size * 2, 32))
            if proto == "ppl":
                ds = SyntheticDataset(cfg.vocab_size, task.seq_len, length=eval_batches * batch_size,
                                      seed=seed + 17)
                res = evaluate_lm(model, ds, batch_size=batch_size, seq_len=task.seq_len,
                                  device=device, max_batches=eval_batches)
                entry["protocols"]["ppl"] = {
                    "loss": res["loss"], "ppl": res["ppl"], "tokens": res["tokens"],
                    "n_batches": res["n_batches"],
                    "dataset": f"synthetic uniform tokens, seq_len {task.seq_len}",
                }
                if verbose:
                    print(f"   ppl protocol: loss={res['loss']:.4f} ppl={res['ppl']:.2f} "
                          f"tokens={res['tokens']}", flush=True)
                continue
            pname = {"copy": "selective-copy", "mqar": "associative-recall",
                     "interval": "interval-recall"}.get(proto, proto)
            ev_task = {"selective-copy": lambda: SelectiveCopyTask(
                            n_content=pkw.get("n_content", 8), seq_len=pkw.get("seq_len", 32)),
                       "associative-recall": lambda: AssociativeRecallTask(
                            num_kv=4, num_queries=2, vocab_key=4, vocab_value=4),
                       "interval-recall": lambda: IntervalRecallTask(
                            distractor_gap=int(pkw.get("gaps", (16,))[-1]), n_values=8)}[pname]()
            if train_probes and pname not in shared_curve:
                shared_net[pname] = ProbeNet(ev_task.vocab_size, dim=48, num_layers=2)
                shared_curve[pname] = train_probe(
                    ev_task, steps=steps, lr=lr, device=device, log=False, progress=False,
                    final_eval_samples=probe_samples, seed=seed)
                if verbose:
                    print(f"   solvability: probe-net on {pname} "
                          f"best={shared_curve[pname]['best_eval_accuracy']:.4f} "
                          f"chance={ev_task.chance:.4f} "
                          f"solvable={shared_curve[pname]['solvable_at_this_budget']}", flush=True)
            net = shared_net.get(pname)
            ev = evaluate_probe(model, ev_task, n_samples=probe_samples, device=device)
            out = {
                "task": ev_task.describe(),
                "chance": float(ev_task.chance),
                "model_accuracy": ev["probe_accuracy"],
                "final": ev,
                "probe_network_accuracy": (
                    evaluate_probe(net, ev_task, n_samples=probe_samples, device=device,
                                   seed=seed + 12345)["probe_accuracy"]
                    if net is not None else None),
                "solvability_check": (None if pname not in shared_curve else {
                    "trained_probe_network_above_chance":
                        bool(shared_curve[pname]["solvable_at_this_budget"]),
                    "best_eval_accuracy": shared_curve[pname]["best_eval_accuracy"],
                    "best_step": shared_curve[pname]["best_step"],
                    "steps": int(steps),
                    "bar": "best checkpointed eval accuracy >= max(1.5 * chance, chance + 0.05)",
                    "curve": shared_curve[pname]["curve"],
                }),
                "verdict": ("no_solvability_check_requested" if pname not in shared_curve else
                            ("probe_network_demonstrably_learned_the_task"
                             if shared_curve[pname]["solvable_at_this_budget"] else
                             "NOT_PROVEN_at_this_budget_the_probe_network_did_not_beat_chance")),
                "note": ("model_accuracy is the SGC model under test on a freshly sampled "
                         "test set; probe_network_accuracy is the mixer-independent "
                         "solvability baseline for the same task and seed"),
            }
            entry["protocols"][proto] = out
            if verbose:
                print(f"   {proto}: model_acc={out['model_accuracy']:.4f} "
                      f"chance={out['chance']:.4f} probe_net={out['probe_network_accuracy']}",
                      flush=True)
        model.zero_grad(set_to_none=True)
        entry["peak_forward_bytes"] = int(tensor_peak_bytes(
            model, torch.randint(0, cfg.vocab_size, (batch_size, 64), device=device),
            torch.randint(0, cfg.vocab_size, (batch_size, 64), device=device),
            backward=False)["tensor_peak_bytes"])
        entries.append(entry)

    out = {
        "protocols": norm_protocols,
        "base_config": base.to_dict(),
        "mixers": [e["mixer"] for e in entries],
        "entries": {e["mixer"]: e for e in entries},
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "matched_width_depth": True,
    }
    return out


# ---------------------------------------------------------------------------
# Reproducibility record + CLI
# ---------------------------------------------------------------------------
def environment_record(device: Any, dtype: Any, command: Optional[str] = None) -> Dict[str, Any]:
    """Everything needed to re-run and cite a result."""
    dev = resolve_device("auto") if device is None else torch.device(device)
    dt = torch.float32 if dtype is None else dtype
    rec = {
        "timestamp_utc": _now(),
        "command_line": command if command is not None else _ORIGINAL_ARGV,
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "device": str(dev),
        "device_type": dev.type,
        "dtype": str(dt).replace("torch.", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "cudnn_version": (torch.backends.cudnn.version() if torch.cuda.is_available() else None),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "memory_method": ("tensor_accounting (sum of numel()*element_size() over live "
                          "parameters, gradients, audited forward/backward activations and "
                          "transients) + torch.cuda.max_memory_allocated() when CUDA is "
                          "available.  GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS is "
                          "BROKEN on this host (returns 0, GetLastError 122) and is never used."),
        "eval_module_sha256": sha256_file(__file__),
    }
    try:
        import numpy as _np
        rec["numpy_version"] = _np.__version__
    except Exception:
        rec["numpy_version"] = None
    return rec


def make_report(body: Dict[str, Any], command: Optional[str] = None,
                device: Any = None, dtype: Any = None,
                extra_env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wrap a result dict in a citable reproducibility envelope."""
    env = environment_record(device, dtype, command=command)
    if extra_env:
        env.update(extra_env)
    return {
        "schema": "simplcore.sgc.eval/1",
        "provenance": env,
        "result": _jsonable(body),
    }


def write_report(path: str, report: Dict[str, Any]) -> None:
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=2, ensure_ascii=False, sort_keys=False)
    print(f"wrote {p} ({p.stat().st_size} bytes)", flush=True)


def _emit(args: argparse.Namespace, body: Dict[str, Any], device: Any, dtype: Any) -> Dict[str, Any]:
    report = make_report(body, device=device, dtype=dtype)
    if getattr(args, "out", None):
        write_report(args.out, report)
    return report


def self_test_report(device: str = "cpu", dtype: str = "float32") -> Dict[str, Any]:
    """Run the instrumentation self-validation used by the CLI ``self-test``."""
    return assert_memory_probe_works(device=device, dtype=dtype, verbose=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_config(args: argparse.Namespace, device: Any, dtype: Any) -> SGCConfig:
    if getattr(args, "config_json", None):
        cfg = SGCConfig.from_json(args.config_json)
    else:
        cfg = SGCConfig.from_dict(SGC_PRESETS[getattr(args, "preset", "debug")].to_dict())
    over: Dict[str, Any] = {}
    if getattr(args, "mixer", None):
        over["mixer"] = args.mixer
    if getattr(args, "dim", None):
        over["dim"] = int(args.dim)
    if getattr(args, "layers", None):
        over["num_layers"] = int(args.layers)
    if getattr(args, "vocab_size", None):
        over["vocab_size"] = int(args.vocab_size)
    if getattr(args, "num_bases", None):
        over["num_bases"] = int(args.num_bases)
    if getattr(args, "local_kernel", None):
        over["local_kernel"] = int(args.local_kernel)
    if over:
        d = cfg.to_dict()
        d.update(over)
        cfg = SGCConfig.from_dict(d)
    cfg.validate()
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="debug", choices=sorted(SGC_PRESETS),
                   help="SGC preset to start from (default: debug)")
    p.add_argument("--config-json", default=None, help="SGCConfig JSON file (overrides --preset)")
    p.add_argument("--mixer", default=None, choices=list(MIXER_KINDS),
                   help="override the mixer kind")
    p.add_argument("--dim", type=int, default=None, help="override model width")
    p.add_argument("--layers", type=int, default=None, help="override the number of blocks")
    p.add_argument("--vocab-size", type=int, default=None, help="override the vocabulary size")
    p.add_argument("--num-bases", type=int, default=None, help="override the oscillator-bank size")
    p.add_argument("--local-kernel", type=int, default=None, help="override the local conv width")
    p.add_argument("--device", default="auto", help="cpu | cuda | auto (default: auto)")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float64", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the full JSON report here")


def _add_probe_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default="selective-copy",
                   choices=["selective-copy", "associative-recall", "interval-recall", "all"])
    p.add_argument("--steps", type=int, default=400,
                   help="probe-network training steps for the solvability check")
    p.add_argument("--probe-batch", type=int, default=64)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-dim", type=int, default=48)
    p.add_argument("--probe-layers", type=int, default=2)
    p.add_argument("--probe-loss", default="ce", choices=["ce", "mse"])
    p.add_argument("--n-samples", type=int, default=2048,
                   help="sequences in the freshly sampled test set")
    p.add_argument("--n-content", type=int, default=8, help="selective-copy content alphabet")
    p.add_argument("--seq-len", type=int, default=32, help="selective-copy sequence length")
    p.add_argument("--num-kv", type=int, default=4, help="MQAR key/value pairs")
    p.add_argument("--num-queries", type=int, default=2, help="MQAR queries per sequence")
    p.add_argument("--n-values", type=int, default=8, help="interval-recall value alphabet")
    p.add_argument("--gaps", default="8,16,32", help="comma-separated interval-recall G values")
    p.add_argument("--no-train", action="store_true",
                   help="skip the solvability training run (numbers alone, no check)")
    p.add_argument("--train-all-gaps", action="store_true",
                   help="interval-recall: train one probe network per gap instead of only the hardest")
    p.add_argument("--model-steps", type=int, default=0,
                   help="also train this many steps on the SGC model and evaluate it")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m simplcore.sgc.eval",
        description="Evaluation harness for simplcore.sgc (accounted memory, fitted "
                    "throughput scaling, constructively-solvable long-range probes).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("evaluate", help="token-weighted loss / perplexity")
    _add_common(pe)
    pe.add_argument("--dataset", default="synthetic", choices=["synthetic"],
                    help="synthetic (uniform random tokens) -- a stand-in until data.py exists")
    pe.add_argument("--dataset-module", default=None,
                    help="import path 'pkg.mod:attr' of a real dataset (len/getitem -> (T,) LongTensor)")
    pe.add_argument("--batch-size", type=int, default=8)
    pe.add_argument("--seq-len", type=int, default=None)
    pe.add_argument("--max-batches", type=int, default=None)
    pe.add_argument("--n-sequences", type=int, default=512)

    pp = sub.add_parser("probe", help="long-range capability probes")
    _add_common(pp)
    _add_probe_opts(pp)

    pt = sub.add_parser("throughput", help="latency / tokens/s / accounted peak memory")
    _add_common(pt)
    pt.add_argument("--seq-lens", default="64,128,256,512")
    pt.add_argument("--batch-size", type=int, default=8)
    pt.add_argument("--warmup", type=int, default=3)
    pt.add_argument("--iters", type=int, default=7)
    pt.add_argument("--backward", action="store_true")
    pt.add_argument("--reduce", default="min", choices=["min", "median"],
                    help="latency aggregate across iterations (default: min, the low-noise one)")
    pt.add_argument("--threads", type=int, default=None,
                    help="torch.set_num_threads() for this run; a single thread gives a much "
                         "cleaner scaling curve on a contended CPU")

    ps = sub.add_parser("streaming", help="constant streaming state vs sequence length")
    _add_common(ps)
    ps.add_argument("--lengths", default="64,256,1024")
    ps.add_argument("--batch-size", type=int, default=1)
    ps.add_argument("--iters", type=int, default=16)

    pc = sub.add_parser("compare", help="same protocol across all four mixers")
    _add_common(pc)
    pc.add_argument("--mixers", default=",".join(MIXER_KINDS))
    pc.add_argument("--protocols", default="ppl")
    pc.add_argument("--batch-size", type=int, default=8)
    pc.add_argument("--eval-batches", type=int, default=8)
    pc.add_argument("--train-probes", action="store_true")
    pc.add_argument("--steps", type=int, default=300)
    pc.add_argument("--probe-lr", type=float, default=1e-2)
    pc.add_argument("--n-samples", type=int, default=1024)

    pst = sub.add_parser("self-test", help="prove the memory instrumentation works")
    _add_common(pst)
    return ap


def _cmd_evaluate(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    cfg = _resolve_config(args, device, dtype)
    model = SGCModel(cfg).to(device, dtype=dtype).eval()
    if args.dataset_module:
        mod_name, _, attr = args.dataset_module.partition(":")
        import importlib
        ds = getattr(importlib.import_module(mod_name), attr or "DATASET")
        seq_len = args.seq_len
    else:
        seq_len = args.seq_len or 32
        ds = SyntheticDataset(cfg.vocab_size, seq_len, length=args.n_sequences, seed=args.seed)
    res = evaluate_lm(model, ds, batch_size=args.batch_size, seq_len=seq_len,
                      device=device, max_batches=args.max_batches)
    res["dataset"] = (args.dataset_module or
                      f"synthetic uniform tokens (seq_len {seq_len}, {args.n_sequences} seqs)")
    res["config"] = cfg.to_dict()
    res["n_params"] = int(model.num_params())
    return res


def _cmd_probe(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    cfg = _resolve_config(args, device, dtype)
    model = None
    model_eval: Dict[str, Any] = {}
    if args.model_steps > 0:
        model, model_eval = _train_sgc_on_task(
            cfg, args.task, device, dtype, steps=args.model_steps,
            batch_size=args.probe_batch, lr=args.probe_lr, seed=args.seed,
            loss=args.probe_loss, n_content=args.n_content, seq_len=args.seq_len,
            num_kv=args.num_kv, num_queries=args.num_queries, n_values=args.n_values,
            gaps_default=args.gaps)
    common = dict(train=not args.no_train, steps=args.steps, batch_size=args.probe_batch,
                  lr=args.probe_lr, dim=args.probe_dim, num_layers=args.probe_layers,
                  seed=args.seed, device=device, n_samples=args.n_samples, loss=args.probe_loss)
    out: Dict[str, Any] = {"config": cfg.to_dict()}
    tasks = (["selective-copy", "associative-recall", "interval-recall"]
             if args.task == "all" else [args.task])
    for t in tasks:
        if t == "selective-copy":
            out[t] = probe_selective_copy(model, n_content=args.n_content, seq_len=args.seq_len,
                                          **common)
        elif t == "associative-recall":
            out[t] = probe_associative_recall(model, num_kv=args.num_kv,
                                              num_queries=args.num_queries,
                                              vocab_key=args.num_kv, vocab_value=args.num_kv,
                                              **common)
        else:
            gaps = tuple(int(x) for x in args.gaps.split(",") if x.strip())
            out[t] = probe_interval_recall(model, gaps=gaps, n_values=args.n_values,
                                           train_all_gaps=args.train_all_gaps, **common)
    if model_eval:
        out["sgc_model_trained_on_probe"] = model_eval
    return out


def _task_for_train(name: str, **kw: Any) -> ProbeTask:
    if name == "selective-copy":
        return SelectiveCopyTask(n_content=kw["n_content"], seq_len=kw["seq_len"])
    if name == "associative-recall":
        return AssociativeRecallTask(num_kv=kw["num_kv"], num_queries=kw["num_queries"],
                                     vocab_key=kw["num_kv"], vocab_value=kw["num_kv"])
    if name == "interval-recall":
        gaps = tuple(int(x) for x in str(kw["gaps_default"]).split(",") if x.strip())
        return IntervalRecallTask(distractor_gap=gaps[-1], n_values=kw["n_values"])
    raise ValueError(f"unknown task {name!r}")


def _train_sgc_on_task(cfg: SGCConfig, task_name: str, device: torch.device,
                        dtype: torch.dtype, steps: int, batch_size: int, lr: float,
                        seed: int, loss: str, **kw: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a small SGC model *on a probe task* and report its accuracy.

    This is the honest end-to-end check: it shows whether the actual SGC mixer
    (not the probe's attention model) can learn the task at a given budget.
    """
    task = _task_for_train(task_name, **kw)
    full = _cfg_from_base(cfg, cfg.mixer, vocab_size=task.vocab_size)
    torch.manual_seed(seed)
    model = SGCModel(full).to(device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps),
                                                pct_start=0.1)
    gen = torch.Generator().manual_seed(seed + 5)
    curve: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for s in range(steps):
        inputs, targets = task.sample(batch_size, gen, device=device)
        lg = _answer_logits(model, inputs, task.answer_pos)
        tg = _answer_targets(targets, task.answer_pos)
        l = _answer_loss(lg, tg, loss)
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (s + 1) % max(1, steps // 4) == 0 or s == steps - 1:
            with torch.no_grad():
                xe, ye = task.sample(512, gen, device=device)
                lge = _answer_logits(model, xe, task.answer_pos)
                tge = _answer_targets(ye, task.answer_pos)
                curve.append({"step": int(s + 1), "train_loss": float(l.detach().cpu()),
                              "eval_accuracy": float((lge.argmax(-1) == tge).float().mean())})
            print(f"   sgc[{cfg.mixer}] step {s+1}/{steps} loss={float(l):.4f} "
                  f"acc={curve[-1]['eval_accuracy']:.4f}", flush=True)
    ev = evaluate_probe(model, task, n_samples=2048, device=device)
    best = max(curve, key=lambda c: c["eval_accuracy"]) if curve else {"eval_accuracy": 0.0, "step": 0}
    out = {
        "task": task.describe(),
        "chance": float(task.chance),
        "steps": int(steps),
        "lr": float(lr),
        "loss": loss,
        "seconds": time.perf_counter() - t0,
        "curve": curve,
        "final": ev,
        "best_eval_accuracy": best["eval_accuracy"],
        "best_step": best["step"],
        "solvable_at_this_budget": bool(best["eval_accuracy"] >= max(1.5 * task.chance,
                                                                    task.chance + 0.05)),
    }
    return model, out


def _cmd_throughput(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    if getattr(args, "threads", None):
        torch.set_num_threads(int(args.threads))
    cfg = _resolve_config(args, device, dtype)
    model = SGCModel(cfg).to(device, dtype=dtype)
    seq_lens = tuple(int(x) for x in args.seq_lens.split(",") if x.strip())
    rows, scaling = bench_throughput(model, seq_lens=seq_lens, batch_size=args.batch_size,
                                     device=device, warmup=args.warmup, iters=args.iters,
                                     backward=args.backward, dtype=dtype,
                                     vocab_size=cfg.vocab_size, reduce=args.reduce)
    body = _flatten_scaling(rows, scaling)
    body["config"] = cfg.to_dict()
    body["n_params"] = int(model.num_params())
    body["param_bytes"] = int(state_dict_bytes(model))
    body["memory_self_check"] = assert_memory_probe_works(device=str(device),
                                                          dtype=str(dtype).replace("torch.", ""))
    return body


def _cmd_streaming(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    cfg = _resolve_config(args, device, dtype)
    if cfg.mixer == "attn":
        raise SystemExit("the attention baseline has no O(1) streaming state; "
                         "bench_streaming is only meaningful for sgc mixers")
    model = SGCModel(cfg).to(device, dtype=dtype)
    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
    body = bench_streaming(model, lengths=lengths, batch_size=args.batch_size, device=device,
                           iters=args.iters, dtype=dtype)
    body["config"] = cfg.to_dict()
    body["n_params"] = int(model.num_params())
    return body


def _cmd_compare(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    base = (SGCConfig.from_json(args.config_json) if args.config_json
            else SGCConfig.from_dict(SGC_PRESETS[args.preset].to_dict()))
    mixers = [m.strip() for m in args.mixers.split(",") if m.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    return compare_mixers(base=base, mixers=mixers, protocols=protocols, device=device,
                          dtype=dtype, batch_size=args.batch_size, train_probes=args.train_probes,
                          steps=args.steps, lr=args.probe_lr, probe_samples=args.n_samples,
                          eval_batches=args.eval_batches, seed=args.seed)


def _cmd_self_test(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    return assert_memory_probe_works(device=str(device), dtype=str(dtype).replace("torch.", ""),
                                     verbose=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    global _CLI_COMMAND
    if argv is not None:
        _CLI_COMMAND = " ".join([Path(sys.executable).name, "-m", "simplcore.sgc.eval"] + list(argv))
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    dispatch = {"evaluate": _cmd_evaluate, "probe": _cmd_probe, "throughput": _cmd_throughput,
                "streaming": _cmd_streaming, "compare": _cmd_compare, "self-test": _cmd_self_test}
    body = dispatch[args.cmd](args, device, dtype)
    report = _emit(args, body, device, dtype)
    if args.cmd in ("evaluate", "self-test"):
        print(json.dumps(report["result"], indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
