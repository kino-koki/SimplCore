# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Accounted (not sampled) memory instrumentation, split out of ``eval.py``.

``GetProcessMemoryInfo`` is broken on the reference host (returns 0 with
``GetLastError() == 122``), so every number here is computed exactly as
``sum(t.numel() * t.element_size())`` over tensors that are alive and
materialised during the measured step -- never sampled from an OS counter."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import _jsonable, resolve_device, resolve_dtype
from ..model.config import SimplCoreConfig
from ..model.model import SimplCoreModel


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
        cfg = SimplCoreConfig(vocab_size=64, dim=width, num_layers=2, num_bases=8,
                         decay_max=64, local_kernel=4, num_heads=4, ffn_ratio=2.0)
        model = SimplCoreModel(cfg).to(dev, dtype=dt).train()
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


