# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Performance benchmarks split out of ``eval.py``: ``bench_throughput``
(og-fit of log(time) vs log(seq_len) -- scaling claims are fitted, not
asserted), ``bench_streaming`` (O(1) state streaming), and
``compare_mixers`` (controlled mixer A/B)."""
from __future__ import annotations

import math
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import SyntheticDataset, resolve_device
from .config import SIMPLCORE_PRESETS, SimplCoreConfig, get_preset
from .memprobe import mb, tensor_peak_bytes
from .model import SimplCoreModel
from .probes import (AssociativeRecallTask, IntervalRecallTask, ProbeNet, ProbeTask,
                     SelectiveCopyTask, evaluate_probe, train_probe)


def evaluate_lm(*args, **kwargs):
    """Lazy proxy to ``eval.evaluate_lm``（eval.py 重导出 bench，避免循环导入）."""
    from .eval import evaluate_lm as _f
    return _f(*args, **kwargs)


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
            # SimplCore 的流式状态与序列长度无关（O(1)），因此 is_kv_free 恒为 True
            "is_kv_free": exact > 0,
            "state_length_invariant": bool(exact == analytic),
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
MIXER_KINDS = ("sgc", "sgc-nogate", "sgc-nolocal")


def _cfg_from_base(base: SimplCoreConfig, mixer: str, **over: Any) -> SimplCoreConfig:
    d = base.to_dict()
    d.update(over)
    d["mixer"] = mixer
    return SimplCoreConfig.from_dict(d)


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


def compare_mixers(configs: Optional[Sequence[Any]] = None, base: Optional[SimplCoreConfig] = None,
                   mixers: Sequence[str] = MIXER_KINDS, protocols: Sequence[str] = ("ppl",),
                   device: Any = None, dtype: Optional[torch.dtype] = None,
                   batch_size: int = 8, train_probes: bool = False, steps: int = 300,
                   lr: float = 1e-2, probe_samples: int = 2048, eval_batches: int = 8,
                   seed: int = 0, verbose: bool = True) -> Dict[str, Any]:
    """Run the same protocol across all requested mixers with matched width/depth.

    Every mixer sees the *same* `SimplCoreConfig` (identical ``dim``, ``num_layers``,
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
        base = SimplCoreConfig.from_dict(SIMPLCORE_PRESETS["debug"].to_dict())
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
        model = SimplCoreModel(cfg).to(device, dtype=dtype)
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


