# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Evaluation harness for the SimplCore package (`simplcore.core`).

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

    python -m simplcore.core.eval evaluate   --preset debug --dataset synthetic --out r.json
    python -m simplcore.core.eval probe      --task selective-copy --steps 400 --out p.json
    python -m simplcore.core.eval throughput --preset debug --seq-lens 64,128,256,512 --backward
    python -m simplcore.core.eval streaming  --preset debug --lengths 64,256,1024
    python -m simplcore.core.eval compare    --preset debug --tasks ppl,selective-copy
    python -m simplcore.core.eval self-test
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

from ..model.config import SimplCoreConfig, SIMPLCORE_PRESETS, get_preset
from ..model.model import SimplCoreModel

# ---------------------------------------------------------------------------
# 切分后的向后兼容再导出（原单文件 eval.py 的公共与内部名称全部保留：
# ``from simplcore.core.eval import X`` 与 ``from . import eval as _eval`` 的
# 既有调用方——tests/test_eval.py、extrapolation.py——无需任何改动）
# ---------------------------------------------------------------------------
from ..common import (EvalBatch, SyntheticDataset, _as_eval_batch, _jsonable,
                     _next_token_targets, _now, _stack_batch,
                     default_config_kwargs, git_commit, git_dirty, resolve_device,
                     resolve_dtype, sha256_file)
from .memprobe import (_TensorAudit, _bytes_of, _discover_tensors, _register_audit,
                       _smoke_stub_model, assert_memory_probe_works,
                       live_tensor_bytes, mb, state_dict_bytes, tensor_peak_bytes)
from .probes import (AssociativeRecallTask, IntervalRecallTask, ProbeNet, ProbeTask,
                     SelectiveCopyTask, _CausalDWConv, _ProbeBlock, _ProbeMixer,
                     _finish, _make_probe_tasks, _probe_report, evaluate_probe,
                     probe_associative_recall, probe_eval_dataset,
                     probe_interval_recall, probe_selective_copy, probe_shift_batches,
                     probe_tasks, train_probe)
from .bench import (MIXER_KINDS, _cfg_from_base, _fit_quality, _flatten_scaling,
                    _measure_phase, _measure_training_step, _median,
                    _reduce_latency, _slope_loglog, _sync, bench_streaming,
                    bench_throughput, compare_mixers,
                    compare_snn_sgc_training)

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
    "compare_snn_sgc_training",
    "tensor_peak_bytes",
    "assert_memory_probe_works",
    "make_report",
    "main",
]

_ORIGINAL_ARGV = (" ".join([Path(sys.executable).name] + sys.argv) if sys.argv
                  else "")
# kept for backwards compatibility with callers that set it explicitly
_CLI_COMMAND = _ORIGINAL_ARGV

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
        "schema": "simplcore.core.eval/1",
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
def _resolve_config(args: argparse.Namespace, device: Any, dtype: Any) -> SimplCoreConfig:
    if getattr(args, "config_json", None):
        cfg = SimplCoreConfig.from_json(args.config_json)
    else:
        cfg = SimplCoreConfig.from_dict(SIMPLCORE_PRESETS[getattr(args, "preset", "debug")].to_dict())
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
        cfg = SimplCoreConfig.from_dict(d)
    cfg.validate()
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="debug", choices=sorted(SIMPLCORE_PRESETS),
                   help="SGC preset to start from (default: debug)")
    p.add_argument("--config-json", default=None, help="SimplCoreConfig JSON file (overrides --preset)")
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
        prog="python -m simplcore.core.eval",
        description="Evaluation harness for simplcore.core (accounted memory, fitted "
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
    model = SimplCoreModel(cfg).to(device, dtype=dtype).eval()
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


def _train_sgc_on_task(cfg: SimplCoreConfig, task_name: str, device: torch.device,
                        dtype: torch.dtype, steps: int, batch_size: int, lr: float,
                        seed: int, loss: str, **kw: Any) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a small SimplCore model *on a probe task* and report its accuracy.

    This is the honest end-to-end check: it shows whether the actual SGC mixer
    (not the probe's attention model) can learn the task at a given budget.
    """
    task = _task_for_train(task_name, **kw)
    full = _cfg_from_base(cfg, cfg.mixer, vocab_size=task.vocab_size)
    torch.manual_seed(seed)
    model = SimplCoreModel(full).to(device, dtype=dtype)
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
    model = SimplCoreModel(cfg).to(device, dtype=dtype)
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
    if cfg.mixer not in ("sgc", "sgc-nogate", "sgc-nolocal", "srf"):
        raise SystemExit("bench_streaming needs an O(1) streaming mixer "
                         "(sgc / sgc-nogate / sgc-nolocal / srf)")
    model = SimplCoreModel(cfg).to(device, dtype=dtype)
    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())
    body = bench_streaming(model, lengths=lengths, batch_size=args.batch_size, device=device,
                           iters=args.iters, dtype=dtype)
    body["config"] = cfg.to_dict()
    body["n_params"] = int(model.num_params())
    return body


def _cmd_compare(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    base = (SimplCoreConfig.from_json(args.config_json) if args.config_json
            else SimplCoreConfig.from_dict(SIMPLCORE_PRESETS[args.preset].to_dict()))
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
        _CLI_COMMAND = " ".join([Path(sys.executable).name, "-m", "simplcore.core.eval"] + list(argv))
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
