# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC training loop.

Designed for 1 x A100 80GB (bf16, sequences of 4096) while remaining fully
runnable on CPU for small-scale convergence proof.

Features that matter for a *real* training run, not a demo:

* bf16 autocast on CUDA (no GradScaler needed) with fp32 parameter/optimizer state;
* TF32 matmuls on Ampere+;
* AdamW with decoupled weight decay applied only to matrices (norms, biases, the
  oscillator ``feed`` term and ``out_scale`` are excluded) -- this matters because
  weight decay on the oscillator parameters destroys long-range memory;
* warmup + cosine decay, gradient clipping, gradient accumulation;
* optional ``torch.compile`` and activation checkpointing, both fenced so a failure
  degrades to eager instead of crashing the run;
* resumable checkpoints (model + optimizer + scheduler + RNG + step);
* JSONL metrics plus a human-readable log, so curves can be replotted later;
* deterministic seeding and a token-budget accounting that is honest about how many
  tokens were actually consumed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import SGCConfig, get_preset
from .model import SGCModel


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class LMWindows(Dataset):
    """Thin adapter exposing ``(inputs, targets)`` windows for the training loop.

    The corpus schema, sharding, SHA-256 verification and cross-shard windowing all
    live in :mod:`simplcore.sgc.data` -- this class deliberately does not
    re-implement them. It asks for windows one token longer than the training
    sequence length so that shifting by one yields exactly ``seq_len`` targets.
    """

    def __init__(self, root: str, split: str, seq_len: int,
                 verify_hashes: bool = True) -> None:
        from .data import ShardedTokenDataset
        self.seq_len = int(seq_len)
        self.inner = ShardedTokenDataset(root, split, seq_len=self.seq_len + 1,
                                        verify_hashes=verify_hashes)
        self.vocab_size = int(getattr(self.inner, "vocab_size", 0) or 0)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        w = self.inner[index]
        if isinstance(w, (tuple, list)):
            w = w[0]
        w = w.long()
        return w[:-1], w[1:]


def resolve_data_dir(explicit: Optional[str]) -> Optional[str]:
    from .data import MANIFEST_NAME
    if explicit:
        return explicit
    for cand in ("data/processed/sgc_zh", "data/processed/sgc_zh_smoke"):
        if os.path.exists(os.path.join(cand, MANIFEST_NAME)):
            return cand
    return None


# ---------------------------------------------------------------------------
# optimiser parameter groups
# ---------------------------------------------------------------------------
def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """Decay matrices only.

    The oscillator parameters (``raw_rho``, ``raw_theta``), the instantaneous
    ``feed`` term, ``out_scale``, all norms and the embeddings are left undecayed:
    weight decay on a decay-rate parameter is a direct attack on long-range memory,
    which is the whole point of this architecture.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2 and not any(k in name for k in ("tok_emb", "head")):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_with_warmup(step: int, total: int, warmup: int, lr: float, lr_min: float) -> float:
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if total <= warmup:
        return lr
    t = min(max((step - warmup) / max(1, total - warmup), 0.0), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model, optim, step: int, tokens: int, cfg: SGCConfig,
                    best_val: float, elapsed: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "tokens": tokens,
        "best_val": best_val,
        "elapsed": elapsed,
        "config": cfg.to_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, optim=None, restore_rng: bool = True) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    if restore_rng:
        try:
            torch.set_rng_state(ckpt["torch_rng"])
            np.random.set_state(ckpt["numpy_rng"])
            random.setstate(ckpt["python_rng"])
        except Exception:
            pass
    return ckpt


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: Optional[int] = None) -> dict:
    """Token-weighted validation loss / perplexity."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
            else torch.autocast("cpu", enabled=False)
        with ctx:
            _, loss = model(x, y)
        n = y.numel()
        total_nll += float(loss) * n
        total_tokens += n
    model.train()
    mean = total_nll / max(total_tokens, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "tokens": total_tokens}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train an SGC language model.")
    ap.add_argument("--config", type=str, default=None,
                    help="path to a JSON config, or a preset name (e.g. sgc-150m)")
    ap.add_argument("--preset", type=str, default=None, help="preset name")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="runs/sgc")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--global-batch-tokens", type=int, default=None,
                    help="target tokens per optimiser step; sets accumulation automatically")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="soft stop after this many minutes (writes a checkpoint)")
    ap.add_argument("--dry-run", action="store_true",
                    help="ignore data, train on random tokens, for stack validation")
    ap.add_argument("--skip-hash-check", action="store_true",
                    help="skip SHA-256 verification of the corpus shards (faster startup)")
    ap.add_argument("--override", type=str, nargs="*", default=[],
                    help="config overrides, e.g. dim=512 num_layers=8")
    return ap.parse_args(argv)


def build_config(args) -> SGCConfig:
    if args.config and os.path.exists(args.config):
        cfg = SGCConfig.from_json(args.config)
    elif args.preset:
        cfg = get_preset(args.preset)
    elif args.config:
        cfg = get_preset(args.config)
    else:
        cfg = get_preset("sgc-150m")
    for item in args.override:
        if "=" not in item:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if not hasattr(cfg, k):
            raise SystemExit(f"unknown config field {k!r}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            setattr(cfg, k, v.lower() in ("1", "true", "yes"))
        elif isinstance(cur, int):
            setattr(cfg, k, int(v))
        elif isinstance(cur, float):
            setattr(cfg, k, float(v))
        elif cur is None:
            setattr(cfg, k, int(v) if v.isdigit() else v)
        else:
            setattr(cfg, k, v)
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # device / dtype
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    elif args.dtype == "fp32":
        amp_dtype = None
    else:
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.out_dir, exist_ok=True)

    # data
    if args.dry_run:
        cfg_pre = build_config(args)
        data_dir = None
        train_ds = val_ds = None
        vocab_size = cfg_pre.vocab_size
    else:
        from .data import MANIFEST_NAME, read_manifest
        data_dir = resolve_data_dir(args.data_dir)
        if data_dir is None:
            raise SystemExit("no packed dataset found; pass --data-dir or use --dry-run")
        manifest = read_manifest(os.path.join(data_dir, MANIFEST_NAME))
        vocab_size = int(manifest["vocab_size"])
        splits = manifest.get("splits") or {}
        if "train" not in splits:
            raise SystemExit(f"{data_dir} has no 'train' split; found {sorted(splits)}")
        train_ds = LMWindows(data_dir, "train", args.seq_len,
                             verify_hashes=not args.skip_hash_check)
        val_ds = LMWindows(data_dir, "val", args.seq_len,
                           verify_hashes=not args.skip_hash_check) if "val" in splits else None
        if val_ds is None:
            print("[train] WARNING: no 'val' split in the manifest -- "
                  "you will have no overfitting early-warning signal")

    cfg = build_config(args)
    cfg.vocab_size = vocab_size
    cfg.grad_checkpoint = args.grad_checkpoint

    model = SGCModel(cfg).to(device)
    n_params = model.num_params()
    amp_dtype = amp_dtype if device.type == "cuda" else None

    groups = build_param_groups(model, args.weight_decay)
    optim = torch.optim.AdamW(groups, lr=args.lr, betas=tuple(args.betas), eps=1e-8)

    if args.compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as exc:                       # pragma: no cover
            print(f"[train] torch.compile unavailable, running eager: {exc}")

    # accumulation
    if args.grad_accum is not None:
        accum = args.grad_accum
    elif args.global_batch_tokens:
        accum = max(1, round(args.global_batch_tokens / (args.micro_batch * args.seq_len)))
    else:
        accum = 1

    start_step, tokens_consumed, best_val, elapsed0 = 0, 0, float("inf"), 0.0
    if args.resume:
        ck = load_checkpoint(args.resume, model, optim)
        start_step = int(ck.get("step", 0))
        tokens_consumed = int(ck.get("tokens", 0))
        best_val = float(ck.get("best_val", float("inf")))
        elapsed0 = float(ck.get("elapsed", 0.0))
        print(f"[train] resumed from {args.resume} at step {start_step}")

    header = {
        "config": cfg.to_dict(),
        "n_params": n_params,
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "micro_batch": args.micro_batch,
        "grad_accum": accum,
        "seq_len": args.seq_len,
        "tokens_per_step": args.micro_batch * args.seq_len * accum,
        "data_dir": data_dir,
        "argv": sys.argv,
    }
    print("[train] " + json.dumps({k: v for k, v in header.items() if k != "config"}))
    print(f"[train] params={n_params:,}  config={cfg.to_dict()}")
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=2, ensure_ascii=False)
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    metrics_fh = open(metrics_path, "a", encoding="utf-8")

    # data loaders
    if train_ds is not None:
        train_loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True,
                                  pin_memory=(device.type == "cuda"),
                                  persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=args.micro_batch, shuffle=False,
                                num_workers=max(0, args.num_workers // 2),
                                drop_last=False,
                                pin_memory=(device.type == "cuda")) if val_ds else None
    else:
        train_loader = val_loader = None

    stop = {"flag": False}

    def _handle(signum, frame):
        stop["flag"] = True
        print(f"[train] signal {signum} received; will checkpoint and stop")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except Exception:
            pass

    model.train()
    t_start = time.time()
    t_log = time.perf_counter()
    step = start_step
    micro = 0
    optim.zero_grad(set_to_none=True)
    pending = 0
    running_loss = 0.0
    running_count = 0
    batch_iter: Iterator = iter(train_loader) if train_loader is not None else None

    while step < args.max_steps and not stop["flag"]:
        lr = cosine_with_warmup(step, args.max_steps, args.warmup, args.lr, args.lr_min)
        for g in optim.param_groups:
            g["lr"] = lr

        # ---- fetch a global batch (accumulated) ----------------------------
        t0 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            if train_loader is None:
                x = torch.randint(0, cfg.vocab_size, (args.micro_batch, args.seq_len),
                                  device=device)
                y = torch.randint(0, cfg.vocab_size, (args.micro_batch, args.seq_len),
                                  device=device)
            else:
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(train_loader)
                    x, y = next(batch_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
                else torch.autocast("cpu", enabled=False)
            with ctx:
                _, loss = model(x, y)
                loss = loss / accum
            loss.backward()
            loss_sum += float(loss.detach())
            pending += y.numel()
            tokens_consumed += y.numel()
            micro += 1

        if args.grad_clip and args.grad_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            gnorm = torch.tensor(0.0)
        optim.step()
        step += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        running_loss += loss_sum
        running_count += 1

        if step % args.log_every == 0 or step == 1:
            # Throughput must be measured over the *whole* logging interval.
            # Using only the last step's ``dt`` with a token count accumulated
            # across several steps inflated ``tokens_per_s`` by a factor of
            # ``log_every`` -- a number that looked 25x better than reality.
            now = time.perf_counter()
            span = max(now - t_log, 1e-9)
            tps = pending / span
            avg = running_loss / max(running_count, 1)
            rec = {"step": step, "loss": round(avg, 4), "lr": lr,
                   "grad_norm": float(gnorm), "step_s": round(dt, 3),
                   "tokens_per_s": round(tps, 1),
                   "tokens": tokens_consumed,
                   "tokens_since_log": pending,
                   "elapsed_s": round(time.time() - t_start + elapsed0, 1)}
            if device.type == "cuda":
                rec["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            print("[train] " + json.dumps(rec))
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()
            running_loss, running_count, pending = 0.0, 0, 0
            t_log = now

        # ---- evaluation -----------------------------------------------------
        if val_loader is not None and (step % args.eval_every == 0 or step == args.max_steps):
            ev = evaluate(model, val_loader, device, amp_dtype, args.eval_batches)
            rec = {"step": step, "val_loss": round(ev["loss"], 4),
                   "val_ppl": round(ev["ppl"], 4), "val_tokens": ev["tokens"],
                   "elapsed_s": round(time.time() - t_start + elapsed0, 1)}
            print("[eval] " + json.dumps(rec))
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()
            if ev["loss"] < best_val:
                best_val = ev["loss"]
                save_checkpoint(os.path.join(args.out_dir, "best.pt"), model, optim, step,
                                tokens_consumed, cfg, best_val,
                                time.time() - t_start + elapsed0)

        if step % args.save_every == 0:
            save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, optim, step,
                            tokens_consumed, cfg, best_val,
                            time.time() - t_start + elapsed0)

        if args.time_budget_min > 0 and (time.time() - t_start) / 60.0 >= args.time_budget_min:
            print(f"[train] time budget {args.time_budget_min} min reached at step {step}")
            stop["flag"] = True

    save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, optim, step,
                    tokens_consumed, cfg, best_val, time.time() - t_start + elapsed0)
    summary = {"steps": step, "tokens": tokens_consumed, "best_val_loss": best_val,
               "wall_s": round(time.time() - t_start + elapsed0, 1),
               "params": n_params, "final_lr": lr}
    print("[train] DONE " + json.dumps(summary))
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    metrics_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
