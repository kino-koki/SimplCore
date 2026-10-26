# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Find the largest micro-batch that fits, and measure real step throughput.

Run this FIRST on the target machine. Training-config arithmetic that assumes a
memory budget without measuring it is how runs die at step 3000. This script:

  * binary-searches the largest micro-batch that fits at a given sequence length;
  * reports measured peak memory (``torch.cuda.max_memory_allocated``) -- a real
    number from the allocator, not an RSS guess;
  * measures forward and forward+backward time and tokens/s;
  * prints the gradient-accumulation factor needed to reach a token budget.

Example::

    python -m scripts.probe_batch --config configs/sgc-1_3b.json --seq-len 4096
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from simplcore.sgc.config import SGCConfig  # noqa: E402
from simplcore.sgc.model import SGCModel  # noqa: E402


def attempt(cfg: SGCConfig, seq_len: int, batch: int, device, amp_dtype,
            backward: bool = True, grad_checkpoint: bool = False) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    cfg.grad_checkpoint = grad_checkpoint
    model = SGCModel(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None \
        else torch.autocast("cpu", enabled=False)
    result = {"batch": batch, "seq_len": seq_len, "ok": False}
    try:
        # warmup
        for _ in range(2):
            with ctx:
                _, loss = model(x, y)
            if backward:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            with ctx:
                _, loss = model(x, y)
            if backward:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        result.update(ok=True, step_s=dt,
                      tokens_per_s=batch * seq_len / dt,
                      loss=float(loss.detach()))
        if device.type == "cuda":
            result["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        result["error"] = "OOM"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result["error"] = "OOM"
        else:
            raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--token-budget", type=int, default=2_000_000,
                    help="tokens per optimiser step, used to suggest grad accumulation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = SGCConfig.from_json(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_dtype = None if args.dtype == "fp32" or device.type == "cpu" else \
        {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    print(f"device={device} amp={amp_dtype} seq_len={args.seq_len} "
          f"vocab={cfg.vocab_size} grad_ckpt={args.grad_checkpoint}")

    # exponential search for the first batch that fails
    hi, last_ok = 1, 0
    first_fail = None
    while hi <= args.max_batch:
        r = attempt(cfg, args.seq_len, hi, device, amp_dtype, not args.forward_only,
                    args.grad_checkpoint)
        print(f"  batch={hi:4d} -> {'OK' if r['ok'] else r.get('error')}"
              + (f"  step={r['step_s']*1000:.1f} ms  {r['tokens_per_s']:.0f} tok/s"
                 f"  peak={r.get('peak_mem_gb', float('nan')):.2f} GB" if r["ok"] else ""))
        if r["ok"]:
            last_ok = hi
            hi *= 2
        else:
            first_fail = hi
            break

    if first_fail is not None:
        lo, hi2 = last_ok, first_fail
        while hi2 - lo > 1:
            mid = (lo + hi2) // 2
            r = attempt(cfg, args.seq_len, mid, device, amp_dtype, not args.forward_only,
                        args.grad_checkpoint)
            if r["ok"]:
                lo = mid
                print(f"  batch={mid:4d} -> OK  {r['tokens_per_s']:.0f} tok/s "
                      f"peak={r.get('peak_mem_gb', float('nan')):.2f} GB")
            else:
                hi2 = mid
                print(f"  batch={mid:4d} -> OOM")
        best = lo
    else:
        best = last_ok

    if best == 0:
        print("RESULT: no batch size fits; try --grad-checkpoint or a shorter --seq-len")
        return 1

    final = attempt(cfg, args.seq_len, best, device, amp_dtype, not args.forward_only,
                    args.grad_checkpoint)
    accum = max(1, round(args.token_budget / max(best * args.seq_len, 1)))
    summary = {
        "config": args.config, "device": str(device), "seq_len": args.seq_len,
        "max_micro_batch": best, "meas": final,
        "suggested_grad_accum_for_tokens_per_step": accum,
        "tokens_per_step": best * args.seq_len * accum,
        "grad_checkpoint": args.grad_checkpoint,
        "backward": not args.forward_only,
    }
    print("\nRESULT " + json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("saved ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
