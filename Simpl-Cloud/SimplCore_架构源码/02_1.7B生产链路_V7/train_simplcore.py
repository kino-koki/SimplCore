#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real next-token adaptation training for the migrated SimplCore V7 student.

Only the SGC/local-path parameters are trainable; teacher embeddings, norms,
SwiGLU FFN and the retained o_proj are frozen.  Before the first optimizer
step the script asserts that EVERY trainable parameter receives a non-zero
gradient -- this is the check that failed for the earlier hard-linked student
(sgc_mixer.coef existed in state_dict but was ignored at runtime).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAIN_KEYS = ("Wk", "Wv", "Wq", "Wc", "Wg", "wgate", "alpha")

# transformers >= 5 renamed torch_dtype -> dtype.  Do not let a version
# mismatch silently load the 1.7B student in fp32 (or fail outright).
_DTYPE_KW = ("dtype" if int(transformers.__version__.split(".")[0]) >= 5
             else "torch_dtype")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="student dir from migrate_qwen3.py")
    ap.add_argument("--data", required=True, help="plain-text corpus for real LM training")
    ap.add_argument("--tokenizer", default=None,
                    help="dir with the REAL Qwen3 tokenizer.json (133 B LFS pointer is not enough)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dt = torch.bfloat16 if a.bf16 else torch.float32
    tdir = a.tokenizer or a.model
    tj = os.path.join(tdir, "tokenizer.json")
    if os.path.exists(tj) and os.path.getsize(tj) < 1024:
        raise SystemExit(
            f"tokenizer.json in {tdir} is a {os.path.getsize(tj)}-byte LFS pointer. "
            "Pass --tokenizer <dir with the real tokenizer.json>.")

    print("[load] tokenizer", tdir, flush=True)
    tok = AutoTokenizer.from_pretrained(tdir, trust_remote_code=True)
    print("[load] model", a.model, dt, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, low_cpu_mem_usage=True,
        **{_DTYPE_KW: dt})
    model.train()

    trainable = []
    for n, p in model.named_parameters():
        if any(k in n for k in TRAIN_KEYS):
            p.requires_grad_(True)
            trainable.append((n, p))
        else:
            p.requires_grad_(False)
    print(f"[freeze] trainable {len(trainable)} / "
          f"{sum(1 for _ in model.named_parameters())} tensors, "
          f"{sum(p.numel() for _, p in trainable):,} params", flush=True)
    if not trainable:
        raise SystemExit("no trainable SGC parameters -- architecture not migrated")

    text = open(a.data, encoding="utf-8").read()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"[data] real text {len(text)} chars -> {len(ids)} tokens", flush=True)
    if len(ids) < a.L + 2:
        raise SystemExit("corpus too short for --L")

    opt = torch.optim.AdamW([p for _, p in trainable], lr=a.lr, weight_decay=0.0)
    rng = torch.Generator().manual_seed(a.seed + 31337)
    curve = []
    t0 = time.time()
    grad_ok = False
    for step in range(1, a.steps + 1):
        hi = len(ids) - a.L - 1
        ix = torch.randint(0, hi, (a.B,), generator=rng)
        batch = torch.stack([ids[i:i + a.L + 1] for i in ix])
        x = batch[:, :-1]
        out = model(input_ids=x, labels=x)
        loss = out.loss if hasattr(out, "loss") else out[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            bad = [n for n, p in trainable
                   if p.grad is None or float(p.grad.abs().sum()) == 0.0]
            grad_ok = len(bad) == 0
            print(f"[grad] first-step non-zero gradient on all trainable "
                  f"params: {grad_ok}" + ("" if grad_ok else f" bad={bad[:4]}"), flush=True)
            if not grad_ok:
                raise SystemExit("trainable parameters are NOT in the forward graph")
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        if step % max(1, a.steps // 8) == 0 or step == 1:
            curve.append(dict(step=step, loss=float(loss.detach()),
                              ppl=float(torch.exp(loss.detach()).clamp(max=1e6))))
            print(f"   step {step:4d} loss {curve[-1]['loss']:.4f}", flush=True)

    out_dir = a.out or a.model
    model.save_pretrained(out_dir, safe_serialization=True)
    import shutil
    shutil.copy(os.path.join(a.model, "modeling_simplcore_v7.py"),
                os.path.join(out_dir, "modeling_simplcore_v7.py"))
    report = dict(
        model=a.model, data=a.data, tokenizer=tdir, dtype=str(dt),
        steps=a.steps, L=a.L, B=a.B, lr=a.lr,
        n_trainable_tensors=len(trainable),
        n_trainable_params=int(sum(p.numel() for _, p in trainable)),
        all_trainable_in_forward_graph=bool(grad_ok),
        initial_loss=curve[0]["loss"], final_loss=curve[-1]["loss"],
        loss_drop=float(curve[0]["loss"] - curve[-1]["loss"]),
        curve=curve, elapsed_s=round(time.time() - t0, 2),
        saved_to=out_dir,
        next="python verify_arch.py --model <out> --tokenizer <real tokenizer dir>")
    json.dump(report, open(os.path.join(out_dir, "train_report.json"), "w"), indent=2)
    print(f"[done] loss {report['initial_loss']:.4f} -> {report['final_loss']:.4f} "
          f"in {report['elapsed_s']}s, saved {out_dir}", flush=True)


if __name__ == "__main__":
    main()
