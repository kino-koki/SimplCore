#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained REAL-corpus training for the shipped SimplCore model.

No cloud repo, no torch, no network.  Everything resolves inside this package:

    python train.py --eval-only                 # evaluate shipped model.npz
    python train.py --steps 20 --smoke         # speed + gradcheck
    python train.py --steps 600 --resume       # continue from the 4200-step ckpt
    python train.py --steps 600 --fresh        # train a new model from scratch

Corpus: 训练/tinyshakespeare.txt (1,115,394 B), chronological 90/10 split,
plus Q/A-formatted dialogue extracted from the script (TRAIN ONLY).  No synthetic
key/value task, no oracle injection, no teacher weights, no renamed checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nn                       # noqa: E402
import simplcore_lm as arch     # noqa: E402

TRAIN_DIR = os.path.join(HERE, "训练")
CORPUS = os.path.join(TRAIN_DIR, "tinyshakespeare.txt")
CKPT = os.path.join(TRAIN_DIR, "step7_ckpt.npz")
SHIPPED = os.path.join(HERE, "model.npz")
os.makedirs(TRAIN_DIR, exist_ok=True)


def build_corpus():
    raw = open(CORPUS, encoding="utf-8").read()
    lines = [l.strip() for l in raw.splitlines()]
    qa = []
    for i, l in enumerate(lines[:-1]):
        if "?" in l and 8 < len(l) < 120 and lines[i + 1] and len(lines[i + 1]) < 160:
            qa.append("Q: %s\nA: %s\n\n" % (l, lines[i + 1]))
    qa_text = "".join(qa)
    cut = int(0.9 * len(raw))
    train_text = raw[:cut] + "\n" + qa_text + "\n" + qa_text
    val_text = raw[cut:]
    chars = sorted(set(train_text + val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    tr = np.array([stoi[c] for c in train_text], dtype=np.int64)
    va = np.array([stoi[c] for c in val_text], dtype=np.int64)
    return tr, va, len(chars), stoi, len(qa)


def batch(data, B, L, rng):
    hi = len(data) - L - 1
    idx = rng.integers(0, hi, size=B)
    X = np.stack([data[i:i + L + 1] for i in idx])
    return X[:, :-1], X[:, 1:]


def eval_bpc(model, val, B, L, seed, n_batch=12):
    rng = np.random.default_rng(seed + 991)
    vals = []
    nn.set_no_grad(True)
    for _ in range(n_batch):
        tok, tgt = batch(val, B, L, rng)
        lg = model.logits(tok)
        z = lg - lg.max(-1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        pt = np.take_along_axis(p, tgt[..., None], -1)[..., 0]
        vals.append(float((-np.log(np.maximum(pt, 1e-12)) / math.log(2.0)).mean()))
    nn.set_no_grad(False)
    return float(np.mean(vals)), float(np.std(vals))


def gradcheck(model, tok, tgt, seed=0, n_params=14):
    rng = np.random.default_rng(seed + 7)
    nn.set_no_grad(False)
    loss = model.loss(tok, tgt)
    loss.backward()
    ps = model.params()
    names = list(ps)
    rng.shuffle(names)
    out = []
    for name in names[:n_params]:
        p = ps[name]
        j = int(rng.integers(p.size))
        ana = 0.0 if p.grad is None else float(p.grad.ravel()[j])
        old = float(p.data.ravel()[j])
        eps = 1e-5
        p.data.ravel()[j] = old + eps
        lp = model.loss(tok, tgt).data.item()
        nn.clear_tape()
        p.data.ravel()[j] = old - eps
        lm = model.loss(tok, tgt).data.item()
        nn.clear_tape()
        p.data.ravel()[j] = old
        num = (lp - lm) / (2 * eps)
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-9)
        out.append(dict(param=name, analytic=ana, numeric=num, rel_err=rel))
    worst = max(o["rel_err"] for o in out)
    return dict(n=len(out), worst_rel_err=worst, passed=bool(worst < 2e-4),
                details=out)


def save(model, st, step, cfg):
    d = {}
    for n, p in model.params().items():
        d["p:" + n] = p.data
        d["m:" + n] = st[n][0]
        d["v:" + n] = st[n][1]
    d["_step"] = np.array(step, dtype=np.int64)
    d["_cfg"] = json.dumps(cfg)
    np.savez_compressed(CKPT, **d)
    # keep the inference weight file in sync (p: keys only + metadata)
    e = {k: v for k, v in d.items() if k.startswith("p:")}
    e["_step"] = d["_step"]
    e["_cfg"] = d["_cfg"]
    e["_arch_step"] = np.array(arch.ARCH_STEP, dtype=np.int64)
    np.savez_compressed(SHIPPED, **e)


def load(model, cfg, resume, fresh):
    if resume and not fresh and os.path.exists(CKPT):
        z = np.load(CKPT)
        ps = model.params()
        for n, p in ps.items():
            np.copyto(p.data, z["p:" + n])
        st = {n: (z["m:" + n].copy(), z["v:" + n].copy()) for n in ps}
        step = int(z["_step"])
        return st, step, np.random.default_rng(cfg["seed"] + 31337 + step)
    if resume and not fresh and os.path.exists(SHIPPED):
        z = np.load(SHIPPED)
        for n, p in model.params().items():
            np.copyto(p.data, z["p:" + n])
        step = int(z["_step"]) if "_step" in z else 0
        return ({n: (np.zeros_like(p.data), np.zeros_like(p.data))
                 for n, p in model.params().items()},
                step, np.random.default_rng(cfg["seed"] + 31337 + step))
    return ({n: (np.zeros_like(p.data), np.zeros_like(p.data))
             for n, p in model.params().items()},
            0, np.random.default_rng(cfg["seed"] + 31337))


def adam(model, st, step, lr0, total):
    lr = lr0 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total)))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for n, p in model.params().items():
        m, v = st[n]
        g = np.zeros_like(p.data) if p.grad is None else p.grad
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        mh = m / (1 - b1 ** step)
        vh = v / (1 - b2 ** step)
        p.data -= lr * mh / (np.sqrt(vh) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--total", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--log-every", type=int, default=200)
    a = ap.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    train, val, V, stoi, n_qa = build_corpus()
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    assert cfg["V"] == V, "vocab mismatch: shipped vocab.json does not match corpus"
    model = arch.SimplCoreLM(V, d=cfg["d"], n_layer=cfg["layers"], W=cfg["W"],
                             rhos=tuple(cfg["rhos"]), lam=cfg["lam"],
                             maxL=cfg["maxL"], seed=cfg["seed"])
    nparam = sum(p.size for p in model.params().values())
    print("corpus train=%d val=%d V=%d qa=%d params=%d arch_step=%d" %
          (len(train), len(val), V, n_qa, nparam, arch.ARCH_STEP), flush=True)

    rng = np.random.default_rng(a.seed + 31337)
    tok, tgt = batch(train, a.batch, a.seq, rng)
    gc = gradcheck(model, tok, tgt, seed=a.seed)
    print("GRADCHECK worst=%.3e %s" % (gc["worst_rel_err"],
                                       "PASS" if gc["passed"] else "FAIL"), flush=True)
    if not gc["passed"]:
        raise SystemExit(2)

    if a.eval_only:
        z = np.load(SHIPPED)
        for n, p in model.params().items():
            np.copyto(p.data, z["p:" + n])
        b, s = eval_bpc(model, val, a.batch, a.seq, a.seed)
        print("SHIPPED model.npz step=%d val_bpc=%.6f std=%.6f" %
              (int(z["_step"]), b, s), flush=True)
        return

    if a.smoke:
        st = {n: (np.zeros_like(p.data), np.zeros_like(p.data))
              for n, p in model.params().items()}
        t0 = time.time()
        for i in range(1, 21):
            tk, tg = batch(train, a.batch, a.seq, rng)
            ls = model.loss(tk, tg)
            ls.backward()
            adam(model, st, i, a.lr, a.total)
            for p in model.params().values():
                p.zero_grad()
            nn.clear_tape()
        dt = time.time() - t0
        b, _ = eval_bpc(model, val, a.batch, a.seq, a.seed, n_batch=3)
        print("SMOKE 20 steps %.2fs val_bpc=%.4f" % (dt, b), flush=True)
        return

    st, step0, rng = load(model, cfg, a.resume or not a.fresh, a.fresh)
    b0, _ = eval_bpc(model, val, a.batch, a.seq, a.seed)
    print("resume step %d  start val_bpc=%.6f" % (step0, b0), flush=True)
    hist = []
    t0 = time.time()
    for i in range(1, a.steps + 1):
        step = step0 + i
        tk, tg = batch(train, a.batch, a.seq, rng)
        for p in model.params().values():
            p.zero_grad()
        nn.clear_tape()
        ls = model.loss(tk, tg)
        ls.backward()
        adam(model, st, step, a.lr, a.total)
        if i % max(1, a.log_every) == 0 or i == a.steps:
            b, s = eval_bpc(model, val, a.batch, a.seq, a.seed, n_batch=6)
            hist.append(dict(step=step, train_loss=float(ls.data),
                             val_bpc=b, val_std=s, sec=round(time.time() - t0, 1)))
            print("step %5d train %.4f val_bpc %.6f %.1fs" %
                  (step, ls.data, b, time.time() - t0), flush=True)
            save(model, st, step, cfg)
    save(model, st, step0 + a.steps, cfg)
    out = dict(arch_step=arch.ARCH_STEP, gradcheck_worst=gc["worst_rel_err"],
               start_step=step0, end_step=step0 + a.steps, params=nparam,
               start_val_bpc=b0, end_val_bpc=hist[-1]["val_bpc"] if hist else b0,
               history=hist, wall_sec=round(time.time() - t0, 2))
    json.dump(out, open(os.path.join(TRAIN_DIR, "train_last_custom.json"), "w"),
              indent=1, ensure_ascii=False)
    print("SAVED model.npz step=%d end_val_bpc=%.6f" %
          (step0 + a.steps, out["end_val_bpc"]), flush=True)


if __name__ == "__main__":
    main()
