# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC-v2 vs Attention 公平对比训练 (PyTorch CPU 后端).

实验设计要点 (可发表诚信):
  - 文档级无泄漏切分 (corpus.json 已保证 val/test 不跨集泄漏);
  - 随机 batch 采样 (每步从不同文档窗口取样, 非固定前 bs 个);
  - 同时报告 train / val / test 困惑度与 train-val gap, 区分"真泛化"与"背答案";
  - 模型规模受限于本地 CPU + 小语料, 论文中如实标注, 不做夸大.
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_pytorch import LMModel, count_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORPUS = os.path.join(RESULTS, "corpus.json")


def load_corpus():
    d = json.load(open(CORPUS))
    return d["vocab"], d["train"], d["val"], d.get("test", d["val"])


def batchify(samples, bs, seq_len, device, rng=None):
    if rng is not None:
        idx = rng.sample(range(len(samples)), min(bs, len(samples)))
    else:
        idx = list(range(min(bs, len(samples))))
    xb = torch.zeros(bs, seq_len, dtype=torch.long)
    yb = torch.zeros(bs, seq_len, dtype=torch.long)
    for i, j in enumerate(idx):
        s = samples[j]
        xb[i] = torch.tensor(s[0][:seq_len])
        yb[i] = torch.tensor(s[1][:seq_len])
    return xb.to(device), yb.to(device)


@torch.no_grad()
def eval_ppl(model, samples, vocab, seq_len, device, max_batches=200):
    """在全部 (或前 max_batches 个) 窗口上算困惑度."""
    model.eval()
    total_loss = 0.0
    n_tok = 0
    bs = 32
    n = len(samples)
    steps = min(max_batches, (n + bs - 1) // bs)
    for bi in range(steps):
        chunk = samples[bi * bs:(bi + 1) * bs]
        xb = torch.zeros(len(chunk), seq_len, dtype=torch.long)
        yb = torch.zeros(len(chunk), seq_len, dtype=torch.long)
        for i, s in enumerate(chunk):
            xb[i] = torch.tensor(s[0][:seq_len])
            yb[i] = torch.tensor(s[1][:seq_len])
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), yb.reshape(-1),
                               reduction="sum")
        total_loss += float(loss.item())
        n_tok += yb.numel()
    return total_loss / max(n_tok, 1)


def run(mixer, vocab, train, val, test, args, device):
    torch.manual_seed(1234)
    model = LMModel(len(vocab), args.dim, args.layers, args.heads,
                    args.seq_len, mixer, sgc_rank=args.rank).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    rng = random.Random(1234)
    # warmup 1 步测单步耗时
    xb, yb = batchify(train, args.batch, args.seq_len, device, rng)
    model.train()
    for _ in range(1):
        opt.zero_grad(); l = model(xb)
        loss = F.cross_entropy(l.reshape(-1, len(vocab)), yb.reshape(-1))
        loss.backward(); opt.step()
    t0 = time.perf_counter()
    train_losses = []
    for step in range(args.steps):
        xb, yb = batchify(train, args.batch, args.seq_len, device, rng)
        opt.zero_grad(); l = model(xb)
        loss = F.cross_entropy(l.reshape(-1, len(vocab)), yb.reshape(-1))
        loss.backward(); opt.step()
        if (step + 1) % max(1, args.steps // 5) == 0:
            train_losses.append(float(loss.item()))
    step_time = (time.perf_counter() - t0) / args.steps

    train_ppl = eval_ppl(model, train, vocab, args.seq_len, device)
    val_ppl = eval_ppl(model, val, vocab, args.seq_len, device)
    test_ppl = eval_ppl(model, test, vocab, args.seq_len, device)
    return {"mixer": mixer, "params": n_params,
            "train_ppl": train_ppl, "val_ppl": val_ppl, "test_ppl": test_ppl,
            "step_time_s": step_time, "steps": args.steps,
            "train_losses": train_losses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    args = ap.parse_args()

    device = "cpu"
    vocab, train, val, test = load_corpus()
    print(f"vocab={len(vocab)} train={len(train)} val={len(val)} test={len(test)} device={device}")

    results = {}
    for mixer in ["attn", "sgc"]:
        r = run(mixer, vocab, train, val, test, args, device)
        results[mixer] = r
        print(f"[{mixer}] params={r['params']:,} train_ppl={r['train_ppl']:.2f} "
              f"val_ppl={r['val_ppl']:.2f} test_ppl={r['test_ppl']:.2f} step={r['step_time_s']:.4f}s")

    json.dump({**results, "args": vars(args)},
              open(os.path.join(RESULTS, "pytorch_compare.json"), "w"), indent=2)
    a, s = results["attn"], results["sgc"]
    print("\n=== 摘要 (SGC vs Attn) ===")
    print(f"参数:  SGC {s['params']:,} / Attn {a['params']:,}  (Δ={s['params']-a['params']:+,})")
    print(f"train_ppl: SGC {s['train_ppl']:.2f} / Attn {a['train_ppl']:.2f}")
    print(f"val_ppl:   SGC {s['val_ppl']:.2f} / Attn {a['val_ppl']:.2f}")
    print(f"test_ppl:  SGC {s['test_ppl']:.2f} / Attn {a['test_ppl']:.2f}")
    print(f"gap(val-train): SGC {s['val_ppl']-s['train_ppl']:+.2f} / Attn {a['val_ppl']-a['train_ppl']:+.2f}")
    print(f"单步:  SGC {s['step_time_s']:.4f}s / Attn {a['step_time_s']:.4f}s "
          f"(speedup {(a['step_time_s']/s['step_time_s']-1)*100:+.0f}%)")


if __name__ == "__main__":
    main()
