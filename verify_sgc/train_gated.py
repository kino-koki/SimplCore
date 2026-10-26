# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC-v2 门控消融实验 (PyTorch CPU 后端) —— 论文 4.4 节.

目的: 验证引理3预言的"内容相关路由缺失"能否用轻量门控补偿.
对比 mixer:
  - attn      : 因果注意力 baseline
  - sgc       : 纯 SGC-v2 (无门控)
  - gated_sgc : SGC-v2 + 内容门控 (g_t = sigmoid(W_g x_t))
  - sgc_nogate: 门控模块但 use_gate=False (等价于纯 SGC, 控制变量)

复用 train_compare.py 的语料加载 (corpus.json 文档级无泄漏) 与评估逻辑,
保证与既有实验可比.
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
from sgc_gated import GatedLMModel, count_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORPUS = os.path.join(RESULTS, "corpus.json")

# 复用 train_compare 的 load_corpus / batchify / eval_ppl
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "train_compare_mod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_compare.py")
)
_tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tc)

load_corpus = _tc.load_corpus
batchify = _tc.batchify
eval_ppl = _tc.eval_ppl


def run(mixer, vocab, train, val, test, args, device):
    torch.manual_seed(1234)
    # 统一用 GatedLMModel 骨架; mixer 决定门控开关
    model = GatedLMModel(len(vocab), args.dim, args.layers, args.heads,
                         args.seq_len, mixer, sgc_rank=args.rank).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    rng = random.Random(1234)
    # warmup 1 步
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
    ap.add_argument("--mixers", type=str, default="attn,sgc,gated_sgc,sgc_nogate")
    args = ap.parse_args()

    device = "cpu"
    vocab, train, val, test = load_corpus()
    print(f"vocab={len(vocab)} train={len(train)} val={len(val)} test={len(test)} device={device}")

    mixers = [m.strip() for m in args.mixers.split(",") if m.strip()]
    results = {}
    for mixer in mixers:
        r = run(mixer, vocab, train, val, test, args, device)
        results[mixer] = r
        print(f"[{mixer}] params={r['params']:,} train_ppl={r['train_ppl']:.2f} "
              f"val_ppl={r['val_ppl']:.2f} test_ppl={r['test_ppl']:.2f} step={r['step_time_s']:.4f}s")

    json.dump({**results, "args": vars(args)},
              open(os.path.join(RESULTS, "gated_compare.json"), "w"), indent=2)

    print("\n=== 门控消融摘要 ===")
    a = results.get("attn")
    s = results.get("sgc")
    g = results.get("gated_sgc")
    if a and s:
        print(f"Attn  vs SGC   : test_ppl {a['test_ppl']:.2f} vs {s['test_ppl']:.2f}")
    if s and g:
        print(f"SGC   vs Gated : test_ppl {s['test_ppl']:.2f} vs {g['test_ppl']:.2f}")
        print(f"  test_ppl 变化: {g['test_ppl']-s['test_ppl']:+.2f}  "
              f"({'门控缓解过拟合' if g['test_ppl'] < s['test_ppl'] else '门控无改善'})")
        print(f"  val-train gap: SGC {s['val_ppl']-s['train_ppl']:+.2f} vs Gated {g['val_ppl']-g['train_ppl']:+.2f}")


if __name__ == "__main__":
    main()
