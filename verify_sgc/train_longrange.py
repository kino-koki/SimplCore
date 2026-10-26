# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""长程联想召回基准: 在 AR 任务上对比各 mixer 的召回准确率与单步耗时.

用法 (每个 mixer 单独跑, 便于后台并行/控时):
  python train_longrange.py --mixer sgc_ag --N 128 --steps 500 --bs 32 --dim 64 --layers 2

输出 results/ar_{mixer}_N{N}.json:
  {mixer, N, L, params, step_time_s, K, chance, train_acc, val_acc, curve:[...]}
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import random

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixers import LMModelV2, count_params
from longrange_task import make_batch, last_pos_accuracy

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
K = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="sgc_ag")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    L = 2 * args.N + 1
    vocab = 2 * K + 2
    rng = random.Random(args.seed)

    model = LMModelV2(vocab, args.dim, args.layers, L, args.mixer,
                      sgc_rank=8, n_heads=4, ssm_state=16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    params = count_params(model)
    step_times = []
    curve = []          # (step, train_acc, val_acc)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        inp, tgt, tv = make_batch(rng, args.N, args.bs, K)
        ts = time.time()
        model.train()
        opt.zero_grad()
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
        loss.backward()
        opt.step()
        step_times.append(time.time() - ts)

        if step % args.eval_every == 0 or step == 1:
            with torch.no_grad():
                train_acc = last_pos_accuracy(logits.detach(), tv, L)
                # 验证集: 固定随机种子生成一批新样本
                vrng = random.Random(args.seed + 999 + step)
                vinp, vtgt, vtv = make_batch(vrng, args.N, 64, K)
                vlogits = model(vinp)
                val_acc = last_pos_accuracy(vlogits.detach(), vtv, L)
            curve.append([step, round(train_acc, 4), round(val_acc, 4)])
            print(f"[{args.mixer} N={args.N}] step {step:4d} "
                  f"loss={loss.item():.3f} train_acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f} step={step_times[-1]*1000:.0f}ms")

    out = {
        "mixer": args.mixer, "N": args.N, "L": L, "vocab": vocab,
        "params": params, "K": K,
        "chance": 1.0 / K,
        "step_time_s": sum(step_times) / len(step_times),
        "total_time_s": time.time() - t0,
        "train_acc": curve[-1][1], "val_acc": curve[-1][2],
        "curve": curve,
    }
    path = os.path.join(RESULTS, f"ar_{args.mixer}_N{args.N}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
