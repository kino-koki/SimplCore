# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""最小训练诊断: 定位损失上升根因.

修复: (1) 每个 step 随机采样 batch (而非固定前8);
      (2) 试不同 lr / weight_decay;
      (3) 打印每步 train_loss, 确认是否下降.
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_pytorch import LMModel, count_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORPUS = os.path.join(RESULTS, "corpus.json")


def random_batch(samples, bs, seq_len):
    idx = random.sample(range(len(samples)), min(bs, len(samples)))
    xb = torch.zeros(bs, seq_len, dtype=torch.long)
    yb = torch.zeros(bs, seq_len, dtype=torch.long)
    for i, j in enumerate(idx):
        s = samples[j]
        xb[i] = torch.tensor(s[:seq_len])
        yb[i] = torch.tensor(s[1:seq_len + 1])
    return xb, yb


def run(mixer, samples, val, vocab, steps, lr, wd, dim=128, layers=3, heads=4, seq_len=128, rank=8):
    torch.manual_seed(1234)
    random.seed(1234)
    model = LMModel(vocab, dim, layers, heads, seq_len, mixer, sgc_rank=rank)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    vx, vy = random_batch(val, 8, seq_len)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        xb, yb = random_batch(samples, 16, seq_len)
        opt.zero_grad()
        l = model(xb)
        loss = F.cross_entropy(l.reshape(-1, vocab), yb.reshape(-1))
        loss.backward()
        opt.step()
        if (step + 1) % max(1, steps // 10) == 0:
            losses.append(float(loss.item()))
    dt = (time.perf_counter() - t0) / steps
    model.eval()
    with torch.no_grad():
        vl = model(vx)
        vloss = F.cross_entropy(vl.reshape(-1, vocab), vy.reshape(-1)).item()
    ppl = float(torch.exp(torch.tensor(vloss)))
    return {"mixer": mixer, "train_losses": losses, "final_train": losses[-1],
            "val_loss": vloss, "val_ppl": ppl, "step_time_s": dt}


def main():
    d = json.load(open(CORPUS))
    vocab, train, val = d["vocab"], d["train"], d["val"]
    print(f"vocab={len(vocab)} train={len(train)} val={len(val)}")
    configs = [
        ("attn", 3e-4, 0.1),
        ("attn", 1e-3, 0.0),
        ("attn", 5e-4, 0.01),
        ("sgc", 3e-4, 0.1),
        ("sgc", 1e-3, 0.0),
        ("sgc", 5e-4, 0.01),
    ]
    out = {}
    for mixer, lr, wd in configs:
        r = run(mixer, train, val, len(vocab), 500, lr, wd)
        key = f"{mixer}_lr{lr}_wd{wd}"
        out[key] = r
        print(f"[{key}] train={r['train_losses']}")
        print(f"         val_loss={r['val_loss']:.3f} ppl={r['val_ppl']:.1f} step={r['step_time_s']:.3f}s")
    json.dump(out, open(os.path.join(RESULTS, "train_diag.json"), "w"), indent=2)
    print("saved -> train_diag.json")


if __name__ == "__main__":
    main()
