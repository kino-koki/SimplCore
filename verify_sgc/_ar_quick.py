# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""精简 AR 对照: 证明 AR 任务注意力可学、O(1) 记忆的 CA/门控不可学 (代价实证)."""
from __future__ import annotations
import os, sys, time, json, random
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_v3 import build_model
from longrange_task import make_batch, last_pos_accuracy

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
K = 256
VOCAB = 2 * K + 2


def run(mixer, N, steps=200, dim=64, layers=2, bs=32, seed=1234, feat=32, chunk=0):
    L = 2 * N + 1
    maxlen = max(L, 2 * 128 + 1)
    kw = {}
    if mixer == "ca":
        kw = dict(ca_feat=feat, ca_chunk=chunk)
    m = build_model(VOCAB, dim, layers, 4, maxlen, mixer, **kw).eval()
    opt = AdamW(m.parameters(), lr=1e-3)
    rng = random.Random(seed)
    t0 = time.time(); acc = 0.0; loss_last = 0.0
    for step in range(1, steps + 1):
        m.train(); opt.zero_grad()
        inp, tgt, tv = make_batch(rng, N, bs, K)
        logits = m(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward(); opt.step()
        loss_last = loss.item()
        if step == steps or step == 1:
            acc = last_pos_accuracy(logits.detach(), tv, L)
    dt = time.time() - t0
    print(f"  {mixer:7s} N={N:3d} feat={feat:3d}  steps={steps}  loss={loss_last:.3f}  "
          f"val_acc={acc:.3f}  wall={dt:.0f}s")
    return {"mixer": mixer, "N": N, "feat": feat, "steps": steps,
            "val_acc": acc, "chance": 1.0 / K, "wall_s": round(dt, 1),
            "loss": loss_last}


def main():
    out = []
    out.append(run("attn", N=16, steps=200))
    out.append(run("attn", N=64, steps=200))
    out.append(run("ca", N=16, steps=200, feat=32))
    out.append(run("ca", N=16, steps=300, feat=128, chunk=16))
    out.append(run("cagate", N=8, steps=150, feat=32, chunk=0))
    path = os.path.join(RESULTS, "ar_quick.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()