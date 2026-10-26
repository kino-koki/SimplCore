# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC 真实 LM 收敛性消融: 定位困惑度爆炸根因并修复.

对比:
  (A) 基线 SGC: out_scale=1 起, lr=3e-4
  (B) out_scale=0 起 (残差先主导, 逐步引入卷积)
  (C) 更高 lr=1e-3
  (D) 内容门控 SGC: 核依赖输入 (类注意力但保持 O(n log n))
每个配置跑固定步数, 记录 train_loss 终值与 val_ppl, 判断是否收敛.
"""
from __future__ import annotations

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_pytorch import LMModel, count_params, AttentionMixer, SpectralGlobalConvV2, FFN, Block

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORPUS = os.path.join(RESULTS, "corpus.json")


class SGCZeroScale(SpectralGlobalConvV2):
    """out_scale 从 0 起, 让残差先主导."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        with torch.no_grad():
            self.out_scale.zero_()


class SGCContentGated(SpectralGlobalConvV2):
    """内容门控: 卷积输出乘以依赖输入的门, 提升真实 LM 容量."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.gate = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)
        y = self._causal_conv(u)
        g = torch.sigmoid(self.gate(h))
        out = self.out_proj(y) * g
        return residual + self.out_scale * out


def build_model(variant, vocab, dim, layers, heads, seq_len, rank):
    if variant == "attn":
        return LMModel(vocab, dim, layers, heads, seq_len, "attn")
    base = LMModel(vocab, dim, layers, heads, seq_len, "sgc", sgc_rank=rank)
    if variant == "sgc":
        return base
    if variant == "sgc_zero":
        # 替换每个 block 的 mixer 为 out_scale=0 版
        for blk in base.blocks:
            d = blk.mixer.dim
            blk.mixer = SGCZeroScale(d, 3, seq_len, rank)
        return base
    if variant == "sgc_gated":
        for blk in base.blocks:
            d = blk.mixer.dim
            blk.mixer = SGCContentGated(d, 3, seq_len, rank)
        return base
    raise ValueError(variant)


def batchify(samples, bs, seq_len):
    xb = torch.zeros(bs, seq_len, dtype=torch.long)
    yb = torch.zeros(bs, seq_len, dtype=torch.long)
    for i in range(min(bs, len(samples))):
        s = samples[i]
        xb[i] = torch.tensor(s[:seq_len])
        yb[i] = torch.tensor(s[1:seq_len + 1])
    return xb, yb


def run(variant, vocab, train, val, steps, lr, dim, layers, heads, seq_len, rank):
    torch.manual_seed(1234)
    model = build_model(variant, vocab, dim, layers, heads, seq_len, rank)
    n = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    xb, yb = batchify(train, 8, seq_len)
    vx, vy = batchify(val, 8, seq_len)
    model.train()
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        l = model(xb)
        loss = F.cross_entropy(l.reshape(-1, vocab), yb.reshape(-1))
        loss.backward()
        opt.step()
        if (step + 1) % max(1, steps // 5) == 0:
            losses.append(float(loss.item()))
    dt = (time.perf_counter() - t0) / steps
    model.eval()
    with torch.no_grad():
        vl = model(vx)
        vloss = F.cross_entropy(vl.reshape(-1, vocab), vy.reshape(-1)).item()
    ppl = float(torch.exp(torch.tensor(vloss)))
    return {"variant": variant, "params": n, "final_train_loss": losses[-1],
            "train_losses": losses, "val_loss": vloss, "val_ppl": ppl,
            "step_time_s": dt, "steps": steps, "lr": lr}


def main():
    d = json.load(open(CORPUS))
    vocab, train, val = d["vocab"], d["train"], d["val"]
    args = dict(dim=256, layers=6, heads=4, seq_len=256, rank=8, steps=1500)
    variants = [
        ("attn", 3e-4),
        ("sgc", 3e-4),
        ("sgc_zero", 3e-4),
        ("sgc_gated", 3e-4),
        ("sgc", 1e-3),
    ]
    out = {}
    for var, lr in variants:
        r = run(var, len(vocab), train, val, args["steps"], lr,
                args["dim"], args["layers"], args["heads"], args["seq_len"], args["rank"])
        out[var if lr == 3e-4 else f"{var}_lr{lr}"] = r
        print(f"[{var} lr={lr}] train_loss={r['final_train_loss']:.3f} "
              f"val_loss={r['val_loss']:.3f} ppl={r['val_ppl']:.1f} "
              f"step={r['step_time_s']:.3f}s")
    json.dump(out, open(os.path.join(RESULTS, "sgc_ablation.json"), "w"), indent=2)
    print("saved -> sgc_ablation.json")


if __name__ == "__main__":
    main()
