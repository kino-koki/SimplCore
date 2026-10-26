# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC-HM 记忆保持实验: 检验"是否线性衰退" + "记忆容量随槽数增长".

任务 (记忆-保持-检索):
    先给出 K 对 (键->值) 再放 G 个干扰 token, 末尾给一个查询键,
    要求模型输出对应值. 关键度量:
      * 召回 vs 距离 G:            距离变大召回是否"掉"(线性注意力的稀释衰退) 还是"平".
      * 召回 vs 对数 K:            需记住的条数变大, 容量(槽数)决定何时开始掉.
    对照: 普通线性注意力 CA (S += k⊗v, 稀释型) vs 门控分层记忆 HM (delta 覆盖写).
"""
from __future__ import annotations
import os, sys, time, json, random, argparse
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_v3 import build_model

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KMAX = 256
VOCAB = 2 * KMAX + 2


def make_batch(rng, K, G, bs):
    """K 对 (键->值) + G 干扰 + 查询键 -> 预测值. L = 2K + 1 + G."""
    L = 2 * K + 1 + G
    inp = torch.zeros(bs, L, dtype=torch.long)
    tgt = torch.zeros(bs, L, dtype=torch.long)
    true_val = torch.zeros(bs, dtype=torch.long)
    j_sel = 0   # 固定查第 0 对, 使任务可学 (键要跨长距留存)
    for i in range(bs):
        keys = rng.sample(range(2, 2 + KMAX), K)          # K 个不同键
        vals = [2 + KMAX + rng.randrange(KMAX) for _ in range(K)]
        seq = []
        for a in range(K):
            seq += [keys[a], vals[a]]                      # 2K 个存储
        # G 个干扰 token, 从"不是任何键/值"的中间区取样
        dist_vocab = [x for x in range(2 + 2 * KMAX, VOCAB)] or [2 + KMAX]
        seq += [rng.choice(dist_vocab) for _ in range(G)]  # 干扰
        q = keys[j_sel]
        seq.append(q)                                     # 查询键
        tv = vals[j_sel]
        inp[i] = torch.tensor(seq)
        tgt[i] = torch.tensor(seq[1:] + [tv])
        true_val[i] = tv
    return inp, tgt, true_val, L


def last_acc(logits, tv, L):
    return (logits[:, -1, :].argmax(-1) == tv).float().mean().item()


def run(mixer, K, G, steps, dim=64, layers=2, bs=16, feat=64, seed=7, chunk=0):
    m = build_model(VOCAB, dim, layers, 4, 2 * 128 + 1 + 1024, mixer, ca_feat=feat,
                    ca_chunk=chunk).eval()
    opt = AdamW(m.parameters(), lr=1e-3)
    rng = random.Random(seed)
    acc = 0.0; loss_last = 0.0; t0 = time.time()
    for step in range(1, steps + 1):
        m.train(); opt.zero_grad()
        inp, tgt, tv, L = make_batch(rng, K, G, bs)
        if L > 2 * 128 + 1 + 1024:
            L = 2 * 128 + 1 + 1024
            inp = inp[:, :L]; tgt = tgt[:, :L]
        logits = m(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward(); opt.step(); loss_last = loss.item()
        if step == steps or step == 1:
            acc = last_acc(logits.detach(), tv, L)
    dt = time.time() - t0
    print(f"  {mixer:5s} K={K:3d} G={G:4d}  loss={loss_last:.3f}  recall@last={acc:.3f}  wall={dt:.0f}s")
    return {"mixer": mixer, "K": K, "G": G, "recall": acc, "steps": steps,
            "chance": 1.0 / KMAX, "wall_s": round(dt, 1), "params": sum(p.numel() for p in m.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", type=int, default=64, help="CA/HM 槽数 r (记忆容量)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--only", default="both", choices=["decay", "capacity", "both"])
    a = ap.parse_args()
    out = []

    if a.only in ("decay", "both"):
        print("=" * 70)
        print("① 线性衰退检验: 固定 K=16 对, 距离 G 从 0 拉到 1024, 召回是否随 G 掉")
        print("=" * 70)
        for G in [0, 128, 512, 1024]:
            out.append(run("ca", 16, G, a.steps, feat=a.feat, chunk=a.chunk))
            out.append(run("hm", 16, G, a.steps, feat=a.feat, chunk=a.chunk))

    if a.only in ("capacity", "both"):
        print("\n" + "=" * 70)
        print("② 容量检验: 固定 G=256 干扰, 需记住的对数 K 从 4 拉到 96, 召回何时被容量卡住")
        print("=" * 70)
        Lcap = 2 * 128 + 1 + 1024
        for K in [4, 16, 48, 96]:
            out.append(run("ca", K, 256, a.steps, feat=a.feat, chunk=a.chunk))
            out.append(run("hm", K, 256, a.steps, feat=a.feat, chunk=a.chunk))

    if a.feat:
        path = os.path.join(RESULTS, f"retention_feat{a.feat}.json")
        json.dump(out, open(path, "w"), indent=2)
        print("saved", path)


if __name__ == "__main__":
    main()