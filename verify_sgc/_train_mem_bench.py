# -*- coding: utf-8 -*-
"""端到端可学的记忆-保持-检索基准 (小词表, 可收敛): 公平对比 attn / ca / hm.

任务: K 对 (键->值) 存入上下文, G 个干扰 token, 末尾给查询键 -> 预测对应值.
  * vocab = 2*KMAX+2 (KMAX 控制键/值域大小, 小词表保证任务可学);
  * 度量: 最后一位 recall (必须靠长程联想召回, 不是局部统计);
  * 三个 mixer 同骨架同规模, 唯一差异是混合器; 训练步数/优化完全一致.
"""
from __future__ import annotations
import os, sys, time, json, random, argparse
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_v3 import build_model, count_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KMAX = 32
VOCAB = 2 * KMAX + 2
MAXLEN = 2 * 128 + 1 + 1024


def make_batch(rng, K, G, bs):
    L = 2 * K + 1 + G
    inp = torch.zeros(bs, L, dtype=torch.long)
    tgt = torch.zeros(bs, L, dtype=torch.long)
    tv = torch.zeros(bs, dtype=torch.long)
    dist_vocab = [x for x in range(2 + 2 * KMAX, VOCAB)] or [2 + KMAX]
    for i in range(bs):
        keys = rng.sample(range(2, 2 + KMAX), K)
        vals = [2 + KMAX + rng.randrange(KMAX) for _ in range(K)]
        seq = []
        for a in range(K):
            seq += [keys[a], vals[a]]
        seq += [rng.choice(dist_vocab) for _ in range(G)]
        q = keys[0]
        seq.append(q)
        tv[i] = vals[0]
        inp[i] = torch.tensor(seq)
        tgt[i] = torch.tensor(seq[1:] + [vals[0]])
    return inp, tgt, tv, L


def last_recall(logits, tv):
    return (logits[:, -1, :].argmax(-1) == tv).float().mean().item()


def run(mixer, K, G, steps, dim, layers, bs, feat, seed, chunk):
    m = build_model(VOCAB, dim, layers, 4, MAXLEN, mixer, ca_feat=feat,
                    ca_chunk=chunk).train()
    opt = AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    rng = random.Random(seed)
    best = 0.0; last = 0.0; loss_last = 0.0
    for step in range(1, steps + 1):
        opt.zero_grad()
        inp, tgt, tv, L = make_batch(rng, K, G, bs)
        logits = m(inp)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward(); opt.step(); loss_last = loss.item()
        if step == steps or step % max(1, steps // 5) == 0 or step == 1:
            with torch.no_grad():
                m.eval(); inp, tgt, tv, L = make_batch(rng, K, G, bs)
                r = last_recall(m(inp).detach(), tv); m.train()
            best = max(best, r); last = r
    print(f"  {mixer:4s} K={K:2d} G={G:4d}  loss={loss_last:.3f}  last_recall={last:.3f} best={best:.3f} "
          f"params={count_params(m):,}")
    return {"mixer": mixer, "K": K, "G": G, "recall_last": last, "best": best,
            "chance": 1.0 / KMAX, "params": count_params(m), "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="all", help="attr: attn|ca|hm|all")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--feat", type=int, default=64)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    mixers = ["attn", "ca", "hm"] if a.mixer == "all" else [a.mixer]
    out = []
    for mx in mixers:
        out.append(run(mx, a.K, a.G, a.steps, a.dim, a.layers, a.bs, a.feat, a.seed, a.chunk))
    path = os.path.join(RESULTS, f"membench_K{a.K}_G{a.G}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()