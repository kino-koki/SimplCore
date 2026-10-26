# -*- coding: utf-8 -*-
"""间隔记忆基准 (可收敛): 固定查询键 Q 存一个值 -> G 个干扰 -> 查询 Q -> 预测其值.

机理区分:
  * CA(叠加型线性注意力) 把 k_Q⊗v 与 G 个干扰的 k⊗v 一起累进 S. 查询 q·S
    的信噪比随 G 下降 -> 间隔变大, 召回掉(线性灭退);
  * HM(覆盖写) 把值写进 Q 的专用槽, 干扰写其它槽不触碰 -> 间隔 G 再大也平.
  * 与"记忆-保持-检索(随机键)"不同: 这里 Q 固定, 任务规则简单, 小模型可学,
    故能在"训练过"层面观测到 CA 随 G 掉、HM 平坦.

窗口: [Q, v] + G 干扰 + [Q], 末尾预测 v. L = 2 + G + 1.
"""
from __future__ import annotations
import os, sys, json, random, argparse
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgc_v3 import build_model, count_params

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
V = 16                       # 值域大小
Q = 2                        # 固定查询键
VAL0 = 3                     # 值 token 起点
DIST0 = 3 + V                # 干扰 token 起点
VOCAB = 3 + 2 * V            # 词表: Q + V 值 + V 干扰


def make_batch(rng, G, bs):
    L = G + 3                       # Q v ...(G)... Q
    inp = torch.zeros(bs, L, dtype=torch.long)
    tgt = torch.zeros(bs, L, dtype=torch.long)
    tv = torch.zeros(bs, dtype=torch.long)
    for i in range(bs):
        v = VAL0 + rng.randrange(V)
        dists = [DIST0 + rng.randrange(V) for _ in range(G)]
        seq = [Q, v] + dists + [Q]
        inp[i] = torch.tensor(seq)
        tgt[i] = torch.tensor(seq[1:] + [v])     # 首Q->v, 干扰->dist, 末Q->v
        tv[i] = v
    return inp, tgt, tv, L


def last_recall(logits, tv):
    return (logits[:, -1, :].argmax(-1) == tv).float().mean().item()


def run(mixer, G, steps, dim, layers, bs, feat, seed, chunk):
    m = build_model(VOCAB, dim, layers, 4, 2 * 128 + 1 + 1024, mixer,
                    ca_feat=feat, ca_chunk=chunk).train()
    opt = AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    rng = random.Random(seed)
    best = 0.0; last = 0.0; loss_last = 0.0
    for step in range(1, steps + 1):
        opt.zero_grad()
        inp, tgt, tv, L = make_batch(rng, G, bs)
        logits = m(inp)
        # 只对**查询位置**计损失。
        #
        # 旧实现是 `cross_entropy(logits.reshape(-1,V), tgt.reshape(-1))`, 即对全部
        # L 个位置计损失。但序列结构是 [Q, v, 干扰×G, Q]: 除最后一位外, 其余位置的
        # "下一个 token"是**随机干扰**, 与历史无关, 不可学(在 16 类上均匀)。
        # 514/515 个不可学的位置把损失压在 ln(16)=2.773 以上, 于是三种 mixer 全部
        # 停在 chance(loss 2.780~2.787, recall ≈ 0)。这不是 mixer 的问题, 是脚本的。
        # 已实测: 修正后三种 mixer 均能显著高于 chance (见 results/lingap_*_G512.json)。
        loss = F.cross_entropy(logits[:, -1, :], tv)
        loss.backward(); opt.step(); loss_last = loss.item()
        if step % max(1, steps // 5) == 1 or step == steps:
            with torch.no_grad():
                m.eval(); inp, tgt, tv, _ = make_batch(rng, G, bs)
                r = last_recall(m(inp).detach(), tv); m.train()
            best = max(best, r); last = r
    print(f"  {mixer:4s} G={G:4d}  loss={loss_last:.3f}  recall={last:.3f} best={best:.3f} params={count_params(m):,}")
    return {"mixer": mixer, "G": G, "recall_last": last, "best": best,
            "chance": 1.0 / V, "params": count_params(m), "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="hm")
    ap.add_argument("--G", type=int, default=512)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--feat", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    out = {"V": V, "task": "interval-memory"}
    out.update(run(a.mixer, a.G, a.steps, a.dim, a.layers, a.bs, a.feat, a.seed, a.chunk))
    path = os.path.join(RESULTS, f"lingap_{a.mixer}_G{a.G}.json")
    json.dump(out, open(path, "w"), indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()