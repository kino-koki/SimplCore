# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""长程联想召回 (Associative Recall, AR) 合成任务生成器.

任务定义 (业界常用于检验长程上下文召回能力, 见 linear-attention /
Mamba 等论文):
  给定 N 个"键-值"对 (k_i, v_i) 后接一个查询键 q (q 等于某个 k_j),
  模型需在 q 之后输出对应的 v_j. 由于 q 与 k_j 之间隔着大量干扰对,
  正确召回要求模型在长程上下文中保留并检索到较早出现的 (k_j -> v_j).

信息论含义:
  - 全局卷积 / 注意力: 状态随 n 增长, 可保留全部 n 个 (k,v) -> 高召回.
  - 线性注意力: 状态 D^2 (有界) -> n>>D 时丢失 -> 召回随 N 退化.
  - Mamba式SSM:   状态 N_ssm·D (有界) -> 同上, 且容量更小 -> 退化更快.

vocab 划分: [0]=PAD, [1]=QUERY_SEP, [2..K+1]=键, [K+2..2K+1]=值.
"""
from __future__ import annotations

import torch


PAD, SEP = 0, 1


def make_sample(rng, N, K):
    """返回单个 AR 样本的输入/目标 token 序列 (长度 2N+1)."""
    keys = list(rng.sample(range(2, 2 + K), N))          # N 个不同键
    vals = [2 + K + int(rng.random() * K) for _ in range(N)]  # N 个值
    seq = []
    for i in range(N):
        seq.append(keys[i])
        seq.append(vals[i])
    j = int(rng.random() * N)                              # 查询哪一对
    q = keys[j]
    seq.append(q)                                          # 查询键
    inp = list(seq)
    # 目标: 输入左移一位 (下一个 token), 末位补上应召回的值 v_j
    tgt = list(seq[1:]) + [vals[j]]
    return inp, tgt, vals[j]


def make_batch(rng, N, bs, K):
    """生成 (bs, 2N+1) 的输入/目标张量, 以及末位应召回的真值."""
    L = 2 * N + 1
    inp = torch.zeros(bs, L, dtype=torch.long)
    tgt = torch.zeros(bs, L, dtype=torch.long)
    true_val = torch.zeros(bs, dtype=torch.long)
    for i in range(bs):
        xi, yt, vj = make_sample(rng, N, K)
        inp[i] = torch.tensor(xi)
        tgt[i] = torch.tensor(yt)
        true_val[i] = vj
    return inp, tgt, true_val


def last_pos_accuracy(logits, true_val, L):
    """仅看末位 (查询之后) 的召回准确率."""
    last = logits[:, -1, :]                 # (bs, vocab)
    pred = last.argmax(dim=-1)
    return (pred == true_val).float().mean().item()


if __name__ == "__main__":
    import random
    rng = random.Random(0)
    K = 256
    for N in [16, 64, 128]:
        inp, tgt, tv = make_batch(rng, N, 4, K)
        print(f"N={N:3d} L={2*N+1:3d} inp[:6]={inp[0][:6].tolist()} "
              f"true_val[0]={int(tv[0])} tgt_last={int(tgt[0][-1])}")
