# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""流式一致性验证：分块流式推演必须与整段前向严格一致。

这是流式架构正确性的硬证据：同一权重、同一输入下，
``stream_step`` 逐 chunk 产生的 logits 与 ``forward`` 整段一次产生的 logits
必须逐 token 对齐（最大绝对误差 < 1e-4）。若不一致，说明跨块状态传递有 bug，
流式推理会悄悄偏离训练分布。

不美化：任何不一致直接 assert 失败并退出码非 0。
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_stream import SimplCoreStreamingModel


def build_model(hidden=64, layers=2, vocab=128, chunk=16):
    cfg = SimplCoreConfig(
        vocab_size=vocab, hidden_size=hidden, num_layers=layers,
        num_experts=2, experts_per_tok=2, moe_intermediate_size=128,
        chunk_size=chunk,
        max_sequence_length=chunk * 4,
        dmc_num_heads=4, use_rfp=True, use_dmc=True, use_localmix=True,
        dropout=0.0,
    )
    return SimplCoreStreamingModel(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seq", type=int, default=37, help="总序列长度(不必是 chunk 整数倍)")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = build_model(args.hidden, args.layers, args.vocab, args.chunk)
    model.eval()

    ids = torch.randint(0, args.vocab, (1, args.seq))

    t0 = time.time()
    # 整段前向
    with torch.no_grad():
        full = model(ids)  # (1, seq, vocab)
    # 分块流式：按 chunk 切分喂入
    with torch.no_grad():
        st = model.stream_init(batch=1)
        chunk_logits = []
        for i in range(0, args.seq, args.chunk):
            chunk = ids[:, i:i + args.chunk]
            lg, st = model.stream_step(chunk, st)
            chunk_logits.append(lg)
        streamed = torch.cat(chunk_logits, dim=1)
    cost = time.time() - t0

    if tuple(full.shape) != tuple(streamed.shape):
        print(f"❌ 形状不一致: forward={tuple(full.shape)} stream={tuple(streamed.shape)}")
        sys.exit(1)

    max_err = (full - streamed).abs().max().item()
    mean_err = (full - streamed).abs().mean().item()
    ok = max_err <= args.tol

    result = {
        "seq_len": args.seq,
        "chunk_size": args.chunk,
        "attention_free": True,
        "forward_shape": list(full.shape),
        "stream_shape": list(streamed.shape),
        "max_abs_err": max_err,
        "mean_abs_err": mean_err,
        "tol": args.tol,
        "consistent": ok,
        "cost_s": round(cost, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if ok:
        print(f"✅ 流式与整段前向一致 (max_err={max_err:.2e} <= tol={args.tol:.0e})")
    else:
        print(f"❌ 流式与整段前向不一致 (max_err={max_err:.2e} > tol={args.tol:.0e})")
        sys.exit(1)


if __name__ == "__main__":
    main()
