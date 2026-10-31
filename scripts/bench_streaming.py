# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""流式推理基准：状态大小、逐 token 延迟、流式 vs 全量一致性。

为什么需要这个脚本
------------------
SGC/SRFN 的卖点是 **O(1) 状态**：生成长度翻 100 倍，推理状态不变大。
加了「推理注意力」之后，这个卖点一度被打破——注意力要 KV cache。

现在推理注意力用**固定容量的环形缓冲**（容量 = 窗口宽度），所以状态重新变成
定长的。但这是"设计如此"，不是"测出来如此"：如果哪天有人把环形缓冲改成
无限增长的 list，这个脚本会立刻报出状态字节数随长度线性增长。

所以它测三件事：

1. **状态字节数 vs 生成长度**——必须是常数；
2. **逐 token 延迟**——必须与上下文长度无关（否则说明有隐性 O(L) 操作）；
3. **流式输出 == 全量前向**——两条路径必须给出同一个答案。

用法::

    python scripts/bench_streaming.py --lengths 128 512 2048
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplcore.core.config import SimplCoreConfig, get_preset
from simplcore.core.model import SimplCoreModel


def build(mixer: str, reason_attn_every=None, dim=320, layers=6,
          num_heads=8, num_kv_heads=2, attn_window=256, vocab=494) -> SimplCoreModel:
    cfg = SimplCoreConfig.from_dict(get_preset("tuan-30m").to_dict())
    cfg.vocab_size = vocab
    cfg.dim = dim
    cfg.num_layers = layers
    cfg.mixer = mixer
    cfg.srf_heads = 32
    cfg.srf_chunk = 128
    cfg.num_heads = num_heads
    cfg.num_kv_heads = num_kv_heads
    cfg.attn_window = attn_window
    cfg.reason_attn_every = reason_attn_every
    cfg.validate()
    m = SimplCoreModel(cfg)
    m.eval()
    return m


@torch.no_grad()
def stream_once(model: SimplCoreModel, tokens: torch.Tensor):
    """逐 token 喂入，返回 (最后 logits, 用时秒)。"""
    st = model.init_stream(1, device=tokens.device, dtype=torch.float32)
    logits = None
    t0 = time.perf_counter()
    for t in range(tokens.shape[1]):
        logits, st = model.stream_step(tokens[:, t], st)
    return logits, st, time.perf_counter() - t0


def bench(name: str, model: SimplCoreModel, lengths, check_full: bool) -> dict:
    torch.manual_seed(0)
    row = {"name": name, "params": model.num_params(),
           "supports_streaming": model.supports_streaming, "lengths": {}}
    for L in lengths:
        tokens = torch.randint(0, model.config.vocab_size, (1, L))
        logits, st, dt = stream_once(model, tokens)
        nbytes = st.num_bytes()
        state_bytes = model.stream_bytes(1)
        entry = {
            "state_bytes": nbytes,
            "stream_bytes_api": state_bytes,
            "seconds": round(dt, 3),
            "ms_per_token": round(dt / L * 1000, 4),
            "last_logit_sum": round(float(logits.sum()), 4),
        }
        if check_full:
            full = model(tokens[:, :L])[0, -1]
            entry["full_vs_stream_maxdiff"] = float(
                (full - logits[0]).abs().max())
        row["lengths"][str(L)] = entry
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    rows = [
        bench("sgc (FFT 固定核)", build("sgc"), args.lengths, True),
        bench("srf (selective SSM)", build("srf"), args.lengths, True),
        bench("srf + 推理注意力", build("srf", reason_attn_every=2),
              args.lengths, True),
    ]

    print(f"{'配置':>22s} {'长度':>7s} {'状态字节':>10s} {'ms/token':>10s} "
          f"{'流式vs全量':>12s}")
    print("-" * 68)
    for r in rows:
        for L, e in r["lengths"].items():
            diff = e.get("full_vs_stream_maxdiff")
            diff_s = f"{diff:.2e}" if diff is not None else "—"
            print(f"{r['name']:>22s} {int(L):>7d} {e['state_bytes']:>10d} "
                  f"{e['ms_per_token']:>10.4f} {diff_s:>12s}")
        print()

    # 断言：状态大小必须是常数
    for r in rows:
        sizes = {e["state_bytes"] for e in r["lengths"].values()}
        assert len(sizes) == 1, f"{r['name']} 的状态大小随长度变化了：{sizes}"
    print("状态大小对所有长度均为常数 ✓")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print(f"写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
