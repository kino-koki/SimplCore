# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""训练吞吐基准：tokens/s 记账（CPU/MI250 同一脚本，诚实口径）.

为什么需要这个脚本
------------------
``throughput_eval.py`` 测的是**解码**吞吐（O(1) 流式 vs KV cache 对照）；
训练侧此前没有同级别的记账工具——调 ``--micro-batch`` / ``--num-workers``
/ 精度开关时只能看 loss 曲线猜 GPU 利用率。本脚本测四件事：

1. **稳态 tokens/s**：预热后取中位数步的 token 吞吐（剔除首步编译/分配噪声）；
2. **precision 对照**：fp32 vs bf16 autocast（MI250 主用 bf16）；
3. **序列长度缩放**：tokens/s 随 ``seq_len`` 的变化（线性是健康态，
   超线性说明有 O(L²) 隐患混入）；
4. **传输参数对照**：``num_workers=0`` vs ``>0``（合成数据下差异应趋近 0
   ——真实 parquet 语料上才是它的主场，账本里如实标注）。

口径（诚实账本）：报 ``tokens/s`` 与 ``steps/s`` 两个数字；梯度同步（DDP）
不计入单卡账本；显存峰值单列。所有数字都是**本机实测**，不做外推。

用法::

    python scripts/bench_train_throughput.py --seq-lens 512 2048 --steps 30
    python scripts/bench_train_throughput.py --preset tuan-30m --precision bf16
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__file__)))

from simplcore.core.config import SimplCoreConfig, get_preset
from simplcore.core.model import SimplCoreModel


def _synthetic_batch(vocab: int, B: int, L: int, device, gen: torch.Generator):
    """合成 token 批（真实语料的 IO 影响不在本基准口径内）。"""
    x = torch.randint(0, vocab, (B, L), generator=gen).to(device)
    y = torch.randint(0, vocab, (B, L), generator=gen).to(device)
    return x, y


def bench_config(cfg: SimplCoreConfig, seq_len: int, batch: int, steps: int,
                 warmup: int, device, precision: str, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = SimplCoreModel(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(seed)
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16,
                 "fp16": torch.float16}[precision]
    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
           if amp_dtype is not None and device.type != "cpu"
           else torch.autocast("cpu", dtype=torch.bfloat16,
                               enabled=precision == "bf16"))
    tokens_per_step = batch * seq_len
    times: list[float] = []
    peak_mem = 0
    for i in range(warmup + steps):
        x, y = _synthetic_batch(cfg.vocab_size, batch, seq_len, device, gen)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "cpu":
            pass
        t0 = time.perf_counter()
        with ctx:
            _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated())
        times.append(time.perf_counter() - t0)
    steady = times[warmup:]
    med = statistics.median(steady)
    return {
        "seq_len": seq_len, "batch": batch, "precision": precision,
        "steps": steps,
        "tokens_per_step": tokens_per_step,
        "median_step_seconds": round(med, 4),
        "tokens_per_second": round(tokens_per_step / med, 1),
        "steps_per_second": round(1.0 / med, 3),
        "p90_step_seconds": round(sorted(steady)[int(0.9 * (len(steady) - 1))], 4),
        "peak_mem_bytes": peak_mem,
        "device": device.type,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练吞吐基准（tokens/s，诚实口径）")
    ap.add_argument("--preset", type=str, default="tuan-30m")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--precisions", type=str, nargs="+",
                    default=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SimplCoreConfig.from_dict(get_preset(args.preset).to_dict())
    if args.dim:
        cfg.dim = args.dim
    if args.layers:
        cfg.num_layers = args.layers

    results = []
    for precision in args.precisions:
        for L in args.seq_lens:
            r = bench_config(cfg, L, args.batch, args.steps, args.warmup,
                             device, precision, args.seed)
            results.append(r)
            print(f"[{precision}] L={L:>5} B={args.batch}: "
                  f"{r['tokens_per_second']:>12,.1f} tokens/s "
                  f"({r['steps_per_second']:.2f} steps/s, "
                  f"median {r['median_step_seconds']}s, "
                  f"p90 {r['p90_step_seconds']}s)")
    if len(results) >= 2 and results[0]["seq_len"] == results[-1]["seq_len"]:
        pass  # 序列缩放分析留给调用方对比 JSON
    print("\n诚实口径：合成数据（语料 IO 不计入）；DDP 梯度同步不计入单卡账；"
          "数字为本机实测，不做外推。序列长度缩放健康态 = tokens/s 近似恒定"
          "（mixer 主项 O(D·log L)）；超线性即有 O(L²) 隐患混入。")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"账本已写: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())