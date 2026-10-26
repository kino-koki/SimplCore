# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 性能基准：对比 eager 与 torch.compile 前向延迟。

提供 ``run_benchmark`` 函数与命令行入口，输出量化加速比，作为性能验证证据。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import torch

from .config import SimplCoreConfig
from .model import SimplCoreModel
from .compile import compile_model
from .fx_passes import optimize_for_inference


@dataclass
class BenchmarkResult:
    eager_ms: float
    compiled_ms: float
    speedup: float
    params: int
    compiled_available: bool
    fx_ms: float = 0.0
    fx_speedup: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "eager_ms": self.eager_ms,
            "compiled_ms": self.compiled_ms,
            "speedup": self.speedup,
            "params": self.params,
            "compiled_available": int(self.compiled_available),
            "fx_ms": self.fx_ms,
            "fx_speedup": self.fx_speedup,
        }


def _time_fn(fn, reps: int) -> float:
    # 预热
    for _ in range(2):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def run_benchmark(
    config: SimplCoreConfig | None = None,
    batch_size: int = 4,
    seq_len: int = 64,
    reps: int = 20,
    device: str = "cpu",
) -> BenchmarkResult:
    """运行 eager vs compiled 前向基准。

    Returns:
        BenchmarkResult (含延迟与加速比)。
    """
    config = config or SimplCoreConfig.get_nano_config()
    model = SimplCoreModel(config).to(device).eval()
    ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        eager_ms = _time_fn(lambda: model(ids), reps)

    compiled_model = compile_model(model, mode="default", backend="inductor", smoke_input=(batch_size, seq_len))
    compiled_available = compiled_model is not model
    with torch.no_grad():
        compiled_ms = _time_fn(lambda: compiled_model(ids), reps)

    speedup = eager_ms / compiled_ms if compiled_ms > 0 else float("nan")

    # FX 图模式 (symbolic_trace + 优化 pass)：跨平台可用，是 Windows 等无
    # torch.compile 环境下的实际可用加速路径。图针对当前 (B, L) 特化。
    fx_graph = None
    fx_ms = 0.0
    fx_speedup = 0.0
    try:
        fx_model = model.trace_fx(example_input_ids=ids)
        fx_graph = optimize_for_inference(fx_model)
        with torch.no_grad():
            fx_ms = _time_fn(lambda: fx_graph(ids), reps)
        fx_speedup = eager_ms / fx_ms if fx_ms > 0 else float("nan")
    except Exception:  # pragma: no cover - 依赖运行环境的 FX 支持
        fx_graph = None

    return BenchmarkResult(
        eager_ms=eager_ms, compiled_ms=compiled_ms, speedup=speedup,
        params=model.num_parameters(), compiled_available=compiled_available,
        fx_ms=fx_ms, fx_speedup=fx_speedup,
    )


def _main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="SimplCore 性能基准")
    p.add_argument("--preset", default="nano")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--reps", type=int, default=20)
    args = p.parse_args()

    cfg = SimplCoreConfig.from_preset(args.preset)
    res = run_benchmark(cfg, batch_size=args.batch, seq_len=args.seq, reps=args.reps)
    print("=" * 60)
    print(f"SimplCore 基准 (preset={args.preset}, B={args.batch}, L={args.seq})")
    print("=" * 60)
    print(f"  参数量          : {res.params:,}")
    print(f"  eager 延迟      : {res.eager_ms:.3f} ms")
    print(f"  compiled 延迟   : {res.compiled_ms:.3f} ms  (torch.compile 可用={res.compiled_available})")
    print(f"    -> 加速比     : {res.speedup:.2f}x")
    if res.fx_ms > 0:
        print(f"  FX 优化图延迟   : {res.fx_ms:.3f} ms  (symbolic_trace + 优化 pass)")
        print(f"    -> 加速比     : {res.fx_speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    _main()
