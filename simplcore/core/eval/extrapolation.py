# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 长上下文外推 + 衰减消融工具.

SimplCore 的核心差异化主张：**没有位置嵌入表、没有 ``max_len``**，因果卷积本身
编码位置，因此训练长度之外的长度"天然有定义"。但此前（``docs/SGC_CORE.md``
§9.4）只有"输出有限"的证据，没有质量证据。本模块提供两把可复现的尺子，
全部复用 :mod:`simplcore.core.eval` 的探针任务 + 可解性协议（``chance`` 水平
+ 训练对照），不把"输出有限"当成"质量保持"：

1. 长度外推探针 :func:`extrapolation_probe`
   -----------------------------------------
   在**固定**的短训练长度 L0 上训练一个可解任务，然后在 L0 的 1/2/4/8/16 倍
   长度上**零微调**评测，报告每个外推倍率下的准确率 + 困惑度 + 流式状态字节数
   （应恒为常数，与 L 无关）。曲线平坦即外推证据。

2. 衰减消融 :func:`decay_ablation`
   --------------------------------
   扫描 ``decay_max``（振荡基时间尺度先验），报告训练困惑度 + 2×外推准确率，
   回答 SGC_SPEC §9.5 的未证明项。

设计约束
--------
* 纯 torch，零第三方；训练/评测循环直接复用 ``eval.train_probe`` /
  ``eval.evaluate_probe``，不重造轮子；
* 可复现：固定 seed，报告记录全部超参；
* ``model_fn`` 钩子让你把任意模型（SGC / 混合 / 注意力基线）喂进同一把尺子，
  做公平的"外推能力"对照。

用法::

    python -m simplcore.core.extrapolation probe --preset debug --train-len 64 \\
        --scale 1,2,4,8 --task selective-copy --out x.json
    python -m simplcore.core.extrapolation decay --preset debug --train-len 64 \\
        --decay-maxes 64,256,1024,4096 --out d.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..model.config import SimplCoreConfig, get_preset
from ..model.model import SimplCoreModel
from . import eval as _eval

__all__ = [
    "ExtrapolationResult",
    "DecayPoint",
    "extrapolation_probe",
    "decay_ablation",
    "sgc_model_factory",
    "main",
]


# ----------------------------------------------------------------------------
# 探针任务构造（复用 eval 包）
# ----------------------------------------------------------------------------
def _make_task(task_name: str, seq_len: Optional[int] = None, **kw: Any):
    """按名字构造 eval 包里的探针任务；``seq_len`` 在构造时传入。"""
    from .eval import SelectiveCopyTask, IntervalRecallTask, AssociativeRecallTask
    if task_name in ("selective-copy", "selective_copy"):
        if seq_len is not None:
            return SelectiveCopyTask(seq_len=int(seq_len), **kw)
        return SelectiveCopyTask(**kw)
    if task_name in ("interval-recall", "interval_recall"):
        return IntervalRecallTask(**kw)
    if task_name in ("associative-recall", "mqar"):
        return AssociativeRecallTask(**kw)
    raise ValueError(f"unknown task {task_name!r}")


# ----------------------------------------------------------------------------
# 1. 长度外推探针
# ----------------------------------------------------------------------------
@dataclass
class ExtrapolationResult:
    """一次外推评测的结果。"""
    train_len: int
    scale: int                      # 外推倍率 k
    eval_len: int                   # L0 * k
    task: str
    chance: float
    acc: float                      # 零微调准确率
    ppl: Optional[float]            # 任务可评测位置上的困惑度
    solvable_at_L0: Optional[bool]  # L0 训练是否达到可解水平（对照基准）
    stream_bytes: int               # 该长度下流式状态字节（应恒为常数）
    wall_ms: float


def extrapolation_probe(model_fn: Callable[[], nn.Module], *,
                        task_name: str = "selective-copy",
                        task_kw: Optional[Dict[str, Any]] = None,
                        train_len: int = 64, scales: Sequence[int] = (1, 2, 4, 8),
                        steps: int = 300, batch_size: int = 64, lr: float = 3e-3,
                        n_samples: int = 512, seed: int = 0,
                        device: Any = "cpu", verbose: bool = True,
                        solvability_check: bool = True) -> List[ExtrapolationResult]:
    """长度外推探针。

    Args:
        model_fn: ``callable() -> model``，每次调用构造一个新模型（SGC / 混合 /
            注意力基线皆可）。训练用第一个实例，评测用它。
        task_name: selective-copy / interval-recall / associative-recall。
        train_len: 训练长度 L0（任务的 seq_len）。
        scales: 外推倍率 k，评测长度 = L0 * k。
        solvability_check: 是否在 L0 上做一次可解性检查（复用 eval 协议），
            作为"该任务在此预算下确实可解"的对照基准。

    协议（与 eval.py 一致）：
      * 在 L0 上训练 model_fn() 得到的模型；
      * 对每个 k，构造 seq_len=L0*k 的同任务评测集，**零微调**评测准确率/困惑度；
      * 报告 ``stream_bytes``（应恒为常数，与 L 无关——这是 SimplCore 的 O(1) 主张）。
    """
    results: List[ExtrapolationResult] = []
    kw = dict(task_kw or {})
    task0 = _make_task(task_name, seq_len=train_len, **kw)
    chance = float(task0.chance)

    model = model_fn()
    t0 = time.time()
    _eval.train_probe(task0, steps=steps, batch_size=batch_size, lr=lr,
                      seed=seed, device=device, model_fn=lambda: model,
                      final_eval_samples=min(n_samples, 1024), log=False,
                      progress=False, eval_batches=1)
    model.eval()
    solvable = None
    if solvability_check:
        sol = _eval.evaluate_probe(model, task0, n_samples=min(n_samples, 1024),
                                   seed=seed, device=device)
        solvable = bool(sol["probe_accuracy"] >= max(1.5 * chance, chance + 0.05))
    if verbose:
        print(f"[extrap] trained on {task_name} @ L={train_len} ({steps} steps, "
              f"{time.time() - t0:.1f}s); solvable={solvable}", flush=True)

    for k in scales:
        eval_len = train_len * k
        task_k = _make_task(task_name, seq_len=eval_len, **kw)
        rep = _eval.evaluate_probe(model, task_k, n_samples=n_samples,
                                   seed=seed + 1000 * k, device=device)
        sbytes = int(model.stream_bytes(1)) if hasattr(model, "stream_bytes") else 0
        r = ExtrapolationResult(
            train_len=train_len, scale=k, eval_len=eval_len, task=task_name,
            chance=chance, acc=float(rep["probe_accuracy"]),
            ppl=(float(rep["probe_ppl"]) if rep.get("probe_ppl") else None),
            solvable_at_L0=solvable, stream_bytes=sbytes,
            wall_ms=(time.time() - t0) * 1000.0,
        )
        results.append(r)
        if verbose:
            print(f"[extrap] k={k:2d} L={eval_len:4d} acc={r.acc:.4f} "
                  f"chance={chance:.4f} ppl={r.ppl if r.ppl else float('nan'):.3f} "
                  f"stream_bytes={sbytes}", flush=True)
    return results


def sgc_model_factory(config: SimplCoreConfig) -> Callable[[], SimplCoreModel]:
    """返回 ``() -> SimplCoreModel``，每次构造一个新模型（同 config）。"""
    def factory() -> SimplCoreModel:
        return SimplCoreModel(config)
    return factory


# ----------------------------------------------------------------------------
# 2. 衰减消融
# ----------------------------------------------------------------------------
@dataclass
class DecayPoint:
    decay_max: float
    train_ppl: float
    extrap_ppl: float          # 2x 长度下的困惑度（外推质量指标）
    acc: float
    extrap_acc: float
    note: str


def decay_ablation(config: SimplCoreConfig, *,
                   decay_maxes: Sequence[float] = (64, 256, 1024, 4096),
                   task_name: str = "selective-copy", train_len: int = 64,
                   steps: int = 300, batch_size: int = 64, lr: float = 3e-3,
                   n_samples: int = 256, seed: int = 0,
                   device: Any = "cpu", verbose: bool = True,
                   ) -> List[DecayPoint]:
    """扫描 ``decay_max``，报告各取值的训练困惑度 + 2×外推准确率/困惑度。

    这是 SGC_SPEC §9.5 的未证明项：默认 8192/16384 是按目标训练长度
    设定的先验，本函数给出"不同时间尺度先验 → 最终质量 + 外推保持"的实测曲线。
    """
    from ..model.config import SimplCoreConfig
    points: List[DecayPoint] = []
    for dm in decay_maxes:
        cfg = SimplCoreConfig.from_dict({**config.to_dict(), "decay_max": float(dm)})
        m = SimplCoreModel(cfg).to(device)
        task0 = _make_task(task_name, seq_len=train_len)
        _eval.train_probe(task0, steps=steps, batch_size=batch_size, lr=lr,
                          seed=seed, device=device, model_fn=lambda: m,
                          final_eval_samples=n_samples, log=False, progress=False,
                          eval_batches=1)
        m.eval()
        rep0 = _eval.evaluate_probe(m, task0, n_samples=n_samples, seed=seed, device=device)
        task2 = _make_task(task_name, seq_len=train_len * 2)
        rep2 = _eval.evaluate_probe(m, task2, n_samples=n_samples, seed=seed + 1,
                                   device=device)
        p = DecayPoint(
            decay_max=float(dm),
            train_ppl=float(rep0["probe_ppl"] or 0.0),
            extrap_ppl=float(rep2["probe_ppl"] or 0.0),
            acc=float(rep0["probe_accuracy"]),
            extrap_acc=float(rep2["probe_accuracy"]),
            note=f"L={train_len}, scales=2x",
        )
        points.append(p)
        if verbose:
            print(f"[decay] decay_max={dm:6g} train_ppl={p.train_ppl:.4f} "
                  f"extrap_ppl={p.extrap_ppl:.4f} acc={p.acc:.4f} "
                  f"extrap_acc={p.extrap_acc:.4f}", flush=True)
    return points


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="length-extrapolation probe")
    p_probe.add_argument("--preset", default="debug")
    p_probe.add_argument("--task", default="selective-copy",
                         choices=["selective-copy", "interval-recall",
                                  "associative-recall"])
    p_probe.add_argument("--train-len", type=int, default=64)
    p_probe.add_argument("--scale", default="1,2,4,8")
    p_probe.add_argument("--steps", type=int, default=300)
    p_probe.add_argument("--n-samples", type=int, default=512)
    p_probe.add_argument("--out", default=None)

    p_decay = sub.add_parser("decay", help="decay_max ablation")
    p_decay.add_argument("--preset", default="debug")
    p_decay.add_argument("--task", default="selective-copy")
    p_decay.add_argument("--train-len", type=int, default=64)
    p_decay.add_argument("--decay-maxes", default="64,256,1024,4096")
    p_decay.add_argument("--steps", type=int, default=300)
    p_decay.add_argument("--n-samples", type=int, default=256)
    p_decay.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    cfg = get_preset(args.preset)

    if args.cmd == "probe":
        scales = tuple(int(x) for x in args.scale.split(","))
        factory = sgc_model_factory(cfg)
        res = extrapolation_probe(factory, task_name=args.task, train_len=args.train_len,
                                  scales=scales, steps=args.steps,
                                  n_samples=args.n_samples)
        report = {"command": "probe", "config": cfg.to_dict(),
                  "results": [asdict(r) for r in res]}
        _dump(report, args.out)
    elif args.cmd == "decay":
        dms = tuple(float(x) for x in args.decay_maxes.split(","))
        pts = decay_ablation(cfg, decay_maxes=dms, task_name=args.task,
                             train_len=args.train_len, steps=args.steps,
                             n_samples=args.n_samples)
        report = {"command": "decay", "config": cfg.to_dict(),
                  "points": [asdict(p) for p in pts]}
        _dump(report, args.out)
    return 0


def _dump(report: Dict[str, Any], out: Optional[str]) -> None:
    s = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(s)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(s, encoding="utf-8")
        print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
