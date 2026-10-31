#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SimplUnity 后台自进化入口（2B 主干 + 多域专家并行训练）。

用法：
    python scripts/unity_evolve_2b.py \
      --checkpoint runs/tuan-2b/last.pt \
      --domains-dir data/experience \
      --out runs/tuan-2b/unity.pt \
      --workers 2 --steps-per-domain 200

``domains-dir`` 下每个子目录是一个域，目录内 ``*.txt`` 是该域经验文本；
每个域按 80/20 切 train/eval，候选专家训练后必须通过 GATE-3 无回归门才会发布。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.tokenization import make_codec
from simplcore.unity import (
    ExperienceLoader, UnityModel, UnitySelfEvolution, for_preset,
)


def _read_domain(path: Path) -> list[str]:
    texts: list[str] = []
    for f in sorted(path.glob("*.txt")):
        raw = f.read_text(encoding="utf-8", errors="ignore")
        texts.extend(t.strip() for t in raw.splitlines() if t.strip())
    return texts


def _split(texts: list[str], eval_ratio: float = 0.2):
    if len(texts) < 2:
        return texts, texts
    n = max(1, int(round(len(texts) * eval_ratio)))
    return texts[:-n], texts[-n:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--domains-dir", required=True)
    ap.add_argument("--out", default="runs/unity/unity.pt")
    ap.add_argument("--preset", default="tuan-2b")
    ap.add_argument("--codec", default="char")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps-per-domain", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-regression", type=float, default=0.02)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu",
                      weights_only=False)
    cfg = SimplCoreConfig(**ckpt["config"])
    base = SimplCoreModel(cfg)
    base.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    base.to(device).eval()

    unity = UnityModel(base, for_preset(args.preset), base_params=None)
    codec = make_codec(args.codec, limit=int(cfg.vocab_size))
    evo = UnitySelfEvolution(unity, device=device, steps_per_domain=args.steps_per_domain,
                             lr=args.lr, num_workers=args.workers)

    domains = [p for p in Path(args.domains_dir).iterdir() if p.is_dir()]
    if not domains:
        raise SystemExit(f"no domain subdirectories in {args.domains_dir}")
    for d in domains:
        texts = _read_domain(d)
        if not texts:
            print(f"[unity] skip empty domain {d.name}")
            continue
        train_texts, eval_texts = _split(texts)
        train_loader = ExperienceLoader(codec, train_texts,
                                        seq_len=args.seq_len,
                                        batch_size=args.batch)
        eval_loader = ExperienceLoader(codec, eval_texts,
                                       seq_len=args.seq_len,
                                       batch_size=args.batch)
        if len(train_loader) == 0 or len(eval_loader) == 0:
            print(f"[unity] skip domain {d.name}: not enough tokens")
            continue
        evo.register_domain(d.name, iter(train_loader), eval_loader=eval_loader,
                            max_regression=args.max_regression)
        print(f"[unity] queued {d.name}: {train_loader.stats()}")

    evo.start()
    deadline = time.time() + args.timeout
    try:
        while evo.num_pending > 0 and time.time() < deadline:
            time.sleep(0.5)
    finally:
        evo.stop()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    unity.save_learned(str(out))
    # 每个专家独立导出，供跨主干拼接（bundle 内含 base_fingerprint/训练量）
    bundle_dir = out.parent / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    for i, ep in enumerate(unity.experts):
        unity.export_expert(i, trained_tokens=args.steps_per_domain).save(
            bundle_dir / f"{ep.name}.pt")
    report = {"status": evo.status(), "out": str(out),
              "bundles": str(bundle_dir)}
    (out.parent / "unity_evolve.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["status"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())