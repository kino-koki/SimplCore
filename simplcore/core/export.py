# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Save / load SimplCore models in the ``safetensors`` format.

``simplcore.core.train`` writes ``torch.save`` checkpoints (pickled optimizer +
RNG state, needed for resumable training). Those are *not* a deployment format:
pickled files are unsafe to load from untrusted sources and are not readable by
the HuggingFace ecosystem.

This module converts between the two worlds:

* ``save_pretrained(model, out_dir)``  -> ``config.json`` + ``model.safetensors``
* ``load_pretrained(out_dir)``         -> a ready :class:`SimplCoreModel`
* ``export_training_ckpt(ckpt, out)``  -> deployment dir from a training checkpoint

Details that matter:

* **Tied weights.** When ``cfg.tie_embeddings`` is set (the default),
  ``head.weight`` and ``tok_emb.weight`` are the *same tensor*. ``safetensors``
  rejects shared tensors, so the duplicate key is dropped on save and the tie is
  re-established on load; the test suite asserts round-trip equality either way.
* **No optimizer / no pickle.** Only ``model.state_dict()`` enters the file.
* Weights are written as saved (fp32 on CPU); bf16/fp16 conversion, if desired,
  belongs to the serving stack, not here.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Tuple

import torch

from .config import SimplCoreConfig
from .model import SimplCoreModel

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
TOKENIZER_NAME = "tokenizer.json"


def save_pretrained(model: SimplCoreModel, out_dir: str,
                    tokenizer_json: Optional[str] = None) -> str:
    """Write ``config.json`` + ``model.safetensors`` (and copy a tokenizer) to ``out_dir``.

    Args:
        model: an :class:`SimplCoreModel`.
        out_dir: destination directory (created if missing).
        tokenizer_json: optional path of a tokenizer to copy next to the weights,
            so the export is self-contained for serving.

    Returns:
        The path of the written weights file.
    """
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    cfg: SimplCoreConfig = model.config
    cfg.to_json(os.path.join(out_dir, CONFIG_NAME))

    tensors = {}
    for name, t in model.state_dict().items():
        if name == "head.weight" and cfg.tie_embeddings:
            continue                      # safetensors forbids shared storage
        tensors[name] = t.detach().cpu().contiguous()

    weights_path = os.path.join(out_dir, WEIGHTS_NAME)
    save_file(tensors, weights_path)

    if tokenizer_json and os.path.exists(tokenizer_json):
        with open(tokenizer_json, "rb") as src, \
                open(os.path.join(out_dir, TOKENIZER_NAME), "wb") as dst:
            dst.write(src.read())
    return weights_path


def load_pretrained(out_dir: str, map_location: str = "cpu") -> SimplCoreModel:
    """Rebuild a model from an export directory written by :func:`save_pretrained`."""
    from safetensors.torch import load_file

    cfg = SimplCoreConfig.from_json(os.path.join(out_dir, CONFIG_NAME))
    state = load_file(os.path.join(out_dir, WEIGHTS_NAME), device=map_location)
    model = SimplCoreModel(cfg)
    if cfg.tie_embeddings:
        state["head.weight"] = state["tok_emb.weight"]   # re-establish the tie
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def export_training_ckpt(ckpt_path: str, out_dir: str,
                         tokenizer_json: Optional[str] = None) -> dict:
    """Convert a resumable training checkpoint (``best.pt`` / ``last.pt``) to an export.

    Returns a small provenance dict (step, tokens, best_val) for the caller to log.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise ValueError(f"{ckpt_path} is not a simplcore.core training checkpoint")
    cfg = SimplCoreConfig.from_dict(ckpt["config"])
    model = SimplCoreModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    save_pretrained(model, out_dir, tokenizer_json=tokenizer_json)
    meta = {k: ckpt.get(k) for k in ("step", "tokens", "best_val")}
    meta["params"] = model.num_params()
    with open(os.path.join(out_dir, "export_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"source_checkpoint": os.path.abspath(ckpt_path), **meta},
                  fh, indent=2, ensure_ascii=False)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="training checkpoint (best.pt / last.pt)")
    ap.add_argument("--out", required=True, help="destination directory")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json to copy alongside the weights")
    args = ap.parse_args(argv)
    meta = export_training_ckpt(args.ckpt, args.out, tokenizer_json=args.tokenizer)
    print(f"exported {args.ckpt} -> {args.out}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
