# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 组件消融基准（论文级）：真实中文语料上逐组件归因。

在**同一数据划分 / 同一训练协议**下，对比 5 个架构变体（全部无注意力），
回答"每个组件对泛化的贡献"这一论文核心问题：

  * full          : LocalInteraction(卷积+递归) + RFP + DMC + MoE  ← 完整架构
  * no_rfp        : LocalInteraction + DMC + MoE                    ← 去掉全局谱卷积
  * no_dmc        : LocalInteraction + RFP + MoE                    ← 去掉固定记忆
  * no_recurrence : LocalInteraction(仅卷积) + RFP + DMC + MoE      ← 去掉线性递归
  * no_conv       : LocalInteraction(仅递归) + RFP + DMC + MoE      ← 去掉因果卷积

协议与 ``bench_corpus_ppl.py`` 完全一致（同一 tokenizer / 文档级留出 / AdamW /
梯度裁剪 / 验证集困惑度），保证消融差异只来自组件本身。

用法:
  python bench_ablation.py --steps 200 --hidden 96 --layers 2 --seq-len 96
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_moe import SimplCoreMoEModel

from bench_corpus_ppl import CharTokenizer, load_corpus, split_documents, make_batches


# ----------------------------------------------------------------------
# 变体配置：同一基线，逐个开关组件
# ----------------------------------------------------------------------
def build_variant_config(vocab: int, hidden: int, layers: int, variant: str) -> SimplCoreConfig:
    base = dict(
        vocab_size=vocab, hidden_size=hidden, num_layers=layers,
        num_experts=2, experts_per_tok=2, moe_intermediate_size=hidden * 2,
        num_waves=8, rfp_num_steps=3, rfp_sigma=1.0,
        num_channels=8, channel_size=16, dmc_num_heads=2,
        local_kernel_size=7, local_dilation=1,
        use_localmix=True, use_rfp=True, use_dmc=True,
        local_use_conv=True, local_use_recurrence=True,
        max_sequence_length=2048, chunk_size=64, dropout=0.0,
        tie_word_embeddings=True,
    )
    if variant == "no_rfp":
        base["use_rfp"] = False
    elif variant == "no_dmc":
        base["use_dmc"] = False
    elif variant == "no_recurrence":
        base["local_use_recurrence"] = False
    elif variant == "no_conv":
        base["local_use_conv"] = False
    elif variant != "full":
        raise ValueError(f"未知变体 {variant!r}")
    return SimplCoreConfig(**base)


# ----------------------------------------------------------------------
# 训练 / 评估（与 bench_corpus_ppl.py 相同协议）
# ----------------------------------------------------------------------
def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               opt: torch.optim.Optimizer, vocab: int) -> float:
    opt.zero_grad(set_to_none=True)
    logits = model(x, training=True)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), ignore_index=0)
    if getattr(model, "last_aux_loss", None) is not None:
        loss = loss + model.last_aux_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate_ppl(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor]],
                 vocab: int) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for x, y in batches:
        logits = model(x, training=False).reshape(-1, vocab)
        yf = y.reshape(-1)
        nll = F.cross_entropy(logits, yf, reduction="none")
        mask = yf != 0
        total_nll += nll[mask].sum().item()
        total_tokens += mask.sum().item()
    model.train()
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


def run_variant(name: str, variant: str, vocab: int, hidden: int, layers: int,
                train_batches, val_batches, steps: int, lr: float) -> Dict:
    cfg = build_variant_config(vocab, hidden, layers, variant)
    model = SimplCoreMoEModel(cfg)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    losses: List[float] = []
    step = 0
    n_batches = len(train_batches)
    while step < steps:
        for x, y in train_batches:
            if step >= steps:
                break
            loss = train_step(model, x, y, opt, vocab)
            losses.append(loss)
            if step % 50 == 0 or step == steps - 1:
                print(f"  [{name}] step {step}/{steps} loss={loss:.4f}", flush=True)
            step += 1
            del x, y, loss
            if step % 100 == 0:
                gc.collect()
    val_ppl = evaluate_ppl(model, val_batches, vocab)
    res = {
        "variant": name,
        "params": params,
        "final_train_loss": round(losses[-1], 4) if losses else None,
        "val_perplexity": round(val_ppl, 4) if val_ppl == val_ppl else None,
        "train_seconds": round(time.time() - t0, 1),
    }
    print(f"  => {res}", flush=True)
    del model
    gc.collect()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-vocab", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench_ablation.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    texts = load_corpus(args.corpus)
    print(f"[data] 语料 {len(texts)} 条, 总字符 {sum(len(t) for t in texts)}")
    tok = CharTokenizer(texts, max_vocab=args.max_vocab)
    print(f"[data] 字符词表 {tok.vocab_size}")

    rng = random.Random(args.seed)
    train_ids, val_ids = split_documents(texts, tok, args.val_frac, args.seed)
    print(f"[data] train {len(train_ids)} tokens, val {len(val_ids)} tokens")
    train_batches = make_batches(train_ids, args.batch, args.seq_len, rng)
    val_batches = make_batches(val_ids, args.batch, args.seq_len, rng)

    variants = ["full", "no_rfp", "no_dmc", "no_recurrence", "no_conv"]
    results = []
    for v in variants:
        print(f"\n=== 变体 {v} ===")
        results.append(run_variant(v, v, tok.vocab_size, args.hidden, args.layers,
                                   train_batches, val_batches, args.steps, args.lr))

    # 相对完整模型的困惑度比（<1 表示该组件反而有害；>1 表示该组件有益）
    full_ppl = next((r["val_perplexity"] for r in results if r["variant"] == "full"), None)
    for r in results:
        if full_ppl and r["val_perplexity"]:
            r["ppl_ratio_vs_full"] = round(r["val_perplexity"] / full_ppl, 4)

    report = {
        "task": "component_ablation_real_chinese_corpus",
        "corpus": str(Path(args.corpus).resolve()),
        "val_split": "document_holdout",
        "vocab_size": tok.vocab_size,
        "steps": args.steps, "hidden": args.hidden, "layers": args.layers,
        "seq_len": args.seq_len, "batch": args.batch, "lr": args.lr,
        "seed": args.seed,
        "attention_free": True,
        "results": results,
        "note": ("组件消融（真实中文语料，CPU 小尺度可复现证据）。全部变体无注意力；"
                 "同一数据划分/训练协议。ppl_ratio_vs_full>1 表示移除该组件后困惑度上升"
                 "（组件有益），<1 表示移除后反而更好（组件在该尺度下冗余）。诚实限制："
                 "语料约 19 万字符、模型在训练集上过拟合（val_ppl 接近 1 属正常），本表回答"
                 "'各组件相对贡献'，非大规模语料 SOTA 对比。"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        raise
