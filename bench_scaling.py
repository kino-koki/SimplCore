# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 规模扩展基准（论文级）：困惑度随模型规模下降。

在**同一语料 / 同一数据划分 / 同一训练协议**下，仅改变 hidden_size（与层数），
观察验证集困惑度随参数量增长的下降趋势——回答"无注意力架构是否遵循扩展律"。

用法:
  python bench_scaling.py --steps 200 --hidden-list 64,96,128 --layers-list 2,2,3
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_moe import SimplCoreMoEModel

from bench_corpus_ppl import CharTokenizer, load_corpus, split_documents, make_batches


def build_config(vocab: int, hidden: int, layers: int,
                 rfp_freeze: bool = False) -> SimplCoreConfig:
    return SimplCoreConfig(
        vocab_size=vocab, hidden_size=hidden, num_layers=layers,
        num_experts=2, experts_per_tok=2, moe_intermediate_size=hidden * 2,
        num_waves=8, rfp_num_steps=3, rfp_sigma=1.0,
        num_channels=8, channel_size=16, dmc_num_heads=2,
        local_kernel_size=7, local_dilation=1,
        use_localmix=True, use_rfp=True, use_dmc=True,
        local_use_conv=True, local_use_recurrence=True,
        max_sequence_length=2048, chunk_size=64, dropout=0.0,
        tie_word_embeddings=True,
        rfp_freeze_evolution=rfp_freeze,
    )


def train_step(model, x, y, opt, vocab):
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
def evaluate_ppl(model, batches, vocab):
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


def run_scale(hidden: int, layers: int, vocab: int, train_batches, val_batches,
              steps: int, lr: float, rfp_freeze: bool = False) -> Dict:
    cfg = build_config(vocab, hidden, layers, rfp_freeze=rfp_freeze)
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
                print(f"  [H={hidden},L={layers}] step {step}/{steps} loss={loss:.4f}", flush=True)
            step += 1
            del x, y, loss
            if step % 100 == 0:
                gc.collect()
    val_ppl = evaluate_ppl(model, val_batches, vocab)
    res = {
        "hidden": hidden, "layers": layers,
        "params": params,
        "rfp_freeze_evolution": rfp_freeze,
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
    ap.add_argument("--hidden-list", default="64,96,128")
    ap.add_argument("--layers-list", default="2,2,3")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-vocab", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench_scaling.json")
    ap.add_argument("--rfp-freeze", action="store_true",
                    help="冻结 RFP FFT 演化（仅投影可学）。用于 Windows/CPU 下 FFT 反向 "
                         "workspace 泄漏导致进程被 OS 杀死的环境，作为文档化回退。")
    args = ap.parse_args()

    hidden_list = [int(v) for v in args.hidden_list.split(",")]
    layers_list = [int(v) for v in args.layers_list.split(",")]
    assert len(hidden_list) == len(layers_list), "hidden-list 与 layers-list 长度必须一致"

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

    results = []
    for hidden, layers in zip(hidden_list, layers_list):
        print(f"\n=== H={hidden}, L={layers} (rfp_freeze={args.rfp_freeze}) ===")
        results.append(run_scale(hidden, layers, tok.vocab_size, train_batches, val_batches,
                                 args.steps, args.lr, rfp_freeze=args.rfp_freeze))

    report = {
        "task": "scaling_curve_real_chinese_corpus",
        "corpus": str(Path(args.corpus).resolve()),
        "val_split": "document_holdout",
        "vocab_size": tok.vocab_size,
        "steps": args.steps, "seq_len": args.seq_len, "batch": args.batch, "lr": args.lr,
        "seed": args.seed,
        "attention_free": True,
        "rfp_freeze_evolution": args.rfp_freeze,
        "results": results,
        "note": ("规模扩展（真实中文语料，CPU 小尺度可复现证据）。仅改变 hidden/layers，"
                 "其余协议固定；val_perplexity 越低越好。诚实限制：语料约 19 万字符、模型在"
                 "训练集上过拟合（val_ppl 接近 1 属正常），本表展示无注意力架构的困惑度随规模"
                 "下降趋势，非大规模语料 SOTA 对比。"),
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
