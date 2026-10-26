# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""真实中文语料困惑度对比基准：SimplCore(无注意力) vs 标准 Transformer 基线。

论文级诚实证据：在**真实中文语料**（corpus/cn_*.jsonl，约 19 万字符）上做自回归
语言建模，对比两条架构路径：

  * SimplCore : ``SimplCoreMoEModel``（LocalInteraction + RFP + DMC + MoE，
                完全无注意力 / 无 Q/K/V / 无 RoPE / 无 SDPA）。
  * Baseline  : 标准 decoder-only Transformer（GQA + RoPE + SwiGLU FFN），
                参数量与 SimplCore 对齐（脚本自动报告双方实际参数量）。

实验协议（保证公平，可复现）：
  1. 同一字符级 tokenizer（由语料构建，0=pad / 1=eos / 2+ 字符）；
  2. 同一数据划分（train/val，固定 seed）；
  3. 同一训练协议（步数 / batch / seq_len / AdamW / 梯度裁剪 / 学习率）；
  4. 双方各自报告 train loss 与 **val perplexity**（exp(mean NLL)，忽略 pad）。

诚实声明：本基准为 CPU 小尺度可复现证据，非大规模语料 SOTA 对比。其目的是回答
"无注意力架构在真实数据上能否与标准 Transformer 竞争"这一论文核心问题。

用法:
  python bench_corpus_ppl.py --steps 300 --hidden 128 --layers 4 --seq-len 96
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


# ----------------------------------------------------------------------
# 字符级 tokenizer（由语料构建，0=pad / 1=eos / 2+ 字符）
# ----------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, texts: List[str], max_vocab: int = 4096):
        chars = sorted(set("".join(texts)))
        if len(chars) + 2 > max_vocab:
            chars = chars[: max_vocab - 2]
        self.stoi = {"<pad>": 0, "<eos>": 1}
        self.itos = {0: "<pad>", 1: "<eos>"}
        for i, ch in enumerate(chars, start=2):
            self.stoi[ch] = i
            self.itos[i] = ch
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in ids)


# ----------------------------------------------------------------------
# 标准 Transformer 基线（GQA + RoPE + SwiGLU），纯 torch
# ----------------------------------------------------------------------
def _precompute_rope(head_dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(max_len).float()
    angles = pos[:, None] * inv[None, :]          # [L, d/2]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)  # [L, d]


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, d], freqs: [L, d]（前半 cos、后半 sin）
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    c, s = freqs[..., : d // 2], freqs[..., d // 2 :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class GQAAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.groups = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, freqs[:T])
        k = _apply_rope(k, freqs[:T])
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class TransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = GQAAttention(dim, n_heads, n_kv_heads, dropout)
        self.norm2 = nn.RMSNorm(dim)
        h = ffn_mult * dim
        self.gate = nn.Linear(dim, h, bias=False)
        self.up = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), freqs))
        h = self.norm2(x)
        x = x + self.dropout(self.down(F.silu(self.gate(h)) * self.up(h)))
        return x


class TransformerBaseline(nn.Module):
    """标准 decoder-only Transformer（GQA + RoPE + SwiGLU），与 SimplCore 对齐参数。"""

    def __init__(self, vocab_size: int, dim: int, layers: int, n_heads: int,
                 n_kv_heads: int, ffn_mult: int = 4, dropout: float = 0.0,
                 max_seq_len: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerLayer(dim, n_heads, n_kv_heads, ffn_mult, dropout)
            for _ in range(layers)
        ])
        self.norm_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.register_buffer("freqs", _precompute_rope(dim // n_heads, max_seq_len))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok_emb(x)
        for layer in self.layers:
            h = layer(h, self.freqs)
        return self.lm_head(self.norm_f(h))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------
# SimplCore 配置（无注意力：LocalInteraction + RFP + DMC + MoE）
# ----------------------------------------------------------------------
def build_simplcore_config(vocab: int, hidden: int, layers: int) -> SimplCoreConfig:
    return SimplCoreConfig(
        vocab_size=vocab, hidden_size=hidden, num_layers=layers,
        num_experts=2, experts_per_tok=2, moe_intermediate_size=hidden * 2,
        num_waves=8, rfp_num_steps=3, rfp_sigma=1.0,
        num_channels=8, channel_size=16, dmc_num_heads=2,
        local_kernel_size=7, local_dilation=1, local_use_recurrence=True,
        use_localmix=True, use_rfp=True, use_dmc=True,
        max_sequence_length=2048, chunk_size=64, dropout=0.0,
        tie_word_embeddings=True,
    )


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def load_corpus(corpus_dir: str, max_chars: Optional[int] = None) -> List[str]:
    texts: List[str] = []
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                texts.append(json.loads(line)["text"])
    if max_chars is not None:
        joined = "".join(texts)
        texts = [joined[:max_chars]]
    return texts


def make_batches(ids: List[int], batch_size: int, seq_len: int,
                 rng: random.Random) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """把 token 序列切成 (batch, seq_len+1) 的 (x, y) 对。"""
    n = len(ids) - 1
    n_chunks = max(1, n // seq_len)
    chunks = []
    for i in range(n_chunks):
        start = i * seq_len
        chunk = ids[start : start + seq_len + 1]
        if len(chunk) < seq_len + 1:
            chunk = chunk + [0] * (seq_len + 1 - len(chunk))
        chunks.append(chunk)
    rng.shuffle(chunks)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(chunks), batch_size):
        group = chunks[i : i + batch_size]
        x = torch.tensor([c[:-1] for c in group], dtype=torch.long)
        y = torch.tensor([c[1:] for c in group], dtype=torch.long)
        batches.append((x, y))
    return batches


def split_documents_detailed(texts: List[str], tok: "CharTokenizer", val_frac: float,
                             seed: int) -> Tuple[List[int], List[int], List, List]:
    """文档级划分，返回 (train_ids, val_ids, train_docs, val_docs)。

    整篇文档留出作验证集（避免同文档内容泄漏到训练集）。docs 元素为 (text, ids)。
    """
    rng = random.Random(seed)
    docs = [(t, tok.encode(t)) for t in texts]
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * val_frac))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    train_ids: List[int] = []
    for _, ids in train_docs:
        train_ids.extend(ids)
        train_ids.append(1)  # eos
    val_ids: List[int] = []
    for _, ids in val_docs:
        val_ids.extend(ids)
        val_ids.append(1)
    return train_ids, val_ids, train_docs, val_docs


def split_documents(texts: List[str], tok: "CharTokenizer", val_frac: float,
                    seed: int) -> Tuple[List[int], List[int]]:
    """文档级划分：整篇文档留出作验证集（避免同文档内容泄漏到训练集）。"""
    train_ids, val_ids, _, _ = split_documents_detailed(texts, tok, val_frac, seed)
    return train_ids, val_ids


# ----------------------------------------------------------------------
# 训练 / 评估
# ----------------------------------------------------------------------
def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               opt: torch.optim.Optimizer, vocab: int, use_aux: bool) -> float:
    opt.zero_grad(set_to_none=True)
    logits = model(x, training=True) if use_aux else model(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), ignore_index=0)
    if use_aux and getattr(model, "last_aux_loss", None) is not None:
        loss = loss + model.last_aux_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate_ppl(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor]],
                 vocab: int, use_aux: bool) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for x, y in batches:
        logits = model(x, training=False) if use_aux else model(x)
        logits = logits.reshape(-1, vocab)
        yf = y.reshape(-1)
        nll = F.cross_entropy(logits, yf, reduction="none")
        mask = yf != 0
        total_nll += nll[mask].sum().item()
        total_tokens += mask.sum().item()
    model.train()
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_model(name: str, model: nn.Module, train_batches, val_batches,
              steps: int, lr: float, vocab: int, use_aux: bool,
              log_every: int = 50) -> Dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    losses: List[float] = []
    n_batches = len(train_batches)
    step = 0
    while step < steps:
        for x, y in train_batches:
            if step >= steps:
                break
            loss = train_step(model, x, y, opt, vocab, use_aux)
            losses.append(loss)
            if step % log_every == 0 or step == steps - 1:
                print(f"  [{name}] step {step}/{steps} loss={loss:.4f}", flush=True)
            step += 1
            del x, y, loss
            if step % 100 == 0:
                gc.collect()
    val_ppl = evaluate_ppl(model, val_batches, vocab, use_aux)
    # 下采样损失曲线（最多 ~40 点）供学习曲线图使用
    curve = []
    stride = max(1, len(losses) // 40)
    for i in range(0, len(losses), stride):
        curve.append({"step": i, "loss": round(losses[i], 4)})
    if curve and curve[-1]["step"] != len(losses) - 1:
        curve.append({"step": len(losses) - 1, "loss": round(losses[-1], 4)})
    return {
        "model": name,
        "params": int(sum(p.numel() for p in model.parameters())),
        "final_train_loss": round(losses[-1], 4) if losses else None,
        "val_perplexity": round(val_ppl, 4) if val_ppl == val_ppl else None,
        "train_seconds": round(time.time() - t0, 1),
        "loss_curve": curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus", help="语料目录（*.jsonl）")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--ffn-mult", type=int, default=4, help="基线 FFN 中间维度倍数（用于参数对齐）")
    ap.add_argument("--max-vocab", type=int, default=4096)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bench_corpus_ppl.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    texts = load_corpus(args.corpus)
    print(f"[data] 语料 {len(texts)} 条, 总字符 {sum(len(t) for t in texts)}")

    tok = CharTokenizer(texts, max_vocab=args.max_vocab)
    print(f"[data] 字符词表 {tok.vocab_size}")

    # 数据划分（文档级留出，固定 seed）
    rng = random.Random(args.seed)
    train_ids, val_ids = split_documents(texts, tok, args.val_frac, args.seed)
    print(f"[data] train {len(train_ids)} tokens, val {len(val_ids)} tokens")

    train_batches = make_batches(train_ids, args.batch, args.seq_len, rng)
    val_batches = make_batches(val_ids, args.batch, args.seq_len, rng)
    print(f"[data] train batches {len(train_batches)}, val batches {len(val_batches)}")

    # ---- SimplCore（无注意力）----
    sc_cfg = build_simplcore_config(tok.vocab_size, args.hidden, args.layers)
    sc_model = SimplCoreMoEModel(sc_cfg)
    sc_params = sum(p.numel() for p in sc_model.parameters())
    print(f"[model] SimplCore params={sc_params}")

    # ---- Transformer 基线（参数对齐）----
    n_heads = max(2, args.hidden // 32)
    n_kv_heads = max(1, n_heads // 2)
    base_model = TransformerBaseline(
        vocab_size=tok.vocab_size, dim=args.hidden, layers=args.layers,
        n_heads=n_heads, n_kv_heads=n_kv_heads, ffn_mult=args.ffn_mult, dropout=0.0,
        max_seq_len=args.seq_len + 8,
    )
    base_params = base_model.num_parameters()
    print(f"[model] Baseline params={base_params} (ffn_mult={args.ffn_mult})")
    if base_params > 0:
        print(f"[model] 参数比 SimplCore/Baseline = {sc_params / base_params:.3f}")

    results = []
    print("\n=== 训练 SimplCore（无注意力）===")
    sc_res = run_model("SimplCore", sc_model, train_batches, val_batches,
                       args.steps, args.lr, tok.vocab_size, use_aux=True)
    results.append(sc_res)
    print(f"  => {sc_res}")
    del sc_model
    gc.collect()

    print("\n=== 训练 Baseline（标准 Transformer）===")
    base_res = run_model("Baseline", base_model, train_batches, val_batches,
                         args.steps, args.lr, tok.vocab_size, use_aux=False)
    results.append(base_res)
    print(f"  => {base_res}")

    report = {
        "task": "real_chinese_corpus_language_modeling",
        "corpus": str(Path(args.corpus).resolve()),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "val_split": "document_holdout",
        "vocab_size": tok.vocab_size,
        "steps": args.steps, "hidden": args.hidden, "layers": args.layers,
        "seq_len": args.seq_len, "batch": args.batch, "lr": args.lr,
        "seed": args.seed,
        "attention_free": True,
        "results": results,
        "note": ("真实中文语料困惑度对比（CPU 小尺度可复现证据）。SimplCore 为无注意力"
                 "架构（LocalInteraction+RFP+DMC+MoE），Baseline 为标准 GQA+RoPE+SwiGLU "
                 "Transformer。验证集为文档级留出（整篇文档不进入训练）。诚实限制：语料仅"
                 "约 19 万字符、词表 424，模型在训练集上严重过拟合（val_ppl 接近 1 属正常），"
                 "本对比回答'无注意力架构能否在真实数据上学习并达到与 Transformer 相当或更优"
                 "的泛化'，非大规模语料 SOTA 对比。val_perplexity 越低越好。"),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))

    if len(results) == 2 and results[0]["val_perplexity"] and results[1]["val_perplexity"]:
        sc_ppl = results[0]["val_perplexity"]
        base_ppl = results[1]["val_perplexity"]
        ratio = sc_ppl / base_ppl
        print(f"\nSimplCore val_ppl={sc_ppl:.2f} | Baseline val_ppl={base_ppl:.2f} | "
              f"ratio={ratio:.3f}（<1 表示 SimplCore 更优）")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback as _tb
        _tb.print_exc()
        raise
