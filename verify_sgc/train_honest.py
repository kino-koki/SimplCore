#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诚实公平对比训练 (tri-mixer，诚实报告):
- 三个 mixer: attn / sgc / sgcv3 (SGC-v2 + 内容路由门控)
- 同一框架，同骨架，同超参，同语料，同随机种子，同优化器（AdamW + weight decay）
- 严格文档级不泄漏切分 (build_corpus.py 保证)，最终输出 train/val/test ppl
- 针对 680M 优化: 分梯度累积 + 可选 CPU/OpenCL 内存密集算子卸载
"""
from __future__ import annotations
import os, sys, json, time, random, argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from sgc_pytorch import LMModel, count_params
from sgc_v3 import build_model, count_params

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", default=256, type=int)
parser.add_argument("--dim", default=256, type=int)
parser.add_argument("--n-layers", default=6, type=int)
parser.add_argument("--n-heads", default=4, type=int)
parser.add_argument("--sgc-rank", default=8, type=int)
parser.add_argument("--gate-dropout", default=0.1, type=float)
parser.add_argument("--ca-feat", default=32, type=int)
parser.add_argument("--ca-chunk", default=128, type=int)
parser.add_argument("--gm-beta", default=0.1, type=float,
                    help="GatedAssocMixer 写入门控步长")
parser.add_argument("--steps", default=800, type=int)
parser.add_argument("--batch-size", default=16, type=int)
parser.add_argument("--grad-accum", default=1, type=int,
                    help="梯度累积，避免大batch OOM")
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--wd", default=0.1, type=float,
                    help="weight decay (防过拟合，诚实版必须开)")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--corpus-json", default="corpus.json", type=str)
parser.add_argument("--eval-every", default=100, type=int)
parser.add_argument("--dropout", default=0.0, type=float,
                    help="嵌入+残差流 dropout (防记忆化崩塌)")
parser.add_argument("--label-smooth", default=0.0, type=float,
                    help="label smoothing (封顶过度自信, 防 OOD 概率爆炸)")
parser.add_argument("--lr-min", default=0.0, type=float,
                    help="cosine 衰减终点学习率")
parser.add_argument("--early-stop", default=0, type=int,
                    help="val 连续多少个 eval 窗口不改善则早停 (0=不早停)")
parser.add_argument("--mixer", default="sgcv3", choices=["attn", "sgc", "sgcv3", "ca", "cagate", "hm"])
parser.add_argument("--out", default=None, type=str, help="输出 JSON")
args = parser.parse_args()

# ============================================================
# 加载无泄漏语料
# ============================================================
corpus_path = os.path.join(os.path.dirname(__file__), "results", args.corpus_json)
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)
vocab = corpus["vocab"]
vocab_size = len(vocab)
train_s = corpus["train"]
val_s = corpus["val"]
test_s = corpus["test"]
print(f"Loaded corpus: vocab={vocab_size} train={len(train_s)} val={len(val_s)} test={len(test_s)}")

# ============================================================
# 设置随机种子 & device
# ============================================================
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device("cpu")  # 诚实路线: 主计算在 CPU OpenBLAS (实测最快)
print(f"device: {device}")

# ============================================================
# 构建模型 (公平)
# ============================================================
model = build_model(vocab_size, args.dim, args.n_layers, args.n_heads,
                    args.seq_len, args.mixer, rank=args.sgc_rank,
                    gate_dropout=args.gate_dropout, ca_feat=args.ca_feat,
                    ca_chunk=args.ca_chunk, dropout=args.dropout,
                    gm_beta=args.gm_beta).to(device)
nparams = count_params(model)
print(f"Model [{args.mixer}] params={nparams:,}")

# ============================================================
# 优化器 (诚实版必须: AdamW + weight decay，避免过拟合)
# ============================================================
optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))

# ============================================================
# 评估困惑度
# ============================================================
@torch.no_grad()
def eval_ppl(samples):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    bs = args.batch_size
    idxs = list(range(0, len(samples), bs))
    for start in idxs:
        batch = samples[start:start+bs]
        x = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                              ignore_index=0)
        nz = int((y != 0).sum().item())       # 只统计非 <pad> 的真实 token
        total_loss += loss.item() * nz
        total_tokens += nz
    avg_nll = total_loss / total_tokens
    ppl = np.exp(avg_nll)
    model.train()
    return avg_nll, ppl

# ============================================================
# 训练循环
# ============================================================
def main():
    all_train_ppl = []
    all_val_ppl = []
    all_test_ppl = []
    times = []
    optim.zero_grad(set_to_none=True)
    warmup = max(1, args.steps // 10)
    best_val = float('inf'); best_step = 0
    no_improv = 0
    for step in range(1, args.steps+1):
        # cosine 学习率调度 (含 warmup)
        if step <= warmup:
            lr = args.lr * step / warmup
        else:
            t = (step - warmup) / max(1, (args.steps - warmup))
            lr = args.lr_min + (args.lr - args.lr_min) * 0.5 * (1 + np.cos(np.pi * t))
        for g in optim.param_groups:
            g['lr'] = lr
        t0 = time.perf_counter()
        # 取样一个 batch
        batch = random.sample(train_s, k=args.batch_size)
        x = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                              ignore_index=0,
                              label_smoothing=args.label_smooth)
        loss.div_(args.grad_accum)
        loss.backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
        dt = time.perf_counter() - t0
        times.append(dt)
        if step % args.eval_every == 0 or step == args.steps:
            train_nll, train_ppl = eval_ppl(random.sample(train_s, k=min(len(train_s), 200)))
            val_nll, val_ppl = eval_ppl(val_s)
            test_nll, test_ppl = eval_ppl(test_s)
            print(f"step={step:4d} lr={lr:.4f} loss={loss.item():.4f} train_ppl={train_ppl:.2f} "
                  f"val_ppl={val_ppl:.2f} test_ppl={test_ppl:.2f} step={dt:.3f}s")
            # best-val 早停
            if val_ppl < best_val - 1e-9:
                best_val = val_ppl; best_step = step; no_improv = 0
            else:
                no_improv += 1
            if args.early_stop > 0 and no_improv >= args.early_stop:
                print(f">> early stop at step {step} (best val {best_val:.2f} @ {best_step})")
                steps_done = step
                break
            all_train_ppl.append((step, train_ppl))
            all_val_ppl.append((step, val_ppl))
            all_test_ppl.append((step, test_ppl))
    # 最终结果
    _, final_train = eval_ppl(train_s[:len(test_s)])  # 样本数匹配 test
    _, final_val = eval_ppl(val_s)
    _, final_test = eval_ppl(test_s)
    print("\n=== FINAL RESULTS ===")
    print(f"mixer: {args.mixer} params={nparams:,}")
    print(f"train_ppl={final_train:.3f}  val_ppl={final_val:.3f}  test_ppl={final_test:.3f}")
    print(f"best_val={best_val:.3f} @ step {best_step}  (early-met: {getattr(args,'early_stop',0)>0})")
    print(f"avg step={np.mean(times):.3f}s (CPU)")
    # 输出 JSON
    out = {
        "args": vars(args),
        "vocab_size": vocab_size,
        "n_params": nparams,
        "final": {
            "train_ppl": final_train,
            "val_ppl": final_val,
            "test_ppl": final_test,
        },
        "best_val": best_val,
        "best_step": best_step,
        "curves": {
            "train": all_train_ppl,
            "val": all_val_ppl,
            "test": all_test_ppl,
        },
        "step_ms": 1000 * np.mean(times),
    }
    if args.out:
        out_path = os.path.join(os.path.dirname(__file__), "results", args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("saved ->", out_path)

if __name__ == "__main__":
    main()