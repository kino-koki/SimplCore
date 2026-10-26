# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""在 *真实中文语料* 上跑一次小尺度训练 (验证 SimplCore 训练流水线端到端可用)。

注意：本机为 AMD Radeon 680M + Windows，PyTorch 不支持 AMD 训练；此处用 CPU
做小规模真实训练 (tiny 配置)，目的是验证「数据 → 分词 → 打包 → 前向 → 反向 →
loss 下降」整链路可运行，而非追求收敛质量。生产训练需在 GPU/ROCm 集群进行。

用法：
    python run_real_train.py --data-dir corpus --steps 60 --seq-len 128
输出：
    real_train_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_moe import SimplCoreMoEModel
from simplcore_pg.training.data import (
    SimpleChineseTokenizer, IterableCorpus, PackedDataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="corpus")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-coeff", type=float, default=1.0)
    args = ap.parse_args()

    # 小配置：CPU 可承受，且启用自研 RFP/DMC 架构
    cfg = SimplCoreConfig(
        vocab_size=4096, hidden_size=128, num_layers=3, num_experts=4,
        experts_per_tok=2, moe_intermediate_size=256, max_sequence_length=args.seq_len,
        dmc_num_heads=4, use_rfp=True, use_dmc=True, dropout=0.0,
        moe_aux_loss_coeff=0.01, moe_z_loss_coeff=0.001,
    )
    cfg.validate()

    tokenizer = SimpleChineseTokenizer(vocab_limit=cfg.vocab_size)
    corpus = IterableCorpus(args.data_dir)
    dataset = PackedDataset(corpus, tokenizer, seq_len=args.seq_len, min_seq=32)
    loader = iter(dataset)

    model = SimplCoreMoEModel(cfg).to(torch.device("cpu"))
    n_params = model.num_parameters()
    optimizer = model.configure_optimizers(lr=args.lr)

    model.train()
    losses, auxes, t0 = [], [], time.time()
    grad_ok = True
    for step in range(1, args.steps + 1):
        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(dataset)
            batch = next(loader)
        input_ids = batch["input_ids"].unsqueeze(0).to(torch.long)
        labels = batch["labels"].unsqueeze(0).to(torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, training=True)
        aux = model.last_aux_loss
        ce = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), labels.reshape(-1), ignore_index=-100
        )
        loss = ce + (args.aux_coeff * aux if aux is not None else 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(ce))
        auxes.append(float(aux) if aux is not None else 0.0)
        if not torch.isfinite(loss):
            grad_ok = False
        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            print(f"[step {step}/{args.steps}] ce={float(ce):.4f} "
                  f"aux={float(aux) if aux is not None else 0:.4f} "
                  f"tok/s={step*input_ids.numel()/max(1e-6,dt):.0f}", flush=True)

    first, last = losses[0], losses[-1]
    result = {
        "device": "cpu",
        "config": {"vocab": cfg.vocab_size, "H": cfg.hidden_size,
                   "layers": cfg.num_layers, "experts": cfg.num_experts,
                   "use_rfp": cfg.use_rfp, "use_dmc": cfg.use_dmc,
                   "seq_len": args.seq_len},
        "n_parameters": n_params,
        "steps": args.steps,
        "loss_first": round(first, 4),
        "loss_last": round(last, 4),
        "loss_min": round(min(losses), 4),
        "loss_delta": round(first - last, 4),
        "loss_decreased": bool(first - last > 0.01),
        "aux_first": round(auxes[0], 4),
        "aux_last": round(auxes[-1], 4),
        "all_finite": grad_ok,
        "note": "CPU 小尺度真实语料训练，验证端到端链路；非收敛质量结论。",
    }
    with open("real_train_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n=== 真实训练结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
