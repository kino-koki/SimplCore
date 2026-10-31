#!/usr/bin/env python
# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""30M 对话能力训练入口（SFT）。

一条命令完成：词表 → 特殊 token → 掩码 → 打包 → 训练 → 对话评测 → 报告。

    # 真机（1 x A100，30M，约 0.6B token 中文对话）
    python scripts/train_30m_dialogue.py --preset tuan-30m \
        --train-jsonl data/sft_zh.jsonl --seq-len 1024 \
        --max-steps 20000 --batch 32 --lr 6e-4 --device cuda

    # 本机 CPU 管线自证（几分钟，验证链路与格式学习，不代表能力）
    python scripts/train_30m_dialogue.py --smoke

注意：``--smoke`` 用合成语料 + 缩小模型，只证明"链路通、格式学得会"。
它**不是**"30M 会对话"的证据，两者的区别见
``docs/DIALOGUE_30M_PLAN.md``。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplcore.core.chat import (ASSISTANT_TOKEN, END_TOKEN, PAD_TOKEN,
                               ChatTemplate, SFTDataset, SpecialTokenCodec,
                               build_memory_encoder, collate_sft)
from simplcore.core.config import SimplCoreConfig, analytic_params, get_preset
from simplcore.core.dialogue_eval import (BASIC_DIALOGUE_SUITE,
                                        CHAT_DIALOGUE_SUITE,
                                        MEMORY_DIALOGUE_SUITE, DialogueReport,
                                        evaluate_dialogue, format_report,
                                        memory_ablation,
                                        memory_projection_report)
from simplcore.core.model import SimplCoreModel
from simplcore.core.scaling import (cost_ledger, embedding_share, gpu_hours,
                                   non_embedding_params, train_flops_per_token)
from simplcore.core.tokenization import CharVocab
from simplcore.core.train import build_param_groups, cosine_with_warmup, save_checkpoint


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------
def prepare_sft_file(path: str, n: int, seed: int) -> str:
    from simplcore.core.dialogue_data import build_records, write_jsonl
    recs = build_records(n, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_jsonl(path, recs)
    return path


def build_vocab(path: str, max_size: int, max_records: Optional[int] = None) -> CharVocab:
    """从**训练文件**统计字符词表（绝不看验证集：那是最常见的静默泄漏）。"""
    texts: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for m in rec.get("messages", []):
                texts.append(str(m.get("content", "")))
    # min_freq=1：对话语料里人名/符号往往只出现几次，砍掉会让「多轮指代」失败
    return CharVocab.train(texts, min_freq=1, max_size=max_size)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def _unpack(batch):
    """DataLoader 可能给 ``(x, y)`` 或 ``(x, y, mem)`` —— 记忆是可选通道。

    必须以**整个 batch** 为参数。早先写成 ``_unpack(x, y)`` 时，第三项永远
    取不到，``mem`` 恒为 None —— 带记忆的训练会静默退化成不带记忆，
    而 loss 曲线、形状、参数量全都正常。这类错只能靠签名本身来防。
    """
    x, y = batch[0], batch[1]
    mem = batch[2] if len(batch) > 2 else None
    return x, y, mem


@torch.no_grad()
def evaluate_loss(model, loader, device, max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    nll, ntok = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, mem = _unpack(batch)
        x, y = x.to(device), y.to(device)
        mem = mem.to(device) if mem is not None else None
        _, loss = model(x, y, memory=mem) if mem is not None else model(x, y)
        if torch.isfinite(loss):
            k = int((y != -100).sum())
            nll += float(loss) * k
            ntok += k
    model.train()
    mean = nll / max(ntok, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "tokens": ntok}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SGC 对话 SFT（30M 级）")
    ap.add_argument("--preset", default="tuan-30m")
    ap.add_argument("--target-params", type=float, default=None,
                    help="按预算自动切分（覆盖 --preset 的 dim/layers）")
    ap.add_argument("--dim", type=int, default=None, help="覆盖宽度")
    ap.add_argument("--layers", type=int, default=None, help="覆盖深度")
    ap.add_argument("--mixer", choices=("sgc", "srf", "attn"), default=None,
                    help="覆盖 mixer 种类；attn 即同骨架 Transformer 基线，"
                         "srf 即谱振流网络（selective SSM）")
    ap.add_argument("--ffn-hidden", type=int, default=None,
                    help="覆盖 SwiGLU 隐藏宽度（用于等参数量对齐基线）")
    ap.add_argument("--reason-attn-every", type=int, default=None,
                    help="每 N 层挂一个推理注意力（None/0 = 关闭）")
    ap.add_argument("--srf-heads", type=int, default=None,
                    help="srf mixer 的 Delta 头数")
    ap.add_argument("--srf-chunk", type=int, default=None,
                    help="srf chunked scan 的块长")
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="生成 N 条合成对话（仅用于管线自证）")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU 自证：合成数据 + 缩小模型 + 少量步数")
    ap.add_argument("--out-dir", default="runs/sft30m")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--lr-min", type=float, default=6e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="训练步数；--smoke 下默认 300，显式传入则以其为准")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--vocab-size", type=int, default=12288)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--system", default=None, help="默认 system 提示（默认不注入）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--use-memory", action="store_true",
                    help="开启 CMT 记忆注入（数据需带 memory 字段，见 dialogue_data.py）")
    ap.add_argument("--memory-dim", type=int, default=None,
                    help="记忆向量维度，默认等于模型宽度")
    ap.add_argument("--memory-alpha", type=float, default=1.0,
                    help="记忆编码器的 df 权重指数（1.0 = 1/df，模板成分压到最小）")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1. 数据 ---------------------------------------------------------
    if args.smoke:
        # CPU 自证规模：目标 ~2 分钟/100 步。参数刻意压到 1M 量级 ——
        # 这里要证明的是"掩码 + 停止符监督能把格式学出来"，不是能力。
        args.synthetic = args.synthetic or 6000
        # 只在**没显式指定**时套用 smoke 的 300 步；
        # 早先写成 min(args.max_steps, 300)，结果 --max-steps 800 静默变成
        # 300 —— 报告上写着"800 步"的预期，实际跑了 300，差 2.7 倍算力。
        args.max_steps = args.max_steps or 300
        args.seq_len = min(args.seq_len, 128)
        args.batch = min(args.batch, 16)
        args.vocab_size = min(args.vocab_size, 4096)
        args.eval_every = min(args.eval_every, 100)
        args.log_every = min(args.log_every, 25)
        args.dim = args.dim or 256
        args.layers = args.layers or 4
        args.warmup = min(args.warmup, 30)
    if args.train_jsonl is None:
        if not args.synthetic:
            raise SystemExit("需要 --train-jsonl，或 --synthetic N / --smoke")
        args.train_jsonl = prepare_sft_file(
            os.path.join(args.out_dir, "synthetic_sft.jsonl"),
            args.synthetic, args.seed)

    # ---- 2. 词表 ---------------------------------------------------------
    vocab = build_vocab(args.train_jsonl, args.vocab_size)
    vocab_path = os.path.join(args.out_dir, "charvocab.json")
    vocab.save(vocab_path)
    codec = SpecialTokenCodec(vocab)
    template = ChatTemplate(system=args.system)
    print(f"[data] 词表 {vocab.size()} 字符 + {len(codec.specials)} 特殊 = {codec.size()}")

    n_records = sum(1 for _ in open(args.train_jsonl, encoding="utf-8"))
    n_val = max(1, int(n_records * args.val_frac))
    n_train = max(1, n_records - n_val)
    # 记忆维度必须在建数据集之前确定（它要编码进每一行）。
    # 用 preset 的宽度或显式 ``--dim``；``--target-params`` 的宽度要等
    # 规划器跑完才知道，那种情况直接拒绝，避免两处 dim 不一致。
    mem_dim = 0
    if args.use_memory:
        if args.memory_dim:
            mem_dim = int(args.memory_dim)
        elif args.dim:
            mem_dim = int(args.dim)
        elif args.target_params is not None:
            raise SystemExit("--use-memory 与 --target-params 同时用时必须给 --memory-dim")
        else:
            mem_dim = int(get_preset(args.preset).dim)
    _coll = collate_sft if args.use_memory else None

    # 编码器只建一次，train / val / 评测共用。各自 fit 会得到两份不同的
    # df 统计，同一段记忆在训练和评测里被编码成不同向量 —— 形状对、loss
    # 照降，但消融数字没有意义。
    mem_enc = None
    if args.use_memory:
        mem_enc = build_memory_encoder(args.train_jsonl, mem_dim,
                                       alpha=args.memory_alpha)
        print(f"[memory] encoder dim={mem_dim} docs={mem_enc.state_dict()['n_docs']} "
              f"grams={len(mem_enc.state_dict()['df'])} "
              f"alpha={mem_enc.state_dict()['alpha']} "
              f"fp={mem_enc.fingerprint()[:12]}")

    train_ds = SFTDataset(args.train_jsonl, codec, template, args.seq_len,
                          max_len=args.seq_len, index_slice=(0, n_train),
                          memory_dim=mem_dim, memory_encoder=mem_enc)
    val_ds = SFTDataset(args.train_jsonl, codec, template, args.seq_len,
                        max_len=args.seq_len, index_slice=(n_train, n_records),
                        memory_dim=mem_dim, memory_encoder=mem_enc)
    print(f"[data] train {train_ds.stats()}")
    print(f"[data] val   {val_ds.stats()}")
    if args.use_memory:
        if not train_ds.with_memory:
            raise SystemExit(
                "--use-memory 但数据里没有任何 memory 字段。"
                "用 dialogue_data.py 的 xsess_* 任务，或在自己的数据里加 memory。")
        if mem_dim != int(args.dim or get_preset(args.preset).dim):
            raise SystemExit(f"记忆维度 {mem_dim} 与模型宽度不一致")

    # ---- 3. 模型 ---------------------------------------------------------
    if args.target_params is not None:
        from simplcore.core.scaling import plan_budget
        cfg = plan_budget(int(args.target_params), top_k=1)[0].cfg
    else:
        cfg = get_preset(args.preset)
    cfg = SimplCoreConfig.from_dict(cfg.to_dict())
    if args.mixer:
        cfg.mixer = args.mixer
        if args.mixer == "attn":
            # 基线对齐：head_dim 固定 64，避免 heads 数随宽度漂移
            cfg.num_heads = max(1, cfg.dim // 64)
    if args.ffn_hidden:
        cfg.ffn_hidden = int(args.ffn_hidden)
    if args.srf_heads:
        cfg.srf_heads = int(args.srf_heads)
    if args.srf_chunk:
        cfg.srf_chunk = int(args.srf_chunk)
    if args.reason_attn_every is not None:
        cfg.reason_attn_every = int(args.reason_attn_every) or None
        if cfg.reason_attn_every and cfg.num_kv_heads is None:
            cfg.num_kv_heads = max(1, cfg.num_heads // 4)
    if args.dim:
        cfg.dim = int(args.dim)
        cfg.num_heads = max(1, cfg.dim // 64) if cfg.mixer == "attn" else cfg.num_heads
    if args.layers:
        cfg.num_layers = int(args.layers)
    if args.smoke:
        cfg.num_bases = min(cfg.num_bases, 16)
        cfg.decay_max = min(cfg.decay_max, 1024)
    cfg.vocab_size = codec.size()
    cfg.use_memory = bool(args.use_memory)
    model = SimplCoreModel(cfg).to(device)
    n_params = model.num_params()
    tag = args.preset if (args.dim is None and args.layers is None) else (
        f"{args.preset}@d{cfg.dim}L{cfg.num_layers}")
    if args.mixer or args.ffn_hidden:
        tag += f"@{cfg.mixer}" + (f"-h{cfg.hidden}" if args.ffn_hidden else "")
    ledger = cost_ledger(cfg, args.seq_len, tag)
    print(f"[model] params={n_params:,}  non-emb={non_embedding_params(cfg):,}  "
          f"emb={embedding_share(cfg)*100:.1f}%  "
          f"train={ledger.train_flops_per_token/1e9:.3f} GFLOPs/token  "
          f"kv_free={ledger.is_kv_free}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=0, collate_fn=_coll)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=0, collate_fn=_coll)

    # ---- 4. 训练前基线 ----------------------------------------------------
    t0 = time.time()
    before = evaluate_dialogue(model, codec, BASIC_DIALOGUE_SUITE, template)
    print("\n[baseline 训练前]")
    print(format_report(before))

    # ---- 5. 训练 ---------------------------------------------------------
    optim = torch.optim.AdamW(build_param_groups(model, args.weight_decay),
                              lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    total_tokens = 0
    best_val = float("inf")
    run = {"config": cfg.to_dict(), "params": n_params,
           "ledger": ledger.as_dict(), "smoke": bool(args.smoke),
           "train_stats": train_ds.stats(), "val_stats": val_ds.stats(),
           "baseline": before.summary(), "curve": []}
    step = 0
    model.train()
    batch_iter = iter(train_loader)
    t_log = time.perf_counter()
    seen_tokens = 0
    stop = False
    args.max_steps = args.max_steps or 3000
    while step < args.max_steps and not stop:
        lr = cosine_with_warmup(step, args.max_steps, args.warmup, args.lr, args.lr_min)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_acc, tok_acc = 0.0, 0
        for _ in range(max(1, args.grad_accum)):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)
            x, y, mem = _unpack(batch)
            x, y = x.to(device), y.to(device)
            mem = mem.to(device) if mem is not None else None
            _, loss = (model(x, y, memory=mem) if mem is not None
                       else model(x, y))
            if not torch.isfinite(loss):
                continue          # 整批都被掩码（极短样本）时 loss 为 nan
            (loss / max(1, args.grad_accum)).backward()
            loss_acc += float(loss.detach())
            tok_acc += int((y != -100).sum())
        if tok_acc == 0:
            step += 1
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        step += 1
        total_tokens += tok_acc
        seen_tokens += tok_acc

        if step % args.log_every == 0 or step == 1:
            now = time.perf_counter()
            span = max(now - t_log, 1e-9)
            rec = {"step": step, "loss": round(loss_acc / max(1, args.grad_accum), 4),
                   "lr": lr, "tokens": total_tokens,
                   "tok_per_s": round(seen_tokens / span, 1),
                   "elapsed_s": round(time.time() - t0, 1)}
            print("[train] " + json.dumps(rec, ensure_ascii=False))
            run["curve"].append(rec)
            seen_tokens, t_log = 0, now

        if step % args.eval_every == 0:
            ev = evaluate_loss(model, val_loader, device)
            print(f"[eval] step={step} val_loss={ev['loss']:.4f} ppl={ev['ppl']:.2f}")
            run["curve"].append({"step": step, "val_loss": round(ev["loss"], 4),
                                 "val_ppl": round(ev["ppl"], 4)})
            if ev["loss"] < best_val:
                best_val = ev["loss"]
                save_checkpoint(os.path.join(args.out_dir, "best.pt"), model,
                                optim, step, total_tokens, cfg, best_val,
                                time.time() - t0)

        if args.time_budget_min > 0 and (time.time() - t0) / 60 >= args.time_budget_min:
            stop = True

    # ---- 6. 训练后评测 ----------------------------------------------------
    after = evaluate_dialogue(model, codec, BASIC_DIALOGUE_SUITE, template)
    print("\n[训练后]")
    print(format_report(after))

    # 开放问题：题目都不在训练语料里，检验"换个问法还会不会接话"。
    chat = evaluate_dialogue(model, codec, CHAT_DIALOGUE_SUITE, template)
    print("\n[开放问题]")
    print(format_report(chat))
    save_checkpoint(os.path.join(args.out_dir, "last.pt"), model, optim, step,
                    total_tokens, cfg, best_val, time.time() - t0)

    # 记忆通道消融：同一 checkpoint、同一套题，只切"给不给记忆"。
    # 只有这张表能说明 memory_in 到底学到了东西 —— 基础套件的通过率
    # 完全可能是靠参数里的文本能力撑起来的，与记忆无关。
    ablation = None
    projection = None
    if args.use_memory:
        # 先判"信息有没有在路上丢掉"：投影塌缩时加步数毫无意义。
        probe = [t.memory for t in MEMORY_DIALOGUE_SUITE if t.memory]
        projection = memory_projection_report(model, probe, encoder=mem_enc)
        print("\n[记忆投影] " + json.dumps({k: v for k, v in projection.items()
                                            if k != "supported"},
                                           ensure_ascii=False))
        ablation = memory_ablation(model, codec, MEMORY_DIALOGUE_SUITE,
                                   template, memory_encoder=mem_enc)
        print("\n[记忆消融] " + ablation["verdict"])
        for r in ablation["per_task"]:
            print(f"  {r['id']:<12} 记忆 {'OK ' if r['with_memory'] else 'FAIL'}"
                  f"  无记忆 {'OK ' if r['without_memory'] else 'FAIL'}"
                  f"  {r['reply_with']!r}")
        print(f"  记忆题 {ablation['memory_pass_without']:.2f} -> "
              f"{ablation['memory_pass_with']:.2f}"
              f"（n={ablation['n_memory_tasks']}，"
              f"翻正={ablation['n_flipped_on']} 翻负={ablation['n_flipped_off']}）")
        print(f"  对照题（该拒答的）{ablation['control_pass_without']:.2f} -> "
              f"{ablation['control_pass_with']:.2f}"
              f"（n={ablation['n_control_tasks']}）")
        with open(os.path.join(args.out_dir, "memory_ablation.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(ablation, fh, ensure_ascii=False, indent=1)
        # 编码器统计必须随 checkpoint 落盘：换一台机器做推理时，
        # 用未拟合的编码器去配已拟合训练出来的 memory_in，向量分布不匹配。
        with open(os.path.join(args.out_dir, "memory_encoder.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(mem_enc.state_dict(), fh, ensure_ascii=False)

    # ---- 7. 报告 ---------------------------------------------------------
    flops = train_flops_per_token(cfg, args.seq_len) * total_tokens
    run.update({
        "steps": step,
        "tokens_trained": total_tokens,
        "wall_s": round(time.time() - t0, 1),
        "best_val_loss": None if best_val == float("inf") else round(best_val, 4),
        "train_flops": flops,
        "a100_hours": round(gpu_hours(flops), 6),
        "after": after.summary(),
        "delta": {
            "task_pass_rate": round(after.task_pass_rate - before.task_pass_rate, 4),
            "stop_rate": round(after.stop_rate - before.stop_rate, 4),
            "turn_leak_rate": round(after.turn_leak_rate - before.turn_leak_rate, 4),
        },
        "after_tasks": after.as_dict()["tasks"],
        "chat": chat.summary(),
        "chat_tasks": chat.as_dict()["tasks"],
        "memory_ablation": ablation,
        "memory_projection": projection,
        "memory_encoder_fingerprint": (mem_enc.fingerprint()
                                       if mem_enc is not None else None),
    })
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print(f"params={n_params/1e6:.2f}M  steps={step}  tokens={total_tokens:,}  "
          f"wall={run['wall_s']}s")
    print(f"task_pass {before.task_pass_rate:.3f} -> {after.task_pass_rate:.3f} | "
          f"stop {before.stop_rate:.3f} -> {after.stop_rate:.3f} | "
          f"leak {before.turn_leak_rate:.3f} -> {after.turn_leak_rate:.3f}")
    if args.smoke:
        print("注：--smoke 只证明链路与格式学习，不构成对话能力证据。")
    print(f"报告：{os.path.join(args.out_dir, 'report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
