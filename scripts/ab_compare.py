#!/usr/bin/env python
# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""受控 A/B：SGC vs attention，同数据 / 同步数 / 同评测。

这是"1B 抵 3B~7B"这类主张**唯一**能落地的证据形式。脚本刻意把两侧
所有能动的东西都钉死：

=========================  ====================================================
固定项                      做法
=========================  ====================================================
数据                       同一份 JSONL、同一个 train/val 切分（按行号切）
词表                       同一份 ``CharVocab``（从训练段统计，绝不用验证段）
模板 / 掩码 / 打包          同一个 ``ChatTemplate`` + ``SFTDataset``
超参                       同一个 ``SFTConfig`` 实例
初始权重                   同 seed；两侧共享 embedding 初始化不被使用，只保证可比
评测                       同一套 ``BASIC_DIALOGUE_SUITE``，贪心解码
=========================  ====================================================

**唯一变化的是 mixer**。

    # CPU 自证（几分钟，验证框架与统计）
    python scripts/ab_compare.py --smoke

    # 真机（30M，A/B 各 20k 步）
    python scripts/ab_compare.py --preset-a tuan-30m --preset-b attn-30m \\
        --train-jsonl data/sft_zh.jsonl --max-steps 20000 --batch 32 --device cuda

输出 ``runs/ab/report.json``，含逐题翻转、配对 bootstrap 区间、McNemar
精确检验，以及两边真实的 token / FLOPs / 参数量 —— 因为"等步数"不等于
"等算力"，这一点必须在报告里看得见，而不是藏在脚注。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplcore.core.ab import PairedOutcome, ab_report, param_efficiency
from simplcore.core.chat import ChatTemplate, SFTDataset, SpecialTokenCodec, collate_sft
from simplcore.core.config import SimplCoreConfig, analytic_params, get_preset
from simplcore.core.dialogue_eval import (BASIC_DIALOGUE_SUITE, DialogueReport,
                                         evaluate_dialogue, format_report)
from simplcore.core.model import SimplCoreModel
from simplcore.core.scaling import (embedding_share, gpu_hours, non_embedding_params,
                                   train_flops_per_token)
from simplcore.core.sft import (SFTConfig, build_optimizer, evaluate_loss,
                              fairness_report, train_sft)
from simplcore.core.tokenization import CharVocab
from simplcore.core.train import save_checkpoint


# ---------------------------------------------------------------------------
def build_vocab(path: str, max_size: int, n_train: Optional[int] = None):
    """从**训练段**统计字符词表。看验证段是最常见的静默泄漏。"""
    texts: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if n_train is not None and i >= n_train:
                break
            line = line.strip()
            if not line:
                continue
            for m in json.loads(line).get("messages", []):
                texts.append(str(m.get("content", "")))
    return CharVocab.train(texts, min_freq=1, max_size=max_size)


def make_model(preset: str, codec_size: int, dim: Optional[int] = None,
               layers: Optional[int] = None, use_memory: bool = False,
               seed: int = 1337) -> Tuple[SimplCoreModel, SimplCoreConfig]:
    cfg = SimplCoreConfig.from_dict(get_preset(preset).to_dict())
    if dim:
        cfg.dim = int(dim)
        if cfg.mixer == "attn":
            cfg.num_heads = max(1, cfg.dim // 64)
    if layers:
        cfg.num_layers = int(layers)
    cfg.vocab_size = int(codec_size)
    cfg.use_memory = bool(use_memory)
    torch.manual_seed(seed)
    return SimplCoreModel(cfg), cfg


def _preset_cfg(preset: str, codec_size: int, layers: Optional[int],
                dim: Optional[int]) -> SimplCoreConfig:
    """按 preset + 覆盖项构造配置。

    ``dim`` 必须参与：``--dim`` 会改写宽度，若层数搜索仍按 preset 的
    原宽度算参数，搜索出的层数会**系统性偏大**，两边参数量对不上，
    而报告里只会看到"参数量差 9%"这种不明所以的告警。
    """
    cfg = SimplCoreConfig.from_dict(get_preset(preset).to_dict())
    if dim:
        cfg.dim = int(dim)
        if cfg.mixer == "attn":
            cfg.num_heads = max(1, cfg.dim // 64)
    cfg.vocab_size = int(codec_size)
    if layers:
        cfg.num_layers = int(layers)
    return cfg


def params_for_layers(preset: str, codec_size: int, layers: int,
                      dim: Optional[int] = None) -> int:
    return analytic_params(_preset_cfg(preset, codec_size, layers, dim))


def match_params(preset: str, codec_size: int, target: int,
                 dim: Optional[int] = None, lo: int = 1, hi: int = 64) -> int:
    """搜索使参数量最接近 ``target`` 的层数（用于 ``--match params``）。"""
    best, best_err = lo, None
    for L in range(lo, hi + 1):
        err = abs(params_for_layers(preset, codec_size, L, dim) - target)
        if best_err is None or err < best_err:
            best, best_err = L, err
    return best


# ---------------------------------------------------------------------------
def run_arm(name: str, args, codec, template, train_ds, val_ds, device,
            layers: Optional[int]) -> Dict[str, Any]:
    model, cfg = make_model(args.preset if name == "a" else args.preset_b,
                            codec.size(), args.dim, layers,
                            use_memory=args.use_memory, seed=args.seed)
    model = model.to(device)
    n_params = model.num_params()
    assert n_params == analytic_params(cfg), (
        f"{name}: 实例化参数 {n_params} 与分析值 {analytic_params(cfg)} 不一致")

    tl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=0,
        collate_fn=collate_sft if train_ds.with_memory else None)
    vl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=0,
        collate_fn=collate_sft if val_ds.with_memory else None)

    # attention baseline 没有 O(1) 流式状态，只能整段重算。这是它真实的
    # 推理代价，不是评测代码的缺陷 —— 但对 A/B 来说是**必须一致**的部分：
    # 两侧都用"同一解码策略"，只是执行路径不同。
    use_state = cfg.mixer not in ("attn",)
    before = evaluate_dialogue(model, codec, BASIC_DIALOGUE_SUITE, template,
                               use_state=use_state)
    print(f"\n[{name}] 训练前 {before.summary()}")

    optim = build_optimizer(model, args.sft)
    t0 = time.time()
    res = train_sft(
        model, tl, vl, args.sft, device, optimizer=optim,
        on_log=lambda r: print(f"[{name}] " + json.dumps(r, ensure_ascii=False)),
        on_eval=lambda s, ev: print(f"[{name}] eval step={s} "
                                    f"val_loss={ev['loss']:.4f} ppl={ev['ppl']:.2f}"),
    )
    after = evaluate_dialogue(model, codec, BASIC_DIALOGUE_SUITE, template,
                              use_state=use_state)
    print(f"[{name}] 训练后 {after.summary()}")

    flops = train_flops_per_token(cfg, args.seq_len) * res.tokens
    out = {
        "name": name,
        "preset": args.preset if name == "a" else args.preset_b,
        "mixer": cfg.mixer,
        "params": n_params,
        "non_emb_params": non_embedding_params(cfg),
        "embedding_share": round(embedding_share(cfg), 4),
        "dim": cfg.dim,
        "layers": cfg.num_layers,
        "use_memory": bool(getattr(cfg, "use_memory", False)),
        "train_flops_per_token": train_flops_per_token(cfg, args.seq_len),
        "train_flops": flops,
        "a100_hours": round(gpu_hours(flops), 6),
        "train": res.as_dict(),
        "wall_s": round(time.time() - t0, 1),
        "before": before.summary(),
        "after": after.summary(),
        "after_tasks": after.as_dict()["tasks"],
        "_report": after,
        "_result": res,
    }
    save_checkpoint(os.path.join(args.out_dir, f"{name}.pt"), model, optim,
                    res.steps, res.tokens, cfg,
                    res.best_val if res.best_val is not None else 0.0,
                    time.time() - t0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="受控 A/B：SGC vs attention")
    ap.add_argument("--preset-a", dest="preset", default="tuan-30m")
    ap.add_argument("--preset-b", default="attn-30m")
    ap.add_argument("--match", choices=["steps", "params", "flops"], default="steps",
                    help="对齐口径：同步数（默认）/ 参数量 / 训练 FLOPs")
    ap.add_argument("--target-params", type=int, default=None,
                    help="--match params 时的目标参数量（默认取 sgc 侧实测值）")
    ap.add_argument("--train-jsonl", default=None)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="CPU 自证")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--layers-a", type=int, default=None)
    ap.add_argument("--layers-b", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--time-budget-min", type=float, default=0.0)
    ap.add_argument("--vocab-size", type=int, default=12288)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--use-memory", action="store_true",
                    help="两侧都开 CMT 记忆注入（需数据带 memory 字段）")
    ap.add_argument("--out-dir", default="runs/ab")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    if args.smoke:
        args.synthetic = args.synthetic or 6000
        args.max_steps = args.max_steps or 250
        args.seq_len = min(args.seq_len, 128)
        args.batch = min(args.batch, 16)
        args.dim = args.dim or 256
        args.layers_a = args.layers_a or 4
        args.layers_b = args.layers_b or 4
        args.vocab_size = min(args.vocab_size, 4096)
        args.eval_every = min(args.eval_every, 125)
        args.log_every = min(args.log_every, 25)
        args.warmup = min(args.warmup, 30)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- 数据（两侧共用一份，切分按行号，杜绝泄漏）-------------------------
    if args.train_jsonl is None:
        if not args.synthetic:
            raise SystemExit("需要 --train-jsonl，或 --synthetic N / --smoke")
        from simplcore.core.dialogue_data import build_records, write_jsonl
        args.train_jsonl = os.path.join(args.out_dir, "synthetic_sft.jsonl")
        write_jsonl(args.train_jsonl, build_records(args.synthetic, seed=args.seed))
    n_rec = sum(1 for _ in open(args.train_jsonl, encoding="utf-8"))
    n_val = max(1, int(n_rec * args.val_frac))
    n_train = max(1, n_rec - n_val)

    vocab = build_vocab(args.train_jsonl, args.vocab_size, n_train)
    vocab.save(os.path.join(args.out_dir, "charvocab.json"))
    codec = SpecialTokenCodec(vocab)
    template = ChatTemplate()
    mem_dim = args.dim or 0
    print(f"[data] 词表 {codec.size()}（训练段 {n_train} 条统计，验证段 {n_val} 条从未参与）")

    train_ds = SFTDataset(args.train_jsonl, codec, template, args.seq_len,
                          max_len=args.seq_len, index_slice=(0, n_train),
                          memory_dim=mem_dim if args.use_memory else 0)
    val_ds = SFTDataset(args.train_jsonl, codec, template, args.seq_len,
                        max_len=args.seq_len, index_slice=(n_train, n_rec),
                        memory_dim=mem_dim if args.use_memory else 0)
    print(f"[data] train {train_ds.stats()}")
    print(f"[data] val   {val_ds.stats()}")
    if args.use_memory and not train_ds.with_memory:
        raise SystemExit("--use-memory 但数据里没有 memory 字段")

    # ---- 超参（两侧同一个实例，这是"受控"的核心）-------------------------
    args.sft = SFTConfig(
        max_steps=args.max_steps or 3000,
        grad_accum=args.grad_accum,
        lr=args.lr, lr_min=args.lr / 10, warmup=args.warmup,
        log_every=args.log_every, eval_every=args.eval_every,
        time_budget_s=args.time_budget_min * 60.0, seed=args.seed,
    )

    # ---- 层数决定（--match params / flops 会改写 B 侧层数）----------------
    layers_a = args.layers_a
    layers_b = args.layers_b
    args.sft_b_steps = args.sft.max_steps       # 默认等步数；下面按需改写
    if args.match == "params":
        p_a = params_for_layers(args.preset, codec.size(),
                                layers_a or get_preset(args.preset).num_layers,
                                args.dim)
        target = args.target_params or p_a
        layers_b = match_params(args.preset_b, codec.size(), target, args.dim)
        p_b = params_for_layers(args.preset_b, codec.size(), layers_b, args.dim)
        print(f"[match] params: A={p_a:,}（{layers_a or get_preset(args.preset).num_layers} 层）"
              f" -> B 取 {layers_b} 层 = {p_b:,}（差 {abs(p_b-p_a)/max(p_a,1)*100:.2f}%）")
    elif args.match == "flops":
        # 先按 A 侧跑满步数算 FLOPs，再折算 B 侧步数（同一 seq_len 下比例恒定）
        ca = _preset_cfg(args.preset, codec.size(), layers_a, args.dim)
        cb = _preset_cfg(args.preset_b, codec.size(), layers_b, args.dim)
        ratio = (train_flops_per_token(ca, args.seq_len)
                 / max(train_flops_per_token(cb, args.seq_len), 1e-9))
        args.sft_b_steps = max(1, int(round(args.sft.max_steps * ratio)))
        print(f"[match] flops: B 侧步数 -> {args.sft_b_steps} "
              f"(A 侧 {args.sft.max_steps}，每 token FLOPs 比 {ratio:.3f})")
    else:
        args.sft_b_steps = args.sft.max_steps

    # ---- 两个 arm ---------------------------------------------------------
    a = run_arm("a", args, codec, template, train_ds, val_ds, device, layers_a)
    steps_a_default = args.sft.max_steps
    args.sft.max_steps = args.sft_b_steps
    b = run_arm("b", args, codec, template, train_ds, val_ds, device, layers_b)
    args.sft.max_steps = steps_a_default

    # ---- 比较 -------------------------------------------------------------
    paired = PairedOutcome.from_reports(a["name"], b["name"],
                                        a["_report"], b["_report"])
    budgets = {"a": a["params"], "b": b["params"]}
    fair = fairness_report({"a": a["_result"], "b": b["_result"]}, budgets)

    report = ab_report([paired], meta={
        "preset_a": args.preset, "preset_b": args.preset_b,
        "match": args.match, "seq_len": args.seq_len,
        "batch": args.batch, "grad_accum": args.grad_accum,
        "sft_config": args.sft.as_dict(), "seed": args.seed,
        "data": {"path": os.path.abspath(args.train_jsonl),
                 "n_train": n_train, "n_val": n_val,
                 "use_memory": bool(args.use_memory)},
        "fairness": fair,
        "param_gap": {
            "a": a["params"], "b": b["params"],
            "ratio": round(max(a["params"], b["params"])
                           / max(min(a["params"], b["params"]), 1), 4),
            "note": ("层数只能取整数，同宽度下 1 层差约 25%，因此 10% 以内的"
                     "参数量差是量化结果而非对齐失败。这个差值必须在报告里"
                     "看得见 —— 它本身就是「更小参数拿到同等分数」的一部分。"),
        },
        "flops_note": (
            "等步数 ≠ 等算力：本次每 token 训练 FLOPs A/B = "
            f"{a['train_flops_per_token']/1e9:.4f} / "
            f"{b['train_flops_per_token']/1e9:.4f} GFLOPs，"
            f"比值 {a['train_flops_per_token']/b['train_flops_per_token']:.3f}。"
            "若 A 侧更省，则 A 在同等墙钟下还能多训 token —— 这个差值本身就是结论的一部分。"
        ),
    })
    drop = ("_report", "_result")
    report["arms"] = {"a": {k: v for k, v in a.items() if k not in drop},
                      "b": {k: v for k, v in b.items() if k not in drop}}
    # 单点数据不给效率比：只有两档以上才有意义
    report["param_efficiency"] = param_efficiency(
        [{"name": "sgc" if a["mixer"] == "sgc" else "attn", "params": a["params"],
          "task_pass_rate": a["after"]["task_pass_rate"]},
         {"name": "sgc" if a["mixer"] == "sgc" else "attn", "params": b["params"],
          "task_pass_rate": b["after"]["task_pass_rate"]}], "task_pass_rate")

    path = os.path.join(args.out_dir, "report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    # ---- 打印 -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"A: {a['preset']:10s} mixer={a['mixer']:5s} params={a['params']:>10,} "
          f"tokens={a['train']['tokens']:>10,} GFLOPs/tok="
          f"{a['train_flops_per_token']/1e9:.4f} pass={a['after']['task_pass_rate']:.3f}")
    print(f"B: {b['preset']:10s} mixer={b['mixer']:5s} params={b['params']:>10,} "
          f"tokens={b['train']['tokens']:>10,} GFLOPs/tok="
          f"{b['train_flops_per_token']/1e9:.4f} pass={b['after']['task_pass_rate']:.3f}")
    p = report["pairs"][0]
    print(f"\nΔpass = {p['delta']:+.4f}  95% CI {p['ci95']}  "
          f"excludes_zero={p['excludes_zero']}")
    print(f"McNemar p={p['mcnemar']['p_value']:.4f} "
          f"(significant={p['mcnemar']['significant']})  {p['evidence_weight']}")
    if p["flipped"]:
        print("翻转题：")
        for f in p["flipped"]:
            print(f"  {f['id']:18s} A={f['first_passed']}  B={f['second_passed']}")
    print(f"\n可比性：{fair['comparable']}")
    for v in fair["violations"]:
        print("  ! " + v)
    print(f"\n报告：{path}")
    if args.smoke:
        print("注：--smoke 用小模型 + 合成数据，只证明框架与统计链路，不构成架构结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
