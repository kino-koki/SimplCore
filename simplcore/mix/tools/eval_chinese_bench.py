#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SimplCore Mix 中文基准评测入口。

把"对标 31B"变成**可打印的分数表**：加载 SimplCoreMix 模型 + tokenizer，
跑 C-Eval 风格 few-shot 选择题，输出 per-subject + 平均正确率。

模型加载（两路）:
  --ckpt <path>     加载 .pt checkpoint（resolve_state_dict 自动处理 DDP/bf16）
  --hf <repo>       加载 HuggingFace 权重（后续接入）

Tokenizer:
  --tokenizer <path>  真 BPE tokenizer.json
  (无则回退字符哈希，但中文基准分数不可信，仅 sanity 验证)

用法:
  # tiny 模型验证管线
  SIMPLCORE_ALLOW_CPU=1 python -m simplcore.mix.tools.eval_chinese_bench \
      --ckpt artifacts/tok_train_test/mix_150m_seed42.pt \
      --builtin tiny --max-new-tokens 4

  # 680M / V100 真模型
  python -m simplcore.mix.tools.eval_chinese_bench \
      --ckpt <7B/3B checkpoint> --tokenizer tokenizer.json --builtin ceval
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import simplcore.tensor as torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simplcore.mix.config import mix_tiny_text, mix_150m_text, mix_7b_text
from simplcore.mix.evaluation.eval_utils import (
    resolve_state_dict,
    build_fewshot_prompt,
    extract_answer,
    run_generation_eval,
)
from simplcore.mix.model import SimplCoreMix
from simplcore.mix.training.tokenizer_adapter import load_tokenizer, make_encoder
from simplcore.mix.training.fast_ops import encode_text_fast


# ============================================================
# 1. 内置中文微调测试集 (tiny / sanity)
# ============================================================
BUILTIN_DATASETS = {
    "tiny": {
        "subject": "tiny_sanity",
        "examples": [
            {"question": "6月是几月份", "choices": ["五", "六", "七", "八"], "answer": "B"},
            {"question": "中国首都是哪里", "choices": ["上海", "北京", "广州", "深圳"], "answer": "B"},
            {"question": "1+1等于多少", "choices": ["1", "2", "3", "4"], "answer": "B"},
        ],
        "test": [
            {"question": "安徽省会在哪里", "choices": ["合肥", "北京", "上海", "广州"], "answer": "A"},
            {"question": "人工智能的英文缩写是", "choices": ["AI", "BI", "CI", "DI"], "answer": "A"},
            {"question": "世界上最大的行星是", "choices": ["地球", "火星", " jupiter", "Jupiter", "土星"], "answer": "D"},
        ],
    },
    "ceval": {
        "subject": "ceval_demo",
        "description": "中英文测试（C-Eval 风格，2-shot）",
        "examples": [
            {"question": "香蚀面菌的繁殖方式是", "choices": ["裂殖", "孢子繁殖", "菌丝节育", "以上都有"], "answer": "D"},
            {"question": "我国古代“十大高楼”之首指的是", "choices": ["塔", "楼", "阁", "亭"], "answer": "A"},
        ],
        "test": [
            {"question": "以下关于细胞学说的叙述中，正确的是",
             "choices": ["细胞是生物体的基本结构单元", "所有生物都是由细胞组成", "细胞只能单独存活", "以上都不正确"], "answer": "A"},
            {"question": "《红楼梦》的作者是", "choices": ["曹雨廉", "曹雪芹", "高翰", "吴语"], "answer": "B"},
            {"question": "下列属于现代文学家的是",
             "choices": ["鲁曲丹", "鲁迅", "茅盾", "巴尔托利"], "answer": "B"},
        ],
    },
}


def _flatten_tests(ds: Dict) -> List[Dict]:
    return ds.get("test", [])


# ============================================================
# 2. 模型与 tokenizer 加载
# ============================================================

def load_model(ckpt_path: Optional[Path], *, cfg_name: str, device) -> SimplCoreMix:
    cfg_fn = {"tiny": mix_tiny_text, "150m": mix_150m_text, "7b": mix_7b_text}[cfg_name]
    model = SimplCoreMix(cfg_fn()).to(device)
    if ckpt_path is not None:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        sd = resolve_state_dict(ckpt)
        model.load_state_dict(sd, strict=False)
        # 报告缺失/冗余
        m, u = model.load_state_dict(sd, strict=False)
        print(f"[eval] checkpoint 加载: missing={len(m)} unexpected={len(u)}")
    else:
        print("[eval] 未提供 checkpoint，用随机初始化模型（仅 sanity）")
    model.eval()
    return model


def load_tokenizer_or_hash(path: Optional[str], vocab_size: int):
    tok = load_tokenizer(path, vocab_size)
    if tok is not None:
        enc = make_encoder(tok, vocab_size)
        dec = tok.decode
        print(f"[eval] tokenizer: BPE (vocab={len(tok)})")
    else:
        enc = lambda text, sl: encode_text_fast(text, vocab_size, sl)
        dec = lambda ids: "".join(chr(max(0, i - 2)) if i > 1 else "" for i in ids)
        print("[eval] tokenizer: 回退字符哈希（仅 sanity）")
    return enc, dec


# ============================================================
# 3. 评测主循环
# ============================================================

def evaluate(
    model,
    encode,
    decode,
    *,
    device,
    dataset_name: str,
    n_shot: int = 2,
    max_new_tokens: int = 8,
) -> Dict:
    ds = BUILTIN_DATASETS[dataset_name]
    examples = ds.get("examples", [])
    tests = _flatten_tests(ds)
    vocab_size = model.cfg.vocab_size

    # 组装 prompts
    prompts = []
    for t in tests:
        prompt = build_fewshot_prompt(examples[:n_shot], t["question"], t["choices"])
        prompts.append(prompt)

    results = run_generation_eval(
        model, encode, decode,
        device=device,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    # 统计
    correct = 0
    for r, t in zip(results, tests):
        gold = t["answer"].strip().upper()
        pred = r["answer"]
        r["gold"] = gold
        r["correct"] = pred == gold
        correct += int(r["correct"])

    acc = correct / max(len(tests), 1)
    return {
        "dataset": dataset_name,
        "subject": ds.get("subject", ""),
        "n_shot": n_shot,
        "n_tests": len(tests),
        "correct": correct,
        "accuracy": round(acc, 4),
        "details": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SimplCore Mix 中文基准评测")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--builtin", choices=list(BUILTIN_DATASETS), default="tiny",
                        help="内置测试集（tiny/sanity 或 ceval-demo）")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--n-shot", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--cfg", choices=["tiny", "150m", "7b"], default="tiny")
    parser.add_argument("--output", type=Path, default=Path("artifacts/eval_out.json"))
    args = parser.parse_args()

    import os as _os
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    print(f"[eval] 设备: {device}")

    model = load_model(args.ckpt, cfg_name=args.cfg, device=device)
    encode, decode = load_tokenizer_or_hash(args.tokenizer, model.cfg.vocab_size)

    summary = evaluate(
        model, encode, decode,
        device=device,
        dataset_name=args.builtin,
        n_shot=args.n_shot,
        max_new_tokens=args.max_new_tokens,
    )

    # 可打印分数表
    print("\n===== 评测结果 =====")
    print(f"  数据集: {summary['dataset']} ({summary['subject']})")
    print(f"  模式:   {summary['n_shot']}-shot")
    print(f"  题目:   {summary['n_tests']} | 正确: {summary['correct']}")
    print(f"  准确率: {summary['accuracy']:.2%}")
    print("\n  详情:")
    for d in summary["details"]:
        mark = "✓" if d["correct"] else "✗"
        print(f"    [{mark}] 金:{d['gold']} 予:{d['answer'] or '(空)'} | {d['prompt'].splitlines()[0][:30]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] 结果保存: {args.output}")


if __name__ == "__main__":
    _env_cpu = __import__("os").environ.get("SIMPLCORE_ALLOW_CPU", "0")
    if not torch.cuda.is_available() and _env_cpu != "1":
        print("[eval] 警告: 无 CUDA，使用 CPU（仅 sanity）。加 SIMPLCORE_ALLOW_CPU=1 压制。")
    main()
