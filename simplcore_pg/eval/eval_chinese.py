# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""中文能力评估 harness (C-Eval / CMMLU 风格 few-shot 选择题)。

目标：把"媲美/超过前沿模型中文能力"变成**可量化、可复现**的对照线。
  * 支持加载真实 C-Eval/CMMLU 的 JSONL (字段: question/choices/answer/subject)；
  * 内置小样本集用于离线冒烟，证明 harness 可运行；
  * few-shot 提示 + 选项字母解析 + 准确率与分科目统计。

解析鲁棒性：从生成文本中取首个出现在 {A..G} 的字母作为预测，容忍自由文本。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from simplcore_pg.training.data import SimpleChineseTokenizer
from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_moe import SimplCoreMoEModel

LETTERS = "ABCDEFG"
_CHOICE_RE = re.compile(r"\b([" + LETTERS + r"])\b")


@dataclass
class BenchmarkSample:
    question: str
    choices: List[str]
    answer: int
    subject: str = "general"
    id: str = ""

    @staticmethod
    def from_dict(d: dict) -> "BenchmarkSample":
        ans = d["answer"]
        if isinstance(ans, str):
            ans = LETTERS.index(ans.upper()) if ans.upper() in LETTERS else -1
        return BenchmarkSample(
            question=d["question"], choices=list(d["choices"]),
            answer=int(ans), subject=d.get("subject", "general"), id=str(d.get("id", "")),
        )


def load_benchmark(path: str) -> List[BenchmarkSample]:
    """加载 JSONL 基准 (每行一个样本)。"""
    samples: List[BenchmarkSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(BenchmarkSample.from_dict(json.loads(line)))
    return samples


# 内置小样本基准 (离线冒烟用；真实评测请加载 C-Eval/CMMLU)
SAMPLE_BENCHMARK: List[dict] = [
    {"id": "cn1", "subject": "常识", "question": "中国的首都是哪里？",
     "choices": ["上海", "北京", "广州", "西安"], "answer": "B"},
    {"id": "cn2", "subject": "古诗", "question": "“床前明月光”的下一句是？",
     "choices": ["疑是地上霜", "低头思故乡", "举头望明月", "玉盘似的月亮"], "answer": "A"},
    {"id": "cn3", "subject": "数学", "question": "一个长方形的长是 8，宽是 5，面积是？",
     "choices": ["13", "40", "26", "35"], "answer": "B"},
    {"id": "cn4", "subject": "古文", "question": "“学而时习之”出自哪部经典？",
     "choices": ["《孟子》", "《论语》", "《道德经》", "《大学》"], "answer": "B"},
    {"id": "cn5", "subject": "常识", "question": "长江是中国第几长的河流？",
     "choices": ["第一", "第二", "第三", "第四"], "answer": "A"},
    {"id": "cn6", "subject": "成语", "question": "“画蛇添足”比喻？",
     "choices": ["多此一举", "锦上添花", "雪中送炭", "精益求精"], "answer": "A"},
    {"id": "cn7", "subject": "地理", "question": "珠穆朗玛峰位于哪两国交界？",
     "choices": ["中国与印度", "中国与尼泊尔", "印度与尼泊尔", "中国与不丹"], "answer": "B"},
    {"id": "cn8", "subject": "历史", "question": "统一六国的秦朝皇帝是？",
     "choices": ["汉武帝", "秦始皇", "唐太宗", "隋文帝"], "answer": "B"},
]


def _build_prompt(tok: SimpleChineseTokenizer, sample: BenchmarkSample,
                  shots: List[BenchmarkSample]) -> List[int]:
    parts: List[str] = []
    for s in shots:
        parts.append(_format_qa(s))
    parts.append(_format_q(simple=sample, with_answer=False))
    text = "以下是选择题，请选出正确答案的字母。\n" + "\n".join(parts) + "答案："
    return tok.encode(text)


def _format_qa(s: BenchmarkSample) -> str:
    return _format_q(simple=s, with_answer=True)


def _format_q(simple: BenchmarkSample, with_answer: bool) -> str:
    lines = [simple.question]
    for i, c in enumerate(simple.choices):
        lines.append(f"{LETTERS[i]}. {c}")
    if with_answer:
        lines.append(f"答案：{LETTERS[simple.answer]}")
    else:
        lines.append("")
    return "\n".join(lines)


def _parse_prediction(text: str) -> Optional[int]:
    m = _CHOICE_RE.search(text)
    if m:
        return LETTERS.index(m.group(1))
    # 退而求其次：匹配 "答案是X" 中的字母
    m2 = re.search(r"答案[是为:：]?\s*([%s])" % LETTERS, text)
    if m2:
        return LETTERS.index(m2.group(1))
    return None


@torch.no_grad()
def evaluate(
    model: SimplCoreMoEModel,
    tok: SimpleChineseTokenizer,
    samples: List[BenchmarkSample],
    *,
    k_shot: int = 4,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
) -> Dict:
    """运行 few-shot 评估，返回总体与各科目准确率。"""
    model.eval()
    max_len = getattr(getattr(model, "config", None), "max_sequence_length", 2048)
    correct = 0
    per_subject: Dict[str, List[int]] = {}
    for idx, s in enumerate(samples):
        shots = [samples[j] for j in range(max(0, idx - k_shot), idx)]
        ids = _build_prompt(tok, s, shots)
        # 超出上下文则截断 (保留尾部：题目与选项 + 答案前缀)；并预留生成空间
        budget = max(1, max_len - max_new_tokens)
        if len(ids) > budget:
            ids = ids[-budget:]
        input_ids = torch.tensor([ids], dtype=torch.long)
        gen = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        pred_text = tok.decode(gen[0].tolist())
        pred = _parse_prediction(pred_text)
        ok = pred is not None and pred == s.answer
        if ok:
            correct += 1
        per_subject.setdefault(s.subject, []).append(1 if ok else 0)

    n = len(samples)
    acc = correct / n if n else 0.0
    subject_acc = {k: sum(v) / len(v) for k, v in per_subject.items()}
    return {"accuracy": acc, "n": n, "correct": correct, "subject_accuracy": subject_acc}


def _main() -> None:
    p = argparse.ArgumentParser(description="SimplCore 中文能力评估")
    p.add_argument("--preset", default="56b-moe")
    p.add_argument("--benchmark", default="", help="C-Eval/CMMLU JSONL 路径；留空用内置样本")
    p.add_argument("--k-shot", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="CPU 小模型冒烟")
    args = p.parse_args()

    if args.smoke:
        cfg = SimplCoreConfig(
            vocab_size=512, hidden_size=64, num_layers=2, num_experts=2,
            experts_per_tok=2, moe_intermediate_size=128, max_sequence_length=256,
            dmc_num_heads=4, use_rfp=False, use_dmc=False, dropout=0.0,
        )
        model = SimplCoreMoEModel(cfg)
    else:
        cfg = SimplCoreConfig.from_preset(args.preset)
        model = SimplCoreMoEModel(cfg)

    tok = SimpleChineseTokenizer(vocab_limit=model.config.vocab_size)
    if args.benchmark:
        samples = load_benchmark(args.benchmark)
    else:
        samples = [BenchmarkSample.from_dict(d) for d in SAMPLE_BENCHMARK]

    res = evaluate(model, tok, samples, k_shot=args.k_shot, max_new_tokens=args.max_new_tokens)
    print("=" * 48)
    print(f"中文评估 (samples={res['n']}, k-shot={args.k_shot})")
    print("=" * 48)
    print(f"  总体准确率 : {res['accuracy']:.3f}")
    for sub, a in res["subject_accuracy"].items():
        print(f"  - {sub:<8s}: {a:.3f}")
    print("=" * 48)
    print("说明：内置/冒烟模型未经训练，准确率约随机；加载真实 C-Eval/CMMLU 后可对照前沿模型分数线。")


if __name__ == "__main__":
    _main()
