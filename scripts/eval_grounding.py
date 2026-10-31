#!/usr/bin/env python
# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""证据绑定 / 拒答门的**实测**脚本（构造标签，非模型自评）。

为什么这样标注
--------------
要给"归因判据准不准"打分，标签不能来自判据自己（那是循环论证），也不能来自
模型（那是自我加冕）。本脚本用**构造**产生真值：

================  ==========================================  ==========
类型              构造方式                                     真值
================  ==========================================  ==========
exact             从上下文逐字复制一整句                          有依据
truncated         复制后删掉约 1/10 字符（不再逐字包含）           有依据
number_swap       复制后把一个数字改成上下文中不存在的数字          无依据
cross_doc         取自**另一份文档**的句子（已校验不在本上下文）     无依据
================  ==========================================  ==========

前两类是**正例**，后两类是**负例**；真值由构造方式保证，与判据无关。

输出
----
``artifacts/grounding_eval.json``：判据的 precision/recall/F1、四态决策分布、
含编造句回答的**拦截率**、代理分数的 ECE/AUC，以及能力边界清单。

用法
----
    python scripts/eval_grounding.py --docs docs --out artifacts/grounding_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from simplcore.core.abstain import (                                 # noqa: E402
    AbstentionState,
    GroundedPolicy,
    evidence_aware_decision,
    expected_calibration_error,
    select_threshold,
)
from simplcore.core.grounding import (                               # noqa: E402
    Claim,
    EvidenceIndex,
    bind_claim,
    coverage_of,
    normalize,
    split_claims,
)

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_SENT_RE = re.compile(r"[。！？!?；;]")


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------
def load_sentences(docs_dir: str, min_chars: int = 18,
                   max_docs: int = 40) -> List[Tuple[str, List[str]]]:
    """返回 ``[(doc_name, [sentence, ...]), ...]``，句子来自真实仓库文档。"""
    out: List[Tuple[str, List[str]]] = []
    if not os.path.isdir(docs_dir):
        return out
    files = sorted(f for f in os.listdir(docs_dir) if f.endswith(".md"))
    for name in files[:max_docs]:
        path = os.path.join(docs_dir, name)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        sents: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            # 跳过 markdown 结构行：标题目录、表格、代码围栏、引用、列表项
            if not line or line[0] in "#|>`-*":
                continue
            for seg in _SENT_RE.split(line):
                seg = seg.strip().strip("|").strip()
                # 只保留"单行单句"的干净句子：不含内联强调符/剩余列表符
                if "*" in seg or "|" in seg or seg.startswith("-"):
                    continue
                if len(normalize(seg)) >= min_chars:
                    sents.append(seg)
        if len(sents) >= 20:
            out.append((name, sents))
    return out


# ---------------------------------------------------------------------------
# 构造样本
# ---------------------------------------------------------------------------
def _truncate(sent: str, rng: random.Random, ratio: float = 0.12) -> str:
    n = max(1, int(len(sent) * ratio))
    idxs = sorted(rng.sample(range(len(sent)), min(n, max(1, len(sent) // 8))), reverse=True)
    chars = list(sent)
    for i in idxs:
        if 0 <= i < len(chars):
            chars[i] = ""
    return "".join(chars)


def _swap_number(sent: str, forbidden: str, rng: random.Random) -> Optional[str]:
    nums = _NUM_RE.findall(sent)
    if not nums:
        return None
    old = rng.choice(nums)
    for _ in range(40):
        candidate = str(rng.randint(11, 999))
        if candidate != old and candidate not in forbidden:
            return sent.replace(old, candidate, 1)
    return None


def _flip_negation(sent: str) -> Optional[str]:
    """在句中插入一个否定标记（制造"一字之差、语义相反"的负例）。"""
    for i in range(len(sent) // 2, len(sent)):
        if sent[i] in "，、, ":
            continue
        return sent[:i] + "不" + sent[i:]
    return None


def build_samples(docs: Sequence[Tuple[str, List[str]]],
                  rng: random.Random,
                  per_type: int = 120) -> List[Dict[str, Any]]:
    """构造四类带真值的样本。每个样本都带 ``context``（证据集合）。"""
    if len(docs) < 2:
        return []
    samples: List[Dict[str, Any]] = []
    type_budget = {t: per_type for t in
                   ("exact", "truncated", "number_swap", "negation", "cross_doc")}

    guard = 0
    while any(v > 0 for v in type_budget.values()) and guard < per_type * 40:
        guard += 1
        name, sents = docs[rng.randrange(len(docs))]
        ctx_sents = [sents[rng.randrange(len(sents))] for _ in range(6)]
        ctx = "".join(s if s.endswith("。") else s + "。" for s in ctx_sents)
        ctx_norm = normalize(ctx)
        sent = ctx_sents[rng.randrange(len(ctx_sents))]
        if len(normalize(sent)) < 18:
            continue

        # --- exact ---
        if type_budget["exact"] > 0:
            type_budget["exact"] -= 1
            samples.append(dict(kind="exact", claim=sent, context=ctx,
                                truth=1, doc=name))
            continue

        # --- truncated（正例，但不再逐字包含）---
        if type_budget["truncated"] > 0:
            t = _truncate(sent, rng)
            if normalize(t) and normalize(t) not in ctx_norm and len(t) > 12:
                type_budget["truncated"] -= 1
                samples.append(dict(kind="truncated", claim=t, context=ctx,
                                    truth=1, doc=name))
                continue

        # --- number_swap（负例）---
        if type_budget["number_swap"] > 0:
            sw = _swap_number(sent, ctx_norm, rng)
            if sw and normalize(sw) not in ctx_norm:
                type_budget["number_swap"] -= 1
                samples.append(dict(kind="number_swap", claim=sw, context=ctx,
                                    truth=0, doc=name))
                continue

        # --- negation（负例：插入否定标记，语义翻转）---
        if type_budget["negation"] > 0:
            cand = _flip_negation(sent)
            if cand and normalize(cand) not in ctx_norm:
                type_budget["negation"] -= 1
                samples.append(dict(kind="negation", claim=cand, context=ctx,
                                    truth=0, doc=name))
                continue

        # --- cross_doc（负例，来自另一文档）---
        if type_budget["cross_doc"] > 0:
            other_name, other_sents = docs[rng.randrange(len(docs))]
            if other_name == name:
                continue
            other = other_sents[rng.randrange(len(other_sents))]
            if len(normalize(other)) >= 18 and normalize(other) not in ctx_norm:
                type_budget["cross_doc"] -= 1
                samples.append(dict(kind="cross_doc", claim=other, context=ctx,
                                    truth=0, doc=other_name))
    return samples


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """秩和法 AUC（并列取平均秩）。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda p: p[0])
    ranks: Dict[int, float] = {}
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k in range(len(ranked)) if ranked[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def confusion(pred: Sequence[int], truth: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "accuracy": round((tp + tn) / max(len(pred), 1), 4),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.path.join(_ROOT, "docs"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "artifacts",
                                                 "grounding_eval.json"))
    ap.add_argument("--per-type", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)
    docs = load_sentences(args.docs)
    if not docs:
        print("no docs found under", args.docs)
        return 2
    samples = build_samples(docs, rng, per_type=args.per_type)
    if not samples:
        print("no samples constructed")
        return 2

    # 划分 dev / test：阈值只能在 dev 上选，指标在 test 上报告（避免阈值泄漏）
    rng.shuffle(samples)
    half = len(samples) // 2
    dev, test = samples[:half], samples[half:]

    def run(part: Sequence[Dict[str, Any]], collect: bool = False):
        pred: List[int] = []
        truth: List[int] = []
        fused: List[float] = []
        kinds: Dict[str, Dict[str, int]] = {}
        st: Dict[str, int] = {s.value: 0 for s in AbstentionState}
        blocked_n = fab_n = 0
        for smp in part:
            idx = EvidenceIndex.from_text(smp["context"], source=smp["doc"])
            b = bind_claim(Claim(smp["claim"]), idx.spans)
            # 归因覆盖率：单句口径 = supported 与否（这才是归因判定的输出，
            # 不能用字符覆盖率 —— 数字被篡改时字符覆盖率仍有 0.95）
            cov = 1.0 if b.supported else 0.0
            # 代理 grounded 分数（未训练 L4 头时的占位）：二元组命中率
            g = b.ngram
            d = evidence_aware_decision(g, cov, 1.0 if len(idx) else 0.0,
                                        GroundedPolicy())
            pred.append(1 if b.supported else 0)
            truth.append(int(smp["truth"]))
            fused.append(float(d.score))
            st[d.state.value] += 1
            k = kinds.setdefault(smp["kind"], {"n": 0, "correct": 0})
            k["n"] += 1
            k["correct"] += int(bool(b.supported) == bool(smp["truth"]))
            if smp["truth"] == 0:
                fab_n += 1
                if d.state is not AbstentionState.ANSWER:
                    blocked_n += 1
        return dict(pred=pred, truth=truth, fused=fused, kinds=kinds,
                    states=st, blocked=blocked_n, fabricated=fab_n)

    dev_r = run(dev)
    test_r = run(test)
    all_r = run(samples)
    per_kind = all_r["kinds"]
    states = all_r["states"]
    blocked, fabricated = all_r["blocked"], all_r["fabricated"]

    # 端到端：两真一假的回答能否被拦（而不是 ANSWER）
    multi_blocked = multi_total = 0
    for _ in range(200):
        name, sents = docs[rng.randrange(len(docs))]
        ctx_sents = [sents[rng.randrange(len(sents))] for _ in range(6)]
        ctx = "".join(x if x.endswith("。") else x + "。" for x in ctx_sents)
        idx = EvidenceIndex.from_text(ctx, source=name)
        good = split_claims("。".join(ctx_sents[:2]) + "。")
        fab = build_cross_claim(docs, name, rng, normalize(ctx))
        if fab is None or not good:
            continue
        answer = "。".join([g.text for g in good] + [fab]) + "。"
        bindings = idx.bind_all(split_claims(answer))
        from simplcore.core.grounding import attribution_coverage
        cov = attribution_coverage(bindings)
        n_unsup = sum(1 for bb in bindings if not bb.supported)
        d = evidence_aware_decision(cov, cov, 1.0, GroundedPolicy(),
                                    n_unsupported=n_unsup)
        multi_total += 1
        if d.state is not AbstentionState.ANSWER:
            multi_blocked += 1

    cm = confusion(all_r["pred"], all_r["truth"])
    cm_test = confusion(test_r["pred"], test_r["truth"])
    # 阈值在 dev 上选，在 test 上评（报告里两者都给，避免"测试集调参"）
    best = select_threshold(dev_r["fused"], dev_r["truth"], metric="f1")
    thr = float(best["threshold"])
    test_pred = [1 if sc >= thr else 0 for sc in test_r["fused"]]
    cm_test_at_thr = confusion(test_pred, test_r["truth"])
    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/eval_grounding.py",
        "seed": args.seed,
        "seconds": round(time.time() - t0, 2),
        "n_samples": len(samples),
        "by_kind": {k: {**v, "acc": round(v["correct"] / max(v["n"], 1), 4)}
                    for k, v in sorted(per_kind.items())},
        "attribution": cm,
        "attribution_test_split": cm_test,
        "n_dev": len(dev),
        "n_test": len(test),
        "proxy_score": {
            "note": "代理分数 = 最佳单句二元组命中率（未训练 L4 头时的占位），"
                    "其 AUC 不是 L4 头的性能，切勿外推",
            "auc": round(auc(all_r["fused"], all_r["truth"]), 4),
            "ece": round(expected_calibration_error(all_r["fused"],
                                                   all_r["truth"], n_bins=10), 4),
            "threshold_from_dev": {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in best.items()},
            "test_at_dev_threshold": cm_test_at_thr,
        },
        "decision_states": states,
        "single_claim_block_rate": round(blocked / fabricated, 4) if fabricated else 0.0,
        "multi_claim_block_rate": round(multi_blocked / multi_total, 4) if multi_total else 0.0,
        "multi_claim_n": multi_total,
        "judge_components": {
            "min_coverage": 0.62,
            "min_ngram": 0.35,
            "numeric_check": "claim 的每个数字必须在 span 中出现",
            "polarity_check": "否定词计数必须相等",
        },
        "interpretation": [
            "supported 判据（LCS>=0.62 且 bigram>=0.35）在 exact 上必然为 1，"
            "真正的考验是 truncated / number_swap / cross_doc 三类。",
            "proxy score 是未训练 L4 头时的占位（最佳单句覆盖率），"
            "其 AUC/ECE 不能当作 L4 头的性能。",
            "阈值在 dev 上选、在 test 上评；attribution 的阈值是判据先验，"
            "未在数据上调参，故全量报告。",
            "single_claim_block_rate = 构造负例中未给出 ANSWER 的比例；"
            "multi_claim_block_rate = 两真一假的回答未给出 ANSWER 的比例。",
        ],
        "honest_bounds": [
            "本脚本测的是文本层归因，不测语义蕴含：逐字出现而被语境误用的情况测不出来。",
            "样本由仓库自身文档构造，领域单一；跨领域分布外表现未知。",
            "四态阈值（0.5/0.8/0.25）为先验，未在人工标注集上调参。",
            "未训练 L4 头，因此不宣称'抗幻觉有效'，只报告判据与规则的可测行为。",
        ],
    }

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("n_samples", "attribution", "proxy_score",
                       "decision_states", "single_claim_block_rate",
                       "multi_claim_block_rate")},
                     ensure_ascii=False, indent=2))
    print("written:", out)
    return 0


def build_cross_claim(docs, current_doc: str, rng: random.Random,
                      ctx_norm: str) -> Optional[str]:
    """取一句不在当前上下文里的句子（来自别的文档）作为编造主张。"""
    for _ in range(30):
        name, sents = docs[rng.randrange(len(docs))]
        if name == current_doc:
            continue
        s = sents[rng.randrange(len(sents))]
        if len(normalize(s)) >= 18 and normalize(s) not in ctx_norm:
            return s
    return None


if __name__ == "__main__":
    raise SystemExit(main())
