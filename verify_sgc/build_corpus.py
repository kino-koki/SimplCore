# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""本地真实语料构建 (外网不可达, 复用项目内中文技术报告).

关键修复 (防止数据泄漏):
  旧版用滑动窗口(seq_len//2 重叠) + 全局 shuffle 取前 10% 当 val,
  导致 val 窗口与 train 窗口高度重叠, 验证集等于"背过" -> val_ppl≈1 是伪结果.
  本版改为 **文档级切分**:
    - 每个源文件 = 一个文档, 整体归入 train / val / test 之一, 绝不跨集拆分;
    - 文档内窗口 **不重叠** (step = seq_len);
    - 词表只用 train 文档构建 (防止 val/test 字符信息泄漏进词表);
    - 固定种子, 可复现.
"""
from __future__ import annotations

import os
import glob
import json
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_documents() -> list[tuple[str, str]]:
    """收集项目内所有 .txt/.md 报告, 每个文件作为一个文档."""
    patterns = [
        os.path.join(PROJECT_ROOT, "*.txt"),
        os.path.join(PROJECT_ROOT, "docs", "**", "*.md"),
        os.path.join(PROJECT_ROOT, "java-high-performance", "*.txt"),
        os.path.join(PROJECT_ROOT, "simplcore", "**", "*.txt"),
    ]
    docs = []
    seen = set()
    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            if fp in seen:
                continue
            seen.add(fp)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    t = f.read()
                t = t.strip()
                if len(t) > 500:       # 过滤过短文件
                    docs.append((fp, t))
            except Exception:
                continue
    return docs


def build_tokenizer(texts: list[str], min_freq: int = 2):
    """character-level tokenizer (中文以字为单位). 只用给定文本构建."""
    freq = {}
    for t in texts:
        for ch in t:
            freq[ch] = freq.get(ch, 0) + 1
    vocab = ["<pad>", "<unk>", "\n"]
    for ch, c in sorted(freq.items(), key=lambda kv: -kv[1]):
        if c >= min_freq and ch not in vocab:
            vocab.append(ch)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    return vocab, stoi


def encode(text: str, stoi: dict) -> list:
    return [stoi.get(ch, stoi["<unk>"]) for ch in text]


def doc_to_windows(ids: list, seq_len: int, step: int):
    """文档内窗口. step<seq_len 时窗口重叠 (同文档内重叠安全, 因整篇归属单一集合,
    不会跨集泄漏). 返回 list[(x, y)] 各长 seq_len."""
    out = []
    for i in range(0, len(ids) - seq_len - 1, step):
        x = ids[i:i + seq_len]
        y = ids[i + 1:i + seq_len + 1]
        out.append((x, y))
    return out


def make_dataset(seq_len: int = 256, val_ratio: float = 0.1,
                 test_ratio: float = 0.1, seed: int = 1234, step: int = None):
    step = step or max(1, seq_len // 4)
    docs = collect_documents()
    # 先编码以统计每篇文档的窗口数 (用于按比例贪心分配)
    # 词表需先建: 用全量文档建临时词表统计窗口数, 最终词表仍只用 train 文档
    tmp_vocab, tmp_stoi = build_tokenizer([t for _, t in docs])
    doc_windows = {}
    for fp, t in docs:
        ids = encode(t, tmp_stoi)
        doc_windows[fp] = len(doc_to_windows(ids, seq_len, step))

    # 按窗口数贪心分配, 保持文档级完整 (整篇归单一集合)
    order = docs[:]
    rng = random.Random(seed)
    rng.shuffle(order)
    total = sum(doc_windows[fp] for fp, _ in order)
    target_val = int(total * val_ratio)
    target_test = int(total * test_ratio)
    train_docs, val_docs, test_docs = [], [], []
    acc_val = acc_test = 0
    for fp, t in order:
        w = doc_windows[fp]
        if acc_val < target_val:
            val_docs.append((fp, t)); acc_val += w
        elif acc_test < target_test:
            test_docs.append((fp, t)); acc_test += w
        else:
            train_docs.append((fp, t))

    # 词表仅用 train 文档
    train_texts = [t for _, t in train_docs]
    vocab, stoi = build_tokenizer(train_texts)

    def windows_for(doc_list):
        all_w = []
        for _, t in doc_list:
            ids = encode(t, stoi)
            all_w.extend(doc_to_windows(ids, seq_len, step))
        return all_w

    train = windows_for(train_docs)
    val = windows_for(val_docs)
    test = windows_for(test_docs)
    return vocab, train, val, test


if __name__ == "__main__":
    seq_len = 256
    vocab, train, val, test = make_dataset(seq_len)
    print("vocab size:", len(vocab))
    print(f"windows: train={len(train)} val={len(val)} test={len(test)}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "corpus.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"vocab": vocab, "train": train, "val": val, "test": test}, open(out, "w"))
    print("saved ->", out)
