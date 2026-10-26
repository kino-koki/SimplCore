# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""bench_corpus_ppl 小规模守护测试：验证真实语料困惑度对比管线可用。"""
import json
import math
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench_corpus_ppl import (
    CharTokenizer,
    TransformerBaseline,
    build_simplcore_config,
    load_corpus,
    make_batches,
    split_documents,
    split_documents_detailed,
)
from simplcore_pg.model_moe import SimplCoreMoEModel


def _contains_subseq(haystack, needle) -> bool:
    """needle 是否作为连续子序列出现在 haystack 中。"""
    n, m = len(haystack), len(needle)
    if m == 0:
        return True
    if m > n:
        return False
    return any(haystack[i : i + m] == needle for i in range(n - m + 1))


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    texts = [
        "海内存知己，天涯若比邻。白日依山尽，黄河入海流。",
        "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
        "落霞与孤鹜齐飞，秋水共长天一色。",
        "书桌上摊开一本旧书，纸页泛黄，墨迹却依旧清晰。",
        "江南的雨总是细密而温柔，落在青石板上，溅起一朵朵小小的水花。",
        "清晨的露珠挂在草叶尖上，在朝阳里闪着晶莹的光。",
        "远山如黛，近水含烟，一幅水墨画在天地间徐徐展开。",
        "他推开木窗，看见院子里那棵老槐树又开满了白色的花。",
        "深夜的图书馆里只有翻书声和笔尖划过纸面的沙沙声。",
        "列车穿过隧道，窗外的风景从青山绿水变成高楼林立。",
        "母亲在厨房里忙碌，锅铲与铁锅碰撞出熟悉的节奏。",
        "雨后的天空格外澄澈，一道彩虹横跨在田野尽头。",
        "他沿着河岸慢慢走，想起许多年前在这里度过的夏天。",
        "实验室的灯光彻夜未熄，数据在屏幕上不断刷新。",
    ]
    p = d / "cn_test.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    return str(d)


def test_tokenizer_and_split(tiny_corpus):
    texts = load_corpus(tiny_corpus)
    tok = CharTokenizer(texts, max_vocab=512)
    assert tok.vocab_size >= 80
    train_ids, val_ids, train_docs, val_docs = split_documents_detailed(
        texts, tok, 0.2, seed=0
    )
    assert len(train_ids) > 0 and len(val_ids) > 0
    # 文档级留出：验证集文档的 token 序列不应作为连续子序列出现在训练集
    # （字符级词表天然高度重叠，故不能只比较 token 集合，须比较整篇文档序列）
    for _, ids in val_docs:
        assert not _contains_subseq(train_ids, ids), "验证集文档泄漏到训练集"
    # 且验证集文档文本与训练集文档文本不相交
    train_texts = {t for t, _ in train_docs}
    for t, _ in val_docs:
        assert t not in train_texts


def test_models_forward(tiny_corpus):
    texts = load_corpus(tiny_corpus)
    tok = CharTokenizer(texts, max_vocab=512)
    sc_cfg = build_simplcore_config(tok.vocab_size, hidden=64, layers=2)
    sc = SimplCoreMoEModel(sc_cfg)
    base = TransformerBaseline(
        vocab_size=tok.vocab_size, dim=64, layers=2,
        n_heads=2, n_kv_heads=1, ffn_mult=4, max_seq_len=64,
    )
    x = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    sc_logits = sc(x)
    base_logits = base(x)
    assert sc_logits.shape == (1, 5, tok.vocab_size)
    assert base_logits.shape == (1, 5, tok.vocab_size)
    assert torch.isfinite(sc_logits).all()
    assert torch.isfinite(base_logits).all()


def test_tiny_training_pipeline(tiny_corpus):
    torch.manual_seed(0)
    random.seed(0)
    texts = load_corpus(tiny_corpus)
    tok = CharTokenizer(texts, max_vocab=512)
    rng = random.Random(0)
    train_ids, val_ids = split_documents(texts, tok, 0.2, seed=0)
    train_batches = make_batches(train_ids, 2, 16, rng=rng)
    val_batches = make_batches(val_ids, 2, 16, rng=rng)

    sc_cfg = build_simplcore_config(tok.vocab_size, hidden=64, layers=2)
    sc = SimplCoreMoEModel(sc_cfg)
    sc_opt = torch.optim.AdamW(sc.parameters(), lr=1e-3)
    for _ in range(3):
        x, y = train_batches[0]
        sc_opt.zero_grad()
        logits = sc(x, training=True)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, tok.vocab_size), y.reshape(-1), ignore_index=0
        )
        if sc.last_aux_loss is not None:
            loss = loss + sc.last_aux_loss
        loss.backward()
        sc_opt.step()
    assert torch.isfinite(loss)

    # 验证集困惑度有限
    sc.eval()
    with torch.no_grad():
        total_nll = 0.0
        total_tokens = 0
        for x, y in val_batches:
            logits = sc(x)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, tok.vocab_size), y.reshape(-1),
                reduction="none", ignore_index=0,
            )
            mask = y.reshape(-1) != 0
            total_nll += nll[mask].sum().item()
            total_tokens += mask.sum().item()
    ppl = math.exp(total_nll / total_tokens)
    assert math.isfinite(ppl) and ppl > 1.0
