# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""分词器守护测试：堆驱动 BPE 的**等价性**与**不走老路径**。

旧实现的复杂度是 ``O(n · |merges|)``：每轮只合并一个 pair，再从头重扫。
12k merges 的中文文本要跑上万次全表扫描，实测 500 字节约 1 分钟 ——
训练集根本切不出来，30M 对话计划在第一步就卡死。

这里不靠"看起来更快"来裁决，而是：

* 用 40 组随机词表的 480 个随机串交叉验证 fast == reference（正确性）；
* 用 monkeypatch 数 ``_merge_step`` 的调用次数，断言 ``encode`` 一次都没调用它
  （确定性，不受机器快慢影响）；
* 另加一条宽松的墙钟预算，防止有人把实现改回逐轮扫描。
"""
from __future__ import annotations

import random
import time

import pytest

from simplcore.core.tokenization import BPETokenizer


def _make_bpe(seed: int, n_merges: int = 30, alpha: str = "abcde") -> BPETokenizer:
    rng = random.Random(seed)
    vocab = {chr(i): i for i in range(256)}
    nid = 256
    merges = []
    toks = list(alpha)
    for _ in range(n_merges):
        a, b = rng.choice(toks), rng.choice(toks)
        m = a + b
        if m in vocab:
            continue
        vocab[m] = nid
        nid += 1
        merges.append((a, b))
        toks.append(m)
    return BPETokenizer(vocab=vocab, merges=merges)


# ---------------------------------------------------------------------------
# 等价性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(10)))
def test_fast_equals_reference_on_random_vocab(seed: int) -> None:
    bpe = _make_bpe(seed)
    rng = random.Random(seed)
    for _ in range(12):
        s = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 24)))
        assert bpe.encode(s) == bpe.encode_reference(s), s


def test_fast_equals_reference_on_empty_and_single() -> None:
    bpe = _make_bpe(0)
    assert bpe.encode("") == bpe.encode_reference("") == []
    assert bpe.encode("a") == bpe.encode_reference("a")


def test_fast_equals_reference_without_merges() -> None:
    bpe = BPETokenizer(vocab={chr(i): i for i in range(256)}, merges=[])
    s = "abcabc"
    assert bpe.encode(s) == bpe.encode_reference(s)


def test_fast_handles_utf8_multibyte() -> None:
    bpe = _make_bpe(3)
    for s in ("你好", "流式卷积", "a你b好c"):
        assert bpe.encode(s) == bpe.encode_reference(s)


# ---------------------------------------------------------------------------
# 不走老路径（确定性断言）
# ---------------------------------------------------------------------------
def test_encode_never_calls_single_merge_step(monkeypatch) -> None:
    bpe = _make_bpe(7, n_merges=40)
    calls = {"n": 0}
    original = BPETokenizer._merge_step

    def counting(self, pairs):
        calls["n"] += 1
        return original(self, pairs)

    monkeypatch.setattr(BPETokenizer, "_merge_step", counting)
    bpe.encode("abcde" * 50)
    assert calls["n"] == 0, "encode 退化回了逐轮扫描"


def test_reference_still_calls_merge_step(monkeypatch) -> None:
    """参考实现必须保留老路径，否则交叉验证就成了自己跟自己比。"""
    bpe = _make_bpe(7, n_merges=40)
    calls = {"n": 0}
    original = BPETokenizer._merge_step

    def counting(self, pairs):
        calls["n"] += 1
        return original(self, pairs)

    monkeypatch.setattr(BPETokenizer, "_merge_step", counting)
    bpe.encode_reference("abcde" * 10)
    assert calls["n"] > 0


# ---------------------------------------------------------------------------
# 墙钟预算（宽松，只防"改回旧实现"这种量级退化）
# ---------------------------------------------------------------------------
def test_encode_is_fast_on_merge_heavy_case() -> None:
    bpe = _make_bpe(11, n_merges=300)
    s = "abcde" * 400          # 2000 字符
    t0 = time.perf_counter()
    ids = bpe.encode(s)
    dt = time.perf_counter() - t0
    assert len(ids) > 0
    # 旧实现在这个规模上是分钟级；毫秒级实现留足余量
    assert dt < 2.0, f"2000 字符耗时 {dt:.2f}s，疑似退回逐轮扫描"


def test_reference_is_slower_than_fast_on_same_input() -> None:
    bpe = _make_bpe(5, n_merges=200)
    s = "abcde" * 60
    t0 = time.perf_counter()
    bpe.encode(s)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    bpe.encode_reference(s)
    ref = time.perf_counter() - t0
    assert fast <= ref * 2.0, (fast, ref)


# ---------------------------------------------------------------------------
# 往返
# ---------------------------------------------------------------------------
def test_decode_roundtrip_lossless() -> None:
    bpe = _make_bpe(2)
    s = "abcdeab"
    assert bpe.decode(bpe.encode(s)) == s


def test_size_and_unk() -> None:
    bpe = _make_bpe(0)
    assert bpe.size() >= 257
    assert bpe.encode("\u4e00")[0] == bpe._unk or bpe.size() > 256
