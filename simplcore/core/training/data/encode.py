# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""分词编码：tokenizer 适配器、EOS 注入、token 计数。"""
from __future__ import annotations



from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple




# ---------------------------------------------------------------------------
# 分词辅助
# ---------------------------------------------------------------------------
def _eos_of(tokenizer: Any) -> Optional[int]:
    """尽量从分词器对象上取 EOS id (兼容我们自己挂的属性与 HF 的原生属性)。"""
    eos = getattr(tokenizer, "eos_id", None)
    if eos is not None:
        return int(eos)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    return None


class _TokenizerAdapter:
    """把 HF ``tokenizers.Tokenizer`` 或自定义分词器包成统一接口。

    提供 :meth:`encode_all` / :meth:`count_tokens` / :attr:`eos_id`。
    HF 的 Rust 实现有 ``encode_batch``, 批编码比逐条 Python 循环快一个量级,
    打包 1.5 GB 语料时这个差别是分钟级的。
    """

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.raw = tokenizer
        self.eos_id = _eos_of(tokenizer)
        enc = getattr(tokenizer, "encode", None)
        if not callable(enc):
            raise TypeError(f"tokenizer of type {type(tokenizer).__name__} has no encode()")
        self._enc = enc

    def encode_all(self, texts: Sequence[str]) -> List[List[int]]:
        """批量编码; 优先走 HF 的 Rust ``encode_batch``。"""
        if not texts:
            return []
        batch_fn = getattr(self.raw, "encode_batch", None)
        if callable(batch_fn):
            return [list(enc.ids) for enc in batch_fn(list(texts))]
        return [list(self._enc(t)) for t in texts]

    def count_tokens(self, texts: Sequence[str]) -> List[int]:
        """每篇文档的 ``len(encode(text)) + 1`` (那个 +1 是文档间追加的 EOS)。"""
        if not texts:
            return []
        batch_fn = getattr(self.raw, "encode_batch", None)
        if callable(batch_fn):
            return [len(enc.ids) + 1 for enc in batch_fn(list(texts))]
        return [len(self._enc(t)) + 1 for t in texts]


def encode_document(tokenizer: Any, text: str, eos_id: Optional[int] = None) -> List[int]:
    """编码单篇文档并在尾部追加 EOS (返回 ``encode(text) + [eos_id]``)。"""
    if eos_id is None:
        eos_id = _eos_of(tokenizer)
    if eos_id is None:
        raise ValueError("eos_id is required (pass it explicitly or put it on the tokenizer)")
    return list(tokenizer.encode(text)) + [int(eos_id)]


def count_tokens(tokenizer: Any, text: str) -> int:
    """单篇文档编码后的 token 数 (含 1 个 EOS)。"""
    return len(tokenizer.encode(text)) + 1

