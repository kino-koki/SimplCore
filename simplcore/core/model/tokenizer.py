# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 分词器 —— 字节级 BPE 与字符级回退 (ByteLevel BPE tokenizer and char fallback).

本模块是 ``simplcore.core.data`` 的唯一分词依赖, 刻意保持"薄":
它只负责把一段文本变成 token id 序列, 以及把 token id 序列变回文本;
分词器之外的一切(清洗、打包、切窗、manifest)都在 ``data.py`` 里。

.. note::
   **与 ``simplcore.core.tokenization`` 的分工**（两者名字相近，容易混）：

   * ``simplcore.core.tokenization`` —— **免训练**的 codec 接口层：
     ``Tokenizer`` / ``CharTokenizer``（固定字符集）/ ``ByteTokenizer`` /
     ``BPETokenizer`` / ``make_codec``。模型、对话、训练链路用的是这一套。
   * 本模块 —— **语料拟合**的训练工具链：``train_bpe`` / ``train_char`` /
     ``train_tokenizer`` / ``load_tokenizer`` / ``save_tokenizer``，
     以及拟合型字符表 :class:`FittedCharTokenizer`。

   历史上两个模块都叫 ``CharTokenizer`` 但语义不同，已把本模块的改名为
   ``FittedCharTokenizer``，避免 ``from simplcore.core import CharTokenizer``
   与 ``from simplcore.core.tokenizer import CharTokenizer`` 拿到不同东西。

两条路线 (two interchangeable backends)
---------------------------------------
1. ``kind="bpe"`` —— HuggingFace ``tokenizers`` 的 **ByteLevel BPE**:
   NFKC 归一化 + 字节级字母表(256 个初值), 因此**任何**输入都可编码,
   不会出现 ``<unk>`` 丢字; 中文常用字的压缩比通常在 1.3-1.8 字符/token。
2. ``kind="char"`` —— 纯 Python 字符表回退: 语料里出现过的字符各占一个 id,
   未见字符回退到 ``<unk>``。训练是 O(N) 计数, 秒级完成,
   用途是"BPE 训练太慢/环境缺依赖时绝不让整条流水线硬失败"。

统一接口 (the duck-typed interface consumed by ``data.py``)
-----------------------------------------------------------
任何"分词器"对象只需提供::

    encode(text: str) -> list[int]
    decode(ids: Sequence[int]) -> str
    eos_id: int
    vocab_size: int

``data.pack_documents`` 只依赖这四项, 所以 BPE / char / 甚至是测试用的
假分词器都可以直接换进去。

**关于"往返是否逐字节一致"**: BPE 后端前置了 NFKC 归一化, 因此
``decode(encode(s))`` 等于 ``NFKC(s)`` 而**不一定**等于 ``s``
(例: 全角逗号 ``,`` 会被折成半角 ``,``, 连字/兼容汉字亦会折叠)。
这是有意为之 —— 训练语料也必须走同一条 NFKC 路径, 否则会出现
"训练看到的字面量 ≠ 推理输入的编码" 的静默不一致。
char 后端不做任何归一化, 因此在词表覆盖范围内是精确往返的。

Copyright 2026 (C) 六安市青奈科技有限责任公司
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_EOS_TOKEN",
    "DEFAULT_SPECIAL_TOKENS",
    "FittedCharTokenizer",
    "TokenizerInfo",
    "compression_ratio",
    "decode",
    "encode",
    "load_tokenizer",
    "main",
    "save_tokenizer",
    "train_bpe",
    "train_tokenizer",
]

_LOG = logging.getLogger("simplcore.core.tokenizer")

#: EOS 用 ASCII 单元分隔符 ``\x1f`` (Unit Separator), 而不是 ``<|eos|>``。
#: 理由: 真实中文网页文本偶尔会出现 ``<|eos|>`` 这类字面量; 若它同时又是
#: 特殊 token, 就会出现 "特殊 token 劫持真实文本" 的静默数据损坏。``\x1f``
#: 在清洗阶段已被 `data.clean_text` 全部剥离, 因此不可能与正文冲突。
DEFAULT_EOS_TOKEN = "\x1f"

#: 默认特殊 token 表。顺序即 id 顺序 (0, 1, 2, ...), 训练/推理必须一致。
DEFAULT_SPECIAL_TOKENS: Tuple[str, ...] = ("<pad>", "<unk>", DEFAULT_EOS_TOKEN)


# ---------------------------------------------------------------------------
# 公开数据类
# ---------------------------------------------------------------------------
@dataclass
class TokenizerInfo:
    """分词器训练结果的度量 (measurements of one trained tokenizer).

    Attributes:
        kind: ``"bpe"`` 或 ``"char"``; 与磁盘上的 ``tokenizer_meta.json`` 同源。
        vocab_size: 实际词表大小 (可能略大于请求值, 因为特殊 token 与字节表被强制保留)。
        requested_vocab_size: 调用方请求的词表大小。
        special_tokens: 特殊 token 列表, 顺序即 id 顺序。
        eos_id: EOS 的 id, 由 ``encode()`` 的调用方在文档之间插入。
        eos_token: EOS 的字面量。
        unk_id: ``<unk>`` 的 id (char 模式下有实际意义)。
        compression_ratio: **实测** chars per token = 语料字符数 / token 数。
            仅在不使用 ``encode_batch`` 的逐条编码路径上测量 (见 ``train_tokenizer``),
            否则该字段为 ``None``。
        sample_chars: 测压缩比时消耗的字符数。
        sample_tokens: 对应的 token 数。
        train_seconds: 训练耗时 (秒)。
        path: 保存目录。
        tokenizer_json: 磁盘上 ``tokenizer.json`` 的路径。
    """

    kind: str
    vocab_size: int
    requested_vocab_size: int
    special_tokens: Tuple[str, ...]
    eos_id: int
    eos_token: str
    unk_id: Optional[int]
    compression_ratio: Optional[float] = None
    sample_chars: int = 0
    sample_tokens: int = 0
    train_seconds: float = 0.0
    path: Optional[str] = None
    tokenizer_json: Optional[str] = None

    def to_dict(self) -> dict:
        """JSON-able view, written into the corpus manifest."""
        return {
            "kind": self.kind,
            "vocab_size": int(self.vocab_size),
            "requested_vocab_size": int(self.requested_vocab_size),
            "special_tokens": list(self.special_tokens),
            "eos_id": int(self.eos_id),
            "eos_token": self.eos_token,
            "unk_id": None if self.unk_id is None else int(self.unk_id),
            "compression_ratio": None if self.compression_ratio is None else round(float(self.compression_ratio), 6),
            "sample_chars": int(self.sample_chars),
            "sample_tokens": int(self.sample_tokens),
            "train_seconds": round(float(self.train_seconds), 3),
            "path": self.path,
            "tokenizer_json": self.tokenizer_json,
        }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def compression_ratio(tokenizer: Any, texts: Iterable[str], max_chars: Optional[int] = None) -> Tuple[float, int, int]:
    """实测压缩比 chars per token。

    Args:
        tokenizer: 任何提供 ``encode`` 的对象。
        texts: 文本可迭代 (generator 亦可, 只消费到 ``max_chars`` 为止)。
        max_chars: 最多统计多少字符; ``None`` 表示全部。

    Returns:
        ``(ratio, chars, tokens)``。ratio = chars / tokens; 无字符时返回 ``(0.0, 0, 0)``。
    """
    chars = 0
    tokens = 0
    for text in texts:
        if not text:
            continue
        chars += len(text)
        tokens += len(tokenizer.encode(text))
        if max_chars is not None and chars >= max_chars:
            break
    ratio = (chars / tokens) if tokens else 0.0
    return ratio, chars, tokens


def _as_token_list(out: Any) -> list:
    """把 HF ``Encoding`` / 原生 list 统一成 ``list[int]``。"""
    if isinstance(out, list):
        return [int(x) for x in out]
    ids = getattr(out, "ids", None)
    if ids is None:
        raise TypeError(f"encode() returned {type(out).__name__}, expected list or Encoding")
    return [int(x) for x in ids]


def _check_special_tokens_for_specials(special_tokens: Sequence[str]) -> None:
    for tok in special_tokens:
        if tok and all(ch.isalnum() or ch in "_-./" for ch in tok):
            _LOG.warning(
                "special token %r looks like ordinary text; a byte-level BPE tokenizer "
                "cannot represent it losslessly. Prefer markup like '<|...|>' or a control char.",
                tok,
            )


# ---------------------------------------------------------------------------
# BPE 后端
# ---------------------------------------------------------------------------
def train_bpe(
    files_or_iter: Any,
    vocab_size: int,
    out_dir: Any,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    *,
    min_frequency: int = 2,
    sample_chars: Optional[int] = None,
    measure_ratio: bool = True,
) -> TokenizerInfo:
    """训练 ByteLevel BPE 并落盘 ``<out_dir>/tokenizer.json``。

    Args:
        files_or_iter: 语料来源。三种形式均可:
            * ``Path`` / ``str`` 指向单个文本文件, 或指向一个目录(递归取 ``*.txt``/``*.jsonl``);
            * 一个可迭代对象, 逐项产出 ``str`` 文本 (``data.iter_documents`` 的输出即可);
            * ``(可迭代对象, 已采样文本列表)`` 二元组 —— 复用已经采样过的文本,
              避免为了测压缩比二次读盘。
        vocab_size: 目标词表大小 (含特殊 token 与 ByteLevel 的 256 个字节)。
        out_dir: 输出目录, 不存在则创建。
        special_tokens: 特殊 token, 顺序即 id 顺序。EOS 必须在内。
        min_frequency: 一个 merge 至少出现的次数; 默认 2。
        sample_chars: 只在这个字符预算内测压缩比 (BPE 训练本身消耗整个迭代器)。
        measure_ratio: 是否另行逐条编码量测压缩比。**大语料上关掉它**,
            否则这一步会显著拖慢 (逐条 Python 循环)。

    Returns:
        :class:`TokenizerInfo`, 含实测压缩比与 ``eos_id``。
    """
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if vocab_size < 16:
        raise ValueError(f"vocab_size must be >= 16, got {vocab_size}")
    special_tokens = tuple(special_tokens)
    _check_special_tokens_for_specials(special_tokens)

    cached_texts: Optional[list] = None
    if isinstance(files_or_iter, tuple) and len(files_or_iter) == 2:
        files_or_iter, cached_texts = files_or_iter
        cached_texts = list(cached_texts)
    elif not isinstance(files_or_iter, (str, os.PathLike)) and measure_ratio:
        # 训练要吃掉整个迭代器, 而压缩比量测需要**可重放**的来源, 所以这里先物化。
        # 调用方若要控制内存, 请显式传 (iter, cached_list) 二元组。
        files_or_iter = list(files_or_iter)

    raw_iter = _as_text_iterator(files_or_iter, cached_texts)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=list(special_tokens),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _LOG.info("training ByteLevel BPE: vocab_size=%d special_tokens=%s", vocab_size, list(special_tokens))
    tok.train_from_iterator(raw_iter, trainer=trainer)
    train_seconds = time.time() - t0

    tok_path = out_dir / "tokenizer.json"
    tok.save(str(tok_path))

    specials = {name: tok.token_to_id(name) for name in special_tokens}
    eos_token = DEFAULT_EOS_TOKEN if DEFAULT_EOS_TOKEN in specials else special_tokens[-1]
    eos_id = specials[eos_token]
    if eos_id is None:
        raise RuntimeError(f"tokenizer did not keep EOS token {eos_token!r}")

    ratio = None
    n_chars = 0
    n_tokens = 0
    if measure_ratio:
        ratio_texts = _ratio_text_source(files_or_iter, cached_texts)
        ratio, n_chars, n_tokens = compression_ratio(tok, ratio_texts, max_chars=sample_chars)

    info = TokenizerInfo(
        kind="bpe",
        vocab_size=int(tok.get_vocab_size()),
        requested_vocab_size=int(vocab_size),
        special_tokens=special_tokens,
        eos_id=int(eos_id),
        eos_token=eos_token,
        unk_id=specials.get("<unk>"),
        compression_ratio=ratio,
        sample_chars=n_chars,
        sample_tokens=n_tokens,
        train_seconds=train_seconds,
        path=str(out_dir),
        tokenizer_json=str(tok_path),
    )
    _write_meta(out_dir, info)
    _LOG.info(
        "BPE ready: vocab=%d eos_id=%d compression=%.4f chars/token (%.1fs)",
        info.vocab_size, info.eos_id, info.compression_ratio or 0.0, info.train_seconds,
    )
    return info


# ---------------------------------------------------------------------------
# char 回退后端
# ---------------------------------------------------------------------------
@dataclass
class FittedCharTokenizer:
    """字符级分词器（**语料拟合**版, character-level fallback backend）。

    .. note::
       与 :class:`simplcore.core.tokenization.CharTokenizer` **不是同一个类**：
       后者是免训练的固定字符集 codec（模型/对话链路用的就是它），
       本类必须先用 :meth:`from_corpus` 或 :meth:`load` 拟合出词表。
       历史上两者同名，导致 ``from simplcore.core import CharTokenizer`` 与
       ``from simplcore.core.tokenizer import CharTokenizer`` 拿到不同语义的对象。

    语料里出现过的字符各占一个 id; 未见字符统一回退到 ``<unk>``。
    训练只是 ``Counter`` 计数, 所以它在任何规模的语料上都是秒级完成,
    用来兜住 "BPE 训练太慢/依赖缺失" 这一类失败。

    Attributes:
        vocab: ``token_id -> 字符`` 的映射, 第 0 项起是特殊 token。
        special_tokens: 特殊 token 列表, 顺序即 id 顺序。
        eos_id: EOS 的 id。
        unk_id: ``<unk>`` 的 id, 未登录字符的落点。
        min_count: 建表时丢弃出现次数低于该值的字符。
        char_count: 真正进入词表的非特殊字符数。
    """

    vocab: list = field(default_factory=list)
    special_tokens: Tuple[str, ...] = DEFAULT_SPECIAL_TOKENS
    eos_id: int = 2
    unk_id: int = 1
    min_count: int = 1
    char_count: int = 0
    #: 实测压缩比 (chars per token); ``None`` 表示未量测。
    compression_ratio: Optional[float] = None
    sample_chars: int = 0
    sample_tokens: int = 0
    _stoi: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # 空词表时用特殊 token 兜底：否则 ``_rebuild`` 找不到 EOS 会 KeyError，
        # 让 ``FittedCharTokenizer()`` 直接崩掉 —— 一个"必须先训练才能用"的类
        # 不该在构造阶段就炸。真正拟合请用 :meth:`from_corpus` / :meth:`load`。
        if not self.vocab:
            self.vocab = list(self.special_tokens)
        self._rebuild()

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_corpus(
        cls,
        texts: Iterable[str],
        *,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
        min_count: int = 1,
        max_chars: Optional[int] = None,
        max_vocab: Optional[int] = None,
        measure_ratio: bool = True,
    ) -> "FittedCharTokenizer":
        """从文本迭代器统计字符表。

        Args:
            texts: 文本可迭代 (可被消费一次; 需要测压缩比时会在内存里留一份列表,
                因此请配合 ``max_chars`` 限制规模)。
            special_tokens: 特殊 token, 顺序即 id 顺序。
            min_count: 出现次数低于该值的字符不进表。
            max_chars: 最多统计多少字符。
            max_vocab: 词表上限 (超出时按频次截断)。
            measure_ratio: 是否实测压缩比。
        """
        special_tokens = tuple(special_tokens)
        counter: Counter = Counter()
        kept_texts: list = [] if measure_ratio else None
        n = 0
        for text in texts:
            if not text:
                continue
            counter.update(text)
            n += len(text)
            if kept_texts is not None:
                kept_texts.append(text)
            if max_chars is not None and n >= max_chars:
                break

        chars = [(ch, c) for ch, c in counter.items() if c >= min_count]
        # 频次降序, 同频按码位稳定排序 —— 保证多进程/多次运行结果一致。
        chars.sort(key=lambda kv: (-kv[1], kv[0]))
        if max_vocab is not None:
            keep = max(int(max_vocab) - len(special_tokens), 1)
            chars = chars[:keep]
        alphabet = [ch for ch, _ in chars]

        vocab = list(special_tokens) + alphabet
        obj = cls(vocab=vocab, special_tokens=special_tokens, min_count=int(min_count), char_count=len(alphabet))
        if measure_ratio and kept_texts:
            ratio, n_chars, n_tokens = compression_ratio(obj, kept_texts)
            obj.compression_ratio = ratio
            obj.sample_chars = n_chars
            obj.sample_tokens = n_tokens
        return obj

    def _rebuild(self) -> None:
        self._stoi = {ch: i for i, ch in enumerate(self.vocab)}
        if "<unk>" in self._stoi:
            self.unk_id = int(self._stoi["<unk>"])
        if DEFAULT_EOS_TOKEN in self._stoi:
            self.eos_id = int(self._stoi[DEFAULT_EOS_TOKEN])
        elif self.special_tokens:
            # 用 .get 而不是 []：词表里没有 EOS 时不该崩，退回默认 eos_id
            eos = self._stoi.get(self.special_tokens[-1])
            if eos is not None:
                self.eos_id = int(eos)

    # -- 接口 ---------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """词表大小 (含特殊 token)。"""
        return len(self.vocab)

    def encode(self, text: str) -> list:
        """字符 -> id; 未登录字符落到 ``unk_id``。"""
        if not text:
            return []
        stoi = self._stoi
        unk = self.unk_id
        return [stoi.get(ch, unk) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        """id -> 文本; 特殊 token 被丢弃 (与 HF ``skip_special_tokens=True`` 同义)。"""
        out = []
        specials = set(self.special_tokens)
        for i in ids:
            ch = self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else ""
            if ch in specials:
                continue
            out.append(ch)
        return "".join(out)

    # -- 落盘 ---------------------------------------------------------------
    def save(self, out_dir: Any) -> str:
        """把词表写成 ``<out_dir>/tokenizer.json`` (本模块自定义的 char 格式)。"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tokenizer.json"
        payload = {
            "simplcore_tokenizer": "char",
            "version": 1,
            "special_tokens": list(self.special_tokens),
            "min_count": int(self.min_count),
            "char_count": int(self.char_count),
            "vocab": list(self.vocab),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(path)

    @classmethod
    def load(cls, path: Any) -> "FittedCharTokenizer":
        """读回 :meth:`save` 写出的词表。"""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vocab=list(payload["vocab"]),
            special_tokens=tuple(payload.get("special_tokens", DEFAULT_SPECIAL_TOKENS)),
            min_count=int(payload.get("min_count", 1)),
            char_count=int(payload.get("char_count", 0)),
        )


def train_char(
    texts: Iterable[str],
    vocab_size: int,
    out_dir: Any,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    *,
    min_count: int = 1,
    sample_chars: Optional[int] = None,
    measure_ratio: bool = True,
) -> TokenizerInfo:
    """训练字符级分词器并落盘 (见 :class:`FittedCharTokenizer`)。"""
    t0 = time.time()
    tok = FittedCharTokenizer.from_corpus(
        texts,
        special_tokens=special_tokens,
        min_count=min_count,
        max_chars=sample_chars,
        max_vocab=int(vocab_size),
        measure_ratio=measure_ratio,
    )
    tok_path = tok.save(out_dir)
    out_dir = Path(out_dir)
    info = TokenizerInfo(
        kind="char",
        vocab_size=tok.vocab_size,
        requested_vocab_size=int(vocab_size),
        special_tokens=tuple(special_tokens),
        eos_id=int(tok.eos_id),
        eos_token=DEFAULT_EOS_TOKEN,
        unk_id=int(tok.unk_id),
        compression_ratio=tok.compression_ratio,
        sample_chars=int(tok.sample_chars),
        sample_tokens=int(tok.sample_tokens),
        train_seconds=time.time() - t0,
        path=str(out_dir),
        tokenizer_json=tok_path,
    )
    _write_meta(out_dir, info)
    _LOG.info(
        "char tokenizer ready: vocab=%d eos_id=%d compression=%.4f chars/token (%.1fs)",
        info.vocab_size, info.eos_id, info.compression_ratio or 0.0, info.train_seconds,
    )
    return info


def train_tokenizer(
    files_or_iter: Any,
    vocab_size: int,
    out_dir: Any,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    *,
    kind: str = "bpe",
    sample_chars: Optional[int] = None,
    min_frequency: int = 2,
    min_count: int = 1,
    measure_ratio: bool = True,
) -> TokenizerInfo:
    """统一入口: ``kind="bpe"`` 或 ``kind="char"``。"""
    if kind == "bpe":
        return train_bpe(
            files_or_iter, vocab_size, out_dir, special_tokens,
            min_frequency=min_frequency, sample_chars=sample_chars, measure_ratio=measure_ratio,
        )
    if kind == "char":
        cached = None
        if isinstance(files_or_iter, tuple) and len(files_or_iter) == 2:
            files_or_iter, cached = files_or_iter
        return train_char(
            _as_text_iterator(files_or_iter, cached), vocab_size, out_dir, special_tokens,
            min_count=min_count, sample_chars=sample_chars, measure_ratio=measure_ratio,
        )
    raise ValueError(f"unknown tokenizer kind: {kind!r} (expected 'bpe' or 'char')")


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------
def load_tokenizer(path: Any):
    """从目录或 ``tokenizer.json`` 路径载入分词器。

    自动判别后端: 文件里带 ``"simplcore_tokenizer": "char"`` 走 :class:`FittedCharTokenizer`,
    否则交给 ``tokenizers.Tokenizer.from_file``。

    Returns:
        提供 ``encode`` / ``decode`` / ``eos_id`` / ``vocab_size`` 的对象。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: JSON 存在但两个后端都不认。
    """
    p = Path(path)
    if p.is_dir():
        p = p / "tokenizer.json"
    if not p.exists():
        raise FileNotFoundError(f"tokenizer not found: {p}")
    head = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(head, dict) and head.get("simplcore_tokenizer") == "char":
        return FittedCharTokenizer.load(p)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(p))

    # HF tokenizers 0.2x 上 added_tokens 可能在顶层, 也可能嵌在 model 下。
    candidates = list(head.get("added_tokens", []) or [])
    model = head.get("model") or {}
    if not candidates and isinstance(model, dict):
        candidates = list(model.get("added_tokens", []) or []) if isinstance(model.get("added_tokens"), list) else []
    specials = [c.get("content") for c in candidates if isinstance(c, dict) and c.get("content")]

    eos_token = DEFAULT_EOS_TOKEN if DEFAULT_EOS_TOKEN in specials else (specials[-1] if specials else None)
    eos_id = None
    if eos_token is not None:
        eos_id = tok.token_to_id(eos_token)
    if eos_id is None:
        _LOG.warning("no EOS token found in %s; pack_documents needs an explicit eos_id", p)
        eos_id = -1
    tok.eos_id = int(eos_id)
    tok.eos_token = eos_token
    tok.vocab_size = int(tok.get_vocab_size())
    unk = tok.token_to_id("<unk>")
    tok.unk_id = None if unk is None else int(unk)
    return tok


def save_tokenizer(tokenizer: Any, out_dir: Any) -> str:
    """把分词器存到 ``<out_dir>/tokenizer.json``, 返回文件路径。

    兼容两种 ``save`` 语义: :class:`FittedCharTokenizer` 收目录,
    HF ``tokenizers.Tokenizer`` 收文件路径。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "tokenizer.json"
    if isinstance(tokenizer, FittedCharTokenizer):
        return tokenizer.save(out_dir)
    if not hasattr(tokenizer, "save"):
        raise TypeError(f"object of type {type(tokenizer).__name__} cannot be saved")
    tokenizer.save(str(target))
    return str(target)


def encode(tokenizer: Any, text: str) -> list:
    """把文本编码成 ``list[int]`` (统一 HF ``Encoding`` 与原生 list 的差异)。"""
    return _as_token_list(tokenizer.encode(text))


def decode(tokenizer: Any, ids: Sequence[int]) -> str:
    """把 id 序列解码回文本。"""
    return tokenizer.decode([int(i) for i in ids])


# ---------------------------------------------------------------------------
# 内部
# ---------------------------------------------------------------------------
def _write_meta(out_dir: Path, info: TokenizerInfo) -> None:
    """写 ``tokenizer_meta.json``, 便于脱离 manifest 单独查分词器配置。"""
    path = Path(out_dir) / "tokenizer_meta.json"
    path.write_text(json.dumps(info.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


_TEXT_SUFFIXES = (".txt", ".jsonl", ".text", ".raw")


def _as_text_iterator(src: Any, cached: Optional[list]) -> Iterator[str]:
    """把 ``str`` / ``Path`` / 可迭代对象统一成文本迭代器。"""
    if cached:
        return iter(cached)
    if isinstance(src, (str, os.PathLike)):
        p = Path(src)
        if p.is_file():
            return _iter_text_file(p)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES)
            if not files:
                raise FileNotFoundError(f"no text files (*.txt/*.jsonl) under {p}")
            return _iter_text_files(files)
        raise FileNotFoundError(f"tokenizer corpus path not found: {p}")
    return iter(src)


def _ratio_text_source(src: Any, cached: Optional[list]) -> Iterable[str]:
    """给压缩比量测准备**可重放**的文本来源。

    ``train_from_iterator`` 会吃掉迭代器, 所以量测绝不能复用同一个迭代器对象,
    否则压缩比会静默地算成 0 (这正是早期版本的一个真实缺陷)。
    语料是**生成器**时无法重放, 只能先物化 —— 因此调用方应主动限制
    ``sample_chars`` (见 :func:`train_bpe` 的 ``(iter, cached_list)`` 约定)。
    """
    if cached:
        return cached
    if isinstance(src, (str, os.PathLike)):
        return _as_text_iterator(src, None)
    return list(src)


def _iter_text_files(files: Sequence[Path]) -> Iterator[str]:
    for f in files:
        yield from _iter_text_file(f)


def _iter_text_file(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """构造 ``python -m simplcore.core.tokenizer --help`` 用的参数解析器。"""
    p = argparse.ArgumentParser(
        prog="python -m simplcore.core.tokenizer",
        description="用文本语料训练 SGC 分词器 (ByteLevel BPE 或 char 回退)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="文本文件或目录 (目录递归取 *.txt/*.jsonl)")
    p.add_argument("--out", required=True, help="输出目录, 写入 tokenizer.json 与 tokenizer_meta.json")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--kind", choices=("bpe", "char"), default="bpe")
    p.add_argument("--sample-chars", type=int, default=50_000_000, help="只在该字符预算内测压缩比")
    p.add_argument("--min-frequency", type=int, default=2, help="BPE merge 的最小频次")
    p.add_argument("--min-count", type=int, default=1, help="char 模式下字符的最小频次")
    p.add_argument("--no-ratio", action="store_true", help="跳过压缩比量测 (大语料更快)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m simplcore.core.tokenizer`` 的入口。"""
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    info = train_tokenizer(
        args.input, args.vocab_size, args.out,
        kind=args.kind,
        sample_chars=args.sample_chars,
        min_frequency=args.min_frequency,
        min_count=args.min_count,
        measure_ratio=not args.no_ratio,
    )
    print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
