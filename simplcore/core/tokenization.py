# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 无词表模式：可插拔编解码器 + 嵌入/头可配.

传统 Transformer 把 tokenize 词表（BPE/SentencePiece）烧死在架构里：
词表 ID 直接就是 embedding 索引，输出层必须与词表 1:1 绑定。
SimplCore 把编解码器**从架构里摘出去**——同一套模型权重可切换
字符 / 字节 / BPE 序列，不重新初始化词表参数，只需适配层尺寸。

设计原则
========
* **纯 Python**：零第三方（不调用 transformers / tokenizers）；
  BPE 编解码基于标准 byte-pair 算法（sennrich 2016），可离线
  训练/加载任意 BPE 词表文件。
* **保守**：默认字符模式（最细粒度，无需外部词表，任意 Unicode 文本
  直接可用），BPE 需要用户提供词表路径。
* **嵌入/头可配**：嵌入层和输出头尺寸由编解码器决定
  （``vocab_size = codec.size()``），SimplCore 内部 mixer/FFN 结构
  对词表大小零假设。

API
---
* ``CharTokenizer()`` / ``ByteTokenizer()`` / ``BPETokenizer(...)``
  实现统一 ``Tokenizer`` 接口（``encode`` / ``decode`` / ``size`` /
  ``max_token_len``），可直接传给 ``SimplCoreConfig.vocab_size = codec.size()``
  或 ``init_simplcore_from_hf`` 的 ``cfg_override``。
* ``Tokenizer.encode(text) -> List[int]``，``decode(ids) -> str``。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

__all__ = ["Tokenizer", "CharTokenizer", "ByteTokenizer", "BPETokenizer",
           "CharVocab"]


# =============================================================================
# 统一接口
# =============================================================================
class Tokenizer:
    """编解码器统一接口：SGC 词表 ID = 该 codec 的 token ID."""

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    def size(self) -> int:
        """词表大小（= SGC embedding dim = vocab_size）."""
        raise NotImplementedError

    def max_token_len(self) -> int:
        """单个 token 可覆盖的最大 Unicode 码点数（用于 RoPE 缩放预算）."""
        return 1


# =============================================================================
# 字符模式：Unicode codepoint 直映（零外部词表）
# =============================================================================
class CharTokenizer(Tokenizer):
    """字符级 codec：每个 Unicode 码点 = 1 token.

    无需外部词表文件；适合快速原型。构造时可指定字符集上限：
    * ``limit`` 给定字符串/集合：只保留这些字符（高频字符集，ID 密集），
      其余字符映射到 0 号 ID（unknown）；
    * ``limit`` 为 int：保留前 ``limit`` 个 codepoint（0..limit-1）；
    * ``limit`` 为 None（默认）：保守取 ASCII 可打印字符 + 常见 CJK
      标点（~95 + 少量，远小于 0x10000，嵌入开销可接受）。
    """

    _DEFAULT_ASCII = "".join(chr(i) for i in range(32, 127)) + "\n\t\r"
    _DEFAULT_PUNCT = "。、！？；：""''（）《》【】—…·"

    def __init__(self, limit: Optional[object] = None,
                 unknown_id: int = 0) -> None:
        self._unknown = int(unknown_id)
        self._chars: List[str] = []
        self._lookup: Dict[str, int] = {}
        self._build(limit)

    def _build(self, limit: Optional[object]) -> None:
        if limit is None:
            charset = self._DEFAULT_ASCII + self._DEFAULT_PUNCT
        elif isinstance(limit, int):
            charset = "".join(chr(i) for i in range(limit))
        else:
            charset = "".join(limit)
        # 去重
        seen = set()
        for ch in charset:
            if ch not in seen:
                seen.add(ch)
                self._lookup[ch] = len(self._chars)
                self._chars.append(ch)

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for ch in text:
            out.append(self._lookup.get(ch, self._unknown))
        return out

    def decode(self, ids: List[int]) -> str:
        return "".join(self._chars[i] for i in ids
                       if 0 <= i < len(self._chars) and i != self._unknown) \
            + "\ufffd" * sum(1 for i in ids if i == self._unknown)

    def size(self) -> int:
        return len(self._chars)

    def max_token_len(self) -> int:
        return 1


# =============================================================================
# 字节模式：8-bit 码流（UTF-8 多字节序列）
# =============================================================================
class ByteTokenizer(Tokenizer):
    """字节级 codec：UTF-8 字节流，词表 = 256 个字节值 + 2 个特殊符号
    （BOS/EOS 占位，SGC 因果卷积不需要，保留仅为接口对齐）。

    词表仅 256，嵌入开销极小；适合 SGC 150M 以下小模型。
    """

    BUCKET = 256
    BOS, EOS = BUCKET, BUCKET + 1
    TOTAL = BUCKET + 2

    def __init__(self, add_bos_eos: bool = False) -> None:
        self._bos = self.BOS if add_bos_eos else None
        self._eos = self.EOS if add_bos_eos else None

    def encode(self, text: str) -> List[int]:
        ids = [b for b in text.encode("utf-8")]
        if self._bos is not None:
            ids = [self._bos] + ids
            if self._eos is not None:
                ids = ids + [self._eos]
        return ids

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i < self.BUCKET]
        return bytes(ids).decode("utf-8", errors="replace")

    def size(self) -> int:
        return self.TOTAL

    def max_token_len(self) -> int:
        return 4  # UTF-8 最多 4 字节/码点


# =============================================================================
# BPE 模式：标准 byte-pair 编解码（sennrich 2016 算法，纯 Python 实现）
# =============================================================================
class BPETokenizer(Tokenizer):
    """BPE codec：加载预训练 BPE 词表（JSON / HF 双文件格式），执行
    标准 byte-pair 合并。

    编码：UTF-8 字节 → 单字节字符序列 → 逐次应用最高优先级 merge，
    输出 token 字符串的 ID 序列。解码：ID → 字符串 → 字节 → UTF-8。
    """

    # 字节 0..255 映射为单字符（GPT-2 byte 编码约定，U+0000..U+00FF）
    _BYTE_TO_CHAR: List[str] = [chr(i) for i in range(256)]
    _CHAR_TO_BYTE: Dict[str, int] = {c: i for i, c in enumerate(_BYTE_TO_CHAR)}

    def __init__(self,
                 vocab: Dict[str, int],
                 merges: List[Tuple[str, str]],
                 unk_id: int = 256) -> None:
        self._vocab = dict(vocab)
        self._merges = list(merges)
        self._unk = unk_id
        self._rank: Dict[Tuple[str, str], int] = \
            {tuple(m): i for i, m in enumerate(merges)}
        # 反向：ID → token 字符串
        self._id2tok: Dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._id2tok[self._unk] = ""

    # -- 工厂方法 -------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str) -> "BPETokenizer":
        """从 JSON 词表文件加载：``{"vocab": {...}, "merges": [[a,b],...]}``."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(vocab=data.get("vocab", {}),
                   merges=[tuple(m) for m in data.get("merges", [])])

    @classmethod
    def from_hf(cls, vocab_json: str, merges_txt: str
                ) -> "BPETokenizer":
        """从 HuggingFace tokenizers 导出的双文件加载.

        ``vocab_json``: ``{token_str: id}``；``merges_txt``: 每行 ``a b``。
        合并顺序即文件行序（行号 = merge 优先级，行 0 最先应用）。
        """
        with open(vocab_json, encoding="utf-8") as fh:
            vocab = json.load(fh)
        merges: List[Tuple[str, str]] = []
        with open(merges_txt, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return cls(vocab=vocab, merges=merges)

    # -- 编解码 ----------------------------------------------------------------
    def _byte_tokenize(self, text: str) -> List[str]:
        """UTF-8 字节流 → 单字节字符序列（GPT-2 约定）."""
        return [self._BYTE_TO_CHAR[b] for b in text.encode("utf-8")]

    def _merge_step(self, pairs: List[str]) -> List[str]:
        """对当前 token 序列执行一次 BPE 合并（最高优先级 merge 优先）."""
        if len(pairs) < 2:
            return pairs
        best_rank = None
        best_pair = None
        for i in range(len(pairs) - 1):
            r = self._rank.get((pairs[i], pairs[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_pair = (pairs[i], pairs[i + 1])
        if best_pair is None:
            return pairs
        merged = best_pair[0] + best_pair[1]
        out: List[str] = []
        i = 0
        n = len(pairs)
        while i < n:
            if i < n - 1 and pairs[i] == best_pair[0] and pairs[i + 1] == best_pair[1]:
                out.append(merged)
                i += 2
            else:
                out.append(pairs[i])
                i += 1
        return out

    def encode(self, text: str) -> List[int]:
        """BPE 编码（堆驱动的 O(n log n) 实现）。

        旧实现每轮只合并**一个** pair，再从头重扫，复杂度是
        ``O(n · |merges|)``：12k merges 的 1KB 中文要跑上万次全表扫描，
        实测 500 字节约 1 分钟 —— 训练集根本切不出来。这不是"慢一点"，
        是"这条路线走不通"。

        现在改成标准的最小堆驱动写法（与 HF tokenizers / tiktoken 同构）：
        把所有可合并的相邻对按 merge 优先级入堆，每次弹出优先级最高的那一对，
        合并后只把新产生的两个邻接对入堆。合并顺序与逐轮扫描**完全等价**，
        复杂度降到 ``O(n log n)``。等价性由
        ``test_tokenization_speed.py::test_fast_matches_reference`` 用随机
        词表交叉验证（不是靠"看起来对"）。
        """
        toks = self._byte_tokenize(text)
        if len(toks) < 2:
            return [self._vocab.get(t, self._unk) for t in toks]
        return self._encode_heap(toks)

    def encode_reference(self, text: str) -> List[int]:
        """逐轮扫描的参考实现（仅用于交叉验证，慢）。"""
        toks = self._byte_tokenize(text)
        while len(toks) >= 2:
            prev = toks
            toks = self._merge_step(toks)
            if toks == prev:
                break
        return [self._vocab.get(t, self._unk) for t in toks]

    def _encode_heap(self, toks: List[str]) -> List[int]:
        import heapq

        n = len(toks)
        nxt = list(range(1, n)) + [-1]
        prev = [-1] + list(range(0, n - 1))
        alive = [True] * n
        heap: List[Tuple[int, int]] = []
        rank = self._rank
        for i in range(n - 1):
            r = rank.get((toks[i], toks[i + 1]))
            if r is not None:
                heap.append((r, i))
        heapq.heapify(heap)                       # 建堆是 O(n)，比逐个 push 快

        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if rank.get((toks[i], toks[j])) != r:
                continue                          # 过期条目（邻接已变）
            toks[i] = toks[i] + toks[j]
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prev[k] = i
            p = prev[i]
            if p != -1:
                rr = rank.get((toks[p], toks[i]))
                if rr is not None:
                    heapq.heappush(heap, (rr, p))
            if k != -1:
                rr = rank.get((toks[i], toks[k]))
                if rr is not None:
                    heapq.heappush(heap, (rr, i))

        out: List[int] = []
        i = 0
        while i != -1:
            out.append(self._vocab.get(toks[i], self._unk))
            i = nxt[i]
        return out

    def decode(self, ids: List[int]) -> str:
        """ID 序列 → 字符串（无损还原源字节流 → UTF-8）.

        标准 BPE 解码：每个 ID 映射到 token 字符串（合并 token 如 "he"
        展开为源字节字符序列），拼接后每个字符取 ord & 0xFF 作为源字节，
        再 UTF-8 decode。未知 ID 落到 ``_BYTE_TO_CHAR[i % 256]``。
        """
        tokens: List[str] = []
        for i in ids:
            tok = self._id2tok.get(i)
            if tok is None:
                tok = self._BYTE_TO_CHAR[i % 256]
            tokens.append(tok)
        byte_str = "".join(tokens)
        return bytes([ord(c) & 0xFF for c in byte_str]).decode(
            "utf-8", errors="replace")

    def size(self) -> int:
        return max(len(self._vocab), 257)

    def max_token_len(self) -> int:
        return 4


# =============================================================================
# 可保存的字符词表（中文小模型的主力 codec）
# =============================================================================
class CharVocab(Tokenizer):
    """从语料统计出来的字符词表，可存盘、可复现、有真正的 ``<unk>``。

    为什么不用现成的 :class:`CharTokenizer`
    ---------------------------------------
    ``CharTokenizer(limit=None)`` 的字符集只有 ASCII 可打印字符加少量中文标点，
    **任何汉字都会落到 ``unknown_id``** —— 中文语料进去是一串 0。它的 ``limit``
    参数虽然能传字符集合，但词表无法存盘，训练与推理必须用同一段代码重新
    构造，任何字符集差异都会静默改变 ID 含义。

    本类把"词表是训练产物"这件事显式化：``train()`` 从语料统计，
    ``save()/load()`` 固定下来，ID 0 恒为 ``<unk>``。

    ID 布局
    -------
    ``0`` = ``<unk>``，``1..n`` = 按频率降序的字符。频率顺序即 ID 顺序，
    这有实际好处：高频字符拿到小 ID，嵌入表的低行先被训好。
    """

    UNK_TOKEN = "<unk>"

    def __init__(self, chars, unk: str = UNK_TOKEN) -> None:
        body = [c for c in chars if c != unk]
        # 去重且保序
        seen, uniq = set(), []
        for c in body:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        self._chars = [unk] + uniq
        self._lookup = {c: i for i, c in enumerate(self._chars)}
        self._unk = 0

    # -- 训练 ---------------------------------------------------------------
    @classmethod
    def train(cls, texts, min_freq: int = 2, max_size: Optional[int] = None,
              extra_chars: Optional[str] = None, unk: str = UNK_TOKEN) -> "CharVocab":
        """统计字符频率并构造词表。

        Args:
            texts: 可迭代的文本片段。
            min_freq: 出现次数下限；低于它的字符归入 ``<unk>``。
            max_size: 词表上限（含 unk）。中文常用字约 3500，
                留出标点/数字/拉丁字母后 4096~6144 足够；上限越大，
                低频尾越长，小模型越训不动。
            extra_chars: 强制保留的字符（如 chat 模板里会出现的符号）。
        """
        from collections import Counter
        cnt: Counter = Counter()
        for t in texts:
            cnt.update(t or "")
        keep = [(c, n) for c, n in cnt.items() if n >= max(1, int(min_freq))]
        keep.sort(key=lambda x: (-x[1], x[0]))
        if max_size is not None:
            keep = keep[: max(1, int(max_size) - 1)]
        chars = [c for c, _ in keep]
        if extra_chars:
            present = set(chars)
            chars.extend(c for c in extra_chars if c not in present)
        vocab = cls(chars, unk=unk)
        vocab.freq = dict(cnt)                    # 便于审计压缩率
        return vocab

    # -- 编解码 -------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        get = self._lookup.get
        return [get(ch, self._unk) for ch in text or ""]

    def decode(self, ids) -> str:
        # <unk> 无法还原原字符，直接跳过（不伪造 "�" 来污染下游统计）
        return "".join(self._chars[i] for i in ids
                       if 0 < i < len(self._chars))

    def size(self) -> int:
        return len(self._chars)

    def max_token_len(self) -> int:
        return 1

    # -- 持久化 -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": "charvocab", "chars": self._chars}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharVocab":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("kind") != "charvocab":
            raise ValueError(f"{path} 不是 CharVocab 文件（kind={data.get('kind')!r}）")
        chars = data["chars"]
        if not chars or chars[0] != cls.UNK_TOKEN:
            raise ValueError(f"{path} 的 0 号 token 必须是 {cls.UNK_TOKEN!r}，"
                             f"否则 ID 含义已经漂移")
        return cls(chars[1:])

    # -- 审计 ---------------------------------------------------------------
    def coverage(self, texts) -> float:
        """在给定语料上的字符覆盖率（被计入词表的字符占比）。"""
        total = hit = 0
        for t in texts:
            for ch in t or "":
                total += 1
                if self._lookup.get(ch, 0) > 0:
                    hit += 1
        return hit / max(total, 1)


# =============================================================================
# 工厂 + 默认 codec
# =============================================================================
def make_codec(kind: str = "char", **kw) -> Tokenizer:
    """按类型名构造编解码器：``char`` / ``byte`` / ``bpe``.

    * ``char``: 透传 ``limit`` / ``unknown_id``；
    * ``byte``: 透传 ``add_bos_eos``；
    * ``bpe``: 需要 ``vocab`` + ``merges``（或 ``vocab_json`` +
      ``merges_txt`` 路径，自动走 ``BPETokenizer.from_hf``）。
    """
    if kind == "char":
        return CharTokenizer(limit=kw.pop("limit", None),
                             unknown_id=kw.pop("unknown_id", 0))
    if kind == "byte":
        return ByteTokenizer(add_bos_eos=kw.pop("add_bos_eos", False))
    if kind == "charvocab":
        # 可存盘的字符词表：``path=`` 加载已有词表，否则必须显式给 ``chars``
        path = kw.pop("path", None)
        if path:
            return CharVocab.load(path)
        chars = kw.pop("chars", None)
        if chars is None:
            raise ValueError("charvocab 需要 path= 或 chars=")
        return CharVocab(chars)
    if kind == "bpe":
        vocab_json = kw.pop("vocab_json", None)
        merges_txt = kw.pop("merges_txt", None)
        if vocab_json and merges_txt:
            codec = BPETokenizer.from_hf(vocab_json, merges_txt)
        else:
            codec = BPETokenizer(vocab=kw.pop("vocab", {}),
                                 merges=kw.pop("merges", []),
                                 unk_id=kw.pop("unk_id", 256))
        return codec
    raise ValueError(f"unknown codec kind: {kind!r}")
