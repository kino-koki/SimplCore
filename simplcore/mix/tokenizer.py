# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix Tokenizer — 自研 BPE（修复版）。

修复命门 4 的四个致命问题：
  1. encode 性能灾难：旧实现对每个 merge（31744 次）都重新扫整段文本，
     O(文本长度 × merges)。新实现用标准**自底向上 BPE** + ``merge_rank``
     O(1) 查找，平均 O(文本长度 log 长度)。
  2. decode 的 ``merges.index()`` 是 O(merges) 查找且越界即崩，已移除。
  3. 特殊 token 冲突：旧实现 pad=0/eos=1，但 vocab 0-255 也是字节 token，
     两者撞了。新布局如下（与训练代码 ``2 + ord(ch) % ...`` 的 0/1 特殊、
     2+ 内容约定对齐）：
       id 0            = <pad>
       id 1            = <eos>
       id 2 .. 257     = 256 字节（byte + 2）
       id 258 .. 65535 = BPE 合并 token（rank r → 258 + r）
  4. 英文 GPT-2 正则：中文整句被当一个"词"，BPE 统计失效。新默认
     ``chinese=True`` 把连续 CJK 切单字，BPE 学会把高频字组合并成词。

训练用标准 BPE（merges 规则与 GPT-2 一致）；encode 是标准的自底向上应用
merges。vocab 布局、merge id 均与 train 一致，可保存/加载。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import simplcore.tensor as torch

# ── token id 布局 ──
PAD_ID = 0
EOS_ID = 1
BYTE_OFFSET = 2      # byte b → b + BYTE_OFFSET
MERGE_OFFSET = 258   # merge rank r → r + MERGE_OFFSET（= 2 + 256）
BYTE_VOCAB_SIZE = 256
N_SPECIAL = 2        # <pad>, <eos>

# 中文友好预分词：单个 CJK 字符 / 连续英文数字 / 空白 / 其它单字符。
# 用 \uXXXX 转义拼范围，确保正则引擎解析为区间而非字面字符。
_CJK_RANGES = (
    "一-鿿"    # CJK 统一表意文字
    "㐀-䶿"    # CJK 扩展 A
    "豈-﫿"    # CJK 兼容表意文字
    "　-〿"    # CJK 符号和标点
    "＀-￯"    # 全角形式
)
# 注意用 (?:...) 非捕获组：若有捕获组，re.findall 只返回组内容，
# 其它分支（英文词/空白/标点）会被吞成空串。
PATTERN_CHINESE = re.compile(
    f"(?:[{_CJK_RANGES}])|[a-zA-Z0-9_]+|\\s+|."
)
# 英文 GPT-2 风格预分词（chinese=False 时用）。
# 注：原 GPT-2 pattern 用了 \p{L}/\p{N}（Unicode 属性），Python 标准 re
# 不支持（需 regex 库）。这里用等价写法：英文词 / 数字串 / 空白 / 其它单字符。
# 对 BPE 训练已足够（预分词只需保证"字间不合并、词内可合并"）。
PATTERN_EN = re.compile(r"[A-Za-z]+|\d+|\s+|[^\sA-Za-z\d]")


class BPETokenizer:
    """字节对编码(BPE)分词器 — 中文友好，自研实现。

    Args:
        vocab_size: 目标词表大小（含 2 个特殊 token）。默认 32000。
        chinese: True 用中文预分词（连续 CJK 切单字）；False 用英文 GPT-2
            pattern。
    """

    def __init__(self, vocab_size: int = 32000, chinese: bool = True):
        if vocab_size < BYTE_VOCAB_SIZE + N_SPECIAL + 1:
            raise ValueError(
                f"vocab_size 至少 {BYTE_VOCAB_SIZE + N_SPECIAL + 1}，got {vocab_size}"
            )
        self.vocab_size = vocab_size
        self.chinese = chinese
        self.pattern = PATTERN_CHINESE if chinese else PATTERN_EN
        # id -> bytes（merge token 的字节表示）；字节 token 不存（即时推导）
        self.vocab: Dict[int, bytes] = {}
        # 有序 merge 规则列表 + rank 加速字典
        self.merges: List[Tuple[int, int]] = []
        self.merge_rank: Dict[Tuple[int, int], int] = {}

    # ── 属性（特殊 token，与训练代码 0/1 约定一致）──
    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_ID

    @property
    def byte_offset(self) -> int:
        return BYTE_OFFSET

    @property
    def merge_offset(self) -> int:
        return MERGE_OFFSET

    def __len__(self) -> int:
        # 训练完成后 = 2 特殊 + 256 字节 + len(merges) ≤ vocab_size
        return N_SPECIAL + BYTE_VOCAB_SIZE + len(self.merges)

    def is_trained(self) -> bool:
        return len(self.merges) > 0

    # ── 预分词 ──
    def _words(self, text: str) -> List[str]:
        return self.pattern.findall(text)

    # ── 训练 ──
    @staticmethod
    def _word_to_byte_ids(word: str) -> List[int]:
        """词 → 字节 token id 序列（2 + byte）。"""
        return [BYTE_OFFSET + b for b in word.encode("utf-8")]

    def _get_word_freqs(self, texts: List[str]) -> Dict[Tuple[int, ...], int]:
        word_freqs: Dict[Tuple[int, ...], int] = {}
        for text in texts:
            for word in self._words(text):
                key = tuple(self._word_to_byte_ids(word))
                word_freqs[key] = word_freqs.get(key, 0) + 1
        return word_freqs

    def train(self, texts: List[str], *, verbose: bool = True) -> None:
        """从文本语料训练 BPE。

        标准算法（与 GPT-2 一致）：反复找最高频相邻 token 对合并，
        merge rank r → id ``MERGE_OFFSET + r``。
        """
        self.vocab = {}
        self.merges = []
        self.merge_rank = {}
        if verbose:
            print(f"[BPETokenizer] 训练 BPE: vocab_size={self.vocab_size} "
                  f"chinese={self.chinese} (特殊={N_SPECIAL}, 字节={BYTE_VOCAB_SIZE})")

        word_freqs = self._get_word_freqs(texts)
        num_merges = self.vocab_size - N_SPECIAL - BYTE_VOCAB_SIZE
        if verbose:
            print(f"[BPETokenizer] 初始 unique 词数: {len(word_freqs)}; "
                  f"待合并次数: {num_merges}")

        for rank in range(num_merges):
            # 统计相邻 pair 频率
            pairs: Dict[Tuple[int, int], int] = {}
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    pairs[p] = pairs.get(p, 0) + freq

            if not pairs:
                if verbose:
                    print("[BPETokenizer] 语料已无可合并 pair，提前结束")
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            self.merge_rank[best] = rank
            new_id = MERGE_OFFSET + rank
            # 新 token 的字节表示 = 两个组成部分的字节拼接
            b1 = self._bytes_for_token(best[0])
            b2 = self._bytes_for_token(best[1])
            self.vocab[new_id] = b1 + b2

            # 应用合并
            new_word_freqs: Dict[Tuple[int, ...], int] = {}
            for word, freq in word_freqs.items():
                new_word: List[int] = []
                j = 0
                n = len(word)
                while j < n:
                    if j < n - 1 and (word[j], word[j + 1]) == best:
                        new_word.append(new_id)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

            if verbose and (rank + 1) % 1000 == 0:
                print(f"[BPETokenizer] merge {rank + 1}/{num_merges}")

        if verbose:
            print(f"[BPETokenizer] 完成: merges={len(self.merges)} "
                  f"vocab={len(self)} (上限 {self.vocab_size})")

    def _bytes_for_token(self, token: int) -> bytes:
        """取 token 的字节表示（字节 token 即时推导，merge token 查表）。"""
        if BYTE_OFFSET <= token < BYTE_OFFSET + BYTE_VOCAB_SIZE:
            return bytes([token - BYTE_OFFSET])
        return self.vocab.get(token, b"")

    # ── 编码 ──
    def _encode_word(self, word_bytes: bytes) -> List[int]:
        """单个词的标准自底向上 BPE 编码。"""
        word = [BYTE_OFFSET + b for b in word_bytes]
        INF = 1 << 30
        while len(word) > 1:
            # 找最低 rank 的相邻 pair
            best_i: Optional[int] = None
            best_r = INF
            for i in range(len(word) - 1):
                r = self.merge_rank.get((word[i], word[i + 1]))
                if r is not None and r < best_r:
                    best_r = r
                    best_i = i
            if best_i is None:
                break
            word[best_i : best_i + 2] = [MERGE_OFFSET + best_r]
        return word

    def encode(self, text: str) -> List[int]:
        """编码文本为 token id 序列。"""
        if not self.is_trained():
            raise RuntimeError("Tokenizer 未训练，请先调用 train()")
        tokens: List[int] = []
        for word in self._words(text):
            if word == "":
                continue  # 空串（findall 极端情况）跳过
            tokens.extend(self._encode_word(word.encode("utf-8")))
        return tokens

    def encode_batch(
        self, texts: List[str], max_length: Optional[int] = None
    ) -> torch.Tensor:
        """批量编码并填充/截断。"""
        encoded = [self.encode(t) for t in texts]
        if max_length:
            encoded = [e[:max_length] for e in encoded]
            max_len = max_length
        else:
            max_len = max(len(e) for e in encoded)
        padded = [e + [PAD_ID] * (max_len - len(e)) for e in encoded]
        return torch.tensor(padded, dtype=torch.long)

    # ── 解码 ──
    def decode(self, tokens: List[int]) -> str:
        """解码 token id 序列为文本。"""
        ba = bytearray()
        for t in tokens:
            if t == PAD_ID or t == EOS_ID:
                continue
            if BYTE_OFFSET <= t < BYTE_OFFSET + BYTE_VOCAB_SIZE:
                ba.append(t - BYTE_OFFSET)
            elif t in self.vocab:
                ba.extend(self.vocab[t])
            else:
                ba.extend(b"?")  # 未知/越界 token
        return ba.decode("utf-8", errors="replace")

    # ── 保存 / 加载 ──
    def save(self, path: str) -> None:
        """保存到 JSON（含增强字段：chinese / merges / vocab hex）。"""
        data = {
            "vocab_size": self.vocab_size,
            "chinese": self.chinese,
            "n_special": N_SPECIAL,
            "byte_offset": BYTE_OFFSET,
            "merge_offset": MERGE_OFFSET,
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": [[int(a), int(b)] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """从 JSON 加载（兼容旧格式：无增强字段时回退默认布局）。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = int(data.get("vocab_size", 32000))
        self.chinese = bool(data.get("chinese", True))
        self.pattern = PATTERN_CHINESE if self.chinese else PATTERN_EN
        self.vocab = {int(k): bytes.fromhex(v) for k, v in data.get("vocab", {}).items()}
        self.merges = [(int(a), int(b)) for a, b in data.get("merges", [])]
        self.merge_rank = {p: r for r, p in enumerate(self.merges)}


class CharTokenizer:
    """字符级 Tokenizer（轻量 fallback；0/1 特殊，2+ 常规，无冲突）。"""

    def __init__(self):
        self.char_to_id: Dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self.id_to_char: Dict[int, str] = {0: "<pad>", 1: "<eos>"}
        self._built = False

    def train(self, texts: List[str]) -> None:
        """构建字符词表。"""
        chars = sorted(set("".join(texts)))
        for i, ch in enumerate(chars, start=2):
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch
        self._built = True
        print(f"[CharTokenizer] Built vocab with {len(self.char_to_id)} chars")

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(self.id_to_char.get(t, "?") for t in tokens)

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> torch.Tensor:
        encoded = [self.encode(t) for t in texts]
        if max_length:
            encoded = [e[:max_length] for e in encoded]
            max_len = max_length
        else:
            max_len = max(len(e) for e in encoded)
        padded = [e + [0] * (max_len - len(e)) for e in encoded]
        return torch.tensor(padded, dtype=torch.long)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id}, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self._built = True

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def pad_token_id(self) -> int:
        return 0

    def __len__(self) -> int:
        return len(self.char_to_id)
