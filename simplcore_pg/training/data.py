# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""中文语料数据流水线。

设计目标 (支撑"纯中文训练语料"策略)：
  * 分词器抽象：默认提供 ``SimpleChineseTokenizer`` (字/词混合 + byte 回退)，
    可在生产环境替换为中文优化 BPE (如训练好的 sentencepiece / Mistral 式
    tokenizer，vocab 含大量汉字与中文词，显著降低中文 token 膨胀——这是中文
    模型性价比的第一杠杆)；
  * 流式加载：``IterableCorpus`` 直接遍历 .jsonl/.txt 目录，支持超大规模语料；
  * 序列打包：``PackedDataset`` 将长文本切分/拼接为固定长度序列，配因果 LM
    标签 (pad 处 label=-100)，最大化训练密度；
  * 质量过滤与去重：提供可插拔钩子 (``quality_filter`` / ``dedup``) 占位，供
    接入真实清洗管线。

仅依赖标准库 + PyTorch，可在 CPU 上冒烟、在集群上扩展。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Callable, Iterable, List, Optional

import torch
from torch.utils.data import IterableDataset


# ----------------------------------------------------------------------
# 分词器
# ----------------------------------------------------------------------
class SimpleChineseTokenizer:
    """可运行的中文回退分词器 (字级 + 常见词表 + byte 回退)。

    生产环境应替换为中文优化 BPE (参数量与压缩率更优)；此处保证流水线可独立
    运行、可被真实 tokenizer 接口无缝替换 (``encode``/``decode``/``vocab_size``)。
    """

    PAD, UNK, BOS, EOS = 0, 1, 2, 3

    def __init__(self, vocab: Optional[dict] = None, max_len: int = 8192,
                 vocab_limit: Optional[int] = None) -> None:
        if vocab is None:
            # 基础字表：标点 + 常用汉字 (GB2312 一级 3755 字) 的占位生成
            vocab = self._build_default_vocab()
        self.token2id = vocab
        self.id2token = {v: k for k, v in vocab.items()}
        self.max_len = max_len
        # vocab_limit: 将超出模型词表的 id 映射为 UNK (保证与模型 vocab 对齐)
        self.vocab_limit = vocab_limit

    @staticmethod
    def _build_default_vocab() -> dict:
        vocab = {
            "<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
            " ": 4, "\n": 5, "\t": 6,
        }
        # 常用中文标点与 ASCII
        for ch in "，。、；：？！“”‘’（）【】《》—…·.,;:?!\"'()[]{}<>_-+*=/%#@&|^~`":
            if ch not in vocab:
                vocab[ch] = len(vocab)
        # 常用汉字 (GB2312 一级，约 3755 字；此处用 Unicode 区间近似生成示例)
        for cp in range(0x4E00, 0x4E00 + 3755):
            ch = chr(cp)
            if ch not in vocab:
                vocab[ch] = len(vocab)
        return vocab

    @classmethod
    def from_pretrained(cls, path: str) -> "SimpleChineseTokenizer":
        """从 JSON 词表加载 (生产可用，也可接入 sentencepiece 模型)。"""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = [self.BOS]
        for ch in text:
            i = self.token2id.get(ch, self.UNK)
            if self.vocab_limit is not None and i >= self.vocab_limit:
                i = self.UNK
            ids.append(i)
        ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            tok = self.id2token.get(int(i), "<unk>")
            if tok in ("<pad>", "<unk>", "<bos>", "<eos>"):
                continue
            out.append(tok)
        return "".join(out)


# ----------------------------------------------------------------------
# 语料迭代
# ----------------------------------------------------------------------
class IterableCorpus:
    """流式遍历语料目录 (.jsonl 每行 {"text":...} 或 .txt 纯文本)。"""

    def __init__(
        self,
        root: str,
        *,
        quality_filter: Optional[Callable[[str], bool]] = None,
        dedup: Optional[Callable[[str], bool]] = None,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.quality_filter = quality_filter
        self.dedup = dedup
        self.seed = seed
        self._seen = set()

    def _iter_files(self) -> Iterable[str]:
        for dirpath, _, files in os.walk(self.root):
            for fn in sorted(files):
                if fn.endswith(".jsonl") or fn.endswith(".txt"):
                    yield os.path.join(dirpath, fn)

    def _iter_texts(self) -> Iterable[str]:
        rng = random.Random(self.seed)
        files = list(self._iter_files())
        rng.shuffle(files)
        for fp in files:
            if fp.endswith(".jsonl"):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            text = obj.get("text") or obj.get("content") or ""
                        except json.JSONDecodeError:
                            continue
                        yield from self._emit(text)
            else:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        yield from self._emit(line)

    def _emit(self, text: str) -> Iterable[str]:
        text = (text or "").strip()
        if len(text) < 16:  # 过短样本丢弃
            return
        if self.quality_filter is not None and not self.quality_filter(text):
            return
        if self.dedup is not None:
            # 旧实现用内置 hash(text)：CPython 的字符串哈希按进程加盐
            # (PYTHONHASHSEED)，因此去重结果**不可复现**。改用稳定摘要。
            key = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
            if key in self._seen:
                return
            self._seen.add(key)
        yield text

    def __iter__(self):
        return self._iter_texts()


# ----------------------------------------------------------------------
# 打包数据集 (因果 LM)
# ----------------------------------------------------------------------
class PackedDataset(IterableDataset):
    """将语料流切分/拼接为固定长度序列，构造 (input_ids, labels)。

    labels 与 input_ids 相同，但序列末尾/填充位置置 -100 (忽略)，实现标准
    因果语言建模 (next-token prediction)。
    """

    def __init__(
        self,
        corpus: Iterable[str],
        tokenizer: SimpleChineseTokenizer,
        seq_len: int = 2048,
        min_seq: int = 32,
        buffer_mult: int = 16,
    ) -> None:
        self.corpus = corpus
        self.tok = tokenizer
        self.seq_len = seq_len
        self.min_seq = min_seq
        self.buffer_mult = buffer_mult

    def _token_stream(self) -> Iterable[int]:
        for text in self.corpus:
            ids = self.tok.encode(text)
            for i in ids:
                yield i

    def __iter__(self):
        buf: List[int] = []
        target = self.seq_len
        for tok_id in self._token_stream():
            buf.append(tok_id)
            if len(buf) >= target * self.buffer_mult:
                # 批量切分，减少小段开销
                while len(buf) >= target:
                    chunk = buf[:target]
                    del buf[:target]
                    yield self._pack(chunk)
                # 注意: 这里**不做** buf = buf[-min_seq:] 截断。
                # 旧实现那样写会静默丢弃最多 (seq_len - min_seq) 个 token
                # （审计实测每遍丢 602/195,226），且与"避免跨块断句丢失"的注释相反。
                # 残余缓冲整段带入下一批，token 零丢失。
        if len(buf) >= self.min_seq:
            # 末段填充到 seq_len；不足 min_seq 的尾巴按设计丢弃（已在文档说明）
            chunk = buf + [self.tok.PAD] * (target - len(buf))
            yield self._pack(chunk[:target])

    def _pack(self, ids: List[int]):
        ids_t = torch.tensor(ids, dtype=torch.long)
        labels = ids_t.clone()
        # 末尾若为 pad，则对应 label 忽略
        labels[ids_t == self.tok.PAD] = -100
        return {"input_ids": ids_t, "labels": labels}
