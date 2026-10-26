# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""真实中文语料加载器 (Real Chinese Corpus Loader).

设计目标
--------
彻底抛弃此前 "随机采样 + 噪声注入" 的合成数据生成器 (fast_dataset.py 之类) ——
那种数据没有语义, 训出来的模型学不到真实语言结构. 本模块加载 **真实、可读、
有语义** 的中文文本:

  * 支持从目录批量读取 UTF-8 纯文本 (.txt);
  * 以字符 (Unit) 为最小单位切分 (无词表, 见 unit_embedding.py);
  * 提供按 "领域 (domain)" 分桶的能力, 以便训练分域专家模型 (SimplUnity);
  * 干净预处理: 仅做空白归一与去不可见控制字符, **不做任何破坏语义的随机噪声**.

本模块不依赖 numpy, 可独立运行; 与 simplcore.tensor 解耦 (上层训练脚本负责
把 Unit 序列喂入 LatticeLM).
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# 默认语料目录 (相对于本文件)
_DEFAULT_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def _clean_text(text: str) -> str:
    """温和的预处理: 归一空白, 丢弃不可见控制字符 (保留中文标点与换行).

    不做任何破坏语义的操作 (如随机替换/插入噪声).
    """
    out = []
    for ch in text:
        o = ord(ch)
        # 丢弃 C0/C1 控制字符 (除换行/制表/回车 外)
        if (0 <= o <= 31 and ch not in "\n\r\t") or (0x7f <= o <= 0x9f):
            continue
        out.append(ch)
    s = "".join(out)
    # 归一空白: 把连续空白压成单空格, 但保留段落换行
    lines = []
    for line in s.split("\n"):
        lines.append(" ".join(line.split()))
    return "\n".join(lines)


@dataclass
class CorpusStats:
    """语料统计."""
    domain: str
    n_chars: int = 0
    n_units: int = 0          # 去重后的 Unit (字符) 数
    n_segments: int = 0       # 段落/样本数
    vocab: set = field(default_factory=set)


class RealCorpus:
    """真实中文语料集: 按领域分桶读取目录下的 .txt 文件.

    目录结构约定:
        corpus/
            classical/   古文 (论语/道德经/史记 等)
            modern/      现代文 (散文/小说节选)
            science/     科普/科技中文
            ...          任意自定义领域 (每个子目录 = 一个领域)

    若没有子目录, 则把该目录下的所有 .txt 当作单一领域 ("default").
    """

    def __init__(self, root_dir: Optional[str] = None) -> None:
        self.root = root_dir or _DEFAULT_CORPUS_DIR
        self.domains: Dict[str, List[str]] = {}      # domain -> list[text]
        self.stats: Dict[str, CorpusStats] = {}
        self._loaded = False

    # ---- 加载 ----------------------------------------------------------
    def load(self, verbose: bool = True) -> "RealCorpus":
        """扫描 root, 按子目录分域读取所有 .txt 内容."""
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"语料根目录不存在: {self.root}")
        self.domains.clear()
        self.stats.clear()

        subdirs = [d for d in os.listdir(self.root)
                   if os.path.isdir(os.path.join(self.root, d)) and not d.startswith(".")]
        if subdirs:
            for d in subdirs:
                self._load_domain(d)
        else:
            self._load_domain("default")

        self._loaded = True
        if verbose:
            total = sum(s.n_chars for s in self.stats.values())
            print(f"[Corpus] 已加载 {len(self.domains)} 个领域, 共 {total:,} 字符")
            for dom, st in self.stats.items():
                print(f"  - {dom}: {st.n_chars:,} 字 / {st.n_segments} 段 / "
                      f"{len(st.vocab)} 去重 Unit")
        return self

    def _load_domain(self, domain: str) -> None:
        dpath = os.path.join(self.root, domain)
        texts: List[str] = []
        n_chars = 0
        vocab: set = set()
        n_seg = 0
        for fp in sorted(glob.glob(os.path.join(dpath, "*.txt"))):
            with open(fp, "r", encoding="utf-8") as f:
                raw = f.read()
            clean = _clean_text(raw)
            # 按自然段落切分为样本 (空段丢弃), 既保留结构也便于批处理
            for seg in clean.split("\n"):
                seg = seg.strip()
                if len(seg) >= 8:        # 过滤过短无意义行
                    texts.append(seg)
                    n_chars += len(seg)
                    vocab.update(seg)
                    n_seg += 1
        self.domains[domain] = texts
        self.stats[domain] = CorpusStats(
            domain=domain, n_chars=n_chars, n_units=len(vocab),
            n_segments=n_seg, vocab=vocab,
        )

    # ---- 取用 ----------------------------------------------------------
    def get_domain_texts(self, domain: str) -> List[str]:
        if not self._loaded:
            self.load()
        return self.domains.get(domain, [])

    def list_domains(self) -> List[str]:
        if not self._loaded:
            self.load()
        return list(self.domains.keys())

    def global_vocab(self) -> set:
        """跨领域合并的去重 Unit 集 (用于构建紧凑 Embedding 表)."""
        if not self._loaded:
            self.load()
        v: set = set()
        for st in self.stats.values():
            v |= st.vocab
        return v

    def make_unit_map(self, min_freq: int = 1) -> "UnitMap":
        """构建 字符->整数索引 的紧凑映射 (仅含语料中实际出现的字符).

        Returns:
            UnitMap: 含 <pad>=0, <unk>=1 与所有出现字符的查表.
        """
        vocab = sorted(self.global_vocab())
        return UnitMap(vocab)


class UnitMap:
    """字符 <-> 整数索引 的紧凑映射 (无词表, 仅覆盖语料实际字符)."""

    PAD = 0
    UNK = 1

    def __init__(self, chars: List[str]) -> None:
        self.idx2ch = ["<pad>", "<unk>"] + list(chars)
        self.ch2idx = {c: i for i, c in enumerate(self.idx2ch)}
        self.size = len(self.idx2ch)

    def encode(self, text: str) -> List[int]:
        return [self.ch2idx.get(ch, self.UNK) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.idx2ch[i] if 0 <= i < self.size else "<unk>"
                       for i in ids)

    def __len__(self) -> int:
        return self.size


# ----------------------------------------------------------------------
# 便捷 API
# ----------------------------------------------------------------------
def load_corpus(root_dir: Optional[str] = None,
                verbose: bool = True) -> RealCorpus:
    """加载真实中文语料集."""
    return RealCorpus(root_dir).load(verbose=verbose)


if __name__ == "__main__":
    c = load_corpus(verbose=True)
    print("\n示例 (classical 前 60 字):")
    txts = c.get_domain_texts("classical")
    if txts:
        print(txts[0][:60])
    um = c.make_unit_map()
    print(f"\n紧凑 Unit 表大小: {len(um)} (含 <pad>/<unk>)")
    print("编码/解码一致性:", um.decode(um.encode("学而时习之")) == "学而时习之")
