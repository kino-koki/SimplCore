# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""无词表单元嵌入 (Unit Embedding).

取代传统的 ``nn.Embedding(vocab_size, d)`` + 离散词表. 本模块把文本的最小处理
单位从 "Token (词表 id)" 改为 **Unit (单元)**: 文本模态下 Unit = Unicode 字符
(字素), 不依赖任何 BPE/WordPiece 词表; 可选字节级模式则下探到 UTF-8 字节.

设计要点
--------
1. **无词表**: 不再有 ``vocab_size`` 概念. 文本直接以字符/字节序列进入模型,
   多模态下各模态有其原生原子 (像素块、音频帧等), 统一称为 Unit.
2. **确定性编码**: ``encode_text`` 把字符串映射为 code-point 索引 (非负整数),
   不经过任何训练得到的分词器, 因此不会引入词表偏见或 OOV 问题.
3. **可学习查表**: 用固定规模的字符/字节嵌入表 (覆盖 Unicode BMP / ASCII 子集),
   规模远小于词表且语义中立; 不在推理期依赖外部 tokenizer 文件.
4. **与现有 LatticeCore 接口兼容**: 输出形状 (B, L, D), 可直接喂入 LatticeBlock.

术语对照 (旧 -> 新)
------------------
  Token / vocab_size      ->  Unit / unit_table_size (仅字符/字节规模, 非词表)
  tokenizer.encode        ->  UnitCodec.encode_text
  Embedding(vocab, d)     ->  UnitEmbedding(d, table_size)
"""

from __future__ import annotations

from typing import List

import simplcore.tensor as torch
import simplcore.tensor.nn as nn


class UnitCodec:
    """把文本编码为 Unit 序列 (字符/字节 code-point 索引), 无词表依赖."""

    def __init__(self, mode: str = "char") -> None:
        # mode: "char" -> Unicode code point; "byte" -> UTF-8 bytes
        self.mode = mode
        if mode == "char":
            # 覆盖常用 BMP + 扩展; 用 int 直接作索引, 无合并/统计
            self.table_size = 0x110000  # Unicode 码空间上限
        elif mode == "byte":
            self.table_size = 256
        else:
            raise ValueError("mode 仅支持 'char' 或 'byte'")

    def encode_text(self, text: str) -> List[int]:
        """字符串 -> Unit 索引列表."""
        if self.mode == "char":
            return [ord(ch) for ch in text]
        # byte mode: 逐字符取 UTF-8 字节
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: List[str]) -> "torch.Tensor":
        """批量编码 -> LongTensor (B, Li), 不等长用 0 左对齐填充."""
        seqs = [self.encode_text(t) for t in texts]
        max_len = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            for j, v in enumerate(s):
                out[i, j] = v
        return out


class UnitEmbedding(nn.Module):
    """无词表单元嵌入.

    Args:
        dim: 输出维度 D.
        mode: "char" 或 "byte".
        freeze: 是否冻结嵌入 (默认可训练).
        scale: 初始化幅度.
    """

    def __init__(self, dim: int, mode: str = "char", scale: float = 0.02):
        super().__init__()
        codec = UnitCodec(mode=mode)
        self.mode = mode
        self.table_size = codec.table_size
        # 可学习查表, 规模 = 字符/字节码空间, 不是词表
        self.weight = nn.Parameter(
            torch.randn(self.table_size, dim) * scale
        )
        self._codec = codec

    @property
    def codec(self) -> UnitCodec:
        return self._codec

    def forward(self, unit_ids: "torch.Tensor") -> "torch.Tensor":
        """unit_ids: (B, L) 非负整数 (code-point / byte). -> (B, L, D)."""
        # 越界保护 (容错, 不抛异常): 截断到合法范围
        idx = unit_ids.clamp(0, self.table_size - 1)
        return self.weight[idx]

    def encode_and_embed(self, text: str) -> "torch.Tensor":
        """便捷: 文本 -> (1, L, D) 嵌入 (无词表)."""
        ids = self._codec.encode_text(text)
        t = torch.tensor(ids, dtype=torch.long).reshape(1, -1)
        return self.forward(t)
