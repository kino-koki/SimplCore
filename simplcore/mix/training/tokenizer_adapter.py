# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""Tokenizer 接入适配层 — 让训练/校准用真 BPE，无则回退字符哈希。

设计目标：
  训练代码只依赖一个统一的 ``encode(text, seq_len) -> LongTensor(seq_len+1)``
  接口。有真 tokenizer 就用 BPE（中文能力地基，修复命门 4）；
  没有就回退字符哈希（现状，保证不破坏任何已有路径）。

  这样你训出 tokenizer.json 后，只需给训练命令加 ``--tokenizer`` 即可启用，
  无需改训练逻辑。

接口：
  load_tokenizer(path, vocab_size) -> BPETokenizer | None
  make_encoder(tokenizer, vocab_size) -> Callable[[str, int], torch.Tensor]
  batch_stream_with_encoder(encoder, texts, batch_size, seq_len) -> Iterator
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import simplcore.tensor as torch

from simplcore.mix.training.fast_ops import encode_text_fast


def load_tokenizer(
    path: Optional[str], vocab_size: int, *, chinese: bool = True
):
    """加载 tokenizer；路径无效或未训练返回 None（调用方回退字符哈希）。"""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[tokenizer_adapter] 警告: tokenizer 文件不存在 {p}，回退字符哈希")
        return None
    try:
        from simplcore.mix.tokenizer import BPETokenizer

        tok = BPETokenizer(vocab_size=vocab_size, chinese=chinese)
        tok.load(str(p))
        if not tok.is_trained():
            print("[tokenizer_adapter] 警告: tokenizer 未训练，回退字符哈希")
            return None
        # vocab 一致性检查
        if len(tok) > vocab_size:
            print(f"[tokenizer_adapter] 警告: tokenizer vocab={len(tok)} 超过 "
                  f"模型 vocab={vocab_size}，将截断；建议重建模型配置")
        return tok
    except Exception as exc:
        print(f"[tokenizer_adapter] 警告: tokenizer 加载失败 ({exc})，回退字符哈希")
        return None


def make_encoder(tokenizer, vocab_size: int) -> Callable[[str, int], torch.Tensor]:
    """构造统一的 encode(text, seq_len) -> LongTensor(seq_len+1)。

    真 BPE：编码 → 截断/补 pad 到 seq_len+1（pad 用 tokenizer.pad_token_id=0，
    与字符哈希的 0 填充一致，保证 x/y shift 语义相同）。
    回退：字符哈希（与旧实现逐 id 一致）。
    """
    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id

        def encode_bpe(text: str, seq_len: int) -> torch.Tensor:
            ids = tokenizer.encode(text)
            ids = ids[: seq_len + 1]
            if len(ids) < seq_len + 1:
                ids = ids + [pad_id] * (seq_len + 1 - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        return encode_bpe

    def encode_hash(text: str, seq_len: int) -> torch.Tensor:
        return encode_text_fast(text, vocab_size, seq_len)

    return encode_hash


def batch_stream_with_encoder(
    encoder: Callable[[str, int], torch.Tensor],
    texts: Iterator[str],
    *,
    batch_size: int,
    seq_len: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """用给定 encoder 组装批次；与 batch_stream_fast 相同语义（不产出尾批）。"""
    batch: List[torch.Tensor] = []
    for text in texts:
        batch.append(encoder(text, seq_len))
        if len(batch) == batch_size:
            tokens = torch.stack(batch, dim=0)
            yield tokens[:, :-1], tokens[:, 1:]
            batch = []
    # 与旧实现一致，不足 batch_size 的尾批不产出
