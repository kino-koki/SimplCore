# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""O(1) 状态的检索容量账（原创清单 B 的可证伪起点）。

先澄清一个容易被压缩比带偏的问题：

* 论文 §5.12 用 "1M token 原文 ≥ 2GB vs 30.4MiB 状态" 得到 1:67 有损压缩；
* 但**信息论容量并不紧**：30.4 MiB ≈ 2.55e8 bit，1M 个 15-bit key + 15-bit value
  只需 ~3.0e7 bit —— 容量比约 8.5，理论上够存。
* 所以"1M needle 必输"不能只靠压缩比论证；真正的瓶颈是**固定维线性读出**
  （query 无法按内容随机寻址到某个槽位），不是状态比特数。

本模块把这两件事分开算：

1. :func:`capacity_ratio` —— 纯信息论容量比（松界，回答"够不够存"）；
2. :func:`linear_readout_snr` —— 固定维线性读出下，目标 key 相对其余 L-1 个
   key 干扰的信噪比（回答"能不能寻址"）。

假设（必须随结论一起引用）
--------------------------
* key 独立同分布、近似各向同性；value 的每个 token 需 ``log2(V)`` bit；
* 状态是 S 个浮点，每个浮点按有效尾数 ``bits_per_float`` 计信息；
* 读出是**线性/低秩**的（这是 SGC/Mamba/RWKV/GLA 这类 O(1) 状态模型的共同结构），
  不是可随机寻址的哈希表。

这不是严格定理，而是一个**可证伪的预测框架**：若实测 needle 成功率显著高于
:func:`linear_readout_snr` 给出的量级，说明模型学到了非线性的内容寻址，
框架需要修；反之则支持"容量不是瓶颈、读出结构才是"。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

__all__ = ["RetrievalLedger", "capacity_ratio", "linear_readout_snr", "ledger"]


@dataclass(frozen=True)
class RetrievalLedger:
    state_floats: int
    bits_per_float: int
    seq_len: int
    vocab_size: int
    state_bits: float
    required_bits: float
    capacity_ratio: float
    readout_snr: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "state_floats": self.state_floats,
            "bits_per_float": self.bits_per_float,
            "seq_len": self.seq_len,
            "vocab_size": self.vocab_size,
            "state_bits": self.state_bits,
            "required_bits": self.required_bits,
            "capacity_ratio": self.capacity_ratio,
            "readout_snr": self.readout_snr,
            "assumptions": [
                "i.i.d. isotropic keys",
                "value needs log2(V) bits per token",
                "linear/low-rank readout (no random addressing)",
            ],
        }


def capacity_ratio(state_floats: int, seq_len: int, vocab_size: int,
                   bits_per_float: int = 24) -> float:
    """信息论容量比：状态能存下 L 个 (key,value) 对的多少倍。

    ``required = L · 2·log2(V)``（key 与 value 各一份）。
    ratio ≥ 1 说明"容量上够存"，不说明"能寻址"。
    """
    if seq_len <= 0 or vocab_size <= 1:
        raise ValueError("seq_len > 0, vocab_size > 1")
    state_bits = float(state_floats) * float(bits_per_float)
    required = float(seq_len) * 2.0 * math.log2(float(vocab_size))
    return state_bits / max(required, 1.0)


def linear_readout_snr(state_floats: int, seq_len: int) -> float:
    """固定维线性读出下，目标 key 相对其余 key 干扰的信噪比。

    对 S 维各向同性 key，单个 key 的投影能量 ~1，其余 L-1 个 key 的干扰
    方差 ~ (L-1)/S；因此 ``SNR ≈ sqrt(S / (L-1))``。SNR ≫ 1 表示线性读出
    在平均意义上可分辨；SNR ≲ 1 表示检索被干扰淹没。
    """
    if seq_len <= 1 or state_floats <= 0:
        raise ValueError("seq_len > 1, state_floats > 0")
    return math.sqrt(float(state_floats) / float(seq_len - 1))


def ledger(state_floats: int, seq_len: int, vocab_size: int,
           bits_per_float: int = 24) -> RetrievalLedger:
    return RetrievalLedger(
        state_floats=int(state_floats), bits_per_float=int(bits_per_float),
        seq_len=int(seq_len), vocab_size=int(vocab_size),
        state_bits=float(state_floats) * float(bits_per_float),
        required_bits=float(seq_len) * 2.0 * math.log2(float(vocab_size)),
        capacity_ratio=capacity_ratio(state_floats, seq_len, vocab_size,
                                      bits_per_float),
        readout_snr=linear_readout_snr(state_floats, seq_len))