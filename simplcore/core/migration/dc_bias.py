# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""TS2SC 直流偏置修复（Direct-Current Bias Transfer）。

问题
----
Qwen / Llama 系用 **RMSNorm**：只按均方根缩放，**不减均值**。
因此残差流的均值向量会原样传到下一层 FFN，改变其激活分布。

把注意力子层换成 SimplCore mixer 后，注入量
``Δ_s = mixer(x) − x`` 的**逐位置均值**与 teacher 的
``attn_out`` 均值并不相同。实测（Qwen3-0.6B，5 段真实中文，
teacher-forced）：

    逐层均值向量余弦全层均值 0.7456
    前 3 层 0.9745 / 0.9218 / 0.9053
    深层 |b_t − b_s| 可达 20 ~ 52（残差 std ≈ 48）

28 层累积后就是一个系统性分布偏移，表现为生成「前言不搭后语」
与高频词重复循环。这不是 pairwise 表达力问题，也不是
Attention / 非 Attention 的赛道问题——是**架构其他地方的静态
参数没有从 teacher 直搬过来**。

修复
----
每层一个静态向量：

    b_i = mean_pos( attn_out_i ) − mean_pos( Δ_s,i )
    y_i = mixer_i(x) + β · b_i        (β 默认 1.0)

性质（全部保持）
  * 零梯度、零蒸馏、零训练步
  * 不引入 q·k 打分 / N×N 矩阵 / KV cache / 任何 Attention 组件
  * mixer 的核仍输入无关、静态、可 IIR 递推
  * O(1) 流式状态**逐字节不变**：偏置加在残差流上，不进递推状态
  * FFT / IIR 对偶不变（偏置是逐位置常数，两条路径同值）
  * 成本：28 × D 个 fp32（D=1024 → 114 KB），一次加法/层

实测（Qwen3-0.6B，argmax 贪心，NNEW=16）
  「中国的首都是」  旧: '城市，有，北京，北京，北京，北京，北京，'  rep3=0.6429
                    修复: '的，是直辖市，人口，是新兴。深圳是新兴。深圳是' rep3=0.2143
  「今天天气很好，」旧: '适合户外，微动，天气晴，绿，花儿在树子'
                    修复: '晴，心情的植物园圃，绿圃，绿园，公园里'

边界（必须随代码一起读）
  * 修复的是**分布漂移**，不是内容寻址。静态平均注意力 profile 的
    表达力上限实测仅 cos 0.3695（首层 0.8846，中后层 0.25），
    模长比 2~7；把 v_proj / o_proj / input_layernorm 全部逐位直搬
    也补不上 pairwise 缺口。详见 docs/TS2SC_coherence_audit.md。
  * b_i 从有限语料估计，是**语料相关的固定估计**；换域可重估。
  * 不能让 base model 学会「回答」：teacher 自己对
    「华为是什么公司」的输出也是续写式追问。那是 SFT / instruct
    基座的行为缺口，与架构无关。
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def estimate_dc_bias(attn_out: torch.Tensor,
                     student_delta: torch.Tensor) -> torch.Tensor:
    """从一对 teacher-forced 张量估计单层直流偏置。

    Args:
        attn_out: teacher 注意力子层输出，形状 ``(B, L, D)``
                  （**不含**残差 x）。
        student_delta: student mixer 注入量 ``mixer(x) − x``，
                  形状 ``(B, L, D)``。
    Returns:
        ``(D,)`` 的静态偏置向量。

    注意：这里对 batch 与位置同时取均值。RMSNorm 不减均值，所以
    均值向量才是决定 FFN 输入分布的那一项；不要用方差或 cos 代替。
    """
    if attn_out.shape != student_delta.shape:
        raise ValueError(
            f"shape mismatch: {tuple(attn_out.shape)} vs "
            f"{tuple(student_delta.shape)}")
    if attn_out.dim() != 3:
        raise ValueError(f"expected (B,L,D), got {tuple(attn_out.shape)}")
    return (attn_out.float().mean(dim=(0, 1))
            - student_delta.float().mean(dim=(0, 1)))


def estimate_layerwise_bias(
        attn_outputs: Sequence[torch.Tensor],
        student_deltas: Sequence[torch.Tensor]) -> torch.Tensor:
    """逐层估计，返回 ``(N_layers, D)``。"""
    if len(attn_outputs) != len(student_deltas):
        raise ValueError("layer count mismatch")
    return torch.stack([
        estimate_dc_bias(a, d)
        for a, d in zip(attn_outputs, student_deltas)
    ])


def apply_dc_bias(y: torch.Tensor, bias: torch.Tensor,
                  beta: float = 1.0) -> torch.Tensor:
    """把第 i 层偏置加到 mixer 输出上（FFN 之前）。

    ``bias`` 形状 ``(D,)`` 或 ``(N_layers, D)`` 配 ``layer`` 参数。
    逐位置常数 → 并行与流式两条路径逐位一致。
    """
    if bias.dim() == 2:
        raise ValueError("pass a single layer's (D,) vector here")
    if bias.dim() != 1 or bias.shape[0] != y.shape[-1]:
        raise ValueError(f"bias must be (D,), got {tuple(bias.shape)}")
    return y + float(beta) * bias.to(y.dtype).view(1, 1, -1)


class LayerwiseDCBias(nn.Module):
    """可随权重包分发的偏置容器（纯 buffer，无梯度、无参数）。

    ``forward(y, layer)`` 直接给出修正后的 mixer 输出。
    用 ``register_buffer`` 是为了让 ``state_dict()`` 自然带上它，
    但**不计入可训练参数**：``sum(p.numel() for p in parameters())``
    对它是 0。
    """

    def __init__(self, bias: torch.Tensor, beta: float = 1.0) -> None:
        super().__init__()
        if bias.dim() != 2:
            raise ValueError(f"expected (N_layers, D), got {tuple(bias.shape)}")
        self.register_buffer("bias", bias.float().clone())
        self.beta = float(beta)

    @property
    def num_layers(self) -> int:
        return int(self.bias.shape[0])

    @property
    def dim(self) -> int:
        return int(self.bias.shape[1])

    def forward(self, y: torch.Tensor, layer: int) -> torch.Tensor:
        return apply_dc_bias(y, self.bias[layer], self.beta)
