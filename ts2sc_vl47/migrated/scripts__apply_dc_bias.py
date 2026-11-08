#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC-VL-1B v2 推理接入示例：直流偏置修复。

只改一处：mixer 输出之后、post_attention_layernorm / FFN 之前，
把第 i 层的静态偏置向量加到残差流上。

    y_i = mixer_i(x) + beta * b_i
    h   = ffn_norm(y_i)
    x_{i+1} = y_i + FFN(h)

b_i 来自 dc_bias.safetensors 的 "model.dc_bias"，shape (28, 1024)，
float32，无梯度。O(1) 递推状态逐字节不变；FFT / IIR 对偶不变；
没有引入任何 q·k 打分、N×N 矩阵或 KV cache。

用法（伪代码，与具体推理框架无关）：

    from safetensors.torch import load_file
    bias = load_file("dc_bias.safetensors")["model.dc_bias"]  # (28,1024)
    for i, layer in enumerate(model.layers):
        y = layer.mixer(x)                       # (B,L,D)
        y = apply_dc_bias(y, bias[i], beta=cfg["dc_bias"]["beta"])
        x = y + layer.mlp(layer.post_attention_layernorm(y))

解析推导与实测见 SimplCore 仓库 docs/TS2SC_coherence_audit.md。
"""
from __future__ import annotations

from typing import Union

import torch


def apply_dc_bias(y: torch.Tensor,
                  bias: torch.Tensor,
                  beta: float = 1.0) -> torch.Tensor:
    """把单层直流偏置加到 mixer 输出。

    Args:
        y: mixer 输出，形状 ``(B, L, D)``（或 ``(L, D)`` / ``(D,)``）。
        bias: 该层偏置，形状 ``(D,)``；也可以传完整 ``(N_layers, D)``
              矩阵配 ``layer`` 参数。
        beta: 强度系数，默认 1.0。0.0 等价于关闭修复。
        layer: 当 bias 是二维时指定层号。
    """
    if bias.dim() == 2:
        raise ValueError("传单层 (D,) 向量；二维请用 bias[layer]")
    if bias.dim() != 1 or bias.shape[0] != y.shape[-1]:
        raise ValueError(
            f"bias 必须是 (D,)，D={y.shape[-1]}，实得 {tuple(bias.shape)}")
    return y + float(beta) * bias.to(y.dtype).view(
        *([1] * (y.dim() - 1)), -1)


def load_bias(path: str = "dc_bias.safetensors",
             key: str = "model.dc_bias") -> torch.Tensor:
    """读取偏置文件并做形状 / 有限性校验。"""
    from safetensors.torch import load_file
    b = load_file(path)[key].float()
    if b.dim() != 2 or b.shape[1] not in (1024,):
        raise ValueError(f"unexpected bias shape {tuple(b.shape)}")
    if not torch.isfinite(b).all():
        raise ValueError("bias contains NaN/Inf")
    return b


if __name__ == "__main__":
    # 自测：不依赖任何模型权重
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="dc_bias.safetensors")
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()
    b = load_bias(a.file)
    print(f"bias shape={tuple(b.shape)} dtype={b.dtype} "
          f"norm_per_layer_first5={[round(float(b[i].norm()), 4) for i in range(5)]}")
    y = torch.zeros(1, 3, b.shape[1])
    o = apply_dc_bias(y, b[0], a.beta)
    assert torch.allclose(o[0, 0], a.beta * b[0], atol=1e-5)
    assert torch.allclose(o[0, 0], o[0, 2], atol=1e-6), \
        "逐位置常数：并行与流式必须逐位一致"
    print(f"✅ self-test passed (beta={a.beta}, positionwise-constant)")
