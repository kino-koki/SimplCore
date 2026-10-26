# -*- coding: utf-8 -*-
"""
理论性能评估: 点阵谐振模型 (Lattice) vs 标准 Transformer
=======================================================

纯标准库实现 (无需 tensorflow/numpy), 用于在联网环境外也能产出可复核的
理论指标: 推理 FLOPs、显存占用、长文本扩展性。

结论可在文档 INNOVATION_SUMMARY / 本脚本输出中查看。运行:
    python simplcore/architecture/lattice_eval.py
"""

from __future__ import annotations

import math

# 80B 配置 (纯中文语料, 大字表)
VOCAB = 150_000
HIDDEN = 8192
LAYERS = 104
NUM_BANDS = 8
KERNEL = 31
NUM_TABLES = 64
SLOTS = 128
FFN_RATIO = 4
CONTEXTS = [2048, 8192, 32768, 131072, 1_048_576]   # 短 -> 1B 上下文
BYTES = 2  # BF16


def transformer_flops_per_token(n, d, layers):
    """标准 Transformer 单层 (含注意力的 2·n²·d + FFN 2·n·d²) × layers, 单 token 步进按 n 上下文."""
    attn = 2.0 * n * n * d
    ffn = 2.0 * n * d * d
    return layers * (attn + ffn)


def lattice_flops_per_token(n, d, layers, bands, K, tables, slots):
    """点阵模型单层: 谐振耦合 O(n·K·bands·d) + 哈希记忆 O(n·tables·slots·d) + FFN."""
    rc = n * K * bands * d
    hm = n * tables * slots * d
    ffn = 2.0 * n * d * d
    return layers * (rc + hm + ffn)


def params_80b():
    d = HIDDEN
    per_block = (
        NUM_BANDS + NUM_BANDS + NUM_BANDS * KERNEL + 1          # RC
        + d * (NUM_TABLES * SLOTS) + d * d + d * d             # HM
        + d * (d * FFN_RATIO) + (d * FFN_RATIO) * d            # FFN
    )
    total = (
        VOCAB * d
        + LAYERS * per_block
        + d
        + d * VOCAB
    )
    return total


def memory_gb(params, ctx, mode):
    """显存占用估算 (GB).

    Transformer: 参数 + KV-cache (每层 2·n·d·2 用于 K,V, BF16, 假设无重算)。
    Lattice:    参数 + 固定记忆 (tables·slots·d, 与 n 解耦) + 激活近似线性。
    """
    p_gb = params * BYTES / 1e9
    if mode == "transformer":
        kv = LAYERS * 2 * ctx * HIDDEN * BYTES / 1e9 * 2   # K,V 两矩阵
        return p_gb + kv
    else:  # lattice
        mem = NUM_TABLES * SLOTS * HIDDEN * BYTES / 1e9
        act = LAYERS * ctx * HIDDEN * BYTES / 1e9 * 4      # 线性激活近似
        return p_gb + mem + act


def main():
    p = params_80b()
    print("=" * 72)
    print("青小团 Lattice 模型 vs 标准 Transformer — 理论性能对比 (80B / 纯中文)")
    print("=" * 72)
    print(f"参数量: {p/1e9:.2f} B  隐藏维度: {HIDDEN}  层数: {LAYERS}")
    print(f"上下文测试点: {[c for c in CONTEXTS]}")
    print("-" * 72)
    print(f"{'上下文(n)':>12} | {'TF FLOPs/T':>14} | {'Lattice FLOPs/T':>16}")
    hdr = f"{'上下文(n)':>12} | {'TF FLOPs/T':>14} | {'Lattice/T':>16} | {'加速比':>8} | {'TF显存(GB)':>11} | {'Lat显存(GB)':>12}"
    print(hdr)
    print("-" * len(hdr))
    for n in CONTEXTS:
        tf = transformer_flops_per_token(n, HIDDEN, LAYERS)
        lat = lattice_flops_per_token(n, HIDDEN, LAYERS, NUM_BANDS, KERNEL, NUM_TABLES, SLOTS)
        ratio = tf / lat
        tf_mem = memory_gb(p, n, "transformer")
        lat_mem = memory_gb(p, n, "lattice")
        nn = n if n < 1e6 else f"{n/1e6:.0f}M"
        print(f"{nn:>12} | {tf/1e9:>12.1f}B | {lat/1e9:>14.1f}B | {ratio:>7.1f}x | {tf_mem:>11.1f} | {lat_mem:>12.1f}")
    print("-" * len(hdr))
    print("说明: FLOPs/T = 每生成一个 token 所需的浮点运算; 加速比 = TF / Lattice。")
    print("Lattice 显存中 '记忆' 项固定 (tables·slots·d), 不随上下文 n 增长;")
    print("Transformer 显存随 n 线性增长 (KV-cache), 在 1B 上下文时主导。")
    print("=" * 72)


if __name__ == "__main__":
    main()
