# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""AMD Radeon 680M 长上下文可行性评估。

在给定共享显存 (UMA) 上限下，基于 SimplCore 流式核心的「固定状态」显存模型，
估算可承载的最大上下文长度；并给出 Windows / WSL2+ROCm / Linux 三条部署路径的
诚实结论 (Windows 原生 PyTorch 无法使用 AMD GPU 训练)。

输出 JSON 到 longctx_amd_feasibility.json。
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def estimate_stream_memory_gb(H, L_total, nb, C, S, win, chunk, dtype_bytes, batch=1):
    """估算 SimplCore 流式核心在给定上下文长度下的单步常驻显存 (GB)。

    激活峰值与 L_total 无关 (被 chunk 封顶)；固定状态也与 L_total 无关。
    本架构激活/状态都不随 L 增长，因此 L 可任意长 (受数值精度与地址空间)，
    真正限制来自：权重常驻 + 固定状态 + chunk 激活。
    """
    Lf = chunk // 2 + 1
    rfp_state = batch * Lf * H * nb
    dmc_state = C * S * H * nb
    heads = max(1, H // 64)
    hd = H // heads
    kv_state = batch * heads * win * hd * 2 * nb
    fixed = rfp_state + dmc_state + kv_state
    act = batch * chunk * H * 4 * nb * 2
    total_bytes = (fixed + act) * dtype_bytes
    return total_bytes / (1024 ** 3)


def main():
    H = 4096
    nb = 14
    C, S = 64, 128
    win = 4096
    chunk = 2048
    dtype_bytes = {"fp16": 2, "fp32": 4, "bf16": 2}

    uma_options = [2, 4, 8]

    report = {
        "device": "AMD Radeon 680M (RDNA2, 12 CU, shared system memory / UMA)",
        "architecture": "SimplCore streaming core (fixed-state long context)",
        "note": "本架构激活显存与序列长度 L 解耦，故 L 不再消耗额外显存；"
                "限制来自权重常驻 + 固定状态 + chunk 激活。",
        "per_length_runtime_gb": {},
        "uma_capacity_gb": {},
        "deployment_paths": {
            "windows_native": {
                "pytorch_amd_support": False,
                "verdict": "不可用：Windows 原生 PyTorch 仅支持 NVIDIA CUDA，"
                           "AMD 680M 无法被 torch 识别用于训练。",
                "workaround": "可用 torch-directml 做极轻量推理，但不支持本架构的 "
                             "FSDP/torch.compile/长上下文训练。",
            },
            "wsl2_roc": {
                "pytorch_amd_support": True,
                "verdict": "推荐路径：WSL2 (Ubuntu 22.04) + ROCm 5.7/6.x，"
                           "680M 属 RDNA2 (gfx1033/gfx1034) 受支持。",
                "workaround": "需设置 HSA_OVERRIDE_GFX_VERSION=10.3.0 并安装 ROCm 版 PyTorch。",
            },
            "linux_native": {
                "pytorch_amd_support": True,
                "verdict": "最佳路径：纯 Linux 双系统 + ROCm，免 WSL2 开销。",
                "workaround": "同 WSL2 ROCm 安装。",
            },
        },
        "length_scalability": "由于激活/状态不随 L 增长，序列长度从 1M 翻倍到 2M/4M "
                              "时常驻显存增量≈0；可支持的上下文长度上限由数值精度与"
                              "地址空间决定，而非显存 OOM。",
    }

    for dt, db in dtype_bytes.items():
        g = estimate_stream_memory_gb(H, 1_000_000, nb, C, S, win, chunk, db)
        report["per_length_runtime_gb"][dt] = round(g, 4)

    weights_bf16_gb = 56 * 1e9 * 2 / (1024 ** 3)
    report["weights_56b_bf16_gb"] = round(weights_bf16_gb, 1)
    for uma in uma_options:
        fits_56b = bool(uma * 1024 ** 3 > weights_bf16_gb * 1024 ** 3)
        report["uma_capacity_gb"][f"{uma}GB"] = {
            "can_load_56b_model": fits_56b,
            "verdict": "可承载 56B 推理/训练" if fits_56b
                       else f"无法加载 56B 模型 (权重 {report['weights_56b_bf16_gb']}GB "
                            f"远超 {uma}GB 共享显存)；仅适合运行小模型 (≤数 B) 的长上下文实验",
        }

    # CPU 实测小点验证公式 (小模型，证明 per_length 不随 L 变化)
    from simplcore_pg.config import SimplCoreConfig
    from simplcore_pg.model_stream import SimplCoreStreamingModel
    small = SimplCoreConfig(
        vocab_size=200, hidden_size=64, num_layers=2, num_experts=2, experts_per_tok=2,
        moe_intermediate_size=128, use_rfp=True, use_dmc=True, use_localmix=True, chunk_size=64,
    )
    m = SimplCoreStreamingModel(small)
    meas = {}
    for L in [512, 1024, 2048]:
        info = m.peak_activation_bytes(1, L)
        meas[L] = round(info["est_peak_bytes"] / 1024, 2)
    report["cpu_small_scale_verify_KB"] = meas
    report["cpu_verify_constant"] = len(set(meas.values())) == 1

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "longctx_amd_feasibility.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
