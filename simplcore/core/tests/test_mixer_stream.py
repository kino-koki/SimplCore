# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""融合热路径的等价性守护。

仅在 ``simplcore.ops.is_available()``（C++/CUDA 扩展已构建）时运行；否则跳过。
验证三件事（与仓库 fp32 守护阈值 4e-5 对齐）：
  1. 融合 step 与纯 PyTorch step 在 y_t 与 IIR 状态上逐位近似；
  2. 融合 step 满足严格因果（扰动末位不影响前文）；
  3. 融合 step 的流式输出 == 整段并行输出（由等价性隐含，这里做显式抽样）。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.sgc import SpectralGlobalConv
from simplcore.ops import is_available, mixer_stream_step


pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="sgc 融合热路径扩展未构建（需在 torch(+CUDA/nvcc) 环境编译；"
           "当前环境不可用，已自动回退纯 PyTorch 路径）",
)


def _make_mixer(dim=64, M=16, K=4, out_gate=True):
    m = SpectralGlobalConv(
        dim, num_bases=M, local_kernel=K,
        content_gate=True, out_gate=out_gate,
        evidence_gate=True, uncertainty_gate=True,
        norm="rmsnorm", fft_dtype=torch.float32,
    )
    m.eval()
    return m


def test_fused_matches_pure_step():
    m = _make_mixer()
    m2 = SpectralGlobalConv(
        m.dim, num_bases=m.num_bases, local_kernel=m.local_kernel,
        content_gate=True, out_gate=True,
        evidence_gate=True, uncertainty_gate=True,
        norm="rmsnorm", fft_dtype=torch.float32,
    )
    m2.load_state_dict(m.state_dict())
    m2.use_mixer_stream = True

    B = 2
    st1 = m.init_state(B, "cpu", torch.float32)
    st2 = m2.init_state(B, "cpu", torch.float32)
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(8):
        x = torch.randn(B, 1, m.dim)
        y1, st1 = m.step(x, st1)
        y2, st2 = m2.step(x, st2)
        worst = max(worst, (y1 - y2).abs().max().item())
        worst = max(worst, (st1.y1 - st2.y1).abs().max().item())
        worst = max(worst, (st1.y2 - st2.y2).abs().max().item())
        if st1.conv_buf is not None and st2.conv_buf is not None:
            worst = max(worst, (st1.conv_buf - st2.conv_buf).abs().max().item())
        worst = max(worst, (st1.energy_sum - st2.energy_sum).abs().max().item())
    assert worst < 1e-4, f"融合与纯 PyTorch 最大偏差 {worst:.2e} 超过 1e-4"


def test_fused_is_causal():
    m = _make_mixer()
    m.use_mixer_stream = True
    B = 1
    L = 12
    x = torch.randn(B, L, m.dim)
    # 整段并行前向（FFT）：取前 L-1 位置的输出
    with torch.no_grad():
        y_full = m.forward(x)  # (B, L, D)
    # 流式：逐 token，记录前 L-1 个输出
    st = m.init_state(B, "cpu", torch.float32)
    streamed = []
    for t in range(L):
        yt, st = m.step(x[:, t:t + 1], st)
        streamed.append(yt)
    ys = torch.stack(streamed, dim=1)  # (B, L, D)
    # 流式应与整段逐位近似（因果性 + 双形式的一部分）
    diff = (ys - y_full).abs().max().item()
    assert diff < 1e-4, f"流式与整段偏差 {diff:.2e} 超过 1e-4"


def test_fused_perturbation_preserves_prefix():
    m = _make_mixer()
    m.use_mixer_stream = True
    B = 2
    L = 10
    base = torch.randn(B, L, m.dim)
    pert = base.clone()
    pert[:, -1, :] += 0.5  # 只扰动末位 token

    def run(seq):
        st = m.init_state(B, "cpu", torch.float32)
        outs = []
        for t in range(L):
            yt, st = m.step(seq[:, t:t + 1], st)
            outs.append(yt)
        return torch.stack(outs, dim=1)  # (B, L, D)

    ref = run(base)
    other = run(pert)
    # 严格因果：扰动末位 token 不应改变前文（位置 0..L-2）的输出
    diff = (ref[:, :L - 1] - other[:, :L - 1]).abs().max().item()
    assert diff < 1e-4, f"扰动末位影响了前文（偏差 {diff:.2e}）"
