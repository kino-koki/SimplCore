# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""流式递推的低精度数值策略守护测试。

守护不变式（``simplcore.core.sgc.recurrent_dtype``）：

* 计算 dtype 为 bf16/fp16 时，所有 mixer 的递推状态（SGC / 门控 SGC 的 y1/y2，
  SRF 的 SSM ``h_re/h_im`` 别名）必须为 fp32 —— 否则 bf16 下 IIR/SSM 递推会逐
  token 累积舍入误差，破坏 O(1) 流式状态"数值可信"的前提（MI250 主用 bf16）。
* fp32 计算时行为与原实现逐位一致（状态即 fp32，零行为变化）。
* 流式输出必须有限（无 NaN / Inf）。

fp32 等价性由 ``test_mixer_stream.py`` / ``test_srf.py`` 等守护；本文件只守
"低精度下递推状态精度"这一新增不变式。无 torch 时整文件被 importorskip 跳过。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel


def _build(mixer: str, compute_dtype):
    torch.manual_seed(0)
    kw = dict(vocab_size=64, dim=32, num_layers=2, num_bases=4, ffn_ratio=2.0,
              local_kernel=3, num_heads=2, srf_heads=2, srf_chunk=4)
    if mixer == "sgc":
        kw["mixer"] = "sgc"
    elif mixer == "srf":
        kw["mixer"] = "srf"
    elif mixer == "gated":
        kw["mixer"] = "sgc"
        kw["gate_ab"] = True
    else:
        raise ValueError(mixer)
    cfg = SimplCoreConfig(**kw)
    cfg.validate()
    m = SimplCoreModel(cfg).to(compute_dtype)
    m.eval()
    return m


@pytest.mark.parametrize("mixer", ["sgc", "srf", "gated"])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float32])
def test_recurrent_state_dtype_and_finite(mixer, compute_dtype):
    """低精度计算时递推状态退化为 fp32；fp32 计算时保持 fp32；输出有限。"""
    m = _build(mixer, compute_dtype)
    B = 2
    x = torch.randint(0, 64, (B, 10))
    with torch.no_grad():
        st = m.init_stream(B, device=torch.device("cpu"), dtype=compute_dtype)
        last = None
        for t in range(x.shape[1]):
            last, st = m.stream_step(x[:, t], st)

    # 递推状态（每层第一个 mixer 的 y1）dtype 必须等于 recurrent_dtype(compute_dtype)
    expected = torch.float32 if compute_dtype in (torch.bfloat16, torch.float16) \
        else compute_dtype
    assert st.layers[0].y1.dtype == expected, (
        f"{mixer}/{compute_dtype}: 递推状态 dtype={st.layers[0].y1.dtype}，期望 {expected}"
    )
    assert torch.isfinite(last).all(), f"{mixer}/{compute_dtype}: 流式输出含 NaN/Inf"


@pytest.mark.parametrize("mixer", ["sgc", "srf", "gated"])
def test_low_precision_state_is_fp32(mixer):
    """针对 MI250（bf16）的关键断言：低精度计算时递推状态必为 fp32。

    这样即使融合扩展未构建（回退纯 PyTorch），MI250 上的 O(1) 流式递推仍然是
    数值可信的 fp32，而非逐 token 累积舍入误差的 bf16。
    """
    m = _build(mixer, torch.bfloat16)
    B = 1
    x = torch.randint(0, 64, (B, 6))
    with torch.no_grad():
        st = m.init_stream(B, device=torch.device("cpu"), dtype=torch.bfloat16)
        for t in range(x.shape[1]):
            _, st = m.stream_step(x[:, t], st)
    assert st.layers[0].y1.dtype == torch.float32
