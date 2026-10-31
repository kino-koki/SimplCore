# -*- coding: utf-8 -*-
"""双形式 fp32 误差随序列长度的实测（短序列阈值不能外推）。

对应 `docs/NUMERICAL_STABILITY.md` 与 `docs/SGC_CORE.md §3.2`：
`<4e-5` 只在 `L ≲ 512` 成立；`L=2048` 已超一个数量级，必须显式承认。

L=8192 的完整测量在 `scripts/measure_long_sequence_error.py`（约 20 s，
不放进常规 pytest）；这里用 L≤2048 守住"误差随 L 增长、短阈值不外推"。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.sgc import SpectralGlobalConv


@pytest.fixture(scope="module")
def errors() -> dict:
    torch.manual_seed(0)
    m = SpectralGlobalConv(dim=64, num_bases=64, local_kernel=4,
                           content_gate=True, evidence_gate=True,
                           uncertainty_gate=True).eval()
    out = {}
    for L in (64, 128, 256, 512, 1024, 2048):
        x = torch.randn(1, L, 64)
        with torch.no_grad():
            full = m(x)
            st = m.init_state(1, x.device, x.dtype)
            outs = []
            for t in range(L):
                y, st = m.step(x[:, t:t + 1], st)
                outs.append(y)
            streamed = torch.cat(outs, dim=1)
        out[L] = float((full - streamed).abs().max())
    return out


def test_short_sequence_stays_below_4e5(errors) -> None:
    assert errors[64] < 4e-5
    assert errors[128] < 4e-5


def test_long_sequence_exceeds_the_short_threshold(errors) -> None:
    """4e-5 的"短序列打包阈值"不能无限外推。

    v1.2 口径注（logfreq 默认网格）：误差特征较 legacy 网格改善——
    L=1024 已降至 4e-5 以内（2.9e-5），L=2048 仍超阈（4.7e-5）。
    阈值外推守护相应更新：以 L=2048 为超阈锚点，L=1024 锁定"接近但
    不超"的新量级，防止文档数字与实现静默漂移。
    """
    assert errors[1024] < 4e-5          # logfreq 下 L=1024 回到阈值内
    assert errors[1024] > 1.5e-5        # 但仍随 L 显著增长，不是平坦
    assert errors[2048] > 4e-5          # 超阈锚点保留
    assert errors[2048] < 1e-3          # 仍是有界的，不是数值爆炸


def test_error_grows_with_length_but_does_not_explode(errors) -> None:
    assert errors[1024] >= errors[512]
    assert errors[2048] >= errors[1024]
    assert errors[2048] < 100 * errors[128]   # 增长受控（约 2 个数量级）


def test_documented_numbers_match(errors) -> None:
    """文档数字（SGC_CORE.md §3.2，v1.2 logfreq 默认网格口径）；
    允许 3× 抖动。legacy 网格的历史数字见文档勘误注——该网格已因
    公共零点被构造期拒绝，误差特征不可复现（这正是口径必须绑定的
    原因）。"""
    assert 5e-7 < errors[64] < 6e-6       # 实测 1.907e-6
    assert 1e-5 < errors[1024] < 9e-5     # 实测 2.943e-5
    assert 1.5e-5 < errors[2048] < 1.5e-4 # 实测 4.673e-5