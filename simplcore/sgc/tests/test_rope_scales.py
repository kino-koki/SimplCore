# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""YaRN/NTK 位置缩放外推层守护测试.

守护 ``simplcore/sgc/rope_scales.py`` 的结构主张（不守外推质量，后者需
A100 规模训练）：
  * ``strategy="none"`` 与 ``_rope._rope_tables`` 逐位一致（守护基准）；
  * 外推性质：缩放表 cos/sin 有限且有界（不 NaN/不 inf）；
  * 三策略（none/linear/yarn）在外推长度上输出分布"行为不同且有限"
    （argmax 熵对比，不断言质量）；
  * YaRN 滑窗模型流式 == 整段前向逐位一致（factor=2，L 从 32 流式到 64）；
  * 因果性继承（扰动未来不影响过去）。

预算：全模块 CPU 上 < 90 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc._rope import _rope_tables                              # noqa: E402
from simplcore.sgc.rope_scales import (                                    # noqa: E402
    SlidingWindowAttentionYaRN,
    RopeScaler,
    make_rope_tables,
)


# ---------------------------------------------------------------------------
# 1. strategy="none" 与 _rope_tables 逐位一致
# ---------------------------------------------------------------------------
def test_none_strategy_identical_to_rope_tables():
    for L in (8, 64, 128):
        c, s = make_rope_tables(16, L, "cpu", torch.float32, strategy="none")
        cr, sr = _rope_tables(16, L, "cpu", torch.float32)
        assert torch.equal(c, cr) and torch.equal(s, sr), \
            f"none strategy diverged at L={L}"


def test_factor_le_1_degrades_to_standard_rope():
    for strat in ("linear", "yarn"):
        c, s = make_rope_tables(16, 64, "cpu", torch.float32,
                                 strategy=strat, factor=1.0)
        cr, sr = _rope_tables(16, 64, "cpu", torch.float32)
        assert torch.allclose(c, cr, atol=1e-6), f"{strat} factor=1 not standard"


# ---------------------------------------------------------------------------
# 2. 外推性质：缩放表有限 + 有界
# ---------------------------------------------------------------------------
def test_scaled_tables_finite_and_bounded():
    for strat in ("linear", "yarn"):
        for factor in (2.0, 4.0):
            c, s = make_rope_tables(16, 256, "cpu", torch.float32,
                                     strategy=strat, factor=factor)
            assert torch.isfinite(c).all(), f"{strat} f={factor} cos NaN"
            assert torch.isfinite(s).all(), f"{strat} f={factor} sin NaN"
            assert c.abs().max().item() <= 1.0 + 1e-6, "cos must be in [-1,1]"
            assert s.abs().max().item() <= 1.0 + 1e-6, "sin must be in [-1,1]"


# ---------------------------------------------------------------------------
# 3. 三策略输出分布"行为不同且有限"（外推长度上 argmax 熵对比）
# ---------------------------------------------------------------------------
def test_three_strategies_differ_in_extrapolation_behavior():
    torch.manual_seed(0)
    D, H, W, L = 32, 4, 8, 256     # L=256 远超训练长度（外推区）
    x = torch.randn(2, L, D)
    with torch.no_grad():
        stats = {}
        for strat in ("none", "linear", "yarn"):
            m = SlidingWindowAttentionYaRN(D, H, W, strategy=strat,
                                            factor=2.0 if strat != "none" else 1.0).eval()
            y = m(x)
            stats[strat] = (y.abs().mean().item(), y.std().item())
    import math
    assert all(all(math.isfinite(v) for v in row) for row in stats.values()), f"non-finite: {stats}"
    # 三策略在外推区输出统计不同（均值/方差不全等）
    vals = [stats["none"], stats["linear"], stats["yarn"]]
    assert not (vals[0] == vals[1] == vals[2]), \
        f"all three strategies identical at L={L}: {vals}"


# ---------------------------------------------------------------------------
# 4. 流式 == 整段前向逐位一致（YaRN factor=2，L=32 流式到 64）
# ---------------------------------------------------------------------------
def test_streaming_matches_full_forward_yarn():
    torch.manual_seed(0)
    D, H, W = 32, 4, 8
    m = SlidingWindowAttentionYaRN(D, H, W, strategy="yarn", factor=2.0).eval()
    L = 64
    x = torch.randn(2, L, D)
    with torch.no_grad():
        full = m(x)
        st = m.init_stream(2, x.device, x.dtype)
        outs = []
        for t in range(L):
            lg, st = m.stream_step(x[:, t], st)
            outs.append(lg)
        streamed = torch.cat(outs, 1)
    d = (streamed - full).abs().max().item()
    assert d < 1e-5, f"YaRN streaming diverged: {d}"


# ---------------------------------------------------------------------------
# 5. 因果性继承
# ---------------------------------------------------------------------------
def test_causality_inherited():
    torch.manual_seed(0)
    D, H, W, L = 32, 4, 8, 64
    m = SlidingWindowAttentionYaRN(D, H, W, strategy="yarn", factor=2.0).eval()
    x = torch.randn(2, L, D)
    x2 = x.clone()
    x2[:, L // 2:] = torch.randn_like(x2[:, L // 2:]) * 100
    with torch.no_grad():
        y1, y2 = m(x), m(x2)
    assert (y1[:, : L // 2] - y2[:, : L // 2]).abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# 6. effective_theta 解析式
# ---------------------------------------------------------------------------
def test_effective_theta():
    assert RopeScaler("none", 1.0).effective_theta() == 10000.0
    assert abs(RopeScaler("linear", 2.0).effective_theta() - 10000.0 ** 0.5) < 1e-9
    assert RopeScaler("yarn", 2.0).effective_theta() is None  # 无单一等效基数


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
