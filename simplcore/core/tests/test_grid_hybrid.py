# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""R2 混合网格 + R3 分层多尺度衰减的守护测试.

R2（复现报告实测）：logfreq 网格低频端绝对间隔趋零 → Ψ 秩缺失
（M=16 rank 14/16、M=128 rank 63/128，cond ≈1e19，与命题 5 的
|sinθ|→0 病态同源）。hybrid = 线性均匀 (m+0.5) 栅格 + 黄金比扰动
（保住最慢基 θ_lo=π/decay_max、打破 legacy 等差的公共零点结构）。

锁定五条：
1. hybrid 在 M=16/64/128/256 全部满秩、条件数有界（守护窗口自适应
   max(512, 4M)——M≥256 需 ≥2M 采样窗，固定 512 是窗口不够不是网格病态）；
2. 守护对 logfreq 网格**确实翻红**（负向变异：把 hybrid 的守护换到
   logfreq 上必须 raise——否则守护是装饰）；
3. hybrid 无公共零点（构造成功即已过 _assert_no_common_zero）；
4. R3 分层衰减：decay_scale≠1 时各层 ρ_max 几何递增（深层更长记忆）；
5. R3 零默认：decay_scale=1.0 与缺省逐位一致；validate 校验 scale>0。

诚实边界：hybrid 的**任务质量收益**（复现报告预期"免费的质量提升"）
未实测——本文件只锁网格数值质量与守护机制。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.arch.sgc import (_assert_numerical_rank,
                                     _grid_rank_condition,
                                     _init_oscillator_grid)
from simplcore.core.model.config import SimplCoreConfig
from simplcore.core.model.model import SimplCoreModel


@pytest.mark.parametrize("M", [16, 64, 128, 256])
def test_hybrid_full_rank_bounded_condition(M: int) -> None:
    _, theta = _init_oscillator_grid(M, 8192.0, mode="hybrid")
    rank, cond = _grid_rank_condition(theta, L=max(512, 4 * M))
    assert rank == M, f"hybrid M={M} 秩缺失: {rank}/{M}"
    assert cond < 1e6, f"hybrid M={M} 条件数超限: {cond:.3e}"
    # 最慢基保留（decay_max 配对设计不被网格替换牺牲）
    assert float(theta.min()) == pytest.approx(3.141592653589793 / 8192.0,
                                               rel=0.5)


def test_rank_guard_flags_logfreq_pathology() -> None:
    """负向变异：守护换到 logfreq 网格上必须翻红（否则守护是装饰）.
    用户实测口径（M=16、decay=8192、rank 14/16）在此精确复现并锁定."""
    _, theta = _init_oscillator_grid(16, 8192.0, mode="logfreq")
    rank, cond = _grid_rank_condition(theta, L=512)
    assert rank < 16, f"预期 logfreq M=16 秩缺失，实测 rank={rank}"
    with pytest.raises(ValueError, match="rank-deficient"):
        _assert_numerical_rank(theta, 16, L=512)


def test_hybrid_passes_common_zero_guard() -> None:
    """hybrid 无公共零点：_init_oscillator_grid 内部已调
    _assert_no_common_zero，构造成功即通过（黄金比扰动打破等差结构）."""
    for M in (16, 64, 128):
        _init_oscillator_grid(M, 8192.0, mode="hybrid")   # 不 raise 即过


# ---------------------------------------------------------------------------
# R3：分层多尺度衰减
# ---------------------------------------------------------------------------
def _tiny(**kw) -> SimplCoreModel:
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=3, num_bases=8,
                          ffn_ratio=2.0, decay_max=256.0, local_kernel=2, **kw)
    return SimplCoreModel(cfg).eval()


def test_layerwise_decay_scale_increases_memory() -> None:
    torch.manual_seed(0)
    m = _tiny(decay_scale=4.0, sgc_init_mode="hybrid")
    with torch.no_grad():
        rhos = [blk.mixer.oscillator_params()[0].max().item()
                for blk in m.blocks]
    assert rhos[0] < rhos[1] < rhos[2], \
        f"decay_scale=4 时深层 ρ_max 必须更大（更长记忆）: {rhos}"
    # hybrid 网格同时生效：构造即过数值秩守护（不 raise）


def test_layerwise_scale_one_bitwise_neutral() -> None:
    torch.manual_seed(0)
    m1 = _tiny(sgc_init_mode="hybrid")
    torch.manual_seed(0)
    m0 = _tiny(decay_scale=1.0, sgc_init_mode="hybrid")
    x = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        assert torch.equal(m0(x), m1(x)), \
            "decay_scale=1.0 必须与缺省逐位一致（opt-in 零默认破坏）"


def test_validate_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="decay_scale"):
        SimplCoreConfig(vocab_size=64, dim=32, num_layers=1, num_bases=8,
                        decay_scale=0.0).validate()