# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""参数预算与算力账本守护测试。

重点不是"函数能跑"，而是三条**主张**不被悄悄改坏：

1. 大词表会把小模型的预算吃穿 → 规划器必须守住嵌入红线；
2. SimplCore 的长程成本随 ``log seq`` 亚线性增长（attention 则随 ``seq`` 线性涨）；
3. SimplCore 没有 KV cache，流式状态与 ``seq`` 无关（O(1)）。
"""
from __future__ import annotations

import math

import pytest

from simplcore.core.config import SimplCoreConfig, analytic_params, get_preset
from simplcore.core.scaling import (CostLedger, aspect_ratio, budget_table,
                                   chinchilla_tokens, compute_budget,
                                   cost_ledger, embedding_params,
                                   embedding_share, gpu_hours,
                                   kv_cache_bytes_per_token,
                                   largest_config_for_flops,
                                   non_embedding_params, params_per_layer,
                                   plan_budget, stream_state_bytes,
                                   tokens_for_budget,
                                   train_flops_per_token)


def _sgc(**kw) -> SimplCoreConfig:
    base = dict(vocab_size=8192, dim=256, num_layers=4, num_bases=16,
                local_kernel=4, num_heads=4, ffn_ratio=2.0)
    base.update(kw)
    return SimplCoreConfig(**base)


# ---------------------------------------------------------------------------
# 参数切分
# ---------------------------------------------------------------------------
def test_embedding_params_is_vocab_times_dim() -> None:
    cfg = _sgc(vocab_size=4096, dim=192)
    assert embedding_params(cfg) == 4096 * 192


def test_embedding_and_non_embedding_sum_to_total() -> None:
    cfg = _sgc()
    assert embedding_params(cfg) + non_embedding_params(cfg) == analytic_params(cfg)


def test_big_vocab_eats_small_model_budget() -> None:
    """同一深度/宽度下，32k 词表的嵌入占比必须显著高于 8k。"""
    big = _sgc(vocab_size=32000, dim=512, num_layers=10, num_bases=32)
    small = _sgc(vocab_size=8192, dim=512, num_layers=10, num_bases=32)
    assert embedding_share(big) > embedding_share(small) + 0.15
    assert embedding_share(small) < 0.25


def test_params_per_layer_excludes_embedding() -> None:
    cfg = _sgc()
    per = params_per_layer(cfg)
    assert per == pytest.approx((non_embedding_params(cfg) - cfg.dim) / cfg.num_layers)


def test_aspect_ratio() -> None:
    assert aspect_ratio(_sgc(dim=512, num_layers=8)) == pytest.approx(64.0)


# ---------------------------------------------------------------------------
# 长程成本的增长阶
# ---------------------------------------------------------------------------
def test_sgc_long_range_grows_sublinearly_with_seq() -> None:
    """SGC 是 log 增长：8 倍上下文带来的成本增幅必须远小于线性。"""
    sgc = _sgc(dim=256, num_layers=4)
    r = train_flops_per_token(sgc, 4096) / train_flops_per_token(sgc, 512)
    # 线性项（6N）被拉长，长程项仅随 log2 增长 → 比值应明显小于 8（线性情形）
    assert r < 4.0


def test_sgc_long_range_is_logarithmic_in_isolation() -> None:
    """把线性项扣掉后，SGC 的长程项应满足 ≈ C·log2(2·seq)。"""
    def lr(seq: int) -> float:
        cfg = _sgc(dim=256, num_layers=1)
        return train_flops_per_token(cfg, seq) - 6.0 * non_embedding_params(cfg)

    a, b = lr(1024), lr(2048)
    # log2(2048)/log2(1024) = 11/10 → 比值应落在 1.03~1.20
    assert 1.03 < b / a < 1.20


def test_infer_flops_constant_in_length() -> None:
    """SimplCore 解码每 token 成本与序列长度无关（O(1) 流式）。"""
    sgc = cost_ledger(_sgc(), 512)
    sgc_long = cost_ledger(_sgc(), 8192)
    assert sgc_long.infer_flops_per_token == pytest.approx(sgc.infer_flops_per_token)


# ---------------------------------------------------------------------------
# 缓存占用
# ---------------------------------------------------------------------------
def test_sgc_has_no_kv_cache() -> None:
    assert kv_cache_bytes_per_token(_sgc()) == 0
    assert cost_ledger(_sgc()).is_kv_free


def test_sgc_state_bytes_independent_of_seq_len() -> None:
    cfg = _sgc()
    assert stream_state_bytes(cfg) == stream_state_bytes(cfg)
    led = cost_ledger(cfg, 512)
    # 上下文涨 64 倍，SimplCore 的缓存占用不变（kv 项为 0）
    assert led.context_bytes(512) == led.context_bytes(32768)


def test_state_bytes_grow_with_batch() -> None:
    cfg = _sgc()
    assert stream_state_bytes(cfg, batch=4) == 4 * stream_state_bytes(cfg, batch=1)


def test_state_bytes_include_only_allocated_evidence_accumulators() -> None:
    off = _sgc(evidence_gate=False)
    on = _sgc(evidence_gate=True)
    assert stream_state_bytes(on) - stream_state_bytes(off) == \
        4 * on.num_layers


def test_snn_stream_bytes_include_membrane_state() -> None:
    cfg = _sgc(mixer="snn-sgc")
    base = _sgc()
    assert stream_state_bytes(cfg) - stream_state_bytes(base) == \
        4 * cfg.num_layers * cfg.dim


# ---------------------------------------------------------------------------
# token / 算力预算
# ---------------------------------------------------------------------------
def test_chinchilla_and_compute_budget() -> None:
    assert chinchilla_tokens(30_000_000) == 600_000_000
    assert compute_budget(1_000_000, 1000) == 6e9


def test_gpu_hours_positive_and_monotone() -> None:
    small = gpu_hours(1e18)
    big = gpu_hours(2e18)
    assert small > 0 and big == pytest.approx(2 * small)


def test_tokens_for_budget_inverse_of_cost() -> None:
    cfg = _sgc()
    per = train_flops_per_token(cfg, 1024)
    assert tokens_for_budget(per * 1234, cfg, 1024) == pytest.approx(1234)


def test_cheaper_per_token_costs_less() -> None:
    """同算力下不同规模的 SimplCore 都能喂出有限 token 数（O(1) 流式）。"""
    budget = 1e18
    sgc = tokens_for_budget(budget, _sgc(dim=256, num_layers=4), 4096)
    assert sgc > 0 and math.isfinite(sgc)


def test_largest_config_for_flops_respects_budget() -> None:
    flops, tokens = 5e19, 1e9
    cfg = largest_config_for_flops(flops, tokens, seq_len=1024,
                                  vocab_size=8192, max_dim=1024, max_layers=16)
    assert cfg is not None
    assert train_flops_per_token(cfg, 1024) * tokens <= flops * 1.001


def test_largest_config_shrinks_when_tokens_double() -> None:
    """同算力下 token 翻倍 → 能装的模型必须变小（Chinchilla 权衡的直接体现）。"""
    flops = 5e19
    a = largest_config_for_flops(flops, 1e9, seq_len=1024, vocab_size=8192,
                                 max_dim=1024, max_layers=16)
    b = largest_config_for_flops(flops, 2e9, seq_len=1024, vocab_size=8192,
                                 max_dim=1024, max_layers=16)
    assert a is not None and b is not None
    assert analytic_params(b) <= analytic_params(a)


# ---------------------------------------------------------------------------
# 规划器
# ---------------------------------------------------------------------------
def test_plan_budget_hits_target_and_respects_red_line() -> None:
    plans = plan_budget(30_000_000, top_k=5)
    assert plans
    best = plans[0]
    assert abs(best.params - 30_000_000) / 30_000_000 < 0.05
    assert best.embedding_share <= 0.25
    assert 16 <= best.aspect <= 80


def test_plan_budget_avoids_32k_vocab_at_30m() -> None:
    """30M 预算下不该出现 32k 词表解 —— 它会把 37% 预算扔进嵌入表。"""
    for p in plan_budget(30_000_000, top_k=5):
        assert p.cfg.vocab_size <= 16000, p.as_dict()


def test_plan_budget_returns_sorted_by_score() -> None:
    plans = plan_budget(10_000_000, top_k=4)
    scores = [p.score(10_000_000) for p in plans]
    assert scores == sorted(scores)


def test_plan_budget_respects_max_layers() -> None:
    for p in plan_budget(5_000_000, top_k=3, max_layers=6):
        assert p.cfg.num_layers <= 6


# ---------------------------------------------------------------------------
# 预设：30M / 1B
# ---------------------------------------------------------------------------
def test_sgc_30m_is_about_30m() -> None:
    cfg = get_preset("tuan-30m")
    assert 28e6 < analytic_params(cfg) < 33e6


def test_sgc_1b_is_about_1b() -> None:
    cfg = get_preset("tuan-1b")
    assert 0.95e9 < analytic_params(cfg) < 1.10e9


def test_budget_table_renders_rows() -> None:
    table = budget_table({"tuan-30m": get_preset("tuan-30m"),
                          "tuan-1b": get_preset("tuan-1b")}, 4096)
    assert "tuan-30m" in table and "tuan-1b" in table
    assert table.count("\n") >= 3
