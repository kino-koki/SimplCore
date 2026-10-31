# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""mathproof 的守护测试：每项数学证明必须保持通过。"""
from __future__ import annotations

import pytest

from simplcore.core.mathproof import (
    _proof_bibo_stability,
    _proof_chunk_carry_law,
    _proof_closed_form,
    _proof_companion_matrix,
    _proof_complexity_audit,
    _proof_evidence_identity,
    _proof_fft_no_aliasing,
    _proof_frozen_base_gradient,
    _proof_gate_bounds,
    _proof_ledger_algebra,
    _proof_shift_covariance,
    _proof_state_sufficiency,
    _proof_stream_equals_parallel,
    run_all,
)


# --------------------------------------------------------------------------
# 逐项证明（失败时 residual 里带具体残差）
# --------------------------------------------------------------------------
def test_p01_closed_form():
    r = _proof_closed_form()
    assert r.passed, r.residual


def test_p02_companion_matrix():
    r = _proof_companion_matrix()
    assert r.passed, r.residual


def test_p03_chunk_carry_law():
    r = _proof_chunk_carry_law()
    assert r.passed, r.residual


def test_p04_fft_no_aliasing():
    r = _proof_fft_no_aliasing()
    assert r.passed, r.residual


def test_p05_bibo_stability():
    r = _proof_bibo_stability()
    assert r.passed, r.residual


def test_p06_stream_equals_parallel():
    r = _proof_stream_equals_parallel()
    assert r.passed, r.residual


def test_p07_evidence_identity():
    r = _proof_evidence_identity()
    assert r.passed, r.residual


def test_p08_gate_bounds():
    r = _proof_gate_bounds()
    assert r.passed, r.residual


def test_p09_shift_covariance():
    r = _proof_shift_covariance()
    assert r.passed, r.residual


def test_p10_state_sufficiency():
    r = _proof_state_sufficiency()
    assert r.passed, r.residual


def test_p11_complexity_audit():
    r = _proof_complexity_audit()
    assert r.passed, r.residual


def test_p12_ledger_algebra():
    r = _proof_ledger_algebra()
    assert r.passed, r.residual


def test_p13_frozen_base_gradient():
    r = _proof_frozen_base_gradient()
    assert r.passed, r.residual


# --------------------------------------------------------------------------
# 汇总一致性
# --------------------------------------------------------------------------
def test_run_all_covers_all_proofs():
    results = run_all()
    assert len(results) == 13
    assert all(r.passed for r in results), [
        f"{r.pid}: {r.residual}" for r in results if not r.passed
    ]


def test_v31_infer_formula_contains_num_bases():
    """v3.1 修正的防回归：infer mixer 项必须含基数 M（旧版 6·D 已废止）。"""
    from simplcore.core.config import SimplCoreConfig
    from simplcore.core.scaling import infer_flops_per_token, non_embedding_params

    cfg = SimplCoreConfig(vocab_size=8192, dim=256, num_layers=4, num_bases=16,
                          local_kernel=4, num_heads=4, ffn_ratio=2.0)
    mixer = infer_flops_per_token(cfg, 4096) - 2.0 * non_embedding_params(cfg)
    assert mixer == pytest.approx(cfg.num_layers * cfg.dim * (5.0 * cfg.num_bases + 22.0))
    # 旧版错误值（6·D·layers）不得再出现
    assert mixer != pytest.approx(6.0 * cfg.dim * cfg.num_layers)
