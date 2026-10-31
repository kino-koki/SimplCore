# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""自研 CUDA/C++ 后端算子契约的守护测试。

守护的主张：
1. 算子清单覆盖 SimplCore 全部内核需求（mixer 三件套 / 闸门 / 优化器 / 状态 / 双 GCD）；
2. 算子名唯一、精度策略齐全（bf16 计算 + fp32 累加）；
3. 禁止项显式点名注意力家族 / KV cache / 位置表（purity 同款红线）；
4. 一致性检查全为硬闸门，且被 campaign.build_plan 的 Phase 0 自动继承；
5. FFT↔IIR 双形式、分块携带、回滚逐位恢复三条逐位一致性必须在契约里。
"""
import pytest

from simplcore.core.backend_contract import (
    DEFAULT_CONTRACT,
    SIMPLCORE_BACKEND_OPS,
    BackendContract,
    BackendOp,
    conformance_checks,
)
from simplcore.core.campaign import build_plan

REQUIRED_OPS = (
    "emb_lookup", "rmsnorm", "gemm", "swiglu", "residual_add",
    "causal_depthwise_conv", "spectral_causal_conv", "chunked_iir_scan",
    "content_gate", "evidence_uncertainty_gate", "ce_loss",
    "stream_state_init", "stream_state_serialize",
    "adamw_frozen_mask", "weight_decay_exempt", "shard_allreduce",
)


def test_ops_complete_and_unique():
    names = [o.name for o in SIMPLCORE_BACKEND_OPS]
    assert len(names) == len(set(names)), "算子名必须唯一"
    for required in REQUIRED_OPS:
        assert required in names, f"缺少必需内核: {required}"


def test_precision_policy_on_every_op():
    """计算内核 bf16+fp32 累加；纯状态内核合法地全 fp32（O(1) 状态本身 fp32）。"""
    for o in SIMPLCORE_BACKEND_OPS:
        assert o.dtype_accum == "fp32", o.name
        if o.kind.startswith("compute"):
            assert o.dtype_compute == "bf16", o.name
        else:  # state / optimizer / collective
            assert o.dtype_compute in ("bf16", "fp32"), o.name
    # 流式状态必须是 fp32 常驻
    for name in ("stream_state_init", "stream_state_serialize"):
        op = next(o for o in SIMPLCORE_BACKEND_OPS if o.name == name)
        assert op.dtype_compute == "fp32" and op.kind == "state"


def test_forbidden_bans_attention_family():
    joined = "\n".join(DEFAULT_CONTRACT.forbidden)
    for banned in ("softmax", "注意力", "KV cache", "位置嵌入表"):
        assert banned in joined, banned


def test_bitexact_trio_present():
    """三条逐位一致性：双形式 / 分块携带 / 回滚恢复。"""
    joined = "\n".join(DEFAULT_CONTRACT.bitexact_requirements)
    assert "4e-5" in joined          # FFT vs IIR
    assert "逐位一致" in joined
    assert "byte-exact" in joined    # 回滚


def test_conformance_checks_all_hard():
    checks = conformance_checks()
    assert len(checks) >= 7
    assert all(c.kind == "hard" for c in checks)


def test_campaign_phase0_inherits_contract_gates():
    """campaign.build_plan 的 Phase 0 闸门必须自动包含契约一致性检查。"""
    plan = build_plan()
    p0 = plan.phase("Phase 0 冻结")
    p0_names = {g.name for g in p0.gates}
    for c in conformance_checks():
        assert c.name in p0_names, f"Phase 0 缺少契约闸门 {c.name}"
    # PyTorch 退役必须写进 Phase 0 目标
    assert "PyTorch 退役" in p0.objective
    assert "自研" in p0.objective


def test_contract_format_and_json():
    txt = DEFAULT_CONTRACT.format()
    assert "后端算子契约" in txt and "禁止进入内核" in txt
    d = DEFAULT_CONTRACT.as_dict()
    assert d["name"] == "simplcore-cuda-backend"
    assert len(d["ops"]) == len(SIMPLCORE_BACKEND_OPS)
    # 一个空契约也应能格式化（不炸）
    empty = BackendContract(name="x", version="0", ops=[],
                            precision_policy=[], bitexact_requirements=[],
                            forbidden=[])
    assert "算子数: 0" in empty.format()
