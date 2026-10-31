# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC v1.2 增强（path_gate / inst_nonlin / init_mode）的测试。

核心不变式：任一增强开启都不得破坏
1) FFT（并行）/ IIR（流式）双形式一致；
2) 严格因果性；
3) 架构纯洁性（无注意力）；
4) analytic_params 与实例参数量逐位一致。
"""
from __future__ import annotations

import math

import pytest
import torch

from simplcore.core.config import SimplCoreConfig, analytic_params
from simplcore.core.model import SimplCoreModel
from simplcore.core.purity import assert_attention_free, assert_pure_config
from simplcore.core.sgc import SpectralGlobalConv, _init_oscillator_grid


def _cfg(**kw) -> SimplCoreConfig:
    base = dict(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                decay_max=128, local_kernel=3, num_heads=2, ffn_ratio=2.0,
                mixer="sgc")
    base.update(kw)
    return SimplCoreConfig(**base)


def _model(**kw) -> SimplCoreModel:
    torch.manual_seed(0)
    cfg = _cfg(**kw)
    assert_pure_config(cfg)
    return SimplCoreModel(cfg).eval()


# ---------------------------------------------------------------------------
# 双形式一致：并行前向 == 逐 token 流式
# ---------------------------------------------------------------------------
def _dual_form_check(m: SimplCoreModel):
    torch.manual_seed(42)
    idx = torch.randint(0, 64, (1, 24))
    with torch.no_grad():
        logits_par = m(idx)
        state = m.init_stream(1)
        logits_stream = None
        for t in range(idx.shape[1]):
            logits_stream, state = m.stream_step(idx[:, t], state)
    assert torch.allclose(logits_par[:, -1, :], logits_stream, atol=1e-4, rtol=1e-4)


def test_legacy_defaults_dual_form():
    _dual_form_check(_model())


def test_path_gate_dual_form():
    _dual_form_check(_model(sgc_path_gate=True))


def test_inst_nonlin_dual_form():
    _dual_form_check(_model(sgc_inst_nonlin=True))


def test_logfreq_init_dual_form():
    _dual_form_check(_model(sgc_init_mode="logfreq"))


def test_all_enhancements_dual_form():
    _dual_form_check(_model(sgc_path_gate=True, sgc_inst_nonlin=True,
                            sgc_init_mode="logfreq", out_gate=True))


# ---------------------------------------------------------------------------
# 因果性：扰动末位 token，前缀 logits 逐位不变
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(sgc_path_gate=True),
    dict(sgc_inst_nonlin=True),
    dict(sgc_init_mode="logfreq"),
])
def test_causality(kw):
    m = _model(**kw)
    torch.manual_seed(3)
    idx = torch.randint(0, 64, (1, 16))
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 7) % 64
    with torch.no_grad():
        l1 = m(idx)
        l2 = m(idx2)
    # 与 test_mixer_stream 的因果性口径一致（FFT 频域乘法会改变整段的
    # 浮点舍入路径，逐位相等不可达；容差 1e-4 对齐仓库 fp32 守护阈值）
    diff = (l1[:, :-1, :] - l2[:, :-1, :]).abs().max().item()
    assert diff < 1e-4, f"扰动末位影响了前文（偏差 {diff:.2e}）"
    last_diff = (l1[:, -1, :] - l2[:, -1, :]).abs().max().item()
    assert last_diff > 1e-4, "扰动末位应改变末位输出"


# ---------------------------------------------------------------------------
# path_gate 的语义：路由确实在"按内容选路"
# ---------------------------------------------------------------------------
def test_path_gate_weights_vary_with_content():
    m = _model(sgc_path_gate=True)
    with torch.no_grad():
        m.blocks[0].mixer.path_gate_proj.weight.add_(0.5)   # 打破均匀初始化
    x1 = torch.randn(1, 8, 32)
    x2 = torch.randn(1, 8, 32)
    with torch.no_grad():
        w1 = torch.softmax(m.blocks[0].mixer.path_gate_proj(m.blocks[0].mixer.norm(x1)), -1)
        w2 = torch.softmax(m.blocks[0].mixer.path_gate_proj(m.blocks[0].mixer.norm(x2)), -1)
    assert not torch.allclose(w1, w2)      # 不同输入 → 不同路由
    assert torch.allclose(w1.sum(-1), torch.ones(1, 8), atol=1e-5)


def test_path_gate_gradients_reach_router():
    m = _model(sgc_path_gate=True)
    m.train()
    idx = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))
    _, loss = m(idx, targets)
    loss.backward()
    g = m.blocks[0].mixer.path_gate_proj.weight.grad
    assert g is not None and torch.isfinite(g).all()


# ---------------------------------------------------------------------------
# init_mode：网格性质与行为差异
# ---------------------------------------------------------------------------
def test_logfreq_grid_covers_low_and_high():
    rho, theta = _init_oscillator_grid(32, decay_max=4096.0, mode="logfreq")
    assert (theta > 0).all() and (theta < torch.pi / 2).all()
    assert theta.min() < torch.pi / 2048          # 最慢基的周期覆盖 decay_max 量级
    assert theta.max() > 1.0                       # 最快基是高频（局部细节）
    # rho 网格两 mode 共享同一公式（v1.2 起 legacy 构造被公共零点守护
    # 拒绝，对照用公式直算而非 legacy 构造路径）
    lam = torch.exp(torch.linspace(0.0, math.log(4096.0), 32))
    assert torch.allclose(rho, torch.exp(-1.0 / lam))


def test_logfreq_changes_initial_kernel():
    """init_mode 真实生效：logfreq 网格与 legacy 公式网格不同。v1.2 起
    legacy 因公共零点被构造期拒绝，对照锚点降为**公式级网格对象**——
    对照的是被弃用网格的数学，不是把它装进模型。"""
    torch.manual_seed(0)
    kw = dict(dim=32, num_bases=16, decay_max=256.0, local_kernel=2)
    m_log = SpectralGlobalConv(**kw, init_mode="logfreq")
    theta_leg = math.pi * (torch.arange(16, dtype=torch.float32) + 0.5) / 16
    theta_log = torch.sigmoid(m_log.raw_theta) * math.pi
    assert not torch.allclose(theta_log, theta_leg)
    # 同一 mode 下确定性：重复构造（固定种子）结果一致
    torch.manual_seed(0)
    m_log2 = SpectralGlobalConv(**kw, init_mode="logfreq")
    assert torch.equal(m_log.build_kernel(64), m_log2.build_kernel(64))


def test_bad_init_mode_rejected():
    with pytest.raises(ValueError):
        _init_oscillator_grid(8, 128.0, mode="linear")
    with pytest.raises(ValueError):
        m = _model(sgc_init_mode="bogus")
        m.validate if False else None
    cfg = _cfg(sgc_init_mode="bogus")
    with pytest.raises(ValueError):
        cfg.validate()


# ---------------------------------------------------------------------------
# 纯洁性与参数账目
# ---------------------------------------------------------------------------
def test_enhanced_model_is_attention_free():
    m = _model(sgc_path_gate=True, sgc_inst_nonlin=True,
               sgc_init_mode="logfreq", out_gate=True)
    assert_attention_free(m)


def test_analytic_params_matches_instantiation():
    """path_gate 的解析参数公式（+3D+3）与实例化参数量逐位一致。"""
    for kw in (dict(), dict(sgc_path_gate=True)):
        cfg = _cfg(**kw)
        n_analytic = analytic_params(cfg)
        n_real = SimplCoreModel(cfg).num_params()
        assert n_analytic == n_real, f"kw={kw}: analytic={n_analytic} real={n_real}"


def test_stream_bytes_unchanged_by_enhancements():
    """增强不新增流式状态（path_gate 权重不进递推态）。"""
    m0 = _model()
    m1 = _model(sgc_path_gate=True, sgc_inst_nonlin=True)
    assert m0.stream_bytes(1) == m1.stream_bytes(1)
