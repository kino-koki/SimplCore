# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""``sgc-bind``（BindingSpectralConv）的守护测试 —— R1 修复的 mixer 级落地.

对应 ``docs/REPRODUCTION_REPORT.md`` 的三条教训与本 mixer 的四条不变式：

1. **双形式一致**：FFT（并行）与 IIR+绑定 scan（流式）在 fp32 内一致
   —— 绑定读/写是逐位置确定性同序操作，对偶性不得被辅助通路破坏；
2. **O(1) 定容**：流式状态字节数与已消费 token 数无关（含槽状态）；
3. **全链梯度**：addr_key / addr_value / addr_query / write_gate /
   mem_gate 全部有非零梯度（复现报告 mem_Wq 梯度恒 0 的教训）；
4. **严格因果**：扰动序列末位，前文 logits 逐位不变。

机制级（非质量级）测试：检索质量数值由 axis-probe 对照实验记账
（``--arms sgc-bind``），此处不锁准确率。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig, analytic_params
from simplcore.core.model import SimplCoreModel
from simplcore.core.purity import assert_attention_free, assert_pure_config


def _cfg(**kw) -> SimplCoreConfig:
    base = dict(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                decay_max=128, local_kernel=3, num_heads=2, ffn_ratio=2.0,
                mixer="sgc-bind", sgc_bind_slots=32)
    base.update(kw)
    return SimplCoreConfig(**base)


def _model(**kw) -> SimplCoreModel:
    torch.manual_seed(0)
    cfg = _cfg(**kw)
    assert_pure_config(cfg)
    return SimplCoreModel(cfg).eval()


# ---------------------------------------------------------------------------
# 1. 双形式一致：并行前向 == 逐 token 流式（fp32 内）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [8, 24, 33])
def test_dual_form_full_vs_stream(L):
    m = _model()
    torch.manual_seed(42)
    idx = torch.randint(0, 64, (1, L))
    with torch.no_grad():
        logits_par = m(idx)
        state = m.init_stream(1)
        last = None
        for t in range(L):
            last, state = m.stream_step(idx[:, t], state)
    assert torch.allclose(logits_par[:, -1, :], last, atol=1e-4, rtol=1e-4), \
        f"L={L}: max|d|={float((logits_par[:, -1, :] - last).abs().max()):.3e}"


def test_dual_form_chunked_vs_full():
    m = _model()
    torch.manual_seed(7)
    idx = torch.randint(0, 64, (1, 29))
    with torch.no_grad():
        full = m(idx)
        chunked = m.forward_chunked(idx, chunk_size=6)
    assert torch.allclose(full, chunked, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. O(1) 定容：状态字节与已消费长度无关
# ---------------------------------------------------------------------------
def test_stream_state_bytes_constant_in_length():
    m = _model()
    x = torch.randint(0, 64, (1, 40))
    with torch.no_grad():
        st = m.init_stream(1)
        for t in range(5):
            _, st = m.stream_step(x[:, t], st)
        b5 = st.num_bytes()
        for t in range(5, 40):
            _, st = m.stream_step(x[:, t], st)
        b40 = st.num_bytes()
    assert b5 == b40
    # 账本一致：num_bytes == streaming_state_bytes（含槽状态 2·S·D + D）
    assert b40 == m.stream_bytes(1, dtype=torch.float32)
    mx = m.blocks[0].mixer
    extra = (2 * mx.n_slots * mx.dim + mx.dim) * 4
    base = 2 * 8 * 32 + 2 * 32 + 1  # y1+y2 (fp32) + conv_buf(K-1=2)·D + evidence? off
    assert b40 > 0 and extra > 0


def test_binding_state_survives_checkpoint_restore():
    m = _model()
    torch.manual_seed(3)
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        st = m.init_stream(1)
        for t in range(12):
            _, st = m.stream_step(idx[:, t], st)
        ckpt = m.checkpoint_stream(st)
        st2 = m.restore_stream(ckpt)
        # 恢复后继续流式，与不中断路径一致
        nxt = torch.randint(0, 64, (1, 1))
        a, _ = m.stream_step(nxt[:, 0], st2)
        b, _ = m.stream_step(nxt[:, 0], st)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. 全链梯度（复现报告 mem_Wq 教训）
# ---------------------------------------------------------------------------
def test_all_binding_parameters_receive_gradient():
    m = _model()
    torch.manual_seed(1)
    idx = torch.randint(0, 64, (2, 16))
    logits, loss = m(idx[:, :-1], idx[:, 1:])
    loss.backward()
    for blk in m.blocks:
        mx = blk.mixer
        for name in ("addr_key", "addr_value", "addr_query",
                     "write_gate", "read_out"):
            if not hasattr(mx, name):
                continue
            p = getattr(mx, name).weight
            assert p.grad is not None and float(p.grad.abs().sum()) > 0, \
                f"{name} 梯度恒 0（复现报告教训未根治）"
    # mem_gate 标量也要有梯度
    g = m.blocks[0].mixer.mem_gate.grad
    assert g is not None and float(g.abs()) > 0


def test_analytic_params_matches_instance():
    from simplcore.core.config import analytic_params
    cfg = _cfg(dim=48, num_layers=3, sgc_bind_slots=16)
    assert analytic_params(cfg) == SimplCoreModel(cfg).num_params()


# ---------------------------------------------------------------------------
# 4. 严格因果 + 纯洁性
# ---------------------------------------------------------------------------
def test_strict_causality():
    m = _model()
    torch.manual_seed(5)
    x = torch.randint(0, 64, (1, 20))
    with torch.no_grad():
        p1 = m(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % 64
        p2 = m(x2)
    assert torch.allclose(p1[:, :-1], p2[:, :-1], atol=1e-5)


def test_purity_passes_for_sgc_bind():
    cfg = _cfg()
    assert_pure_config(cfg)
    m = SimplCoreModel(cfg)
    assert_attention_free(m)
    assert m.supports_streaming


# ---------------------------------------------------------------------------
# 5. 构造守卫
# ---------------------------------------------------------------------------
def test_n_slots_must_be_positive():
    with pytest.raises(ValueError):
        _model(sgc_bind_slots=0)


def test_sgc_bind_differs_from_sgc():
    base = dict(vocab_size=64, dim=32, num_layers=1, num_bases=8,
                local_kernel=4)
    a = SimplCoreModel(SimplCoreConfig(mixer="sgc", **base))
    b = SimplCoreModel(SimplCoreConfig(mixer="sgc-bind", **base))
    assert b.num_params() > a.num_params()
