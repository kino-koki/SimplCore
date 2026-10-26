# -*- coding: utf-8 -*-
"""核心子层与 Block 测试：ResonanceField / DMC / TAR / SelfAttention / SimplCoreBlock。"""
import pytest
import torch

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.resonance import ResonanceField
from simplcore_pg.memory import DynamicMemoryChannels, MemoryState
from simplcore_pg.routing import TopologyAwareRouter
from simplcore_pg.attention import SelfAttention
from simplcore_pg.localmix import GatedLinearRecurrence, LocalInteraction
from simplcore_pg.block import SimplCoreBlock
from simplcore_pg.exceptions import ModelError


def _cfg(**kw):
    base = dict(hidden_size=32, num_layers=2, num_waves=4, num_channels=4,
                channel_size=8, num_paths=4, min_depth=1, max_depth=2,
                dmc_num_heads=4, tar_topk=2, ffn_intermediate_ratio=2.0,
                max_sequence_length=64, num_modalities=3, modality_dims=(32, 32, 32),
                dropout=0.0, rfp_num_steps=2, use_rollout=False)
    base.update(kw)
    return SimplCoreConfig(**base)


# ---- ResonanceField ----
def test_resonance_forward_shape():
    m = ResonanceField(32, num_waves=4, num_steps=2)
    x = torch.randn(2, 10, 32)
    y = m(x)
    assert y.shape == x.shape


def test_resonance_grad_flows():
    m = ResonanceField(16, num_waves=2, num_steps=1)
    x = torch.randn(1, 6, 16, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_resonance_1d_sequence():
    m = ResonanceField(16, num_waves=2, num_steps=1)
    y = m(torch.randn(1, 1, 16))
    assert torch.isfinite(y).all()


def test_resonance_bad_input():
    m = ResonanceField(16, num_waves=2, num_steps=1)
    with pytest.raises(AssertionError):
        m(torch.randn(2, 10))  # 2D -> torch._assert 触发 AssertionError


# ---- DMC ----
def test_dmc_read_shape():
    m = DynamicMemoryChannels(32, num_channels=4, channel_size=8, num_heads=4)
    x = torch.randn(2, 10, 32)
    y = m(x)
    assert y.shape == x.shape


def test_dmc_grad_flows():
    m = DynamicMemoryChannels(16, num_channels=2, channel_size=4, num_heads=2)
    x = torch.randn(1, 5, 16, requires_grad=True)
    m(x).sum().backward()
    assert x.grad is not None


def test_dmc_runtime_state_write_reset():
    m = DynamicMemoryChannels(16, num_channels=2, channel_size=4, num_heads=2)
    st = MemoryState(2, 4, 4, device=torch.device("cpu"))
    st.write(torch.randn(3, 4))
    assert st.write_ptr == 3
    st.reset()
    assert st.write_ptr == 0 and torch.allclose(st.bank, torch.zeros_like(st.bank))


# ---- TAR ----
def test_tar_forward_shape():
    m = TopologyAwareRouter(32, num_paths=4, topk=2)
    x = torch.randn(2, 8, 32)
    y = m(x, training=False)
    assert y.shape == x.shape


def test_tar_topk_equals_paths():
    m = TopologyAwareRouter(16, num_paths=3, topk=3)
    y = m(torch.randn(1, 4, 16))
    assert y.shape == (1, 4, 16)


def test_tar_training_dropout_flag():
    m = TopologyAwareRouter(16, num_paths=3, topk=2, dropout=0.5)
    x = torch.randn(1, 4, 16)
    y1 = m(x, training=True)
    y2 = m(x, training=False)
    # 训练与非训练路径数值可能不同（dropout 仅训练期）
    assert y1.shape == y2.shape == (1, 4, 16)


# ---- SelfAttention ----
def test_attention_shape():
    m = SelfAttention(32, num_heads=4, max_seq_len=64)
    x = torch.randn(2, 12, 32)
    y = m(x)
    assert y.shape == x.shape


def test_attention_mask_effect():
    m = SelfAttention(16, num_heads=2, max_seq_len=64)
    x = torch.randn(1, 4, 16)
    mask = torch.tensor([[True, True, False, False]])
    y = m(x, attention_mask=mask)
    assert y.shape == x.shape


def test_attention_maxlen_guard():
    m = SelfAttention(16, num_heads=2, max_seq_len=8)
    with pytest.raises(AssertionError):
        m(torch.randn(1, 16, 16))  # 触发 torch._assert


# ---- Block ----
def test_block_forward_shape_and_grad():
    block = SimplCoreBlock(_cfg())
    x = torch.randn(2, 10, 32, requires_grad=True)
    y = block(x, training=True)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_block_without_attention():
    block = SimplCoreBlock(_cfg(use_attention=False))
    y = block(torch.randn(2, 8, 32))
    assert y.shape == (2, 8, 32)


# ---- GatedLinearRecurrence：并行实现 vs 顺序参考 ----
def _sequential_recurrence(gate, x):
    """参考实现：严格按 h_t = a_t h_{t-1} + (1-a_t) x_t 顺序展开。"""
    a = torch.sigmoid(gate(x))
    B, L, H = x.shape
    out = torch.zeros_like(x)
    h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    for t in range(L):
        h = a[:, t] * h + (1.0 - a[:, t]) * x[:, t]
        out[:, t] = h
    return out


def test_recurrence_parallel_matches_sequential():
    torch.manual_seed(0)
    for L, H in [(1, 8), (5, 16), (40, 32), (64, 64)]:
        m = GatedLinearRecurrence(H)
        x = torch.randn(2, L, H)
        y_par = m(x)
        y_seq = _sequential_recurrence(m.gate, x)
        assert y_par.shape == y_seq.shape == (2, L, H)
        assert torch.allclose(y_par, y_seq, atol=1e-5, rtol=1e-4), f"L={L}, H={H}"


def test_recurrence_grad_flows_and_finite():
    torch.manual_seed(1)
    m = GatedLinearRecurrence(16)
    x = torch.randn(1, 20, 16, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_recurrence_cross_chunk_matches_sequential():
    """跨块边界：L > chunk_size 时分块状态传播必须与顺序展开逐位一致。"""
    torch.manual_seed(2)
    for L, C, H in [(129, 128, 16), (200, 64, 32), (300, 128, 8), (500, 128, 16)]:
        m = GatedLinearRecurrence(H, chunk_size=C, max_seq_len=512)
        x = torch.randn(2, L, H)
        y_par = m(x)
        y_seq = _sequential_recurrence(m.gate, x)
        assert y_par.shape == y_seq.shape == (2, L, H)
        err = float((y_par - y_seq).abs().max())
        assert err < 1e-4, f"L={L}, C={C}, H={H}: max_err={err:.2e}"


def test_recurrence_long_sequence_stable():
    """长序列数值稳定：无 NaN/Inf，且与顺序展开误差随长度保持有界。"""
    torch.manual_seed(3)
    m = GatedLinearRecurrence(16, chunk_size=128, max_seq_len=2048)
    x = torch.randn(1, 1024, 16)
    y_par = m(x)
    assert torch.isfinite(y_par).all()
    y_seq = _sequential_recurrence(m.gate, x)
    err = float((y_par - y_seq).abs().max())
    assert err < 1e-4, f"L=1024 max_err={err:.2e}"


def test_recurrence_padding_does_not_leak():
    """填充位置不影响有效输出：L 非 chunk 整数倍时结果与顺序展开一致。"""
    torch.manual_seed(4)
    m = GatedLinearRecurrence(8, chunk_size=64, max_seq_len=128)
    x = torch.randn(1, 100, 8)  # 100 = 64 + 36，非整数块
    y_par = m(x)
    y_seq = _sequential_recurrence(m.gate, x)
    assert torch.allclose(y_par, y_seq, atol=1e-5, rtol=1e-4)


def test_recurrence_rejects_over_max_len():
    """L > max_seq_len 必须显式拒绝，禁止静默裁剪（裁剪会破坏因果性）。"""
    m = GatedLinearRecurrence(8, chunk_size=64, max_seq_len=128)
    x = torch.randn(1, 200, 8)
    with pytest.raises(AssertionError):
        m(x)


def test_recurrence_slow_decay_numerically_stable():
    """极端 gate（a≈0.95 慢衰减）下并行扫描与顺序展开仍逐位一致、无上溢。"""
    torch.manual_seed(5)
    m = GatedLinearRecurrence(16, chunk_size=128, max_seq_len=2048)
    with torch.no_grad():
        m.gate.weight.fill_(3.0)
        m.gate.bias.fill_(3.0)  # sigmoid(3)≈0.95，慢衰减，状态长期累积
    x = torch.randn(1, 1024, 16)
    y_par = m(x)
    assert torch.isfinite(y_par).all()
    y_seq = _sequential_recurrence(m.gate, x)
    err = float((y_par - y_seq).abs().max())
    assert err < 1e-4, f"慢衰减 max_err={err:.2e}"


def test_recurrence_non_power_of_two_chunk():
    """非 2 的幂 chunk_size 下并行扫描仍与顺序展开一致（掩码轮数正确）。"""
    torch.manual_seed(6)
    for C in (100, 40, 7):
        m = GatedLinearRecurrence(8, chunk_size=C, max_seq_len=256)
        x = torch.randn(1, 250, 8)
        y_par = m(x)
        y_seq = _sequential_recurrence(m.gate, x)
        err = float((y_par - y_seq).abs().max())
        assert err < 1e-4, f"C={C} max_err={err:.2e}"


def test_localinteraction_conv_only_and_recur_only():
    """local_use_conv / local_use_recurrence 开关：单路径局部层仍可前向/反向。"""
    torch.manual_seed(7)
    for use_conv, use_recur in [(True, False), (False, True)]:
        m = LocalInteraction(16, kernel_size=3, use_conv=use_conv, use_recurrence=use_recur)
        x = torch.randn(2, 10, 16, requires_grad=True)
        y = m(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):
        LocalInteraction(16, use_conv=False, use_recurrence=False)


def test_localinteraction_fx_traceable():
    """LocalInteraction 必须可被 torch.fx 符号追踪（无数据相关控制流）。"""
    import torch.fx
    m = LocalInteraction(16, kernel_size=3, use_recurrence=True)
    x = torch.randn(1, 8, 16)
    gm = torch.fx.symbolic_trace(m)
    with torch.no_grad():
        y = gm(x)
    assert y.shape == (1, 8, 16)
    assert torch.isfinite(y).all()
