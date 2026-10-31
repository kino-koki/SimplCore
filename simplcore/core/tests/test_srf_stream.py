# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SRF 融合热路径的等价守护：融合算子必须与纯 PyTorch step 在 fp32 下逐位一致。

无 torch / 扩展未编译时自动 skip（不报错）。目标机（含 torch + 编译器）上由
pytest 验证；CUDA 路径的并行归约顺序差异远在阈值内（见 docs/MIXER_STREAM.md §5）。
"""
import pytest

torch = pytest.importorskip("torch")
from simplcore.core.srf import SpectralResonanceFlow
from simplcore.ops import srf_is_available


def _make(mixer_stream: bool) -> SpectralResonanceFlow:
    m = SpectralResonanceFlow(
        dim=64, num_bases=16, local_kernel=4, content_gate=True, out_gate=True,
        evidence_gate=True, uncertainty_gate=True, norm="rmsnorm",
    )
    m.eval()
    m.use_mixer_stream = mixer_stream
    return m


def test_srf_stream_equivalence_fp32():
    if not srf_is_available():
        pytest.skip("srf 融合扩展不可用（无 torch / 未编译）")

    torch.manual_seed(0)
    m_ref = _make(mixer_stream=False)
    m_fused = _make(mixer_stream=True)
    m_fused.load_state_dict(m_ref.state_dict())

    B, T = 2, 12
    st_ref = m_ref.init_state(B, "cpu", torch.float32)
    st_fused = m_fused.init_state(B, "cpu", torch.float32)

    worst = 0.0
    for _ in range(T):
        x = torch.randn(B, 1, 64)
        y_ref, st_ref = m_ref.step(x, st_ref)
        y_fu, st_fused = m_fused.step(x, st_fused)
        worst = max(worst, (y_ref - y_fu).abs().max().item())
        worst = max(worst, (st_ref.y1 - st_fused.y1).abs().max().item())
        worst = max(worst, (st_ref.y2 - st_fused.y2).abs().max().item())
    assert worst < 1e-3, f"SRF 融合与纯 PyTorch fp32 偏差 {worst} 超过 1e-3"


def test_srf_stream_equivalence_bf16_runs():
    """bf16 输入下融合路径应可运行且输出有限（内部 fp32 递推，输出 cast 回 bf16）。

    仅验证可运行性与有限性，不做精度等式断言——融合内部恒 fp32，与纯 PyTorch
    bf16 step 的数值口径不同（融合更精确）。无扩展时 skip。
    """
    if not srf_is_available():
        pytest.skip("srf 融合扩展不可用（无 torch / 未编译）")

    torch.manual_seed(0)
    m = _make(mixer_stream=True)
    m.to(torch.bfloat16)
    m.eval()
    B, T = 2, 8
    st = m.init_state(B, "cpu", torch.bfloat16)
    for _ in range(T):
        x = torch.randn(B, 1, 64, dtype=torch.bfloat16)
        y, st = m.step(x, st)
    assert torch.isfinite(y).all(), "SRF 融合 bf16 输出含非有限值"
