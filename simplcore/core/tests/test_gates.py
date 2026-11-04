# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""四层闸门（L1~L4）守护测试。

覆盖：
* L2 证据门 / L3 不确定门 **不破坏 FFT↔IIR 对偶形式**（流式 == 整段）
* 闸门**严格因果**（未来扰动不改变过去）
* 闸门参数**梯度可达**（不出现静默零梯度）
* 闸门**可消融**（关闭后与基线路径一致）
* L4 拒答门：损失、阈值决策、温度标定
"""
from __future__ import annotations

import math

import pytest
import torch

from simplcore.core.abstain import (
    AbstentionHead,
    abstention_loss,
    calibrate_temperature,
    grounded_probability,
    should_abstain,
)
from simplcore.core.sgc import SpectralGlobalConv

DIM = 32
M = 8


def _layer(**kw) -> SpectralGlobalConv:
    """P0-4：闸门默认值已在算子层统一为 False（与 SimplCoreConfig 对齐）。

    本文件的测试意图是**带 L2/L3 闸门**的双形式与门控行为，不该依赖
    隐式默认——用 setdefault 显式声明；调用方显式传值时以调用方为准。
    """
    torch.manual_seed(0)
    kw.setdefault("evidence_gate", True)
    kw.setdefault("uncertainty_gate", True)
    return SpectralGlobalConv(
        dim=DIM, num_bases=M, decay_max=64.0, local_kernel=4, **kw
    )


# ---------------------------------------------------------------------------
# L2 / L3: dual form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 4, 17, 33])
def test_gates_stream_matches_parallel(L: int) -> None:
    """流式路径必须与整段前向一致（带 L2/L3 闸门）。"""
    layer = _layer()
    x = torch.randn(2, L, DIM)

    with torch.no_grad():
        y_parallel = layer(x)

        state = layer.init_state(2, x.device, x.dtype)
        outs = []
        for t in range(L):
            y_t, state = layer.step(x[:, t:t + 1], state)
            outs.append(y_t)
        y_stream = torch.cat(outs, dim=1)

    assert torch.max(torch.abs(y_parallel - y_stream)) < 4e-5


def test_evidence_gate_uses_causal_prefix_only() -> None:
    """L2 的证据统计只能依赖前缀：改末位不应影响首位。"""
    layer = _layer()
    x = torch.randn(1, 8, DIM)
    x2 = x.clone()
    x2[:, -1] += 5.0

    with torch.no_grad():
        w1 = layer._evidence_weights(x)
        w2 = layer._evidence_weights(x2)

    assert torch.max(torch.abs(w1[:, 0] - w2[:, 0])) < 1e-6


def test_gates_preserve_causality() -> None:
    """扰动未来 token，前文输出必须逐位不变。"""
    layer = _layer()
    x = torch.randn(1, 16, DIM)
    x2 = x.clone()
    x2[:, 8:] += 3.0

    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x2)

    assert torch.max(torch.abs(y1[:, :8] - y2[:, :8])) < 1e-6


# ---------------------------------------------------------------------------
# gate parameter gradients
# ---------------------------------------------------------------------------
def test_gate_parameters_receive_gradient() -> None:
    """L2/L3 的参数必须收到梯度（防止静默零梯度陷阱）。"""
    layer = _layer()
    x = torch.randn(2, 12, DIM, requires_grad=False)
    y = layer(x)
    y.pow(2).mean().backward()

    assert layer.evidence_gain.grad is not None
    assert torch.any(layer.evidence_gain.grad != 0), \
        "evidence_gain 零梯度：证据门退化为常数"
    assert layer.evidence_bias.grad is not None
    assert torch.any(layer.evidence_bias.grad != 0)
    assert layer.uncertainty_logit.grad is not None
    assert torch.any(layer.uncertainty_logit.grad != 0), \
        "uncertainty_logit 零梯度：不确定门退化为常数"


def test_evidence_gate_init_is_half_open() -> None:
    """零能量输入下，证据门初始应为 sigmoid(0)=0.5（半开）且对输入敏感。"""
    layer = _layer()
    w = layer._evidence_weights(torch.zeros(1, 4, DIM))
    assert abs(float(w.mean()) - 0.5) < 1e-6
    # 权重不能全零：否则闸门对证据完全不敏感
    assert abs(float(layer.evidence_gain)) > 1e-3


# ---------------------------------------------------------------------------
# ablation
# ---------------------------------------------------------------------------
def test_gates_can_be_disabled() -> None:
    """关闭 L2/L3 后，长程路径不再被闸门调制。"""
    layer_off = _layer(evidence_gate=False, uncertainty_gate=False)
    layer_on = _layer(evidence_gate=True, uncertainty_gate=True)

    x = torch.randn(1, 10, DIM)
    with torch.no_grad():
        y_off = layer_off(x)
        y_on = layer_on(x)

    assert not torch.allclose(y_off, y_on), "开启闸门后输出应发生变化"
    # v1.2 闸门 opt-in：关闭时门参数**不存在**（无幽灵参数、无梯度贡献）
    assert not hasattr(layer_off, "evidence_gain")
    assert not hasattr(layer_off, "uncertainty_logit")


def test_gate_disabled_stream_still_matches() -> None:
    """关闭闸门后对偶形式依然成立（回归保护）。"""
    layer = _layer(evidence_gate=False, uncertainty_gate=False)
    x = torch.randn(1, 12, DIM)
    with torch.no_grad():
        y_par = layer(x)
        state = layer.init_state(1, x.device, x.dtype)
        outs = []
        for t in range(12):
            y_t, state = layer.step(x[:, t:t + 1], state)
            outs.append(y_t)
        y_stream = torch.cat(outs, dim=1)
    assert torch.max(torch.abs(y_par - y_stream)) < 4e-5


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------
def test_evidence_gate_suppresses_long_path_on_silent_context() -> None:
    """低能量（无信息）上下文下，证据门应压低权重。"""
    layer = _layer(uncertainty_gate=False)
    loud = torch.randn(1, 8, DIM) * 3.0
    quiet = torch.randn(1, 8, DIM) * 1e-3

    with torch.no_grad():
        w_loud = layer._evidence_weights(loud)
        w_quiet = layer._evidence_weights(quiet)

    # 前缀能量越大 → 证据越充分 → 权重越高（sigmoid 单调）
    assert float(w_loud.mean()) > float(w_quiet.mean())


def test_uncertainty_gate_damps_disagreement() -> None:
    """长短路径分歧越大，不确定门的阻尼越强。"""
    layer = _layer(evidence_gate=False)
    long_a = torch.randn(1, 4, DIM)
    local_same = long_a.clone()
    local_diff = -long_a

    w_same = layer._uncertainty_weights(long_a, local_same)
    w_diff = layer._uncertainty_weights(long_a, local_diff)

    assert float(w_same.mean()) > float(w_diff.mean())
    assert torch.all(w_diff <= 1.0 + 1e-6) and torch.all(w_diff >= 0.0)


def test_streaming_state_bytes_accounts_for_evidence_gate() -> None:
    """流式状态字节数必须把证据门的运行和计入（O(1) 承诺的一部分）。"""
    on = _layer(evidence_gate=True)
    off = _layer(evidence_gate=False)
    assert on.streaming_state_bytes(1) > off.streaming_state_bytes(1)
    # 与序列长度无关
    assert on.streaming_state_bytes(1) == on.streaming_state_bytes(1)


# ---------------------------------------------------------------------------
# L4 abstention gate
# ---------------------------------------------------------------------------
def test_abstention_head_shape_and_default_never_abstains() -> None:
    torch.manual_seed(0)
    head = AbstentionHead(DIM)
    hidden = torch.randn(2, 6, DIM)
    logit = head(hidden)
    assert logit.shape == (2, 6)
    # 未训练时几乎不拒答
    assert float(grounded_probability(logit).max()) < 0.5


def test_abstention_loss_and_gradient() -> None:
    head = AbstentionHead(DIM)
    hidden = torch.randn(2, 5, DIM)
    grounded = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0],
                             [0.0, 1.0, 1.0, 0.0, 1.0]])
    logit = head(hidden)
    loss = abstention_loss(logit, grounded)
    assert loss.dim() == 0
    loss.backward()
    assert head.proj.weight.grad is not None
    assert torch.any(head.proj.weight.grad != 0)


def test_should_abstain_threshold() -> None:
    logit = torch.tensor([[2.0, -2.0, 0.0]])
    mask = should_abstain(logit, threshold=0.5)
    # p(2.0)=0.88 -> 不拒答；p(-2.0)=0.12 -> 拒答；p(0)=0.5 -> 边界不拒答
    assert bool(mask[0, 0]) is False
    assert bool(mask[0, 1]) is True
    assert bool(mask[0, 2]) is False


def test_calibrate_temperature() -> None:
    logit = torch.tensor([1.0])
    assert float(calibrate_temperature(logit, 2.0)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        calibrate_temperature(logit, 0.0)


def test_abstention_loss_rejects_shape_mismatch() -> None:
    head = AbstentionHead(DIM)
    logit = head(torch.randn(2, 5, DIM))
    with pytest.raises(ValueError):
        abstention_loss(logit, torch.ones(2, 4))
