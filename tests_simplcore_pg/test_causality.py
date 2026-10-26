# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""因果性回归测试（此前仓库完全缺失，正是它让 RFP 的非因果缺陷存活至今）。

判据: 只扰动序列的**未来**位置, 早期位置的输出必须逐位不变。
对自回归语言模型这是硬约束 —— 违反它会让交叉熵被"偷看答案"压低,
困惑度数字失真, 生成过程不合法。
"""
from __future__ import annotations

import pytest
import torch

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.resonance import ResonanceField, band_to_causal_kernel, causal_spectral_conv


def _causal_delta(module, x, split, forward_kwargs=None):
    """Return max |y[:, :split] - y2[:, :split]| after perturbing x[:, split:]."""
    forward_kwargs = forward_kwargs or {}
    with torch.no_grad():
        y1 = module(x, **forward_kwargs)
        x2 = x.clone()
        x2[:, split:, :] = torch.randn_like(x2[:, split:, :]) * 10.0
        y2 = module(x2, **forward_kwargs)
    return (y1[:, :split] - y2[:, :split]).abs().max().item()


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def test_causal_kernel_is_future_blind():
    torch.manual_seed(0)
    L, W = 32, 3
    band = torch.rand(L // 2 + 1, W) + 0.1
    kernel = band_to_causal_kernel(band, L)
    assert kernel.shape == (L, W)
    x = torch.randn(2, L, W)
    y = causal_spectral_conv(x, kernel)
    # perturb the future, check the past
    x2 = x.clone()
    x2[:, 20:, :] = torch.randn(2, L - 20, W) * 10
    y2 = causal_spectral_conv(x2, kernel)
    delta = (y[:, :20] - y2[:, :20]).abs().max().item()
    assert delta < 1e-5, f"causal_spectral_conv leaks the future: {delta}"


def test_causal_spectral_conv_matches_explicit():
    """Must equal the explicit causal sum, not merely be future-blind."""
    torch.manual_seed(1)
    L, W = 16, 2
    x = torch.randn(1, L, W)
    kernel = torch.randn(L, W)
    got = causal_spectral_conv(x, kernel)
    ref = torch.zeros_like(x)
    for t in range(L):
        for s in range(t + 1):
            ref[:, t, :] += x[:, s, :] * kernel[t - s].unsqueeze(0)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


# ---------------------------------------------------------------------------
# ResonanceField
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,split", [(64, 48), (128, 100), (256, 200)])
def test_resonance_field_is_causal(L, split):
    torch.manual_seed(0)
    layer = ResonanceField(hidden_size=32, num_waves=4, num_steps=2,
                           max_seq_len=L).eval()
    x = torch.randn(2, L, 32)
    delta = _causal_delta(layer, x, split, {"seq_len": L})
    assert delta < 1e-5, f"ResonanceField leaks the future: max|delta|={delta}"


def test_resonance_field_default_is_causal_flag():
    layer = ResonanceField(hidden_size=8, num_waves=2)
    assert layer.causal is True, "RFP must be causal by default"


def test_non_causal_mode_is_actually_non_causal():
    """Guard the对照 path: causal=False must be the *old* leaking behaviour.

    If this ever stops leaking, the historical comparisons are no longer
    reproducible and the docstring warning is stale.
    """
    torch.manual_seed(0)
    layer = ResonanceField(hidden_size=32, num_waves=4, num_steps=2,
                           max_seq_len=64, causal=False).eval()
    x = torch.randn(2, 64, 32)
    delta = _causal_delta(layer, x, 48, {"seq_len": 64})
    assert delta > 1e-4, "causal=False no longer reproduces the non-causal behaviour"


def test_resonance_field_gradients_flow():
    torch.manual_seed(0)
    layer = ResonanceField(hidden_size=16, num_waves=4, num_steps=2,
                           max_seq_len=32, use_checkpoint=False)
    x = torch.randn(2, 32, 16)
    layer(x, seq_len=32).pow(2).mean().backward()
    for name in ("in_proj.weight", "out_proj.weight", "omega", "beta", "out_scale"):
        p = dict(layer.named_parameters())[name]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} has no gradient"


def test_damping_is_not_advertised_as_trainable():
    """damping/dt are buffers: the docstring must not claim they backprop."""
    layer = ResonanceField(hidden_size=8, num_waves=2)
    assert "damping" not in dict(layer.named_parameters())
    doc = (ResonanceField.__doc__ or "") + __import__(
        "simplcore_pg.resonance", fromlist=["x"]).__doc__
    assert "梯度可达" not in doc or "不参与梯度" in doc


# ---------------------------------------------------------------------------
# model level
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_rfp", [True, False])
def test_simplcore_model_is_causal(use_rfp):
    torch.manual_seed(0)
    cfg = SimplCoreConfig.get_attention_free_config(hidden_size=64, num_layers=2,
                                                    vocab_size=128)
    cfg.use_rfp = use_rfp
    cfg.max_sequence_length = 128
    from simplcore_pg.model import SimplCoreModel
    model = SimplCoreModel(cfg).eval()
    L, split = 64, 48
    idx = torch.randint(0, cfg.vocab_size, (1, L))

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids, **kw):
            return self.m(ids.float() if False else ids)

    with torch.no_grad():
        y1 = model(idx)
        idx2 = idx.clone()
        idx2[:, split:] = torch.randint(0, cfg.vocab_size, (1, L - split))
        y2 = model(idx2)
    delta = (y1[:, :split] - y2[:, :split]).abs().max().item()
    assert delta < 1e-4, f"SimplCoreModel (use_rfp={use_rfp}) leaks the future: {delta}"


def test_simplcore_moe_model_is_causal():
    torch.manual_seed(0)
    cfg = SimplCoreConfig.get_attention_free_config(hidden_size=64, num_layers=2,
                                                    vocab_size=128)
    cfg.num_experts = 4
    cfg.max_sequence_length = 128
    from simplcore_pg.model_moe import SimplCoreMoEModel
    model = SimplCoreMoEModel(cfg).eval()
    L, split = 64, 48
    idx = torch.randint(0, cfg.vocab_size, (1, L))
    with torch.no_grad():
        y1 = model(idx)
        idx2 = idx.clone()
        idx2[:, split:] = torch.randint(0, cfg.vocab_size, (1, L - split))
        y2 = model(idx2)
    delta = (y1[:, :split] - y2[:, :split]).abs().max().item()
    assert delta < 1e-4, f"SimplCoreMoEModel leaks the future: {delta}"

