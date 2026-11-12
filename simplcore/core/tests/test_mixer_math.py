# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Numerical correctness tests for the SGC mixer.

These are the tests that justify the architectural claims in docs/SGC_CORE.md:

* the FFT (parallel) form and the IIR (streaming) form are the *same operator*;
* the operator is strictly causal;
* the receptive field is not capped by any ``max_len``;
* the streaming state does not grow with sequence length;
* every parameter, including the oscillator decay/frequency parameters, receives
  gradient;
* full-forward generation and O(1)-state streaming generation agree.
"""
from __future__ import annotations

import math

import pytest
import torch

from simplcore.core.config import SimplCoreConfig, get_preset
from simplcore.core.sgc import SpectralGlobalConv, fft_causal_conv
from simplcore.core.model import SimplCoreModel


# ---------------------------------------------------------------------------
# reference implementation
# ---------------------------------------------------------------------------
def explicit_causal_conv(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Direct O(L^2) causal convolution, used as ground truth."""
    B, L, D = u.shape
    y = torch.zeros_like(u)
    for t in range(L):
        for s in range(t + 1):
            y[:, t, :] += u[:, s, :] * kernel[t - s].unsqueeze(0)
    return y


def _random_mixer(dim=16, num_bases=6, local_kernel=4, seed=0) -> SpectralGlobalConv:
    torch.manual_seed(seed)
    m = SpectralGlobalConv(dim=dim, num_bases=num_bases, local_kernel=local_kernel)
    with torch.no_grad():
        m.coef.normal_(0, 0.3)
        m.in_proj.weight.normal_(0, 0.2)
        m.out_proj.weight.normal_(0, 0.2)
        m.local_conv.weight.normal_(0, 0.3)
    return m


# ---------------------------------------------------------------------------
# 1. conv correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 2, 7, 16, 33, 64])
def test_fft_causal_conv_matches_explicit(L):
    torch.manual_seed(0)
    D = 8
    u = torch.randn(2, L, D)
    k = torch.randn(L, D)
    got = fft_causal_conv(u, k, torch.float32)
    ref = explicit_causal_conv(u, k)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


def test_mixer_kernel_matches_explicit():
    """The mixer output equals an explicit causal convolution with its own K."""
    m = _random_mixer()
    L = 24
    x = torch.randn(2, L, m.dim)
    with torch.no_grad():
        h = m.norm(x)
        u = m.in_proj(h) * torch.sigmoid(m.gate_proj(h))
        K = m.build_kernel(L)
        local = m._local_conv_full(u)
        long = explicit_causal_conv(u, K)
        if m.evidence_gate:
            long = long * m._evidence_weights(u)
        if m.uncertainty_gate:
            long = long * m._uncertainty_weights(long, local)
        manual = long + local + m.feed * u
        manual = m.out_proj(manual)
        expected = x + m.out_scale * manual
        got = m(x)
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


# ---------------------------------------------------------------------------
# 2. the dual form: FFT (parallel) == IIR (streaming)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 4, 17, 64, 129])
def test_streaming_matches_parallel(L):
    m = _random_mixer()
    m.eval()
    x = torch.randn(2, L, m.dim)
    with torch.no_grad():
        y_par = m(x)
        st = m.init_state(2, x.device, x.dtype)
        outs = []
        for t in range(L):
            y_t, st = m.step(x[:, t], st)      # y_t: (B, D)
            outs.append(y_t)
        y_seq = torch.stack(outs, dim=1)       # (B, L, D)
    err = (y_par - y_seq).abs().max().item()
    assert err < 1e-4, f"parallel/streaming mismatch {err}"


def test_streaming_state_is_constant_in_length():
    m = _random_mixer(num_bases=6)
    sizes = []
    for L in (8, 128, 2048):
        st = m.init_state(1, torch.device("cpu"), torch.float32)
        x = torch.randn(1, 1, m.dim)
        with torch.no_grad():
            for _ in range(min(L, 32)):          # state shape is set at init
                _, st = m.step(x, st)
        sizes.append(st.y1.numel() + st.y2.numel()
                     + (st.conv_buf.numel() if st.conv_buf is not None else 0))
    assert len(set(sizes)) == 1, f"state grew with length: {sizes}"
    assert m.streaming_state_bytes(1) == (
        2 * 6 * m.dim * 4 + (m.local_kernel - 1) * m.dim * 4
        # L2 证据门的运行前缀能量和：每层一个 fp32 标量
        + (4 if m.evidence_gate else 0)
    )


# ---------------------------------------------------------------------------
# 3. causality and receptive field
# ---------------------------------------------------------------------------
def test_mixer_is_strictly_causal():
    m = _random_mixer()
    m.eval()
    L = 32
    x = torch.randn(1, L, m.dim)
    with torch.no_grad():
        y1 = m(x)
        x2 = x.clone()
        x2[:, L // 2:, :] = torch.randn(1, L - L // 2, m.dim) * 100
        y2 = m(x2)
    delta = (y1[:, : L // 2] - y2[:, : L // 2]).abs().max().item()
    assert delta < 1e-5, f"future leaks into the past: {delta}"


def test_receptive_field_is_not_capped():
    """A change at position 0 must still affect the last position (no max_len cap)."""
    m = _random_mixer(num_bases=8)
    m.eval()
    L = 512
    x = torch.zeros(1, L, m.dim)
    with torch.no_grad():
        y1 = m(x)
        x2 = x.clone()
        x2[:, 0, :] = 5.0                     # perturb the very first token only
        y2 = m(x2)
    influence = (y1[:, -1] - y2[:, -1]).abs().max().item()
    assert influence > 1e-6, "position 0 has no influence at position L-1"


# ---------------------------------------------------------------------------
# 4. gradients
# ---------------------------------------------------------------------------
def test_all_parameters_receive_gradient():
    m = _random_mixer()
    L = 16
    x = torch.randn(2, L, m.dim)
    m(x).pow(2).mean().backward()
    missing = []
    for name, p in m.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            missing.append(name)
    assert not missing, f"parameters without gradient: {missing}"


def test_oscillator_params_get_gradient_through_kernel():
    """rho/theta must be reachable, i.e. coef is not initialised to zero."""
    torch.manual_seed(0)
    m = SpectralGlobalConv(dim=8, num_bases=4)
    x = torch.randn(2, 12, 8)
    m(x).pow(2).mean().backward()
    assert m.raw_rho.grad is not None and m.raw_rho.grad.abs().sum() > 0
    assert m.raw_theta.grad is not None and m.raw_theta.grad.abs().sum() > 0


def test_no_positional_embedding_table():
    cfg = get_preset("debug")
    m = SimplCoreModel(cfg)
    names = dict(m.named_parameters())
    assert not any("pos_emb" in n or "position" in n for n in names), \
        "SGC must not carry a positional-embedding table"


# ---------------------------------------------------------------------------
# 5. model level
# ---------------------------------------------------------------------------
def test_model_forward_backward():
    cfg = SimplCoreConfig(vocab_size=128, dim=32, num_layers=2, num_bases=8,
                     local_kernel=4, num_heads=4)
    m = SimplCoreModel(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 32, cfg.vocab_size)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


@pytest.mark.parametrize("preset", ["debug", "toy"])
def test_preset_builds_and_runs(preset):
    cfg = get_preset(preset)
    m = SimplCoreModel(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        out = m(idx)
    assert out.shape == (1, 24, cfg.vocab_size)
    assert m.num_params() > 0


def test_streaming_generation_matches_full_forward():
    """Greedy streaming generation must equal greedy full-forward generation."""
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                     local_kernel=4, dropout=0.0)
    torch.manual_seed(0)
    m = SimplCoreModel(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        a = m.generate(prompt, max_new_tokens=6, temperature=0.0, use_state=True)
        b = m.generate(prompt, max_new_tokens=6, temperature=0.0, use_state=False)
    assert torch.equal(a, b), f"streaming {a.tolist()} != full {b.tolist()}"


def test_length_extrapolation_beyond_training_length():
    """Inference longer than anything seen in training stays finite (no pos table)."""
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8, local_kernel=4)
    m = SimplCoreModel(cfg).eval()
    with torch.no_grad():
        for L in (16, 1024):
            out = m(torch.randint(0, cfg.vocab_size, (1, L)))
            assert torch.isfinite(out).all(), f"non-finite output at L={L}"


def test_non_sgc_mixer_runs():
    """Any non-attention mixer (here SRFN) must run, backward, and shape correctly."""
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, mixer="srf",
                         srf_heads=2, srf_chunk=8, num_heads=2)
    m = SimplCoreModel(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(idx, idx)
    loss.backward()
    assert logits.shape == (2, 16, cfg.vocab_size)


# ---------------------------------------------------------------------------
# 6. structural ablations must actually change the model
# ---------------------------------------------------------------------------
def test_ablation_variants_differ():
    base = dict(vocab_size=64, dim=32, num_layers=1, num_bases=8, local_kernel=4)
    full = SimplCoreModel(SimplCoreConfig(mixer="sgc", **base))
    nogate = SimplCoreModel(SimplCoreConfig(mixer="sgc-nogate", **base))
    nolocal = SimplCoreModel(SimplCoreConfig(mixer="sgc-nolocal", **base))
    assert full.num_params() > nogate.num_params()
    assert full.num_params() > nolocal.num_params()


def test_analytic_params_matches_real_model():
    """``analytic_params`` must equal ``SimplCoreModel(cfg).num_params()`` exactly.

    The preset docs quote analytic counts for models too large to instantiate
    (1.3B/3B); this test anchors that formula against real construction for every
    mixer kind, plus a scaled-up config with ``out_gate`` enabled.
    """
    from simplcore.core.config import analytic_params, SIMPLCORE_PRESETS

    for mixer, extra in [("sgc", {}), ("sgc-nogate", {}), ("sgc-nolocal", {}),
                         ("snn-sgc", {}),
                         ("srf", {"srf_heads": 2, "srf_chunk": 8}),
                         ("sgc", {"gate_ab": True, "num_heads": 4})]:
        cfg = SimplCoreConfig(vocab_size=128, dim=32, num_layers=3, num_bases=8,
                         local_kernel=4, mixer=mixer, **extra)
        assert analytic_params(cfg) == SimplCoreModel(cfg).num_params(), mixer

    cfg = SimplCoreConfig(vocab_size=512, dim=192, num_layers=4, num_bases=64,
                     decay_max=1024, local_kernel=4, out_gate=True)
    assert analytic_params(cfg) == SimplCoreModel(cfg).num_params()

    for name in SIMPLCORE_PRESETS:
        c = SimplCoreConfig.from_dict(SIMPLCORE_PRESETS[name].to_dict())
        assert analytic_params(c) > 0
