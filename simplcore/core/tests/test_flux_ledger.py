# -*- coding: utf-8 -*-
"""Correctness tests for the independent SimplCore FluxLedger prototype."""
from __future__ import annotations

import torch

from simplcore.core.arch.flux_ledger import (
    FluxLedgerMixer,
    SimplCoreFluxLanguageModel,
)


def test_flux_ledger_is_causal() -> None:
    torch.manual_seed(7)
    model = SimplCoreFluxLanguageModel(
        vocab_size=41, dim=24, num_layers=2, state_dim=12)
    model.eval()
    tokens = torch.randint(0, 41, (2, 20))
    changed = tokens.clone()
    changed[:, 11:] = torch.randint(0, 41, changed[:, 11:].shape)

    with torch.no_grad():
        logits, _, _ = model(tokens)
        changed_logits, _, _ = model(changed)

    torch.testing.assert_close(logits[:, :11], changed_logits[:, :11], rtol=0, atol=0)
    assert not torch.equal(logits[:, 11:], changed_logits[:, 11:])


def test_chunked_execution_matches_full_sequence() -> None:
    torch.manual_seed(13)
    model = SimplCoreFluxLanguageModel(
        vocab_size=37, dim=16, num_layers=3, state_dim=7)
    model.eval()
    tokens = torch.randint(0, 37, (2, 23))

    with torch.no_grad():
        full_logits, _, full_states = model(tokens)
        first_logits, _, states = model(tokens[:, :8])
        second_logits, _, states = model(tokens[:, 8:17], states=states)
        third_logits, _, states = model(tokens[:, 17:], states=states)
    chunk_logits = torch.cat((first_logits, second_logits, third_logits), dim=1)

    torch.testing.assert_close(chunk_logits, full_logits, rtol=0, atol=0)
    for expected, actual in zip(full_states, states):
        torch.testing.assert_close(expected.baseline, actual.baseline, rtol=0, atol=0)
        torch.testing.assert_close(expected.surprise, actual.surprise, rtol=0, atol=0)


def test_state_size_is_constant_and_values_remain_finite() -> None:
    torch.manual_seed(29)
    mixer = FluxLedgerMixer(dim=32, state_dim=9)
    state = mixer.init_state(batch_size=3)
    initial_numel = state.baseline.numel() + state.surprise.numel()
    for _ in range(500):
        sample = torch.randn(3, 32) * 100
        _, state = mixer.step(sample, state)
    assert state.baseline.numel() + state.surprise.numel() == initial_numel
    assert state.baseline.shape == (3, 9)
    assert state.surprise.shape == (3, 9)
    assert torch.isfinite(state.baseline).all()
    assert torch.isfinite(state.surprise).all()
    assert torch.all(state.surprise >= 0)


def test_training_loss_and_gradients_are_finite() -> None:
    torch.manual_seed(31)
    model = SimplCoreFluxLanguageModel(
        vocab_size=53, dim=24, num_layers=2, state_dim=10)
    inputs = torch.randint(0, 53, (4, 17))
    targets = torch.randint(0, 53, (4, 17))
    targets[0, :3] = -100

    logits, loss, states = model(inputs, targets)
    assert logits.shape == (4, 17, 53)
    assert loss is not None and torch.isfinite(loss)
    assert len(states) == 2
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters()]
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_model_rejects_malformed_input_and_state() -> None:
    model = SimplCoreFluxLanguageModel(
        vocab_size=17, dim=8, num_layers=2, state_dim=4)
    try:
        model(torch.ones(2, 3, 4, dtype=torch.long))
    except ValueError as exc:
        assert "input_ids" in str(exc)
    else:
        raise AssertionError("non-rank-2 token IDs should be rejected")

    try:
        model(torch.ones(2, 3, dtype=torch.long), states=())
    except ValueError as exc:
        assert "layer states" in str(exc)
    else:
        raise AssertionError("wrong number of layer states should be rejected")
