# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Tests for :mod:`simplcore.sgc.eval`.

Budget: the whole module must finish in well under 120 s on CPU.  Everything
here is deliberately tiny (dim <= 64, seq <= 128, <= 6 throughput points); the
heavy sweeps belong in the CLI, not in the default test run.

What is asserted, and why it matters:

* the token-weighted loss reproduces a hand-computed ``sum(NLL)/count`` exactly,
  including the ``-100`` padding mask (a mean-of-batch-means implementation
  cannot pass the uneven-mask part of ``test_exact_hand_computed_loss_and_mask``);
* the memory instrumentation reports a positive, width-scaling tensor byte count
  -- the guard against this host's broken ``PROCESS_MEMORY_COUNTERS`` path that
  silently reported 0.0 MB;
* the throughput log-log slope is in a sane range and the accounted peak memory
  grows with sequence length;
* the streaming state byte count is identical for every sequence length, and the
  state's tensor inventory is structurally identical.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.sgc.config import SGCConfig              # noqa: E402
from simplcore.sgc.eval import (                          # noqa: E402
    AssociativeRecallTask,
    EvalBatch,
    IntervalRecallTask,
    ProbeNet,
    SelectiveCopyTask,
    SyntheticDataset,
    _bytes_of,
    _slope_loglog,
    assert_memory_probe_works,
    bench_streaming,
    bench_throughput,
    evaluate_lm,
    evaluate_probe,
    make_report,
    state_dict_bytes,
    tensor_peak_bytes,
    train_probe,
)
from simplcore.sgc.model import SGCModel                 # noqa: E402

DEVICE = torch.device("cpu")


def eval_cfg(dim: int = 32, layers: int = 2, vocab: int = 64, mixer: str = "sgc") -> SGCConfig:
    """dim is always a multiple of 8 so 4 heads divide it exactly."""
    return SGCConfig(vocab_size=vocab, dim=dim, num_layers=layers, num_bases=8,
                      decay_max=128, local_kernel=4, num_heads=4, ffn_ratio=2.0,
                      mixer=mixer)


def tiny_cfg(dim: int = 32) -> SGCConfig:
    return eval_cfg(dim=dim)


class TinyDictDataset:
    """Training-style dataset: ``__getitem__`` returns ``(inputs, targets)`` pairs."""

    def __init__(self, n: int = 6, seq_len: int = 8, vocab: int = 11) -> None:
        self.n, self.seq_len, self.vocab = n, seq_len, vocab

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(1000 + i)
        x = torch.randint(0, self.vocab, (self.seq_len,), generator=g)
        t = torch.full_like(x, -100)
        if i % 3 != 2:                       # indices 2 and 5 carry no counted token
            t[1:] = x[1:]
        return x, t


def hand_reference(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor):
    """float64 ``sum(NLL)`` over every position that is not ``-100``."""
    logits = model(inputs).double()
    logp = F.log_softmax(logits, dim=-1)
    mask = targets != -100
    nll = -logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return float(nll[mask].sum().item()), int(mask.sum().item())


@pytest.fixture(scope="module")
def small_model() -> SGCModel:
    torch.manual_seed(0)
    return SGCModel(tiny_cfg(32)).to(DEVICE).eval()


# ---------------------------------------------------------------------------
# exact token-weighted loss
# ---------------------------------------------------------------------------
def test_exact_hand_computed_loss_and_mask():
    """Exact hand computation: loss = sum(NLL)/count over unmasked tokens only."""
    ds = TinyDictDataset()
    model = SGCModel(tiny_cfg(32)).to(DEVICE).eval()

    nll, count = 0.0, 0
    with torch.no_grad():
        for i in range(len(ds)):
            x, t = ds[i]
            s, c = hand_reference(model, x.unsqueeze(0), t.unsqueeze(0))
            nll += s
            count += c
    expected_loss = nll / count

    res = evaluate_lm(model, ds, batch_size=2, seq_len=8, device=DEVICE)
    assert res["tokens"] == count == 28, (res["tokens"], count)
    assert res["loss"] == pytest.approx(expected_loss, rel=0, abs=1e-5)
    assert res["ppl"] == pytest.approx(math.exp(expected_loss), rel=1e-5)
    assert res["n_batches"] == 3
    assert res["max_abs_dev_from_exact"] < 1e-5

    class Uneven(TinyDictDataset):
        """First row has 4 counted tokens instead of 8 -> uneven batch denominators."""

        def __getitem__(self, i: int):
            x, t = super().__getitem__(i)
            if i == 0:
                t = t.clone()
                t[4:] = -100
            return x, t

    uneven = Uneven()
    res_u = evaluate_lm(model, uneven, batch_size=2, seq_len=8, device=DEVICE)
    nll_u, count_u = 0.0, 0
    means = []
    with torch.no_grad():
        for i in range(len(uneven)):
            x, t = uneven[i]
            s, c = hand_reference(model, x.unsqueeze(0), t.unsqueeze(0))
            nll_u += s
            count_u += c
            if c:
                means.append(s / c)
    assert res_u["tokens"] == count_u == 24
    assert res_u["loss"] == pytest.approx(nll_u / count_u, abs=1e-5)
    # the token-weighted loss MUST differ from the mean of batch means here
    assert abs(res_u["loss"] - sum(means) / len(means)) > 1e-6


def test_shift_objective_counts_seq_len_minus_one():
    ds = SyntheticDataset(vocab_size=17, seq_len=8, length=4, seed=3)
    model = SGCModel(tiny_cfg(32)).to(DEVICE).eval()
    res = evaluate_lm(model, ds, batch_size=2, seq_len=8, device=DEVICE)
    assert res["tokens"] == 4 * 7, res["tokens"]
    assert res["n_batches"] == 2
    assert res["weighting"] == "token_weighted_sum_nll_over_counted_tokens"
    assert res["loss"] > 0 and math.isfinite(res["ppl"])


def test_eval_batch_pair_is_used_verbatim():
    """When the dataset supplies explicit targets, the harness must not re-shift."""
    model = SGCModel(tiny_cfg(32)).to(DEVICE).eval()
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 64, (3, 10), generator=g)
    t = torch.full_like(x, -100)
    t[:, 9] = x[:, 0]
    ds = [EvalBatch(x[i], t[i]) for i in range(3)]
    res = evaluate_lm(model, ds, batch_size=3, seq_len=10, device=DEVICE)
    assert res["tokens"] == 3, res["tokens"]
    ref, cnt = hand_reference(model, x, t)
    assert cnt == 3
    assert res["loss"] == pytest.approx(ref / cnt, abs=1e-5)


def test_evaluate_lm_rejects_all_masked_and_empty_datasets():
    model = SGCModel(tiny_cfg(32)).to(DEVICE).eval()
    with pytest.raises(ValueError):
        evaluate_lm(model, [], batch_size=1, seq_len=8, device=DEVICE)
    x = torch.zeros(2, 8, dtype=torch.long)
    t = torch.full((2, 8), -100, dtype=torch.long)
    with pytest.raises(ValueError):
        evaluate_lm(model, [EvalBatch(x, t)], batch_size=1, seq_len=8, device=DEVICE)


# ---------------------------------------------------------------------------
# memory instrumentation
# ---------------------------------------------------------------------------
def test_memory_probe_is_positive_and_scales_with_width():
    r32 = tensor_peak_bytes(SGCModel(tiny_cfg(32)).to(DEVICE).train(),
                            torch.randint(0, 64, (2, 32)), torch.randint(0, 64, (2, 32)))
    r64 = tensor_peak_bytes(SGCModel(tiny_cfg(64)).to(DEVICE).train(),
                            torch.randint(0, 64, (2, 32)), torch.randint(0, 64, (2, 32)))
    for r in (r32, r64):
        assert r["tensor_peak_bytes"] > 0
        assert r["accounted_bytes"] > 0
        assert r["param_bytes"] > 0
        assert r["grad_bytes"] > 0
        assert r["fwd_bytes"] > 0
        assert r["bwd_bytes"] > 0
        assert r["tensor_peak_bytes"] >= r["param_bytes"]
    assert r64["param_bytes"] > r32["param_bytes"]
    assert r64["accounted_bytes"] > r32["accounted_bytes"]
    assert r64["tensor_peak_bytes"] > r32["tensor_peak_bytes"]
    assert r64["param_bytes"] == state_dict_bytes(SGCModel(tiny_cfg(64)))


def test_memory_probe_self_check_and_growth():
    out = assert_memory_probe_works(device="cpu", dtype="float32")
    assert out["ok"] is True
    ws = out["checks"]["width_scaling"]
    assert ws["accounted_bytes_32"] > 0
    assert ws["accounted_bytes_64"] > ws["accounted_bytes_32"]
    assert ws["param_bytes_64"] > ws["param_bytes_32"]
    assert out["checks"]["stub_exact"]["fwd_bytes"] == out["checks"]["stub_exact"]["expected"]
    assert _bytes_of(torch.zeros(10, 3, dtype=torch.float32)) == 120


def test_memory_probe_distinguishes_frozen_from_untrained_parameters():
    """The probe measures real tensors: frozen params contribute 0 grad bytes.

    An OS-counter-based implementation could not distinguish these two cases,
    which is exactly how the broken RSS path silently returned 0.0 MB.
    """
    inputs = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    live = SGCModel(tiny_cfg(32)).to(DEVICE).train()
    r_live = tensor_peak_bytes(live, inputs, targets)
    assert r_live["grad_bytes"] > 0
    assert r_live["bwd_bytes"] > 0

    frozen = SGCModel(tiny_cfg(32)).to(DEVICE).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    r_frozen = tensor_peak_bytes(frozen, inputs, targets, backward=False)
    assert r_frozen["param_bytes"] == r_live["param_bytes"] > 0
    assert r_frozen["grad_bytes"] == 0
    assert r_frozen["accounted_bytes"] > r_frozen["param_bytes"]


# ---------------------------------------------------------------------------
# throughput scaling
# ---------------------------------------------------------------------------
def test_throughput_slope_and_memory_growth():
    """Slope sanity + accounted memory growth.

    The lengths are chosen so that kernel time dominates Python overhead: on a
    tiny model at seq 32 one forward is a few hundred microseconds and the
    log-log fit is dominated by measurement noise (a negative slope on a trivially
    linear operation is pure noise).  The assertion band is therefore deliberately
    wide -- it is a *sanity* check that the estimate is a finite, non-quadratic
    exponent, not a precision claim.
    """
    torch.manual_seed(0)
    model = SGCModel(eval_cfg(32)).to(DEVICE)
    lengths = (128, 256, 512)
    # single thread: with 8 worker threads the per-iteration spread on this box is
    # ~2x, which swamps the slope (a *fitted* exponent is only meaningful when the
    # measurement noise is small relative to the effect).  `reduce="min"` is the
    # companion choice for the same reason.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows, scaling = bench_throughput(model, seq_lens=lengths, batch_size=8,
                                         device=DEVICE, warmup=3, iters=7, backward=False,
                                         vocab_size=64, reduce="min")
    finally:
        torch.set_num_threads(prev_threads)
    times = [r["median_latency_ms"] for r in rows]
    mems = [r["peak_accounted_bytes"] for r in rows]
    assert all(t > 0 for t in times), times
    assert all(m > 0 for m in mems), mems
    assert all(r["tokens_per_s"] > 0 for r in rows)
    assert all(r["param_bytes"] > 0 for r in rows)
    slope = scaling["latency_loglog_slope"]
    assert math.isfinite(slope)
    # the fit must be self-consistent, and the min-of-iters estimator must agree
    # with the reduced one when reduction IS min
    assert slope == pytest.approx(_slope_loglog(lengths, times), rel=1e-12)
    assert scaling["latency_loglog_slope_min_of_iters"] == pytest.approx(slope, rel=1e-9)
    # Ratio form of the exponent: seq_len grows 4x across these three lengths, so a
    # linear operator implies a ~4x latency ratio and a quadratic one ~16x.  The
    # ratio is far more robust than a fitted slope when the per-point latency is
    # only a few milliseconds.
    ratio = times[-1] / times[0]
    assert 2.0 <= ratio <= 8.0, (ratio, times, slope)
    assert slope < 1.5, (slope, scaling)
    # accounted memory must grow with the sequence length (no silent 0)
    assert scaling["peak_memory_loglog_slope"] > 0.3, scaling
    assert mems[2] > mems[0] > 0
    # latency must grow with the sequence length overall.  Single-pair comparisons
    # are unreliable on a shared CPU (the 512-point FFT can beat the 128-point one
    # by luck of thermal/scheduling state), so require the trend, not every pair.
    assert times[-1] > times[0], times


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------
def test_streaming_state_is_length_invariant():
    torch.manual_seed(0)
    model = SGCModel(eval_cfg(32, layers=2)).to(DEVICE)
    out = bench_streaming(model, lengths=(16, 64), batch_size=1, device=DEVICE, iters=4)
    rows = out["rows"]
    assert out["state_bytes_length_invariant"] is True
    assert out["state_bytes_by_length"]["16"] == out["state_bytes_by_length"]["64"] > 0
    assert len({r["state_bytes"] for r in rows}) == 1
    assert all(r["state_matches_formula"] for r in rows)
    # length invariance is structural, not a timing artefact: the state tensor
    # inventory is identical for both lengths
    inventory = {}
    for L in (16, 64):
        state = model.init_stream(1, device=DEVICE, dtype=torch.float32)
        state.length = L                      # pretend L tokens were consumed
        shapes = tuple(sorted(tuple(t.shape) for st in state.layers
                              for t in (st.y1, st.y2, st.conv_buf, st.energy_sum)
                              if t is not None))
        inventory[L] = (shapes, state.num_bytes())
    assert inventory[16] == inventory[64]
    expected = sum(2 * blk.mixer.num_bases * blk.mixer.dim
                   + max(blk.mixer.local_kernel - 1, 0) * blk.mixer.dim
                   # L2 证据门的运行前缀能量和：每层一个 fp32 标量
                   + (1 if getattr(blk.mixer, "evidence_gate", False) else 0)
                   for blk in model.blocks) * 4
    assert inventory[16][1] == expected > 0
    per_tok = {r["length"]: r["state_bytes_per_token"] for r in rows}
    assert per_tok[64] < per_tok[16]
    assert all(r["step_latency_ms"] > 0 for r in rows)


def test_streaming_state_bytes_do_not_depend_on_length_or_content():
    torch.manual_seed(0)
    model = SGCModel(eval_cfg(32, layers=2)).to(DEVICE)
    seen = set()
    for L in (8, 32, 256):
        state = model.init_stream(1, device=DEVICE, dtype=torch.float32)
        x = torch.randint(0, 64, (1, L))
        with torch.no_grad():
            for t in range(L):
                _, state = model.stream_step(x[:, t], state)
        seen.add(state.num_bytes())
    assert len(seen) == 1
    assert model.stream_bytes(1, torch.float32) == seen.pop()


# ---------------------------------------------------------------------------
# probes: mechanics, causality, (gated) solvability
# ---------------------------------------------------------------------------
def test_probe_task_invariants():
    g = torch.Generator().manual_seed(0)
    copy = SelectiveCopyTask(n_content=8, seq_len=32)
    x, y = copy.sample(16, g)
    assert x.shape == y.shape == (16, 32)
    assert int((y != -100).sum()) == 16
    for b in range(16):
        mk = (x[b] == copy.marker).nonzero().flatten()
        assert mk.numel() == 1
        assert int(x[b, mk[0] + 1]) == int(y[b, copy.answer_pos])
        assert int(x[b, copy.answer_pos - 1]) == copy.query
    assert copy.chance == pytest.approx(1 / 8)

    mqar = AssociativeRecallTask(num_kv=4, num_queries=3, vocab_key=4, vocab_value=4)
    x, y = mqar.sample(16, g)
    for b in range(16):
        table = {int(x[b, 1 + 2 * i]): int(x[b, 2 + 2 * i]) for i in range(4)}
        for j in range(3):
            assert table[int(x[b, mqar.query_pos[j]])] == int(y[b, mqar.answer_pos[j]])
    assert int((y != -100).sum()) == 16 * 3
    assert mqar.chance == pytest.approx(1 / 4)
    assert mqar.vocab_size > max(mqar.pad, mqar.value_base)

    for G in (0, 4, 32):
        iv = IntervalRecallTask(distractor_gap=G, n_values=8)
        x, y = iv.sample(8, g)
        assert int(x[0, 1]) == iv.query_key and int(x[0, 3 + G]) == iv.query_key
        assert int(x[0, 2]) == int(y[0, iv.answer_pos])
        assert int((y != -100).sum()) == 8
        assert iv.answer_pos == iv.seq_len - 1 == G + 4
        assert 0 <= int(y[0, iv.answer_pos]) < iv.n_values
        if G > 0:
            distractors = x[:, 3:3 + G]
            assert int(distractors.min()) >= iv.content_base
            assert int(distractors.max()) < iv.query_key
            # distractors must never collide with a legal answer value
            assert int(distractors.max()) >= iv.n_values
            # and they must actually vary, otherwise the task is solvable by a
            # fixed-offset copy instead of by key retrieval
            assert len({int(v) for v in distractors.flatten()}) > 1
    assert IntervalRecallTask(distractor_gap=4, n_values=8).chance == pytest.approx(1 / 8)


def test_probe_batches_are_always_in_range_after_allocator_reuse():
    """Regression: uninitialised ``torch.empty`` memory must never reach the model.

    The MQAR sampler used to allocate its id buffer with ``torch.empty`` and then
    write only some slots; after a few training steps the allocator recycled
    storage from freed float64 tensors and the model received ids around 4e18.
    This test frees float64 tensors first, then crosses the batch boundaries where
    the bug appeared.
    """
    scratch = [torch.rand((32, 13), dtype=torch.float64) for _ in range(8)]
    scratch = [s * 1e-29 for s in scratch]      # bit patterns in the observed range
    del scratch
    task = AssociativeRecallTask(num_kv=4, num_queries=2, vocab_key=4, vocab_value=4)
    gen = torch.Generator().manual_seed(991)
    for _ in range(40):
        x, y = task.sample(32, gen, device=torch.device("cpu"))
        assert int(x.min()) >= 0 and int(x.max()) < task.vocab_size
        assert int(y[y != -100].min()) >= 0 and int(y.max()) < task.vocab_size
    # and the same guard fires for a hostile custom device-object call pattern
    for dev in ("cpu", torch.device("cpu")):
        x, _ = task.sample(8, gen, device=dev)
        assert x.device.type == "cpu" and x.dtype == torch.long


def test_probe_eval_is_deterministic_and_scores_only_answer_positions():
    torch.manual_seed(0)
    task = AssociativeRecallTask(num_kv=4, num_queries=2, vocab_key=4, vocab_value=4)
    model = ProbeNet(task.vocab_size, dim=32, num_layers=2, global_kind="conv").to(DEVICE).eval()
    a = evaluate_probe(model, task, n_samples=64, batch_size=16, seed=7, device=DEVICE)
    b = evaluate_probe(model, task, n_samples=64, batch_size=16, seed=7, device=DEVICE)
    assert a["probe_accuracy"] == b["probe_accuracy"]
    assert a["n_scored_positions"] == 64 * 2
    assert 0.0 <= a["probe_accuracy"] <= 1.0
    assert a["chance"] == pytest.approx(0.25)
    assert a["probe_loss"] > 0
    assert "freshly sampled" in a["test_set"]


def test_probe_network_is_causal_and_deterministic():
    """Changing the future must not change the past (guards the audit finding).

    The audited ``verify_sgc`` mixer was non-causal and non-deterministic; this
    asserts both properties for the probe network's global and local paths.
    """
    torch.manual_seed(0)
    task = SelectiveCopyTask(n_content=8, seq_len=24)
    for kind in ("attn", "conv"):
        model = ProbeNet(task.vocab_size, dim=32, num_layers=2, global_kind=kind)
        model.eval()
        g = torch.Generator().manual_seed(1)
        x = torch.randint(0, task.vocab_size, (2, 24), generator=g)
        with torch.no_grad():
            full = model(x)
            head = model(x[:, :12])
            again = model(x)
        assert torch.allclose(full[:, :12], head, atol=1e-6), kind
        assert torch.equal(full, again), kind


@pytest.mark.skipif(os.environ.get("SGC_PROBE_TEST") != "1",
                    reason="probe training is the slow part; set SGC_PROBE_TEST=1 to include it")
def test_train_probe_learning_curve_beats_chance():
    t0 = time.perf_counter()
    task = AssociativeRecallTask(num_kv=4, num_queries=2, vocab_key=4, vocab_value=4)
    r = train_probe(task, steps=150, batch_size=64, lr=1e-2, dim=32, num_layers=2,
                    device=DEVICE, log=False, progress=False, final_eval_samples=256)
    assert len(r["curve"]) >= 2
    assert r["final_accuracy"] >= task.chance
    assert time.perf_counter() - t0 < 60


# ---------------------------------------------------------------------------
# report envelope
# ---------------------------------------------------------------------------
def test_report_records_provenance():
    rep = make_report({"x": 1}, command="python -m simplcore.sgc.eval self-test",
                      device=DEVICE, dtype=torch.float32)
    p = rep["provenance"]
    assert p["command_line"] == "python -m simplcore.sgc.eval self-test"
    assert p["torch_version"] == torch.__version__
    assert p["device"] == "cpu"
    assert p["dtype"] == "float32"
    assert "git_commit" in p
    assert "tensor_accounting" in p["memory_method"]
    assert len(p["eval_module_sha256"]) == 64
    assert rep["schema"] == "simplcore.sgc.eval/1"
    assert rep["result"] == {"x": 1}
