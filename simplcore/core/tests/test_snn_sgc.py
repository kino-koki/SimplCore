"""Tests for the experimental signed-LIF SGC mixer."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from simplcore.core.arch.registry import build_mixer
from simplcore.core.arch.sgc import SGCStreamState
from simplcore.core.data import ShardedTokenDataset
from simplcore.core.config import SimplCoreConfig, analytic_params
from simplcore.core.dream import Insight, ReflectionConfig, TrainExample
from simplcore.core.envlearn import ForwardAdapter, LearnedAdapter
from simplcore.core.eval import compare_snn_sgc_training
from simplcore.core.memory import MemoryCompressor
from simplcore.core.model import SimplCoreModel
from simplcore.core.smm import SMMConfig, SimplModelMemory
from simplcore.core.tokenization import CharTokenizer
from simplcore.core.training.train.data_windows import unpack_lm_batch
from simplcore.core.training.train.data_windows import LMWindows
from simplcore.core.training.data import DataPipelineError, compute_sha256
from simplcore.core.training.train.evaluate import evaluate


def _config(**overrides) -> SimplCoreConfig:
    values = dict(
        vocab_size=32,
        dim=16,
        num_layers=1,
        num_bases=4,
        local_kernel=3,
        num_heads=2,
        ffn_ratio=2.0,
        mixer="snn-sgc",
    )
    values.update(overrides)
    return SimplCoreConfig(**values)


def test_config_rejects_invalid_snn_settings() -> None:
    for field, value in (
        ("snn_decay", -0.1),
        ("snn_decay", 1.0),
        ("snn_decay", float("nan")),
        ("snn_threshold", 0.0),
        ("snn_threshold", float("inf")),
        ("snn_surrogate_slope", -1.0),
        ("snn_surrogate_slope", float("nan")),
    ):
        with pytest.raises(ValueError):
            _config(**{field: value}).validate()
    with pytest.raises(ValueError, match="gate_ab"):
        _config(gate_ab=True).validate()


def test_config_rejects_invalid_ncm_settings() -> None:
    for field, value in (
        ("ncm_dim", 0),
        ("ncm_dim", 17),
        ("ncm_alpha", -0.1),
        ("ncm_alpha", float("nan")),
        ("ncm_beta", 1.1),
        ("ncm_beta", float("inf")),
    ):
        with pytest.raises(ValueError):
            _config(use_ncm=True, **{field: value}).validate()


def test_snn_spikes_are_ternary_and_surrogate_is_differentiable() -> None:
    mixer = build_mixer(_config())
    current = torch.zeros(1, 1, 16, requires_grad=True)
    with torch.no_grad():
        current[0, 0, :3] = torch.tensor([-2.0, 0.2, 2.0])

    spikes, _ = mixer._spike_sequence(current)
    assert set(spikes.detach().unique().tolist()) <= {-1.0, 0.0, 1.0}
    spikes.sum().backward()
    assert current.grad is not None
    assert torch.count_nonzero(current.grad) > 0


def test_snn_surrogate_gradient_reaches_sgc_parameters() -> None:
    mixer = build_mixer(_config())
    x = torch.randn(2, 5, mixer.dim)
    mixer(x).square().mean().backward()
    assert mixer.coef.grad is not None
    assert torch.isfinite(mixer.coef.grad).all()
    assert torch.count_nonzero(mixer.coef.grad) > 0


def test_snn_model_training_step_updates_backbone() -> None:
    cfg = _config()
    model = SimplCoreModel(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.randint(0, cfg.vocab_size, (2, 6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.blocks[0].mixer.coef.detach().clone()
    _, loss = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].mixer.coef.grad is not None
    optimizer.step()
    assert not torch.equal(model.blocks[0].mixer.coef, before)


def test_snn_temporal_recurrence_is_chunk_invariant() -> None:
    mixer = build_mixer(_config()).eval()
    current = torch.randn(2, 11, 16)

    full, full_membrane = mixer._spike_sequence(current)
    first, membrane = mixer._spike_sequence(current[:, :4])
    second, chunk_membrane = mixer._spike_sequence(current[:, 4:], membrane)

    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    torch.testing.assert_close(chunk_membrane, full_membrane)


def test_snn_full_chunk_and_token_stream_agree() -> None:
    mixer = build_mixer(_config()).eval()
    x = torch.randn(2, 11, mixer.dim)
    with torch.no_grad():
        full = mixer(x)

        chunk_state = mixer.init_state(2, x.device, x.dtype)
        first, chunk_state = mixer.forward_chunk(x[:, :4], chunk_state)
        second, chunk_state = mixer.forward_chunk(x[:, 4:], chunk_state)
        chunked = torch.cat((first, second), dim=1)

        token_state = mixer.init_state(2, x.device, x.dtype)
        token_outputs = []
        for token in x.unbind(dim=1):
            output, token_state = mixer.step(token, token_state)
            token_outputs.append(output.unsqueeze(1))
        streamed = torch.cat(token_outputs, dim=1)

    torch.testing.assert_close(chunked, full, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(streamed, full, atol=2e-4, rtol=2e-4)
    assert token_state.spike_membrane.shape == (2, mixer.dim)
    assert SGCStreamState(layers=[token_state]).num_bytes() == \
        mixer.streaming_state_bytes(2, x.dtype)


def test_snn_stream_state_and_checkpoint_preserve_membrane() -> None:
    cfg = _config()
    model = SimplCoreModel(cfg).eval()
    state = model.init_stream(2, device="cpu", dtype=torch.float32)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        for token in ids.unbind(dim=1):
            _, state = model.stream_step(token, state)

    assert state.layers[0].spike_membrane is not None
    assert state.layers[0].spike_membrane.shape == (2, cfg.dim)
    assert state.num_bytes() == model.stream_bytes(2, torch.float32)
    checkpoint = model.checkpoint_stream(state)
    assert checkpoint["layers"][0]["spike_membrane"] is not None

    restored = model.restore_stream(checkpoint)
    torch.testing.assert_close(
        restored.layers[0].spike_membrane,
        state.layers[0].spike_membrane)
    next_token = torch.randint(0, cfg.vocab_size, (2, 1))
    with torch.no_grad():
        logits_a, _ = model.stream_step(next_token, state)
        logits_b, _ = model.stream_step(next_token, restored)
    torch.testing.assert_close(logits_a, logits_b)


def test_snn_membrane_is_preserved_and_detached_with_stream_state() -> None:
    model = SimplCoreModel(_config())
    state = model.init_stream(1, device="cpu", dtype=torch.float32)
    state.layers[0].spike_membrane.requires_grad_()

    detached = model._detach_stream_state(state)
    assert detached.layers[0].spike_membrane is not \
        state.layers[0].spike_membrane
    assert not detached.layers[0].spike_membrane.requires_grad
    state.detach()
    assert not state.layers[0].spike_membrane.requires_grad


def test_ncm_is_opt_in_and_compatible_with_standard_forward() -> None:
    cfg = _config(use_ncm=True, ncm_dim=8)
    model = SimplCoreModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (3, 5))

    logits = model(x)
    assert logits.shape == (3, 5, cfg.vocab_size)
    assert model.ncm is not None

    baseline = SimplCoreModel(_config()).eval()
    baseline.load_state_dict(model.state_dict(), strict=False)
    out_base = baseline(x)
    assert torch.isfinite(logits).all()
    assert not torch.equal(logits, out_base)
    torch.testing.assert_close(model(x), logits)
    assert model.num_params() == analytic_params(cfg)


def test_ncm_is_causal_batch_independent_and_trainable() -> None:
    cfg = _config(use_ncm=True, ncm_dim=8)
    model = SimplCoreModel(cfg).eval()
    x = torch.randn(2, 6, cfg.dim)
    changed = x.clone()
    changed[:, 4:] += 10.0
    with torch.no_grad():
        prefix, _, _ = model.ncm.forward_sequence(x)
        changed_prefix, _, _ = model.ncm.forward_sequence(changed)
    torch.testing.assert_close(prefix[:, :4], changed_prefix[:, :4])

    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        together = model(tokens)
        separate = torch.cat([model(tokens[i:i + 1]) for i in range(2)], dim=0)
    torch.testing.assert_close(together, separate)
    left = model.init_stream(1, device=tokens.device)
    right = model.init_stream(1, device=tokens.device)
    with torch.no_grad():
        _, left = model.stream_step(tokens[:1, 0], left)
        _, right = model.stream_step(tokens[1:, 0], right)
    assert left.concept_state is not right.concept_state
    assert not torch.equal(left.concept_state, right.concept_state)

    model.train()
    _, loss = model(tokens[:, :-1], tokens[:, 1:])
    loss.backward()
    assert model.ncm.proj.weight.grad is not None
    assert model.ncm.out.weight.grad is not None
    assert torch.count_nonzero(model.ncm.proj.weight.grad) > 0


def test_ncm_all_objectives_train_with_optional_targets() -> None:
    cfg = _config(
        use_ncm=True,
        ncm_dim=8,
        ncm_label_classes=5,
        ncm_group_classes=3,
    )
    model = SimplCoreModel(cfg).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    labels = torch.randint(0, cfg.ncm_label_classes, (2, 6))
    groups = torch.randint(0, cfg.ncm_group_classes, (2, 6))
    labels[0, 0] = -100
    _, total_loss, aux = model(
        tokens, tokens, ncm_label_targets=labels,
        ncm_group_targets=groups, return_ncm_aux=True)
    assert set(aux) >= {
        "future_latent_loss", "concept_labels_loss", "token_groups_loss",
        "ncm_loss",
    }
    assert torch.isfinite(total_loss)
    assert aux["future_latent_count"] == tokens.numel()
    assert aux["concept_labels_count"] == tokens.numel() - 1
    assert aux["token_groups_count"] == tokens.numel()
    total_loss.backward()
    assert model.ncm.future_predictor.weight.grad is not None
    assert model.ncm.label_predictor.weight.grad is not None
    assert model.ncm.group_predictor.weight.grad is not None


def test_ncm_legacy_checkpoint_loads_strictly() -> None:
    cfg = _config(use_ncm=True, ncm_dim=8)
    source = SimplCoreModel(cfg)
    legacy = source.state_dict()
    legacy.pop("ncm.future_predictor.weight")
    restored = SimplCoreModel(cfg)
    restored.load_state_dict(legacy, strict=True)
    assert torch.isfinite(restored.ncm.future_predictor.weight).all()


def test_ncm_auxiliary_losses_match_across_chunk_boundaries() -> None:
    cfg = _config(
        use_ncm=True,
        ncm_dim=8,
        ncm_label_classes=5,
        ncm_group_classes=3,
    )
    model = SimplCoreModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.ncm_label_classes, tokens.shape)
    groups = torch.randint(0, cfg.ncm_group_classes, tokens.shape)
    with torch.no_grad():
        full_logits, full_loss, full_aux = model(
            tokens, tokens, ncm_label_targets=labels,
            ncm_group_targets=groups, return_ncm_aux=True)
        state = model.init_stream(tokens.shape[0], device=tokens.device)
        parts = []
        ce_sum = 0.0
        aux_sums = {name: 0.0 for name in (
            "future_latent", "concept_labels", "token_groups")}
        aux_counts = {name: 0 for name in aux_sums}
        for start in range(0, tokens.shape[1], 3):
            end = min(start + 3, tokens.shape[1])
            logits, loss_sum, _, state, aux = model.forward_chunk_step(
                tokens[:, start:end], tokens[:, start:end], state,
                ncm_label_targets=labels[:, start:end],
                ncm_group_targets=groups[:, start:end],
                return_ncm_aux=True)
            parts.append(logits)
            ce_sum += float(loss_sum)
            for name in aux_sums:
                aux_sums[name] += float(aux[f"{name}_loss_sum"])
                aux_counts[name] += int(aux[f"{name}_count"])
            state = model._detach_stream_state(state)

    chunked_logits = torch.cat(parts, dim=1)
    torch.testing.assert_close(chunked_logits, full_logits, atol=3e-4, rtol=3e-4)
    weighted_aux = sum(
        weight * aux_sums[source] / max(aux_counts[source], 1)
        for source, weight in (
            ("future_latent", cfg.ncm_future_weight),
            ("concept_labels", cfg.ncm_label_weight),
            ("token_groups", cfg.ncm_group_weight),
        )
    )
    chunked_total = ce_sum / tokens.numel() + weighted_aux
    torch.testing.assert_close(full_loss, torch.tensor(chunked_total), atol=3e-4, rtol=3e-4)
    assert full_aux["ncm_loss"].item() == pytest.approx(weighted_aux, abs=1e-5)


def test_ncm_optional_targets_fail_clearly_without_configured_head() -> None:
    cfg = _config(use_ncm=True, ncm_dim=8)
    model = SimplCoreModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with pytest.raises(ValueError, match="class count is not configured"):
        model(x, x, ncm_label_targets=x, return_ncm_aux=True)


def test_lm_batch_unpack_accepts_optional_ncm_sources() -> None:
    x = torch.randint(0, 10, (2, 4))
    y = torch.randint(0, 10, (2, 4))
    labels = torch.randint(0, 3, (2, 4))
    groups = torch.randint(0, 2, (2, 4))
    got = unpack_lm_batch({
        "input_ids": x, "targets": y,
        "ncm_label_targets": labels, "ncm_group_targets": groups,
    })
    assert all(a is b for a, b in zip(got, (x, y, labels, groups)))
    with pytest.raises(ValueError, match="same shape"):
        unpack_lm_batch({"input_ids": x, "targets": y, "ncm_label_targets": labels[:, :2]})


def test_ncm_targets_load_from_aligned_dataset_sidecars(tmp_path) -> None:
    tokens = np.arange(8, dtype=np.uint32)
    labels = np.arange(8, dtype=np.int32) % 4
    groups = np.arange(8, dtype=np.int16) % 2
    for name, array in (
        ("tokens-a.bin", tokens[:3]),
        ("tokens-b.bin", tokens[3:]),
        ("labels.bin", labels),
        ("groups-a.bin", groups[:4]),
        ("groups-b.bin", groups[4:]),
    ):
        array.tofile(tmp_path / name)
    manifest = {
        "seq_len": 4,
        "dtype": "uint32",
        "vocab_size": 16,
        "eos_id": 0,
        "splits": {
            "train": {
                "n_tokens": 8,
                "n_windows": 2,
                "files": [
                    {"path": "tokens-a.bin", "n_tokens": 3},
                    {"path": "tokens-b.bin", "n_tokens": 5},
                ],
                "ncm_label_targets": {
                    "dtype": "int32",
                    "files": [{
                        "path": "labels.bin",
                        "n_tokens": 8,
                        "sha256": compute_sha256(tmp_path / "labels.bin"),
                    }],
                },
                "ncm_group_targets": {
                    "dtype": "int16",
                    "files": [
                        {
                            "path": "groups-a.bin",
                            "n_tokens": 4,
                            "sha256": compute_sha256(tmp_path / "groups-a.bin"),
                        },
                        {
                            "path": "groups-b.bin",
                            "n_tokens": 4,
                            "sha256": compute_sha256(tmp_path / "groups-b.bin"),
                        },
                    ],
                },
            }
        },
    }
    (tmp_path / "index.json").write_text(json.dumps(manifest), encoding="utf-8")

    raw = ShardedTokenDataset(tmp_path, "train")
    item = raw[0]
    assert torch.equal(item["tokens"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(item["ncm_label_targets"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(item["ncm_group_targets"], torch.tensor([0, 1, 0, 1]))
    assert raw.num_bytes() == sum((tmp_path / name).stat().st_size for name in (
        "tokens-a.bin", "tokens-b.bin", "labels.bin", "groups-a.bin", "groups-b.bin"))
    restored = pickle.loads(pickle.dumps(raw))
    assert torch.equal(restored[1]["ncm_group_targets"], torch.tensor([0, 1, 0, 1]))
    lm_item = LMWindows(str(tmp_path), "train", seq_len=3,
                        verify_hashes=True)[0]
    assert torch.equal(lm_item["input_ids"], torch.tensor([0, 1, 2]))
    assert torch.equal(lm_item["targets"], torch.tensor([1, 2, 3]))
    assert torch.equal(lm_item["ncm_label_targets"], torch.tensor([0, 1, 2]))
    batch = next(iter(DataLoader(
        LMWindows(str(tmp_path), "train", seq_len=3), batch_size=2)))
    x, y, batch_labels, batch_groups = unpack_lm_batch(batch)
    assert x.shape == y.shape == batch_labels.shape == batch_groups.shape == (2, 3)
    invalid_manifest = json.loads(json.dumps(manifest))
    invalid_manifest["splits"]["train"]["ncm_label_targets"]["files"][0]["n_tokens"] = 7
    (tmp_path / "index.json").write_text(
        json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(DataPipelineError, match="holds 8 tokens, manifest says 7"):
        ShardedTokenDataset(tmp_path, "train", verify_hashes=False)
    invalid_manifest = json.loads(json.dumps(manifest))
    invalid_manifest["splits"]["train"]["ncm_label_targets"]["files"][0][
        "sha256"] = "0" * 64
    (tmp_path / "index.json").write_text(
        json.dumps(invalid_manifest), encoding="utf-8")
    with pytest.raises(DataPipelineError, match="sha256 mismatch"):
        ShardedTokenDataset(tmp_path, "train")


@pytest.mark.parametrize("chunk_size", [0, 2])
def test_ncm_sidecars_reach_training_cli(tmp_path, chunk_size: int) -> None:
    from simplcore.core.training.train.main import main as train_main

    data_dir = tmp_path / "corpus"
    out_dir = tmp_path / f"run-{chunk_size}"
    data_dir.mkdir()
    split_specs = {}
    for split in ("train", "val"):
        tokens = (np.arange(10, dtype=np.uint32) + (split == "val")) % 16
        labels = (np.arange(10, dtype=np.int32) + (split == "val")) % 3
        groups = (np.arange(10, dtype=np.int16) + (split == "val")) % 2
        shard_specs = {}
        for source, array, dtype in (
            ("tokens", tokens, "uint32"),
            ("labels", labels, "int32"),
            ("groups", groups, "int16"),
        ):
            name = f"{split}-{source}.bin"
            path = data_dir / name
            array.tofile(path)
            shard_specs[source] = {
                "path": name,
                "n_tokens": int(array.size),
                "sha256": compute_sha256(path),
            }
        split_specs[split] = {
            "n_tokens": 10,
            "n_windows": 2,
            "files": [shard_specs["tokens"]],
            "ncm_label_targets": {
                "dtype": "int32",
                "files": [shard_specs["labels"]],
            },
            "ncm_group_targets": {
                "dtype": "int16",
                "files": [shard_specs["groups"]],
            },
        }
    manifest = {
        "seq_len": 5,
        "dtype": "uint32",
        "vocab_size": 16,
        "eos_id": 0,
        "splits": split_specs,
    }
    (data_dir / "index.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = [
        "--preset", "debug",
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
        "--seq-len", "4",
        "--max-steps", "1",
        "--micro-batch", "1",
        "--num-workers", "0",
        "--eval-every", "1",
        "--eval-batches", "1",
        "--log-every", "1",
        "--save-every", "10",
        "--max-kernel-gib", "0",
        "--device", "cpu",
        "--dtype", "fp32",
        "--override",
        "vocab_size=16", "dim=16", "ffn_hidden=32", "num_layers=1",
        "num_heads=2", "use_ncm=true", "ncm_dim=8",
        "ncm_label_classes=3", "ncm_group_classes=2",
    ]
    if chunk_size:
        args.extend(["--train-chunk", str(chunk_size)])

    assert train_main(args) == 0
    records = [
        json.loads(line)
        for line in (out_dir / "metrics.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    train_record = next(record for record in records if "loss" in record)
    eval_record = next(record for record in records if "val_loss" in record)
    assert np.isfinite(train_record["ncm_aux_loss"])
    assert train_record["ncm_aux_loss"] > 0
    assert eval_record["ncm_aux"].keys() == {
        "future_latent", "concept_labels", "token_groups"}
    assert all(np.isfinite(value) for value in eval_record["ncm_aux"].values())
    assert (out_dir / "last.pt").is_file()


def test_ncm_evaluation_reports_auxiliary_metrics_and_ce_perplexity() -> None:
    cfg = _config(
        use_ncm=True, ncm_dim=8,
        ncm_label_classes=5, ncm_group_classes=3)
    model = SimplCoreModel(cfg).eval()
    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (2, 6)),
        "targets": torch.randint(0, cfg.vocab_size, (2, 6)),
        "ncm_label_targets": torch.randint(0, 5, (2, 6)),
        "ncm_group_targets": torch.randint(0, 3, (2, 6)),
    }
    full = evaluate(model, [batch], device=torch.device("cpu"), amp_dtype=None)
    chunked = evaluate(
        model, [batch], device=torch.device("cpu"),
        amp_dtype=None, chunk_size=2)
    assert full["ncm_aux"].keys() == {
        "future_latent", "concept_labels", "token_groups"}
    assert full["loss"] == pytest.approx(chunked["loss"], abs=3e-4)
    for source in full["ncm_aux"]:
        assert full["ncm_aux"][source] == pytest.approx(
            chunked["ncm_aux"][source], abs=3e-4)


def test_ncm_full_chunk_stream_and_checkpoint_are_equivalent() -> None:
    cfg = _config(use_ncm=True, ncm_dim=8)
    model = SimplCoreModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 9))
    with torch.no_grad():
        full = model(tokens)
        chunked = model.forward_chunked(tokens, chunk_size=3, detach_state=False)

        state = model.init_stream(2, device=tokens.device)
        streamed = []
        for token in tokens.unbind(dim=1):
            logits, state = model.stream_step(token, state)
            streamed.append(logits.unsqueeze(1))
        streamed = torch.cat(streamed, dim=1)
    torch.testing.assert_close(chunked, full, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(streamed, full, atol=3e-4, rtol=3e-4)
    assert state.concept_state is not None
    assert state.concept_state.shape == (2, cfg.ncm_dim)

    checkpoint = model.checkpoint_stream(state)
    restored = model.restore_stream(checkpoint)
    torch.testing.assert_close(restored.concept_state, state.concept_state)
    next_token = torch.randint(0, cfg.vocab_size, (2,))
    with torch.no_grad():
        logits_a, _ = model.stream_step(next_token, state)
        logits_b, _ = model.stream_step(next_token, restored)
    torch.testing.assert_close(logits_a, logits_b)

    expected_bytes = model.stream_bytes(2, model.tok_emb.weight.dtype)
    assert state.num_bytes() == expected_bytes


def test_ncm_low_precision_keeps_recurrent_state_fp32() -> None:
    cfg = _config(use_ncm=True, ncm_dim=8)
    model = SimplCoreModel(cfg).to(torch.bfloat16).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        full = model(tokens)
        state = model.init_stream(1, device=tokens.device, dtype=torch.bfloat16)
        streamed = []
        for token in tokens.unbind(dim=1):
            logits, state = model.stream_step(token, state)
            streamed.append(logits.unsqueeze(1))

    assert state.concept_state.dtype == torch.float32
    torch.testing.assert_close(
        torch.cat(streamed, dim=1), full, atol=3e-2, rtol=3e-2)
    assert state.num_bytes() == model.stream_bytes(1, torch.bfloat16)


def test_snn_old_stream_checkpoint_without_membrane_restores() -> None:
    model = SimplCoreModel(_config()).eval()
    state = model.init_stream(1, device="cpu", dtype=torch.float32)
    checkpoint = model.checkpoint_stream(state)
    checkpoint["layers"][0].pop("spike_membrane")

    restored = model.restore_stream(checkpoint)
    assert restored.layers[0].spike_membrane is None
    with torch.no_grad():
        logits, restored = model.stream_step(torch.tensor([3]), restored)
    assert torch.isfinite(logits).all()
    assert restored.layers[0].spike_membrane.shape == (1, model.config.dim)


def test_snn_full_and_chunked_paths_return_finite_outputs() -> None:
    mixer = build_mixer(_config()).eval()
    x = torch.randn(2, 9, 16)
    state = mixer.init_state(2, x.device, x.dtype)

    with torch.no_grad():
        full = mixer(x)
        chunked, next_state = mixer.forward_chunk(x, state)

    assert full.shape == x.shape
    assert chunked.shape == x.shape
    assert torch.isfinite(full).all()
    assert torch.isfinite(chunked).all()
    assert next_state is not None
    assert next_state.spike_membrane is not None
    assert SGCStreamState(layers=[next_state]).num_bytes() == \
        mixer.streaming_state_bytes(2, x.dtype)


def test_snn_chunk_without_state_update_does_not_mutate_membrane() -> None:
    mixer = build_mixer(_config()).eval()
    state = mixer.init_state(1, torch.device("cpu"), torch.float32)
    before = state.spike_membrane.clone()
    with torch.no_grad():
        _, returned = mixer.forward_chunk(
            torch.randn(1, 4, mixer.dim), state, update_state=False)

    assert returned is state
    torch.testing.assert_close(state.spike_membrane, before)


def test_snn_cpu_bfloat16_keeps_recurrent_state_in_fp32() -> None:
    cfg = _config()
    model = SimplCoreModel(cfg).to(torch.bfloat16).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        state = model.init_stream(1, device="cpu", dtype=torch.bfloat16)
        for token in tokens.unbind(dim=1):
            logits, state = model.stream_step(token, state)

    assert state.layers[0].y1.dtype == torch.float32
    assert state.layers[0].spike_membrane.dtype == torch.float32
    assert torch.isfinite(logits).all()
    assert state.num_bytes() == model.stream_bytes(1, torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_snn_cuda_bfloat16_training_and_streaming() -> None:
    cfg = _config()
    model = SimplCoreModel(cfg).to(device="cuda", dtype=torch.bfloat16).train()
    tokens = torch.randint(0, cfg.vocab_size, (2, 6), device="cuda")
    targets = torch.randint(0, cfg.vocab_size, (2, 6), device="cuda")
    _, loss = model(tokens, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.blocks[0].mixer.coef.grad).all()

    model.eval()
    with torch.no_grad():
        state = model.init_stream(
            2, device=torch.device("cuda"), dtype=torch.bfloat16)
        for token in tokens.unbind(dim=1):
            logits, state = model.stream_step(token, state)
    assert state.layers[0].spike_membrane.dtype == torch.float32
    assert state.num_bytes() == model.stream_bytes(2, torch.bfloat16)
    assert torch.isfinite(logits).all()


def test_snn_works_with_smm_recall_injection_and_sleep() -> None:
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=24, num_layers=1, num_bases=4, local_kernel=3,
        ffn_ratio=2.0, mixer="snn-sgc", use_memory=True)
    model = SimplCoreModel(cfg).eval()
    smm = SimplModelMemory(
        model, codec,
        SMMConfig(dream=ReflectionConfig(
            mode="memory", require_regression_check=False),
            think_max_iters=1))
    smm.experience("abcd", "efgh", grounded=True, evidence="efgh")
    memory, score, _ = smm.recall("abcd")
    assert memory is not None and score > 0

    result = smm.answer(torch.tensor([codec.encode("abc")]), max_new_tokens=2,
                        query="abcd", temperature=0)
    assert result["memory"] is not None
    assert result["tokens"].shape[1] == len(codec.encode("abc")) + 2
    assert smm.ledger.injected == 1
    assert smm.stats()["working"]["stream_bytes_1"] == model.stream_bytes(1)
    persisted_memory = smm.state_dict()
    model_copy = SimplCoreModel(cfg).eval()
    smm_copy = SimplModelMemory(model_copy, codec, SMMConfig())
    smm_copy.load_state_dict(persisted_memory)
    recalled, _, _ = smm_copy.recall("abcd")
    assert recalled is not None
    report = smm.sleep()
    assert report.round_index == 1
    assert model.config.mixer == "snn-sgc"


def test_snn_hidden_state_api_supports_cmt_memory_compressor() -> None:
    cfg = _config()
    model = SimplCoreModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 5))
    compressor = MemoryCompressor(dim=cfg.dim, k=3)
    with torch.no_grad():
        memory_tokens = compressor(model, ids)
    assert memory_tokens.shape == (2, 3, cfg.dim)
    assert torch.isfinite(memory_tokens).all()


def test_snn_dreaming_rollback_restores_mixer_weights_and_memory() -> None:
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=24, num_layers=1, num_bases=4, local_kernel=3,
        ffn_ratio=2.0, mixer="snn-sgc", use_memory=True)
    model = SimplCoreModel(cfg).eval()
    smm = SimplModelMemory(
        model, codec,
        SMMConfig(dream=ReflectionConfig(
            mode="both", require_regression_check=True)))
    smm.experience("abcd", "efgh", grounded=True, evidence="efgh")
    smm.dream.reflector = lambda _model, trajectories: [
        Insight(text="verified", source=trajectories[0],
                repair="efgh", evidence="efgh")]
    smm.dream.synthesizer = lambda _model, _insights: [
        TrainExample(prompt="abcd", target="efgh")]
    scores = iter(({"score": [1.0]}, {"score": [0.0]}))
    smm.dream.verifier = lambda _model: next(scores)
    smm.dream.verifier_is_default = False

    mixer = model.blocks[0].mixer
    original = mixer.coef.detach().clone()
    bank_size = len(smm.bank)
    def mutate_weights(target, _examples):
        with torch.no_grad():
            target.blocks[0].mixer.coef.add_(1.0)
        return {"steps": 1}
    smm.dream.consolidator = mutate_weights

    report = smm.sleep()
    assert report.rolled_back
    assert report.wrote_memory > 0
    assert smm.ledger.rollbacks == 1
    assert len(smm.bank) == bank_size
    torch.testing.assert_close(mixer.coef, original)
    assert model.config.mixer == "snn-sgc"


def test_snn_works_with_frozen_base_environment_adapter() -> None:
    cfg = _config()
    base = SimplCoreModel(cfg)
    adapter = LearnedAdapter(dim=cfg.dim, num_blocks=1)
    adapted = ForwardAdapter(base, adapter)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    targets = torch.randint(0, cfg.vocab_size, (2, 6))
    _, loss = adapted(tokens, targets)
    loss.backward()

    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in adapter.parameters())


def test_default_sgc_config_does_not_opt_into_snn() -> None:
    cfg = SimplCoreConfig(vocab_size=32, dim=16, num_layers=1, num_bases=4)
    assert cfg.mixer == "sgc"
    model = SimplCoreModel(cfg)
    assert model.blocks[0].mixer.__class__.__name__ == "SpectralGlobalConv"


def test_snn_config_roundtrip_preserves_opt_in_settings() -> None:
    cfg = _config(snn_decay=0.73, snn_threshold=0.24,
                  snn_surrogate_slope=3.5)
    restored = SimplCoreConfig.from_dict(cfg.to_dict())
    assert restored.mixer == "snn-sgc"
    assert restored.snn_decay == cfg.snn_decay
    assert restored.snn_threshold == cfg.snn_threshold
    assert restored.snn_surrogate_slope == cfg.snn_surrogate_slope


def test_snn_and_sgc_weights_load_strictly_in_both_directions() -> None:
    sgc_cfg = _config(mixer="sgc")
    snn_cfg = _config()
    sgc_model = SimplCoreModel(sgc_cfg)
    snn_model = SimplCoreModel(snn_cfg)

    snn_model.load_state_dict(sgc_model.state_dict(), strict=True)
    sgc_model.load_state_dict(snn_model.state_dict(), strict=True)
    assert set(sgc_model.state_dict()) == set(snn_model.state_dict())


def test_snn_training_comparison_reports_matched_budget_and_quality_gate() -> None:
    report = compare_snn_sgc_training(
        _config(mixer="sgc"), token_budget=56, seq_len=8, batch_size=2,
        device="cpu", seed=3, data_seed=4)
    assert report["same_parameter_count"]
    assert report["same_tokens_trained"]
    assert report["arms"]["sgc"]["tokens_trained"] == 56
    assert report["arms"]["snn-sgc"]["tokens_trained"] == 56
    assert isinstance(
        report["comparison"]["meets_80_percent_training_time_reduction"], bool)
    assert isinstance(
        report["comparison"]["snn_improves_synthetic_validation_loss"], bool)
    assert report["limitations"]


def test_snn_training_comparison_rejects_unsupported_gate_ab() -> None:
    with pytest.raises(ValueError, match="gate_ab"):
        compare_snn_sgc_training(
            _config(mixer="sgc", gate_ab=True), token_budget=56,
            seq_len=8, batch_size=2, device="cpu")
