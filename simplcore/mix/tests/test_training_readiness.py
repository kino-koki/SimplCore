import pytest
import simplcore.tensor as torch


def test_150m_and_7b_configs_enable_training_readiness_capabilities():
    from simplcore.mix.config import mix_150m_text, mix_7b_text

    cfg_150m = mix_150m_text()
    cfg_7b = mix_7b_text()

    assert cfg_150m.gradient_checkpointing is True
    assert cfg_7b.gradient_checkpointing is True
    assert cfg_7b.iterative_refinement_enabled is True
    assert cfg_7b.web_search_enabled is True
    assert cfg_7b.max_refinement_loops >= cfg_7b.max_loop_iters


def test_model_uses_gradient_checkpointing_when_enabled(monkeypatch):
    import simplcore.tensor.utils.checkpoint as torch_checkpoint
    from simplcore.mix import SimplCoreMix, mix_tiny_text

    calls = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return function(*args)

    monkeypatch.setattr(torch_checkpoint, "checkpoint", fake_checkpoint)

    cfg = mix_tiny_text()
    cfg.gradient_checkpointing = True
    model = SimplCoreMix(cfg)
    model.train()

    tokens = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    logits = model(tokens, n_loops=1)

    assert logits.shape == (1, 8, cfg.vocab_size)
    assert calls
    assert all(call is False for call in calls)


def test_loss_guard_hard_stops_above_maximum():
    from simplcore.mix.training.loss_guard import LossGuard, LossGuardAction

    guard = LossGuard(target_loss=1.5, max_loss=3.9)

    decision = guard.check(4.0, step=12)

    assert decision.action == LossGuardAction.STOP
    assert "超过最高损失上限" in decision.reason


def test_loss_guard_warns_between_target_and_maximum():
    from simplcore.mix.training.loss_guard import LossGuard, LossGuardAction

    guard = LossGuard(target_loss=1.5, max_loss=3.9)

    decision = guard.check(2.0, step=13)

    assert decision.action == LossGuardAction.WARN
    assert decision.should_continue is True


def test_expand_tensor_to_shape_tiles_source_values():
    from simplcore.mix.stacking.mapper import expand_tensor_to_shape

    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    expanded = expand_tensor_to_shape(source, (3, 5))

    assert expanded.shape == (3, 5)
    assert torch.equal(expanded[:2, :2], source)
    assert expanded[2, 4].item() == source[0, 0].item()


def test_calibration_plan_records_loss_limits_and_sources():
    from simplcore.mix.calibration import build_calibration_plan
    from simplcore.mix.stacking.manifest import SourceCheckpoint, StackManifest

    manifest = StackManifest(
        target_name="mix-7b-text",
        sources=[SourceCheckpoint(path="150m-a.pt", role="prelude", shard_id=0, loss=1.4)],
    )

    plan = build_calibration_plan(manifest, target_loss=1.5, max_loss=3.9)

    assert plan["target_name"] == "mix-7b-text"
    assert plan["loss_guard"]["target_loss"] == 1.5
    assert plan["loss_guard"]["max_loss"] == 3.9
    assert plan["sources"][0]["eligible_for_stacking"] is True


def test_7b_readiness_report_checks_paths(tmp_path):
    from simplcore.mix.readiness import build_7b_readiness_report

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "000000.parquet").write_bytes(b"sample")
    user_docx = tmp_path / "user_docx_corpus.jsonl"
    user_docx.write_text('{"text":"青小团"}\n', encoding="utf-8")

    report = build_7b_readiness_report(
        public_corpus_dir=corpus_dir,
        user_docx_jsonl=user_docx,
    )

    assert report["target"] == "SimplCore Mix 7B stacking calibration"
    assert report["public_parquet_count"] == 1
    assert report["required_loss"]["max_loss"] == 3.9
