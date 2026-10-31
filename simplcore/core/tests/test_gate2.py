# -*- coding: utf-8 -*-
"""GATE-2 的判定与端到端管线测试（不训练、不做大规模实测）。"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.gate2 import evaluate_gate2, summarize_gate2
from simplcore.core.longctx_eval import NeedleCorpus, score_trial
from simplcore.core.model import SimplCoreModel


def _model() -> SimplCoreModel:
    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=4,
                          ffn_ratio=2.0, local_kernel=4)
    return SimplCoreModel(cfg).eval()


# ---------------------------------------------------------------------------
# 1. 纯判定逻辑（不依赖模型）
# ---------------------------------------------------------------------------
def _trials(n: int, ok: int, depth: float = 0.5):
    return [{"correct": i < ok, "depth": depth} for i in range(n)]


def test_summary_pass_at_chance_below_chance() -> None:
    pass_ = summarize_gate2(_trials(8, 8), n_values=8)
    assert pass_["status"] == "pass" and pass_["passed"] is True
    assert pass_["recall"] == 1.0 and pass_["chance"] == 0.125

    at = summarize_gate2(_trials(8, 1), n_values=8)
    assert at["status"] == "at_chance" and at["passed"] is True

    below = summarize_gate2(_trials(8, 0), n_values=8)
    assert below["status"] == "below_chance" and below["passed"] is False


def test_summary_reports_worst_depth_not_just_average() -> None:
    trials = ([{"correct": True, "depth": 0.25} for _ in range(4)]
              + [{"correct": False, "depth": 0.75} for _ in range(4)])
    s = summarize_gate2(trials, n_values=8)
    assert s["recall"] == 0.5
    assert s["worst_depth_recall"] == 0.0
    assert s["per_depth"]["0.75"] == 0.0


def test_summary_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize_gate2([], n_values=8)


# ---------------------------------------------------------------------------
# 2. chunked 评分与 parallel 一致（小长度，eval 模式无 dropout）
# ---------------------------------------------------------------------------
def test_chunked_score_matches_parallel_on_small_length() -> None:
    m = _model()
    corpus = NeedleCorpus(512, seed=2026)
    tr = corpus.make_trial(0, 0.5)
    with torch.no_grad():
        a = score_trial(m, tr, mode="parallel")
        b = score_trial(m, tr, mode="chunked", chunk_size=128)
    assert a["argmax"] == b["argmax"]
    assert a["correct"] == b["correct"]


def test_chunked_mode_requires_forward_chunked() -> None:
    class NoChunk:
        pass
    corpus = NeedleCorpus(128, seed=1)
    with pytest.raises(ValueError, match="forward_chunked"):
        score_trial(NoChunk(), corpus.make_trial(0, 0.5), mode="chunked")


# ---------------------------------------------------------------------------
# 3. 端到端：小长度 + toy 模型，不碰 1M/4M
# ---------------------------------------------------------------------------
def test_gate2_end_to_end_small(tmp_path) -> None:
    m = _model()
    out = tmp_path / "gate2.json"
    rep = evaluate_gate2(m, lengths=(512,), depths=(0.25, 0.75),
                         trials_per_depth=1, chunk_size=128,
                         seed=2026, out_path=out)
    assert rep["protocol"] == "GATE-2"
    assert rep["overall"]["n_trials"] == 2
    assert rep["overall"]["chance"] == pytest.approx(1 / 8)
    assert rep["status"] in ("pass", "at_chance", "below_chance")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "GATE-2" in text and "below_chance" in text


def test_gate2_rejects_empty_lengths() -> None:
    with pytest.raises(ValueError):
        evaluate_gate2(_model(), lengths=())

def test_gate2_cli_loads_checkpoint(tmp_path) -> None:
    """4M 验收用路径：从 checkpoint 加载模型，不启动训练。"""
    from simplcore.core.gate2 import main

    m = _model()
    ckpt = {"config": m.config.to_dict(), "model": m.state_dict()}
    path = tmp_path / "tiny.pt"
    torch.save(ckpt, path)
    out = tmp_path / "gate2.json"
    rc = main(["--checkpoint", str(path), "--lengths", "512",
               "--depths", "0.5", "--trials-per-depth", "1",
               "--chunk-size", "128", "--out", str(out)])
    assert rc in (0, 2)          # 0=passed, 2=at/below chance
    assert out.exists()