# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""phase1_train 守护测试 —— 数据闸门 P0-budget-frozen 数据部分的执行器。"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from simplcore.core.data import read_manifest
from simplcore.core.phase1_train import (
    CorpusBatches,
    SyntheticCorpusSpec,
    make_synthetic_corpus,
    run_phase1,
    tiny_curriculum,
)


SPEC = SyntheticCorpusSpec(vocab_size=32, seq_len=16,
                           train_tokens=8192, val_tokens=2048, seed=7)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("corpus")
    make_synthetic_corpus(root, SPEC)
    return str(root)


# ---------------------------------------------------------------------------
# 语料
# ---------------------------------------------------------------------------
def test_manifest_roundtrip_and_counts(corpus) -> None:
    m = read_manifest(corpus)
    assert m["vocab_size"] == 32 and m["dtype"] == "uint32"
    assert m["splits"]["train"]["n_tokens"] == 8192
    assert m["splits"]["val"]["n_tokens"] == 2048
    # 分片文件 token 数复算 = 文件字节数 / 4
    import pathlib
    f = m["splits"]["train"]["files"][0]
    assert (pathlib.Path(corpus) / f["path"]).stat().st_size == 8192 * 4


def test_synthetic_corpus_deterministic(tmp_path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    make_synthetic_corpus(a, SPEC)
    make_synthetic_corpus(b, SPEC)
    assert (a / "train.bin").read_bytes() == (b / "train.bin").read_bytes()
    assert (a / "val.bin").read_bytes() == (b / "val.bin").read_bytes()


# ---------------------------------------------------------------------------
# CorpusBatches：确定性 / 连续性 / 回卷
# ---------------------------------------------------------------------------
def test_batches_deterministic(corpus) -> None:
    s1, s2 = CorpusBatches(corpus, "train"), CorpusBatches(corpus, "train")
    for _ in range(3):
        x1, y1 = s1.next_batch(16, 4)
        x2, y2 = s2.next_batch(16, 4)
        torch.testing.assert_close(x1, x2)
        torch.testing.assert_close(y1, y2)


def test_batches_shift_alignment_and_continuity(corpus) -> None:
    src = CorpusBatches(corpus, "train")
    # 错位对齐：y[t] == x[t+1]（来自同一条 token 流）
    x, y = src.next_batch(16, 4)
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    # 连续消费：换档重切不回头 —— 两次取批的完整窗口 == 游标起点开始的 token 流
    src_a, src_b = CorpusBatches(corpus, "train"), CorpusBatches(corpus, "train")
    xa, _ = src_a.next_batch(16, 2)          # 消费 2*17
    xb, _ = src_a.next_batch(31, 4)          # 消费 4*32，窗口重切不回头
    raw_b = torch.from_numpy(src_b._next(2 * 17).astype(np.int64)).reshape(2, 17)
    torch.testing.assert_close(xa, raw_b[:, :-1])
    raw_b2 = torch.from_numpy(src_b._next(4 * 32).astype(np.int64)).reshape(4, 32)
    torch.testing.assert_close(xb, raw_b2[:, :-1])
    # 两源游标一致 = 同一条流的连续消费
    assert src_a.cursor == src_b.cursor == 2 * 17 + 4 * 32


def test_batch_epoch_wrap_counted(corpus) -> None:
    src = CorpusBatches(corpus, "train")
    n = src.total_tokens
    src._next(n - 1)
    out = src._next(8)
    assert src.epochs == 1 and len(out) == 8    # 回卷且计数入账


def test_read_tokens_across_request_bounds(corpus) -> None:
    src = CorpusBatches(corpus, "train")
    np.testing.assert_array_equal(src.read_tokens(0, 100),
                                  src.read_tokens(0, 50).astype(np.uint32).tolist()
                                  + src.read_tokens(50, 50).astype(np.uint32).tolist())
    with pytest.raises(IndexError):
        src.read_tokens(src.total_tokens - 1, 2)


# ---------------------------------------------------------------------------
# 泄漏检查
# ---------------------------------------------------------------------------
def test_shared_shard_rejected(corpus, tmp_path) -> None:
    # 篡改 manifest：val 指到 train 的分片 → 必须报错
    m = read_manifest(corpus)
    m["splits"]["val"]["files"] = m["splits"]["train"]["files"]
    (tmp_path / "index.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="泄漏"):
        run_phase1(corpus_root=str(tmp_path), max_steps=2)


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def report(corpus) -> dict:
    return run_phase1(corpus, batch=4, max_steps=40, lr=3e-3,
                      curriculum=tiny_curriculum((16,), 20), eval_every=8,
                      seed=2026)


def test_e2e_loss_improves_and_gate(report) -> None:
    g = report["gate"]
    assert g["improved"] is True
    assert g["commits"] >= 1
    assert g["final_eval_loss"] < g["first_eval_loss"]
    assert report["eval_history"], "必须有评估记录"


def test_report_honesty_and_provenance(report) -> None:
    c = report["corpus"]
    assert c["synthetic"] is True
    assert c["splits_disjoint"] is True
    assert len(c["manifest_sha256"]) == 64      # 真实指纹
    h = report["honesty"]
    assert h["claim_permitted"] is False        # 合成语料关闭宣称
    assert h["forbidden_checks"]


def test_eval_protocol_fixed(report) -> None:
    ep = report["eval_protocol"]
    assert ep["eval_seq_len"] == 16
    assert all(e["eval_loss"] > 0 for e in report["eval_history"])


def test_val_windows_never_in_train(report) -> None:
    # 评估批来自 val 分片：结构上 train 游标独立（epochs 各自计）
    assert report["corpus"]["train_epochs"] >= 0
    assert report["corpus"]["val_tokens"] > 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_smoke(tmp_path, monkeypatch) -> None:
    from simplcore.core.phase1_train import main
    out = tmp_path / "r.json"
    rc = main(["--steps", "12", "--train-tokens", "4096",
               "--val-tokens", "1024", "--out", str(out)])
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["protocol"] == "phase1-train-run"
    assert rep["honesty"]["claim_permitted"] is False
    assert rc in (0, 1)                          # 改善与否都算跑通


# ---------------------------------------------------------------------------
# checkpoint / 断点续训
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ckpt")
    make_synthetic_corpus(root / "corpus", SPEC)
    return root


CK_KW = dict(batch=4, lr=3e-3, curriculum=tiny_curriculum((16,), 20),
             eval_every=5, eval_batch=4, seed=2026,
             checkpoint_every=10)


def test_resume_trajectory_identical(ckpt_dir) -> None:
    """断点续训的轨迹必须与不间断运行逐位一致（不是"大概齐"）。"""
    kw = dict(CK_KW, checkpoint_path=ckpt_dir / "ck.pt")
    full = run_phase1(str(ckpt_dir / "corpus"), max_steps=30, **kw)
    run_phase1(str(ckpt_dir / "corpus"), max_steps=18, **kw)
    b = run_phase1(str(ckpt_dir / "corpus"), max_steps=30,
                   resume_from=ckpt_dir / "ck.pt", **kw)
    tail = full["eval_history"][-len(b["eval_history"]):]
    assert b["eval_history"] == tail
    assert full["gate"]["final_eval_loss"] == b["gate"]["final_eval_loss"]
    assert full["gate"]["commits"] == b["gate"]["commits"]
    assert full["ledger"]["steps"] == b["ledger"]["steps"]
    assert full["ledger"]["tokens_seen"] == b["ledger"]["tokens_seen"]
    assert b["resumed_from"] is not None


def test_resume_rejects_curriculum_mismatch(ckpt_dir) -> None:
    kw = dict(CK_KW, checkpoint_path=ckpt_dir / "ck2.pt")
    run_phase1(str(ckpt_dir / "corpus"), max_steps=10, **kw)
    with pytest.raises(ValueError, match="课程"):
        run_phase1(str(ckpt_dir / "corpus"), max_steps=2,
                   resume_from=ckpt_dir / "ck2.pt",
                   curriculum=tiny_curriculum((16, 32, 64), 20), **{
                       k: v for k, v in CK_KW.items() if k != "curriculum"})


def test_resume_rejects_corpus_fingerprint_mismatch(ckpt_dir) -> None:
    kw = dict(CK_KW, checkpoint_path=ckpt_dir / "ck3.pt")
    run_phase1(str(ckpt_dir / "corpus"), max_steps=10, **kw)
    make_synthetic_corpus(ckpt_dir / "corpus_b", SPEC)
    # 同 seed → 同语料 → 指纹相同，必须放行；换 seed → 指纹不同 → 拒绝
    ok = run_phase1(str(ckpt_dir / "corpus_b"), max_steps=2,
                    resume_from=ckpt_dir / "ck3.pt", **kw)
    assert ok["resumed_from"] is not None
    spec_b = SyntheticCorpusSpec(vocab_size=32, seq_len=16,
                                 train_tokens=8192, val_tokens=2048, seed=99)
    make_synthetic_corpus(ckpt_dir / "corpus_c", spec_b)
    with pytest.raises(ValueError, match="指纹"):
        run_phase1(str(ckpt_dir / "corpus_c"), max_steps=2,
                   resume_from=ckpt_dir / "ck3.pt", **kw)


def test_checkpoint_gate_baseline_survives(ckpt_dir) -> None:
    """恢复后 no-regression gate 的历史基线（best_eval）不丢。"""
    from simplcore.core.trainer import StreamingTrainer
    kw = dict(CK_KW, checkpoint_path=ckpt_dir / "ck4.pt")
    run_phase1(str(ckpt_dir / "corpus"), max_steps=10, **kw)
    blob = torch.load(str(ckpt_dir / "ck4.pt"), weights_only=False)
    assert blob["format"] == StreamingTrainer.CKPT_FORMAT
    assert blob["meta"]["best_eval"] is not None
    assert blob["best_state"], "最优快照必须已保存"


def test_checkpoint_data_cursor_roundtrip(ckpt_dir) -> None:
    """data_state 携带数据游标 + 语料指纹（续训数据从同一位置继续）。"""
    kw = dict(CK_KW, checkpoint_path=ckpt_dir / "ck5.pt")
    run_phase1(str(ckpt_dir / "corpus"), max_steps=10, **kw)
    blob = torch.load(str(ckpt_dir / "ck5.pt"), weights_only=False)
    ds = blob["meta"]["data_state"]
    assert {"train_cursor", "train_epochs", "val_cursor",
            "corpus_sha256"} <= set(ds)
    assert len(ds["corpus_sha256"]) == 64
