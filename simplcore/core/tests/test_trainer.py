# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""分块流式训练驱动（StreamingTrainer）的守护测试。

覆盖四条训练语义：
1. loss 在周期数据上可观测下降（驱动 + 优化器 + BPTT 全链路活着）；
2. 冻结掩码：frozen_names 之外逐位不动、之内更新（Phase 2 语义）；
3. no-regression gate：退化 → 逐位回滚到最优快照（C-rollback 同源）；
4. 课程按 token 预算推进。
"""
import pytest
import torch

from simplcore.core.adapter_model import AdapterModel
from simplcore.core.conformance_runner import TorchReferenceAdapter
from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.trainer import (ContextCurriculum, CurriculumStage,
                                    StreamingTrainer, main)

CFG = SimplCoreConfig(vocab_size=32, dim=32, num_layers=2, num_bases=8,
                      local_kernel=4, ffn_ratio=2.0, decay_max=256.0,
                      tie_embeddings=False)


def _make_trainer(lr=3e-3, frozen=None, stages=None):
    torch.manual_seed(7)
    native = SimplCoreModel(CFG)
    ad = AdapterModel(CFG, TorchReferenceAdapter())
    ad.load_native(native.state_dict())
    if stages is None:
        stages = [CurriculumStage(16, 16 * 4 * 50), CurriculumStage(32, 32 * 4 * 50)]
    tr = StreamingTrainer(ad, ad.adapter, ContextCurriculum(stages),
                          lr=lr, frozen_names=frozen)
    return tr


def _periodic_batch(seq_len, B=4):
    base = torch.arange(seq_len + 1) % CFG.vocab_size
    return (base[:-1].unsqueeze(0).repeat(B, 1),
            base[1:].unsqueeze(0).repeat(B, 1))


# ---------------------------------------------------------------------------
# 1. loss 下降
# ---------------------------------------------------------------------------
def test_loss_decreases_on_periodic_data():
    tr = _make_trainer()
    first = last = None
    for _ in range(24):
        si, st = tr.stage()
        x, y = _periodic_batch(st.seq_len)
        r = tr.train_step(x, y)
        if first is None:
            first = r["loss"]
        last = r["loss"]
    assert last < first - 0.1, (first, last)
    assert tr.ledger.steps == 24 and tr.ledger.tokens_seen > 0


# ---------------------------------------------------------------------------
# 2. 冻结掩码（Phase 2 语义）
# ---------------------------------------------------------------------------
def test_frozen_base_bitwise_untouched():
    frozen = {n for n in
              AdapterModel(CFG, TorchReferenceAdapter()).param_names()
              if n != "head.weight"}
    tr = _make_trainer(frozen=frozen)
    before = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    for _ in range(3):
        si, st = tr.stage()
        x, y = _periodic_batch(st.seq_len)
        tr.train_step(x, y)
    for n, p in tr.model.named_parameters():
        if n in frozen:
            assert torch.equal(before[n], p.detach()), f"冻结参数被改动: {n}"
        else:
            assert not torch.equal(before[n], p.detach()), f"可训参数没动: {n}"


# ---------------------------------------------------------------------------
# 3. no-regression gate
# ---------------------------------------------------------------------------
def test_rollback_restores_best_bitwise():
    tr = _make_trainer(lr=3e-3)
    x, y = _periodic_batch(16)
    gate1 = tr.maybe_commit(x, y, chunk=16)
    assert gate1["committed"] is True and gate1["rollback"] is False
    best = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    # 破坏性扰动 → 评估必然退化超容差（未训练模型对小幅扰动不敏感，须大扰动）
    with torch.no_grad():
        for p in tr.model.parameters():
            p.add_(5.0)
    gate2 = tr.maybe_commit(x, y, chunk=16)
    assert gate2["rollback"] is True
    assert tr.ledger.rollbacks == 1
    for n, p in tr.model.named_parameters():
        assert torch.equal(best[n], p.detach()), f"回滚不逐位: {n}"


def test_commit_keeps_improvement():
    tr = _make_trainer()
    x, y = _periodic_batch(16)
    g1 = tr.maybe_commit(x, y, chunk=16)
    # 若干正常步（周期数据上 loss 应改善）
    for _ in range(5):
        tr.train_step(x, y)
    g2 = tr.maybe_commit(x, y, chunk=16)
    assert g2["rollback"] is False
    assert g2["best"] <= g1["eval_loss"] + 1e-9


# ---------------------------------------------------------------------------
# 4. 课程推进
# ---------------------------------------------------------------------------
def test_curriculum_advances_on_token_budget():
    tr = _make_trainer(stages=[CurriculumStage(8, 8 * 4 * 2),
                               CurriculumStage(16, 16 * 4 * 2)])
    assert tr.stage()[0] == 0
    for _ in range(2):
        x, y = _periodic_batch(8)
        tr.train_step(x, y)
    assert tr.stage()[0] == 1                     # token 预算耗尽 → 升档
    x, y = _periodic_batch(16)
    r = tr.train_step(x, y)
    assert r["stage"] == 1
    with pytest.raises(ValueError):
        tr.train_step(*_periodic_batch(8))        # 不按当前档长度 → 拒绝


def test_curriculum_total_tokens():
    c = ContextCurriculum([CurriculumStage(8, 64), CurriculumStage(16, 128)])
    assert c.total_tokens == 192
    assert c.stage_for(0)[0] == 0 and c.stage_for(63)[0] == 0
    assert c.stage_for(64)[0] == 1 and c.stage_for(10_000)[0] == 1
    assert c.exhausted(192) and not c.exhausted(191)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_smoke_loss_decreases(capsys):
    assert main(["--steps", "20"]) == 0
    out = capsys.readouterr().out
    assert "下降 ✓" in out and "gate:" in out
