# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Dreaming 循环与 CMT 压缩记忆库的守护测试。"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig, analytic_params
from simplcore.core.dream import (
    DreamConfig,
    DreamingLoop,
    DreamStage,
    ExperienceBuffer,
    Insight,
    TrainExample,
    Trajectory,
    default_reflector,
    default_synthesizer,
)
from simplcore.core.memory import (
    CompressedMemoryBank,
    MemoryAwareObjective,
    MemoryCompressor,
    self_match_scores,
    top_aggregate,
)
from simplcore.core.model import SimplCoreModel


DIM = 16


def _cfg(**kw) -> SimplCoreConfig:
    base = dict(vocab_size=64, dim=DIM, num_layers=2, num_heads=2,
                num_bases=4, local_kernel=2)
    base.update(kw)
    return SimplCoreConfig(**base)


# ===========================================================================
# CMT: self-matching / top-aggregation
# ===========================================================================
def test_self_match_scores_is_cosine() -> None:
    q = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    s = self_match_scores(q, keys)
    assert s.shape == (1, 3)
    assert pytest.approx(1.0, abs=1e-6) == float(s[0, 0])
    assert pytest.approx(0.0, abs=1e-6) == float(s[0, 1])
    assert pytest.approx(1.0, abs=1e-6) == float(s[0, 2])


def test_top_aggregate_picks_top_k_and_weights() -> None:
    q = torch.tensor([[1.0, 0.0]])
    keys = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    agg, scores = top_aggregate(q, keys, values, top_k=2, temperature=0.01)
    assert agg.shape == (1, 2)
    assert scores.shape == (1, 3)
    # 前两条被选中，第三条（相似度 0）不参与
    assert not torch.allclose(agg, torch.tensor([[5.0, 5.0]]))


def test_memory_aware_objective_uses_delta() -> None:
    obj = MemoryAwareObjective()
    with_mem = torch.randn(2, 3, 10)
    without = torch.randn(2, 3, 10)
    targets = torch.randint(0, 10, (2, 3))
    loss = obj(with_mem, without, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)


# ===========================================================================
# CMT: 记忆库
# ===========================================================================
def test_memory_bank_add_and_retrieve() -> None:
    bank = CompressedMemoryBank(dim=DIM, capacity=8)
    keys = torch.randn(5, DIM)
    values = torch.randn(5, DIM)
    bank.add(keys, values)
    assert len(bank) == 5
    picked, scores = bank.retrieve(torch.randn(2, DIM), top_k=3)
    assert picked.shape == (2, 3, DIM)
    assert scores.shape == (2, 3)
    # 检索出的第一条应当是真正最相似的
    q = keys[3:4]
    p, sc = bank.retrieve(q, top_k=1)
    assert pytest.approx(1.0, abs=1e-5) == float(sc[0, 0])


def test_memory_bank_evicts_least_retrieved() -> None:
    bank = CompressedMemoryBank(dim=DIM, capacity=3)
    for i in range(3):
        bank.add(torch.full((1, DIM), float(i)), torch.full((1, DIM), float(i)))
    # 检索第 0 条，使它成为最热的
    bank.retrieve(torch.full((1, DIM), 0.0), top_k=1)
    bank.add(torch.full((1, DIM), 9.0), torch.full((1, DIM), 9.0))
    assert len(bank) == 3
    kept = bank.keys[:, 0].tolist()
    assert 9.0 in kept            # 新条目留下
    assert 0.0 in kept            # 被检索过的热条目留下
    assert 1.0 not in kept        # 从未检索过的冷条目被驱逐


def test_memory_bank_aggregate_shape() -> None:
    bank = CompressedMemoryBank(dim=DIM, capacity=8)
    bank.add(torch.randn(4, DIM), torch.randn(4, DIM))
    mem, scores = bank.aggregate(torch.randn(3, DIM), top_k=2)
    assert mem.shape == (3, DIM)
    assert scores.shape == (3, 4)


def test_memory_bank_empty_is_safe() -> None:
    bank = CompressedMemoryBank(dim=DIM)
    mem, scores = bank.aggregate(torch.randn(2, DIM))
    assert mem.shape == (2, DIM)
    assert scores.shape[1] == 0


def test_memory_compressor_shape() -> None:
    model = SimplCoreModel(_cfg()).eval()
    comp = MemoryCompressor(DIM, k=4)
    with torch.no_grad():
        out = comp(model, torch.randint(0, 64, (2, 10)))
    assert out.shape == (2, 4, DIM)


# ===========================================================================
# Dreaming 循环
# ===========================================================================
def _traj(prompt="q", grounded=False, confidence=0.2, score=0.0, corrected=None):
    meta = {"corrected_response": corrected} if corrected else {}
    return Trajectory(prompt=prompt, response="a", confidence=confidence,
                      grounded=grounded, score=score, meta=meta)


def test_buffer_priority_prefers_valuable() -> None:
    buf = ExperienceBuffer(capacity=16)
    buf.add(_traj("low", score=0.1))
    buf.add(_traj("high", score=5.0))
    buf.add(_traj("mid", score=1.0))
    picked = [t.prompt for t in buf.sample(3, strategy="priority")]
    assert "high" in picked


def test_buffer_capacity_enforced() -> None:
    buf = ExperienceBuffer(capacity=4)
    for i in range(20):
        buf.add(_traj(f"p{i}", score=float(i)))
    assert len(buf) == 4


def test_default_reflector_flags_ungrounded() -> None:
    ins = default_reflector(None, [_traj("a", grounded=False),
                                   _traj("b", grounded=True, confidence=0.9)])
    assert len(ins) == 1
    assert ins[0].source.prompt == "a"


def test_default_synthesizer_needs_corrected_response() -> None:
    ins = default_reflector(None, [_traj("a", grounded=False)])
    assert default_synthesizer(None, ins) == []
    ins2 = default_reflector(None, [_traj("a", grounded=False, corrected="fixed")])
    ex = default_synthesizer(None, ins2)
    assert len(ex) == 1 and ex[0].target == "fixed"


def test_dreaming_empty_buffer_is_idle() -> None:
    loop = DreamingLoop(model=None, buffer=ExperienceBuffer())
    rep = loop.step()
    assert rep.stage is DreamStage.IDLE
    assert rep.harvested == 0


def test_dreaming_full_cycle_commits() -> None:
    buf = ExperienceBuffer()
    buf.add(_traj("a", grounded=False, corrected="fixed"))
    loop = DreamingLoop(
        model=None, buffer=buf,
        config=DreamConfig(mode="memory", require_regression_check=False),
    )
    rep = loop.step()
    assert rep.stage is DreamStage.COMMIT
    assert rep.harvested == 1
    assert rep.insights >= 1
    assert rep.synthesized == 1
    assert rep.consolidated is True


def test_dreaming_writes_to_memory_bank() -> None:
    buf = ExperienceBuffer()
    buf.add(_traj("a", grounded=False, corrected="fixed"))
    bank = CompressedMemoryBank(dim=DIM)
    loop = DreamingLoop(
        model=None, buffer=buf,
        config=DreamConfig(mode="memory", require_regression_check=False),
        memory_bank=bank,
    )
    loop.step()
    assert len(bank) >= 1


def test_dreaming_mode_none_does_not_consolidate() -> None:
    buf = ExperienceBuffer()
    buf.add(_traj("a", grounded=False, corrected="fixed"))
    loop = DreamingLoop(
        model=None, buffer=buf,
        config=DreamConfig(mode="none", require_regression_check=False),
    )
    rep = loop.step()
    assert rep.consolidated is False
    assert rep.stage is DreamStage.IDLE


def test_dreaming_budget_caps_insights_and_examples() -> None:
    buf = ExperienceBuffer()
    for i in range(10):
        buf.add(_traj(f"p{i}", grounded=False, corrected="fixed"))
    loop = DreamingLoop(
        model=None, buffer=buf,
        config=DreamConfig(mode="memory", max_insights=2, max_examples=1,
                            require_regression_check=False),
    )
    rep = loop.step()
    assert rep.insights <= 2
    assert rep.synthesized <= 1


def test_dreaming_regression_triggers_rollback() -> None:
    """no-regression gate：分数退化必须回滚。"""
    class Model:
        def __init__(self):
            self.snapshots = 0
            self.restores = 0

        def snapshot(self):
            self.snapshots += 1
            return {"w": 1}

        def restore(self, snap):
            self.restores += 1

    class BadConsolidator:
        def __call__(self, model, examples):
            return {"changed": True}

    buf = ExperienceBuffer()
    buf.add(_traj("a", grounded=False, corrected="fixed"))
    model = Model()
    scores = iter([1.0, 0.5])          # 基线 1.0，巩固后 0.5 → 退化
    loop = DreamingLoop(
        model=model, buffer=buf,
        config=DreamConfig(mode="weights", regression_tolerance=0.0),
        consolidator=BadConsolidator(),
        verifier=lambda m: next(scores),
    )
    rep = loop.step()
    assert rep.stage is DreamStage.ROLLBACK
    assert rep.rolled_back is True
    assert rep.score_delta < 0
    assert model.restores == 1


def test_dreaming_no_regression_commits() -> None:
    class Model:
        def snapshot(self):
            return {"w": 1}

        def restore(self, snap):
            raise AssertionError("不应回滚")

    buf = ExperienceBuffer()
    buf.add(_traj("a", grounded=False, corrected="fixed"))
    scores = iter([1.0, 1.2])
    loop = DreamingLoop(
        model=Model(), buffer=buf,
        config=DreamConfig(mode="weights", regression_tolerance=0.0),
        consolidator=lambda m, ex: None,
        verifier=lambda m: next(scores),
    )
    rep = loop.step()
    assert rep.stage is DreamStage.COMMIT
    assert rep.score_delta > 0


def test_dreaming_reflector_exception_is_contained() -> None:
    def boom(model, trajs):
        raise RuntimeError("critic down")

    buf = ExperienceBuffer()
    buf.add(_traj("a"))
    loop = DreamingLoop(model=None, buffer=buf, reflector=boom)
    rep = loop.step()
    assert rep.stage is DreamStage.FAILED
    assert any("critic down" in n for n in rep.notes)


def test_dreaming_run_stops_on_failure() -> None:
    def boom(model, trajs):
        raise RuntimeError("x")

    buf = ExperienceBuffer()
    buf.add(_traj("a"))
    loop = DreamingLoop(model=None, buffer=buf, reflector=boom)
    reps = loop.run(max_rounds=5)
    assert len(reps) == 1


# ===========================================================================
# 记忆注入（SGC 架构层）
# ===========================================================================
def test_model_memory_injection_is_noop_by_default() -> None:
    """零初始化的 memory_in → 注入记忆不改变基座输出。"""
    torch.manual_seed(0)
    model = SimplCoreModel(_cfg(use_memory=True)).eval()
    ids = torch.randint(0, 64, (2, 8))
    mem = torch.randn(2, DIM)
    with torch.no_grad():
        a = model(ids)
        b = model(ids, memory=mem)
    assert torch.allclose(a, b, atol=1e-6)


def test_model_memory_injection_changes_output_when_trained() -> None:
    torch.manual_seed(0)
    model = SimplCoreModel(_cfg(use_memory=True)).eval()
    with torch.no_grad():
        model.memory_in.weight.copy_(torch.eye(DIM))
    ids = torch.randint(0, 64, (2, 8))
    mem = torch.randn(2, DIM)
    with torch.no_grad():
        a = model(ids)
        b = model(ids, memory=mem)
    assert not torch.allclose(a, b, atol=1e-5)


def test_model_without_memory_flag_rejects_memory() -> None:
    """没开 ``use_memory`` 却收到记忆：必须报错，**不能**静默忽略。

    旧实现是 ``if memory is None or self.memory_in is None: return None``。
    后果是带记忆的样本被当成"没记忆"处理 —— 训练照跑、loss 照降，
    但模型其实在瞎猜，而 loss 曲线、形状断言、参数统计全都看不出异常。
    这是最难排查的一类失败，所以改成当场报错。
    """
    torch.manual_seed(0)
    model = SimplCoreModel(_cfg()).eval()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a = model(ids)                       # 不带记忆：正常
    assert a.shape == (2, 8, _cfg().vocab_size)
    with pytest.raises(ValueError, match="use_memory"):
        with torch.no_grad():
            model(ids, memory=torch.randn(2, DIM))


def test_analytic_params_accounts_for_memory_projection() -> None:
    cfg = _cfg(use_memory=True)
    model = SimplCoreModel(cfg)
    assert model.num_params() == analytic_params(cfg)


def test_stream_state_carries_memory() -> None:
    torch.manual_seed(0)
    model = SimplCoreModel(_cfg(use_memory=True)).eval()
    mem = torch.randn(1, DIM)
    state = model.init_stream(1, memory=mem)
    assert state.memory is not None
    assert state.memory.shape == (1, DIM)
    with torch.no_grad():
        tok = torch.randint(0, 64, (1,))
        logits, state = model.stream_step(tok, state)
    assert logits.shape[0] == 1
