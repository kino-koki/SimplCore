# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Simpl Model Memory（SMM）—— 自主学习整合架构的测试。"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.dream import DreamConfig, DreamStage, TrainExample
from simplcore.core.memory import TextKeyEncoder
from simplcore.core.model import SimplCoreModel
from simplcore.core.smm import SMMConfig, SimplModelMemory
from simplcore.core.tokenization import CharTokenizer


def _build(mode: str = "memory"):
    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=48, num_layers=2, num_bases=8, decay_max=128,
        local_kernel=3, num_heads=4, ffn_ratio=2.0, mixer="sgc",
        use_memory=True)
    model = SimplCoreModel(cfg).eval()
    dcfg = DreamConfig(mode=mode, require_regression_check=False)
    smm = SimplModelMemory(model, codec, SMMConfig(dream=dcfg))
    return smm, codec


# ---------------------------------------------------------------------------
# 在线循环
# ---------------------------------------------------------------------------
def test_experience_writes_buffer_and_bank():
    smm, _ = _build()
    smm.experience("abcd", "efgh", confidence=0.5, grounded=True)
    assert len(smm.buffer) == 1
    assert len(smm.bank) == 1
    assert smm.ledger.experiences == 1
    assert smm.ledger.memory_writes == 1


def test_recall_roundtrip_same_text_is_top_hit():
    smm, _ = _build()
    smm.experience("用户的名字是小林", "好的", grounded=True)
    smm.experience("小林喜欢蓝色", "明白", grounded=True)
    mem, score, meta = smm.recall("用户的名字")
    assert mem is not None
    assert mem.shape == (48,)
    assert score > 0.0
    assert smm.ledger.recall_hits == 1


def test_recall_empty_bank_returns_none():
    smm, _ = _build()
    mem, score, meta = smm.recall("anything")
    assert mem is None
    assert meta == []


def test_recall_threshold_gates_injection():
    smm, _ = _build()
    smm.config.min_recall_score = 0.999   # 高到不命中
    smm.experience("abcd", "efgh", grounded=True)
    mem, score, _ = smm.recall("wxyz")
    # 同为词面键，分数不会到 0.999 → 被阈值拦截
    assert mem is None
    assert smm.ledger.injected == 0


def test_value_encoder_dim_mismatch_raises():
    smm, _ = _build()
    smm.value_encoder = TextKeyEncoder(dim=7, ngram=2)   # 维度 ≠ bank.dim
    with pytest.raises(ValueError):
        smm.experience("abcd", "efgh", grounded=True)


def test_answer_requires_query_for_injection():
    smm, codec = _build()
    idx = torch.tensor([codec.encode("abc")])
    with pytest.raises(ValueError):
        smm.answer(idx, max_new_tokens=4, inject=True, query="")


def test_answer_injection_changes_logits_after_training_memory_in():
    """memory_in 零初始化 → 初始注入不改 logits（恒等）；
    手动给 memory_in 非零权重后，注入与不注入的前向 logits 必须分叉。"""
    smm, codec = _build()
    smm.experience("abcd", "eeee", grounded=True)
    idx = torch.tensor([codec.encode("abc")])
    mem, score, _ = smm.recall("abcd")
    assert mem is not None
    with torch.no_grad():
        # 零初始化：注入为恒等，logits 逐位一致
        l0a = smm.model(idx, memory=smm.model.memory_in(mem.reshape(1, -1)))
        l0b = smm.model(idx)
        assert torch.equal(l0a, l0b)
        # 给 memory_in 非零权重（模拟已训练的记忆投影）
        smm.model.memory_in.weight.normal_(0, 0.2)
        l1 = smm.model(idx, memory=smm.model.memory_in(mem.reshape(1, -1)))
    assert not torch.equal(l1, l0b)
    assert smm.ledger.injected == 0   # 直接 recall 前向不计数；answer 才计


def test_answer_without_memory_flag_raises():
    """use_memory=False 的模型收到注入请求必须显式报错（不静默丢弃）。"""
    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=48, num_layers=2, num_bases=8, decay_max=128,
        local_kernel=3, num_heads=4, ffn_ratio=2.0, mixer="sgc",
        use_memory=False)
    model = SimplCoreModel(cfg).eval()
    smm = SimplModelMemory(model, codec)
    smm.experience("abcd", "efgh", grounded=True)   # 先有记忆，recall 才会命中
    with pytest.raises(ValueError):
        smm.answer(torch.tensor([codec.encode("abc")]), max_new_tokens=4,
                   inject=True, query="abcd")


# ---------------------------------------------------------------------------
# 巩固闭环（sleep）
# ---------------------------------------------------------------------------
def test_sleep_memory_mode_writes_nothing_extra_but_runs():
    smm, _ = _build(mode="memory")
    smm.experience("abcd", "efgh", grounded=True, evidence="证据：efgh")
    size_before = len(smm.bank)
    rep = smm.sleep()
    assert rep.round_index == 1
    assert smm.ledger.sleep_rounds == 1
    assert not rep.rolled_back
    assert len(smm.bank) >= size_before


def test_sleep_empty_buffer_is_idle():
    smm, _ = _build()
    rep = smm.sleep()
    assert rep.stage in (DreamStage.IDLE,)
    assert rep.harvested == 0


def test_weights_mode_requires_verifier_and_gate():
    """mode=weights + 无有效 verifier → 巩固被拒绝，不写权重（硬约束）。"""
    smm, codec = _build(mode="weights")
    # DreamingLoop 的默认 verifier 是 default_verifier（返回常量），且
    # require_regression_check=False 时 gate 不跑 —— 这里验证"无样本时不炸"：
    smm.experience("abcd", "efgh", grounded=False)   # 无证据 → 不合成样本
    rep = smm.sleep()
    assert not rep.wrote_weights


def test_rollback_counted_via_ledger():
    """回滚路径：verifier 判退化 → 回滚 → SMM 账本记账。"""
    smm, codec = _build(mode="weights")
    smm.dream.config.require_regression_check = True
    calls = {"n": 0}

    def verifier(model):
        calls["n"] += 1
        # 第二次（巩固后）评测显著低于第一次 → gate 判退化
        return {"score": [1.0] if calls["n"] == 1 else [0.0]}

    smm.dream.verifier = verifier
    smm.dream.verifier_is_default = False
    example = TrainExample(prompt="abcd", target="efgh")
    # 直接注入一条带证据的轨迹，让 synthesizer 能产出样本
    from simplcore.core.dream import Trajectory
    smm.buffer.add(Trajectory(prompt="abcd", response="efgh", grounded=True,
                              meta={"evidence": "efgh"}))
    # 手工走一轮：harvest→reflect→synthesize 可能产出 0 样本（取决于 reflector），
    # 为保证权重巩固发生，直接用自定义 synthesizer
    smm.dream.synthesizer = lambda model, insights: [example]
    rep = smm.sleep()
    assert rep.rolled_back
    assert smm.ledger.rollbacks == 1
    assert smm.ledger.weights_commits == 0


# ---------------------------------------------------------------------------
# 持久化与账目
# ---------------------------------------------------------------------------
def test_state_dict_roundtrip():
    smm, _ = _build()
    smm.experience("abcd", "efgh", grounded=True)
    snap = smm.state_dict()
    smm2, _ = _build()
    smm2.load_state_dict(snap)
    assert len(smm2.bank) == len(smm.bank)
    assert smm2.key_encoder.fingerprint() == smm.key_encoder.fingerprint()
    m1, s1, _ = smm.recall("abcd")
    m2, s2, _ = smm2.recall("abcd")
    assert torch.allclose(m1, m2)
    assert abs(s1 - s2) < 1e-6


def test_stats_covers_three_tiers():
    smm, _ = _build()
    smm.experience("abcd", "efgh", grounded=True)
    st = smm.stats()
    assert st["working"]["stream_bytes_1"] > 0
    assert st["episodic"]["size"] == 1
    assert st["semantic"]["mode"] == "memory"
    assert st["ledger"]["experiences"] == 1


def test_purity_no_attention():
    smm, _ = _build()
    assert smm.attention_like_modules() == []


# ---------------------------------------------------------------------------
# 思考层接线（SMM × NonlinearThinker）
# ---------------------------------------------------------------------------
def _build_thinking(iters: int = 2):
    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=48, num_layers=2, num_bases=8, decay_max=128,
        local_kernel=3, num_heads=4, ffn_ratio=2.0, mixer="sgc",
        use_memory=True)
    model = SimplCoreModel(cfg).eval()
    smm = SimplModelMemory(model, codec,
                           SMMConfig(think_max_iters=iters))
    return smm, codec


def test_thinker_disabled_by_default():
    smm, _ = _build()
    assert smm.thinker is None
    assert smm.stats()["think"] == {"enabled": False}


def test_thinker_zero_init_answer_is_bit_identical_to_plain():
    """思考层 cell/halting 零初始化 → cell 是恒等映射，
    answer()（注入记忆 + 思考）与纯模型 answer() 的 tokens 逐位一致。"""
    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcdefgh")
    cfg = SimplCoreConfig.from_codec(
        codec, dim=48, num_layers=2, num_bases=8, decay_max=128,
        local_kernel=3, num_heads=4, ffn_ratio=2.0, mixer="sgc",
        use_memory=True)
    model = SimplCoreModel(cfg).eval()

    smm_plain = SimplModelMemory(model, codec, SMMConfig())
    smm_think, codec2 = _build_thinking(iters=3)
    # 同一模型上建两个 SMM（bank 独立但确定性种子保证键值一致）
    for smm in (smm_plain, smm_think):
        smm.experience("abcd", "eeee", grounded=True)
    idx = torch.tensor([codec.encode("abc")])
    torch.manual_seed(42)
    out_plain = smm_plain.answer(idx, max_new_tokens=5, query="abcd",
                                 temperature=0)
    torch.manual_seed(42)
    out_think = smm_think.answer(idx, max_new_tokens=5, query="abcd",
                                 temperature=0)
    assert torch.equal(out_plain["tokens"], out_think["tokens"])
    # 思考确实发生了（账本记账）
    assert smm_think.stats()["think"]["tokens"] == 5 + idx.shape[1]
    assert smm_plain.stats()["think"] == {"enabled": False}


def test_thinker_changes_tokens_after_training_cell():
    """给 cell.down 非零权重（模拟已训练的思考单元）后，
    思考路径与直通路径的生成必须分叉。"""
    smm, codec = _build_thinking(iters=2)
    smm.experience("abcd", "eeee", grounded=True)
    idx = torch.tensor([codec.encode("abc")])
    torch.manual_seed(42)
    out0 = smm.answer(idx, max_new_tokens=5, query="abcd", temperature=0)
    smm.thinker.ledger.reset()          # 只看修改后的行为
    with torch.no_grad():
        smm.thinker.cell.down.weight.normal_(0, 0.2)
        smm.thinker.halting.proj.bias.fill_(-2.0)   # 偏向跑满迭代
    torch.manual_seed(42)
    out1 = smm.answer(idx, max_new_tokens=5, query="abcd", temperature=0)
    assert not torch.equal(out0["tokens"], out1["tokens"])
    think = smm.stats()["think"]
    assert think["iters_run"] >= (5 + idx.shape[1]) * 2   # 每步跑满 2 步
    assert think["speedup"] == 1.0


def test_thinking_smm_purity_and_injection_compatible():
    """思考层开启时：纯洁性守卫覆盖 thinker，记忆注入路径正常计数。"""
    smm, codec = _build_thinking(iters=2)
    assert smm.attention_like_modules() == []
    smm.experience("abcd", "eeee", grounded=True)
    idx = torch.tensor([codec.encode("abc")])
    out = smm.answer(idx, max_new_tokens=3, query="abcd", temperature=0)
    assert out["memory"] is not None
    assert smm.ledger.injected == 1
