# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""环境学习模块的测试。"""
from __future__ import annotations

import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.tokenization import CharTokenizer
from simplcore.core.envlearn import (
    WeightBudget,
    LearnedAdapter,
    ForwardAdapter,
    EnvTrajectory,
    LocalCodeEnvironment,
    EnvironmentLearner,
    summarize_failures,
    EvalScores,
)
from simplcore.core.purity import attention_like_modules


def _pair():
    torch.manual_seed(0)
    codec = CharTokenizer(limit="abcd")
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_bases=8,
                                    decay_max=128, local_kernel=3, num_heads=2,
                                    ffn_ratio=2.0, mixer="sgc")
    m = SimplCoreModel(cfg).eval()
    adapter = LearnedAdapter(dim=32, num_blocks=2)
    fa = ForwardAdapter(m, adapter, codec)
    return m, adapter, fa, codec


# --------------------------------------------------------------------------
# WeightBudget
# --------------------------------------------------------------------------
def test_budget_refuses_over_cap():
    b = WeightBudget(base_params=1_000_000_000)
    assert b.can_allocate(2_000_000_000)
    b.allocate(2_000_000_000)
    assert not b.can_allocate(1)
    try:
        b.allocate(1)
        assert False
    except ValueError:
        pass


def test_budget_vram_is_only_learned():
    b = WeightBudget(base_params=1_000_000_000)
    b.allocate(500_000_000)
    # 显存只随可学权重增长，不随总权重
    assert b.vram_params() == 500_000_000
    assert b.vram_params() == b.learned_params


# --------------------------------------------------------------------------
# ForwardAdapter: 冻结基座、只训适配器
# --------------------------------------------------------------------------
def test_forward_adapter_freezes_base_trains_adapter():
    from simplcore.core.dream import default_consolidator
    m, adapter, fa, codec = _pair()
    base_before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    ad_before = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    trajs = [EnvTrajectory(prompt="ab", response="x", passed=False, correct="cd")]
    ex = summarize_failures(trajs)
    default_consolidator(fa, ex, codec=codec, lr=1e-2, epochs=2,
                        batch_size=1, seq_len=32)
    changed_base = any(not torch.equal(m.state_dict()[k], base_before[k])
                      for k in base_before)
    changed_adapter = any(not torch.equal(adapter.state_dict()[k], ad_before[k])
                         for k in ad_before)
    assert not changed_base
    assert changed_adapter


# --------------------------------------------------------------------------
# EnvironmentLearner: 提交 / 回滚 / 拒绝
# --------------------------------------------------------------------------
class _ConstV:
    def __init__(self, val): self.val = val
    def __call__(self, gen): return EvalScores.from_any({"pass_rate": [self.val] * 3})


class _DegradingV:
    def __init__(self): self.n = 0
    def __call__(self, gen):
        self.n += 1
        v = 1.0 if self.n == 1 else 0.0   # 第1次=巩固前 base，第2次=巩固后 new
        return EvalScores.from_any({"pass_rate": [v] * 3})


def test_stage_commits_when_improving():
    m, adapter, fa, codec = _pair()
    el = EnvironmentLearner(m, adapter, codec=codec, verifier=_ConstV(1.0),
                           generator=lambda p: "def add(a,b):\n return a+b\n",
                           consolidate_lr=1e-2, epochs=2)
    rep = el.stage([EnvTrajectory(prompt="ab", response="x", passed=False,
                                 correct="cd")])
    assert rep.wrote_weights and not rep.rolled_back


def test_stage_rolls_back_on_regression():
    m, adapter, fa, codec = _pair()
    el = EnvironmentLearner(m, adapter, codec=codec, verifier=_DegradingV(),
                           generator=lambda p: "def add(a,b):\n return a-b\n",
                           consolidate_lr=1e-2, epochs=2)
    snap = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
    rep = el.stage([EnvTrajectory(prompt="ab", response="x", passed=False,
                                 correct="cd")])
    restored = all(torch.equal(adapter.state_dict()[k], snap[k]) for k in snap)
    assert rep.rolled_back and restored


def test_stage_refuses_without_verifier():
    m, adapter, fa, codec = _pair()
    el = EnvironmentLearner(m, adapter, codec=codec, verifier=None,
                           generator=lambda p: "x", consolidate_lr=1e-2)
    rep = el.stage([EnvTrajectory(prompt="ab", response="x", passed=False,
                                 correct="cd")])
    assert rep.rolled_back and not rep.wrote_weights


# --------------------------------------------------------------------------
# Adapter purity / grow
# --------------------------------------------------------------------------
def test_adapter_has_no_attention():
    ad = LearnedAdapter(dim=32, num_blocks=2)
    assert attention_like_modules(ad) == []


def test_adapter_grow_respects_budget():
    bud = WeightBudget(base_params=1_000_000_000, max_learned=5000,
                       total_cap=3_000_000_000)
    ad = LearnedAdapter(dim=32, num_blocks=1)
    added = ad.grow(10, budget=bud)
    # 单个 AdapterBlock 已远超 5000 上限，grow 应一颗都不加
    assert added == 0


# --------------------------------------------------------------------------
# LocalCodeEnvironment（复用 code_exec 沙箱）
# --------------------------------------------------------------------------
def test_local_code_environment_rollout():
    from simplcore.core.code_exec import make_smoke_problems
    env = LocalCodeEnvironment(make_smoke_problems())
    good = ("def add(a, b):\n    return a + b\n"
            "def is_even(n):\n    return n % 2 == 0\n"
            "def reverse_str(s):\n    return s[::-1]\n")
    results = env.rollout(lambda p: good)
    assert len(results) == 3
    assert all(isinstance(r.passed, bool) for r in results)
