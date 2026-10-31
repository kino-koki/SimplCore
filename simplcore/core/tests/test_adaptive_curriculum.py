# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""自适应序列长度课程测试（meso→macro 反馈闭环）。

覆盖：
1. 就绪度探针：场能量越过代谢天花板 → 0.0（系统失衡，课程须推迟）；
   场稳定且有余量 → 就绪度高；
2. 自适应阶段推进：就绪度不足 → **推迟**拉长；就绪度达标 → 按调度推进；
   宽限期耗尽 → 强制推进（避免课程死锁）；
3. 零回归：自适应关闭时课程行为与调度表完全一致。
"""
import argparse
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.seqlen_curriculum import SeqLenCurriculum, build_curriculum
from simplcore.core.bio_computation import BioEnhancer


@pytest.fixture(autouse=True)
def _preserve_global_rng():
    """守住全局 RNG 状态：本文件若改写种子会污染套件里的确定性测试。"""
    torch_state = torch.random.get_rng_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        random.setstate(py_state)


def _toy():
    cfg = SimplCoreConfig(dim=32, num_layers=2, num_bases=8, vocab_size=64,
                          dropout=0.0, tie_embeddings=True)
    return SimplCoreModel(cfg)


# ---------------------------------------------------------------------------
# 就绪度探针
# ---------------------------------------------------------------------------
def test_readiness_in_range_and_zero_when_over_budget():
    bio = BioEnhancer(_toy(), metabolic_budget=1.0)
    bio.field.reset(2, torch.device("cpu"), torch.float32)
    r0 = bio.readiness()
    assert 0.0 <= r0 <= 1.0

    # 场能量冲破代谢天花板（budget × D = 1.0 × 32 = 32）→ 就绪度归零
    with torch.no_grad():
        bio.field.phi.fill_(100.0)
    r1 = bio.readiness()
    assert r1 == 0.0, f"超代谢预算时就绪度应为 0，实际 {r1}"


def test_readiness_higher_when_stable():
    bio = BioEnhancer(_toy(), metabolic_budget=1.5)
    bio.field.reset(2, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        bio.field.phi.fill_(1.0)          # 非零场（0 场无法体现波动）
    # 连续若干步、场不变 → EMA 收敛、场稳定 → 就绪度高
    for _ in range(20):
        bio.readiness()
    stable = bio.readiness()

    # 场突然暴涨 → 偏离 EMA → 稳定度下降 → 就绪度下降
    with torch.no_grad():
        bio.field.phi.mul_(8.0)
    unstable = bio.readiness()
    assert unstable < stable, f"场不稳时就绪度应更低：{unstable} vs {stable}"


# ---------------------------------------------------------------------------
# 自适应阶段推进
# ---------------------------------------------------------------------------
def _adaptive_cur():
    cur = SeqLenCurriculum.from_cli("4096:0.1,16384:0.25,65536:0.45,4000000", 1000)
    cur.adaptive = True
    return cur


def test_gate_delays_promotion_when_not_ready():
    cur = _adaptive_cur()
    # step 100 已越过第一阶段边界（0.1*1000=100），但就绪度不足 → 不推进
    assert cur.promote(100, readiness=0.1, gate_threshold=0.5, grace_steps=200) is False
    assert cur.achieved_stage == 0
    assert cur.seq_len_at(100) == 4096
    assert cur.seq_len_at(500) == 4096      # 仍停在第一阶段（被推迟）
    print("[ok] gate delays promotion when readiness low")


def test_promotes_when_ready():
    cur = _adaptive_cur()
    assert cur.promote(100, readiness=0.9, gate_threshold=0.5, grace_steps=200) is True
    assert cur.achieved_stage == 1
    assert cur.seq_len_at(100) == 16384
    # 下一阶段边界在 250 步：未到不推进（绝不提前）
    assert cur.promote(200, readiness=0.9, gate_threshold=0.5, grace_steps=200) is False
    assert cur.achieved_stage == 1
    assert cur.promote(250, readiness=0.9, gate_threshold=0.5, grace_steps=200) is True
    assert cur.achieved_stage == 2
    print("[ok] promotes on schedule when readiness high")


def test_grace_forces_promotion_to_avoid_deadlock():
    cur = _adaptive_cur()
    # 就绪度一直不足，但越过边界超过宽限步数 → 强制推进（课程不死锁）
    assert cur.promote(100, readiness=0.0, gate_threshold=0.5, grace_steps=200) is False
    assert cur.achieved_stage == 0
    assert cur.promote(400, readiness=0.0, gate_threshold=0.5, grace_steps=200) is True
    assert cur.achieved_stage >= 1
    print("[ok] grace forces promotion (no deadlock)")


def test_zero_regression_when_adaptive_off():
    cur = SeqLenCurriculum.from_cli("4096:0.1,16384:0.25,65536:0.45,4000000", 1000)
    assert cur.adaptive is False
    # 与调度表一致（不受就绪度影响）
    assert cur.seq_len_at(0) == 4096
    assert cur.seq_len_at(100) == 16384
    assert cur.seq_len_at(250) == 65536
    assert cur.seq_len_at(999) == 4000000
    # promote 在非自适应模式下是 no-op
    assert cur.promote(250, readiness=0.0, gate_threshold=0.5, grace_steps=0) is False
    assert cur.seq_len_at(250) == 65536
    print("[ok] zero regression when adaptive off")


def test_no_readiness_signal_still_advances():
    """没开 --enhance bio 时无就绪度信号（None）→ 退化为按调度推进。

    否则「自适应课程但忘了开 bio」会把课程永远卡在第 0 阶段（静默退化）。
    """
    cur = _adaptive_cur()
    assert cur.promote(100, readiness=None, gate_threshold=0.5, grace_steps=200) is True
    assert cur.achieved_stage == 1
    assert cur.seq_len_at(100) == 16384
    print("[ok] no readiness signal -> scheduled advance (no stall)")


def test_build_curriculum_adaptive_flag():
    args = argparse.Namespace(seq_len_curriculum="4096:0.1,16384",
                              bio_adaptive_curriculum=True)
    cur = build_curriculum(args, type("C", (), {"context": None})(), 1000)
    assert cur.adaptive is True
    args2 = argparse.Namespace(seq_len_curriculum="4096:0.1,16384")
    cur2 = build_curriculum(args2, type("C", (), {"context": None})(), 1000)
    assert cur2.adaptive is False
    print("[ok] build_curriculum honours adaptive flag")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL ADAPTIVE CURRICULUM TESTS PASSED")
