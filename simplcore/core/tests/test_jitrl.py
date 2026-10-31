# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""JitRL 式测试时优势调制层的数值与行为守护。

对应 ``jitrl.py``（v1.2，agent 域）。守护目标：

* **恒等性**：``alpha=0`` / 空记忆 / ``logit_bias_fn=None`` 时，所有
  生成路径与未接入**逐位一致**（默认关闭的硬承诺）；
* **优势估计数学**：``A = Σ sim·r / (Σ sim + eps)`` 对已知构造记忆的
  精确值，以及"无经验 → 优势 0"的空转防护；
* **偏置编译**：优势 → 词表级 bias 的落点/零点/None 语义；
* **闭环**：成功经验回填后，同任务检索出的偏置会**提高**该动作的
  logit（采样的确定性核对，不宣称成功率提升——那是环境实测的事）；
* **生成钩子**：``generate(logit_bias_fn=...)`` 与手工
  ``logits + bias`` 的一致性，及 None 钩子的逐位恒等。

诚实边界：本模块不含任何环境基准（WebArena/Jericho）数字——那些属于
原论文的外部 API 模型；效果判据见论文 §7.4 W4。
"""
from __future__ import annotations

import math

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.jitrl import (AdvantageEstimator, JitRLController,
                                  TrajectoryMemory, Transition,
                                  advantage_logit_bias, apply_logit_bias)
from simplcore.core.model import SimplCoreModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _model() -> SimplCoreModel:
    torch.manual_seed(0)
    return SimplCoreModel(SimplCoreConfig(vocab_size=64, dim=32, num_layers=2,
                                          num_bases=8, local_kernel=4)).eval()


# ---------------------------------------------------------------------------
# 1. identity guarantees (default-off)
# ---------------------------------------------------------------------------
def test_zero_alpha_is_identity():
    """alpha=0：bias_for 恒 None，advantage_logit_bias 恒 None。"""
    ctrl = JitRLController(vocab_size=64, alpha=0.0)
    ctrl.record([1.0, 0.0], action_token=7, reward=1.0)
    assert ctrl.bias_for([1.0, 0.0]) is None
    assert advantage_logit_bias({7: 1.0}, 64, alpha=0.0) is None


def test_empty_memory_is_identity():
    """空记忆：有 alpha 也无偏置——无经验即无主张（不空转）。"""
    ctrl = JitRLController(vocab_size=64, alpha=2.0)
    assert ctrl.bias_for([1.0, 0.0]) is None


def test_bias_none_when_all_advantages_zero():
    """候选动作全无经验（优势 0）→ None 而非零张量（避免无谓加法）。"""
    mem = TrajectoryMemory()
    mem.add(Transition(task_key=(1.0, 0.0), action_token=5, reward=1.0))
    est = AdvantageEstimator(mem)
    advs = est.action_advantages([0.0, 1.0], [9, 10])   # 与经验正交的动作
    assert all(v == 0.0 for v in advs.values())
    assert advantage_logit_bias(advs, 64, alpha=2.0) is None


# ---------------------------------------------------------------------------
# 2. advantage estimation math
# ---------------------------------------------------------------------------
def test_advantage_exact_value():
    """构造已知记忆，断言 A = Σ sim·r / (Σ sim + eps) 的精确值。"""
    mem = TrajectoryMemory()
    # 两条与查询 (1,0) 的相似度可控的经验：同向 r=1、反向（sim=0，不投票）
    mem.add(Transition(task_key=(1.0, 0.0), action_token=7, reward=1.0))
    mem.add(Transition(task_key=(0.0, 1.0), action_token=7, reward=0.0))
    est = AdvantageEstimator(mem)
    a = est.advantage([1.0, 0.0], 7)
    assert a == pytest.approx(1.0 / (1.0 + 1e-6), rel=1e-6)
    # 两条同向：一条 r=1 一条 r=0，等权 → A ≈ 0.5
    mem.add(Transition(task_key=(1.0, 0.0), action_token=7, reward=0.0))
    a2 = est.advantage([1.0, 0.0], 7)
    assert 0.4 < a2 < 0.6


def test_advantage_zero_without_experience():
    mem = TrajectoryMemory()
    mem.add(Transition(task_key=(1.0, 0.0), action_token=5, reward=1.0))
    est = AdvantageEstimator(mem)
    assert est.advantage([1.0, 0.0], 99) == 0.0    # 未出现过的动作
    assert est.advantage([0.0, 0.0], 5) == 0.0     # 零向量查询（sim=0 不投票）


def test_memory_capacity_eviction():
    mem = TrajectoryMemory(capacity=3)
    for i in range(5):
        mem.add(Transition(task_key=(float(i),), action_token=i, reward=1.0))
    assert len(mem) == 3
    assert [tr.action_token for tr in mem.transitions] == [2, 3, 4]


# ---------------------------------------------------------------------------
# 3. bias compilation
# ---------------------------------------------------------------------------
def test_bias_compilation_and_apply():
    bias = advantage_logit_bias({7: 0.8, 3: 0.2}, vocab_size=64, alpha=2.0)
    assert bias is not None and bias.shape == (64,)
    assert bias[7].item() == pytest.approx(1.6)
    assert bias[3].item() == pytest.approx(0.4)
    assert bias[10].item() == 0.0
    logits = torch.randn(2, 5, 64)
    out = apply_logit_bias(logits, bias)
    assert torch.allclose(out, logits + bias)
    assert out is not logits
    # None → 恒等（同一对象，不复制）
    assert apply_logit_bias(logits, None) is logits


def test_bias_out_of_vocab_ignored():
    """越界 token id 被忽略（不写入、不越界）：只有合法 id 生效。"""
    bias = advantage_logit_bias({999: 1.0, 7: 1.0}, vocab_size=64, alpha=1.0)
    assert bias is not None and bias.shape == (64,)
    assert bias[7].item() == pytest.approx(1.0)
    assert float(bias.abs().sum().item()) == pytest.approx(1.0)  # 999 被丢弃


# ---------------------------------------------------------------------------
# 4. closed loop: record success → bias favors the action
# ---------------------------------------------------------------------------
def test_record_then_bias_favors_action():
    ctrl = JitRLController(vocab_size=64, alpha=10.0)
    task = [1.0, 0.0]
    ctrl.record(task, action_token=7, reward=1.0)      # verifier 判定成功
    ctrl.record(task, action_token=8, reward=0.0)      # 失败
    bias = ctrl.bias_for(task)
    assert bias is not None
    assert bias[7].item() > bias[8].item()             # 成功动作占优
    assert bias[7].item() > 0.0 and bias[8].item() == pytest.approx(0.0)


def test_controller_rejects_bad_args():
    with pytest.raises(ValueError):
        JitRLController(vocab_size=0)
    with pytest.raises(ValueError):
        JitRLController(vocab_size=64, alpha=-1.0)


# ---------------------------------------------------------------------------
# 5. generation hook: bit-exact identity + manual-bias equivalence
# ---------------------------------------------------------------------------
def test_generate_identity_without_hook():
    """logit_bias_fn=None（默认）：两条流式路径与历史行为逐位一致。"""
    m = _model()
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        a = m.generate(idx, max_new_tokens=6, temperature=0.0, use_state=True)
        b = m.generate(idx, max_new_tokens=6, temperature=0.0, use_state=True,
                       logit_bias_fn=None)
        c = m.generate(idx, max_new_tokens=6, temperature=0.0, use_state=False,
                       logit_bias_fn=None)
        d = m.generate(idx, max_new_tokens=6, temperature=0.0, use_state=False)
    assert torch.equal(a, b)
    assert torch.equal(c, d)


def test_generate_with_bias_matches_manual_addition():
    """钩子的语义 = 每步加性偏置：把某动作 token 顶到 argmax，核对生成
    序列与手工 `logits + bias` 逐步一致。"""
    m = _model()
    idx = torch.randint(0, 64, (1, 8))
    target = 63
    bias = advantage_logit_bias({target: 50.0}, 64, alpha=1.0)

    with torch.no_grad():
        hooked = m.generate(idx, max_new_tokens=4, temperature=0.0,
                            use_state=True, logit_bias_fn=lambda _lg: bias)
        # 手工参考：逐步取 stream_step 的 logits 加同一 bias 再 argmax
        state = m.init_stream(1, idx.device, torch.float32)
        logits = None
        ref = idx.clone()
        for t in range(idx.shape[1]):
            logits, state = m.stream_step(idx[:, t], state)
        biased = logits + bias.to(logits.dtype)
        for _ in range(4):
            nxt = biased.argmax(dim=-1, keepdim=True)
            ref = torch.cat([ref, nxt], dim=1)
            if int(nxt.item()) == target:
                break
            logits, state = m.stream_step(nxt, state)
            biased = logits + bias.to(logits.dtype)
    assert hooked[0, -1].item() == target          # 偏置把目标顶到 argmax
    assert torch.equal(hooked[0, idx.shape[1]:idx.shape[1] + 1],
                       ref[0, idx.shape[1]:idx.shape[1] + 1])


def test_hook_receives_logits_and_can_return_none():
    """钩子收到的就是未调制 logits；返回 None 时该步恒等。"""
    m = _model()
    idx = torch.randint(0, 64, (1, 8))
    seen = []

    def spy(logits):
        seen.append(logits)
        return None                                 # 恒等

    with torch.no_grad():
        a = m.generate(idx, max_new_tokens=3, temperature=0.0,
                       use_state=True, logit_bias_fn=spy)
        b = m.generate(idx, max_new_tokens=3, temperature=0.0, use_state=True)
    assert torch.equal(a, b)
    assert len(seen) == 3                           # 每个解码步调用一次
    assert all(lg.shape[-1] == 64 for lg in seen)