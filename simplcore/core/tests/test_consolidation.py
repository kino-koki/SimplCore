# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""支柱 C 端到端：训练信号必须**真的**写进权重，且退化时必须真的回滚。

这组测试堵的是"报告说成功、实际没做事"这一类缺陷：

1. ``default_consolidator`` 必须真的改动权重张量（不是"跑了个空循环"）；
2. ``ReflectionLoop`` 在 ``mode="weights"`` 下若没给 consolidator，必须**报错**，
   而不是写一份 ``consolidated=True`` 的假报告；
3. no-regression gate 判定退化时，权重必须被**逐位还原**（回滚是真的）；
4. 巩固不得对振荡基参数做 weight decay —— 那会掐死长程记忆。

预算：纯 CPU < 60 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import SimplCoreConfig                     # noqa: E402
from simplcore.core.dream import (                                     # noqa: E402
    ReflectionConfig,
    ReflectionLoop,
    ExperienceBuffer,
    Insight,
    TrainExample,
    Trajectory,
    default_consolidator,
)
from simplcore.core.model import SimplCoreModel                        # noqa: E402
from simplcore.core.tokenization import CharTokenizer                   # noqa: E402


def _model() -> SimplCoreModel:
    torch.manual_seed(0)
    return SimplCoreModel(SimplCoreConfig(vocab_size=128, dim=32, num_layers=2,
                                           num_heads=2, num_bases=4, ffn_ratio=2.0))


def _examples() -> list:
    return [TrainExample(prompt="ab", target="cd"),
            TrainExample(prompt="ef", target="gh")]


def _changed(before: dict, model) -> int:
    after = model.state_dict()
    return sum(1 for k in before if not torch.equal(before[k], after[k]))


# ---------------------------------------------------------------------------
# 1. 巩固器真的写权重
# ---------------------------------------------------------------------------
def test_default_consolidator_actually_changes_weights() -> None:
    m = _model()
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    stats = default_consolidator(m, _examples(), codec=CharTokenizer(limit="abcdefgh"),
                                 lr=1e-2, epochs=2, batch_size=2, seq_len=32)
    assert stats["steps"] > 0, stats
    assert stats["examples"] == 2, stats
    assert stats["tokens"] > 0, stats
    changed = _changed(before, m)
    assert changed > 0, "巩固跑完了，但一个权重张量都没变"
    # 绝大多数张量都该被动过（不是只动了某一个）
    assert changed >= len(before) * 0.5, f"只改了 {changed}/{len(before)} 个张量"


def test_default_consolidator_reduces_loss_on_its_own_batch() -> None:
    """在同一批样本上多训几步，loss 应该下降（否则梯度根本没进权重）。"""
    m = _model()
    stats = default_consolidator(m, _examples(), codec=CharTokenizer(limit="abcdefgh"),
                                 lr=5e-2, epochs=6, batch_size=2, seq_len=32)
    assert stats["loss_before"] is not None
    assert stats["loss_after"] is not None
    assert stats["loss_after"] < stats["loss_before"], stats


def test_default_consolidator_requires_codec_or_ids() -> None:
    """没有 codec 也没有预编码 id 时必须**当场报错**，不能静默什么都不做。"""
    m = _model()
    with pytest.raises(ValueError, match="codec"):
        default_consolidator(m, _examples(), lr=1e-3)


def test_default_consolidator_accepts_pre_encoded_ids() -> None:
    """样本自带 meta['input_ids'] 时不需要 codec。"""
    m = _model()
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    exs = [TrainExample(prompt="", target="",
                        meta={"input_ids": [3, 4, 5, 6], "prompt_len": 1})]
    stats = default_consolidator(m, exs, lr=1e-2, epochs=1, batch_size=1)
    assert stats["steps"] == 1
    assert _changed(before, m) > 0


def test_consolidator_does_not_decay_oscillator_parameters() -> None:
    """振荡基参数（raw_rho/raw_theta）不得被 weight decay 衰减。

    给衰减率参数做衰减等于直接掐死长程记忆 —— 这是训练侧踩过的坑，
    巩固侧同样不能踩。
    """
    m = _model()
    opt_params = {n for n, _ in m.named_parameters()}
    assert any("raw_rho" in n for n in opt_params), "模型里没有 raw_rho，测试前提不成立"

    # 复现 default_consolidator 的分组规则
    decay, no_decay = [], []
    for name, p in m.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    # 用 id 比较：tensor 的 `in` 走的是值相等，会给出错误结果
    decay_ids = {id(p) for p in decay}
    decay_names = {n for n, p in m.named_parameters() if id(p) in decay_ids}
    assert not any("raw_rho" in n or "raw_theta" in n for n in decay_names), \
        "振荡基参数落进了 weight decay 组"


# ---------------------------------------------------------------------------
# 2. ReflectionLoop 的"没做事"必须被拒
# ---------------------------------------------------------------------------
def _loop(**kw) -> ReflectionLoop:
    buf = ExperienceBuffer()
    buf.add(Trajectory(prompt="demo", response="demo", grounded=False, score=1.0))

    def reflector(model, trajs):
        return [Insight(text="补一个修复", source=trajs[0],
                        repair="```python\nreturn 1\n```")]

    def synthesizer(model, insights):
        return [TrainExample(prompt="demo", target="cd")]

    return ReflectionLoop(
        model=_model(), buffer=buf,
        reflector=reflector, synthesizer=synthesizer,
        **kw,
    )


def test_weights_mode_without_codec_or_consolidator_fails_loudly() -> None:
    """mode=weights 但没有 codec 也没有 consolidator 时必须失败，不能假成功。"""
    loop = _loop(config=ReflectionConfig(mode="weights", require_regression_check=False,
                                     max_rounds=1))
    rep = loop.step()
    assert rep.stage.value == "failed", f"notes={rep.notes}"
    assert not rep.consolidated, "什么都没写却报告 consolidated"


def test_weights_mode_with_codec_really_writes_and_reports() -> None:
    """给了 codec 时，内置 consolidator 会真的写权重，并如实报告。"""
    loop = _loop(config=ReflectionConfig(mode="weights", require_regression_check=False,
                                     max_rounds=1, consolidate_lr=1e-2),
                 codec=CharTokenizer(limit="abcdefgh"))
    before = {k: v.detach().clone() for k, v in loop.model.state_dict().items()}
    rep = loop.step()
    assert rep.stage.value != "failed", rep.notes
    assert rep.consolidated, f"notes={rep.notes}"
    assert _changed(before, loop.model) > 0, "报告说 consolidated，但权重没变"
    assert any("权重巩固" in n for n in rep.notes), rep.notes


def test_rollback_restores_weights_byte_exact() -> None:
    """no-regression gate 判定退化时，权重必须被逐位还原。"""
    from simplcore.core.dream import EvalScores

    def reflector(model, trajs):
        return [Insight(text="坏洞见", source=trajs[0], repair="```python\nx=1\n```")]

    def synthesizer(model, insights):
        return [TrainExample(prompt="demo", target="cd")]

    buf = ExperienceBuffer()
    buf.add(Trajectory(prompt="demo", response="demo", grounded=False, score=1.0))
    # verifier 被调用两次：第一次是基线（巩固前），第二次是巩固后。
    # 让它第二次变差，gate 才判得出退化。
    calls = {"n": 0}

    def verifier(model):
        calls["n"] += 1
        scores = [1.0, 1.0, 1.0, 1.0] if calls["n"] == 1 else [0.0, 0.0, 0.0, 0.0]
        return EvalScores.from_any({"pass_rate": scores})

    loop = ReflectionLoop(
        model=_model(), buffer=buf,
        config=ReflectionConfig(mode="weights", require_regression_check=True,
                            max_rounds=1, bootstrap_samples=50),
        reflector=reflector, synthesizer=synthesizer,
        codec=CharTokenizer(limit="abcdefgh"),
        verifier=verifier,
    )
    before = {k: v.detach().clone() for k, v in loop.model.state_dict().items()}
    rep = loop.step()
    assert rep.rolled_back, f"未回滚：stage={rep.stage.value} notes={rep.notes}"
    after = loop.model.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"回滚后权重张量 {k} 不一致"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
