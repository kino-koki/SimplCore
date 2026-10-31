# -*- coding: utf-8 -*-
"""端到端：ReflectionLoop + 代码执行 verifier 应当真正走到 COMMIT。

这是支柱 C 的"解锁证明"：``mode="weights"`` 在旧实现里因为没有独立 verifier
会被直接 ROLLBACK；现在用真实执行结果作 verifier，gate 能判定并提交。
"""
from __future__ import annotations

from simplcore.core.code_exec import CodeExecutionVerifier, make_smoke_problems
from simplcore.core.config import SimplCoreConfig
from simplcore.core.dream import (
    ReflectionConfig,
    ReflectionLoop,
    ExperienceBuffer,
    Insight,
    Trajectory,
    TrainExample,
)
from simplcore.core.model import SimplCoreModel


def _model() -> SimplCoreModel:
    return SimplCoreModel(SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_heads=2,
                              num_bases=4, ffn_ratio=2.0))


def test_weights_mode_commits_with_code_verifier() -> None:
    probs = make_smoke_problems()
    by_name = {p.name: p for p in probs}
    solutions = {
        by_name["add"].prompt: "```python\ndef add(a, b):\n    return a + b\n```",
        by_name["is_even"].prompt: "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        by_name["reverse_str"].prompt: "```python\ndef reverse_str(s):\n    return s[::-1]\n```",
    }
    verifier = CodeExecutionVerifier(probs, lambda p: solutions[p], timeout_s=20)

    buf = ExperienceBuffer()
    buf.add(Trajectory(prompt="demo", response="demo", grounded=False, score=1.0))

    def reflector(model, trajs):
        return [Insight(text="补一个可执行的修复", source=trajs[0],
                        repair="```python\nreturn 1\n```")]

    def synthesizer(model, insights):
        return [TrainExample(prompt="demo", target="return 1")]

    loop = ReflectionLoop(
        model=_model(),
        buffer=buf,
        config=ReflectionConfig(mode="weights", require_regression_check=True,
                           max_rounds=1, bootstrap_samples=50),
        reflector=reflector,
        synthesizer=synthesizer,
        consolidator=lambda m, ex: None,
        verifier=verifier,
    )
    rep = loop.step()

    assert rep.stage.value == "commit", f"未提交，notes={rep.notes}"
    assert rep.consolidated and not rep.rolled_back
    assert rep.baseline_score == 1.0 and rep.new_score == 1.0
    assert rep.ci_lower.get("pass_rate", 0.0) >= 0.0
