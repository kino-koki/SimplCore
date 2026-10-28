# -*- coding: utf-8 -*-
"""代码沙箱 + verifier 的测试。"""
from __future__ import annotations

import os

import pytest

from simplcore.core.code_exec import (
    CodeExecutionVerifier,
    LocalPythonSandbox,
    extract_code,
    make_smoke_problems,
    run_python_tests,
)
from simplcore.core.dream import EvalScores


def test_correct_code_passes() -> None:
    r = run_python_tests("def add(a, b):\n    return a + b",
                         ["assert add(2, 3) == 5"], timeout_s=20)
    assert r.ok and r.score == 1.0 and r.passed == 1 and r.total == 1


def test_wrong_code_fails_with_assertion_exit() -> None:
    r = run_python_tests("def add(a, b):\n    return a - b",
                         ["assert add(2, 3) == 5"], timeout_s=20)
    assert not r.ok and r.score == 0.0
    assert r.outcomes[0].exit_code == 3


def test_syntax_error_is_reported() -> None:
    r = run_python_tests("def add(a, b)\n    return a + b",
                         ["assert True"], timeout_s=20)
    assert not r.ok and not r.outcomes[0].passed
    assert "SyntaxError" in r.outcomes[0].stderr or r.outcomes[0].exit_code == 1


def test_infinite_loop_times_out() -> None:
    r = run_python_tests("while True:\n    pass", ["assert True"], timeout_s=1.2)
    assert r.timed_out and not r.ok


def test_extract_fenced_code() -> None:
    txt = "这是代码：\n```python\ndef f():\n    return 1\n```\n完毕"
    assert extract_code(txt) == "def f():\n    return 1"


def test_extract_raw_code() -> None:
    txt = "def f():\n    return 1"
    assert extract_code(txt) == "def f():\n    return 1"


def test_verifier_returns_per_problem_scores() -> None:
    probs = make_smoke_problems()[:2]
    good = {
        probs[0].prompt: "```python\ndef add(a, b):\n    return a + b\n```",
        probs[1].prompt: "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
    }
    v = CodeExecutionVerifier(probs, lambda p: good[p], timeout_s=20)
    out = v(None)
    assert list(out.keys()) == ["pass_rate"]
    assert out["pass_rate"] == [1.0, 1.0]
    # 必须能被 dream 的 EvalScores 归一（逐样本列表，不是单点）
    scores = EvalScores.from_any(out)
    assert len(scores.metrics["pass_rate"]) == 2


def test_verifier_detects_broken_code() -> None:
    probs = make_smoke_problems()[:1]
    v = CodeExecutionVerifier(probs, lambda p: "```python\ndef add(a, b):\n    return 0\n```",
                              timeout_s=20)
    assert v(None)["pass_rate"] == [0.0]
    assert v.results[0]["name"] == "add"


def test_win_job_object_enforces_memory_limit() -> None:
    """Windows：Job Object 内存配额必须真实生效（呼应 POSIX 的 RLIMIT_AS）.

    64MB 配额下分配 200MB 应当触发 MemoryError → 脚本约定 exit 4。
    """
    if os.name != "nt":
        pytest.skip("Windows Job Object 专属测试")
    box = LocalPythonSandbox(timeout_s=30, memory_limit_mb=64)
    out = box.run("x = bytearray(200 * 1024 * 1024)", "assert True")
    assert not out.passed, "64MB 配额下 200MB 分配必须失败"
    assert out.exit_code in (1, 4), f"意外的退出码: {out.exit_code}, stderr={out.stderr[:200]}"


def test_win_job_object_allows_normal_allocation() -> None:
    """Windows：配额内的正常计算不受影响（默认 2048MB 档）。"""
    if os.name != "nt":
        pytest.skip("Windows Job Object 专属测试")
    box = LocalPythonSandbox(timeout_s=30)
    out = box.run("x = bytearray(16 * 1024 * 1024)\nassert len(x) == 16 * 1024 * 1024",
                  "assert True")
    assert out.passed, f"正常分配不应失败: {out.stderr[:200]}"
