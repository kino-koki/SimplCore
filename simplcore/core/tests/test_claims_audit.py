# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""``simplcore.core.eval.claims_audit`` 的守护测试。

这个模块把 ``docs/THEORETICAL_EDGE.md`` 的每条**机制层**主张变成可执行断言。
本测试保证：

1. 所有检查都能跑通（没有检查自身崩掉）；
2. 所有主张当前都成立；
3. 每个被文档点名的模块都有至少一条检查（防止"文档写了、代码里没人验"）；
4. CLI 能正常输出、失败时返回非零退出码。

预算：全模块 CPU 上 < 60 s。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.eval.claims_audit import (                        # noqa: E402
    Check,
    format_table,
    main,
    run_all,
)

#: ``docs/THEORETICAL_EDGE.md`` §3 优势总表里逐条点名的模块
DOCUMENTED_MODULES = (
    "sgc.py",
    "srf.py",
    "selective.py",
    "gates.py",
    "abstain.py",
    "grounding.py",
    "memory.py",
    "dream.py",
    "code_exec.py",
    "extrapolation.py",
    "scaling.py",
    "purity.py",
)


@pytest.fixture(scope="module")
def results() -> list:
    return run_all()


def test_every_check_actually_ran(results) -> None:
    """没有任何一条检查因为自身异常而"没跑"。"""
    broken = [c for c in results if c.claim == "检查本身未能执行"]
    assert not broken, [f"{c.module}: {c.evidence}" for c in broken]


def test_all_claims_hold(results) -> None:
    """当前所有机制层主张都成立。"""
    failed = [f"{c.module} :: {c.claim} ({c.evidence})"
              for c in results if not c.passed]
    assert not failed, "\n".join(failed)


def test_every_documented_module_is_covered(results) -> None:
    """文档里点名的每个模块都必须有对应检查。"""
    covered = {c.module for c in results}
    missing = [m for m in DOCUMENTED_MODULES if m not in covered]
    assert not missing, f"文档声明了但没有自动验证的模块: {missing}"


def test_evidence_is_reported_not_just_a_boolean(results) -> None:
    """每条检查都要给出可核对的实测数字，而不是只给 True/False。"""
    empty = [f"{c.module} :: {c.claim}" for c in results if not c.evidence]
    assert not empty, empty


def test_table_renders_and_cli_exits_zero(capsys) -> None:
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "checks passed" in out
    for m in DOCUMENTED_MODULES:
        assert m in out, f"{m} 未出现在报告里"


def test_cli_json_output(capsys) -> None:
    import json

    rc = main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["total"] == len(payload["checks"])
    assert payload["passed"] == payload["total"]
    assert all(c["passed"] for c in payload["checks"])


def test_format_table_handles_empty() -> None:
    assert format_table([]) == "(no checks ran)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
