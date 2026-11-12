# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""paper_audit 守护测试 —— 论文数字必须能从代码重算。"""
from __future__ import annotations

import json

import pytest

from simplcore.core.paper_audit import run_audit


@pytest.fixture(scope="module")
def rep() -> dict:
    # collect_tests=False：不跑 pytest 收集（C16/C17 属慢项，CLI 全量跑覆盖）
    return run_audit(".", collect_tests=False)


def test_all_fast_checks_pass(rep) -> None:
    assert rep["protocol"] == "paper-audit"
    assert rep["failed_ids"] == [], rep["failed_ids"]
    assert all(c["match"] for c in rep["checks"])


def test_core_ledger_items_present(rep) -> None:
    ids = {c["id"] for c in rep["checks"]}
    # 部署账、流式状态、FLOPs、KV、Chinchilla、战役日历、证明/守卫计数
    for need in ("C01-deploy-1m-total", "C04-stream-state", "C05-infer-flops",
                 "C06-kv-1m", "C07-kv-paper-text", "C08-chinchilla-1b",
                 "C10-campaign-days", "C12-proofs", "C13-theory-guards"):
        assert need in ids


def test_deploy_cap_is_hard(rep) -> None:
    c = {x["id"]: x for x in rep["checks"]}["C02-deploy-cap"]
    assert c["actual"] is True


def test_honesty_note_marks_s7_as_speculation(rep) -> None:
    assert "推测" in rep["honesty"]["note"]


def test_kv_single_source_matches_longctx_claim(rep) -> None:
    """KV 单一事实源 = 120 GiB（longctx_eval 同一口径），复现报告必须一致。"""
    c = {x["id"]: x for x in rep["checks"]}
    assert c["C06-kv-1m"]["actual"] == 120 * 2 ** 30
    assert c["C07-kv-paper-text"]["match"] is True


def test_cli_writes_report(tmp_path) -> None:
    from simplcore.core.paper_audit import main
    out = tmp_path / "audit.json"
    rc = main(["--out", str(out), "--no-collect"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_checks"] >= 15
    assert rc == 0

# ---------------------------------------------------------------------------
# 状态字节断言：配置必须声明 + 量级分档
# ---------------------------------------------------------------------------
def test_state_byte_audit_tiers() -> None:
    from simplcore.core.paper_audit import audit_state_bytes

    # 0.05% → PASS
    assert audit_state_bytes(1000, 1000.5)["status"] == "PASS"
    # 0.5% → WARN（可接受但需标注）
    assert audit_state_bytes(1000, 1005)["status"] == "WARN"
    # 2% → FAIL
    assert audit_state_bytes(1000, 1020)["status"] == "FAIL"
    # 整数量级不匹配 → FAIL（"凑不出"的典型）
    assert audit_state_bytes(1000, 10_000)["status"] == "FAIL"
    assert audit_state_bytes(1000, 10_000)["reason"] == "magnitude_mismatch"
    # 未声明配置 → FAIL，即使数字本身对
    v = audit_state_bytes(1000, 1000, has_config=False)
    assert v["status"] == "FAIL" and v["reason"] == "missing_config"


def test_state_byte_claims_are_registered_and_pass(rep) -> None:
    c = {x["id"]: x for x in rep["checks"]}
    for cid in ("S01-paper-5.3-single", "S02-paper-5.3-24layer",
                "S03-readme-deploy-1b", "S04-native-120b",
                "S05-theoretical-edge"):
        assert cid in c, cid
        assert c[cid]["match"] is True, (cid, c[cid])
        assert c[cid]["config"]


def test_undocumented_state_number_is_detected(tmp_path) -> None:
    from simplcore.core.paper_audit import scan_state_mentions

    (tmp_path / "README.md").write_text(
        "O(1) 流式状态 35840 B，与序列长度无关。\n", encoding="utf-8")
    hits = scan_state_mentions(tmp_path)
    assert hits and hits[0]["bytes"] == 35_840


def test_product_docs_have_no_35840_left() -> None:
    """35840 只能作为**勘误/旧值**出现，不能再被当论文事实引用。"""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    paths = [root / "README.md", root / "SimplCore_Paper.md"]
    paths += sorted((root / "docs").glob("*.md"))
    for p in paths:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"35,?840", text):
            context = text[max(0, m.start() - 30):m.end() + 30]
            assert ("旧值" in context or "勘误" in context or "误报" in context), \
                f"{p}: 35840 只能作为勘误/旧值出现，上下文={context!r}"
