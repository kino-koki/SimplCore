# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""Pilot（A1/A2 闭环）冒烟测试：真实沙箱、真实判分，tiny 规模几分钟内完成。"""
from __future__ import annotations

import json

from simplcore.core.pilot import (
    PilotCodeEnvironment,
    PilotConfig,
    default_reference_problems,
    run_pilot,
)


def _tiny_pcfg() -> PilotConfig:
    return PilotConfig(rounds=1, max_new_tokens=24, consolidate_epochs=1,
                       batch_size=2, n_boot=100, seed=7, timeout_s=5.0)


def test_reference_problems_have_references():
    problems = default_reference_problems()
    assert len(problems) >= 5
    for p in problems:
        assert p.meta.get("reference"), f"题目 {p.name} 缺参考实现"
        assert p.tests, f"题目 {p.name} 缺测试"


def test_pilot_env_fills_reference_on_failure():
    problems = default_reference_problems()
    env = PilotCodeEnvironment(problems, timeout_s=5.0)
    # 常败生成器：输出语法非法代码 → 全部失败 → 必须回填参考实现
    rollouts = env.rollout(lambda prompt: "def broken(:\n    return")
    assert len(rollouts) == len(problems)
    for r in rollouts:
        assert not r.passed
        assert r.correct, f"失败轨迹未回填参考实现：{r.prompt[:20]}"


def test_run_pilot_end_to_end(tmp_path):
    rep = run_pilot(out_dir=tmp_path / "pilot", pcfg=_tiny_pcfg())
    # 结构完整
    assert rep["protocol"] == "pilot-a1a2"
    assert rep["verdict"] in ("pass", "fail", "inconclusive")
    assert rep["a2"]["stages"] == 1
    assert rep["honesty"]["claim_permitted"] is False
    # A1 锚点都是真实沙箱逐题分数（0..1）
    for score in rep["a1"].get("_", []) or []:
        assert 0.0 <= score <= 1.0
    assert 0.0 <= rep["a1"]["baseline_pass_rate"] <= 1.0
    assert 0.0 <= rep["a1"]["final_pass_rate"] <= 1.0
    # 账本落盘且逐行可解析
    ledger = (tmp_path / "pilot" / "ledger.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in ledger]
    assert events[0] == "baseline" and events[-1] == "final"
    assert events.count("round") == 1
    # 报告落盘
    report = json.loads((tmp_path / "pilot" / "report.json").read_text(
        encoding="utf-8"))
    assert report["verdict"] == rep["verdict"]
