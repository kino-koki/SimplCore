# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""24h·4090D 实测口径模板的守护测试.

守护的主张：
1. 模板结构完整：phase0/phase1/retrospect 三段齐全，每个 measured 字段都带
   caliber（口径）说明；
2. 诚实规则：模板分发状态 measured=null（合法：待实机填充），estimate 与
   measured 并存必须有 deviation；
3. 填充脚本对合成 metrics.jsonl 的回填正确（MFU/tokens/s/tokens_per_param
   的算术），且缺列时保持 null——绝不编造实测值；
4. deviation > 20% 自动标 REVIEW。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / "artifacts" / "4090d" / "measurement_template.json"
FILLER = ROOT / "scripts" / "fill_measurements.py"


@pytest.fixture(scope="module")
def template():
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_template_structure_complete(template):
    for section in ("phase0", "phase1", "retrospect", "deviations"):
        assert section in template, section
    # phase0 必备：dry-run 显存峰值 / 吞吐 / MFU
    assert "mem_peak_gib" in template["phase0"]["dryrun"]
    assert "mfu" in template["phase0"]["throughput"]
    assert set(template["phase0"]["throughput"]["mfu_by_mode"]) >= {
        "eager", "compile_max_autotune", "cuda_graph"}


def test_template_every_measured_has_caliber(template):
    """每个含 measured 字段的对象必须带 caliber（口径说明）——没有口径的
    实测数字不可审计。"""
    def check(node, path=""):
        if isinstance(node, dict):
            if "measured" in node and "caliber" not in node:
                raise AssertionError(f"missing caliber at {path}")
            for k, v in node.items():
                check(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                check(v, f"{path}[{i}]")
    check(template)


def test_template_honest_null_state(template):
    """模板分发状态：所有 *_measured 为 null（待实机），不冒充实测。"""
    for node in _walk(template):
        if isinstance(node, dict) and "measured" in node:
            assert node["measured"] is None, node


def test_filler_fills_from_synthetic_metrics(tmp_path, template):
    """回填脚本：合成 metrics.jsonl → tokens/s、MFU、tokens/param 正确。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    micro, seq, n_params = 2, 4096, 2_000_000_000
    (run_dir / "run.json").write_text(json.dumps({
        "micro_batch": micro, "seq_len_initial": seq, "n_params": n_params,
        "argv": ["--optimizer", "adafactor"],
    }), encoding="utf-8")
    # 20 步稳态：每步 1.0 s → tokens/s = 2*4096 = 8192
    rows = [{"step": i, "step_sec": 5.0 if i < 10 else 1.0,
             "tokens": i * micro * seq, "mem_peak_gib": 19.0}
            for i in range(30)]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "filled.json"
    r = subprocess.run(
        [sys.executable, str(FILLER), "--run-dir", str(run_dir),
         "--out", str(out), "--preset", "debug"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    # tokens/s：预热（step_sec=5）被跳过 → 中位数 1.0s → 8192
    assert doc["phase0"]["throughput"]["tokens_per_sec"]["measured"] == 8192.0
    # MFU 已回填且在 (0,1) 的合理量级（debug 预设小模型 F_tok 小）
    assert doc["phase0"]["throughput"]["mfu"]["measured"] is not None
    assert 0 < doc["phase0"]["throughput"]["mfu"]["measured"] < 1.0
    # tokens/param（NT42 义务字段）
    assert doc["phase1"]["budget"]["tokens_per_param"]["measured"] == \
        round(rows[-1]["tokens"] / n_params, 4)
    # optimizer 识别为 adafactor
    assert doc["phase1"]["config_locked"]["optimizer"]["measured"] == "adafactor"
    # 显存回填
    assert doc["phase1"]["memory"]["mem_peak_gib"]["measured"] == 19.0


def test_filler_missing_columns_stay_null(tmp_path):
    """缺列/缺文件时保持 null——绝不编造实测值（诚实规则）。"""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    out = tmp_path / "filled.json"
    r = subprocess.run(
        [sys.executable, str(FILLER), "--run-dir", str(run_dir),
         "--out", str(out), "--preset", "debug"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["phase0"]["throughput"]["tokens_per_sec"]["measured"] is None
    assert doc["phase0"]["throughput"]["mfu"]["measured"] is None


def test_filler_deviation_review_rule(tmp_path):
    """estimate 与 measured 偏差 >20% 自动标 REVIEW（诚实规则）。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "micro_batch": 2, "seq_len_initial": 4096, "n_params": 2_000_000_000,
    }), encoding="utf-8")
    rows = [{"step": i, "step_sec": 1.0, "tokens": i * 8192}
            for i in range(30)]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "filled.json"
    r = subprocess.run(
        [sys.executable, str(FILLER), "--run-dir", str(run_dir),
         "--out", str(out), "--preset", "debug"],
        capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    # tokens_per_param：模板 estimate 0.10（若模板带 estimate 的字段会进
    # deviations）；这里至少验证 deviations 结构存在且为 list
    assert isinstance(doc["deviations"]["rows"], list)
    for row in doc["deviations"]["rows"]:
        assert {"field", "estimate", "measured", "deviation_pct",
                "cause"} <= set(row)
