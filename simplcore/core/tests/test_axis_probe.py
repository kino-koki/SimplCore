# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""轴 C/D 能力对照协议（X-axis-probe）的守护测试.

只锁**机制与可复现性**，不锁实验结果（谁赢是实测数据）：
1. 任务生成器正确性（needle 存在/计数正确/Markov 转移合法）；
2. chance 口径与臂构造（参数量报告、SGC 少一个 D² 投影的事实性校验）；
3. 同 seed 重跑**确定性一致**（可复现性的核心承诺）；
4. 纯净性：axis_probe.py 在 purity 定义级白名单内、TS 臂不进 mixer 注册表；
5. 全流程冒烟（最小步数，四任务 × sgc/ts 臂跑通，报告字段齐）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from simplcore.core.eval.axis_probe import (TASK_SPECS, _make_arm,
                                            gen_aggregate_count,
                                            gen_counting, gen_markov,
                                            gen_precise_pointer, run_probe)


# ---------------------------------------------------------------------------
# 1. 任务生成器
# ---------------------------------------------------------------------------
def test_precise_pointer_generator() -> None:
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        seq, value = gen_precise_pointer(64, g)
        assert seq.shape[0] == 64 and 0 <= value < 16
        # 末位是 MARKER（查询位），中段存在 (MARKER, value) 对
        assert seq[-1].item() == 31
        body = seq[:-1].tolist()
        pairs = [(body[i], body[i + 1]) for i in range(len(body) - 1)]
        assert (31, value) in pairs, "needle 对必须存在且 value 与目标一致"


def test_aggregate_and_counting_generators() -> None:
    g = torch.Generator().manual_seed(1)
    seq, tgt = gen_aggregate_count(129, g)
    special = (seq == 30).sum().item()
    assert tgt == special % 4
    seq2, tgt2 = gen_counting(96, g)
    assert tgt2 == min((seq2 == 30).sum().item(), 8)
    assert 0 <= tgt2 < 9


def test_markov_generator_valid_transitions() -> None:
    g = torch.Generator().manual_seed(2)
    seq, tgt = gen_markov(64, g)
    assert seq.shape[0] == 64 and tgt == seq[-1].item()
    assert (seq < 16).all()


# ---------------------------------------------------------------------------
# 2. 臂构造与参数口径
# ---------------------------------------------------------------------------
def test_arm_params_sgc_fewer_than_ts() -> None:
    """同 D 下 SGC 少一个 D² 投影——参数量对照口径对 SGC 有利（事实锁定）."""
    _, sgc_info = _make_arm("sgc", dim=32, layers=2, vocab_out=48, seed=0)
    _, ts_info = _make_arm("ts", dim=32, layers=2, vocab_out=48, seed=0)
    assert sgc_info["params"] < ts_info["params"]
    _, local_info = _make_arm("local", dim=32, layers=2, vocab_out=48, seed=0)
    assert local_info["params"] > 0
    with pytest.raises(ValueError):
        _make_arm("nope", 32, 2, 48, 0)


# ---------------------------------------------------------------------------
# 3. 同 seed 确定性一致（可复现性承诺）
# ---------------------------------------------------------------------------
def test_same_seed_reproducible() -> None:
    a = run_probe(tasks=["counting"], arms=["sgc"], steps=8, dim=16,
                  layers=1, batch=4, n_eval=32, seed=7)
    b = run_probe(tasks=["counting"], arms=["sgc"], steps=8, dim=16,
                  layers=1, batch=4, n_eval=32, seed=7)
    assert a.rows["counting"]["sgc"] == b.rows["counting"]["sgc"], \
        "同 seed 重跑必须逐位一致（对照协议的核心承诺）"
    acc = a.rows["counting"]["sgc"]["acc"]
    assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# 4. purity 白名单与报告字段
# ---------------------------------------------------------------------------
def test_purity_whitelist_and_report_fields() -> None:

    from simplcore.core.arch.purity import scan_source_attention_free
    root = Path(__file__).resolve().parents[3] / "simplcore"
    assert scan_source_attention_free(root) == [], \
        "axis_probe 的 TS 臂必须被定义级白名单豁免，且其余模块保持纯净"
    text = (root / "core" / "eval" / "axis_probe.py").read_text(encoding="utf-8")
    assert "purity: allow-attention-reference" in text[:3000]
    rep = run_probe(tasks=["precise_pointer"], arms=["sgc", "ts"], steps=6,
                    dim=16, layers=1, batch=4, n_eval=16, seed=0)
    row = rep.rows["precise_pointer"]
    assert set(row) == {"chance", "sgc", "ts"}
    assert row["chance"] == pytest.approx(1 / 16)
    for arm in ("sgc", "ts"):
        assert 0.0 <= row[arm]["acc"] <= 1.0
        assert row[arm]["params"] > 0
    assert "不" in rep.note and "显著性" in rep.note   # 诚实口径在账本里

# ---------------------------------------------------------------------------
# 5. 负面结果声明不得被悄悄删除（文档守护）
# ---------------------------------------------------------------------------
def test_axis_findings_persist_in_docs() -> None:
    """2026-10-31 复现的三条结论必须留在文档里——防漂移（负面结果照账
    是仓库方法学的一部分，删掉它们等于改历史）。"""
    root = Path(__file__).resolve().parents[3]
    report = (root / "repro" / "ICML_revised_report.md").read_text(encoding="utf-8")
    edge = (root / "docs" / "THEORETICAL_EDGE.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    # The manuscript filename now contains only an archive/status stub; the
    # revised reproducibility report owns these measured-result claims.
    assert "0.398" in report and "0.3125" in report and "0.938" in report
    assert "结构性劣势" in report and "未复现" in report
    assert "追平" in report and "可修的工程缺陷" in report
    # THEORETICAL_EDGE：实测对照节 + 三条结论
    assert "实测对照（0.1-α 独立复现" in edge
    assert "结构性劣势" in edge and "engineering trade-off" in edge
    # README：是/不是区域的实测定位 + sidecar 联动
    assert "结构性劣势" in readme and "axis_probe" in readme

# ---------------------------------------------------------------------------
# 6. 多 seed + bootstrap CI（二期复现报告的统计门槛）
# ---------------------------------------------------------------------------
def test_bootstrap_ci_contract() -> None:
    from simplcore.core.eval.axis_probe import bootstrap_ci
    # 常量序列：CI 坍缩到该值（正确性）
    lo, hi = bootstrap_ci([0.5] * 5)
    assert lo == hi == 0.5
    # 分离序列：CI 包含均值且有序
    lo, hi = bootstrap_ci([0.0, 0.0, 0.0, 1.0, 1.0])
    assert 0.0 <= lo <= hi <= 1.0 and lo < hi


def test_run_probe_multiseed_structure() -> None:
    from simplcore.core.eval.axis_probe import run_probe_multiseed
    rep = run_probe_multiseed(tasks=["counting"], arms=["sgc"],
                              seeds=(0, 1, 2), steps=6, dim=16, layers=1,
                              batch=4, n_eval=16, n_boot=200)
    row = rep.rows["counting"]["sgc"]
    assert len(row["accs"]) == 3 and "ci95" in row and "mean" in row
    lo, hi = row["ci95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert "噪声内" in rep.note        # 判定纪律在账本里
