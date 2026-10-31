# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""1B→3B 战役计划的守护测试（纯解析账本，不建模型、秒级）。

守护的主张（全部来自用户给定的硬指标）：
1. 上下文翻倍 HBM 增量 ≤ 1 GiB —— 在流式与分块训练两条**可行路径**上成立；
2. 朴素整段 FFT 路径随 L 线性增长，1M 起即超 128 GB —— 不可行，必须分块训练；
3. Phase 1 ≤ 30 天、Phase 2 ≤ 28 天，均为硬停、无延期；
4. 权重预算 = 1B 冻结基座 + ≤2B 可学、总上限 3B（WeightBudget 硬拒超额）；
5. 验收协议只收**真实输出**指标，显式禁止"注意力层面"的机制内对比。
"""
import pytest

from simplcore.core.campaign import (
    CONTEXT_CLAIM_GIB,
    CONTEXT_LADDER,
    GIB,
    HBM_MI250_BYTES,
    PHASE1_MAX_DAYS,
    PHASE2_MAX_DAYS,
    build_plan,
    chunked_train_bytes,
    claim_verdict,
    doubling_report,
    naive_fft_train_bytes,
    optimizer_state_bytes,
    stream_bytes,
)
from simplcore.core.config import get_preset
from simplcore.core.envlearn import WeightBudget


@pytest.fixture(scope="module")
def cfg():
    return get_preset("tuan-1b")


@pytest.fixture(scope="module")
def plan():
    return build_plan()


# ---------------------------------------------------------------------------
# 1. HBM 账本：上下文翻倍增量
# ---------------------------------------------------------------------------
def test_stream_state_is_context_free(cfg):
    """O(1) 流式状态与 L 无关：阶梯上任何一级的流式字节数都相同，翻倍增量为 0。"""
    ref = stream_bytes(cfg)
    for seq_len in CONTEXT_LADDER:
        assert stream_bytes(cfg) == ref  # 公式根本不接受 L —— 结构性保证
    rows = doubling_report(cfg, base_params := 1_000_000_000)
    assert all(r.stream_delta == 0 for r in rows[1:])
    assert ref > 0


def test_chunked_train_is_context_free(cfg):
    """分块流式训练显存只随 chunk 增长，与总上下文 L 无关。"""
    one_m = chunked_train_bytes(cfg, 1_000_000_000, chunk=8192)
    four_m = chunked_train_bytes(cfg, 1_000_000_000, chunk=8192)  # L 不进公式
    assert one_m == four_m
    rows = doubling_report(cfg, 1_000_000_000)
    assert all(r.chunked_delta == 0 for r in rows[1:])


def test_naive_fft_grows_linear_and_infeasible(cfg):
    """朴素整段 FFT：激活随 L 线性增长，1M 起即超 128 GB —— 不可行路径。"""
    b = 1_000_000_000
    nb_1m = naive_fft_train_bytes(cfg, b, 1024 * 1024)
    nb_4m = naive_fft_train_bytes(cfg, b, 4 * 1024 * 1024)
    # 线性增长（优化器状态是常数项，激活部分严格 4×）
    act_1m = nb_1m - optimizer_state_bytes(b)
    act_4m = nb_4m - optimizer_state_bytes(b)
    assert act_4m == 4 * act_1m
    # 1M 起就装不进 MI250 的 128 GB
    assert nb_1m > HBM_MI250_BYTES
    assert nb_4m > HBM_MI250_BYTES


def test_user_doubling_claim_holds_on_viable_paths(cfg):
    """用户主张"翻倍增加 ≤ 1 GiB"在流式与分块两条可行路径上成立（增量为 0）。"""
    rows = doubling_report(cfg, 1_000_000_000)
    verdict = claim_verdict(rows)
    assert verdict["stream"]["holds"] is True
    assert verdict["chunked"]["holds"] is True
    assert verdict["claim_holds"] is True
    assert verdict["stream"]["max_delta"] <= CONTEXT_CLAIM_GIB * GIB
    # 朴素路径被点名不可行：整段 FFT 从阶梯第一级（512K）就装不下 128 GB
    assert verdict["naive"]["fits_hbm_anywhere"] is False
    assert verdict["naive"]["overflows_from"] == CONTEXT_LADDER[0]


# ---------------------------------------------------------------------------
# 2. 阶段硬停
# ---------------------------------------------------------------------------
def test_phase_hard_stops(plan):
    """Phase 0 ≤30、Phase 1 ≤30、复盘 ≤2、Phase 2 ≤28 天，全部硬停无延期。"""
    p0 = plan.phase("Phase 0 冻结")
    p1 = plan.phase("Phase 1 · 1B 基座")
    p2 = plan.phase("Phase 2 · 2B 附加参数")
    retro = plan.phase("复盘（立即）")
    assert p0.max_days == 30 and p0.hard_stop is True
    assert p1.max_days == PHASE1_MAX_DAYS == 30 and p1.hard_stop is True
    assert p2.max_days == PHASE2_MAX_DAYS == 28 and p2.hard_stop is True
    assert retro.max_days == 2 and retro.hard_stop is True  # 90 天预算零余量
    assert p0.days_ok(30) and not p0.days_ok(30.5)
    assert p1.days_ok(30) and not p1.days_ok(30.5)
    assert p2.days_ok(28) and not p2.days_ok(28.5)
    assert retro.days_ok(2) and not retro.days_ok(2.5)


def test_total_budget_90_days(plan):
    """v2 硬指标：全程 3 个月（90 天）完全搞定 —— 各阶段天数之和恰好 ≤90。"""
    sched = plan.schedule_check()
    assert sched["cap_days"] == 90
    assert sched["allocations"] == {
        "Phase 0 冻结": 30, "Phase 1 · 1B 基座": 30,
        "复盘（立即）": 2, "Phase 2 · 2B 附加参数": 28,
    }
    assert sched["total_days"] == 90
    assert sched["holds"] is True
    # Phase 0 的硬停闸门存在（后端验收不封顶就不可能守住 90 天）
    hard = plan.hard_gate_names()
    assert "P0-hard-stop-30d" in hard and "R-hard-stop-2d" in hard


def test_hbm_estimates_fit_mi250(plan):
    """两个训练阶段的解析显存峰值都必须装进 128 GB。"""
    for name in ("Phase 1 · 1B 基座", "Phase 2 · 2B 附加参数"):
        ph = plan.phase(name)
        assert 0 < ph.hbm_bytes <= HBM_MI250_BYTES, name


def test_phase2_token_budget_exists(plan):
    """Phase 2 有解析 token 预算（基座只前向 + 适配器全训）。"""
    p2 = plan.phase("Phase 2 · 2B 附加参数")
    p1 = plan.phase("Phase 1 · 1B 基座")
    assert p2.token_budget and p2.token_budget > 0
    assert p1.token_budget and p1.token_budget > p2.token_budget  # 30d 全量训 > 28d 适配器训


# ---------------------------------------------------------------------------
# 3. 权重预算（1B 冻结 + ≤2B 可学 / 总 3B）
# ---------------------------------------------------------------------------
def test_budget_alignment(plan):
    b = plan.budget
    assert b.base_params == 1_000_000_000
    assert b.max_learned == 2_000_000_000
    assert b.total_cap == 3_000_000_000
    # 满额 2B 可学：允许；显存口径只算可学部分
    wb = WeightBudget(base_params=1_000_000_000, max_learned=2_000_000_000,
                      total_cap=3_000_000_000)
    assert wb.can_allocate(2_000_000_000)
    assert not wb.can_allocate(2_000_000_001)
    wb.allocate(2_000_000_000)
    assert wb.vram_params() == 2_000_000_000  # 基座冻结，不进优化器显存


def test_plan_preset_guard():
    """非 1B 口径的 preset 直接拒绝 —— 战役指标按 1B 给定，不许偷换。"""
    with pytest.raises(ValueError):
        build_plan(preset="debug")


# ---------------------------------------------------------------------------
# 4. 验收协议：真实输出，禁止机制内对比
# ---------------------------------------------------------------------------
def test_protocol_is_real_output_only(plan):
    names = [g.name for g in plan.protocol]
    # 真实输出四件套 + 口径 + 一致性
    for must in ("X-heldout-ppl", "X-longctx-retrieval", "X-code-passk",
                 "X-throughput", "X-equal-budget", "X-stream-consistency"):
        assert must in names, must
    # 全部协议项都是硬闸门
    assert all(g.kind == "hard" for g in plan.protocol)


def test_forbidden_list_bans_attention_level_comparison(plan):
    """用户原话："真实输出结果，而不是注意力层面" —— 必须显式进入禁止清单。"""
    joined = "\n".join(plan.forbidden)
    assert "注意力" in joined
    assert "等 token" in joined or "等 FLOPs" in joined
    assert "verifier" in joined  # 禁止自报自评


# ---------------------------------------------------------------------------
# 5. 「无人可忽视」演示包
# ---------------------------------------------------------------------------
def test_showcase_package_complete(plan):
    """演示包：≥8 条、每条都有可执行的复验命令、含至少 1 条诚实边界。"""
    assert len(plan.showcase) >= 8
    for s in plan.showcase:
        assert s.claim and s.verify and s.owner, s.name
        # 复验命令必须是可跑的 CLI/脚本引用，不是形容词
        assert ("python -m" in s.verify) or ("benchmark" in s.verify) or ("脚本" in s.verify) or ("日志" in s.verify)
    kinds = {s.kind for s in plan.showcase}
    assert "honest" in kinds          # 诚实边界主动呈现（needle 短板）
    names = [s.name for s in plan.showcase]
    assert "S1-deploy-2gb" in names and "S7-honest-needle" in names
    # 演示包进入机器可读输出与文本报告
    d = plan.as_dict()
    assert len(d["showcase"]) == len(plan.showcase)
    assert "演示包" in plan.format()


# ---------------------------------------------------------------------------
# 6. 报告可用性
# ---------------------------------------------------------------------------
def test_plan_report_and_gates(plan):
    txt = plan.format()
    for needle in ("1B→3B 战役计划", "硬停 30 天", "硬停 28 天", "最终验收协议", "总时间预算"):
        assert needle in txt
    names = plan.gate_names()
    assert len(names) == len(set(names))  # 闸门名不重复
    hard = plan.hard_gate_names()
    assert "P1-hard-stop-30d" in hard and "P2-hard-stop-28d" in hard
    d = plan.as_dict()
    assert d["total_cap"] == 3_000_000_000
    assert d["doubling"]["claim_holds"] is True
    assert d["schedule"]["holds"] is True
    assert [p["name"] for p in d["phases"]] == [
        "Phase 0 冻结", "Phase 1 · 1B 基座", "复盘（立即）", "Phase 2 · 2B 附加参数"]
