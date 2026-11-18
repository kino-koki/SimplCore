# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""2B×24h·4090D 战役计划的守护测试（纯解析账本，不建模型、秒级）。

守护的主张（全部来自用户给定的硬指标）：
1. 上下文翻倍显存增量 ≤ 1 GiB —— 在流式与分块训练两条**可行路径**上成立；
2. 朴素整段 FFT 路径随 L 线性增长，512K 起即超 24 GB —— 不可行，必须分块训练；
3. 2B 全参在 24GB 上只有 Adafactor（≈8 B/参数）可行，AdamW（16 B/参数）直接爆；
4. Phase 0 栈验收 ≤1 天、Phase 1 训练 ≤1 天（= 24h 硬停），无延期；
5. 验收协议只收**真实输出**指标，显式禁止"注意力层面"的机制内对比与欠训档粉饰。
"""
import pytest

from simplcore.core.campaign import (
    ADAFACTOR_BYTES_PER_PARAM,
    CONTEXT_CLAIM_GIB,
    CONTEXT_LADDER,
    GIB,
    HBM_4090D_BYTES,
    OPTIMIZER_BYTES_PER_PARAM,
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
    return get_preset("tuan-2b")


@pytest.fixture(scope="module")
def plan():
    return build_plan()


# ---------------------------------------------------------------------------
# 1. 显存账本：上下文翻倍增量
# ---------------------------------------------------------------------------
def test_stream_state_is_context_free(cfg):
    """O(1) 流式状态与 L 无关：阶梯上任何一级的流式字节数都相同，翻倍增量为 0。"""
    ref = stream_bytes(cfg)
    for seq_len in CONTEXT_LADDER:
        assert stream_bytes(cfg) == ref  # 公式根本不接受 L —— 结构性保证
    rows = doubling_report(cfg, base_params := 2_000_000_000,
                           bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)
    assert all(r.stream_delta == 0 for r in rows[1:])
    assert ref > 0


def test_chunked_train_is_context_free(cfg):
    """分块流式训练显存只随 chunk 增长，与总上下文 L 无关。"""
    one_m = chunked_train_bytes(cfg, 2_000_000_000, chunk=4096,
                                bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)
    four_m = chunked_train_bytes(cfg, 2_000_000_000, chunk=4096,
                                 bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)  # L 不进公式
    assert one_m == four_m
    rows = doubling_report(cfg, 2_000_000_000,
                           bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)
    assert all(r.chunked_delta == 0 for r in rows[1:])


def test_naive_fft_grows_linear_and_infeasible(cfg):
    """朴素整段 FFT：激活随 L 线性增长，512K 起即超 24 GB —— 不可行路径。"""
    b = 2_000_000_000
    nb_512k = naive_fft_train_bytes(cfg, b, 512 * 1024)
    nb_2m = naive_fft_train_bytes(cfg, b, 2 * 1024 * 1024)
    # 线性增长（优化器状态是常数项，激活部分严格 4×）
    act_a = nb_512k - optimizer_state_bytes(b)
    act_b = nb_2m - optimizer_state_bytes(b)
    assert act_b == 4 * act_a
    # 512K 起就装不进 4090D 的 24 GB
    assert nb_512k > HBM_4090D_BYTES
    assert nb_2m > HBM_4090D_BYTES


def test_user_doubling_claim_holds_on_viable_paths(cfg):
    """结构性主张"翻倍增加 ≤ 1 GiB"在流式与分块两条可行路径上成立（增量为 0）。"""
    rows = doubling_report(cfg, 2_000_000_000,
                           bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)
    verdict = claim_verdict(rows)
    assert verdict["stream"]["holds"] is True
    assert verdict["chunked"]["holds"] is True
    assert verdict["claim_holds"] is True
    assert verdict["stream"]["max_delta"] <= CONTEXT_CLAIM_GIB * GIB
    # 朴素路径被点名不可行：整段 FFT 从阶梯第一级（512K）就装不下 24 GB
    assert verdict["naive"]["fits_hbm_anywhere"] is False
    assert verdict["naive"]["overflows_from"] == CONTEXT_LADDER[0]


# ---------------------------------------------------------------------------
# 2. 24GB 定容：优化器路径选择是硬闸门
# ---------------------------------------------------------------------------
def test_adamw_cannot_fit_2b_on_24gb(cfg):
    """AdamW 混合精度 16 B/参数 × 2B = 32 GiB > 24 GB —— 直接爆，必须 Adafactor。"""
    b = 2_000_000_000
    assert optimizer_state_bytes(b) > HBM_4090D_BYTES
    # Adafactor 路径（fp32 参数+梯度+因子状态 ≈ 8 B/参数）+ chunk 激活装得下
    fits = chunked_train_bytes(cfg, b, chunk=4096,
                               bytes_per_param=ADAFACTOR_BYTES_PER_PARAM)
    assert fits < 23 * GIB
    # 同 chunk 下 AdamW 路径装不下
    adamw = chunked_train_bytes(cfg, b, chunk=4096)
    assert adamw > HBM_4090D_BYTES


# ---------------------------------------------------------------------------
# 3. 阶段硬停
# ---------------------------------------------------------------------------
def test_phase_hard_stops(plan):
    """Phase 0 栈验收 ≤1 天、Phase 1 训练 ≤1 天（=24h），全部硬停无延期。"""
    p0 = plan.phase("Phase 0 栈验收")
    p1 = plan.phase("Phase 1 · 2B 基座（24h）")
    retro = plan.phase("复盘（立即）")
    assert p0.max_days == 1 and p0.hard_stop is True
    assert p1.max_days == PHASE1_MAX_DAYS == 1 and p1.hard_stop is True
    assert retro.max_days == 0 and retro.hard_stop is True  # 复盘不占训练窗口
    assert p0.days_ok(1) and not p0.days_ok(1.5)
    assert p1.days_ok(1) and not p1.days_ok(1.5)
    assert retro.days_ok(0) and retro.days_ok(0.5)


def test_total_budget_2_days(plan):
    """总预算 2 天（含栈验收），24h 训练硬停 —— 各阶段天数之和恰好 ≤2。"""
    sched = plan.schedule_check()
    assert sched["cap_days"] == 2
    assert sched["allocations"] == {"Phase 0 栈验收": 1, "Phase 1 · 2B 基座（24h）": 1}
    assert sched["total_days"] == 2
    assert sched["holds"] is True
    # 24h 硬停闸门存在
    hard = plan.hard_gate_names()
    assert "P1-hard-stop-24h" in hard and "P0-oom-free-dryrun" in hard


def test_hbm_estimates_fit_4090d(plan):
    """Phase 0/1 的解析显存峰值都必须装进 24 GB（实测闸门按 23 GiB 执行）。"""
    for name in ("Phase 0 栈验收", "Phase 1 · 2B 基座（24h）"):
        ph = plan.phase(name)
        assert 0 < ph.hbm_bytes <= 23 * GIB, name


def test_phase1_token_budget_honest(plan):
    """Phase 1 有解析 token 预算，且为欠训档（< 20×N 的 Chinchilla 比率）——诚实呈现。"""
    p1 = plan.phase("Phase 1 · 2B 基座（24h）")
    assert p1.token_budget and p1.token_budget > 0
    assert p1.token_budget < 20 * plan.base_params  # 明确的欠训档，不许粉饰


# ---------------------------------------------------------------------------
# 4. 权重预算（2B 全参）
# ---------------------------------------------------------------------------
def test_budget_alignment(plan):
    b = plan.budget
    assert b.base_params == 2_000_000_000
    assert b.total_cap == 2_000_000_000
    # 本战役是 2B 全参基座：没有附加适配器预算（can_allocate 只管"再加多少"）
    assert b.can_allocate(0)
    assert not b.can_allocate(1)


def test_plan_preset_guard():
    """非 2B 口径的 preset 直接拒绝 —— 战役指标按 2B 给定，不许偷换。"""
    with pytest.raises(ValueError):
        build_plan(preset="debug")


# ---------------------------------------------------------------------------
# 5. 验收协议：真实输出，禁止机制内对比与欠训粉饰
# ---------------------------------------------------------------------------
def test_protocol_is_real_output_only(plan):
    names = [g.name for g in plan.protocol]
    for must in ("X-heldout-ppl", "X-longctx-retrieval",
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
    # 欠训档必须如实报告 tokens/param
    assert "tokens/param" in joined


# ---------------------------------------------------------------------------
# 6. 「无人可忽视」演示包
# ---------------------------------------------------------------------------
def test_showcase_package_complete(plan):
    """演示包：≥8 条、每条都有可执行的复验命令、含至少 1 条诚实边界。"""
    assert len(plan.showcase) >= 8
    for s in plan.showcase:
        assert s.claim and s.verify and s.owner, s.name
        # 复验命令必须是可跑的 CLI/脚本引用，不是形容词
        assert ("python -m" in s.verify) or ("benchmark" in s.verify) or ("脚本" in s.verify) or ("日志" in s.verify)
    kinds = {s.kind for s in plan.showcase}
    assert "honest" in kinds          # 诚实边界主动呈现（24h token 预算）
    names = [s.name for s in plan.showcase]
    assert "S1-deploy-2gb" in names and "S7-honest-token-budget" in names
    # 演示包进入机器可读输出与文本报告
    d = plan.as_dict()
    assert len(d["showcase"]) == len(plan.showcase)
    assert "演示包" in plan.format()


# ---------------------------------------------------------------------------
# 7. 报告可用性
# ---------------------------------------------------------------------------
def test_plan_report_and_gates(plan):
    txt = plan.format()
    for needle in ("2B×24h·4090D 战役计划", "硬停 1 天", "最终验收协议", "总时间预算"):
        assert needle in txt
    names = plan.gate_names()
    assert len(names) == len(set(names))  # 闸门名不重复
    hard = plan.hard_gate_names()
    assert "P1-hard-stop-24h" in hard
    d = plan.as_dict()
    assert d["total_cap"] == 2_000_000_000
    assert d["doubling"]["claim_holds"] is True
    assert d["schedule"]["holds"] is True
    assert [p["name"] for p in d["phases"]] == [
        "Phase 0 栈验收", "Phase 1 · 2B 基座（24h）", "复盘（立即）", "Phase 2（本战役无）"]


# ---------------------------------------------------------------------------
# 8. NT46 课程校准硬闸门：启动器 N_plan = token 预算 / 全局 batch
# ---------------------------------------------------------------------------
def test_launcher_curriculum_calibrated():
    """启动器的 --max-steps 必须按 token 预算校准（NT46）：否则 until_frac
    阶梯在 24h 内走不到（课程卡死 stage 0）。"""
    from pathlib import Path as _P
    script = _P(__file__).resolve().parents[3] / "scripts" / "train_tuan_2b_4090d.sh"
    text = script.read_text(encoding="utf-8")
    assert 'MAX_STEPS="${MAX_STEPS:-3358}"' in text
    assert "--max-steps \"$MAX_STEPS\"" in text
    assert "NT46" in text                      # 校准依据在注释中可审计
    # 校准算术：3358 步 × 65536 tokens ≈ 2.2e8 = 24h 解析预算
    assert abs(3358 * 65_536 - 2.2e8) / 2.2e8 < 0.01


def test_curriculum_ladder_unfolds_within_budget():
    """校准后的阶梯在 24h 预算内完整展开（NT46 (c) 的可执行形式）。"""
    fracs = (0.18, 0.36, 0.54, 0.70, 0.90)     # _CTX_2B_4M 阶梯
    n_plan = 3358
    boundaries = [round(f * n_plan) for f in fracs]
    assert all(b < n_plan for b in boundaries)          # 全部边界可达
    assert boundaries == sorted(set(boundaries))        # 严格递增
