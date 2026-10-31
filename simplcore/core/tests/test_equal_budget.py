# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""等预算对照 harness（X-equal-budget）的守护测试。

覆盖六条对照语义：
1. FLOPs 账：TS 逐项公式精确可复算；attention 项随 s 线性、SC 亚线性；
2. TS 臂因果性：改未来 token 不影响过去位置的 logits（对照资格的根本前提）；
3. 参数对齐：match_ts_layers 与 SimplCore 非嵌入参数偏差 < 5%；
4. 窗口流确定性：同 seed 逐位相同、consumed_hash 可复算；
5. 端到端对照：两臂 token 恒等、loss 曲线下降、配对统计自洽、闸门诚实；
6. CLI 产出 schema 合法的 runs/eval_equal_token.json。
"""
import json
import math

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.equal_budget import (SyntheticWindowSource, TSConfig,
                                         TransformerBaseline, WindowStream,
                                         match_ts_layers, paired_eval,
                                         run_comparison,
                                         ts_non_embedding_params,
                                         ts_train_flops_per_token)
from simplcore.core.scaling import non_embedding_params, train_flops_per_token

SMOKE_KW = dict(token_budget=8000, lr=2e-3, eval_every_tokens=2000)


# ---------------------------------------------------------------------------
# 1. FLOPs 账
# ---------------------------------------------------------------------------
def test_ts_flops_formula_is_exact() -> None:
    """TS 每 token FLOPs = 6·N_nonemb + 12·s·D·L，逐项可复算。"""
    L, D, H, s = 6, 128, 512, 1024
    n = ts_non_embedding_params(L, D, H)
    assert n == L * (4 * D * D + 3 * D * H + 4 * D) + 2 * D
    assert ts_train_flops_per_token(L, D, H, s) == pytest.approx(
        6.0 * n + 12.0 * s * D * L, rel=1e-12)


def test_attention_term_linear_vs_sgc_sublinear() -> None:
    """s 翻倍：TS 每 token FLOPs 的增量 > SC 的增量（亚线性优势的账面体现）。"""
    cfg = SimplCoreConfig(vocab_size=64, dim=128, num_layers=6, num_bases=32,
                          ffn_ratio=4.0, tie_embeddings=False)
    for s1, s2 in ((1024, 2048), (4096, 8192)):
        d_ts = ts_train_flops_per_token(6, 128, 512, s2) - ts_train_flops_per_token(6, 128, 512, s1)
        d_sc = train_flops_per_token(cfg, s2) - train_flops_per_token(cfg, s1)
        assert d_ts > d_sc > 0, f"s: {s1}->{s2}: TS Δ={d_ts:.1f} SC Δ={d_sc:.1f}"


# ---------------------------------------------------------------------------
# 2. TS 臂因果性
# ---------------------------------------------------------------------------
def test_ts_baseline_is_causal() -> None:
    """改写窗口后半段的 token，前半段位置的 logits 必须逐位不变。"""
    torch.manual_seed(0)
    cfg = TSConfig(vocab_size=64, dim=32, num_layers=2, ffn_hidden=64,
                   num_heads=2)
    m = TransformerBaseline(cfg).eval()
    x = torch.randint(0, 64, (2, 32))
    with torch.no_grad():
        full = m(x, None)[0] if isinstance(m(x, None), tuple) else m(x, None)
    x2 = x.clone()
    x2[:, 16:] = torch.randint(0, 64, (2, 16))     # 只改未来
    with torch.no_grad():
        full2 = m(x2, None)[0] if isinstance(m(x2, None), tuple) else m(x2, None)
    torch.testing.assert_close(full[:, :16], full2[:, :16], rtol=0, atol=0)
    assert not torch.allclose(full[:, 16:], full2[:, 16:])   # 改动确实生效


# ---------------------------------------------------------------------------
# 3. 参数对齐
# ---------------------------------------------------------------------------
def test_match_ts_layers_aligns_params() -> None:
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                          ffn_ratio=2.0, tie_embeddings=False)
    sc_n = int(non_embedding_params(cfg))
    D, H = cfg.dim, int(cfg.ffn_hidden or cfg.ffn_ratio * cfg.dim)
    layers, ts_n = match_ts_layers(sc_n, D, H, cfg.num_layers)
    assert layers >= 1
    assert abs(ts_n - sc_n) / sc_n < 0.05


# ---------------------------------------------------------------------------
# 4. 窗口流确定性
# ---------------------------------------------------------------------------
def test_window_stream_deterministic_and_hashable() -> None:
    a, b = WindowStream(100, seed=42), WindowStream(100, seed=42)
    c = WindowStream(100, seed=43)
    sa, sb, sc = a.take(250), b.take(250), c.take(250)
    assert sa == sb                                # 同 seed 逐位相同
    assert sa != sc                                # 不同 seed 不同序列
    # 周期循环：250 = 2×100 + 50，第 200 个应等于第 0 个
    assert sa[200] == sa[0]
    assert a.consumed_hash() == b.consumed_hash()
    with pytest.raises(ValueError):
        WindowStream(0)


def test_synthetic_source_deterministic() -> None:
    s1 = SyntheticWindowSource(vocab_size=64, seq_len=32, n_windows=16, seed=7)
    s2 = SyntheticWindowSource(vocab_size=64, seq_len=32, n_windows=16, seed=7)
    s3 = SyntheticWindowSource(vocab_size=64, seq_len=32, n_windows=16, seed=8)
    assert torch.equal(s1.window(3), s2.window(3))
    assert not torch.equal(s1.window(3), s3.window(3))
    with pytest.raises(IndexError):
        s1.window(16)


# ---------------------------------------------------------------------------
# 5. 端到端对照（smoke 规模，秒级）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_report() -> dict:
    return run_comparison(preset="smoke", mode="equal_token", **SMOKE_KW)


def test_equal_token_budgets_identical(smoke_report: dict) -> None:
    g = smoke_report["gates"]["X-equal-budget"]
    assert smoke_report["mode"] == "equal_token"
    assert g["tokens_sc"] == g["tokens_ts"] > 0
    assert g["satisfied_by_construction"] is True
    assert g["passed"] is True
    # 等 FLOPs 参考行必须给出（另一种口径的账不能缺席）
    assert g["equal_flops_ts_tokens_reference"] > 0


def test_both_arms_learn(smoke_report: dict) -> None:
    for arm in ("simplcore", "transformer"):
        curve = smoke_report["arms"][arm]["loss_curve_head"]
        assert len(curve) >= 2
        assert curve[-1][1] < curve[0][1], f"{arm} loss did not decrease"


def test_paired_stats_self_consistent(smoke_report: dict) -> None:
    p = smoke_report["paired"]
    # 数学恒等：mean(diff) == mean(sc) - mean(ts)
    assert p["mean_loss_diff_sc_minus_ts"] == pytest.approx(
        p["mean_loss_simplcore"] - p["mean_loss_transformer"], abs=1e-9)
    # ppl 比值恒等式
    assert p["ppl_ratio_sc_over_ts"] == pytest.approx(
        math.exp(p["mean_loss_simplcore"] - p["mean_loss_transformer"]), rel=1e-9)
    assert p["sc_wins"] + p["ts_wins"] <= p["n_windows"]
    lo, hi = p["ci95_diff"]
    assert lo <= p["mean_loss_diff_sc_minus_ts"] <= hi
    assert "worst_window_for_sc" in p               # 最差 case 必须在报告里


def test_honesty_fields_consistent(smoke_report: dict) -> None:
    h = smoke_report["honesty"]
    gates = smoke_report["gates"]
    all_pass = all(g["passed"] for g in gates.values())
    assert h["unqualified_claim_permitted"] == all_pass
    fc = h["forbidden_checks"]
    assert all(fc.values())                          # 四条禁止项逐条核验
    assert "合成语料不可替代真实语料结论" in "".join(h["notes"])


def test_dry_run_refuses_claims() -> None:
    """没训练就没有数据支撑宣称 —— 诚实性字段必须关死。"""
    rep = run_comparison(preset="smoke", dry_run=True)
    assert rep["honesty"]["unqualified_claim_permitted"] is False
    assert rep["gates"]["X-heldout-ppl"]["passed"] is False
    assert rep["paired"]["n_windows"] == 0


def test_equal_flops_mode_uses_reference_budget() -> None:
    rep = run_comparison(preset="smoke", mode="equal_flops", **SMOKE_KW)
    g = rep["gates"]["X-equal-budget"]
    assert rep["mode"] == "equal_flops"
    # TS 预算按 FLOPs 账反算 → 实际 FLOPs 比值必须落在 1% 容差内
    assert abs(g["flops_ratio_ts_over_sc"] - 1.0) <= 0.01
    assert g["passed"] is True
    assert g["tokens_ts"] != g["tokens_sc"]          # token 数本来就应不同


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def test_cli_writes_valid_report(tmp_path) -> None:
    from simplcore.core.equal_budget import main
    out = tmp_path / "eval_equal_token.json"
    rc = main(["--preset", "smoke", "--tokens", "4000",
               "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["protocol"] == "X-equal-budget"
    assert rep["report_version"] == 1
    for key in ("budget", "arms", "paired", "gates", "honesty"):
        assert key in rep
    assert set(rep["gates"]) == {"X-equal-budget", "X-heldout-ppl"}


def test_paired_eval_on_frozen_models(tmp_path) -> None:
    """配对评估独立可调：固定两臂（不训练）时逐窗统计仍自洽。"""
    src = SyntheticWindowSource(vocab_size=64, seq_len=32, n_windows=64, seed=5)
    torch.manual_seed(0)
    ts = TransformerBaseline(TSConfig(vocab_size=64, dim=32, num_layers=1,
                                      ffn_hidden=64, num_heads=2))
    torch.manual_seed(1)
    sc_cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=1, num_bases=8,
                             ffn_ratio=2.0, tie_embeddings=False)
    from simplcore.core.model import SimplCoreModel
    sc = SimplCoreModel(sc_cfg)
    p = paired_eval(sc, ts, src, n_windows=32, n_boot=200)
    assert p["n_windows"] == 32
    assert p["sc_wins"] + p["ts_wins"] <= 32
    assert len(p["window_indices_head"]) == 16
