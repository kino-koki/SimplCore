# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""长上下文检索评测（X-longctx-retrieval + X-stream-consistency）的守护测试。

覆盖五条评测语义：
1. needle 语料：确定性、needle 落点正确、key→value 映射全语料一致；
2. 流式==整段：同一模型两模式 argmax 逐 trial 相同、logit 差在容差内
   （X-stream-consistency 的实质，不是装饰字段）；
3. 账本：TS KV cache 公式、放不下如实标记不可跑（不跑假对照）；
4. 端到端报告：schema、闸门、near-chance 诚实警告、S7 信息论边界行；
5. 学习冒烟：needle 结构上 CE 可观测下降（key→value 是序列最强结构）。
"""
import json
import math

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.equal_budget import TransformerBaseline, TSConfig
from simplcore.core.longctx_eval import (NeedleCorpus, NeedleTrial,
                                         run_longctx_eval, score_trial,
                                         _ts_kv_bytes)
from simplcore.core.model import SimplCoreModel
from simplcore.core.scaling import stream_state_bytes

CORPUS_KW = dict(n_keys=4, n_values=8, needles_per_seq=4)


# ---------------------------------------------------------------------------
# 1. needle 语料
# ---------------------------------------------------------------------------
def test_corpus_deterministic_and_consistent() -> None:
    c1 = NeedleCorpus(128, seed=9, **CORPUS_KW)
    c2 = NeedleCorpus(128, seed=9, **CORPUS_KW)
    c3 = NeedleCorpus(128, seed=10, **CORPUS_KW)
    t1, t2, t3 = c1.make_trial(0, 0.5), c2.make_trial(0, 0.5), c3.make_trial(0, 0.5)
    assert torch.equal(t1.tokens, t2.tokens)
    assert not torch.equal(t1.tokens, t3.tokens)
    assert t1.tokens.shape[0] == 128              # 定长
    # key→value 映射与 seed 绑定（全语料一致 —— 否则任务无解）
    assert c1.value_token((9 + 2) % 8) == t1.value_token or t1.key_id != 2


def test_needle_placed_at_requested_depth() -> None:
    c = NeedleCorpus(256, seed=3, **CORPUS_KW)
    tr = c.make_trial(0, 0.8)
    k1, k2 = c.key_tokens(0)
    # 查询后缀是末 4 token：[OPEN, k1, k2, CLOSE]
    assert tr.tokens[-4:].tolist() == [0, k1, k2, 1]
    assert tr.value_token == c.value_token((3 + 0) % 8)
    # needle 块在序列里真实存在
    needle = [0, k1, k2, 1, tr.value_token]
    toks = tr.tokens.tolist()
    hits = [i for i in range(len(toks) - 4) if toks[i:i + 5] == needle]
    assert len(hits) == 1                          # 每 needle 恰好出现一次
    pos = hits[0]
    assert 0.6 <= pos / (256 - 9) <= 1.0           # depth=0.8 的落点在深度带内


def test_depth_bounds_enforced() -> None:
    c = NeedleCorpus(128, **CORPUS_KW)
    with pytest.raises(ValueError):
        c.make_trial(0, 0.0)
    with pytest.raises(ValueError):
        c.make_trial(0, 1.0)
    with pytest.raises(ValueError):
        NeedleCorpus(32)                            # seq_len < 64


# ---------------------------------------------------------------------------
# 2. 流式==整段
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_models():
    torch.manual_seed(0)
    c = NeedleCorpus(128, seed=2026, **CORPUS_KW)
    cfg = SimplCoreConfig(vocab_size=c.vocab_size, dim=32, num_layers=2,
                          num_bases=8, ffn_ratio=2.0, local_kernel=4,
                          decay_max=256.0, tie_embeddings=False)
    sc = SimplCoreModel(cfg).eval()
    torch.manual_seed(1)
    ts = TransformerBaseline(TSConfig(vocab_size=c.vocab_size, dim=32,
                                      num_layers=2, ffn_hidden=64,
                                      num_heads=2)).eval()
    return c, cfg, sc, ts


def test_streaming_equals_parallel_logits(tiny_models) -> None:
    """X-stream-consistency 的实质：同一 trial 两模式 argmax 相同、logit 差 < 4e-5。"""
    c, cfg, sc, _ = tiny_models
    for i in range(3):
        tr = c.make_trial(i, 0.1 + 0.2 * i)
        r_stream = score_trial(sc, tr, mode="streaming")
        r_par = score_trial(sc, tr, mode="parallel")
        assert r_stream["argmax"] == r_par["argmax"]
        assert r_stream["within_tol"] is True
        assert r_stream["max_logit_diff"] < 4e-5
        assert r_stream["value_nll"] == pytest.approx(r_par["value_nll"], abs=1e-5)


def test_ts_arm_causal_on_long_context(tiny_models) -> None:
    """TS 臂前缀不变性：同一前缀在两种延续长度下，前缀位置 logits 逐位相同
    （长上下文因果性的正确测法 —— 只能扰动"未来"，不能扰动过去）。"""
    c, _, _, ts = tiny_models
    tr = c.make_trial(0, 0.9)                      # needle 都放在前缀之后
    x_full = tr.tokens.unsqueeze(0)                # (1, L)
    x_prefix = x_full[:, :64]                      # (1, 64)
    with torch.no_grad():
        r1 = ts(x_prefix, None)
        l1 = (r1[0] if isinstance(r1, tuple) else r1)
        r2 = ts(x_full, None)
        l2 = (r2[0] if isinstance(r2, tuple) else r2)
    torch.testing.assert_close(l1[:, :64], l2[:, :64], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3. 账本
# ---------------------------------------------------------------------------
def test_kv_bytes_formula() -> None:
    assert _ts_kv_bytes(2, 32, 256) == 2 * 2 * 256 * 32 * 2     # K+V×L×D×bf16
    assert _ts_kv_bytes(12, 2048, 1048576) == 2 * 12 * 1048576 * 2048 * 2


def test_ledger_only_reports_and_fails_gates() -> None:
    """ledger_only：不跑模型，闸门必须 FAIL（没有数据就没有 PASS），账面照报。"""
    rep = run_longctx_eval(preset="smoke", lengths=(256, 8192),
                           trials_per_depth=1, ledger_only=True)
    assert all(not r.get("evaluated") for r in rep["lengths"])
    assert rep["gates"]["X-longctx-retrieval"]["passed"] is False
    assert rep["gates"]["X-stream-consistency"]["passed"] is False
    assert rep["honesty"]["unqualified_claim_permitted"] is False
    # S7 信息论行永远在：SC 状态字节必须与 scaling 单一事实源一致
    cfg = SimplCoreConfig(vocab_size=NeedleCorpus(256, **CORPUS_KW).vocab_size,
                          dim=32, num_layers=2, num_bases=8, ffn_ratio=2.0,
                          local_kernel=4, decay_max=512.0, tie_embeddings=False)
    row = rep["lengths"][0]
    assert row["simplcore_stream_state_bytes"] == stream_state_bytes(cfg)


def test_ts_infeasible_at_1b_1m() -> None:
    """1B 形态 1M：TS KV(同形 bf16) > 80GiB → 不可跑，SC 状态 = 30.4 MiB 量级。"""
    rep = run_longctx_eval(preset="1b", lengths=(1048576,), ledger_only=True)
    row = rep["lengths"][0]
    assert row["ts_runnable_on_device"] is False
    assert row["ts_kv_cache_bytes"] > 80 * (1 << 30)
    assert 30 * (1 << 20) <= row["simplcore_stream_state_bytes"] <= 32 * (1 << 20)


# ---------------------------------------------------------------------------
# 4. 端到端实跑报告
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e2e_report() -> dict:
    return run_longctx_eval(preset="smoke", lengths=(256,), trials_per_depth=1,
                            seed=2026)


def test_e2e_report_schema(e2e_report: dict) -> None:
    assert e2e_report["protocol"] == "X-longctx-retrieval"
    assert e2e_report["report_version"] == 1
    row = e2e_report["lengths"][0]
    assert row["evaluated"] is True
    for key in ("simplcore", "transformer", "stream_consistency"):
        assert key in row
    sc = row["simplcore"]
    assert sc["n_trials"] == 5                     # 5 depths × 1
    assert 0.0 <= sc["accuracy"] <= 1.0
    assert set(sc["per_depth_accuracy"]) == {"0.10", "0.30", "0.50", "0.70", "0.90"}
    # X-stream-consistency：SC 流式与整段逐 trial 一致
    assert row["stream_consistency"]["X-stream-consistency_passed"] is True
    assert row["stream_consistency"]["max_logit_diff"] < 4e-5


def test_e2e_honesty(e2e_report: dict) -> None:
    h = e2e_report["honesty"]
    row = e2e_report["lengths"][0]
    # 未训练模型在 chance 附近 → near_chance_warning 必须亮、不许宣称
    assert e2e_report["gates"]["X-longctx-retrieval"]["near_chance_warning"] is True
    assert h["unqualified_claim_permitted"] is False
    assert all(h["forbidden_checks"].values())
    s7 = h["S7_honest_needle"]["rows"][0]
    assert s7["seq_len"] == 256
    assert s7["ts_kv_over_sc_state"] > 0
    assert s7["needle_original_bytes"] == 5 * 4 * 2


# ---------------------------------------------------------------------------
# 5. 学习冒烟
# ---------------------------------------------------------------------------
def test_needle_structure_learnable() -> None:
    """needle 任务在 CE 下可学：loss 下降（key→value 是序列中最强结构）。"""
    c = NeedleCorpus(128, seed=11, **CORPUS_KW)
    cfg = SimplCoreConfig(vocab_size=c.vocab_size, dim=32, num_layers=2,
                          num_bases=8, ffn_ratio=2.0, local_kernel=4,
                          decay_max=256.0, tie_embeddings=False)
    torch.manual_seed(0)
    m = SimplCoreModel(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    trials = [c.make_trial(i, 0.1 + 0.2 * (i % 5)) for i in range(8)]
    xs = torch.stack([t.tokens for t in trials])

    def loss_of() -> float:
        m.eval()
        with torch.no_grad():
            _, loss = m(xs, xs)                    # 全序列 CE（value 也在其中）
        m.train()
        return float(loss)

    first = loss_of()
    for _ in range(30):
        _, loss = m(xs, xs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    last = loss_of()
    assert last < first - 0.1, f"loss {first:.4f} -> {last:.4f} did not decrease"


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def test_cli(tmp_path) -> None:
    from simplcore.core.longctx_eval import main
    out = tmp_path / "eval_longctx.json"
    assert main(["--preset", "smoke", "--lengths", "256",
                 "--trials-per-depth", "1", "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["protocol"] == "X-longctx-retrieval"
    assert set(rep["gates"]) == {"X-longctx-retrieval", "X-stream-consistency"}
    out2 = tmp_path / "ledger.json"
    assert main(["--preset", "1b", "--lengths", "1048576",
                 "--ledger-only", "--out", str(out2)]) == 0
    rep2 = json.loads(out2.read_text(encoding="utf-8"))
    assert rep2["lengths"][0]["evaluated"] is False
