# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""自进化链的守护测试：反思产出真实信号 / 记忆可召回 / gate 判定可信。

这三件事此前都是"接口对、逻辑空"的状态，测试只校验形状所以全绿：

1. ``default_reflector`` 把轨迹打标记而不是产出可训练的修订，于是
   ``synthesized`` 恒为 0 —— 自进化在跑，但学不到任何东西；
2. ``_write_memory`` 用一个标量填满整条向量当键，所有记忆共线（余弦恒 1.0），
   检索退化成"随便返回一条"，且内置 ``hash()`` 带进程随机盐；
3. ``no-regression gate`` 比两个标量点估计，在小评测集上被噪声主导，
   而且默认验证器恒返回 0.0 时 gate 恒通过 —— 等于没有 gate。

本文件逐条钉死这些行为。
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from simplcore.core.dream import (
    DreamConfig,
    DreamingLoop,
    DreamStage,
    EvalScores,
    ExperienceBuffer,
    Insight,
    Trajectory,
    default_reflector,
    default_synthesizer,
    paired_bootstrap_ci,
    regression_gate,
)
from simplcore.core.memory import CompressedMemoryBank, TextKeyEncoder


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
EVIDENCE = ("SGC 的流式状态是 2·M·D。推理期用二阶递推，与序列长度无关。"
            "实测吞吐在 8k 上下文下提升 1.86 倍。")

GROUNDED_RESPONSE = "SGC 的流式状态是 2·M·D。实测吞吐在 8k 上下文下提升 1.86 倍。"
TAMPERED_RESPONSE = "SGC 的流式状态是 2·M·D。实测吞吐在 8k 上下文下提升 2.86 倍。"
FABRICATED_RESPONSE = "SGC 的流式状态是 2·M·D。该架构已在三家芯片厂量产。"

FILLER = "这是与本问题无关的填充句，内容涉及天气、股票、菜谱等随机话题。"


def _traj(response=TAMPERED_RESPONSE, prompt="SGC 的流式状态与吞吐？",
          context=EVIDENCE, **kw) -> Trajectory:
    meta = kw.pop("meta", {})
    if context is not None:
        meta.setdefault("context", context)
    return Trajectory(prompt=prompt, response=response, meta=meta, **kw)


# ===========================================================================
# 1. 反思器：从外部证据产出可训练的修订
# ===========================================================================
def test_reflector_repairs_tampered_number_with_evidence_verbatim() -> None:
    """数字篡改能被发现，且 ``repair`` 是**证据原文**而不是模型改写。"""
    ins = default_reflector(None, [_traj()])
    assert len(ins) == 1
    got = ins[0]
    assert got.kind == "partially_supported"
    assert got.repair == "实测吞吐在 8k 上下文下提升 1.86 倍"
    assert got.evidence == got.repair
    assert got.meta["numbers_ok"] is False
    assert "2.86" in got.meta["claim"]


def test_reflector_reports_fabrication_without_repair() -> None:
    """证据里根本没有的句子：标记为 unsupported，且**不给** repair。

    纯编造的句子没有原文可引用，"修好它"只能靠模型自己编 —— 那正是要防的，
    所以这里必须留 repair=None，让合成器拒收。
    """
    ins = default_reflector(None, [_traj(response=FABRICATED_RESPONSE)])
    assert len(ins) == 1
    assert ins[0].kind == "unsupported"
    assert ins[0].repair is None
    assert "三家芯片厂" in ins[0].meta["claim"]
    assert default_synthesizer(None, ins) == []


def test_reflector_skips_fully_grounded_response() -> None:
    """全部可归因 → 没有可反思项（不给自进化制造噪声）。"""
    assert default_reflector(None, [_traj(response=GROUNDED_RESPONSE)]) == []


def test_reflector_without_evidence_never_fabricates_repair() -> None:
    """没有外部证据时只能标记 ``no_evidence``，绝不能凭空产出修订目标。"""
    ins = default_reflector(None, [_traj(context=None, grounded=False, confidence=0.2)])
    assert len(ins) == 1
    assert ins[0].kind == "no_evidence"
    assert ins[0].repair is None
    assert ins[0].evidence is None


def test_reflector_reads_retrieval_hits() -> None:
    """``meta['hits']``（检索命中列表）与 ``meta['context']`` 等价可用。"""
    traj = Trajectory(
        prompt="吞吐提升多少？",
        response="实测吞吐在 8k 上下文下提升 2.86 倍。",
        meta={"hits": [{"text": "实测吞吐在 8k 上下文下提升 1.86 倍。",
                        "source": "doc#3"}]},
    )
    ins = default_reflector(None, [traj])
    assert len(ins) == 1
    assert ins[0].repair == "实测吞吐在 8k 上下文下提升 1.86 倍"
    assert ins[0].meta["evidence_source"] == "doc#3"


# ===========================================================================
# 2. 合成器：只接受带证据原文的洞见
# ===========================================================================
def test_synthesizer_rejects_insights_without_repair() -> None:
    traj = _traj(context=None, grounded=False)
    ins = [Insight(text="无证据", source=traj, kind="no_evidence")]
    assert default_synthesizer(None, ins) == []


def test_synthesizer_produces_minimal_edit_target() -> None:
    """目标 = 原响应里只替换出错的那一句，而不是整段换成一句证据。"""
    ins = default_reflector(None, [_traj()])
    ex = default_synthesizer(None, ins)
    assert len(ex) == 1
    assert ex[0].target == GROUNDED_RESPONSE
    assert "2.86" not in ex[0].target
    assert ex[0].meta["from_evidence"] is True
    assert ex[0].meta["kind"] == "partially_supported"


def test_synthesizer_keeps_corrected_response_verbatim() -> None:
    traj = _traj(meta={"corrected_response": "修正后的完整回答"})
    ex = default_synthesizer(None, default_reflector(None, [traj]))
    assert [e.target for e in ex] == ["修正后的完整回答"]


def test_synthesizer_dedupes_identical_pairs() -> None:
    traj = _traj()
    ins = default_reflector(None, [traj]) * 3
    assert len(default_synthesizer(None, ins)) == 1


def test_loop_synthesizes_from_evidence_end_to_end() -> None:
    """端到端：``synthesized > 0``（旧实现此处恒为 0）。"""
    buf = ExperienceBuffer()
    buf.add(_traj())
    loop = DreamingLoop(
        model=None, buffer=buf,
        config=DreamConfig(mode="memory", require_regression_check=False),
    )
    rep = loop.step()
    assert rep.harvested == 1
    assert rep.insights >= 1
    assert rep.synthesized == 1
    assert rep.stage is DreamStage.COMMIT


# ===========================================================================
# 3. 记忆：键有区分度、能召回、可回滚
# ===========================================================================
def test_text_key_encoder_is_deterministic_and_discriminative() -> None:
    enc = TextKeyEncoder(64)
    a1, a2 = enc("SGC 的流式状态是 2·M·D"), enc("SGC 的流式状态是 2·M·D")
    b = enc("天气与股票市场综述")
    assert pytest.approx(1.0, abs=1e-6) == float(F.normalize(a1, dim=0) @ F.normalize(a2, dim=0))
    assert float(F.normalize(a1, dim=0) @ F.normalize(b, dim=0)) < 0.5


def test_text_key_encoder_survives_process_restart() -> None:
    """内置 ``hash()`` 带进程随机盐；blake2b 不会。这里验证的是"每次构造都一致"。"""
    assert torch.equal(TextKeyEncoder(32)("同一段文本"), TextKeyEncoder(32)("同一段文本"))


def _build_bank(n_filler: int, dim: int = 64) -> tuple:
    docs = [
        ("SGC 流式状态多大？", "流式状态 2·M·D，二阶递推与长度无关。"),
        ("吞吐提升多少？", "实测吞吐在 8k 上下文下提升 1.86 倍。"),
        ("四层闸门是什么？", "四层闸门是 selective/evidence/uncertainty/abstention。"),
        ("CMT 注入点在哪？", "记忆注入点在 SGC 残差流。"),
    ]
    buf = ExperienceBuffer(capacity=4096)
    bank = CompressedMemoryBank(dim=dim, capacity=8192)
    loop = DreamingLoop(
        model=None, buffer=buf, memory_bank=bank,
        config=DreamConfig(mode="memory", batch_size=4096,
                           max_memory_writes=8192, require_regression_check=False),
    )
    for q, ev in docs:
        buf.add(Trajectory(prompt=q, response=ev, confidence=0.9, grounded=True,
                           meta={"context": ev}))
    for i in range(n_filler):
        buf.add(Trajectory(prompt=f"{FILLER}编号{i}", response=f"{FILLER}编号{i}",
                           confidence=0.9, grounded=True, meta={"context": FILLER}))
    loop.step()
    return bank, docs


@pytest.mark.parametrize("n_filler", [0, 64, 128, 512])
def test_memory_retrieval_hits_the_right_entry(n_filler: int) -> None:
    """干扰规模 0/64/128/512 下，top-1 都必须命中正确条目。

    旧实现的键全部共线，本测试在 ``n_filler >= 64`` 时必然失败 —— 这正是
    "记忆写了但从不被召回"的可复现指标。
    """
    bank, docs = _build_bank(n_filler)
    assert len(bank) >= len(docs)
    enc = TextKeyEncoder(bank.dim)
    keys = F.normalize(bank.keys, dim=-1)
    metas = bank.metas()
    for q, _ev in docs:
        scores = keys @ F.normalize(enc(q).unsqueeze(0), dim=-1).squeeze(0)
        top = int(torch.argmax(scores))
        assert metas[top]["prompt"] == q, f"干扰 {n_filler}：检索到 {metas[top]['prompt']}"
        assert metas[top]["kind"] == "knowledge"


def test_grounded_round_writes_knowledge_without_training_signal() -> None:
    """全部答对的一轮也要写记忆 —— 记忆巩固与"有没有训练样本"无关。"""
    bank, docs = _build_bank(0)
    assert len(bank) == len(docs)
    kinds = {m["kind"] for m in bank.metas()}
    assert kinds == {"knowledge"}
    for m in bank.metas():
        assert m["from_evidence"] is True
        assert m["evidence"].strip()


def test_memory_bank_state_roundtrip() -> None:
    bank = CompressedMemoryBank(dim=8, capacity=4)
    bank.add(torch.randn(2, 8), torch.randn(2, 8), [{"a": 1}, {"a": 2}])
    snap = bank.state_dict()
    bank.add(torch.randn(1, 8), torch.randn(1, 8), [{"a": 3}])
    assert len(bank) == 3
    bank.load_state_dict(snap)
    assert len(bank) == 2
    assert [m["a"] for m in bank.metas()] == [1, 2]


def test_memory_snapshot_is_deep_copy() -> None:
    bank = CompressedMemoryBank(dim=4, capacity=4)
    bank.add(torch.ones(1, 4), torch.ones(1, 4), [{"k": 1}])
    snap = bank.state_dict()
    snap["meta"][0]["k"] = 99
    assert bank.metas()[0]["k"] == 1


# ===========================================================================
# 4. no-regression gate：多指标 + 置信下界
# ===========================================================================
def test_eval_scores_accepts_three_return_shapes() -> None:
    assert EvalScores.from_any(0.5).primary() == 0.5
    assert EvalScores.from_any({"a": 0.25, "b": 0.75}).primary() == 0.25
    s = EvalScores.from_any({"a": [0.0, 1.0]})
    assert s.aggregate()["a"] == 0.5 and len(s) == 2


def test_eval_scores_rejects_unusable_return() -> None:
    with pytest.raises(TypeError):
        EvalScores.from_any("oops")
    with pytest.raises(TypeError):
        EvalScores.from_any({})


def test_bootstrap_interval_is_paired() -> None:
    """配对 bootstrap：方差大的样本对不会把区间撑成"什么都可能"。"""
    base = [1.0] * 5
    new = [2.0, 2.0, 2.0, 0.5, 0.5]
    lower, upper = paired_bootstrap_ci(base, new, 0.95, 2000, 0)
    assert lower < 0.0 < upper


def test_gate_rolls_back_when_point_estimate_improves_but_lower_bound_is_negative() -> None:
    """点估计 +0.4，但置信下界 -0.2 → 判退化。这是升级 gate 的核心理由。"""
    base = EvalScores({"m": [1.0] * 5})
    new = EvalScores({"m": [2.0, 2.0, 2.0, 0.5, 0.5]})
    verdict = regression_gate(base, new, tolerance=0.0, n_boot=2000, seed=0)
    assert verdict.score_delta is not None and verdict.score_delta > 0
    assert not verdict.passed
    assert verdict.failed_metrics == ["m"]
    assert verdict.ci_lower["m"] < 0.0


def test_gate_fails_when_any_single_metric_regresses() -> None:
    """总分上涨不能抵消某项崩掉：任一指标失守即回滚。"""
    base = EvalScores({"grounding": [0.8] * 6, "abstain": [0.9] * 6})
    new = EvalScores({"grounding": [0.95] * 6, "abstain": [0.4] * 6})
    verdict = regression_gate(base, new, tolerance=0.0, n_boot=500, seed=0)
    assert not verdict.passed
    assert verdict.failed_metrics == ["abstain"]
    assert "grounding" in verdict.ci_lower and verdict.ci_lower["grounding"] > 0


def test_gate_passes_on_consistent_improvement() -> None:
    base = EvalScores({"grounding": [0.8] * 6, "abstain": [0.9] * 6})
    new = EvalScores({"grounding": [0.85] * 6, "abstain": [0.92] * 6})
    verdict = regression_gate(base, new, tolerance=0.0, n_boot=500, seed=0)
    assert verdict.passed, verdict.notes


def test_gate_tolerates_small_dip_within_tolerance() -> None:
    base = EvalScores({"m": [1.0] * 8})
    new = EvalScores({"m": [0.99] * 8})
    assert regression_gate(base, new, tolerance=0.05, n_boot=200, seed=0).passed


def test_gate_reports_metrics_only_on_one_side() -> None:
    base = EvalScores({"m": [1.0] * 4, "legacy": [0.5] * 4})
    new = EvalScores({"m": [1.0] * 4, "fresh": [0.5] * 4})
    verdict = regression_gate(base, new, n_boot=100, seed=0)
    assert verdict.passed
    assert any("legacy" in n for n in verdict.notes)
    assert any("fresh" in n for n in verdict.notes)


# ===========================================================================
# 5. gate 接进循环：拒绝在没有独立评测时写回权重
# ===========================================================================
class _Model:
    def __init__(self) -> None:
        self.snapshots = 0
        self.restores = 0

    def snapshot(self):
        self.snapshots += 1
        return {"w": 1}

    def restore(self, snap):
        self.restores += 1


def _weights_loop(verifier, bank=None, mode="weights"):
    buf = ExperienceBuffer()
    buf.add(_traj())
    return DreamingLoop(
        model=_Model(), buffer=buf, memory_bank=bank,
        config=DreamConfig(mode=mode, regression_tolerance=0.0,
                           bootstrap_samples=400),
        consolidator=lambda m, ex: None,
        verifier=verifier,
    )


def test_loop_refuses_weight_writeback_without_verifier() -> None:
    """默认验证器恒返回常量 → gate 恒通过 → 必须拒绝写回权重。"""
    loop = _weights_loop(verifier=None)
    rep = loop.step()
    assert rep.stage is DreamStage.ROLLBACK
    assert rep.rolled_back is True
    assert loop.model.restores == 1
    assert any("拒绝写回权重" in n for n in rep.notes)


def test_loop_rolls_back_on_high_variance_improvement() -> None:
    base = [1.0] * 5
    new = [2.0, 2.0, 2.0, 0.5, 0.5]
    scores = iter([base, new])
    rep = _weights_loop(verifier=lambda m: {"grounding": next(scores)}).step()
    assert rep.stage is DreamStage.ROLLBACK
    assert rep.failed_metrics == ["grounding"]
    assert rep.ci_lower["grounding"] < 0.0


def test_loop_commits_on_confident_improvement() -> None:
    scores = iter([[1.0] * 5, [1.2] * 5])
    rep = _weights_loop(verifier=lambda m: {"grounding": next(scores)}).step()
    assert rep.stage is DreamStage.COMMIT
    assert rep.rolled_back is False
    assert rep.ci_lower["grounding"] > 0


def test_loop_rolls_back_when_evaluator_raises() -> None:
    """评测不可用 ≠ 没退化。保守回滚。"""
    def boom(_model):
        raise RuntimeError("held-out 集缺失")

    loop = _weights_loop(verifier=boom)
    rep = loop.step()
    assert rep.stage is DreamStage.ROLLBACK
    assert any("评测失败" in n for n in rep.notes)


def test_rollback_also_restores_memory() -> None:
    """只回滚权重会把学错的东西留在记忆库，下一轮继续注入。"""
    bank = CompressedMemoryBank(dim=32, capacity=64)
    bank.add(torch.ones(1, 32), torch.ones(1, 32), [{"kind": "knowledge", "prompt": "旧"}])
    scores = iter([[1.0] * 5, [0.1] * 5])
    loop = _weights_loop(verifier=lambda m: {"g": next(scores)}, bank=bank, mode="both")
    rep = loop.step()
    assert rep.stage is DreamStage.ROLLBACK
    assert len(bank) == 1
    assert bank.metas()[0]["prompt"] == "旧"


def test_report_carries_multi_metric_fields() -> None:
    scores = iter([[0.8] * 4, [0.9] * 4])
    rep = _weights_loop(verifier=lambda m: {"grounding": next(scores)}).step()
    d = rep.as_dict()
    assert d["metrics_base"] == {"grounding": 0.8}
    assert d["metrics_new"] == {"grounding": 0.9}
    assert d["ci_lower"]["grounding"] > 0
    assert d["failed_metrics"] == []


# ===========================================================================
# 6. CMT：压缩器真的可训练 + memory-aware 双路前向（G1 / G2）
# ===========================================================================
def _memory_model(dim: int = 32):
    from simplcore.core.config import SimplCoreConfig
    from simplcore.core.model import SimplCoreModel

    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=dim, num_layers=2, num_heads=2,
                    num_bases=4, local_kernel=2, use_memory=True)
    model = SimplCoreModel(cfg)
    with torch.no_grad():
        model.memory_in.weight.copy_(torch.eye(dim) * 0.5)
    return model


def test_trainable_compressor_params_actually_change() -> None:
    """G1：旧 ``MemoryCompressor`` 是训练无关读出，参数永不变化。

    这里验证新压缩器**确实在训练**，并且基座权重全程冻结、事后恢复。
    """
    from simplcore.core.memory import (
        TrainableMemoryCompressor, compressor_param_snapshot, train_compressor_step,
    )

    model = _memory_model()
    comp = TrainableMemoryCompressor(32, k=4, layers=2, kernel=3)
    opt = torch.optim.Adam(comp.parameters(), lr=3e-2)
    ids = torch.randint(0, 64, (2, 12))

    before = compressor_param_snapshot(comp)
    losses = [train_compressor_step(comp, model, ids, opt)["loss"] for _ in range(12)]
    after = compressor_param_snapshot(comp)

    assert float((after - before).abs().max()) > 1e-3, "压缩器参数没有变化，等于没在训练"
    assert losses[-1] < losses[0], f"增量损失未下降：{losses[0]:.4f} → {losses[-1]:.4f}"
    assert all(p.requires_grad for p in model.parameters()), "基座冻结没有恢复"


def test_static_compressor_is_training_invariant() -> None:
    """对照组：``MemoryCompressor`` 的参数在任何调用后都不变（即 G1 的旧缺陷）。"""
    from simplcore.core.memory import MemoryCompressor

    model = _memory_model()
    comp = MemoryCompressor(32, k=4)
    before = torch.cat([p.detach().reshape(-1).clone() for p in comp.parameters()])
    with torch.no_grad():
        for _ in range(3):
            comp(model, torch.randint(0, 64, (2, 8)))
    after = torch.cat([p.detach().reshape(-1) for p in comp.parameters()])
    assert float((after - before).abs().max()) == 0.0


def test_memory_aware_loss_needs_two_forwards_and_is_differentiable() -> None:
    """G2：memory-aware 目标必须是"有记忆 vs 无记忆"的双路前向。"""
    from simplcore.core.memory import memory_aware_loss

    model = _memory_model()
    ids = torch.randint(0, 64, (2, 10))
    mem = torch.randn(2, 32, requires_grad=True)

    assert float(memory_aware_loss(model, ids, memory=None)) == 0.0

    loss = memory_aware_loss(model, ids, memory=mem)
    assert torch.isfinite(loss)
    loss.backward()
    assert mem.grad is not None and float(mem.grad.abs().sum()) > 0.0


def test_memory_aware_loss_has_uniform_floor_at_zero_increment() -> None:
    """增量 logits 全零时，损失**恰好是 ln(V)**（均匀分布的交叉熵），不是 0。

    这是该目标函数的定义性质，必须钉住（否则"记忆没起作用"会被误读成"零损失"）：
    客观度量的是"增量本身能不能预测下一个 token"，增量恒零意味着记忆分支
    没有携带任何信息，损失因此等于随机猜测的熵。
    """
    import math

    from simplcore.core.config import SimplCoreConfig
    from simplcore.core.memory import memory_aware_loss
    from simplcore.core.model import SimplCoreModel

    torch.manual_seed(0)
    model = SimplCoreModel(SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_heads=2,
                               num_bases=4, local_kernel=2, use_memory=True))
    ids = torch.randint(0, 64, (2, 10))
    loss = float(memory_aware_loss(model, ids, memory=torch.randn(2, 32)).detach())
    assert loss == pytest.approx(math.log(64), abs=1e-5)

    # 打开注入后增量不再为零 → 损失偏离均匀熵（证明注入真的进了前向）
    with torch.no_grad():
        model.memory_in.weight.copy_(torch.eye(32) * 0.5)
    loss2 = float(memory_aware_loss(model, ids, memory=torch.randn(2, 32)).detach())
    assert abs(loss2 - math.log(64)) > 1e-4
