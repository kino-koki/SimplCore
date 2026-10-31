# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""A/B 对照与训练器的守护测试。

这些测试防的是同一类错误：**框架看起来很对，但两侧其实不可比**。
它不会让任何形状断言失败，只会让"我们做了受控对照"变成一句空话。
"""
from __future__ import annotations

import json

import pytest
import torch

from simplcore.core.ab import (PairedOutcome, bootstrap_diff_ci, mcnemar,
                             param_efficiency, _binom_two_sided_p)
from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.sft import (SFTConfig, SFTResult, evaluate_loss,
                               fairness_report, train_sft)


# ---------------------------------------------------------------------------
# 统计：精确检验与区间
# ---------------------------------------------------------------------------
def test_mcnemar_no_discordant_is_not_significant() -> None:
    r = mcnemar(0, 0)
    assert r["p_value"] == 1.0
    assert not r["significant"]


def test_mcnemar_symmetric_in_b_and_c() -> None:
    """H0 下 b 与 c 可交换 —— 检验不能偏向"第一个模型更好"。"""
    a, b = mcnemar(7, 2), mcnemar(2, 7)
    assert a["p_value"] == pytest.approx(b["p_value"])


def test_mcnemar_exact_value_is_binomial_not_normal_approx() -> None:
    """b=9,c=0,n=9：精确双侧 = 2 × (1/512) ≈ 0.0039。

    正态近似的 McNemar（含连续性校正）会给出约 0.011 —— 差 3 倍。
    小样本上用近似会把"没证据"读成"有证据"。
    """
    r = mcnemar(9, 0)
    assert r["p_value"] == pytest.approx(2.0 / 512.0, rel=1e-9)


def test_mcnemar_requires_five_discordant_for_significance() -> None:
    """实际有 4 题翻转、全部同向时 p=0.125 > 0.05 —— 必须不显著。

    这是"看起来惊人的结果其实只是 4 道题"的量化版本。
    """
    r = mcnemar(4, 0)
    assert r["p_value"] == pytest.approx(2.0 / 16.0)
    assert not r["significant"]


def test_binom_p_never_exceeds_one() -> None:
    for b in range(12):
        for c in range(12):
            assert 0.0 <= _binom_two_sided_p(b, c) <= 1.0


def test_bootstrap_ci_excludes_zero_for_consistent_gap() -> None:
    a = [1.0] * 10
    b = [0.0] * 10
    ci = bootstrap_diff_ci(a, b, n_boot=500, seed=0)
    assert ci["diff"] == pytest.approx(1.0)
    assert ci["ci_low"] > 0
    assert ci["excludes_zero"]


def test_bootstrap_ci_includes_zero_for_no_difference() -> None:
    a = [1.0, 0.0] * 5
    b = [0.0, 1.0] * 5
    ci = bootstrap_diff_ci(a, b, n_boot=500, seed=0)
    assert ci["diff"] == pytest.approx(0.0)
    assert not ci["excludes_zero"]


def test_bootstrap_requires_paired_lengths() -> None:
    with pytest.raises(ValueError):
        bootstrap_diff_ci([1.0, 0.0], [1.0])


def test_bootstrap_is_deterministic_given_seed() -> None:
    a = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    b = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert bootstrap_diff_ci(a, b, seed=3, n_boot=400) == \
        bootstrap_diff_ci(a, b, seed=3, n_boot=400)


# ---------------------------------------------------------------------------
# 配对结果
# ---------------------------------------------------------------------------
class _FakeReport:
    def __init__(self, tasks):
        self.per_task = tasks


def _tasks(pairs):
    return [{"id": k, "passed": v} for k, v in pairs]


def test_paired_outcome_requires_aligned_task_ids() -> None:
    """两侧题目集合不同必须报错，绝不能按位置对齐。

    按位置对齐会把"少一道题"变成"题目整体错位"，Δ 会完全失真，
    而报告里看不出任何异常。
    """
    a = _FakeReport(_tasks([("x", True), ("y", True)]))
    b = _FakeReport(_tasks([("x", True), ("z", True)]))
    with pytest.raises(ValueError, match="不对齐"):
        PairedOutcome.from_reports("a", "b", a, b)


def test_paired_outcome_counts_flips_correctly() -> None:
    a = _FakeReport(_tasks([("x", True), ("y", False), ("z", True)]))
    b = _FakeReport(_tasks([("x", False), ("y", False), ("z", True)]))
    p = PairedOutcome.from_reports("a", "b", a, b)
    assert p.discordant() == (1, 0)          # 只有 x 是"只 a 过"
    assert p.rate_first() == pytest.approx(2 / 3)
    assert p.rate_second() == pytest.approx(1 / 3)
    assert p.flipped_ids() == [{"id": "x", "first_passed": True,
                                "second_passed": False}]


def test_paired_summary_reports_evidence_weight() -> None:
    """报告必须自带"有几题翻转"，否则 Δ=0.1 会被当成能力差。"""
    a = _FakeReport(_tasks([(f"t{i}", i < 6) for i in range(10)]))
    b = _FakeReport(_tasks([(f"t{i}", i < 5) for i in range(10)]))
    s = PairedOutcome.from_reports("a", "b", a, b).summary()
    assert s["delta"] == pytest.approx(0.1)
    assert "1/10" in s["evidence_weight"]
    assert s["mcnemar"]["significant"] is False


# ---------------------------------------------------------------------------
# 参数效率
# ---------------------------------------------------------------------------
def test_param_efficiency_requires_two_points_per_family() -> None:
    out = param_efficiency([
        {"name": "sgc", "params": 30_000_000, "task_pass_rate": 0.7},
        {"name": "attn", "params": 30_000_000, "task_pass_rate": 0.6},
    ])
    assert out["comparisons"] == []
    assert any("两档" in c for c in out["caveats"])


def test_param_efficiency_interpolates() -> None:
    """sgc 在 20M 拿到 0.6 分，attn 在 20M 是 0.4、40M 是 0.8 → 等价 30M，比值 1.5。"""
    out = param_efficiency([
        {"name": "sgc", "params": 20_000_000, "task_pass_rate": 0.6},
        {"name": "sgc", "params": 40_000_000, "task_pass_rate": 0.9},
        {"name": "attn", "params": 20_000_000, "task_pass_rate": 0.4},
        {"name": "attn", "params": 40_000_000, "task_pass_rate": 0.8},
    ])
    c = out["comparisons"][0]
    assert c["equivalent_params"] == 30_000_000
    assert c["ratio"] == pytest.approx(1.5)


def test_param_efficiency_refuses_to_extrapolate() -> None:
    """目标分数超过对照族最高分：不许外推，只能如实说"没达到"。"""
    out = param_efficiency([
        {"name": "sgc", "params": 20_000_000, "task_pass_rate": 0.95},
        {"name": "sgc", "params": 40_000_000, "task_pass_rate": 0.99},
        {"name": "attn", "params": 20_000_000, "task_pass_rate": 0.3},
        {"name": "attn", "params": 40_000_000, "task_pass_rate": 0.5},
    ])
    c = out["comparisons"][0]
    assert c["ratio"] is None
    assert "不能报效率比" in c["note"]


def test_param_efficiency_always_carries_caveats() -> None:
    out = param_efficiency([
        {"name": "sgc", "params": 20_000_000, "task_pass_rate": 0.6},
        {"name": "sgc", "params": 40_000_000, "task_pass_rate": 0.9},
        {"name": "attn", "params": 20_000_000, "task_pass_rate": 0.4},
        {"name": "attn", "params": 40_000_000, "task_pass_rate": 0.8},
    ])
    assert any("不是'能力等价'的证明" in c for c in out["caveats"])


# ---------------------------------------------------------------------------
# 训练器
# ---------------------------------------------------------------------------
def _tiny_model(vocab=64, dim=32, layers=2, use_memory=False):
    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=vocab, dim=dim, num_layers=layers,
                    num_heads=2, num_bases=4, local_kernel=2,
                    use_memory=use_memory)
    return SimplCoreModel(cfg), cfg


class _FixedLoader:
    """产出固定 batch 的 loader（无随机性，便于断言更新方向）。"""

    def __init__(self, batches):
        self.batches = list(batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _batch(B=2, L=8, V=64, mem=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (B, L), generator=g)
    y = x.clone()
    if mem is not None:
        return x, y, mem
    return x, y


def test_train_sft_returns_result_with_curve() -> None:
    model, _ = _tiny_model()
    cfg = SFTConfig(max_steps=6, log_every=2, eval_every=0, warmup=1, lr=1e-2)
    res = train_sft(model, _FixedLoader([_batch()]), None, cfg,
                    torch.device("cpu"))
    assert isinstance(res, SFTResult)
    assert res.steps == 6
    assert res.tokens > 0
    assert res.stop_reason == "max_steps"
    assert any("loss" in r for r in res.curve)


def test_train_sft_grad_accum_is_token_weighted() -> None:
    """梯度累积必须等价于"拼成大批再算平均 loss"。

    等权写法 ``(loss / ga).backward()`` 在长短样本混排时会给长样本
    系统性低估 —— 等价于偷偷重采样了数据，而曲线看不出来。
    这里用两个 token 数差很多的 micro-batch 验证：梯度必须与
    "手工加权平均 loss 再反传"一致。
    """
    model, _ = _tiny_model()
    # grad_clip 会**缩放**梯度（范数超过阈值时），与"加权是否正确"无关，
    # 所以这里关掉它，单独考察归一化口径。
    cfg = SFTConfig(max_steps=1, grad_accum=2, warmup=1, lr=0.0,
                    log_every=999, eval_every=0, grad_clip=1e9)

    g = torch.Generator().manual_seed(1)
    x1 = torch.randint(0, 64, (2, 8), generator=g)
    y1 = x1.clone()
    x2 = torch.randint(0, 64, (2, 8), generator=g)
    y2 = x2.clone()
    y2[0, 4:] = -100        # 第二个 micro-batch 只有一半 token 受监督

    train_sft(model, _FixedLoader([(x1, y1), (x2, y2)]), None, cfg,
              torch.device("cpu"))
    got = torch.cat([p.grad.reshape(-1).clone()
                     for p in model.parameters() if p.grad is not None])

    # 参考实现：手算 token 加权平均
    ref, _ = _tiny_model()
    _, l1 = ref(x1, y1)
    _, l2 = ref(x2, y2)
    n1 = int((y1 != -100).sum())
    n2 = int((y2 != -100).sum())
    ((l1 * n1 + l2 * n2) / (n1 + n2)).backward()
    want = torch.cat([p.grad.reshape(-1) for p in ref.parameters()
                      if p.grad is not None])
    assert torch.allclose(got, want, atol=1e-5), \
        f"梯度累积不是 token 加权：max|diff|={float((got-want).abs().max())}"


def test_train_sft_skips_all_masked_batches() -> None:
    """整批被掩码时不做更新（否则等于用全零目标训练）。"""
    model, _ = _tiny_model()
    cfg = SFTConfig(max_steps=3, log_every=999, eval_every=0, warmup=1, lr=0.1)
    x = torch.zeros((2, 8), dtype=torch.long)
    y = torch.full((2, 8), -100, dtype=torch.long)
    before = [p.detach().clone() for p in model.parameters()]
    res = train_sft(model, _FixedLoader([(x, y)]), None, cfg, torch.device("cpu"))
    if res.tokens == 0:
        for a, b in zip(before, model.parameters()):
            assert torch.equal(a, b.detach()), "空批不应该改变参数"


def test_train_sft_respects_time_budget() -> None:
    model, _ = _tiny_model()
    cfg = SFTConfig(max_steps=100000, log_every=999, eval_every=0,
                    time_budget_s=1e-6, warmup=1, lr=1e-3)
    res = train_sft(model, _FixedLoader([_batch()]), None, cfg, torch.device("cpu"))
    assert res.stop_reason == "time_budget"


def test_train_sft_raises_when_memory_given_but_model_lacks_it() -> None:
    """带记忆的样本喂给不支持的模型必须报错，不能静默丢掉记忆。

    静默丢掉会让"有记忆"的样本退化成"没记忆"，训练照跑、loss 照降，
    但模型学到的是瞎猜 —— 这是最坏的一种失败。
    """
    model, _ = _tiny_model(use_memory=False)
    cfg = SFTConfig(max_steps=1, log_every=999, eval_every=0, warmup=1)
    batch = _batch(mem=torch.randn(2, 32))
    with pytest.raises(ValueError, match="use_memory"):
        train_sft(model, _FixedLoader([batch]), None, cfg, torch.device("cpu"))


def test_model_raises_on_memory_when_use_memory_false() -> None:
    """模型层单独一条：这是"静默丢记忆"的唯一防线。"""
    model, _ = _tiny_model(use_memory=False)
    with pytest.raises(ValueError, match="use_memory"):
        model(torch.zeros((1, 4), dtype=torch.long), memory=torch.randn(1, 32))


def test_train_sft_memory_reaches_injection_projection() -> None:
    """记忆必须真的流过 ``memory_in``：零初始化也要有非零梯度。

    零初始化只保证"初始输出不变"，不阻断梯度 —— ``W·x`` 对 ``W`` 的
    梯度是 ``dL/dy · xᵀ``，与 ``W`` 的初值无关。这一条把"零初始化
    = 不学习"这个常见误解钉死。
    """
    model, _ = _tiny_model(use_memory=True)
    cfg = SFTConfig(max_steps=3, log_every=999, eval_every=0, warmup=1, lr=1e-2)
    mem = torch.randn(2, 32)
    train_sft(model, _FixedLoader([_batch(mem=mem)]), None, cfg, torch.device("cpu"))
    assert float(model.memory_in.weight.abs().max()) > 0, \
        "memory_in 零初始化后仍然是全 0 —— 记忆没有被训练"


def test_zero_memory_row_does_not_change_logits() -> None:
    """零记忆行必须等价于"不注入"（无 bias 线性层 → 数学保证）。"""
    model, _ = _tiny_model(use_memory=True)
    with torch.no_grad():
        model.memory_in.weight.normal_(0, 0.1)
    model.eval()
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a = model(x)
        b = model(x, memory=torch.zeros(2, 32))
    assert torch.equal(a, b)


def test_nonzero_memory_row_changes_logits() -> None:
    """非零记忆必须真的改变输出，否则注入点接错了。"""
    model, _ = _tiny_model(use_memory=True)
    with torch.no_grad():
        model.memory_in.weight.normal_(0, 0.1)
    model.eval()
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        a = model(x)
        b = model(x, memory=torch.randn(2, 32))
    assert not torch.allclose(a, b)


def test_evaluate_loss_is_token_weighted() -> None:
    """验证损失也按 token 加权，否则短样本多的批次会压低报告值。"""
    model, _ = _tiny_model()
    model.eval()
    g = torch.Generator().manual_seed(2)
    x1 = torch.randint(0, 64, (2, 8), generator=g)
    y1 = x1.clone()
    x2 = torch.randint(0, 64, (2, 8), generator=g)
    y2 = x2.clone()
    y2[:, 2:] = -100
    got = evaluate_loss(model, _FixedLoader([(x1, y1), (x2, y2)]),
                        torch.device("cpu"))["loss"]
    with torch.no_grad():
        _, l1 = model(x1, y1)
        _, l2 = model(x2, y2)
    n1, n2 = int((y1 != -100).sum()), int((y2 != -100).sum())
    want = (float(l1) * n1 + float(l2) * n2) / (n1 + n2)
    assert got == pytest.approx(want, rel=1e-5)


# ---------------------------------------------------------------------------
# 可比性自检
# ---------------------------------------------------------------------------
def _res(steps=100, tokens=10000, nn=0, stop="max_steps"):
    return SFTResult(steps=steps, tokens=tokens, n_nonfinite=nn, stop_reason=stop)


def test_fairness_passes_for_identical_runs() -> None:
    r = fairness_report({"a": _res(), "b": _res()})
    assert r["comparable"]
    assert r["violations"] == []


def test_fairness_flags_step_mismatch() -> None:
    r = fairness_report({"a": _res(steps=3000), "b": _res(steps=2800)})
    assert not r["comparable"]
    assert any("步数" in v for v in r["violations"])


def test_fairness_flags_token_mismatch() -> None:
    r = fairness_report({"a": _res(tokens=1200), "b": _res(tokens=900)})
    assert not r["comparable"]
    assert any("token" in v for v in r["violations"])


def test_fairness_flags_early_stop_and_nonfinite() -> None:
    r = fairness_report({"a": _res(stop="time_budget"), "b": _res(nn=2)})
    assert not r["comparable"]
    assert any("提前结束" in v for v in r["violations"])
    assert any("非有限" in v for v in r["violations"])


def test_fairness_flags_param_budget_gap() -> None:
    """差 30% 必须报出来。15% 以内的差是层数量化结果，不算违规。"""
    r = fairness_report({"a": _res(), "b": _res()},
                        budgets={"a": 30_085_298, "b": 40_000_000})
    assert not r["comparable"]
    assert any("参数量" in v for v in r["violations"])


def test_fairness_allows_layer_quantization_gap() -> None:
    """10% 的差在整数层数的粒度内，不该被当成违规（但会写进报告）。"""
    r = fairness_report({"a": _res(), "b": _res()},
                        budgets={"a": 2_481_684, "b": 2_705_664})
    assert r["comparable"]


def test_fairness_tolerates_small_param_gap() -> None:
    """0.64% 的参数量差（tuan-30m vs attn-30m）不该被当成不可比。"""
    r = fairness_report({"a": _res(), "b": _res()},
                        budgets={"a": 30_085_298, "b": 29_894_144})
    assert r["comparable"]
