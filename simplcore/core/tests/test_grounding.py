# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""证据绑定（grounding）与证据感知拒答（abstention）守护测试。

守的是**结构性主张**，不守"抗幻觉有效"这个经验结论（后者由
``scripts/eval_grounding.py`` 用构造标签实测，数字落在 artifacts JSON 里）：

1. 判据是确定性的、可跨进程复现的（不依赖 ``hash()`` / dict 顺序）；
2. 共享单字造成的**伪匹配必须被拒绝**（LCS 虚高 → bigram 判据拦下）；
3. 四态决策的四个状态都真实可达，且判定顺序不会把"信息不足"误判成"拒答"；
4. 校准函数在可分离数据上给出的阈值/温度合理；
5. 运行时管线（检索→生成→归因→决策）端到端可跑，且拒答时**不输出原文**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.abstain import (                                 # noqa: E402
    AbstentionState,
    GroundedAnswerer,
    GroundedPolicy,
    evidence_aware_decision,
    expected_calibration_error,
    fit_temperature,
    select_threshold,
)
from simplcore.core.grounding import (                               # noqa: E402
    Claim,
    EvidenceIndex,
    EvidenceSpan,
    attribution_coverage,
    bind_claim,
    bigram_coverage_of,
    coverage_of,
    char_token_offsets,
    grounded_label_tensor,
    normalize,
    split_claims,
    token_labels_from_bindings,
    unsupported_claims,
)

CTX = ("阻尼振荡基字典用 M 个阻尼振荡基的线性组合构成卷积核。"
       "常驻状态为 固定字节数（由代码计算），与序列长度无关。"
       "四层闸门逐层收紧，每层对应一种不同的失败模式。")
FABRICATED = "该架构在 3B 规模上击败了所有公开基线，训练成本降低八成。"


# ---------------------------------------------------------------------------
# 1. 归一化与逐字判据
# ---------------------------------------------------------------------------
def test_normalize_strips_punct_and_space() -> None:
    assert normalize(" A b，c 。") == "abc"
    assert normalize("") == ""


def test_exact_substring_is_supported() -> None:
    cov, exact = coverage_of("常驻状态为 固定字节数（由代码计算），与序列长度无关", CTX)
    assert exact is True and cov == 1.0


def test_fabricated_claim_rejected_despite_shared_chars() -> None:
    """纯编造的中文句子与真证据常有 0.1–0.5 的 LCS 覆盖率，必须被 bigram 拦下。"""
    cov, exact = coverage_of(FABRICATED, CTX)
    ng = bigram_coverage_of(FABRICATED, CTX)
    assert exact is False
    assert ng < 0.35, f"二进制判据失效：ngram={ng:.3f}"
    span = EvidenceSpan("s1", CTX, "doc")
    b = bind_claim(Claim(FABRICATED), [span])
    assert b.supported is False, (
        f"伪匹配被误判为有依据（coverage={b.coverage:.3f}, ngram={b.ngram:.3f}）")


def test_bigram_coverage_bounds() -> None:
    assert bigram_coverage_of("abcd", "xxabcdxx") == 1.0
    assert bigram_coverage_of("abcd", "zzzz") == 0.0
    assert bigram_coverage_of("a", "zzz") == 1.0      # 无二元组可比 → 不设限


# ---------------------------------------------------------------------------
# 2. 确定性 / 可复现
# ---------------------------------------------------------------------------
def test_binding_is_deterministic_and_order_independent() -> None:
    spans = [EvidenceSpan("s1", "甲" * 20 + "常驻状态为 固定字节数（由代码计算）", "a"),
             EvidenceSpan("s2", "常驻状态为 固定字节数（由代码计算）与序列长度无关", "b")]
    claim = Claim("常驻状态为 固定字节数（由代码计算）与序列长度无关")
    b1 = bind_claim(claim, spans)
    b2 = bind_claim(claim, list(reversed(spans)))
    assert b1.span.span_id == b2.span.span_id == "s2"
    assert b1.coverage == b2.coverage


# ---------------------------------------------------------------------------
# 3. 切句与聚合
# ---------------------------------------------------------------------------
def test_split_claims_drops_short_fragments() -> None:
    claims = split_claims("好的。这是第一句有实质内容的话。嗯。这是第二句实质内容。")
    texts = [c.text for c in claims]
    assert len(texts) == 2
    assert all("。" not in t for t in texts)


def test_split_claims_respects_max() -> None:
    text = "。".join(f"第{i}句有足够长度的实质内容用于测试" for i in range(50))
    assert len(split_claims(text, max_claims=7)) == 7


def test_attribution_coverage_weights_long_claims() -> None:
    idx = EvidenceIndex.from_text(CTX, "doc")
    ans = "常驻状态为 固定字节数（由代码计算），与序列长度无关。" + FABRICATED
    bindings = idx.bind_all(split_claims(ans))
    cov_w = attribution_coverage(bindings, weight_by_length=True)
    cov_u = attribution_coverage(bindings, weight_by_length=False)
    assert 0.0 < cov_w < 1.0
    assert cov_u == pytest.approx(0.5)
    assert len(unsupported_claims(bindings)) == 1


def test_empty_answer_has_zero_coverage() -> None:
    assert attribution_coverage([]) == 0.0


# ---------------------------------------------------------------------------
# 4. 索引构建
# ---------------------------------------------------------------------------
def test_index_from_text_offsets_track_source() -> None:
    idx = EvidenceIndex.from_text(CTX, "doc")
    assert len(idx) >= 3
    sp = idx.spans[0]
    assert sp.start >= 0 and CTX[sp.start:sp.start + 4] == sp.text[:4]


def test_index_from_retrieval_accepts_dicts_and_objects() -> None:
    class Hit:
        def __init__(self, text, id):
            self.text, self.id = text, id

    idx = EvidenceIndex.from_retrieval([{"text": "甲乙丙丁戊", "id": "h1"},
                                        Hit("己庚辛壬癸", "h2")])
    assert len(idx) == 2
    assert idx.spans[0].span_id == "h1" and idx.spans[1].source == "h2"


# ---------------------------------------------------------------------------
# 5. 四态决策
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("g,cov,rc,expect", [
    (0.9, 0.95, 1.0, AbstentionState.ANSWER),     # 证据充分
    (0.9, 0.60, 1.0, AbstentionState.HEDGE),      # 部分归因
    (0.1, 0.05, 0.0, AbstentionState.CLARIFY),    # 检索没命中 → 要资料，不是拒答
    (0.1, 0.10, 1.0, AbstentionState.ABSTAIN),    # 有命中却归因不上 → 拒答
])
def test_four_states_reachable(g, cov, rc, expect) -> None:
    d = evidence_aware_decision(g, cov, rc)
    assert d.state is expect, f"{g},{cov},{rc} → {d.state} ({d.reason})"
    assert 0.0 <= d.score <= 1.0


def test_clarify_takes_precedence_over_abstain_when_nothing_retrieved() -> None:
    """信息不足 ≠ 模型在编。判定顺序必须是"先判资料，再判归因"。"""
    assert evidence_aware_decision(0.0, 0.0, 0.0).state is AbstentionState.CLARIFY


def test_policy_validation_rejects_bad_bounds() -> None:
    with pytest.raises(ValueError):
        GroundedPolicy(tau_grounded=1.5).validate()
    with pytest.raises(ValueError):
        GroundedPolicy(coverage_answer=0.3, coverage_hedge=0.7).validate()
    with pytest.raises(ValueError):
        GroundedPolicy(tau_retrieval=-0.1).validate()


# ---------------------------------------------------------------------------
# 6. 校准
# ---------------------------------------------------------------------------
def test_select_threshold_recovers_separable_split() -> None:
    scores = [0.05, 0.1, 0.15, 0.2, 0.85, 0.9, 0.95, 0.99]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    best = select_threshold(scores, labels)
    assert best["f1"] == 1.0
    assert 0.2 < best["threshold"] <= 0.85


def test_fit_temperature_is_positive_and_monotone_sane() -> None:
    logits = [-2.0, -1.0, 0.5, 1.5, 3.0]
    labels = [0, 0, 1, 1, 1]
    t = fit_temperature(logits, labels)
    assert t > 0
    assert fit_temperature([], []) == 1.0


def test_ece_zero_when_perfectly_calibrated() -> None:
    probs = [0.0, 1.0, 0.0, 1.0]
    labels = [0, 1, 0, 1]
    assert expected_calibration_error(probs, labels, n_bins=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. 运行时管线端到端
# ---------------------------------------------------------------------------
def _answerer(reply: str) -> GroundedAnswerer:
    return GroundedAnswerer(generate=lambda prompt, ctx: reply)


def test_pipeline_answers_when_grounded() -> None:
    r = _answerer("常驻状态为 固定字节数（由代码计算），与序列长度无关。").answer("常驻状态多大？", context=CTX)
    assert r.decision.state is AbstentionState.ANSWER
    assert r.text == r.raw_answer
    assert r.attribution["n_unsupported"] == 0


def test_pipeline_abstains_and_hides_hallucination() -> None:
    r = _answerer(FABRICATED).answer("能击败所有基线吗？", context=CTX)
    assert r.decision.state is AbstentionState.ABSTAIN
    assert FABRICATED not in r.text, "拒答时不得把编造内容返回给调用方"
    assert r.raw_answer == FABRICATED


def test_pipeline_clarifies_without_retrieval() -> None:
    r = _answerer("我不知道。").answer("问题", context="")
    assert r.decision.state is AbstentionState.CLARIFY


def test_pipeline_marks_proxy_grounded_source() -> None:
    r = _answerer("常驻状态为 固定字节数（由代码计算），与序列长度无关。").answer("q", context=CTX)
    assert r.attribution["grounded_source"] == "proxy"


def test_pipeline_uses_injected_scorer() -> None:
    a = GroundedAnswerer(generate=lambda p, c: "常驻状态为 固定字节数（由代码计算），与序列长度无关。",
                         grounded_scorer=lambda p, ans: 0.99)
    r = a.answer("q", context=CTX)
    assert r.attribution["grounded_source"] == "head"
    assert r.decision.grounded_prob == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 8. 结构化篡改判据（数字 / 否定）
# ---------------------------------------------------------------------------
def test_number_tampering_is_rejected() -> None:
    """单个数字被改写时字符覆盖率仍有 ~0.9，必须由数字判据拦下。"""
    span = EvidenceSpan("s1", "SGC 在 L=2048 时比标准因果注意力快 5.64×。", "doc")
    tampered = "SGC 在 L=2048 时比标准因果注意力快 9.64×。"
    cov, exact = coverage_of(tampered, span.text)
    assert cov > 0.85 and not exact          # 文本重叠几乎无损
    b = bind_claim(Claim(tampered), [span])
    assert b.numbers_ok is False
    assert b.supported is False
    
    
def test_negation_flip_is_rejected() -> None:
    span = EvidenceSpan("s1", "常驻状态与序列长度无关。", "doc")
    flipped = "常驻状态与序列长度有关。"
    cov, _ = coverage_of(flipped, span.text)
    assert cov > 0.8
    b = bind_claim(Claim(flipped), [span])
    assert b.polarity_ok is False
    assert b.supported is False


def test_exact_claim_inside_longer_span_stays_supported() -> None:
    """回归：claim 是长 span 的一部分时，不能用整段否定词计数判它极性不一致。"""
    span = EvidenceSpan("s1",
                        "这里不引入任何注意力，也不引入 KV 缓存。"
                        "常驻状态与序列长度无关。", "doc")
    b = bind_claim(Claim("常驻状态与序列长度无关。"), [span])
    assert b.exact is True
    assert b.polarity_ok is True
    assert b.supported is True


def test_numbers_supported_helper() -> None:
    from simplcore.core.grounding import extract_numbers, numbers_supported
    assert extract_numbers("1,024 与 5.64") == ["1024", "5.64"]
    assert numbers_supported("快 5.64 倍", "实测快 5.64 倍")
    assert not numbers_supported("快 5.64 倍", "实测快 9.64 倍")


def test_single_unsupported_claim_caps_answer_to_hedge() -> None:
    """逐句性质：一句假话不能靠两句真话的长度把整体拉成"可直接作答"。"""
    ctx = ("常驻状态为 固定字节数（由代码计算），与序列长度无关。"
           "四层闸门逐层收紧，每层对应一种不同的失败模式。"
           "阻尼振荡基字典用 M 个阻尼振荡基的线性组合构成卷积核。")
    idx = EvidenceIndex.from_text(ctx, "doc")
    answer = ("常驻状态为 固定字节数（由代码计算），与序列长度无关。"
              "阻尼振荡基字典用 M 个阻尼振荡基的线性组合构成卷积核。"
              + FABRICATED)
    r = GroundedAnswerer(generate=lambda p, c: answer).answer("q", context=ctx)
    assert r.attribution["n_unsupported"] == 1
    assert r.decision.state is not AbstentionState.ANSWER
    assert "无归因依据" in r.decision.reason


# ===========================================================================
# 句子级标签 → token 位置（L4 头真正的训练标签）
# ===========================================================================
ALIGN_ANSWER = ("SGC 流式状态是 2·M·D。"
          "该架构已在三家芯片厂量产。"
          "实测吞吐提升 1.86 倍")
ALIGN_CTX = "SGC 流式状态是 2·M·D。实测吞吐提升 1.86 倍。"


def test_split_claims_records_char_spans() -> None:
    """每句都要带回原文字符跨度，且跨度切出来**就是这句本身**。"""
    claims = split_claims(ALIGN_ANSWER)
    assert len(claims) == 3
    for c in claims:
        assert c.start >= 0 and c.end > c.start
        assert ALIGN_ANSWER[c.start:c.end] == c.text


def test_split_claims_span_survives_trailing_fragment() -> None:
    """最后一段没有句末标点时也要被切出来并定位。"""
    claims = split_claims("常驻状态与序列长度无关")
    assert len(claims) == 1
    assert ALIGN_ANSWER[:0] == ""
    assert claims[0].text == "常驻状态与序列长度无关"


def test_token_labels_align_to_sentences() -> None:
    """逐位置标签：每句区间内取值一致，句间空隙取 default。"""
    idx = EvidenceIndex.from_text(ALIGN_CTX, "doc")
    claims = split_claims(ALIGN_ANSWER)
    bindings = idx.bind_all(claims)
    assert [b.supported for b in bindings] == [True, False, True]

    labels = token_labels_from_bindings(bindings, char_token_offsets(ALIGN_ANSWER))
    assert len(labels) == len(ALIGN_ANSWER)
    for c, b in zip(claims, bindings):
        seg = set(labels[c.start:c.end])
        assert seg == {(1.0 if b.supported else 0.0)}, f"第 {c.index} 句标签不纯：{seg}"


def test_token_labels_default_outside_claims() -> None:
    """不属于任何句子的位置（标点/空隙）取 default，不被邻居污染。"""
    idx = EvidenceIndex.from_text(ALIGN_CTX, "doc")
    claims = split_claims(ALIGN_ANSWER)
    bindings = idx.bind_all(claims)
    labels = token_labels_from_bindings(bindings, char_token_offsets(ALIGN_ANSWER), default=-1.0)
    covered = set()
    for c in claims:
        covered.update(range(c.start, c.end))
    for i, v in enumerate(labels):
        if i not in covered:
            assert v == -1.0


def test_token_labels_handle_word_level_tokenizer_offsets() -> None:
    """非字符级 tokenizer 也要能用：token 跨度与句子区间相交即继承标签。"""
    idx = EvidenceIndex.from_text(ALIGN_CTX, "doc")
    claims = split_claims("SGC 流式状态是 2·M·D。该架构已在三家芯片厂量产")
    bindings = idx.bind_all(claims)
    offsets = [(0, 15), (16, 28)]              # 两个"词"恰好等于两句
    labels = token_labels_from_bindings(bindings, offsets)
    assert labels == [1.0, 0.0]


def test_grounded_label_tensor_matches_list_version() -> None:
    idx = EvidenceIndex.from_text(ALIGN_CTX, "doc")
    bindings = idx.bind_all(split_claims(ALIGN_ANSWER))
    offs = char_token_offsets(ALIGN_ANSWER)
    lst = token_labels_from_bindings(bindings, offs)
    ten = grounded_label_tensor(bindings, offs)
    assert ten.shape == (len(ALIGN_ANSWER),)
    assert ten.tolist() == lst
