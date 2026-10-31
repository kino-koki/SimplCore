# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""对话运行时：记忆检索 + 四态处置的守护测试。

做法：把**生成**用注入的 generator 固定成一段已知文本，这样
"归因 → 四态"这段逻辑可以被精确验证，而不需要先训一个模型。
生成质量是另一回事（见 ``test_dialogue_eval.py``），这里只测闸门。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.abstain import AbstentionState
from simplcore.core.chat import ChatTemplate, SpecialTokenCodec
from simplcore.core.dialogue_runtime import (ChatSession, DialogMemory,
                                            MemoryHit, MIN_RECALL_SCORE)
from simplcore.core.tokenization import CharVocab


# ---------------------------------------------------------------------------
def _codec() -> SpecialTokenCodec:
    # 词表必须覆盖会用到的**全部**字符：CharVocab 把未见字符映射到 id 0，
    # decode 会丢弃 unk —— 词表不全时"断言某字符不出现"会假通过。
    chars = ("用户的名字是小林青奈科技营业收入是人员专利数量市场占有率"
             "不知道一二三四五六七八九十"
             "我叫什么名字？请原样重复这句子不要加其他内容：流式卷积做"
             "该架构已在三家芯片厂量产吞吐提升倍实测的阻尼振荡残差")
    return SpecialTokenCodec(CharVocab.train([chars], min_freq=1))


def _fixed(text: str, stopped: bool = True):
    """构造一个"永远输出这段文本"的生成器。"""
    def gen(model, prompt, codec, **kw):
        return {"text": text, "tokens": len(text), "stopped": stopped,
                "leaked": False}
    return gen


@pytest.fixture()
def codec() -> SpecialTokenCodec:
    return _codec()


def _mem(dim: int = 64) -> DialogMemory:
    return DialogMemory(dim=dim)


# ---------------------------------------------------------------------------
# 记忆库
# ---------------------------------------------------------------------------
def test_memory_rejects_empty_and_blank() -> None:
    m = _mem()
    assert not m.add("")
    assert not m.add("   \n\t")
    assert len(m) == 0


def test_memory_write_and_recall() -> None:
    m = _mem()
    assert m.add("青奈科技的营业收入是 128 人。")
    assert m.add("用户的名字是小林。")
    hits = m.search("用户的名字是小林。", top_k=2)
    assert hits, "同文本必须能召回自己"
    assert hits[0].text == "用户的名字是小林。"
    assert hits[0].score > 0.9


def test_memory_recall_orders_by_score() -> None:
    m = _mem()
    m.add("用户的名字是小林。")
    m.add("完全无关的一句：今天下雨了。")
    hits = m.search("用户的名字是小林。", top_k=2)
    assert hits[0].score >= hits[-1].score


def test_memory_key_controls_retrieval_independently_of_value() -> None:
    """``key`` 是检索键，``text`` 是内容 —— 两者分开是为了绕过字面编码的局限。

    ``TextKeyEncoder`` 是 3-gram 哈希，"我叫什么名字" 与 "用户的名字是小林"
    的 3-gram 交集为空，直接拿内容当键检索不到。这正是 key 参数存在的理由。
    这条测试同时锁住"改 key 不改内容"的语义。
    """
    m = _mem()
    m.add("用户的名字是小林。", key="我叫什么名字 名字 我是谁")
    hits = m.search("我叫什么名字？", top_k=1)
    assert hits and hits[0].text == "用户的名字是小林。"


def test_memory_hit_meta_carries_payload() -> None:
    m = _mem()
    m.add("青奈科技的营业收入是 128 人。", meta={"kind": "fact", "src": "年报"})
    h = m.search("青奈科技的营业收入是 128 人。", top_k=1)[0]
    assert h.meta["kind"] == "fact"
    assert h.meta["src"] == "年报"


def test_memory_aggregate_is_normalized_unit_vector() -> None:
    m = _mem(dim=64)
    m.add("用户的名字是小林。")
    m.add("青奈科技的营业收入是 128 人。")
    vec = m.aggregate(m.search("用户的名字是小林。", top_k=2))
    assert vec is not None
    assert float(vec.norm()) == pytest.approx(1.0, abs=1e-5)


def test_memory_aggregate_of_no_hits_is_none() -> None:
    m = _mem()
    assert m.aggregate([]) is None


def test_memory_snapshot_restore_roundtrip() -> None:
    m = _mem()
    m.add("用户的名字是小林。")
    st = m.snapshot()
    m.add("青奈科技的营业收入是 128 人。")
    assert len(m) == 2
    m.restore(st)
    assert len(m) == 1


# ---------------------------------------------------------------------------
# 四态：每种状态都要可达，且拒答时**不返回原文**
# ---------------------------------------------------------------------------
def test_no_evidence_forces_clarify_and_hides_raw(codec) -> None:
    s = ChatSession(None, codec, ChatTemplate(), memory=None,
                    generator=_fixed("青奈科技 2026 年营业收入是 88 亿。"))
    r = s.ask("青奈科技 2026 年的营业收入是多少？")
    assert r.state == AbstentionState.CLARIFY.value
    assert r.suppressed
    assert "88" not in r.text, "拒答时绝不能把编造的原文一起返回"
    assert "88" in r.raw_text, "原文要留在 raw_text 里供审计"


def test_supported_claim_is_answered(codec) -> None:
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("青奈科技的营业收入是 128 人。"))
    r = s.ask("青奈科技的营业收入是多少？")
    assert r.state == AbstentionState.ANSWER.value
    assert not r.suppressed
    assert r.text == r.raw_text
    assert r.coverage == pytest.approx(1.0)


def test_one_fabricated_sentence_caps_answer_to_hedge(codec) -> None:
    """两句里有一句编造：必须降到 HEDGE，且**标注**哪句没依据。

    这是"逐句"原则的直接检验：覆盖率 0.5 对 2 句里的 1 句，但关键是
    不能因为另一句是真的就整体放行。
    """
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("青奈科技的营业收入是 128 人。"
                                     "该架构已在三家芯片厂量产。"))
    r = s.ask("青奈科技的营业收入和量产情况？")
    assert r.state == AbstentionState.HEDGE.value
    assert not r.suppressed, "HEDGE 仍要作答，只是加限定语"
    assert r.text.startswith("（不确定）")
    assert r.n_unsupported >= 1
    bad = [c for c in r.claims if not c["supported"]]
    assert any("芯片厂" in c["text"] for c in bad), r.claims


def test_empty_generation_is_never_answered(codec) -> None:
    """空回复永远要拦。放行会让"立刻输出 end"的模型拿到满分格式指标。"""
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed(""))
    r = s.ask("青奈科技的营业收入是多少？")
    assert r.state == AbstentionState.CLARIFY.value
    assert r.suppressed


def test_punctuation_only_generation_is_clarify(codec) -> None:
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("。。。"))
    r = s.ask("青奈科技的营业收入是多少？")
    assert r.state == AbstentionState.CLARIFY.value


def test_number_tampering_breaks_support(codec) -> None:
    """把 128 改成 999：文本重叠度仍然很高，但数字核对必须拦住。"""
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("青奈科技的营业收入是 999 人。"),
                    use_evidence_gate=True)
    r = s.ask("青奈科技的营业收入是多少？")
    assert r.coverage < 0.8, f"数字篡改仍被当作有依据：{r.claims}"
    assert r.state != AbstentionState.ANSWER.value


def test_negation_flip_breaks_support(codec) -> None:
    """把"是"改成"不是"：字面几乎相同，极性核对必须拦住。"""
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("青奈科技的营业收入不是 128 人。"))
    r = s.ask("青奈科技的营业收入是多少？")
    assert r.state != AbstentionState.ANSWER.value


def test_external_evidence_counts_same_as_memory(codec) -> None:
    """外部证据（检索文档 / 工具返回值）与记忆命中必须等价。

    只认记忆会让"有检索但没进记忆库"的链路整条失效，而这正是
    RAG 最常见的接法。
    """
    bare = ChatSession(None, codec, ChatTemplate(), memory=None,
                       generator=_fixed("青奈科技的营业收入是 128 人。"))
    assert bare.ask("营业收入？").state == AbstentionState.CLARIFY.value

    with_ev = ChatSession(None, codec, ChatTemplate(), memory=None,
                          generator=_fixed("青奈科技的营业收入是 128 人。"))
    r = with_ev.ask("营业收入？", evidence="青奈科技的营业收入是 128 人。")
    assert r.state == AbstentionState.ANSWER.value


def test_gate_off_returns_raw_for_ablation(codec) -> None:
    """关闸门必须真的放行原文 —— 否则 A/B 的"无闸门"对照组不成立。"""
    s = ChatSession(None, codec, ChatTemplate(), memory=None,
                    generator=_fixed("随便编一个数字 88 亿。"),
                    use_evidence_gate=False)
    r = s.ask("营业收入？")
    assert r.state == AbstentionState.ANSWER.value
    assert r.text == r.raw_text


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------
def test_suppressed_turn_is_kept_out_of_history(codec) -> None:
    """被拒答的那句不能进历史，否则下一轮模型会把它当已确认的事实继续编。"""
    s = ChatSession(None, codec, ChatTemplate(), memory=None,
                    generator=_fixed("青奈科技 2026 年营业收入是 88 亿。"))
    s.ask("青奈科技 2026 年的营业收入是多少？")
    roles = [m["role"] for m in s.history]
    assert roles == ["user"], f"拒答回合不该写入 assistant：{s.history}"


def test_answered_turn_is_kept_in_history(codec) -> None:
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("青奈科技的营业收入是 128 人。"))
    s.ask("青奈科技的营业收入是多少？")
    assert [m["role"] for m in s.history] == ["user", "assistant"]


def test_reset_clears_history_but_not_memory(codec) -> None:
    """会话历史与跨会话记忆是两件事：重置对话不该忘掉用户是谁。"""
    mem = _mem()
    mem.add("用户的名字是小林。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("用户的名字是小林。"))
    s.ask("我叫什么？")
    assert s.history
    s.reset()
    assert s.history == []
    assert len(mem) == 1


def test_report_counts_states(codec) -> None:
    mem = _mem()
    mem.add("青奈科技的营业收入是 128 人。")
    s = ChatSession(None, codec, ChatTemplate(), memory=mem,
                    generator=_fixed("青奈科技的营业收入是 128 人。"))
    s.ask("营业收入？")
    rep = s.report()
    assert rep["turns"] == 1
    assert rep["answer_rate"] == pytest.approx(1.0)
    assert rep["memory"]["size"] == 1


def test_min_recall_score_is_low_enough_for_literal_encoder() -> None:
    """召回阈值必须低到能让字面编码工作。

    ``TextKeyEncoder`` 在"表述不同但指同一实体"的中文上余弦常常**接近 0**。
    阈值设在 0.2 以上等于静默关掉检索，而这个失效在报告里看起来
    只是"模型记忆力差"。
    """
    assert MIN_RECALL_SCORE <= 0.1
    m = _mem()
    m.add("用户的名字是小林。", key="我叫什么名字")
    assert m.search("我叫什么名字？", top_k=1), \
        "按 key 检索仍然召不回，阈值或编码器有问题"


def test_memory_hit_dataclass_is_serializable() -> None:
    h = MemoryHit(text="甲", score=0.5, meta={"k": 1})
    import json
    assert json.loads(json.dumps(h.as_dict()))["text"] == "甲"


# ---------------------------------------------------------------------------
# 与真实模型的接线（形状级）
# ---------------------------------------------------------------------------
def test_session_runs_with_real_model_and_memory(codec) -> None:
    """真实模型 + 真实记忆注入：只验证"不崩、四态可达"，不验证内容。"""
    from simplcore.core.config import SimplCoreConfig
    from simplcore.core.model import SimplCoreModel

    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=codec.size(), dim=64, num_layers=2, num_bases=4,
                    decay_max=256, local_kernel=2, num_heads=2, use_memory=True)
    model = SimplCoreModel(cfg)
    with torch.no_grad():
        model.memory_in.weight.normal_(0, 0.02)

    mem = DialogMemory(dim=cfg.dim)
    mem.add("用户的名字是小林。")
    s = ChatSession(model, codec, ChatTemplate(), memory=mem, max_new_tokens=4)
    r = s.ask("我叫什么？")
    assert r.state in {st.value for st in AbstentionState}
    assert isinstance(r.text, str)


def test_session_raises_if_memory_given_but_model_lacks_projection(codec) -> None:
    """模型没开 use_memory 却给了记忆：必须报错，不能悄悄忽略。"""
    from simplcore.core.config import SimplCoreConfig
    from simplcore.core.model import SimplCoreModel

    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=codec.size(), dim=64, num_layers=2, num_bases=4,
                    decay_max=256, local_kernel=2, num_heads=2, use_memory=False)
    model = SimplCoreModel(cfg)
    mem = DialogMemory(dim=cfg.dim)
    # 必须用 key 让检索**真的命中**，否则 mem_vec 为 None、注入路径根本
    # 不被走到，这条测试会变成一条永远通过的空断言。
    mem.add("用户的名字是小林。", key="我叫什么")
    s = ChatSession(model, codec, ChatTemplate(), memory=mem, max_new_tokens=2)
    assert mem.search("我叫什么？", top_k=1), "检索没命中，测试前提不成立"
    with pytest.raises(ValueError, match="use_memory"):
        s.ask("我叫什么？")
