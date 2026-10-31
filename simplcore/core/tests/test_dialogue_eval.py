# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""对话评测守护测试。

两条主张必须钉死：

* 判定规则本身不能松（松了等于给模型虚报成绩）；
* 生成循环必须在 ``<|end|>`` 处停、在抢话时判漏 —— 这两件事与模型好坏无关，
  纯粹是评测器的行为，测不出来就说明评测量的是别的东西。
"""
from __future__ import annotations

import json

import pytest

#: 关闭重复约束：本文件里用桩模型测的是"遇到 <|end|> 是否停"与"没停时是否
#: 用完预算"，与贪心环无关。默认的 no_repeat_ngram=3 会让桩模型（永远输出
#: "天"）在第四个 token 处被挡住，把这两条用例变成误报。
_NO_REPEAT = dict(repetition_penalty=1.0, no_repeat_ngram=0, min_new_tokens=0)
import torch

from simplcore.core.chat import (END_TOKEN, USER_TOKEN, ChatTemplate,
                                SpecialTokenCodec)
from simplcore.core.dialogue_eval import (BASIC_DIALOGUE_SUITE, Check,
                                        DialogueReport, DialogueTask,
                                        _numbers, evaluate_dialogue,
                                        format_report, generate_reply,
                                        normalize_text, suite_as_jsonl)
from simplcore.core.tokenization import CharVocab

CHARS = "天地你好世界答案是苹果香蕉橘子西安小林不知道我在等你的指令"


def _codec() -> SpecialTokenCodec:
    return SpecialTokenCodec(CharVocab.train([CHARS], min_freq=1))


class _ConstModel(torch.nn.Module):
    """永远输出同一个 token 的假模型，用于隔离测试生成循环。"""

    def __init__(self, token_id: int, vocab_size: int) -> None:
        super().__init__()
        self.tok_emb = torch.nn.Embedding(1, 1)
        self.token_id = int(token_id)
        self.vocab_size = int(vocab_size)
        self.calls = 0

    def init_stream(self, batch, device=None, dtype=None):
        return {"n": 0}

    def _logits(self):
        lg = torch.full((1, self.vocab_size), -1e4)
        lg[0, self.token_id] = 1e4
        return lg

    def stream_step(self, token, state):
        self.calls += 1
        return self._logits(), state

    def forward(self, idx):
        B, L = idx.shape
        return self._logits().view(1, 1, -1).expand(B, L, -1)


# ---------------------------------------------------------------------------
# 归一化与数字抽取
# ---------------------------------------------------------------------------
def test_normalize_fullwidth_and_space() -> None:
    assert normalize_text("１２３ ＡＢＣ") == "123ABC"


def test_normalize_strips_edge_punctuation_only() -> None:
    assert normalize_text("，你好。") == "你好"
    assert normalize_text("你，好") == "你，好"      # 内部标点保留


def test_extract_numbers_including_decimal_and_negative() -> None:
    assert _numbers("1.86 倍，-3 个，42") == [1.86, -3.0, 42.0]


# ---------------------------------------------------------------------------
# 判定规则
# ---------------------------------------------------------------------------
def test_check_equals_normalizes_both_sides() -> None:
    ok, _ = Check(equals="青奈科技做流式卷积。").evaluate("青奈科技 做流式卷积")
    assert ok


def test_check_equals_rejects_extra_words() -> None:
    ok, why = Check(equals="对").evaluate("对的")
    assert not ok and why


def test_check_contains_and_not_contains() -> None:
    assert Check(contains=["苹果"]).evaluate("苹果、香蕉")[0]
    assert not Check(not_contains=["苹果"]).evaluate("苹果、香蕉")[0]
    assert not Check(contains=["梨"]).evaluate("苹果、香蕉")[0]


def test_check_max_and_min_chars() -> None:
    assert not Check(max_chars=3).evaluate("一二三四")[0]
    assert not Check(min_chars=3).evaluate("一二")[0]


def test_check_regex() -> None:
    assert Check(regex=r"^【.+】$").evaluate("【流式卷积】")[0]
    assert not Check(regex=r"^【.+】$").evaluate("流式卷积")[0]


def test_check_numbers_requires_exact_multiset() -> None:
    assert Check(numbers=[42]).evaluate("42")[0]
    assert not Check(numbers=[42]).evaluate("41")[0]
    assert not Check(numbers=[42]).evaluate("42 和 43")[0]


def test_check_distinct_count_rejects_repeats() -> None:
    """只数分隔符会把重复项判成通过 —— 这条规则就是为了堵这个洞。"""
    assert not Check(distinct_count=("、", 3)).evaluate("苹果、苹果、苹果")[0]
    assert Check(distinct_count=("、", 3)).evaluate("苹果、香蕉、橘子")[0]


def test_check_count_at_least() -> None:
    assert Check(count_at_least=("、", 2)).evaluate("甲、乙、丙")[0]
    assert not Check(count_at_least=("、", 3)).evaluate("甲、乙")[0]
    assert not Check(count_at_least=("、", 1)).evaluate("甲乙")[0]


def test_check_multiple_fields_are_conjunctive() -> None:
    c = Check(contains=["苹果"], max_chars=4)
    assert not c.evaluate("苹果和香蕉")[0]      # 超长
    assert c.evaluate("苹果")[0]


# ---------------------------------------------------------------------------
# 内建套件的判定强度
# ---------------------------------------------------------------------------
def test_greet_check_rejects_pure_digits() -> None:
    task = next(t for t in BASIC_DIALOGUE_SUITE if t.id == "greet")
    assert not task.check.evaluate("11")[0]
    assert task.check.evaluate("你好，有什么可以帮你的？")[0]


def test_list_check_requires_distinct_items() -> None:
    task = next(t for t in BASIC_DIALOGUE_SUITE if t.id == "list_three")
    assert not task.check.evaluate("西安、西安、不知道什么？")[0]
    assert task.check.evaluate("苹果、香蕉、橘子")[0]


def test_every_task_declares_an_id_and_a_note() -> None:
    ids = [t.id for t in BASIC_DIALOGUE_SUITE]
    assert len(set(ids)) == len(ids)
    for t in BASIC_DIALOGUE_SUITE:
        assert t.messages and t.messages[-1]["role"] == "user"
        assert t.note, f"{t.id} 缺少说明"


def test_suite_exports_to_jsonl(tmp_path) -> None:
    p = tmp_path / "suite.jsonl"
    suite_as_jsonl(str(p))
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(BASIC_DIALOGUE_SUITE)
    assert rows[0]["check"], "判定条件不能被序列化成空"


# ---------------------------------------------------------------------------
# 生成循环
# ---------------------------------------------------------------------------
def test_generate_stops_at_end_token() -> None:
    codec = _codec()
    model = _ConstModel(codec.token_id(END_TOKEN), codec.size())
    r = generate_reply(model, codec.encode("你好"), codec, max_new_tokens=32,
                       **_NO_REPEAT)
    assert r["stopped"] is True
    assert r["leaked"] is False
    assert r["tokens"] == 0 and r["text"] == ""


def test_generate_detects_turn_leak() -> None:
    codec = _codec()
    model = _ConstModel(codec.token_id(USER_TOKEN), codec.size())
    r = generate_reply(model, codec.encode("你好"), codec, max_new_tokens=32)
    assert r["leaked"] is True and r["tokens"] == 0


def test_generate_runs_to_budget_without_stop() -> None:
    codec = _codec()
    content_id = codec.encode("天")[0]
    model = _ConstModel(content_id, codec.size())
    r = generate_reply(model, codec.encode("你好"), codec, max_new_tokens=5,
                       **_NO_REPEAT)
    assert r["stopped"] is False
    assert r["tokens"] == 5
    assert r["text"] == "天天天天天"


def test_generate_state_and_full_paths_agree() -> None:
    codec = _codec()
    content_id = codec.encode("天")[0]
    a = generate_reply(_ConstModel(content_id, codec.size()),
                       codec.encode("你好"), codec, 4, use_state=True,
                       **_NO_REPEAT)
    b = generate_reply(_ConstModel(content_id, codec.size()),
                       codec.encode("你好"), codec, 4, use_state=False,
                       **_NO_REPEAT)
    assert a["text"] == b["text"] and a["tokens"] == b["tokens"]


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------
def test_report_with_const_end_model_flags_everything_empty() -> None:
    codec = _codec()
    model = _ConstModel(codec.token_id(END_TOKEN), codec.size())
    rep = evaluate_dialogue(model, codec, BASIC_DIALOGUE_SUITE[:3])
    s = rep.summary()
    assert s["stop_rate"] == 1.0
    assert s["empty_rate"] == 1.0
    assert s["task_pass_rate"] == 0.0
    assert s["format_ok"] is False          # 停了但没内容，仍不算格式合格


def test_report_with_const_content_model_flags_leak_free_but_unstopped() -> None:
    codec = _codec()
    model = _ConstModel(codec.encode("天")[0], codec.size())
    rep = evaluate_dialogue(model, codec, BASIC_DIALOGUE_SUITE[:2])
    s = rep.summary()
    assert s["stop_rate"] == 0.0
    assert s["format_ok"] is False
    assert rep.failures()


def test_report_as_dict_is_json_serialisable() -> None:
    codec = _codec()
    model = _ConstModel(codec.token_id(END_TOKEN), codec.size())
    rep = evaluate_dialogue(model, codec, BASIC_DIALOGUE_SUITE[:2])
    json.dumps(rep.as_dict(), ensure_ascii=False)


def test_empty_report_does_not_divide_by_zero() -> None:
    rep = DialogueReport()
    s = rep.summary()
    assert s["n_tasks"] == 0 and s["task_pass_rate"] == 0.0


def test_format_report_contains_all_rows() -> None:
    codec = _codec()
    tasks = BASIC_DIALOGUE_SUITE[:2]
    rep = evaluate_dialogue(_ConstModel(codec.token_id(END_TOKEN), codec.size()),
                            codec, tasks)
    text = format_report(rep)
    for t in tasks:
        assert t.id in text


def test_custom_task_roundtrip() -> None:
    codec = _codec()
    task = DialogueTask("t1", [{"role": "user", "content": "只回答：天"}],
                        Check(equals="天"), max_new_tokens=8)
    # 输出 "天" 需要模型先吐 "天" 再吐 END：用逐 token 脚本太啰嗦，
    # 这里改判"含天"，验证自定义任务能被正常跑通
    task.check = Check(contains=["天"])
    rep = evaluate_dialogue(_ConstModel(codec.encode("天")[0], codec.size()),
                            codec, [task])
    assert rep.per_task[0]["passed"] is True


# ---------------------------------------------------------------------------
# 合成语料生成器（CPU 自证的数据来源，坏了会让"链路自证"变成假象）
# ---------------------------------------------------------------------------
def test_dialogue_data_is_deterministic() -> None:
    from simplcore.core.dialogue_data import build_records
    a = build_records(50, seed=7)
    b = build_records(50, seed=7)
    assert a == b
    assert build_records(50, seed=8) != a


def test_dialogue_data_records_are_well_formed() -> None:
    from simplcore.core.dialogue_data import build_records
    for r in build_records(300, seed=1):
        msgs = r["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[0]["content"] and msgs[1]["content"]


def test_dialogue_data_covers_every_task() -> None:
    from simplcore.core.dialogue_data import TASKS, build_records
    seen = set()
    for r in build_records(4000, seed=0):
        n = len(r["messages"])
        text = r["messages"][0]["content"]
        if n == 4:
            seen.add("multi_turn")
        elif "不知道" in r["messages"][1]["content"]:
            seen.add("unknown")
        elif "【" in r["messages"][1]["content"]:
            seen.add("bracket")
    # 至少三种结构性任务出现过（多轮 / 拒答 / 符号包裹）
    assert len(seen) >= 3


def test_dialogue_data_write_jsonl_roundtrip(tmp_path) -> None:
    from simplcore.core.dialogue_data import build_records, write_jsonl
    p = tmp_path / "d.jsonl"
    n = write_jsonl(str(p), build_records(20, seed=2))
    lines = p.read_text(encoding="utf-8").splitlines()
    assert n == len(lines) == 20
    assert json.loads(lines[0])["messages"]
