# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""对话数据层守护测试：特殊 token、模板、掩码、偏移、打包、词表。

这些断言针对的是"**不会让形状测试失败、只会让训练白跑**"的缺陷。
最典型的一条是偏移：``SFTDataset`` 若返回未错位的 ``(x, x)``，
``cross_entropy`` 学到的就是"复制输入"，loss 会漂亮地掉到 0.1 以下，
而模型一个字也不会说。
"""
from __future__ import annotations

import json

import pytest
import torch

from simplcore.core.chat import (ASSISTANT_TOKEN, END_TOKEN, IGNORE_INDEX,
                                PAD_TOKEN, SPECIAL_TOKENS, SYSTEM_TOKEN,
                                USER_TOKEN, ChatTemplate, SFTDataset,
                                SpecialTokenCodec, build_sft_dataset,
                                collate_sft, normalise_record, pack_sequences)
from simplcore.core.tokenization import CharVocab


# ---------------------------------------------------------------------------
# 词表
# ---------------------------------------------------------------------------
def test_charvocab_id_zero_is_unk_and_order_is_frequency() -> None:
    v = CharVocab.train(["aaabbc", "bb"], min_freq=1)
    assert v._chars[0] == CharVocab.UNK_TOKEN
    assert v.encode("b")[0] < v.encode("a")[0] < v.encode("c")[0]   # 频率降序
    assert v.encode("z") == [0]


def test_charvocab_roundtrip_and_save_load(tmp_path) -> None:
    v = CharVocab.train(["你好世界", "世界和平"], min_freq=1)
    s = "你好世界"
    assert v.decode(v.encode(s)) == s
    p = tmp_path / "v.json"
    v.save(str(p))
    v2 = CharVocab.load(str(p))
    assert v2.size() == v.size()
    assert v2.encode(s) == v.encode(s)


def test_charvocab_is_deterministic_across_instances() -> None:
    """同一语料两次训练必须给出同一 ID 映射（跨进程可复现）。"""
    texts = ["甲乙丙丁", "丙丁戊"]
    a = CharVocab.train(texts, min_freq=1)
    b = CharVocab.train(texts, min_freq=1)
    assert a._chars == b._chars


def test_charvocab_min_freq_and_max_size() -> None:
    v = CharVocab.train(["aab", "aac", "d"], min_freq=2, max_size=3)
    assert v.size() <= 3
    assert v.encode("d") == [0]        # 只出现一次 → unk


def test_charvocab_rejects_foreign_file(tmp_path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"kind": "bpe", "chars": ["<unk>"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        CharVocab.load(str(p))


def test_charvocab_rejects_shifted_id_zero(tmp_path) -> None:
    p = tmp_path / "bad2.json"
    p.write_text(json.dumps({"kind": "charvocab", "chars": ["你", "好"]}),
                 encoding="utf-8")
    with pytest.raises(ValueError):
        CharVocab.load(str(p))


# ---------------------------------------------------------------------------
# 特殊 token
# ---------------------------------------------------------------------------
# 词表刻意覆盖测试里会用到的**全部**字符：``CharVocab`` 会把未见字符映射到
# id 0，而 ``decode`` 会丢弃 unk —— 词表不全时"断言字符不出现"会假通过。
_ALL_CHARS = "你好世界客户端天地一二三四问答题 abcdefg很长用户问题只有没有"


def _codec(texts=(_ALL_CHARS,)) -> SpecialTokenCodec:
    return SpecialTokenCodec(CharVocab.train(list(texts), min_freq=1))


def test_special_codec_keeps_base_ids_unchanged() -> None:
    base = CharVocab.train(["你好世界"], min_freq=1)
    codec = SpecialTokenCodec(base)
    assert codec.encode("你好") == base.encode("你好")
    assert codec.size() == base.size() + len(SPECIAL_TOKENS)


def test_special_ids_are_unique_and_ordered() -> None:
    codec = _codec()
    ids = [codec.token_id(t) for t in SPECIAL_TOKENS]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert codec.token_id(PAD_TOKEN) == codec.offset


def test_special_roundtrip_both_modes() -> None:
    codec = _codec()
    s = f"{USER_TOKEN}你好{END_TOKEN}{ASSISTANT_TOKEN}世界{END_TOKEN}"
    ids = codec.encode(s)
    assert codec.decode(ids) == s
    assert codec.decode(ids, keep_special=False) == "你好世界"


def test_special_decode_handles_empty_and_plain() -> None:
    codec = _codec()
    assert codec.decode([]) == ""
    assert codec.decode(codec.encode("你好")) == "你好"


def test_unknown_special_is_rejected_at_construction() -> None:
    with pytest.raises(NotImplementedError):
        SpecialTokenCodec(CharVocab.train(["a"]), allow_unknown_special=True)


def test_token_id_raises_for_unregistered() -> None:
    with pytest.raises(KeyError):
        _codec().token_id("<|tool|>")


# ---------------------------------------------------------------------------
# 模板
# ---------------------------------------------------------------------------
def test_template_renders_exact_layout() -> None:
    t = ChatTemplate()
    s = t.render([{"role": "user", "content": "问"},
                  {"role": "assistant", "content": "答"}])
    assert s == f"{USER_TOKEN}问{END_TOKEN}{ASSISTANT_TOKEN}答{END_TOKEN}"


def test_template_generation_prompt_ends_with_assistant() -> None:
    t = ChatTemplate()
    s = t.render([{"role": "user", "content": "问"}], add_generation_prompt=True)
    assert s.endswith(ASSISTANT_TOKEN)
    assert USER_TOKEN in s


def test_template_system_injection() -> None:
    t = ChatTemplate(system="你是助手")
    s = t.render([{"role": "user", "content": "问"}])
    assert s.startswith(f"{SYSTEM_TOKEN}你是助手{END_TOKEN}")


def test_template_explicit_system_wins_over_default() -> None:
    t = ChatTemplate(system="默认")
    s = t.render([{"role": "system", "content": "显式"},
                  {"role": "user", "content": "问"}])
    assert "显式" in s and "默认" not in s


def test_template_rejects_unknown_role() -> None:
    with pytest.raises(ValueError):
        ChatTemplate().render([{"role": "tool", "content": "x"}])


# ---------------------------------------------------------------------------
# 掩码
# ---------------------------------------------------------------------------
def test_only_assistant_content_and_end_are_supervised() -> None:
    codec = _codec()
    ids, labels = ChatTemplate().encode_example(
        [{"role": "user", "content": "你好"},
         {"role": "assistant", "content": "世界"}], codec)
    trained = codec.decode([i for i, v in zip(ids, labels) if v != IGNORE_INDEX])
    assert trained == f"世界{END_TOKEN}"
    # prompt 段一个都不能被训练
    prompt = ChatTemplate().prompt_ids(codec, [{"role": "user", "content": "你好"}])
    n_prompt = len(prompt)
    assert all(v == IGNORE_INDEX for v in labels[:n_prompt])


def test_end_token_is_supervised_because_it_is_the_stop_action() -> None:
    codec = _codec()
    ids, labels = ChatTemplate().encode_example(
        [{"role": "user", "content": "问"},
         {"role": "assistant", "content": "答"}], codec)
    end_id = codec.token_id(END_TOKEN)
    # 最后一个 END 必须被训练
    assert labels[-1] == end_id
    assert ids[-1] == end_id


def test_multi_turn_masks_every_prompt_segment() -> None:
    codec = _codec()
    ids, labels = ChatTemplate().encode_example(
        [{"role": "user", "content": "一"},
         {"role": "assistant", "content": "二"},
         {"role": "user", "content": "三"},
         {"role": "assistant", "content": "四"}], codec)
    trained = codec.decode([i for i, v in zip(ids, labels) if v != IGNORE_INDEX])
    assert "二" in trained and "四" in trained
    assert "一" not in trained and "三" not in trained


def test_truncation_right_keeps_head_left_keeps_tail() -> None:
    codec = _codec()
    msgs = [{"role": "user", "content": "一二三四"},
            {"role": "assistant", "content": "abcdefg"}]
    r = ChatTemplate().encode_example(msgs, codec, max_len=8, truncate="right")
    l = ChatTemplate().encode_example(msgs, codec, max_len=8, truncate="left")
    assert r is not None and l is not None
    assert codec.decode(r[0]).startswith(USER_TOKEN)
    assert codec.decode(l[0]).endswith(END_TOKEN)


def test_drop_when_no_target_survives_truncation() -> None:
    codec = _codec()
    msgs = [{"role": "user", "content": "很长很长的用户问题"},
            {"role": "assistant", "content": "答"}]
    # 只保留前 3 个 token → 一个 assistant token 都不剩
    assert ChatTemplate().encode_example(msgs, codec, max_len=3) is None


# ---------------------------------------------------------------------------
# 偏移（关键回归）
# ---------------------------------------------------------------------------
def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_sft_targets_are_shifted_by_one_token(tmp_path) -> None:
    """``labels[t]`` 必须是 ``input_ids[t+1]`` 位置的监督信号。

    若这条失败，训练就在学"复制输入"：loss 会掉得极快，生成却只有标记。
    """
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [
        {"messages": [{"role": "user", "content": "你好"},
                      {"role": "assistant", "content": "世界"}]},
    ] * 3)
    codec = _codec()
    ds = SFTDataset(str(path), codec, seq_len=64)
    x, y = ds[0]
    # 偏移后长度少一
    assert x.shape[0] == y.shape[0] == 63
    ids, labels = ChatTemplate().encode_example(
        [{"role": "user", "content": "你好"},
         {"role": "assistant", "content": "世界"}], codec)
    assert x.tolist()[:len(ids) - 1] == ids[:-1]
    assert y.tolist()[:len(labels) - 1] == labels[1:]


def test_shifted_first_target_after_assistant_marker_is_content() -> None:
    path_text = [
        {"messages": [{"role": "user", "content": "问"},
                      {"role": "assistant", "content": "答"}]}] * 2
    codec = _codec()
    msgs = path_text[0]["messages"]
    ids, labels = ChatTemplate().encode_example(msgs, codec)
    a_pos = ids.index(codec.token_id(ASSISTANT_TOKEN))
    # 模型在 a_pos 处看到的输入，其目标必须是 assistant 的第一个内容 token
    assert labels[a_pos + 1] == ids[a_pos + 1]
    assert codec.decode([labels[a_pos + 1]]) == "答"


# ---------------------------------------------------------------------------
# 打包
# ---------------------------------------------------------------------------
def test_pack_fills_rows_and_pads_labels_with_ignore() -> None:
    ex = [([1, 2, 3], [IGNORE_INDEX, 2, 3]), ([4, 5], [4, 5])]
    x, y = pack_sequences(ex, seq_len=4, pad_id=0)
    assert x.shape == y.shape == (2, 4)
    # 3 + 2 > 4，所以第二条另起一行（贪心但绝不越界）
    assert x[0].tolist() == [1, 2, 3, 0]
    assert x[1].tolist() == [4, 5, 0, 0]
    assert y[0].tolist() == [IGNORE_INDEX, 2, 3, IGNORE_INDEX]
    assert y[1].tolist() == [4, 5, IGNORE_INDEX, IGNORE_INDEX]


def test_pack_never_exceeds_seq_len() -> None:
    ex = [([7] * 10, [1] * 10), ([8] * 10, [1] * 10)]
    x, y = pack_sequences(ex, seq_len=8, pad_id=0)
    assert x.shape[1] == 8
    assert all(int(v) <= 8 for v in x.flatten())


def test_pack_truncates_overlong_example() -> None:
    x, y = pack_sequences([([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])], seq_len=3, pad_id=0)
    assert x[0].tolist() == [1, 2, 3]


def test_pack_greedily_joins_when_it_fits() -> None:
    ex = [([1, 2], [1, 2]), ([3, 4], [3, 4])]
    x, y = pack_sequences(ex, seq_len=4, pad_id=0)
    assert x[0].tolist() == [1, 2, 3, 4]
    assert y[0].tolist() == [1, 2, 3, 4]


def test_pack_drop_remainder_and_empty() -> None:
    x, _ = pack_sequences([([1, 2], [1, 2])], seq_len=8, pad_id=0, drop_remainder=True)
    assert x.shape == (0, 8)
    x2, y2 = pack_sequences([], seq_len=8, pad_id=0)
    assert x2.shape == (0, 8) and y2.shape == (0, 8)


def test_pack_rejects_bad_seq_len() -> None:
    with pytest.raises(ValueError):
        pack_sequences([([1], [1])], seq_len=0, pad_id=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_dataset_stats_report_supervised_ratio(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "问"},
                                      {"role": "assistant", "content": "答"}]}] * 20)
    ds = SFTDataset(str(path), _codec(), seq_len=32)
    st = ds.stats()
    assert st["records"] == 20 and st["records_used"] == 20
    assert 0.0 < st["supervised_ratio"] < 1.0


def test_dataset_index_slice_is_disjoint(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": f"问{i}"},
                                      {"role": "assistant", "content": "答"}]}
                        for i in range(20)])
    codec = _codec(("问0123456789", "答"))
    a = SFTDataset(str(path), codec, seq_len=32, index_slice=(0, 15))
    b = SFTDataset(str(path), codec, seq_len=32, index_slice=(15, 20))
    assert a.stats()["records"] == 15 and b.stats()["records"] == 5


def test_dataset_skips_prompt_only_records(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "只有问题没有回答"}]}] * 3
                 + [{"messages": [{"role": "user", "content": "问"},
                                  {"role": "assistant", "content": "答"}]}])
    ds = SFTDataset(str(path), _codec(), seq_len=32)
    assert ds.stats()["records_skipped"] == 3


def test_dataset_empty_raises_with_actionable_message(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "x"}]}])
    with pytest.raises(ValueError, match="一条可用样本都没有"):
        SFTDataset(str(path), _codec(), seq_len=16)


def test_collate_stacks_fixed_rows(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "问"},
                                      {"role": "assistant", "content": "答"}]}] * 8)
    ds = SFTDataset(str(path), _codec(), seq_len=16)
    x, y = collate_sft([ds[0], ds[1]])
    assert x.shape == y.shape == (2, 15)


# ---------------------------------------------------------------------------
# 兼容三种 schema
# ---------------------------------------------------------------------------
def test_normalise_messages_schema() -> None:
    r = normalise_record({"messages": [{"role": "user", "content": "a"}]})
    assert r["messages"][0]["role"] == "user"


def test_normalise_sharegpt_schema() -> None:
    r = normalise_record({"conversations": [["human", "你好"], ["gpt", "你好呀"]]})
    assert [m["role"] for m in r["messages"]] == ["user", "assistant"]


def test_normalise_prompt_response_schema() -> None:
    r = normalise_record({"prompt": "q", "response": "a"})
    assert r["messages"] == [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "a"}]


def test_normalise_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="无法识别"):
        normalise_record({"q": 1})


def test_build_sft_dataset_helper(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, [{"messages": [{"role": "user", "content": "问"},
                                      {"role": "assistant", "content": "答"}]}] * 4)
    ds = build_sft_dataset(str(path), _codec(), seq_len=32)
    assert len(ds) > 0
