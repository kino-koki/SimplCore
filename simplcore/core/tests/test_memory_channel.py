# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""记忆通道的守护测试（SFT 数据层 → 模型注入 → 训练梯度）。

这组测试防的是"记忆看起来接上了，其实没有"：
形状都对、loss 照降、参数统计正常，但记忆从未真正影响预测。
"""
from __future__ import annotations

import json

import pytest
import torch

from simplcore.core.chat import (ChatTemplate, SFTDataset, SpecialTokenCodec,
                                build_memory_encoder, collate_sft,
                                pack_sequences, _pack_rows)
from simplcore.core.config import SimplCoreConfig
from simplcore.core.dialogue_data import (MEMORY_TASKS, build_records, write_jsonl)
from simplcore.core.model import SimplCoreModel
from simplcore.core.tokenization import CharVocab

_ALL = ("用户的名字是小林阿青老周小何青奈科技的营业收入是人员专利数量"
        "市场占有率不知道一二三四五六七八九十流式卷积做"
        "我叫什么名字？请原样重复这句子不要加其他内容")


def _codec() -> SpecialTokenCodec:
    return SpecialTokenCodec(CharVocab.train([_ALL], min_freq=1))


def _suite_codec() -> SpecialTokenCodec:
    """覆盖记忆套件全部期望答案的词表。

    用 ``_codec()`` 那种小词表跑套件会被可编码性校验拦下来（这是对的：
    词表里没有"老苏"，那题本来就不可能过）。
    """
    from simplcore.core.dialogue_eval import MEMORY_DIALOGUE_SUITE
    parts = [_ALL]
    for t in MEMORY_DIALOGUE_SUITE:
        parts.append(t.memory)
        parts.append(t.check.equals or "")
        parts.extend(t.check.contains or ())
        parts.extend(m["content"] for m in t.messages)
    return SpecialTokenCodec(CharVocab.train(parts, min_freq=1))


def _write(tmp_path, records):
    p = tmp_path / "d.jsonl"
    write_jsonl(str(p), records)
    return str(p)


# ---------------------------------------------------------------------------
# 打包：记忆行必须独占
# ---------------------------------------------------------------------------
def test_exclusive_rows_are_never_packed_together() -> None:
    """带记忆的样本独占一行。

    记忆是**整行一个向量**。一旦与别的样本混行，邻居样本也会被注入
    本来不属于它的记忆 —— 而"邻居是谁"取决于打包顺序，等于给训练
    注入了一个看不见的随机变量。
    """
    ex = [([1, 2], [1, 2]), ([3], [3]), ([4, 5], [4, 5])]
    ids, lab, excl = _pack_rows(ex, seq_len=8, pad_id=0,
                                exclusive=[False, True, False])
    assert excl.count(True) == 1
    mem_row = excl.index(True)
    # 该行只含它自己那一个 token，前后都是 padding
    assert ids[mem_row].tolist() == [3] + [0] * 7


def test_exclusive_flushes_preceding_partial_row() -> None:
    """独占样本进来前，已攒的半行必须先收掉，不能被它"顶到下一行"。"""
    ex = [([1, 2], [1, 2]), ([9], [9])]
    ids, _, excl = _pack_rows(ex, seq_len=8, pad_id=0, exclusive=[False, True])
    assert excl == [False, True]
    assert ids[0].tolist() == [1, 2, 0, 0, 0, 0, 0, 0]
    assert ids[1].tolist() == [9, 0, 0, 0, 0, 0, 0, 0]


def test_pack_sequences_unchanged_without_exclusive() -> None:
    """不传 exclusive 时行为必须与旧实现逐位一致（回归保护）。"""
    ex = [([1, 2, 3], [1, 2, 3]), ([4, 5], [4, 5])]
    a = pack_sequences(ex, seq_len=4, pad_id=0)
    b = _pack_rows(ex, seq_len=4, pad_id=0)[:2]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_exclusive_length_must_match() -> None:
    with pytest.raises(ValueError, match="等长"):
        _pack_rows([([1], [1])], seq_len=4, pad_id=0, exclusive=[True, False])


# ---------------------------------------------------------------------------
# 数据集：记忆行与文本对齐
# ---------------------------------------------------------------------------
def test_dataset_returns_memory_rows_only_for_memory_records(tmp_path) -> None:
    recs = [
        {"messages": [{"role": "user", "content": "你好"},
                      {"role": "assistant", "content": "你好呀"}]},
        {"messages": [{"role": "user", "content": "我叫什么？"},
                      {"role": "assistant", "content": "小林"}],
         "memory": "用户的名字是小林。", "memory_required": True},
    ]
    ds = SFTDataset(_write(tmp_path, recs), _codec(), ChatTemplate(),
                    seq_len=48, max_len=48, memory_dim=32)
    assert ds.with_memory
    assert ds.n_memory_rows == 1
    mem = ds.memory_tensor()
    assert mem.shape[1] == 32
    nonzero = [i for i in range(mem.shape[0]) if float(mem[i].norm()) > 0]
    assert len(nonzero) == 1
    assert float(mem[nonzero[0]].norm()) == pytest.approx(1.0, abs=1e-5)


def test_dataset_item_is_three_tuple_with_memory(tmp_path) -> None:
    recs = [{"messages": [{"role": "user", "content": "我叫什么？"},
                          {"role": "assistant", "content": "小林"}],
             "memory": "用户的名字是小林。", "memory_required": True}]
    ds = SFTDataset(_write(tmp_path, recs), _codec(), ChatTemplate(),
                    seq_len=48, max_len=48, memory_dim=32)
    item = ds[0]
    assert len(item) == 3
    assert item[2].shape == (32,)


def test_dataset_item_is_two_tuple_without_memory(tmp_path) -> None:
    recs = [{"messages": [{"role": "user", "content": "你好"},
                          {"role": "assistant", "content": "你好呀"}]}]
    ds = SFTDataset(_write(tmp_path, recs), _codec(), ChatTemplate(),
                    seq_len=48, max_len=48)
    assert len(ds[0]) == 2
    assert ds.memory_tensor() is None


def test_memory_required_without_fact_is_unscored(tmp_path) -> None:
    """``memory_required`` 但没给 memory：整行不监督。

    否则等于逼模型把答案背成默认值：换个人名就错，而 loss 曲线看不出区别。
    """
    recs = [{"messages": [{"role": "user", "content": "我叫什么？"},
                          {"role": "assistant", "content": "小林"}],
             "memory_required": True}]
    ds = SFTDataset(_write(tmp_path, recs), _codec(), ChatTemplate(),
                    seq_len=48, max_len=48, memory_dim=32)
    assert int((ds._rows[1] != -100).sum()) == 0


def test_memory_without_required_flag_is_still_scored(tmp_path) -> None:
    """没标 ``memory_required`` 时正常监督 —— 记忆是额外信息，不是必需品。"""
    recs = [{"messages": [{"role": "user", "content": "我叫什么？"},
                          {"role": "assistant", "content": "小林"}],
             "memory": "用户的名字是小林。"}]
    ds = SFTDataset(_write(tmp_path, recs), _codec(), ChatTemplate(),
                    seq_len=48, max_len=48, memory_dim=32)
    assert int((ds._rows[1] != -100).sum()) > 0


def test_dataset_stats_report_memory_rows(tmp_path) -> None:
    recs = build_records(200, seed=0)
    ds = SFTDataset(_write(tmp_path, recs), _codec(), ChatTemplate(),
                    seq_len=96, max_len=96, memory_dim=32)
    st = ds.stats()
    assert "memory_rows" in st
    assert st["memory_rows"] > 0


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------
def test_collate_sft_stacks_memory() -> None:
    b = [(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
          torch.ones(8)),
         (torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
          torch.zeros(8))]
    x, y, m = collate_sft(b)
    assert m.shape == (2, 8)
    assert float(m[0].sum()) == 8.0 and float(m[1].sum()) == 0.0


def test_collate_sft_two_tuple_still_works() -> None:
    b = [(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))]
    out = collate_sft(b)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# 模型：零向量 = 不注入（数学保证）
# ---------------------------------------------------------------------------
def test_zero_memory_row_is_identity() -> None:
    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_heads=2,
                    num_bases=4, local_kernel=2, use_memory=True)
    m = SimplCoreModel(cfg)
    with torch.no_grad():
        m.memory_in.weight.normal_(0, 0.1)
    m.eval()
    x = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        a = m(x)
        b = m(x, memory=torch.zeros(2, 32))
    assert torch.equal(a, b), "零记忆行改变了输出 —— 记忆投影带了 bias？"


def test_mixed_batch_only_affects_memory_rows() -> None:
    """一个 batch 里"有记忆"和"无记忆"混排：只有前者受影响。"""
    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_heads=2,
                    num_bases=4, local_kernel=2, use_memory=True)
    m = SimplCoreModel(cfg)
    with torch.no_grad():
        m.memory_in.weight.normal_(0, 0.1)
    m.eval()
    x = torch.randint(0, 64, (2, 6))
    mem = torch.stack([torch.randn(32), torch.zeros(32)])
    with torch.no_grad():
        base = m(x)
        mixed = m(x, memory=mem)
    assert not torch.allclose(base[0], mixed[0]), "有记忆的那行没受影响"
    assert torch.allclose(base[1], mixed[1]), "无记忆的那行被误伤"


# ---------------------------------------------------------------------------
# 跨会话数据
# ---------------------------------------------------------------------------
def test_cross_session_records_carry_memory_and_flag() -> None:
    recs = build_records(600, seed=0)
    xs = [r for r in recs if r.get("memory_required")]
    assert xs, "没有任何跨会话样本"
    for r in xs:
        assert r.get("memory"), "memory_required 样本必须带 memory"
        # 记忆内容不能出现在**提问**里，否则模型不需要记忆。
        # 不能检查整条记录：复述题的标准答案就等于记忆原文，
        # 那是标签，必须存在（推理时模型看不到 assistant 轮）。
        prompt = "".join(m["content"] for m in r["messages"]
                         if m["role"] == "user")
        assert r["memory"] not in prompt, (
            f"记忆内容泄漏进了提问，这条样本不需要记忆：{prompt!r}")


def test_cross_session_name_answer_not_in_dialogue() -> None:
    """提问里不能有名字 —— 否则"被问名字就答小林"能拿满分。

    注意只看**提问侧**（user 轮）。assistant 那一轮就是名字本身，
    它是监督标签，必须存在；模型在推理时只看得到 prompt。
    """
    from simplcore.core.dialogue_data import NAMES, _cross_session_name_record
    import random
    rng = random.Random(0)
    for _ in range(40):
        rec = _cross_session_name_record(rng)
        name = [n for n in NAMES if n in rec["memory"]][0]
        users = [m["content"] for m in rec["messages"] if m["role"] == "user"]
        assert users, "样本没有 user 轮"
        for q in users:
            assert name not in q, f"{name} 泄漏进了提问：{q!r}"
        # 反过来也要成立：答案确实等于记忆里的名字，否则学的是别的东西
        assert rec["messages"][-1]["content"] == name


def test_cross_session_unknown_has_unknown_as_answer() -> None:
    """记忆里没有该字段 → 正确答案是"不知道"，不是编一个。"""
    from simplcore.core.dialogue_data import _cross_session_unknown_record
    import random
    rng = random.Random(0)
    for _ in range(20):
        rec = _cross_session_unknown_record(rng)
        assert rec["messages"][1]["content"] == "不知道"
        assert rec["memory_required"]


def test_memory_tasks_are_registered() -> None:
    from simplcore.core.dialogue_data import DEFAULT_WEIGHTS, TASKS
    for name in MEMORY_TASKS:
        assert name in TASKS and name in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS[name] > 0


def test_default_weights_give_memory_enough_share() -> None:
    """记忆任务占比必须够高，否则 memory_in 收不到足够梯度。"""
    from simplcore.core.dialogue_data import DEFAULT_WEIGHTS
    total = sum(DEFAULT_WEIGHTS.values())
    mem = sum(DEFAULT_WEIGHTS[k] for k in MEMORY_TASKS)
    assert mem / total >= 0.15


def test_records_with_memory_survive_jsonl_roundtrip(tmp_path) -> None:
    """JSONL 必须保留 memory / memory_required —— 序列化时丢字段是最常见的坑。"""
    recs = build_records(60, seed=1)
    p = _write(tmp_path, recs)
    back = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert any("memory" in r for r in back)
    assert any(r.get("memory_required") for r in back)


# ---------------------------------------------------------------------------
# 评测：记忆是否**真的**被注入，以及消融结论是否成立
# ---------------------------------------------------------------------------
def _tiny(use_memory: bool = True) -> SimplCoreModel:
    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_heads=2,
                    num_bases=4, local_kernel=2, use_memory=use_memory)
    return SimplCoreModel(cfg).eval()


def test_evaluate_injects_encoded_memory_only_when_enabled(monkeypatch) -> None:
    """带 ``memory`` 的题必须注入非零向量；不带的题必须注入 None。

    这条防的是"评测看起来有记忆、其实 generate_reply 收到的是 None"：
    那种情况下开/关记忆跑出完全一样的数字，很容易被解读成"记忆没用"。
    """
    import simplcore.core.dialogue_eval as de

    seen = []

    def fake(model, prompt, codec, max_new_tokens, temperature=0.0,
             use_state=True, memory=None):
        seen.append(None if memory is None else tuple(memory.shape))
        return {"text": "小林", "tokens": 1, "stopped": True, "leaked": False}

    monkeypatch.setattr(de, "generate_reply", fake)
    model, codec = _tiny(), _suite_codec()

    de.evaluate_dialogue(model, codec, de.MEMORY_DIALOGUE_SUITE,
                         use_memory=True)
    assert seen, "没有题目被跑到"
    assert all(s == (1, 32) for s in seen), (
        f"带 memory 的题都该注入 (1, 32)，实际 {seen}")

    seen.clear()
    de.evaluate_dialogue(model, codec, de.MEMORY_DIALOGUE_SUITE,
                         use_memory=False)
    assert all(s is None for s in seen), (
        f"关掉记忆后不该注入任何向量，实际 {seen}")


def test_memory_ablation_detects_gain(monkeypatch) -> None:
    """只有给了记忆才答对 → 必须报"有效"，且翻正数等于带记忆通过数。"""
    import simplcore.core.dialogue_eval as de

    def fake(model, prompt, codec, max_new_tokens, temperature=0.0,
             use_state=True, memory=None):
        # 有记忆就说出只有 mem_name_a 才对的名字；没记忆就说不知道
        text = "小林" if memory is not None else "不知道"
        return {"text": text, "tokens": 1, "stopped": True, "leaked": False}

    monkeypatch.setattr(de, "generate_reply", fake)
    rep = de.memory_ablation(_tiny(), _suite_codec())

    assert not rep["unencodable_targets"], "测试词表没覆盖套件答案"
    assert rep["memory_supported"]
    assert rep["n_tasks"] == len(de.MEMORY_DIALOGUE_SUITE)
    # 记忆题口径：拒答对照题不计入（否则"永远说不知道"白拿分）
    assert rep["memory_pass_with"] > rep["memory_pass_without"]
    assert rep["delta"] > 0
    assert rep["n_flipped_on"] >= 1
    assert rep["n_control_tasks"] == 1
    assert "有效" in rep["verdict"] and "作废" not in rep["verdict"]


def test_memory_ablation_flags_vacuous_suite(monkeypatch) -> None:
    """关掉记忆也全对 → 必须判"题目没考住记忆"，不能当成记忆有效。"""
    import simplcore.core.dialogue_eval as de

    def fake(model, prompt, codec, max_new_tokens, temperature=0.0,
             use_state=True, memory=None):
        # 无论有没有记忆都答"小林"：说明这题靠参数就能过
        return {"text": "小林", "tokens": 1, "stopped": True, "leaked": False}

    monkeypatch.setattr(de, "generate_reply", fake)
    rep = de.memory_ablation(_tiny(), _suite_codec(),
                             tasks=de.MEMORY_DIALOGUE_SUITE[:1])
    assert rep["memory_pass_with"] == rep["memory_pass_without"] == 1.0
    assert rep["delta"] == 0
    assert "作废" in rep["verdict"]


def test_memory_ablation_all_control_tasks(monkeypatch) -> None:
    """只有对照题时如实说"无法评估"，不能报"有效"或"作废"。"""
    import simplcore.core.dialogue_eval as de

    def fake(model, prompt, codec, max_new_tokens, temperature=0.0,
             use_state=True, memory=None):
        return {"text": "不知道", "tokens": 1, "stopped": True, "leaked": False}

    monkeypatch.setattr(de, "generate_reply", fake)
    rep = de.memory_ablation(_tiny(), _suite_codec(),
                             tasks=de.MEMORY_DIALOGUE_SUITE[-1:])
    assert rep["n_memory_tasks"] == 0
    assert "无法评估" in rep["verdict"]


def test_memory_ablation_without_memory_module() -> None:
    """模型没开 use_memory：如实报"只有无记忆基线"，不伪造消融。"""
    from simplcore.core.dialogue_eval import memory_ablation
    rep = memory_ablation(_tiny(use_memory=False), _codec(),
                          tasks=())
    assert rep["memory_supported"] is False
    assert "没有 use_memory" in rep["verdict"]


# ---------------------------------------------------------------------------
# 编码器：同模板可分离性 + 训练/评测同源
# ---------------------------------------------------------------------------
def test_fitted_encoder_separates_same_template_names() -> None:
    """同模板、只差人名的两条记忆，拟合后必须可分。

    未拟合时余弦 0.71（最坏 0.83），且**换维度无效**——相似度只由共享
    n-gram 数决定。这意味着 ``memory_in`` 收到的两条向量几乎相同，模型
    没有信息可用来选名字，只能从参数里挑一个。实测训练时正是这样：
    注入"阿岚"，模型输出了训练集里的"阿青"。
    """
    from simplcore.core.memory import TextKeyEncoder
    from simplcore.core.dialogue_data import NAMES

    texts = [f"用户的名字是{n}。" for n in NAMES]
    raw = TextKeyEncoder(192)
    fit = TextKeyEncoder(192).fit([f"用户的名字是{n}。" for n in NAMES] * 8
                                  + ["青奈科技的营业收入是 128 人。"] * 8)

    def max_pair_cos(enc):
        V = torch.stack([enc(t) for t in texts])
        S = V @ V.T
        n = len(texts)
        return max(float(S[i, j]) for i in range(n) for j in range(n) if i != j)

    before, after = max_pair_cos(raw), max_pair_cos(fit)
    assert before > 0.6, f"前提不成立：未拟合时本以为会高度相似，实际 {before:.3f}"
    assert after < 0.35, f"拟合后仍然高度相似（{after:.3f}），绑定不可能学会"


def test_encoder_state_roundtrip_is_bitwise_identical() -> None:
    from simplcore.core.memory import TextKeyEncoder
    corpus = ["用户的名字是小林。", "用户的名字是阿岚。", "青奈科技的营业收入是 128 人。"]
    enc = TextKeyEncoder(64).fit(corpus * 3)
    back = TextKeyEncoder.from_state(enc.state_dict())
    assert back.fingerprint() == enc.fingerprint()
    for t in corpus:
        assert torch.equal(back(t), enc(t))


def test_unfitted_encoder_matches_legacy_behaviour() -> None:
    """未 fit 必须与旧实现逐位一致：已有记忆库不能因为这次改动换键。"""
    from simplcore.core.memory import TextKeyEncoder, _norm_text
    enc = TextKeyEncoder(64)
    t = "用户的名字是小林。"
    # 旧实现：所有 gram 权重为 1
    import hashlib
    v = torch.zeros(64)
    for g in enc._grams(t):
        d = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
        v[d % 64] += 1.0 if (int.from_bytes(
            hashlib.blake2b((g + "#").encode(), digest_size=8).digest(), "big") & 1) else -1.0
    v = v / v.norm()
    assert torch.allclose(enc(t), v, atol=1e-6)


def test_build_memory_encoder_and_datasets_share_state(tmp_path) -> None:
    """train / val 数据集必须用同一份统计：同一条记忆编码结果必须相同。"""
    from simplcore.core.chat import build_memory_encoder

    recs = build_records(400, seed=0)
    path = _write(tmp_path, recs)
    enc = build_memory_encoder(path, 32)

    assert enc.fitted, "没有从数据里统计到任何 memory 文本"
    ds1 = SFTDataset(path, _codec(), ChatTemplate(), seq_len=96,
                     max_len=96, index_slice=(0, 200), memory_dim=32,
                     memory_encoder=enc)
    ds2 = SFTDataset(path, _codec(), ChatTemplate(), seq_len=96,
                     max_len=96, index_slice=(200, 400), memory_dim=32,
                     memory_encoder=enc)
    assert ds1.memory_encoder.fingerprint() == ds2.memory_encoder.fingerprint()

    # 同一条记忆文本，两侧编码必须逐位相同
    probe = recs[0].get("memory") or next(r["memory"] for r in recs if r.get("memory"))
    assert torch.equal(ds1.memory_encoder(probe), ds2.memory_encoder(probe))
    assert torch.equal(enc(probe), ds1.memory_encoder(probe))


def test_dataset_rejects_encoder_dim_mismatch(tmp_path) -> None:
    from simplcore.core.memory import TextKeyEncoder
    recs = build_records(40, seed=0)
    path = _write(tmp_path, recs)
    enc = TextKeyEncoder(16).fit(["用户的名字是小林。"] * 4)
    with pytest.raises(ValueError, match="维度"):
        SFTDataset(path, _codec(), ChatTemplate(), seq_len=64, max_len=64,
                   memory_dim=32, memory_encoder=enc)


# ---------------------------------------------------------------------------
# 编码器：同模板可分离性 + 训练/评测同源
# ---------------------------------------------------------------------------
def test_fitted_encoder_separates_same_template_names() -> None:
    """同模板、只差人名的两条记忆，拟合后必须可分。

    未拟合时余弦 0.71（最坏 0.83），且**换维度无效**——相似度只由共享
    n-gram 数决定。这意味着 ``memory_in`` 收到的两条向量几乎相同，模型
    没有信息可用来选名字，只能从参数里挑一个。实测训练时正是这样：
    注入"阿岚"，模型输出了训练集里的"阿青"。
    """
    from simplcore.core.memory import TextKeyEncoder
    from simplcore.core.dialogue_data import NAMES

    texts = [f"用户的名字是{n}。" for n in NAMES]
    raw = TextKeyEncoder(192)
    fit = TextKeyEncoder(192).fit([f"用户的名字是{n}。" for n in NAMES] * 8
                                  + ["青奈科技的营业收入是 128 人。"] * 8)

    def max_pair_cos(enc):
        V = torch.stack([enc(t) for t in texts])
        S = V @ V.T
        n = len(texts)
        return max(float(S[i, j]) for i in range(n) for j in range(n) if i != j)

    before, after = max_pair_cos(raw), max_pair_cos(fit)
    assert before > 0.6, f"前提不成立：未拟合时本以为会高度相似，实际 {before:.3f}"
    assert after < 0.35, f"拟合后仍然高度相似（{after:.3f}），绑定不可能学会"


def test_encoder_state_roundtrip_is_bitwise_identical() -> None:
    from simplcore.core.memory import TextKeyEncoder
    corpus = ["用户的名字是小林。", "用户的名字是阿岚。", "青奈科技的营业收入是 128 人。"]
    enc = TextKeyEncoder(64).fit(corpus * 3)
    back = TextKeyEncoder.from_state(enc.state_dict())
    assert back.fingerprint() == enc.fingerprint()
    for t in corpus:
        assert torch.equal(back(t), enc(t))


def test_unfitted_encoder_matches_legacy_behaviour() -> None:
    """未 fit 必须与旧实现逐位一致：已有记忆库不能因为这次改动换键。"""
    from simplcore.core.memory import TextKeyEncoder, _norm_text
    enc = TextKeyEncoder(64)
    t = "用户的名字是小林。"
    # 旧实现：所有 gram 权重为 1
    import hashlib
    v = torch.zeros(64)
    for g in enc._grams(t):
        d = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
        v[d % 64] += 1.0 if (int.from_bytes(
            hashlib.blake2b((g + "#").encode(), digest_size=8).digest(), "big") & 1) else -1.0
    v = v / v.norm()
    assert torch.allclose(enc(t), v, atol=1e-6)


def test_build_memory_encoder_and_datasets_share_state(tmp_path) -> None:
    """train / val 数据集必须用同一份统计：同一条记忆编码结果必须相同。"""
    from simplcore.core.chat import build_memory_encoder

    recs = build_records(400, seed=0)
    path = _write(tmp_path, recs)
    enc = build_memory_encoder(path, 32)

    assert enc.fitted, "没有从数据里统计到任何 memory 文本"
    ds1 = SFTDataset(path, _codec(), ChatTemplate(), seq_len=96,
                     max_len=96, index_slice=(0, 200), memory_dim=32,
                     memory_encoder=enc)
    ds2 = SFTDataset(path, _codec(), ChatTemplate(), seq_len=96,
                     max_len=96, index_slice=(200, 400), memory_dim=32,
                     memory_encoder=enc)
    assert ds1.memory_encoder.fingerprint() == ds2.memory_encoder.fingerprint()

    # 同一条记忆文本，两侧编码必须逐位相同
    probe = recs[0].get("memory") or next(r["memory"] for r in recs if r.get("memory"))
    assert torch.equal(ds1.memory_encoder(probe), ds2.memory_encoder(probe))
    assert torch.equal(enc(probe), ds1.memory_encoder(probe))


def test_dataset_rejects_encoder_dim_mismatch(tmp_path) -> None:
    from simplcore.core.memory import TextKeyEncoder
    recs = build_records(40, seed=0)
    path = _write(tmp_path, recs)
    enc = TextKeyEncoder(16).fit(["用户的名字是小林。"] * 4)
    with pytest.raises(ValueError, match="维度"):
        SFTDataset(path, _codec(), ChatTemplate(), seq_len=64, max_len=64,
                   memory_dim=32, memory_encoder=enc)


# ---------------------------------------------------------------------------
# 评测题的静态可行性：答案必须能被词表编码出来
# ---------------------------------------------------------------------------
def test_memory_suite_targets_are_all_encodable() -> None:
    """套件里每一题的期望答案都必须能编码 —— 否则那题永远不可能通过。

    实测踩过：``mem_name_b`` 曾用答案 ``阿岚``，而 ``岚`` 不在训练语料里，
    编码为 ``[0]``（<unk>）。模型无法生成该字符，题目从写下那一刻就没有解，
    但报告只会显示"失败"，读起来像"记忆没学会"。
    """
    from simplcore.core.dialogue_eval import (MEMORY_DIALOGUE_SUITE,
                                             unencodable_targets)

    recs = build_records(800, seed=0)
    codec = SpecialTokenCodec(CharVocab.train(
        [m["content"] for r in recs for m in r["messages"]], min_freq=1))
    bad = unencodable_targets(MEMORY_DIALOGUE_SUITE, codec)
    assert not bad, f"以下题目的答案不可编码，永远无法通过：{bad}"


def test_unencodable_targets_detects_unk() -> None:
    from simplcore.core.dialogue_eval import DialogueTask, Check, unencodable_targets

    codec = _codec()
    assert 0 in codec.encode("岚"), "前提：岚 不在词表里"
    ok = DialogueTask("ok", [{"role": "user", "content": "问"}],
                      Check(contains=["小林"]))
    bad = DialogueTask("bad", [{"role": "user", "content": "问"}],
                       Check(contains=["阿岚"]))
    got = unencodable_targets([ok, bad], codec)
    assert [g["id"] for g in got] == ["bad"]
    assert got[0]["target"] == "阿岚"


def test_ablation_refuses_when_target_unencodable(monkeypatch) -> None:
    """有不可编码的题时，结论必须是"结果无意义"，不能报有效/无效。"""
    import simplcore.core.dialogue_eval as de

    def fake(model, prompt, codec, max_new_tokens, temperature=0.0,
             use_state=True, memory=None):
        return {"text": "小林", "tokens": 1, "stopped": True, "leaked": False}

    monkeypatch.setattr(de, "generate_reply", fake)
    task = de.DialogueTask("bad", [{"role": "user", "content": "我叫什么？"}],
                           de.Check(contains=["阿岚"]), memory="用户的名字是阿岚。")
    rep = de.memory_ablation(_tiny(), _codec(), tasks=[task])
    assert rep["unencodable_targets"], "没有检出不可编码的目标"
    assert "无意义" in rep["verdict"]


def test_encoder_survives_disk_roundtrip_for_inference(tmp_path) -> None:
    """训练落盘的编码器，推理侧读回后必须逐位一致。

    否则 ``memory_in`` 见到的是另一种"记忆语言"：不报错、输出仍是人话，
    只是答非所问。这类错在服务端最难查。
    """
    from simplcore.core.chat import load_memory_encoder
    from simplcore.core.dialogue_runtime import DialogMemory

    recs = build_records(300, seed=0)
    path = _write(tmp_path, recs)
    enc = build_memory_encoder(path, 32)

    fp = tmp_path / "encoder.json"
    fp.write_text(json.dumps(enc.state_dict(), ensure_ascii=False),
                  encoding="utf-8")
    back = load_memory_encoder(str(fp), dim=32)
    assert back.fingerprint() == enc.fingerprint()

    probe = next(r["memory"] for r in recs if r.get("memory"))
    assert torch.equal(back(probe), enc(probe))

    with pytest.raises(ValueError, match="维度"):
        load_memory_encoder(str(fp), dim=48)

    # 运行时记忆库用同一份编码器 → 写入的键与训练完全一致
    mem = DialogMemory(dim=32, encoder=back)
    mem.add(probe)
    assert torch.equal(mem._encoder(probe), enc(probe))


def test_projection_report_detects_rank_one_collapse() -> None:
    """秩一 memory_in → 所有记忆投到同一条线上，必须判塌缩。"""
    from simplcore.core.dialogue_eval import memory_projection_report
    from simplcore.core.memory import TextKeyEncoder

    m = _tiny()
    D = 32
    enc = TextKeyEncoder(D).fit(
        [f"用户的名字是{n}。" for n in ("小林", "老周", "阿苏", "小何")] * 4)
    texts = ["用户的名字是小林。", "用户的名字是老周。", "用户的名字是阿苏。"]

    with torch.no_grad():
        u = torch.randn(D)
        m.memory_in.weight.copy_(torch.outer(u, u))     # 秩一
    rep = memory_projection_report(m, texts, encoder=enc)
    assert rep["supported"]
    assert rep["collapsed"], f"没判出塌缩：{rep}"
    assert rep["projected"]["abs_max"] > 0.95


def test_projection_report_passes_for_separating_map() -> None:
    """接近恒等的投影应保持区分度，不报塌缩。"""
    from simplcore.core.dialogue_eval import memory_projection_report
    from simplcore.core.memory import TextKeyEncoder

    m = _tiny()
    enc = TextKeyEncoder(32).fit(
        [f"用户的名字是{n}。" for n in ("小林", "老周", "阿苏", "小何")] * 4)
    texts = ["用户的名字是小林。", "用户的名字是老周。", "用户的名字是阿苏。"]
    with torch.no_grad():
        m.memory_in.weight.copy_(torch.eye(32) * 2.0)
    rep = memory_projection_report(m, texts, encoder=enc)
    assert not rep["collapsed"], rep
    assert rep["input"]["abs_max"] < 0.5


def test_projection_report_detects_rank_one_collapse() -> None:
    """秩一 memory_in → 所有记忆投到同一条线上，必须判塌缩。"""
    from simplcore.core.dialogue_eval import memory_projection_report
    from simplcore.core.memory import TextKeyEncoder

    m = _tiny()
    D = 32
    enc = TextKeyEncoder(D).fit(
        [f"用户的名字是{n}。" for n in ("小林", "老周", "阿苏", "小何")] * 4)
    texts = ["用户的名字是小林。", "用户的名字是老周。", "用户的名字是阿苏。"]

    with torch.no_grad():
        u = torch.randn(D)
        m.memory_in.weight.copy_(torch.outer(u, u))     # 秩一
    rep = memory_projection_report(m, texts, encoder=enc)
    assert rep["supported"]
    assert rep["collapsed"], f"没判出塌缩：{rep}"
    assert rep["projected"]["abs_max"] > 0.95


def test_projection_report_passes_for_separating_map() -> None:
    """接近恒等的投影应保持区分度，不报塌缩。"""
    from simplcore.core.dialogue_eval import memory_projection_report
    from simplcore.core.memory import TextKeyEncoder

    m = _tiny()
    enc = TextKeyEncoder(32).fit(
        [f"用户的名字是{n}。" for n in ("小林", "老周", "阿苏", "小何")] * 4)
    texts = ["用户的名字是小林。", "用户的名字是老周。", "用户的名字是阿苏。"]
    with torch.no_grad():
        m.memory_in.weight.copy_(torch.eye(32) * 2.0)
    rep = memory_projection_report(m, texts, encoder=enc)
    assert not rep["collapsed"], rep
    assert rep["input"]["abs_max"] < 0.5
