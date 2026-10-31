# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC 数据管道测试 (fast tests for the real-data pipeline).

设计原则 (design rules):
  * **不需要完整下载**: 全部数据都是 ``tmp_path`` 里现场生成的假 parquet;
  * **快**: 不训练 32k 词表, 只训练几百个 token; 整个文件目标 < 30 s;
  * **测真东西**: 断言 manifest 的 SHA-256 与实际文件一致、train/val/test 在
    文档级互不相交、``ShardedTokenDataset`` 的窗口形状/内容与直接读 ``.bin`` 一致。

为什么"文档级不相交"要这样测 (how the leak test works)
-------------------------------------------------------
每个假文档都带一个**唯一**编号 ``编号####``, 并且按归属组携带只在组内出现的短语
(``蓝色海豚`` 只在 test 文档里 / ``红色鲸鱼`` 只在 val 文档里)。若切分把同一篇
文档的 token 同时放进两个 split, 那么在"另一个" split 的 token 流里一定能扫到
该文档的编号, 断言就会失败。这比"窗口索引不重叠"强得多, 因为窗口索引本来就不可能重叠。
"""
from __future__ import annotations

import hashlib
import json
import re
import time

import numpy as np
import pytest
import torch

from simplcore.core import data as D
from simplcore.core import tokenizer as T

EOS_CHAR = T.DEFAULT_EOS_TOKEN
TEST_MARKER = "蓝色海豚"
VAL_MARKER = "红色鲸鱼"
DOC_MARKER = "编号"

#: 两个假分片的行数 (缺一不可, 测试里直接断言这个和)。
SHARD1_DOCS = 15 + 10 + 6
SHARD2_DOCS = 15 + 10
TOTAL_DOCS = SHARD1_DOCS + SHARD2_DOCS

#: 清洗后的假文档长度必须明显大于 :data:`CLEAN.min_chars`。
CLEAN = D.CleanConfig(min_chars=100, max_chars=200_000, min_keep_ratio=0.05)

_ARTICLE = (
    "这是一段用于本地快速测试的中文教育类文本,它讨论自然科学与历史研究中的若干问题,"
    "并且包含足够多的字符以保证清洗之后不会因为长度门限而被丢弃。"
)


def make_doc(index: int, marker: str = "", repeat: int = 8) -> str:
    """生成一篇带唯一编号的假文档 (长度约 ``repeat * 70`` 个字符, 需大于 CLEAN.min_chars)。"""
    head = f"{DOC_MARKER}{index:04d}"
    if marker:
        head += f" {marker}"
    return f"{head}: " + _ARTICLE * repeat


def write_parquet(path, docs) -> None:
    """写一个只有 ``text`` 列的 parquet 文件 (与 fineweb 的真实 schema 一致)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = str(path)
    pq.write_table(pa.table({"text": pa.array(list(docs), type=pa.string())}), path)


@pytest.fixture(scope="module")
def fake_parquet(tmp_path_factory):
    """构造假语料目录: 2 个可读分片 + 1 个 ``.incomplete`` + 1 个 ``.lock``。

    行数写死成常量 (:data:`SHARD1_DOCS` / :data:`SHARD2_DOCS`), 因为测试里要断言
    "迭代出的文档数 == 两个分片的行数之和"; 用切片拼列表容易算错, 所以直接引用常量。
    """
    root = tmp_path_factory.mktemp("parquet")
    nested = root / "subset_a"
    nested.mkdir()

    train_like = [make_doc(i) for i in range(40, 70)]
    val_like = [make_doc(i, VAL_MARKER) for i in range(30, 40)]
    test_like = [make_doc(i, TEST_MARKER) for i in range(20, 30)]
    junk = [
        "",
        "太短",
        "====" * 40,
        "2024-01-31 12:00:00 ERROR: Traceback (most recent call last)",
        "def f(x):\n    return x + 1\n",
        "\x00\x01\x02 只有控制字符 \x03\x04",
    ]

    first = train_like[:15] + val_like + junk
    second = train_like[15:] + test_like
    assert len(first) == SHARD1_DOCS, f"fixture drifted: {len(first)} != {SHARD1_DOCS}"
    assert len(second) == SHARD2_DOCS, f"fixture drifted: {len(second)} != {SHARD2_DOCS}"

    write_parquet(nested / "00001.parquet", first)
    write_parquet(root / "00002.parquet", second)
    (root / "00003.parquet.incomplete").write_bytes(b"not a parquet file at all")
    (root / "00004.parquet.lock").write_bytes(b"")
    return root


# ---------------------------------------------------------------------------
# 1. 清洗
# ---------------------------------------------------------------------------
def test_clean_text_removes_noise_and_normalizes():
    """NFKC、控制字符、URL、噪声行、空行折叠、长度门限都要按预期生效。"""
    raw = (
        "中文段落一。\n\n\n\n\n"
        "2024-01-31 12:00:00 INFO: 日志行 should go\n"
        "========================\n"
        "Traceback (most recent call last):\n"
        "    File \"a.py\", line 12, in <module>\n"
        "https://example.com/very/long/url?q=1 邮箱 a.b@example.com\n"
    ) + _ARTICLE * 4

    out = D.clean_text(raw, CLEAN)

    assert out, "a mostly-normal Chinese document must survive cleaning"
    assert "\x1f" not in out
    assert "日志行" not in out
    assert "Traceback" not in out
    assert "http" not in out
    assert "a.b@example.com" not in out
    assert "======" not in out
    # 5 个连续空行的输入被压到最多 2 个空行 (= 最多 3 个换行符)
    assert "\n\n\n\n" not in out, "runs of blank lines must collapse"
    assert raw.count("\n\n\n\n\n") >= 1, "the fixture must actually contain a long blank run"
    assert "<url>" in out and "<email>" in out, "URLs/emails become placeholders"


def test_clean_text_unicode_nfkc():
    """全角/兼容字符折叠到 NFKC 形式。"""
    out = D.clean_text("Ｈｅｌｌｏ，，中文①②③ " + _ARTICLE * 4, CLEAN)
    assert "Hello" in out
    assert "①②③" not in out


def test_clean_text_enforces_length_gate():
    """太短的整篇丢弃; 太长的整篇丢弃 (而不是截断)。"""
    assert D.clean_text("很短的一段话。", CLEAN) == ""
    assert D.classify_text("很短的一段话。", CLEAN)[1] == "too_short"

    # _ARTICLE 只有 70 字, 所以要超过 CLEAN.max_chars (200_000) 需要 2858 次重复;
    # 这里只比门限多一点点, 免得测试为了造一篇"太长"的文档白烧 CPU。
    repeats = CLEAN.max_chars // len(_ARTICLE) + 2
    long_doc = _ARTICLE * repeats
    assert len(long_doc) > CLEAN.max_chars, f"_ARTICLE * {repeats} is only {len(long_doc)} chars"
    assert D.classify_text(long_doc, CLEAN)[1] == "too_long"
    assert D.clean_text(long_doc, CLEAN) == ""
    # 同一篇文档, 只把上限放宽就该留下 —— 证明丢弃原因是长度而不是内容
    lenient = D.CleanConfig(min_chars=100, max_chars=10_000_000, min_keep_ratio=0.05)
    assert D.classify_text(long_doc, lenient)[1] == "kept"


def test_clean_text_drops_mostly_noise_document():
    """整篇几乎都是代码/符号时按保留率丢弃。"""
    noise = "\n".join(["{ } ] ) ; ( [ < > + - * / % ^ & | ~ ` ! @ # $"] * 60)
    reason = D.classify_text(noise, CLEAN)[1]
    assert reason in ("no_renderable_text", "too_short", "too_noisy")


def test_looks_like_noise_heuristics():
    """启发式: 时间戳/Traceback/分隔线/高符号占比判为噪声, 正常中文不判。"""
    assert D.looks_like_noise("2024-01-31 12:00:00 ERROR: boom")
    assert D.looks_like_noise("Traceback (most recent call last):")
    assert D.looks_like_noise("================")
    assert D.looks_like_noise("}}}]]]];;;:::***")
    assert not D.looks_like_noise("这是一句完全正常的中文, 带有标点。")
    assert not D.looks_like_noise("")
    # 回归: 短行 (少于 min_symbol_ratio_chars) 的符号占比天然很高 ("前。" 是 0.5),
    # 早期版本纯按占比判, 会把正文里的短句成段整行删掉。长度门限必须挡住这一误判。
    assert not D.looks_like_noise("前。")
    assert not D.looks_like_noise("后。")
    assert not D.looks_like_noise("序言:")
    assert D.looks_like_noise("}}}]]]")  # 纯符号短行仍然要被拦下
    assert D.looks_like_noise("###")


def test_clean_text_keeps_short_prose_lines():
    """短句成段的正文必须保住 (回归测试: 曾经整行被当成日志噪声删掉)。"""
    body = "这是一段用于测试的中文教育类文本, 内容足够长以便通过最小长度门限。" * 6
    raw = "前。\n\n\n\n\n\n后。\n" + body
    out = D.clean_text(raw, D.CleanConfig(min_chars=50))
    assert out.startswith("前。"), out[:10]
    assert out.rstrip().endswith("。")
    assert "后。" in out
    # 6 个连续空行折叠到最多 2 个空行 (= 最多 3 个换行符)
    runs = [len(g) for g in __import__("re").findall(r"\n+", out)]
    assert runs and max(runs) <= 3, runs
    # 夹在空白之间的噪声行被删掉后, 两侧空白仍属于同一段, 不能因此多留空行
    raw2 = "前。\n\n\n\n=====\n\n\n\n后。\n" + body
    out2 = D.clean_text(raw2, D.CleanConfig(min_chars=50))
    runs2 = [len(g) for g in __import__("re").findall(r"\n+", out2)]
    assert max(runs2) <= 3, runs2
    assert "前。" in out2 and "后。" in out2


# ---------------------------------------------------------------------------
# 2. parquet 读取
# ---------------------------------------------------------------------------
def test_iter_documents_skips_incomplete_and_locks(fake_parquet):
    """``.incomplete`` / ``.lock`` 必须被跳过, 且文本列自动识别。"""
    docs = list(D.iter_documents(fake_parquet))
    assert len(docs) == TOTAL_DOCS, "wrong document count: a skipped/duplicated shard?"
    assert all(isinstance(d, str) for d in docs)

    files = D._iter_parquet_files(fake_parquet)
    names = sorted(f.name for f in files)
    assert names == ["00001.parquet", "00002.parquet"], names
    assert not any(".incomplete" in n or ".lock" in n for n in names)
    # 排序是"按完整路径"的, 所以根目录下的 00002 排在 subset_a/00001 前面 —— 有意为之:
    # 分片顺序在不同机器/文件系统上必须确定, 只能靠路径字典序。
    assert [f.name for f in files] == ["00002.parquet", "00001.parquet"]


def test_iter_documents_limits_and_columns(fake_parquet):
    """``max_docs`` 生效; 显式指定列名与错误的列名行为正确。"""
    assert len(list(D.iter_documents(fake_parquet, max_docs=7))) == 7
    assert len(list(D.iter_documents(fake_parquet, text_column="text"))) == TOTAL_DOCS
    with pytest.raises(KeyError):
        list(D.iter_documents(fake_parquet, text_column="raw_content"))


def test_pick_text_column():
    """列名优先级与大小写处理。"""
    assert D.pick_text_column(["a", "text", "content"]) == "text"
    assert D.pick_text_column(["content", "raw_content"]) == "content"
    assert D.pick_text_column(["raw_content"]) == "raw_content"
    assert D.pick_text_column(["TEXT"]) == "TEXT"
    assert D.pick_text_column(["foo", "bar"]) is None


# ---------------------------------------------------------------------------
# 3. 分词器
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bpe_dir(tmp_path_factory, fake_parquet):
    """在假语料上训练一个很小的 BPE 分词器 (模块级复用)。"""
    out = tmp_path_factory.mktemp("bpe")
    texts = [d for d in D.iter_documents(fake_parquet) if isinstance(d, str) and len(d) > 100]
    info = T.train_bpe((iter(texts), texts), 600, out, measure_ratio=True)
    return out, info


def test_train_bpe_roundtrip_and_ratio(bpe_dir):
    """BPE: 词表尺寸、EOS id、压缩比可用、解码恢复 NFKC 后的文本。"""
    import unicodedata

    out, info = bpe_dir
    assert info.kind == "bpe"
    assert info.vocab_size > 256, "ByteLevel BPE must keep the 256-byte alphabet"
    assert info.eos_id >= 0
    assert info.compression_ratio and info.compression_ratio > 1.0
    assert info.sample_chars > 0

    tok = T.load_tokenizer(out)
    assert tok.eos_id == info.eos_id
    sample = make_doc(1234) + " 混排 ASCII text."
    ids = T.encode(tok, sample)
    assert T.decode(tok, ids) == unicodedata.normalize("NFKC", sample)
    assert T.encode(tok, "") == []


def test_char_fallback_tokenizer(tmp_path):
    """char 回退: 建表、往返、未登录字符落到 <unk>。"""
    texts = [make_doc(i) for i in range(20)]
    out = tmp_path / "char"
    info = T.train_tokenizer((iter(texts), texts), 100_000, out, kind="char", measure_ratio=True)
    assert info.kind == "char"
    assert info.unk_id == 1
    tok = T.load_tokenizer(out)
    sample = make_doc(3)
    assert T.decode(tok, T.encode(tok, sample)) == sample
    assert T.encode(tok, "\U0001F600") == [tok.unk_id]
    # 词表 = 3 个特殊 token + 全语料出现过的不同字符 (而不是这一篇文档的字符集)
    assert tok.vocab_size == len(set("".join(texts))) + 3
    assert tok.vocab_size < 100, "char vocab must stay tiny on this fixture"


def test_compression_ratio_helper():
    """压缩比工具函数: chars / tokens, 空输入返回 0。"""

    class FakeTok:
        def encode(self, text):
            return list(range(len(text) // 2))

    ratio, chars, tokens = T.compression_ratio(FakeTok(), ["abcdefghij"])
    assert (chars, tokens) == (10, 5)
    assert ratio == 2.0
    assert T.compression_ratio(FakeTok(), [])[0] == 0.0


# ---------------------------------------------------------------------------
# 4. 打包 (核心)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def packed(tmp_path_factory, fake_parquet, bpe_dir):
    """用假 parquet + 小 BPE 跑一次完整 pack_documents。"""
    out = tmp_path_factory.mktemp("corpus")
    _, info = bpe_dir
    tokenizer = T.load_tokenizer(bpe_dir[0])
    t0 = time.time()
    result = D.pack_documents(
        D.iter_documents(fake_parquet),
        tokenizer,
        seq_len=32,
        out_dir=out,
        val_frac=0.25,
        test_frac=0.25,
        seed=1234,
        clean_cfg=CLEAN,
        shard_tokens=1024,
        tokenizer_info=info,
        source_files=["00001.parquet", "00002.parquet"],
    )
    return out, result, tokenizer, time.time() - t0


def test_pack_produces_parseable_bins(packed):
    """(a) 产出 .bin 可解析: 尺寸是 itemsize 整数倍, token 数/manifest 一致。"""
    out, result, tokenizer, _ = packed
    manifest = D.read_manifest(out)
    assert manifest["dtype"] == "uint32"
    assert manifest["byteorder"] == "little"
    assert manifest["format"] == "simplcore.core.corpus"
    assert manifest["seq_len"] == 32
    assert manifest["eos_id"] == tokenizer.eos_id
    assert manifest["vocab_size"] == tokenizer.vocab_size

    for split in ("train", "val", "test"):
        entry = manifest["splits"][split]
        assert entry["files"], f"{split} has no shards"
        total = 0
        for f in entry["files"]:
            path = out / f["path"]
            assert path.exists()
            assert path.stat().st_size % 4 == 0
            arr = D.load_shard(path)
            assert arr.dtype == np.uint32
            assert arr.size == f["n_tokens"]
            assert int(arr.max(initial=0)) < max(manifest["vocab_size"], 1)
            assert f["n_tokens"] == f["bytes"] // 4
            total += arr.size
        assert total == entry["n_tokens"]
        assert entry["n_windows"] * 32 <= entry["n_tokens"]
        # n_tokens = 落盘的 token = 写入的 token; remainder 是被 drop_last 丢掉的尾巴,
        # 它**不在**文件里, 所以不能加进 n_tokens, 只能和 n_tokens_encoded 对账。
        assert entry["n_tokens"] == entry["n_windows"] * 32
        assert entry["n_tokens"] + entry["remainder_tokens"] == entry["n_tokens_encoded"]
        assert entry["remainder_tokens"] < 32

    assert result.n_docs_kept + result.n_docs_dropped + result.n_docs_empty == result.n_docs_in
    assert sum(s["n_docs"] for s in manifest["splits"].values()) == result.n_docs_kept
    assert set(result.drop_reasons) <= {"too_short", "too_long", "too_many_lines", "too_noisy", "empty", "no_renderable_text"}
    assert result.drop_reasons.get("too_short", 0) >= 1
    assert manifest["documents"]["n_kept"] == result.n_docs_kept
    assert manifest["cleaning"] == CLEAN.to_dict(), "the exact cleaning config must be recorded"


def test_manifest_hashes_match_files(packed):
    """(d) manifest 的 SHA-256 与实际文件一致, 且改动文件会被发现。"""
    out, result, _, _ = packed
    manifest = D.read_manifest(out)
    for split in ("train", "val", "test"):
        for f in manifest["splits"][split]["files"]:
            path = out / f["path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == f["sha256"], f"hash mismatch for {path}"
            assert D.compute_sha256(path) == f["sha256"]

    # 篡改一个分片 -> ShardedTokenDataset 必须在加载时拒绝它
    victim = out / manifest["splits"]["val"]["files"][0]["path"]
    backup = victim.read_bytes()
    try:
        victim.write_bytes(backup[:-4] + b"\x00\x00\x00\x00")
        with pytest.raises(D.DataPipelineError, match="sha256 mismatch"):
            D.ShardedTokenDataset(out, "val", verify_hashes=True)
        # 关掉校验后仍然能读 (便于调试)
        D.ShardedTokenDataset(out, "val", verify_hashes=False)
    finally:
        victim.write_bytes(backup)
    D.ShardedTokenDataset(out, "val", verify_hashes=True)


def test_splits_are_document_disjoint(packed):
    """(b) train/val/test 在**文档级**互不相交 (没有 token 序列泄漏)。

    做法: 用 EOS 把每个 split 的 token 流切回"逐篇文档的 token 序列",
    然后对三个 split 的文档集合两两求交 —— 交集必须为空。
    这比"窗口索引不重叠"强得多 (窗口索引本来就不可能重叠), 也比匹配某个短语的
    token ngram 可靠: ByteLevel BPE 会因为上下文不同把同一个短语压成不同的 id
    (实测 ``蓝色海豚`` 单独编码是 3 个 token, 夹在长句里只有 1 个)。
    """
    out, result, tokenizer, _ = packed
    eos = int(tokenizer.eos_id)
    manifest = D.read_manifest(out)

    def stream(split):
        return list(D.ShardedTokenDataset(out, split, verify_hashes=False).token_stream())

    def split_documents(seq):
        """EOS 定界: 返回 ``[(doc_token_tuple, 原始位置), ...]``。

        尾部不足一窗的 token 在 ``drop_last=True`` 下不会落盘, 所以最后一个文档
        可能不完整 —— 最后一段残片被显式丢弃, 不做"半个文档"的比较。
        """
        docs = []
        cur = []
        for t in seq:
            if t == eos:
                docs.append(tuple(cur))
                cur = []
            else:
                cur.append(t)
        return docs

    seqs = {s: stream(s) for s in ("train", "val", "test")}
    docs = {s: split_documents(seqs[s]) for s in ("train", "val", "test")}

    # 每篇文档都以 EOS 结尾, 因此 EOS 数 >= 文档数 - 1 (最后一篇的 EOS 可能落在被丢的尾巴里)
    for split in ("train", "val", "test"):
        n_docs = manifest["splits"][split]["n_docs"]
        n_eos = seqs[split].count(eos)
        assert n_eos >= n_docs - 1, f"{split}: only {n_eos} EOS for {n_docs} documents"

    # 主断言: 三个 split 的文档集合两两不相交
    sets = {s: set(docs[s]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[a] & sets[b]
        assert not overlap, f"{a} and {b} share {len(overlap)} identical documents"

    # 覆盖性: 窗口边界不保证与文档边界对齐, 因此"最后一段残片"所在的文档会丢,
    # 所以只要求绝大多数文档可恢复 (这里给 10% 余量)。关键是上界: 绝不凭空多出文档。
    total_recovered = sum(len(sets[s]) for s in sets)
    assert total_recovered <= result.n_docs_kept
    assert total_recovered >= int(0.9 * result.n_docs_kept)
    for split in ("train", "val", "test"):
        n_docs = manifest["splits"][split]["n_docs"]
        assert 0 < len(sets[split]) <= n_docs
        assert all(d for d in sets[split]), f"{split} recovered an empty document"

    # 方向性: 每篇假文档都以唯一编号 (编号####) 开头。把每个 split 的 token 流整体
    # 解码, 抽出其中出现过的**文档编号**, 三个集合必须两两不相交 —— 这就是
    # "同一篇文档的 token 只出现在一个 split 里" 的直接、可读的证据。
    id_re = re.compile(DOC_MARKER + r"\d{4}")

    def doc_ids(split):
        found = set()
        # 按块解码 (整条流一次解码在真实语料上会很占内存)
        seq = seqs[split]
        for i in range(0, len(seq), 4096):
            for m in id_re.finditer(tokenizer.decode(seq[i:i + 4096])):
                found.add(m.group(0))
        return found

    ids = {s: doc_ids(s) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = ids[a] & ids[b]
        assert not shared, f"{a} and {b} both contain documents {sorted(shared)[:5]}"
    assert sum(len(ids[s]) for s in ids) >= int(0.9 * result.n_docs_kept)

    # 标记归属: 只有 val 文档带 VAL_MARKER, 只有 test 文档带 TEST_MARKER。
    # 用"编号 -> 所属 split"的映射来判定, 避免依赖窗口边界。
    id_to_split = {}
    for split, chunk in ids.items():
        for doc_id in chunk:
            id_to_split.setdefault(doc_id, split)

    def marker_ids(split, marker):
        """找出该 split 里含指定字符标记的文档编号 (按单个窗口解码判断)。"""
        ds = D.ShardedTokenDataset(out, split, verify_hashes=False)
        hits = set()
        for i in range(len(ds)):
            text = tokenizer.decode([int(t) for t in ds[i]])
            if marker in text:
                for m in id_re.finditer(text):
                    hits.add(m.group(0))
        return hits

    test_ids = marker_ids("test", TEST_MARKER)
    val_ids = marker_ids("val", VAL_MARKER)
    assert test_ids, "TEST_MARKER does not appear anywhere in the test split"
    assert val_ids, "VAL_MARKER does not appear anywhere in the val split"
    for doc_id in test_ids:
        assert id_to_split.get(doc_id) == "test", f"test document {doc_id} also shows up in {id_to_split.get(doc_id)}"
    for doc_id in val_ids:
        assert id_to_split.get(doc_id) == "val", f"val document {doc_id} also shows up in {id_to_split.get(doc_id)}"


# ---------------------------------------------------------------------------
# 5. 数据集
# ---------------------------------------------------------------------------
def test_dataset_window_shape_and_contents(packed):
    """(c) ShardedTokenDataset 产出形状正确的 ``(seq_len,)`` 窗口, 内容与文件一致。"""
    out, result, tokenizer, _ = packed
    manifest = D.read_manifest(out)

    for split in ("train", "val", "test"):
        ds = D.ShardedTokenDataset(out, split)
        assert len(ds) == manifest["splits"][split]["n_windows"]
        assert ds.seq_len == 32
        assert ds.vocab_size == tokenizer.vocab_size
        assert ds.eos_id == tokenizer.eos_id
        assert ds.num_tokens == manifest["splits"][split]["n_tokens"]

        # 逻辑视图: 所有分片首尾相连的 token 流
        stream = np.concatenate([D.load_shard(out / f["path"]) for f in manifest["splits"][split]["files"]])
        assert stream.size == ds.num_tokens

        for index in (0, 1, len(ds) // 2, len(ds) - 1):
            window = ds[index]
            assert isinstance(window, torch.Tensor)
            assert window.shape == (32,), window.shape
            assert window.dtype == torch.long
            assert int(window.max()) < manifest["vocab_size"]
            expected = stream[index * 32:(index + 1) * 32]
            assert np.array_equal(window.numpy(), expected), f"{split}[{index}] does not match the .bin"

        # 负索引与越界
        assert torch.equal(ds[-1], ds[len(ds) - 1])
        with pytest.raises(IndexError):
            _ = ds[len(ds)]
        with pytest.raises(IndexError):
            _ = ds[-len(ds) - 1]

        # 小窗口切法: 同一个 split 用不同 seq_len 切, 窗口数应随之变化
        ds_small = D.ShardedTokenDataset(out, split, seq_len=8)
        assert len(ds_small) == ds.num_tokens // 8
        assert ds_small[0].shape == (8,)
        assert np.array_equal(ds_small[3].numpy(), stream[24:32])

        ds.close()


def test_dataset_labels_transform_and_pickle(packed):
    """``return_labels`` / ``transform`` / ``num_workers`` 需要的可序列化性。"""
    import pickle

    out, _, _, _ = packed
    ds = D.ShardedTokenDataset(out, "train", return_labels=True, transform=lambda t: t + 1)
    x, y = ds[0]
    assert x.shape == y.shape == (32,)
    assert torch.equal(y, x)

    plain = D.ShardedTokenDataset(out, "val")
    clone = pickle.loads(pickle.dumps(plain))
    assert len(clone) == len(plain)
    assert torch.equal(clone[0], plain[0])
    assert clone.num_tokens == plain.num_tokens
    # 默认 dtype 来自 manifest
    assert plain.dtype == np.dtype("uint32")
    assert plain.num_bytes() > 0
    assert "ShardedTokenDataset" in repr(plain)


def test_dataset_rejects_bad_inputs(packed):
    """缺分片 / split 名错误 / 未知目录都要报清晰的错。"""
    out, _, _, _ = packed
    with pytest.raises(KeyError):
        D.ShardedTokenDataset(out, "nope")
    with pytest.raises(FileNotFoundError):
        D.ShardedTokenDataset(out / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        D.load_shard(out / "missing.bin")

    # 截断成非 itemsize 整数倍 -> 报 truncated
    manifest = D.read_manifest(out)
    victim = out / manifest["splits"]["test"]["files"][0]["path"]
    backup = victim.read_bytes()
    try:
        victim.write_bytes(backup[:len(backup) - 3])
        with pytest.raises(D.DataPipelineError, match="truncated"):
            D.load_shard(victim)
    finally:
        victim.write_bytes(backup)


def test_load_split_helper(packed):
    """``load_split`` 返回 ``(windows, seq_len)`` 数组且与数据集一致。"""
    out, _, _, _ = packed
    arr = D.load_split(out, "val", max_windows=3)
    assert arr.shape == (3, 32)
    ds = D.ShardedTokenDataset(out, "val", verify_hashes=False)
    assert np.array_equal(arr[2], ds[2].numpy())


# ---------------------------------------------------------------------------
# 6. 切分逻辑与边界条件
# ---------------------------------------------------------------------------
def test_plan_document_splits():
    """文档级切分: 向下取整 + 至少 1 篇 + 训练集不被挤空。"""
    assert D._plan_document_splits(0, 0.1, 0.1) == {"train": 0, "val": 0, "test": 0}
    assert D._plan_document_splits(1000, 0.01, 0.01) == {"train": 980, "val": 10, "test": 10}
    assert D._plan_document_splits(4, 0.005, 0.005) == {"train": 2, "val": 1, "test": 1}
    assert D._plan_document_splits(2, 0.5, 0.5) == {"train": 1, "val": 1, "test": 0}
    assert D._plan_document_splits(3, 0.9, 0.9) == {"train": 1, "val": 1, "test": 1}


def test_pack_documents_rejects_empty_corpus(tmp_path):
    """全是被丢弃的文档时必须抛 DataPipelineError, 而不是悄悄产出空语料。"""
    docs = ["短", "更短的内容", "", "2024-01-01 00:00:00 ERROR: boom"]
    with pytest.raises(D.DataPipelineError, match="no documents survived"):
        D.pack_documents(docs, _TinyTokenizer(), 8, tmp_path / "empty", clean_cfg=CLEAN)


def test_pack_documents_validates_args(tmp_path):
    """非法 seq_len / frac / dtype / eos_id 都要立刻报错。"""
    docs = [make_doc(i) for i in range(10)]
    tok = _TinyTokenizer()
    with pytest.raises(ValueError, match="seq_len"):
        D.pack_documents(docs, tok, 0, tmp_path / "a", clean_cfg=CLEAN)
    with pytest.raises(ValueError, match="val_frac"):
        D.pack_documents(docs, tok, 8, tmp_path / "b", val_frac=1.5, clean_cfg=CLEAN)
    with pytest.raises(ValueError, match="must be < 1"):
        D.pack_documents(docs, tok, 8, tmp_path / "c", val_frac=0.6, test_frac=0.6, clean_cfg=CLEAN)
    with pytest.raises(ValueError, match="unsigned"):
        D.pack_documents(docs, tok, 8, tmp_path / "d", dtype="int32", clean_cfg=CLEAN)
    with pytest.raises(ValueError, match="eos_id"):
        D.pack_documents(docs, _TinyTokenizer(eos_id=-1), 8, tmp_path / "e", clean_cfg=CLEAN)


def test_pack_documents_is_deterministic(tmp_path):
    """同一输入 + 同一 seed -> 完全相同的 token 流与 manifest 统计。"""
    docs = [make_doc(i) for i in range(40)]
    tok = _TinyTokenizer()
    streams = []
    for run in ("a", "b"):
        res = D.pack_documents(docs, tok, 16, tmp_path / run, val_frac=0.2, test_frac=0.2,
                               seed=7, clean_cfg=CLEAN, shard_tokens=512)
        streams.append(np.concatenate([
            D.load_shard(tmp_path / run / f["path"])
            for split in ("train", "val", "test")
            for f in res.splits[split]["files"]
        ]))
        assert res.n_docs_kept == 40
    assert np.array_equal(streams[0], streams[1])

    # 换 seed -> 切分不同 (文档顺序变化 -> 逐 split 的 token 流不同)
    res3 = D.pack_documents(docs, tok, 16, tmp_path / "c", val_frac=0.2, test_frac=0.2,
                            seed=99, clean_cfg=CLEAN, shard_tokens=512)
    stream3 = np.concatenate([
        D.load_shard(tmp_path / "c" / f["path"]) for f in res3.splits["train"]["files"]
    ])
    stream_a = np.concatenate([
        D.load_shard(tmp_path / "a" / f["path"])
        for f in D.read_manifest(tmp_path / "a")["splits"]["train"]["files"]
    ])
    assert not np.array_equal(stream_a, stream3)


def test_pack_documents_uint16_and_single_file(tmp_path):
    """``uint16`` 与 ``shard_tokens=0`` (每 split 单文件) 两条路径都要能跑。"""
    docs = [make_doc(i) for i in range(30)]
    tok = _TinyTokenizer(vocab_size=200)
    res = D.pack_documents(docs, tok, 16, tmp_path / "u16", val_frac=0.1, test_frac=0.1,
                           dtype="uint16", clean_cfg=CLEAN, shard_tokens=0)
    manifest = D.read_manifest(tmp_path / "u16")
    assert manifest["dtype"] == "uint16"
    assert [f["path"] for f in manifest["splits"]["train"]["files"]] == ["train.bin"]
    assert (tmp_path / "u16" / "train.bin").stat().st_size % 2 == 0
    ds = D.ShardedTokenDataset(tmp_path / "u16", "train")
    assert ds.dtype == np.dtype("uint16")
    assert ds[0].shape == (16,)

    # uint16 + 大词表必须被拒绝
    with pytest.raises(ValueError, match="uint16"):
        D.pack_documents(docs, _TinyTokenizer(vocab_size=70000), 16, tmp_path / "big",
                         dtype="uint16", clean_cfg=CLEAN)


def test_pack_documents_rejects_split_without_a_full_window(tmp_path):
    """某个 split 凑不满一窗时必须明确报错, 而不是写出 0 字节分片 (numpy 无法 mmap)。"""
    docs = [make_doc(i) for i in range(5)]
    tok = _TinyTokenizer()
    with pytest.raises(D.DataPipelineError, match="no full"):
        D.pack_documents(docs, tok, 4096, tmp_path / "tiny", val_frac=0.5, test_frac=0.4,
                         clean_cfg=CLEAN)
    assert not list((tmp_path / "tiny").glob("*.bin")), "no shard should be left behind"


# ---------------------------------------------------------------------------
# 7. 端到端 (parquet -> 分词器 -> 打包 -> 数据集)
# ---------------------------------------------------------------------------
def test_end_to_end_small(tmp_path, fake_parquet):
    """整条流水线在小规模上跑通: parquet -> char 分词器 -> 打包 -> DataLoader。"""
    from torch.utils.data import DataLoader

    out = tmp_path / "e2e"
    docs = list(D.iter_documents(fake_parquet))
    info = T.train_tokenizer((iter(docs), docs), 2000, out, kind="char")
    assert info.eos_id >= 0

    cfg = D.CleanConfig(min_chars=100, max_chars=100_000, min_keep_ratio=0.05)
    result = D.pack_documents(
        D.iter_documents(fake_parquet), T.load_tokenizer(out), 16, out,
        val_frac=0.2, test_frac=0.2, seed=5, clean_cfg=cfg, tokenizer_info=info,
    )
    assert result.total_tokens > 0
    # char 分词器一个字符一个 token, 所以压缩比恰好 ~1.0 (BPE 会明显高于 1)
    assert 0.9 <= result.compression_ratio <= 1.1
    assert result.total_tokens_encoded >= result.total_tokens

    ds = D.ShardedTokenDataset(out, "train")
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert batch.shape == (4, 16)

    manifest = D.read_manifest(out)
    assert manifest["tokenizer"]["kind"] == "char"
    assert manifest["documents"]["n_kept"] == result.n_docs_kept
    assert manifest["elapsed_seconds"] >= 0.0
    assert json.loads((out / "index.json").read_text(encoding="utf-8"))["seq_len"] == 16


class _TinyTokenizer:
    """最小可用分词器: 每个字符一个 id, 带 EOS (用来验证适配层与边界条件)。"""

    def __init__(self, vocab_size: int = 300, eos_id: int = 2) -> None:
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def encode(self, text):
        return [1 + (ord(ch) % (self.vocab_size - 3)) for ch in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)
