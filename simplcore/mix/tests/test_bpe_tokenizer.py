"""自研 BPE tokenizer 测试（修复命门 4）。

覆盖：
  - 训练→编码→解码 round-trip 一致
  - 特殊 token id 布局（pad=0/eos=1，无冲突）
  - 编码 id 范围合法（全部 < vocab_size）
  - 中文预分词把连续 CJK 切单字
  - 保存/加载后编码结果一致
  - encode 性能 sanity（不退化）
"""
import tempfile
import time
from pathlib import Path

import simplcore.tensor as torch

from simplcore.mix.tokenizer import BPETokenizer, CharTokenizer, PAD_ID, EOS_ID

CORPUS = [
    "安徽省六安市皋城中学初二学生",
    "我热爱人工智能和深度学习编程",
    "训练大规模语言模型需要大量中文语料",
    "华为的余承东是高管的传奇人物",
    "参数堆叠技术可以把小模型变成大模型",
    "操作系统和内核开发是计算机科学的核心",
    "深度学习框架PyTorch和CUDA加速训练",
    "中文自然语言处理需要好的分词器",
    "Hello world, this is English text with numbers 123",
    "混合中英文的测试 Mixed Chinese English 🚀",
] * 5


def _train(vocab_size: int = 500) -> BPETokenizer:
    tok = BPETokenizer(vocab_size=vocab_size, chinese=True)
    tok.train(CORPUS, verbose=False)
    return tok


def test_roundtrip_identity():
    tok = _train()
    for text in CORPUS:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, f"round-trip 失败: {text!r} -> {decoded!r}"


def test_special_token_layout():
    tok = _train()
    assert tok.pad_token_id == PAD_ID == 0
    assert tok.eos_token_id == EOS_ID == 1
    assert tok.byte_offset == 2
    assert tok.merge_offset == 258


def test_encode_ids_in_range():
    tok = _train()
    for text in CORPUS:
        ids = tok.encode(text)
        for t in ids:
            assert 0 <= t < tok.vocab_size, f"id 越界: {t} >= {tok.vocab_size}"
        # 特殊 token 不会被当作内容产出
        assert EOS_ID not in ids, "内容中不应出现 eos id"


def test_chinese_pretokenize_splits_cjk():
    from simplcore.mix.tokenizer import PATTERN_CHINESE
    words = PATTERN_CHINESE.findall("我爱中国")
    # 每个汉字独立一个词（可能混入空白）
    han = [w for w in words if w and w.strip()]
    assert "我" in han and "爱" in han and "中" in han and "国" in han


def test_save_load_preserves_encode():
    tok = _train()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "tok.json")
        tok.save(path)
        tok2 = BPETokenizer()
        tok2.load(path)
        for text in CORPUS[:5]:
            assert tok.encode(text) == tok2.encode(text), f"加载后不一致: {text!r}"


def test_encode_performance_sanity():
    """encode 不应退化到 O(N×merges)。长文本编码应在毫秒级。"""
    tok = _train()
    long_text = "深度学习与人工智能技术正在改变世界。自然语言处理是核心方向。" * 50  # ~500字
    t0 = time.time()
    tok.encode(long_text)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"长文本编码过慢: {elapsed:.2f}s (可能回退 O(N*M))"


def test_char_tokenizer_no_conflict():
    ct = CharTokenizer()
    ct.train(["abcdef"])
    assert ct.pad_token_id == 0 and ct.eos_token_id == 1
    ids = ct.encode("abc")
    assert all(i >= 2 for i in ids), "内容 id 不应与特殊 token 冲突"
    assert ct.decode(ids) == "abc"


def test_vocab_size_floor():
    import pytest
    with pytest.raises(ValueError):
        BPETokenizer(vocab_size=100)  # 小于最小 259


def test_tokenizer_adapter_modes():
    """适配层：真 BPE 与字符哈希回退都应产出合法 id。"""
    from simplcore.mix.training.tokenizer_adapter import (
        load_tokenizer,
        make_encoder,
    )

    tok = _train()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "tok.json")
        tok.save(path)
        loaded = load_tokenizer(path, vocab_size=500)
        assert loaded is not None
        enc_bpe = make_encoder(loaded, 500)
        ids = enc_bpe("我热爱人工智能", 8)
        assert tuple(ids.shape) == (9,)  # seq_len+1
        assert all(0 <= t < 500 for t in ids.tolist())

        # 回退模式
        enc_hash = make_encoder(None, 500)
        ids2 = enc_hash("我热爱人工智能", 8)
        assert tuple(ids2.shape) == (9,)
        assert all(0 <= t < 500 for t in ids2.tolist())


def test_load_nonexistent_returns_none():
    from simplcore.mix.training.tokenizer_adapter import load_tokenizer
    assert load_tokenizer("/no/such/file.json", 500) is None
