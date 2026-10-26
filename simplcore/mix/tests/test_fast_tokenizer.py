"""快速 tokenizer id 一致性测试 — 质量不变的硬证据。

断言 ``fast_ops.encode_text_fast`` 与旧 ``train_150m.encode_text`` 在
代表性中文文本（超长/短/空/astral 字符/标点混合）上逐 id 完全一致。
"""
import simplcore.tensor as torch

from simplcore.mix.training.fast_ops import encode_text_fast, batch_stream_fast
from simplcore.mix.training.train_150m import encode_text as encode_text_old
from simplcore.mix.training.train_150m import batch_stream as batch_stream_old


SAMPLES = [
    "你好，世界！",
    "安徽省六安市皋城中学初二年级学生",
    "基于SimplCore Mix架构的7B小模型，完全基于中文优化",
    "一二三四五六七八九十一二三四五六七八九十",  # 超长（> seq_len）
    "a" * 5000,                                    # 极长
    "emoji 😀🎉 and 中文 🚀 混合文本",              # astral 字符
    "",                                            # 空串
    "，。！？…——   \n\t  标点与空白",              # 标点/空白
    "1234567890 abcdefghijklmnopqrstuvwxyz 中文",
]


def test_encode_id_identical_for_all_samples():
    for seq_len in (8, 16, 64):
        for vocab in (64, 32000):
            for text in SAMPLES:
                old = encode_text_old(text, vocab, seq_len)
                new = encode_text_fast(text, vocab, seq_len)
                assert tuple(old.shape) == tuple(new.shape), (
                    f"shape 不符 seq_len={seq_len} vocab={vocab} text={text!r}"
                )
                assert torch.equal(old, new), (
                    f"id 不一致 seq_len={seq_len} vocab={vocab} text={text!r}"
                )


def test_encode_batches_identical():
    texts = SAMPLES + ["测试段落" * 30]
    for seq_len in (8, 32):
        for vocab in (32000, 64):
            old_batches = list(batch_stream_old(iter(texts), vocab_size=vocab, batch_size=3, seq_len=seq_len))
            new_batches = list(batch_stream_fast(iter(texts), vocab_size=vocab, batch_size=3, seq_len=seq_len))
            assert len(old_batches) == len(new_batches)
            for (ox, oy), (nx, ny) in zip(old_batches, new_batches):
                assert torch.equal(ox, nx), "x id 不一致"
                assert torch.equal(oy, ny), "y id 不一致"


def test_cache_rows_identical_to_fast_encoder():
    """行缓存产出的行应与 encode_text_fast 一致。"""
    from pathlib import Path
    import tempfile

    from simplcore.mix.training.fast_ops import TokenizedRowCache, _HAS_NUMPY

    texts = SAMPLES
    with tempfile.TemporaryDirectory() as tmp:
        cache = TokenizedRowCache(Path(tmp), "test", seq_len=16, vocab_size=32000)
        cache.build(iter(texts))
        cached = list(cache.iterate())
        assert len(cached) == len(texts)
        for text, row in zip(texts, cached):
            expect = encode_text_fast(text, 32000, 16)
            assert torch.equal(row, expect), f"缓存行不一致: {text!r}"
        # 二次加载（load_known）也应一致（仅 numpy 可用时；本机无 numpy 跳过）
        if _HAS_NUMPY:
            cache2 = TokenizedRowCache(Path(tmp), "test", seq_len=16, vocab_size=32000)
            assert cache2.load_known() is True
            cached2 = list(cache2.iterate())
            assert len(cached2) == len(texts)
            for row1, row2 in zip(cached, cached2):
                assert torch.equal(row1, row2)
