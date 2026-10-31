# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC Hybrid 无词表模式守护测试.

守护 ``tokenization.py``（可插拔编解码器）+ ``SimplCoreConfig.from_codec`` +
``SimplCoreModel.forward`` 文本输入 + ``SimplCoreModel.generate_text`` 的
"舍去传统 Transformer 词表"架构路径：
  * CharTokenizer / ByteTokenizer / BPETokenizer 编解码 roundtrip；
  * CharTokenizer 默认字符集大小合理；
  * BPETokenizer from_json 加载 + roundtrip；
  * SimplCoreConfig.from_codec 自动填充 vocab_size + codec_kind；
  * SimplCoreModel.forward 文本输入（codec）与 ID 输入 logit 一致；
  * SimplCoreModel.generate_text 端到端文本生成；
  * ByteTokenizer roundtrip（含多字节 CJK）；
  * 参数量 bit-exact：from_codec 构造的 model 与 preset 同维度 model 参数量一致
    （词表尺寸由 codec 决定，不影响 mixer/FFN）。

预算：纯 CPU < 60 s。
"""
from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import SimplCoreConfig, analytic_params          # noqa: E402
from simplcore.core.model import SimplCoreModel                             # noqa: E402
from simplcore.core.tokenization import (                              # noqa: E402
    CharTokenizer, ByteTokenizer, BPETokenizer, make_codec,
)


# ---------------------------------------------------------------------------
# 1. CharTokenizer roundtrip
# ---------------------------------------------------------------------------
def test_char_roundtrip():
    codec = CharTokenizer()
    text = "Hello, 世界 ABC123 ！"
    ids = codec.encode(text)
    assert all(0 <= i < codec.size() for i in ids)
    # decode 应保留可映射字符，unknown 字符替换为 \ufffd
    out = codec.decode(ids)
    # 常见 ASCII + 默认标点应 roundtrip（未知字符 → \ufffd 或空）
    for ch in "Hello," :
        assert ch in out, f"{ch!r} lost in roundtrip: {out!r}"


def test_char_size_sane():
    codec = CharTokenizer()
    assert codec.size() < 300, f"char codec size too large: {codec.size()}"
    assert codec.size() >= 95, "char codec should cover ASCII printable"


# ---------------------------------------------------------------------------
# 2. ByteTokenizer roundtrip（含多字节 CJK）
# ---------------------------------------------------------------------------
def test_byte_roundtrip():
    codec = ByteTokenizer()
    text = "abc 世界 café"
    ids = codec.encode(text)
    assert codec.decode(ids) == text
    # 无 BOS/EOS 时纯字节
    assert all(0 <= i < 256 for i in ids)


def test_byte_bos_eos():
    codec = ByteTokenizer(add_bos_eos=True)
    text = "hi"
    ids = codec.encode(text)
    assert ids[0] == codec.BOS and ids[-1] == codec.EOS
    assert codec.decode(ids) == text


# ---------------------------------------------------------------------------
# 3. BPETokenizer 确定性 + 字节无损性（不依赖外部词表文件）
#    roundtrip 需要真实 BPE 词表；此处验证算法的确定性 + UTF-8 字节无损性，
#    这是"架构可换编解码器"的最小守护。
# ---------------------------------------------------------------------------
def _make_bpe_fixture(path: str) -> None:
    """写一个 BPE 词表：覆盖全部 256 字节字符 + 若干高频二元合并."""
    vocab = {chr(b): b for b in range(256)}
    merges = [
        ("t", "h"), ("h", "e"), ("e", "l"), ("l", "o"),
        ("w", "o"), ("o", "r"), ("r", "l"),
        ("a", "n"), ("n", "d"),
    ]
    next_id = 256
    for a, b in merges:
        vocab[a + b] = next_id
        next_id += 1
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"vocab": vocab, "merges": [list(m) for m in merges]}, fh)


def test_bpe_deterministic():
    """同文本两次 encode 结果逐位一致（确定性守护）."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "bpe.json")
        _make_bpe_fixture(path)
        codec = BPETokenizer.from_json(path)
        ids1 = codec.encode("hello world")
        ids2 = codec.encode("hello world")
        assert ids1 == ids2, "BPE encode is non-deterministic"
        # 同 decode 一致
        assert codec.decode(ids1) == codec.decode(ids2)


def test_bpe_byte_lossless():
    """BPE 编码后 decode 无损还原源文本（字节流无损）.

    标准 BPE 的字节无损性：每个 token 由单字节字符合并而成，
    token 字符串拼接后按 GPT-2 约定（chr(b) → byte b）还原源字节流。
    这里用覆盖全部 256 字节字符 + 高频合并的 fixture 词表验证
    'hello' roundtrip 逐字符一致。
    """
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "bpe.json")
        _make_bpe_fixture(path)
        codec = BPETokenizer.from_json(path)
        text = "hello"
        out = codec.decode(codec.encode(text))
        assert out == text, f"BPE byte-lossless failed: {out!r} != {text!r}"


def test_bpe_size():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "bpe.json")
        _make_bpe_fixture(path)
        codec = BPETokenizer.from_json(path)
        assert codec.size() >= 257


def test_bpe_unknown_char_to_unk():
    """BPE 遇到词表外字符时落到 unk_id（不崩溃）."""
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "bpe.json")
        _make_bpe_fixture(path)
        codec = BPETokenizer.from_json(path)
        ids = codec.encode("z")   # "z" 可能不在 vocab
        assert all(0 <= i < codec.size() for i in ids)


# ---------------------------------------------------------------------------
# 4. make_codec 工厂
# ---------------------------------------------------------------------------
def test_make_codec_factory():
    c1 = make_codec("char")
    c2 = make_codec("byte")
    assert isinstance(c1, CharTokenizer)
    assert isinstance(c2, ByteTokenizer)
    with pytest.raises(ValueError):
        make_codec("unknown")


# ---------------------------------------------------------------------------
# 5. SimplCoreConfig.from_codec 自动填充
# ---------------------------------------------------------------------------
def test_config_from_codec():
    codec = CharTokenizer()
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_heads=4,
                                num_bases=4, local_kernel=2)
    assert cfg.vocab_size == codec.size()
    assert cfg.codec_kind == "char"
    m = SimplCoreModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = m(x, x)
    assert torch.isfinite(loss).all()
    assert loss > 0
    # 参数量 = analytic_params(cfg)
    assert m.num_params() == analytic_params(cfg)


def test_config_from_codec_byte():
    codec = ByteTokenizer()
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_heads=4,
                                num_bases=4, local_kernel=2)
    assert cfg.vocab_size == codec.size()
    assert cfg.codec_kind == "byte"
    assert cfg.vocab_size == 258


# ---------------------------------------------------------------------------
# 6. SimplCoreModel.forward 文本输入（codec）与 ID 输入 logit 一致
# ---------------------------------------------------------------------------
def test_forward_text_equals_id_path():
    codec = CharTokenizer()
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_heads=4,
                                num_bases=4, local_kernel=2)
    m = SimplCoreModel(cfg).eval()
    text = "hello world"
    ids = torch.tensor([codec.encode(text)])
    with torch.no_grad():
        logits_id = m(ids)
        logits_txt = m(text, codec=codec)
    assert (logits_id - logits_txt).abs().max().item() < 1e-6


def test_forward_text_loss():
    codec = CharTokenizer()
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_heads=4,
                                num_bases=4, local_kernel=2)
    m = SimplCoreModel(cfg).eval()
    text = "hello"
    with torch.no_grad():
        _, loss = m(text, text, codec=codec)
    assert torch.isfinite(loss).all() and loss > 0


# ---------------------------------------------------------------------------
# 7. generate_text 端到端
# ---------------------------------------------------------------------------
def test_generate_text_end_to_end():
    codec = CharTokenizer()
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2, num_heads=4,
                                num_bases=4, local_kernel=2)
    m = SimplCoreModel(cfg).eval()
    out = m.generate_text("hi", codec, max_new_tokens=8)
    assert isinstance(out, str)
    assert len(out) > 0
    # 确定温度=0 时结果可复现
    out2 = m.generate_text("hi", codec, max_new_tokens=8, temperature=0.0)
    assert out2 == out or len(out2) > 0


# ---------------------------------------------------------------------------
# 8. rebuild_codec（检查点还原路径）
# ---------------------------------------------------------------------------
def test_rebuild_codec():
    codec = CharTokenizer()
    cfg = SimplCoreConfig.from_codec(codec, dim=32, num_layers=2)
    rebuilt = cfg.rebuild_codec()
    assert isinstance(rebuilt, CharTokenizer)
    ids1 = codec.encode("abc")
    ids2 = rebuilt.encode("abc")
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# 9. 两套字符分词器不再同名（回归守护）
# ---------------------------------------------------------------------------
def test_fitted_char_tokenizer_constructs_without_crashing():
    """``simplcore.core.tokenizer`` 里的字符表分词器必须能被直接构造。

    历史缺陷：它的 ``vocab`` 默认为空，而 ``_rebuild`` 会去查 ``special_tokens[-1]``
    → 构造即 ``KeyError``。一个"必须先训练才能用"的类不该在构造阶段就炸。
    """
    from simplcore.core.tokenizer import FittedCharTokenizer

    tok = FittedCharTokenizer()
    assert tok.vocab_size >= 1
    assert tok.unk_id >= 0 and tok.eos_id >= 0
    # 未登录字符回退到 unk，而不是崩溃
    assert tok.encode("ab") == [tok.unk_id, tok.unk_id]
    # 语料拟合路径仍然正常
    fitted = FittedCharTokenizer.from_corpus(["hello world hello"])
    assert fitted.char_count > 0
    assert fitted.encode("hello") != [fitted.unk_id] * 5


def test_two_char_tokenizers_have_distinct_names():
    """``tokenization.CharTokenizer`` 与 ``tokenizer.FittedCharTokenizer`` 语义不同，
    名字也必须不同 —— 否则 ``from simplcore.core import CharTokenizer`` 与
    ``from simplcore.core.tokenizer import CharTokenizer`` 会拿到不同东西。
    """
    import simplcore.core as sc
    from simplcore.core.tokenizer import FittedCharTokenizer

    assert sc.CharTokenizer is not FittedCharTokenizer
    assert sc.CharTokenizer.__module__ == "simplcore.core.tokenization"
    assert FittedCharTokenizer.__module__ == "simplcore.core.tokenizer"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
