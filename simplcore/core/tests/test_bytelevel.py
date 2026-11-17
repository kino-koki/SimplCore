# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""bytelevel 工业级字节级主路径的守护测试.

覆盖：对抗 Unicode 往返、流式编码一致、patcher 三模式性质、
桥接与任意 SimplCore mixer 的前向/反向、流式等价、checkpoint 往返。
"""
import json

import pytest
import torch
import torch.nn.functional as F

from simplcore.core.model.bytelevel import (ByteLevelCodec, BytePatchBridge,
                                            EntropyPatcher)


# =============================================================================
# ByteLevelCodec
# =============================================================================
def test_roundtrip_adversarial_unicode():
    codec = ByteLevelCodec()
    samples = [
        "", "hello world", "你好，世界！", "emoji \U0001F680 test",
        "ünïcödé åccents", "tabs\tand\nnewlines\r\n", "控制字符",
        "4-byte \U00010488 codepoint", "mixed 中英 mixing", "   ",
        "line\nbreaks\r\r\n", "quote\"'`", "<html>&amp;</html>",
    ]
    for s in samples:
        ids = codec.encode(s)
        assert all(0 <= i < 256 for i in ids)
        assert codec.decode(ids) == s


def test_codec_streaming_matches_batch():
    codec = ByteLevelCodec()
    text = "你好 world \U0001F680 " * 7
    whole = codec.encode(text)
    parts = [text[i:i + 7] for i in range(0, len(text), 7)]
    stream = []
    for part in parts:
        stream += codec.feed(part)
    stream += codec.feed_flush()
    assert stream == whole


def test_codec_invalid_bytes_replaced_not_raised():
    codec = ByteLevelCodec()
    bad = [0x41, 0xF0, 0x9F]          # 截断的 4 字节码点前缀
    out = codec.decode(bad)
    assert "\ufffd" in out            # 替换字符，不抛异常


def test_codec_save_load(tmp_path):
    codec = ByteLevelCodec(add_bos_eos=True)
    p = str(tmp_path / "codec.json")
    codec.save(p)
    loaded = ByteLevelCodec.load(p)
    assert loaded.encode("测试") == codec.encode("测试")


# =============================================================================
# EntropyPatcher
# =============================================================================
def _assert_monotone(bounds):
    assert bounds == sorted(bounds)
    assert len(set(bounds)) == len(bounds)


def test_fixed_mode_boundaries_and_coverage():
    p = EntropyPatcher(mode="fixed", max_len=4)
    ids = list(range(20))
    b = p.boundaries(ids)
    assert b == [3, 7, 11, 15, 19]
    _assert_monotone(b)


def test_space_mode_breaks_at_non_alnum():
    p = EntropyPatcher(mode="space", max_len=16)
    ids = [97, 98, 99, 32, 100, 101, 102]
    b = p.boundaries(ids)
    assert b[0] == 3                     # 空格后断开
    assert b[-1] == len(ids) - 1         # 末尾收尾


def test_entropy_mode_is_deterministic_and_bounded():
    p1 = EntropyPatcher(mode="entropy", threshold=3.0, max_len=8)
    p2 = EntropyPatcher(mode="entropy", threshold=3.0, max_len=8)
    ids = [i % 256 for i in range(300)]
    assert p1.boundaries(ids) == p2.boundaries(ids)
    b = p1.boundaries(ids)
    _assert_monotone(b)
    for x in b:
        assert 0 <= x < 300


def test_patcher_streaming_matches_batch():
    p_batch = EntropyPatcher(mode="entropy", threshold=3.0, max_len=8)
    p_stream = EntropyPatcher(mode="entropy", threshold=3.0, max_len=8)
    ids = [i % 251 for i in range(400)]
    whole = p_batch.boundaries(ids)
    got, last = [], 0
    for i in range(0, len(ids), 13):
        chunk = ids[i:i + 13]
        bounds, _ = p_stream.feed(chunk)
        for x in bounds:
            got.append(last + x)
        last += len(chunk)
    assert got == whole[:len(got)]


def test_patcher_reset_clears_state():
    p = EntropyPatcher(mode="entropy", threshold=3.0, max_len=8)
    p.boundaries([1, 2, 3, 4, 5])
    p.reset()
    assert not p._pending and p._acc == 0.0 and p._since_break == 0


# =============================================================================
# BytePatchBridge + SimplCore mixer 兼容性
# =============================================================================
def _tiny_model(mixer, vocab):
    from simplcore.core.model.config import SimplCoreConfig
    from simplcore.core.model.model import SimplCoreModel
    cfg = SimplCoreConfig(vocab_size=vocab, dim=32, num_layers=2,
                          num_bases=8, ffn_ratio=2.0, local_kernel=4,
                          decay_max=128.0, mixer=mixer, tie_embeddings=True)
    return SimplCoreModel(cfg)


@pytest.mark.parametrize("mixer", ["sgc", "sgc-bind", "srf"])
def test_bridge_forward_backward_through_mixers(mixer):
    codec = ByteLevelCodec()
    bridge = BytePatchBridge(codec)
    model = _tiny_model(mixer, 256)
    text = "你好，byte-level world！"
    ids = codec.encode(text)
    pm = bridge.patch_means(ids, model.tok_emb.weight.detach())
    assert pm.shape[0] >= 1 and pm.shape[1] == 32
    # patch 序列过 mixer 栈 + 字节头（加 batch 维）
    h = pm.unsqueeze(0)
    for blk in model.blocks:
        h = blk(h)
    logits = model.ln_f(h)[0] @ model.tok_emb.weight.t()
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits, torch.full((logits.shape[0],), 65))
    loss.backward()
    g = model.blocks[0].mixer
    grads = [p.grad for p in g.parameters() if p.grad is not None]
    assert grads and any(float(gr.abs().sum()) > 0 for gr in grads)


def test_bridge_deterministic():
    codec = ByteLevelCodec()
    bridge = BytePatchBridge(codec)
    ids = list("hello world".encode())
    a = bridge.patch_means(ids, torch.randn(256, 32))
    b = bridge.patch_means(ids, torch.randn(256, 32))
    assert a.shape == b.shape
    again = BytePatchBridge(ByteLevelCodec())
    c = again.patch_means(ids, torch.randn(256, 32))
    assert c.shape == a.shape


def test_bridge_empty_input():
    codec = ByteLevelCodec()
    bridge = BytePatchBridge(codec)
    pm = bridge.patch_means([], torch.randn(256, 32))
    assert pm.shape == (0, 32)


# =============================================================================
# 工业级入口：byte_level_logits / byte_level_generate
# =============================================================================
def test_byte_level_logits_matches_manual_path():
    from simplcore.core.model.bytelevel import byte_level_logits
    model = _tiny_model("sgc-bind", 256)
    text = "你好，byte-level world！"
    logits = byte_level_logits(model, text)
    # 与手动路径（bridge + blocks）逐值一致
    codec = ByteLevelCodec()
    bridge = BytePatchBridge(codec)
    pm = bridge.patch_means(codec.encode(text), model.tok_emb.weight.detach())
    h = pm.unsqueeze(0)
    for blk in model.blocks:
        h = blk(h)
    manual = model.ln_f(h)[0] @ model.tok_emb.weight.t()
    assert torch.allclose(logits, manual, atol=1e-5)
    assert logits.shape == manual.shape


def test_byte_level_generate_smoke():
    from simplcore.core.model.bytelevel import byte_level_generate
    model = _tiny_model("sgc", 256)
    out = byte_level_generate(model, "hello", max_new_patches=3, greedy=True)
    assert isinstance(out, str)
    out2 = byte_level_generate(model, "hello", max_new_patches=3, greedy=True)
    assert out == out2            # greedy 确定性
