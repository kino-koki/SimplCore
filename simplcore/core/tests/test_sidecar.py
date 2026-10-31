# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""三层精确引用（L1 旁路 + L2 指针取回）的守护测试。

覆盖六条语义：
1. L1 档案逐 token 精确（"第几个字"级的物理前提）；
2. 指针命中：理想查询（= 块 gist）下 top-1 命中 needle 块，
   read_topk 读回的 token 与写入逐个相等；
3. 账本分账：HBM 常驻 O(1)（与 N 无关）vs Host 档案 O(N)，4M 档数字锁定；
4. 注入零初始化纪律：inject_vector 初始恒为 0（与 CMT memory_in 同源）；
5. purity 兼容：纯净通路无注意力算子，exact_attn 文件靠显式标记豁免；
6. 局部精确注意力实验件：前向形状正确、键集与 L 无关。

诚实边界：指针头零初始化时分布均匀（无判别力）——测试中的"命中"是
把投影设为单位阵（训练收敛的理想状态）后的**机制级**验证，不构成
检索质量主张。
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from simplcore.core.memory.sidecar import (PointerHead, SidecarArchive,
                                           SidecarRetriever, sidecar_ledger)
from simplcore.core.model.config import SimplCoreConfig
from simplcore.core.model.model import SimplCoreModel

ROOT = None  # 由 purity 测试模块统一维护；此处用相对定位


def _model(dim: int = 32, layers: int = 1) -> SimplCoreModel:
    codec_vocab = 256
    cfg = SimplCoreConfig(vocab_size=codec_vocab, dim=dim, num_layers=layers,
                          num_bases=8, ffn_ratio=2.0, decay_max=256.0,
                          local_kernel=2)
    return SimplCoreModel(cfg).eval()


# ---------------------------------------------------------------------------
# 1. L1 档案：逐 token 精确
# ---------------------------------------------------------------------------
def test_archive_exact_roundtrip() -> None:
    a = SidecarArchive(block_size=16)
    seg1 = torch.randint(0, 256, (37,))
    seg2 = torch.randint(0, 256, (11,))
    s1 = a.append(seg1)
    s2 = a.append(seg2)
    assert (s1, s2) == (0, 37) and len(a) == 48
    # 逐 token 精确（含跨块、跨段、单点区间）
    assert torch.equal(a.read(0, 37), seg1)
    assert torch.equal(a.read(37, 48), seg2)
    assert torch.equal(a.read(10, 40), torch.cat([seg1[10:], seg2[:3]]))
    assert a.read(20, 20).numel() == 0
    with pytest.raises(IndexError):
        a.read(0, 49)
    # 块寻址
    lo, hi = a.block_span(2)
    assert (lo, hi) == (32, 48) and a.num_blocks == 3


# ---------------------------------------------------------------------------
# 2. 指针命中 + 精确读取（机制级：proj=identity，训练收敛的理想状态）
# ---------------------------------------------------------------------------
def test_needle_pointer_hits_and_reads_exact() -> None:
    model = _model()
    ret = SidecarRetriever(model, block_size=8, top_k=4)
    d = model.config.dim
    torch.manual_seed(3)
    background = torch.randint(0, 256, (3000,))
    needle = torch.randint(0, 256, (8,))          # 恰好一个块
    ret.observe(background)
    needle_start = ret.observe(needle)
    assert needle_start == 3000
    assert len(ret.gist) == ret.archive.num_blocks

    # 机制级理想查询：proj=identity → h=needle gist 与自身点积最大
    with torch.no_grad():
        ret.pointer.proj.weight.copy_(torch.eye(d))
        ret.pointer.proj.bias.zero_()
    needle_block = needle_start // 8
    h = ret.gist.gists[needle_block].to(torch.float32)
    scores, ids = ret.retrieve(h, k=3)
    assert ids.reshape(-1)[0].item() == needle_block
    # 分布语义：完整 softmax 分布和为 1；topk 是其中的子集（降序、≤1）
    full = torch.softmax(ret.pointer.logits(h, ret.gist.gists), dim=-1)
    assert torch.allclose(full.sum(), torch.tensor(1.0), atol=1e-6)
    assert (scores[:-1] >= scores[1:] - 1e-7).all()
    assert float(scores.max()) <= 1.0 and float(scores.min()) >= 0.0
    # 精确读取：逐 token 与写入一致（"第几个字"级）
    got = ret.read_topk(h, k=1)
    assert got[0][0] == needle_block
    assert torch.equal(got[0][1], needle)
    # 区间读回整段 needle（start 句柄可寻址）
    assert torch.equal(ret.archive.read(needle_start, needle_start + 8), needle)


# ---------------------------------------------------------------------------
# 3. 账本分账：HBM O(1) vs Host O(N)
# ---------------------------------------------------------------------------
def test_ledger_resident_o1_archive_on() -> None:
    small = sidecar_ledger(131_072, dim=1024)
    big = sidecar_ledger(4_194_304, dim=1024)
    # O(1) 常驻：与 N 无关
    assert small["resident_hbm_bytes"] == big["resident_hbm_bytes"]
    # O(N) 档案：精确线性（32×）
    ratio = big["archive_host_bytes"] / small["archive_host_bytes"]
    assert ratio == pytest.approx(32.0)
    # 4M 档数字：int32 原文 = 16 MiB；gist 按块（64 token/块）
    assert big["archive_host_bytes"] == 4_194_304 * 4
    assert big["archive_host_mib"] == pytest.approx(16.0)
    assert big["gist_host_bytes"] == (4_194_304 // 64) * 1024 * 4
    # 实例账本与设计口径一致（同 dim=32 口径）；HBM 常驻与 N 无关
    model = _model()
    ret = SidecarRetriever(model, block_size=64)
    ret.observe(torch.randint(0, 256, (640,)))
    led = ret.ledger()
    assert led["num_tokens"] == 640 and led["num_blocks"] == 10
    ref = sidecar_ledger(640, dim=model.config.dim, block_size=64)
    assert led["resident_hbm_bytes"] == ref["resident_hbm_bytes"]
    assert led["archive_host_bytes"] == ref["archive_host_bytes"]


# ---------------------------------------------------------------------------
# 4. 注入零初始化纪律（与 CMT memory_in 同源）
# ---------------------------------------------------------------------------
def test_inject_zero_init_then_trainable() -> None:
    model = _model()
    ret = SidecarRetriever(model, block_size=8)
    ret.observe(torch.randint(0, 256, (100,)))
    h = torch.randn(model.config.dim)
    # 零初始化：注入恒为 0（初始恒等，不扰动残差流）
    assert float(ret.inject_vector(h).abs().max()) == 0.0
    # 训练后（投影非零）：注入非零且可反传
    with torch.no_grad():
        ret.inject_proj.weight.normal_(0, 0.05)
        ret.pointer.proj.weight.normal_(0, 0.1)
    inj = ret.inject_vector(h)
    assert float(inj.abs().max()) > 0.0
    loss = ret.inject_vector(h.requires_grad_(True)).sum()
    loss.backward()
    assert ret.pointer.proj.weight.grad is not None
    assert ret.inject_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# 5. purity 兼容：纯净通路无注意力；exact_attn 靠显式标记豁免
# ---------------------------------------------------------------------------
def test_purity_scan_passes_with_marked_exact_attn() -> None:
    from pathlib import Path

    from simplcore.core.arch.purity import scan_source_attention_free
    root = Path(__file__).resolve().parents[3] / "simplcore"
    assert scan_source_attention_free(root) == [], \
        "sidecar 纯净通路必须无注意力；exact_attn 必须靠文件头标记豁免"
    text = (root / "core" / "memory" / "sidecar_exact_attn.py") \
        .read_text(encoding="utf-8")
    assert "purity: allow-attention-reference" in text[:3000]
    # 纯净模块本身不得含标记（否则失去守卫意义）
    pure = (root / "core" / "memory" / "sidecar.py").read_text(encoding="utf-8")
    assert "allow-attention-reference" not in pure


# ---------------------------------------------------------------------------
# 6. 局部精确注意力实验件
# ---------------------------------------------------------------------------
def test_exact_attn_forward_shape_and_keys() -> None:
    from simplcore.core.memory.sidecar_exact_attn import LocalExactCrossAttention
    model = _model(dim=32, layers=1)
    ret = SidecarRetriever(model, block_size=8, top_k=4)
    ret.observe(torch.randint(0, 256, (2000,)))
    attn = LocalExactCrossAttention(ret, num_heads=4)
    h = torch.randn(1, model.config.dim)
    out, w = attn(h, k=4, need_weights=True)
    assert out.shape == (1, model.config.dim)
    # 键集与 L 无关：4 块 × 8 token = 32 个键（不是 2000）
    assert w.shape[-1] == 32
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(dim=-1)),
                          atol=1e-5)