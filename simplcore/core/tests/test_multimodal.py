# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""多模态入口（0.1-α）的守护测试。

覆盖五条语义：
1. adapter 形状契约（图像 patch 数 / 音频帧下采样）；
2. **无模态时逐位等于纯文本路径**（同一 block 栈，测试锁定）；
3. 模态前缀的因果性：模态在前、文本在后；训练可反传（模态侧有梯度）；
4. 流式一致性：模态前缀 prefill（inputs_embeds）+ 文本流式 的最终 logits
   与整段路径一致（fp32 口径 < 1e-4，L 小）；
5. 基座 stream_step 的 inputs_embeds 缺省行为**逐位不变**（回归锁定）。

诚实边界：adapter 未在真实多模态数据上训练验证——本文件只锁结构与通路。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.model.config import SimplCoreConfig
from simplcore.core.model.model import SimplCoreModel
from simplcore.core.model.multimodal import (AudioAdapter, ImageAdapter,
                                             SimplCoreMultimodal)


def _model(dim: int = 32, layers: int = 1) -> SimplCoreModel:
    cfg = SimplCoreConfig(vocab_size=256, dim=dim, num_layers=layers,
                          num_bases=8, ffn_ratio=2.0, decay_max=256.0,
                          local_kernel=2)
    return SimplCoreModel(cfg).eval()


# ---------------------------------------------------------------------------
# 1. adapter 形状契约
# ---------------------------------------------------------------------------
def test_image_adapter_shape() -> None:
    a = ImageAdapter(dim=32, patch_size=16)
    x = torch.randn(2, 3, 64, 64)
    out = a(x)
    assert out.shape == (2, 16, 32)            # (64/16)^2 = 16 patches
    with pytest.raises(ValueError):
        ImageAdapter(dim=32, patch_size=0)
    with pytest.raises(ValueError):
        a(torch.randn(2, 3, 64))


def test_audio_adapter_shape() -> None:
    a = AudioAdapter(dim=32, n_mels=80, downsample=4)
    out = a(torch.randn(1, 80, 100))
    assert out.shape == (1, 25, 32)            # 100 // 4
    with pytest.raises(ValueError):
        a(torch.randn(1, 80))


# ---------------------------------------------------------------------------
# 2. 无模态时逐位等于纯文本路径
# ---------------------------------------------------------------------------
def test_no_modality_bitwise_matches_base() -> None:
    torch.manual_seed(0)
    model = _model()
    mm = SimplCoreMultimodal(model)
    mm.eval()
    idx = torch.randint(0, 256, (2, 24))
    lg_base = model(idx)
    lg_mm = mm(idx)
    assert torch.equal(lg_base, lg_mm), \
        "无模态输入时多模态包装必须与纯文本路径逐位一致"
    logits, loss = mm(idx, targets=idx)
    _, loss_base = model(idx, targets=idx)
    assert torch.equal(loss, loss_base)


# ---------------------------------------------------------------------------
# 3. 模态前缀：因果拼接 + 梯度通路
# ---------------------------------------------------------------------------
def test_modality_prefix_grads_flow() -> None:
    torch.manual_seed(0)
    model = _model()
    mm = SimplCoreMultimodal(model)
    mm.train()
    idx = torch.randint(0, 256, (1, 12))
    img = torch.randn(1, 3, 32, 32)            # 2x2=4 个 patch
    logits = mm(idx, image=img)
    assert logits.shape == (1, 12 + 4 + 1, 256)  # 前缀 + 分隔 + 文本
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 256), torch.randint(0, 256, (17,)))
    loss.backward()
    assert mm.image.patchify.weight.grad is not None
    assert mm.image.patchify.weight.grad.abs().sum() > 0
    assert mm.sep.grad is not None
    assert model.tok_emb.weight.grad is not None

    audio = torch.randn(1, 80, 40)
    logits2 = mm(idx, audio=audio)
    assert logits2.shape == (1, 12 + 10 + 1, 256)  # 40//4=10 帧


# ---------------------------------------------------------------------------
# 4. 流式一致性：模态前缀 prefill + 文本流式 ≈ 整段路径
# ---------------------------------------------------------------------------
def test_stream_matches_full_with_modality_prefix() -> None:
    torch.manual_seed(0)
    model = _model()
    mm = SimplCoreMultimodal(model)
    mm.eval()
    idx = torch.randint(0, 256, (1, 8))
    img = torch.randn(1, 3, 32, 32)            # 4 patches
    with torch.no_grad():
        # 整段：embed_sequence → block 栈 → head，取最后一个位置的 logits
        x = mm.embed_sequence(idx, image=img)
        for blk in model.blocks:
            x = blk(x, None)
        full_last = model.head(model.ln_f(x))[:, -1, :]
        # 流式：模态前缀 inputs_embeds prefill + 文本逐 token
        soft = mm.embed_sequence(torch.empty(1, 0, dtype=torch.long), image=img)
        state = model.init_stream(1, device=idx.device,
                                  dtype=model.tok_emb.weight.dtype)
        lg = None
        for i in range(soft.shape[1]):
            lg, state = model.stream_step(None, state,
                                          inputs_embeds=soft[:, i:i + 1])
        for t in range(idx.shape[1]):
            lg, state = model.stream_step(idx[:, t], state)
    err = float((lg - full_last).abs().max())
    assert err < 1e-4, f"流式（含模态前缀）与整段偏差超 fp32 口径: {err}"


# ---------------------------------------------------------------------------
# 5. inputs_embeds 缺省行为逐位不变（基座回归锁定）
# ---------------------------------------------------------------------------
def test_stream_step_default_bitwise_unchanged() -> None:
    torch.manual_seed(0)
    model = _model()
    model.eval()
    idx = torch.randint(0, 256, (1, 6))
    with torch.no_grad():
        st1 = model.init_stream(1, device=idx.device,
                                dtype=model.tok_emb.weight.dtype)
        lg1 = None
        for t in range(idx.shape[1]):
            lg1, st1 = model.stream_step(idx[:, t], st1)
        st2 = model.init_stream(1, device=idx.device,
                                dtype=model.tok_emb.weight.dtype)
        lg2 = None
        for t in range(idx.shape[1]):
            lg2, st2 = model.stream_step(idx[:, t], st2, inputs_embeds=None)
        # inputs_embeds 显式传入 tok_emb 输出也应与 token 路径一致
        st3 = model.init_stream(1, device=idx.device,
                                dtype=model.tok_emb.weight.dtype)
        lg3 = None
        for t in range(idx.shape[1]):
            emb = model.tok_emb(idx[:, t])
            lg3, st3 = model.stream_step(None, st3, inputs_embeds=emb)
    assert torch.equal(lg1, lg2)
    assert torch.equal(lg1, lg3)


# ---------------------------------------------------------------------------
# 6. 生成：模态条件下输出形状与提前停止
# ---------------------------------------------------------------------------
def test_generate_with_modality_prefix() -> None:
    torch.manual_seed(0)
    model = _model()
    mm = SimplCoreMultimodal(model)
    mm.eval()
    idx = torch.randint(0, 256, (1, 5))
    img = torch.randn(1, 3, 32, 32)
    out = mm.generate(idx, max_new_tokens=7, image=img)
    assert out.shape == (1, 5 + 7)