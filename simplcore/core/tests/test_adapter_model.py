# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""纯算子重组模型（AdapterModel）的守护测试。

三个命题：
1. **等价**：同一份权重下，AdapterModel 与原生 SimplCoreModel 的并行前向、
   流式前向、梯度在数值上一致 —— 算子重组没有改变数学。
2. **覆盖**：并行 + 流式前向覆盖契约全部前向算子 —— 16 算子契约对
   "搭出完整 SimplCore" 是完备的（这就是本模块存在的意义）。
3. **可训练**：梯度流经每个参数（无静默零梯度）。
"""
import pytest
import torch

from simplcore.core.adapter_model import AdapterModel, RecordingAdapter
from simplcore.core.conformance_runner import OP_TO_METHODS, TorchReferenceAdapter
from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel

# 闸门显式开启（v1.2 起默认 opt-in）：契约审计必须覆盖 L2/L3 门算子，
# 梯度 parity 也要遍历门标量参数。
CFG = SimplCoreConfig(vocab_size=32, dim=32, num_layers=2, num_bases=8,
                      local_kernel=4, ffn_ratio=2.0, decay_max=256.0,
                      evidence_gate=True, uncertainty_gate=True,
                      tie_embeddings=False)


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(11)
    native = SimplCoreModel(CFG)
    rec = RecordingAdapter(TorchReferenceAdapter())
    ad = AdapterModel(CFG, rec)
    ad.load_native(native.state_dict())
    return native, ad, rec


def _batch(vocab=32, B=2, L=64, seed=3):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (B, L), generator=g)
    tgt = torch.randint(0, vocab, (B, L), generator=g)
    return idx, tgt


# ---------------------------------------------------------------------------
# 1. 等价
# ---------------------------------------------------------------------------
def test_parallel_forward_parity(pair):
    native, ad, _ = pair
    idx, tgt = _batch()
    with torch.no_grad():
        lo_na, loss_na = native(idx, tgt)
        lo_ad, loss_ad = ad.forward(idx, tgt)
    assert torch.allclose(lo_na, lo_ad, rtol=1e-4, atol=1e-5), \
        float((lo_na - lo_ad).abs().max())
    assert abs(float(loss_na) - float(loss_ad)) < 1e-5


def test_streaming_forward_parity_vs_native_step(pair):
    """分块流式（chunk=L 一次喂入）与原生逐 token stream_step 等价。"""
    native, ad, _ = pair
    idx, _ = _batch()
    with torch.no_grad():
        state = native.init_stream(idx.shape[0])
        toks = []
        for t in range(idx.shape[1]):
            logits, state = native.stream_step(idx[:, t], state)
            toks.append(logits)                      # (B, V)
        lo_native = torch.stack(toks, dim=1)         # (B, T, V)
        lo_ad, _ = ad.forward_streaming(idx, chunk=idx.shape[1])
    assert torch.allclose(lo_native, lo_ad, rtol=1e-4, atol=1e-5), \
        float((lo_native - lo_ad).abs().max())


def test_streaming_vs_parallel_dual_form(pair):
    native, ad, _ = pair
    idx, _ = _batch()
    with torch.no_grad():
        lo_par, _ = ad.forward(idx)
        lo_str, _ = ad.forward_streaming(idx, chunk=16)
    dev = float((lo_par - lo_str).abs().max())
    assert dev < 4e-5, dev


def test_gradient_parity(pair):
    native, ad, _ = pair
    idx, tgt = _batch(seed=5)
    _, loss_na = native(idx, tgt)
    loss_na.backward()
    _, loss_ad = ad.forward(idx, tgt)
    loss_ad.backward()
    # 每个参数都有梯度（无静默零梯度）
    for name, p in ad.named_parameters():
        assert p.grad is not None, f"无梯度: {name}"
        assert p.grad.abs().sum() > 0 or name.endswith("evidence_pos_gain"), name
    # 抽样梯度与原生一致
    ng = dict(native.named_parameters())
    for name in ("tok_emb.weight", "blocks.0.mixer.in_proj.weight",
                 "blocks.0.mixer.coef", "blocks.1.ffn.down.weight"):
        g_na, g_ad = ng[name].grad, ad.P[name].grad
        assert torch.allclose(g_na, g_ad, rtol=1e-3, atol=1e-7), \
            (name, float((g_na - g_ad).abs().max()))


# ---------------------------------------------------------------------------
# 2. 契约覆盖审计（完备性证明）
# ---------------------------------------------------------------------------
def test_contract_ops_coverage(pair):
    """并行 + 流式前向后，被调用的适配器方法必须覆盖契约的全部前向算子。"""
    _, ad, rec = pair
    idx, tgt = _batch()
    with torch.no_grad():
        ad.forward(idx, tgt)
        ad.forward_streaming(idx, chunk=16)
    called = set(rec.called_ops())
    forward_ops = {"emb_lookup", "rmsnorm", "gemm", "swiglu", "residual_add",
                   "causal_depthwise_conv", "spectral_causal_conv",
                   "chunked_iir_scan", "content_gate",
                   "evidence_gate_parallel", "evidence_gate_chunk",
                   "uncertainty_gate", "ce_loss"}
    missing = forward_ops - called
    assert not missing, f"契约前向算子未被用到: {missing}"
    # 被调用的每个方法都映射回某个契约算子（无契约外内核）
    mapped = {m for ms in OP_TO_METHODS.values() for m in ms}
    outside = called - mapped
    assert not outside, f"契约外内核被调用: {outside}"


# ---------------------------------------------------------------------------
# 3. 装载语义
# ---------------------------------------------------------------------------
def test_load_native_missing_key_raises():
    torch.manual_seed(11)
    native = SimplCoreModel(CFG)
    ad = AdapterModel(CFG, TorchReferenceAdapter())
    sd = dict(native.state_dict())
    sd.pop("ln_f.weight")
    with pytest.raises(KeyError):
        ad.load_native(sd)


def test_rejects_unsupported_variants():
    with pytest.raises(ValueError):
        AdapterModel(SimplCoreConfig(vocab_size=32, dim=8, num_layers=1,
                                     num_bases=4, gate_ab=True),
                     TorchReferenceAdapter())
