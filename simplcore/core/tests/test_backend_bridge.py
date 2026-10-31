# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""backend_bridge 守护测试 —— C-ABI 编组层经真实 ctypes 指针验证。"""
from __future__ import annotations

import ctypes
from typing import Any

import pytest
import torch

from simplcore.core.backend_bridge import (
    ABI_SPEC,
    CtypesBackendAdapter,
    FakeCLib,
    REQUIRED_SYMBOLS,
    symbol_report,
)
from simplcore.core.conformance_runner import TorchReferenceAdapter


@pytest.fixture(scope="module")
def br() -> CtypesBackendAdapter:
    return CtypesBackendAdapter(FakeCLib())


REF = TorchReferenceAdapter()


# ---------------------------------------------------------------------------
# 逐算子 parity（fp32 路径，经真实指针读写）
# ---------------------------------------------------------------------------
def test_parity_emb_lookup(br) -> None:
    torch.manual_seed(1)
    w = torch.randn(16, 8)
    ids = torch.randint(0, 16, (5,))
    torch.testing.assert_close(br.emb_lookup(w, ids), REF.emb_lookup(w, ids))


def test_parity_rmsnorm_gemm_swiglu_residual(br) -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 5, 8)
    w = torch.randn(8)
    torch.testing.assert_close(br.rmsnorm(x, w, 1e-5), REF.rmsnorm(x, w, 1e-5))
    a, b = torch.randn(3, 4), torch.randn(4, 6)
    torch.testing.assert_close(br.gemm(a, b), REF.gemm(a, b))
    g, u = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    torch.testing.assert_close(br.swiglu(g, u), REF.swiglu(g, u))
    torch.testing.assert_close(br.residual_add(g, u), REF.residual_add(g, u))


def test_parity_convs(br) -> None:
    torch.manual_seed(3)
    u = torch.randn(2, 7, 4)
    k = torch.randn(7, 4)                    # 核库 (L, D)
    torch.testing.assert_close(br.spectral_causal_conv(u, k),
                               REF.spectral_causal_conv(u, k))
    wc = torch.randn(4, 1, 3)
    y1, nb1 = br.causal_depthwise_conv(u, wc, None)
    y2, nb2 = REF.causal_depthwise_conv(u, wc, None)
    torch.testing.assert_close(y1, y2)
    torch.testing.assert_close(nb1, nb2)
    buf = torch.randn(2, 2, 4)
    y3, nb3 = br.causal_depthwise_conv(u, wc, buf)
    y4, nb4 = REF.causal_depthwise_conv(u, wc, buf)
    torch.testing.assert_close(y3, y4)
    assert nb3.shape == nb4.shape
    torch.testing.assert_close(nb3, nb4)


def test_parity_chunked_iir_scan_both_paths(br) -> None:
    torch.manual_seed(4)
    B, C, D, M = 2, 6, 4, 3
    u = torch.randn(B, C, D)
    y1 = torch.randn(B, M, D)
    y2 = torch.randn(B, M, D)
    a, b, c = torch.randn(M), torch.randn(M), torch.randn(M)
    o1, p1, q1 = br.chunked_iir_scan(u, y1, y2, a, b, c)
    o2, p2, q2 = REF.chunked_iir_scan(u, y1, y2, a, b, c)
    torch.testing.assert_close(o1, o2)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(q1, q2)
    coef = torch.randn(M, D)
    l1, _, _ = br.chunked_iir_scan(u, y1, y2, a, b, c, coef=coef)
    l2, _, _ = REF.chunked_iir_scan(u, y1, y2, a, b, c, coef=coef)
    torch.testing.assert_close(l1, l2)          # 融合投影 epilogue 路径


def test_parity_gates(br) -> None:
    torch.manual_seed(5)
    u = torch.randn(2, 6, 4)
    torch.testing.assert_close(
        br.evidence_gate_parallel(u, 0.3, 0.1, -0.2),
        REF.evidence_gate_parallel(u, 0.3, 0.1, -0.2))
    es = torch.randn(2, 1)
    w1, e1 = br.evidence_gate_chunk(u, es, 10, 0.3, 0.1, -0.2)
    w2, e2 = REF.evidence_gate_chunk(u, es, 10, 0.3, 0.1, -0.2)
    torch.testing.assert_close(w1, w2)
    torch.testing.assert_close(e1, e2)
    w3, e3 = br.evidence_gate_chunk(u, None, 0, 0.3, 0.1, -0.2)
    w4, e4 = REF.evidence_gate_chunk(u, None, 0, 0.3, 0.1, -0.2)
    torch.testing.assert_close(w3, w4)
    torch.testing.assert_close(e3, e4)
    long_, local = torch.randn(2, 6, 4), torch.randn(2, 6, 4)
    torch.testing.assert_close(br.uncertainty_gate(long_, local, 0.7, 1e-6),
                               REF.uncertainty_gate(long_, local,
                                                    torch.tensor(0.7), 1e-6))
    h, w = torch.randn(2, 8), torch.randn(3, 8)
    torch.testing.assert_close(br.content_gate(h, w), REF.content_gate(h, w))


def test_parity_ce_loss_and_adamw(br) -> None:
    torch.manual_seed(6)
    logits = torch.randn(4, 10)
    labels = torch.tensor([1, 2, -100, 5])
    torch.testing.assert_close(br.ce_loss(logits, labels),
                               REF.ce_loss(logits, labels))
    p = torch.randn(6)
    g = torch.randn(6)
    m = torch.zeros(6)
    v = torch.zeros(6)
    frozen = torch.tensor([True, False, False, True, False, False])
    kw = (0.01, 0.9, 0.999, 1e-8, 0.1)
    r1 = br.adamw_frozen_mask(p, g, m, v, frozen, *kw, 3)
    r2 = REF.adamw_frozen_mask(p, g, m, v, frozen, *kw, 3)
    for x, y in zip(r1, r2):
        torch.testing.assert_close(x, y)
    # 冻结位逐位不动
    torch.testing.assert_close(r1[0][frozen], p[frozen])


def test_noncontiguous_input_autofix(br) -> None:
    """非连续输入必须被正确处理（contiguous 化后指针才有效）。"""
    torch.manual_seed(7)
    big = torch.randn(4, 8)
    a = big[:, ::2]                     # 非连续视图
    b = torch.randn(4, 4)
    torch.testing.assert_close(br.gemm(a, b), REF.gemm(a.contiguous(), b))


def test_dtype_contract(br) -> None:
    """dtype 契约：fp32 单一契约 + int64 严格项。

    * 浮点输入走 ``_to_fp32`` **自动升精度**（v3 契约，与 ops/mixer_stream
      同哲学），不得拒绝——但结果必须与 fp32 参考逐位一致，并回写原 dtype；
    * 整数契约项（ce_loss 的 labels）仍然**严格 int64**，传错必须 TypeError。
    """
    # rmsnorm：低精度 weight 自动升精度，结果与 fp32 参考一致、回写原 dtype
    w32 = torch.randn(4)
    x32 = torch.randn(2, 4)
    ref = br.rmsnorm(x32, w32, 1e-5)
    got = br.rmsnorm(x32.to(torch.float64), w32.to(torch.float64), 1e-5)
    assert got.dtype == torch.float64
    torch.testing.assert_close(got.float(), ref)
    # ce_loss：labels 必须 int64（fp32 labels 现在也不许静默升精度——语义会错）
    with pytest.raises(TypeError, match="dtype"):
        br.ce_loss(torch.randn(4, 8), torch.randn(4))
    with pytest.raises(TypeError, match="dtype"):
        br.ce_loss(torch.randn(4, 8), torch.arange(4, dtype=torch.int32))


def test_kernel_error_code_propagates() -> None:
    class BadLib(FakeCLib):
        def _op_gemm(self, *args: Any) -> int:
            return 7

    br = CtypesBackendAdapter(BadLib())
    with pytest.raises(RuntimeError, match="gemm.*7"):
        br.gemm(torch.randn(2, 2), torch.randn(2, 2))


# ---------------------------------------------------------------------------
# 符号校验 / 加载
# ---------------------------------------------------------------------------
def test_missing_symbol_rejected() -> None:
    lib = FakeCLib()
    del lib.sc_gemm                       # 实例属性（__init__ 经 setattr 绑定）
    with pytest.raises(RuntimeError, match="sc_gemm"):
        CtypesBackendAdapter(lib)


def test_symbol_report_and_version_gate(tmp_path) -> None:
    with pytest.raises(OSError):
        symbol_report(tmp_path / "nope.dll")
    # ABI_SPEC 覆盖完备性：14 个设备内核；宿主侧 5 个不进 ABI
    assert len(REQUIRED_SYMBOLS) == 14
    host = {"stream_state_init", "stream_state_serialize",
            "stream_state_deserialize", "weight_decay_exempt", "shard_allreduce"}
    device = {m for m in vars(TorchReferenceAdapter)
              if not m.startswith("_") and m not in host and m != "ops_registry"
              and callable(getattr(TorchReferenceAdapter, m))}
    assert set(ABI_SPEC) == device, set(ABI_SPEC) ^ device


# ---------------------------------------------------------------------------
# 宿主侧算子：与参考适配器 byte 兼容
# ---------------------------------------------------------------------------
def test_host_ops_byte_compat(br) -> None:
    state = {"a": torch.randn(2, 3), "b": torch.randn(4)}
    assert br.stream_state_serialize(state) == REF.stream_state_serialize(state)
    back = br.stream_state_deserialize(REF.stream_state_serialize(state))
    for k in state:
        torch.testing.assert_close(back[k], state[k])
    init = br.stream_state_init({"s": (2, 3)})
    assert init["s"].dtype == torch.float32 and (init["s"] == 0).all()
    assert br.weight_decay_exempt(["rho", "w", "theta_x"]) == [True, False, True]
    g = [torch.randn(3), torch.randn(3)]
    torch.testing.assert_close(br.shard_allreduce(g), REF.shard_allreduce(g))


# ---------------------------------------------------------------------------
# 端到端：AdapterModel 全链路（桥接 vs 参考适配器，同权重同输入）
# ---------------------------------------------------------------------------
def test_adapter_model_full_parity_with_bridge(br) -> None:
    from simplcore.core.adapter_model import AdapterModel
    from simplcore.core.config import SimplCoreConfig
    from simplcore.core.model import SimplCoreModel

    torch.manual_seed(11)
    cfg = SimplCoreConfig(vocab_size=32, dim=32, num_layers=2, num_bases=8,
                          local_kernel=4, ffn_ratio=2.0, norm="rmsnorm",
                          decay_max=256.0, tie_embeddings=False)
    native = SimplCoreModel(cfg)
    sd = native.state_dict()
    m_ref = AdapterModel(cfg, TorchReferenceAdapter())
    m_ref.load_native(sd)
    m_br = AdapterModel(cfg, br)
    m_br.load_native(sd)
    idx = torch.randint(0, 32, (2, 24))
    targets = torch.randint(0, 32, (2, 24))
    with torch.no_grad():
        lo, l1 = m_ref.forward(idx, targets)
        lb, l2 = m_br.forward(idx, targets)
    torch.testing.assert_close(lo, lb)
    torch.testing.assert_close(l1, l2, rtol=0, atol=1e-6)
    # 流式路径（forward_streaming：分块流式前向）
    with torch.no_grad():
        lo_s, _ = m_ref.forward_streaming(idx, 8, None)
        lb_s, _ = m_br.forward_streaming(idx, 8, None)
    torch.testing.assert_close(lo_s, lb_s)


# 类型引用（保持 import 有意义）
_ = ctypes
