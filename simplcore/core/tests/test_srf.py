# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""谱振流网络（SRFN）的守护测试。

这组测试防的是"看起来接上了，其实没接上"的几类错误：

1. **训练/推理两条路径不一致** —— chunk 并行算出的结果和逐 token 递推不同。
   这种 bug 不会在训练里暴露（loss 照降），只在线上生成时表现为"答非所问"，
   事后极难定位。所以用 1e-4 的硬阈值卡住。
2. **selective 名不副实** —— Delta 若与输入无关，模型和旧的固定卷积没区别。
   用"改输入 -> Delta 必须变"来卡。
3. **泄漏未来** —— chunk 内是下三角矩阵，一旦 mask 方向写反（上三角保留下三角
   置零），模型会偷看答案，训练 loss 会异常地好。
4. **推理状态必须是 O(1) 定长** —— SRFN 的流式状态与序列长度无关（无 KV cache）。
"""
from __future__ import annotations

import torch
import pytest

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.ssm import SelectiveSpectralSSM


# ---------------------------------------------------------------------------
# SelectiveSpectralSSM
# ---------------------------------------------------------------------------
def _ssm(dim=8, bases=4, chunk=4) -> SelectiveSpectralSSM:
    torch.manual_seed(0)
    return SelectiveSpectralSSM(dim, num_bases=bases, chunk_size=chunk)


def test_parallel_matches_streaming() -> None:
    """chunked 并行 scan 必须与逐 token 递推逐位一致。"""
    m = _ssm()
    u = torch.randn(2, 12, 8)
    par = m(u)

    st = m.init_state(2, u.device, u.dtype)
    seq = []
    for t in range(u.shape[1]):
        y, st = m.step(u[:, t:t + 1], st)
        seq.append(y)
    seq = torch.cat(seq, dim=1)

    assert torch.allclose(par, seq, atol=1e-5), \
        f"并行与串行不一致 max|diff|={(par - seq).abs().max().item():.3e}"


@pytest.mark.parametrize("chunk", [1, 3, 5, 12, 16])
def test_chunk_size_invariance(chunk: int) -> None:
    """块长只是实现细节，不应改变计算结果。"""
    m = _ssm(chunk=4)
    m2 = _ssm(chunk=chunk)
    m2.load_state_dict(m.state_dict())
    u = torch.randn(2, 12, 8)
    assert torch.allclose(m(u), m2(u), atol=1e-5)


def test_no_future_leak() -> None:
    """改未来的 token 不能影响过去的输出。"""
    m = _ssm()
    u = torch.randn(2, 12, 8)
    y = m(u)
    u2 = u.clone()
    u2[:, 6:] += 10.0
    assert (m(u2)[:, :6] - y[:, :6]).abs().max().item() < 1e-6


def test_step_size_is_input_dependent() -> None:
    """selective 的判据：Delta 必须随输入变化。

    这条测试挡的是"Delta 写成了常数"——那样模型退化成固定卷积，
    名字里带 selective 但实际没有选择性。
    """
    m = _ssm()
    a = torch.zeros(1, 1, 8)
    b = torch.ones(1, 1, 8) * 3.0
    da, db = m.step_size(a), m.step_size(b)
    assert (da - db).abs().max().item() > 1e-4, "Delta 与输入无关，selective 失效"


def test_state_size_independent_of_length() -> None:
    """状态字节数只与 (batch, bases, dim) 有关，与序列长度无关。"""
    m = _ssm()
    short, _ = m(torch.randn(1, 4, 8), return_state=True)
    long, st_long = m(torch.randn(1, 512, 8), return_state=True)
    st_short = m.init_state(1, torch.device("cpu"), torch.float32)
    assert st_long.num_bytes() == st_short.num_bytes()
    assert m.streaming_state_bytes(1) == st_long.num_bytes()


def test_gradients_reach_all_parameters() -> None:
    """没有"接上了但拿不到梯度"的悬挂参数。"""
    m = _ssm()
    u = torch.randn(2, 8, 8, requires_grad=True)
    m(u).sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not missing, f"这些参数没有梯度：{missing}"


# ---------------------------------------------------------------------------
# 整机：SRFN 作为 SimplCore 的 mixer
# ---------------------------------------------------------------------------
def _tiny(**kw) -> SimplCoreModel:
    base = dict(vocab_size=64, dim=32, num_layers=2, num_bases=4, ffn_ratio=2.0,
                mixer="srf", num_heads=2, srf_heads=2, srf_chunk=4, local_kernel=3)
    base.update(kw)
    cfg = SimplCoreConfig(**base)
    cfg.validate()
    return SimplCoreModel(cfg)


def test_srf_model_streaming_matches_full() -> None:
    torch.manual_seed(0)
    m = _tiny()
    assert m.supports_streaming
    m.eval()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        full = m(x)[0, -1]
        st = m.init_stream(1, device=x.device, dtype=torch.float32)
        last = None
        for t in range(x.shape[1]):
            last, st = m.stream_step(x[:1, t], st)
    assert (full - last).abs().max().item() < 1e-3


def test_srf_model_causal() -> None:
    torch.manual_seed(0)
    m = _tiny()
    m.eval()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = m(x)
        x2 = x.clone()
        x2[:, 8:] = 0
        b = m(x2)
    assert (a[:, :8] - b[:, :8]).abs().max().item() < 1e-6


def test_srf_close_to_sgc_in_params() -> None:
    """同宽度下 srf 与 sgc 参数量应在 5% 以内 —— 否则 A/B 不公平。"""
    torch.manual_seed(0)
    sgc = _tiny(mixer="sgc")
    srf = _tiny(mixer="srf")
    ratio = srf.num_params() / sgc.num_params()
    assert 0.95 < ratio < 1.05, f"参数比 {ratio:.3f} 超出 5%"


def test_srf_streaming_state_is_o1() -> None:
    """SRFN 的流式状态是 O(1) 定长：生成 6 个与 200 个 token 的状态字节数相同。"""
    torch.manual_seed(0)
    m = _tiny()
    m.eval()
    x = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        st_short = m.init_stream(1, device=x.device, dtype=torch.float32)
        for t in range(x.shape[1]):
            _, st_short = m.stream_step(x[:, t], st_short)
        b_short = st_short.num_bytes()
        st_long = m.init_stream(1, device=x.device, dtype=torch.float32)
        for t in range(200):
            _, st_long = m.stream_step(x[:, t % x.shape[1]], st_long)
    assert st_long.num_bytes() == b_short, "SRFN 流式状态随生成长度增长了（应为 O(1)）"


def test_srf_checkpoint_roundtrip() -> None:
    """存档 → 恢复 → 继续生成，结果必须和不中断的生成一致。"""
    torch.manual_seed(0)
    m = _tiny()
    m.eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        st = m.init_stream(1, device=x.device, dtype=torch.float32)
        for t in range(4):
            _, st = m.stream_step(x[:, t], st)
        ckpt = m.checkpoint_stream(st)
        st_a = m.restore_stream(ckpt)
        st_b = m.init_stream(1, device=x.device, dtype=torch.float32)
        for t in range(4):
            _, st_b = m.stream_step(x[:, t], st_b)
        for t in range(4, 8):
            la, st_a = m.stream_step(x[:, t], st_a)
            lb, st_b = m.stream_step(x[:, t], st_b)
    assert (la - lb).abs().max().item() < 1e-4, \
        f"恢复后的流式输出对不上 max|diff|={(la - lb).abs().max().item():.3e}"


@pytest.mark.parametrize("kw", [
    dict(mixer="srf"),
    dict(mixer="srf", out_gate=True),
])
def test_analytic_params_match_model(kw) -> None:
    """解析参数量必须位精确 —— 预算规划器靠它，差 1 都算错。

    ``analytic_params`` 决定 ``--target-params`` 怎么切宽深比，它一旦和真实
    模型不符，所有"等参数 A/B"的前提就是假的。
    """
    from simplcore.core.config import analytic_params
    torch.manual_seed(0)
    m = _tiny(**kw)
    assert analytic_params(m.config) == m.num_params()
