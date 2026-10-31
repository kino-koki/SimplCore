# -*- coding: utf-8 -*-
"""分块训练内核测试：chunked 前向必须与整段并行/逐 token 流式数值等价。

这是 ``--train-chunk`` 能被信任的前提。覆盖三层：

1. 闸门：``evidence_weights_chunk`` ≡ ``evidence_weights``（分段 cumsum）；
2. mixer：``forward_chunk`` 拼接 ≡ ``forward``，含重根极限 / 无门 / K=1；
3. model：``forward_chunked`` 的 logits/loss ≡ ``forward``，且 chunk=L 时梯度一致。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.gates import evidence_weights, evidence_weights_chunk
from simplcore.core.model import SimplCoreModel
from simplcore.core.sgc import SpectralGlobalConv


def _mixer(**kw) -> SpectralGlobalConv:
    torch.manual_seed(0)
    return SpectralGlobalConv(dim=16, num_bases=8, **kw).eval()


def _chunked_mixer(m: SpectralGlobalConv, x: torch.Tensor, chunk: int) -> torch.Tensor:
    state = m.init_state(x.shape[0], x.device, x.dtype)
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[1], chunk):
            y, state = m.forward_chunk(x[:, i:i + chunk], state)
            outs.append(y)
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# 1. 证据门的分块形式
# ---------------------------------------------------------------------------
def test_evidence_chunk_matches_full_parallel() -> None:
    torch.manual_seed(0)
    u = torch.randn(3, 64, 16)
    gain = torch.tensor(1.0)
    pos = torch.tensor(0.2)
    bias = torch.tensor(-0.1)
    full, _ = evidence_weights(u, gain, pos, bias), None
    energy = None
    outs = []
    length = 0
    for i in range(0, 64, 11):
        part, energy = evidence_weights_chunk(
            u[:, i:i + 11], energy, length, gain, pos, bias)
        length += 11
        outs.append(part)
    chunked = torch.cat(outs, dim=1)
    assert torch.allclose(full, chunked, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. mixer 级
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [1, 5, 16, 64, 200])
def test_mixer_chunk_matches_full(chunk: int) -> None:
    m = _mixer(local_kernel=4, content_gate=True, evidence_gate=True,
               uncertainty_gate=True, path_gate=True, out_gate=True)
    x = torch.randn(2, 64, 16)
    with torch.no_grad():
        full = m(x)
    chunked = _chunked_mixer(m, x, chunk)
    assert torch.allclose(full, chunked, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("kw", [
    {"init_mode": "logfreq"},                       # 小 theta 的重根极限支
    {"local_kernel": 1},                            # 无局部卷积缓冲
    {"content_gate": False, "evidence_gate": False,
     "uncertainty_gate": False, "path_gate": False, "out_gate": False},
])
def test_mixer_variants_chunk_matches_full(kw: dict) -> None:
    m = _mixer(**kw)
    x = torch.randn(2, 53, 16)
    with torch.no_grad():
        full = m(x)
    chunked = _chunked_mixer(m, x, 5)
    assert torch.allclose(full, chunked, atol=1e-5, rtol=1e-4)


def test_chunked_state_is_constant_size() -> None:
    m = _mixer()
    st = m.init_state(2, torch.device("cpu"), torch.float32)
    _, st = m.forward_chunk(torch.randn(2, 64, 16), st)
    _, st2 = m.forward_chunk(torch.randn(2, 64, 16), st)
    assert st2.y1.shape == st.y1.shape == (2, 8, 16)
    assert st2.length == 128
    assert st2.conv_buf.shape == (2, m.local_kernel - 1, 16)


# ---------------------------------------------------------------------------
# 3. model 级
# ---------------------------------------------------------------------------
def _model(**over) -> SimplCoreModel:
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=6,
                          ffn_ratio=2.0, local_kernel=4, dropout=0.0,
                          evidence_gate=True, uncertainty_gate=True,
                          sgc_path_gate=True, **over)
    torch.manual_seed(0)
    return SimplCoreModel(cfg).eval()


@pytest.mark.parametrize("chunk", [1, 7, 16, 33, 97, 128])
def test_model_chunked_matches_full(chunk: int) -> None:
    m = _model()
    x = torch.randint(0, 64, (2, 97))
    with torch.no_grad():
        logits, loss = m(x, x)
        logits_c, loss_c = m.forward_chunked(x, x, chunk_size=chunk)
    assert torch.allclose(logits, logits_c, atol=1e-5, rtol=1e-4)
    assert torch.allclose(loss, loss_c, atol=1e-5)


def test_model_chunk_equals_total_keeps_gradients() -> None:
    """chunk=L 时是同一张计算图，梯度必须与整段逐位一致。"""
    x = torch.randint(0, 64, (2, 32))
    m1 = _model()
    _, loss1 = m1(x, x)
    loss1.backward()
    g1 = {n: p.grad.clone() for n, p in m1.named_parameters() if p.grad is not None}

    m2 = _model()
    m2.load_state_dict(m1.state_dict())
    m2.zero_grad(set_to_none=True)
    _, loss2 = m2.forward_chunked(x, x, chunk_size=32)
    loss2.backward()
    for name, p in m2.named_parameters():
        if p.grad is None:
            continue
        assert torch.allclose(p.grad, g1[name], atol=1e-6, rtol=1e-4), name


def test_model_chunk_gradients_finite_and_truncated() -> None:
    x = torch.randint(0, 64, (2, 40))
    m = _model()
    _, loss = m.forward_chunked(x, x, chunk_size=8)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters())


def test_unsupported_mixer_fails_loudly() -> None:
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=6,
                          ffn_ratio=2.0, mixer="srf", srf_heads=2, srf_chunk=8)
    m = SimplCoreModel(cfg).eval()
    with pytest.raises(ValueError, match="forward_chunk"):
        m.forward_chunked(torch.randint(0, 64, (1, 16)))

def test_forward_chunk_step_matches_forward_chunked() -> None:
    """训练用的逐块接口与便利接口必须给出同一组 logits/loss。"""
    m = _model()
    x = torch.randint(0, 64, (2, 40))
    with torch.no_grad():
        ref_logits, ref_loss = m.forward_chunked(x, x, chunk_size=8)
        st = m.init_stream(2, device=x.device, dtype=m.tok_emb.weight.dtype)
        outs = []
        total, n = 0.0, 0
        for pos in range(0, 40, 8):
            part, loss_sum, n_part, st = m.forward_chunk_step(
                x[:, pos:pos + 8], x[:, pos:pos + 8], st)
            outs.append(part)
            total += float(loss_sum)
            n += n_part
            st = m._detach_stream_state(st)
        got_logits = torch.cat(outs, dim=1)
        got_loss = total / n
    assert torch.allclose(ref_logits, got_logits, atol=1e-5, rtol=1e-4)
    assert abs(ref_loss.item() - got_loss) < 1e-5