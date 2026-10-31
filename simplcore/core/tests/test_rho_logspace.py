# -*- coding: utf-8 -*-
"""log-space 极点参数化：让 4M 上下文的最慢基真正可达。

旧参数化 ``rho = sigmoid(raw).clamp(max=1-1e-6)`` 有两道上限：

* fp32 sigmoid 在 raw ≳ 17.3 时饱和到 1.0；
* 初始化 ``_logit(..., eps=1e-4)`` 把 rho 夹到 0.9999（时间常数 ~1e4），
  所以 ``decay_max=4e6`` 的预设实际只铺到 1e4。

新参数化 ``rho = exp(-softplus(raw_rho_log))`` 无饱和，可表达
``0.99999975``（4M 所需）；旧 checkpoint 通过 ``_load_from_state_dict`` 迁移。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig, get_preset
from simplcore.core.sgc import SpectralGlobalConv


def _logspace(**kw) -> SpectralGlobalConv:
    torch.manual_seed(0)
    return SpectralGlobalConv(dim=16, num_bases=8, decay_max=4_000_000,
                              local_kernel=4, content_gate=True,
                              evidence_gate=False, uncertainty_gate=False,
                              init_mode="logfreq", rho_param="logspace",
                              **kw)


def test_logspace_reaches_four_million_tokens() -> None:
    m = _logspace().eval()
    with torch.no_grad():
        rho, _ = m.oscillator_params()
    assert float(rho.max()) >= 0.9999997, float(rho.max())
    # 4M 所需的时间常数 ≈ 4e6；fp32 在 1 附近的分辨率仍够。
    assert float(rho.max()) < 1.0


def test_sigmoid_cannot_reach_four_million_tokens() -> None:
    """对照组：旧参数化在 decay_max=4e6 时仍被夹在 0.9999。"""
    m = SpectralGlobalConv(dim=16, num_bases=8, decay_max=4_000_000,
                           content_gate=True, evidence_gate=False,
                           uncertainty_gate=False, rho_param="sigmoid").eval()
    with torch.no_grad():
        rho, _ = m.oscillator_params()
    assert float(rho.max()) <= 0.99991, float(rho.max())


def test_logspace_dual_form_matches() -> None:
    m = _logspace().eval()
    x = torch.randn(1, 2048, 16)
    with torch.no_grad():
        full = m(x)
        st = m.init_state(1, x.device, x.dtype)
        outs = []
        for t in range(2048):
            y, st = m.step(x[:, t:t + 1], st)
            outs.append(y)
        streamed = torch.cat(outs, dim=1)
    err = float((full - streamed).abs().max())
    assert err < 1e-3, err


def test_default_is_sigmoid_no_regression() -> None:
    cfg = SimplCoreConfig()
    assert cfg.sgc_rho_param == "sigmoid"
    cfg.validate()          # 不抛异常即默认行为未变


def test_tuan_2b_enables_logspace_for_4m() -> None:
    cfg = get_preset("tuan-2b")
    assert cfg.sgc_rho_param == "logspace"
    assert cfg.sgc_init_mode == "logfreq"
    assert cfg.decay_max == 4_000_000
    cfg.validate()


def test_logspace_rejected_for_non_sgc_mixer() -> None:
    cfg = SimplCoreConfig(mixer="srf", srf_heads=2, sgc_rho_param="logspace")
    with pytest.raises(ValueError, match="仅支持 SGC"):
        cfg.validate()


def test_invalid_rho_param_rejected() -> None:
    with pytest.raises(ValueError, match="sgc_rho_param"):
        SimplCoreConfig(sgc_rho_param="tanh").validate()
    with pytest.raises(ValueError, match="rho_param"):
        SpectralGlobalConv(dim=16, num_bases=4, rho_param="tanh")


def test_checkpoint_migrates_between_parameterizations() -> None:
    torch.manual_seed(0)
    old = SpectralGlobalConv(dim=16, num_bases=8, decay_max=8192,
                             content_gate=True, evidence_gate=False,
                             uncertainty_gate=False, rho_param="sigmoid")
    with torch.no_grad():
        old.raw_rho.copy_(old.raw_rho + 0.3)
    new = SpectralGlobalConv(dim=16, num_bases=8, decay_max=8192,
                             content_gate=True, evidence_gate=False,
                             uncertainty_gate=False, rho_param="logspace")
    new.load_state_dict(old.state_dict())
    with torch.no_grad():
        rho_old, theta_old = old.oscillator_params()
        rho_new, theta_new = new.oscillator_params()
    assert torch.allclose(rho_old, rho_new, atol=1e-7)
    assert torch.allclose(theta_old, theta_new, atol=1e-7)