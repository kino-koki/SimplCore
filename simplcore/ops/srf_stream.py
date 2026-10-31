# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 谱振流（SRF）mixer 融合热路径的 JIT 加载与分发。

把 ``simplcore/csrc`` 里的 C++/CUDA 融合算子（SRF 专属）编译成可加载扩展，并提供
与 :class:`~simplcore.core.srf.SpectralResonanceFlow.step` 等价的 ``srf_stream_step``
入口。SGC 与 SRF 是不同 mixer 家族（前者 time-invariant IIR，后者 input-dependent
selective SSM），数学不同，故分模块、分守护测试（见 ``test_srf_stream.py``）。

设计约束（延续仓库"诚实纪律"）：
* 无 torch / 无编译器 / 无 CUDA 时 ``is_available()`` 返回 ``False``，调用方自动
  回退到纯 PyTorch 路径——**绝不悄悄出错**。
* 融合算子产出的 y_t 与状态更新必须与纯 PyTorch step **逐位近似**（fp32 阈值 1e-3）；
  等价性由 ``simplcore/core/tests/test_srf_stream.py`` 守护。
* 融合路径接受 ``float32`` / ``bfloat16`` / ``float16`` 输入：非 fp32 精度在 Python 层
  **归一化为 fp32** 后再下沉到 C++/CUDA（数值敏感的复数 SSM 递推恒在 fp32 执行），
  输出再 cast 回原精度。这既让 bf16 的 MI250 推理/服务路径用得上融合算子，又杜绝了
  "bf16 权重/状态被当作 fp32 字节误读"的潜在 bug；对 bf16 而言融合路径比纯 PyTorch
  的 bf16 step 更精确。fp32 输入走零拷贝快路径（与守护逐位一致）。
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

_EXT = None
_AVAILABLE: Optional[bool] = None
_BUILD_ERROR: Optional[BaseException] = None


def _csrc_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc"
    )


def _build() -> None:
    """惰性编译并加载 SRF 融合扩展（仅首次调用时）。"""
    global _EXT, _AVAILABLE, _BUILD_ERROR
    try:
        import torch  # noqa: F401
    except ImportError:
        _AVAILABLE = False
        return
    try:
        from torch.utils.cpp_extension import load

        sources = [os.path.join(_csrc_dir(), "srf_stream_ext.cpp")]
        define_macros = []
        if torch.cuda.is_available():
            cu = os.path.join(_csrc_dir(), "srf_stream_cuda.cu")
            if os.path.exists(cu):
                sources.append(cu)
                define_macros.append(("SRF_STREAM_BUILD_CUDA", "1"))
        _EXT = load(
            name="srf_stream_hotpath",
            sources=sources,
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=["-O2"],
            define_macros=define_macros,
            build_directory=os.path.join(_csrc_dir(), "build"),
            verbose=False,
        )
        _AVAILABLE = True
    except Exception as exc:  # 编译/链接失败：保守降级
        _AVAILABLE = False
        _BUILD_ERROR = exc


def is_available() -> bool:
    """SRF 融合热路径扩展是否已就绪（编译失败 / 无 torch 时为 False）。"""
    global _AVAILABLE
    if _AVAILABLE is None:
        _build()
    return bool(_AVAILABLE)


def build_error() -> Optional[BaseException]:
    """返回最近一次构建失败的异常（用于诊断），无则返回 None。"""
    if _AVAILABLE is None:
        _build()
    return _BUILD_ERROR


def srf_stream_step(mixer, x_t, state) -> Tuple[object, object]:
    """融合版逐 token SRF mixer step；接口与 ``mixer.step(x_t, state)`` 一致。

    从 mixer 组装权重张量（含 selective SSM 的 b_in / coef / skip / dt_proj）并调用
    底层算子；复数状态 (y1/y2 = h_re/h_im)、conv_buf、energy_sum 原地更新后写回 state。

    精度约定与 ``mixer_stream_step`` 同（详见模块 docstring）：fp32 内部递推，bf16/fp16
    在 Python 层归一化下沉，输出 cast 回原精度；fp32 走零拷贝快路径。
    """
    import torch

    if not is_available():
        raise RuntimeError(
            "srf 融合热路径不可用，请勿直接调用；请回退纯 PyTorch 路径"
        )

    fp32 = torch.float32
    accepted = (fp32, torch.bfloat16, torch.float16)
    squeeze = x_t.dim() == 3
    x_bd = x_t.squeeze(1) if squeeze else x_t
    orig_dtype = x_bd.dtype
    if orig_dtype not in accepted:
        raise RuntimeError(
            f"srf 融合热路径不支持 dtype {orig_dtype}；请回退纯 PyTorch 路径"
        )
    need_cast = orig_dtype != fp32

    device = x_bd.device
    x_bd = x_bd.contiguous().to(fp32)
    B, D = x_bd.shape

    st = state
    ssm = mixer.ssm
    K = mixer.local_kernel

    def prep(t):
        return t.detach().to(fp32) if need_cast else t.detach()

    # 复数状态（与 state.y1/state.y2 别名一致）
    y1_in = prep(st.y1)
    y2_in = prep(st.y2)
    if st.conv_buf is not None:
        conv_in = prep(st.conv_buf)
    else:
        conv_in = torch.zeros(B, 1, D, device=device, dtype=fp32)
    if st.energy_sum is not None:
        energy_in = prep(st.energy_sum).reshape(B)
    else:
        energy_in = torch.zeros(B, device=device, dtype=fp32)

    # 权重：fp32 路径即原张量；非 fp32 路径下 .to(fp32) 副本。内核恒收 fp32。
    wnorm = prep(mixer.norm.weight)
    in_w = prep(mixer.in_proj.weight)
    gate_w = prep(mixer.gate_proj.weight) if mixer.content_gate \
        else torch.zeros(D, D, device=device, dtype=fp32)
    out_w = prep(mixer.out_proj.weight)
    outgate_w = prep(mixer.out_gate_proj.weight) if mixer.out_gate \
        else torch.zeros(D, D, device=device, dtype=fp32)
    b_in = prep(ssm.b_in)
    coef = prep(ssm.coef)
    skip = prep(ssm.skip)
    dt_w = prep(ssm.dt_proj.weight.reshape(D).contiguous())
    dt_bias = float(ssm.dt_proj.bias.detach().reshape(()).to(fp32))
    rho = prep(torch.sigmoid(ssm.raw_rho))
    theta = prep(torch.sigmoid(ssm.raw_theta) * math.pi)
    local_w = prep(mixer.local_conv.weight.reshape(D, K).contiguous())

    y_t = _EXT.srf_stream_step(
        x_bd, y1_in, y2_in, conv_in, energy_in, wnorm, in_w, gate_w,
        out_w, outgate_w, b_in, coef, skip, dt_w, float(dt_bias), rho, theta,
        local_w, int(st.length), bool(mixer.content_gate),
        bool(mixer.evidence_gate), bool(mixer.uncertainty_gate),
        bool(mixer.out_gate), float(mixer.out_scale), float(mixer.norm.eps),
        1e-6, float(getattr(mixer, "evidence_gain", 1.0)),
        float(getattr(mixer, "evidence_pos_gain", 0.0)),
        float(getattr(mixer, "evidence_bias", 0.0)),
        float(getattr(mixer, "uncertainty_logit", 0.0)),
    )
    y_t = y_t.to(orig_dtype)

    # 状态回写：fp32 快路径是同一张量（原地已更新）；非 fp32 路径拷回原精度状态。
    if need_cast:
        st.y1.copy_(y1_in)
        st.y2.copy_(y2_in)
        if st.conv_buf is not None:
            st.conv_buf.copy_(conv_in)
        if st.energy_sum is not None:
            st.energy_sum = energy_in.reshape(B, 1).to(orig_dtype)
    else:
        st.y1 = y1_in
        st.y2 = y2_in
        st.conv_buf = conv_in if st.conv_buf is not None else None
        if st.energy_sum is not None:
            st.energy_sum = energy_in.reshape(B, 1)
    # 同步持久 ssm_state 别名（若 SRF 走融合路径，ssm.step 不被调用，scratch 留空）
    if st.ssm_state is not None:
        st.ssm_state.h_re = st.y1
        st.ssm_state.h_im = st.y2
        st.ssm_state.scratch = None
    st.length += 1
    if squeeze:
        y_t = y_t.unsqueeze(1)
    return y_t, state


def self_test(tol: float = 1e-3) -> bool:
    """在已加载扩展的前提下，对一份随机 SRF mixer 做一次融合 vs 纯 PyTorch 的
    等价性自检；返回是否通过。不可用时返回 False。"""
    import torch

    if not is_available():
        return False
    from simplcore.core.srf import SpectralResonanceFlow

    cfg_dim, cfg_M, cfg_K = 64, 16, 4
    m = SpectralResonanceFlow(cfg_dim, num_bases=cfg_M, local_kernel=cfg_K,
                              content_gate=True, out_gate=True,
                              evidence_gate=True, uncertainty_gate=True,
                              norm="rmsnorm")
    m.eval()
    m2 = SpectralResonanceFlow(cfg_dim, num_bases=cfg_M, local_kernel=cfg_K,
                               content_gate=True, out_gate=True,
                               evidence_gate=True, uncertainty_gate=True,
                               norm="rmsnorm")
    m2.load_state_dict(m.state_dict())
    m2.use_mixer_stream = True

    B = 2
    st1 = m.init_state(B, "cpu", torch.float32)
    st2 = m2.init_state(B, "cpu", torch.float32)
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(8):
        x = torch.randn(B, 1, cfg_dim)
        y1, st1 = m.step(x, st1)
        y2, st2 = m2.step(x, st2)
        worst = max(worst, (y1 - y2).abs().max().item())
        worst = max(worst, (st1.y1 - st2.y1).abs().max().item())
        worst = max(worst, (st1.y2 - st2.y2).abs().max().item())
    return worst < tol
