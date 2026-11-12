# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 门控增强 + 多头谱基 mixer: 当前最强 linear-attention 结构并集.

设计动机（对齐 GatedDeltaNet / Gated Linear Attention / Mamba2）
================================================================
:class:`GatedSpectralConv2` 在 :class:`SpectralGlobalConv` 的结构骨架上
叠加三项结构升级，全部保持 严格因果 + O(L log L) FFT 训练 + O(1) 状态流式
+ 内容门控 + 无位置嵌入表：

a) **GLA/Mamba2 式 channel 独立双门**（``gate_ab=True``）：
   * ``gamma(t, c) = sigmoid(W_gamma h[t, c])`` —— Mamba2 的时间常数内容化
     （B_t 思想）：逐位置逐通道调制衰减写入强度；
   * ``beta(t, c)  = sigmoid(W_beta h[t, c])`` —— GatedDeltaNet 的写入门：
     逐位置逐通道调制状态写入强度。
   两个门都是**逐位置瞬时**乘法（不进核、不增状态），FFT 形式不变：
   ``u_gated = in_proj(h) * sigmoid(gate_proj(h)) * sigmoid(gamma(h)) * sigmoid(beta(h))``
   （content_gate 开启时与旧 SGC 同一 ``gate_proj`` 同路叠加），长程项仍走
   ``fft_causal_conv(u_gated, K)``，局部项 / feed 项同用 ``u_gated``。
   流式 IIR 路径的递推输入端 ``c_m * u[t]`` 换成 ``c_m * u_gated[t]``，
   状态形状不变 —— 两条路径对门是同一确定性逐位置乘法，天然逐位一致。

b) **多头谱基**（``num_heads > 1``）：振荡基系数 ``coef`` 从 ``(M, D)`` 升级为
   ``(M, G, head_dim)``（G = num_heads，head_dim = D // G）：M 个基 × 每 head
   独立系数组，把 M 维全局字典细化为 head 局部字典，提升内容路由表达力。
   核 ``K[tau, g, hd] = sum_m coef[m, g, hd] psi_m[tau]``，FFT 路径把
   ``(M, G*hd)`` 核广播回 ``(L, D)`` 全通道卷积，复杂度不变；流式 IIR
   state y1/y2 形状 ``(B, M, G, head_dim)`` —— 总元素数 M*G*hd == M*D，
   **O(1) 状态字节数与单头完全相同**。

c) **输出门**（``out_gate=True``，与 SGC 同名参数一致）：双门（输入 gamma/beta
   + 输出 out_gate）可同时启用。

诚实边界
--------
* 本模块只保证 数学正确 / 因果 / 双形式等价（FFT == IIR 逐位）/ O(1) 状态 /
  参数量精确；"质量更强"需 A100 训练实测（与 ``eval.py`` / ``extrapolation.py``
  探针协议配合）；
* 纯 torch，零第三方。

API
---
* :class:`GatedSpectralConv2`
* :func:`analytic_gated_spectral_params` —— 精确参数量（不实例化，bit-exact）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import (
    evidence_weight_step,
    evidence_weights,
    init_energy_sum,
    uncertainty_weights,
)
from .sgc import (
    THETA_CLAMP_HI,
    LayerStreamState,
    _init_oscillator_grid,
    _logit,
    depthwise_causal_conv,
    fft_causal_conv,
    make_norm,
    recurrent_dtype,
    theta_clamp_floor,
)


# =============================================================================
# 门控增强 + 多头谱基 mixer
# =============================================================================
class GatedSpectralConv2(nn.Module):
    """门控增强 + 多头谱基 SGC mixer（详见模块 docstring）。

    Args:
        dim: 通道宽度 D。
        num_bases: 振荡基数量 M（默认 64）。
        decay_max: 最长衰减时间常数（默认 8192）。
        local_kernel: depthwise 因果卷积宽度 K（默认 4）。
        num_heads: 谱基分组数 G（默认 1 即退化为原 SGC 单头谱基，向后兼容）；
            要求 ``dim % num_heads == 0``。
        content_gate: 是否保留 SGC 的内容门（默认 True）。
        out_gate: 输出门（默认 False）。
        gate_ab: 是否启用 GLA 式 gamma/beta 双门（默认 True；False 退化为
            原 SpectralGlobalConv 行为，便于 1:1 消融对照）。
        norm / fft_dtype / eps: 同 SGC。
    """

    def __init__(self, dim: int, num_bases: int = 64, decay_max: float = 8192.0,
                 local_kernel: int = 4, num_heads: int = 1,
                 content_gate: bool = True, out_gate: bool = False,
                 gate_ab: bool = True, norm: str = "rmsnorm",
                 fft_dtype: Optional[torch.dtype] = torch.float32,
                 eps: float = 1e-6,
                 evidence_gate: bool = False,     # P0-4: 与 SimplCoreConfig 对齐
                 uncertainty_gate: bool = False) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_bases = num_bases
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.local_kernel = local_kernel
        self.content_gate = content_gate
        self.out_gate = out_gate
        self.gate_ab = gate_ab
        self.fft_dtype = fft_dtype

        self.norm = make_norm(norm, dim, eps=eps)

        # --- 投影 / 门 -------------------------------------------------------
        self.in_proj = nn.Linear(dim, dim, bias=False)
        if content_gate:
            self.gate_proj = nn.Linear(dim, dim, bias=False)
        if gate_ab:
            # GLA/Mamba2 双门（逐位置逐通道）。
            #
            # 门偏置用**独立 nn.Parameter**（而不是 Linear.bias）：SimplCoreModel.
            # _init_weights 会把所有 nn.Linear 的 bias 归零，那样"初始写入受抑"的
            # 意图就失效了。偏置取 -2.0 → sigmoid(-2) ≈ 0.12（与 AbstentionHead
            # 的约定一致）：初始时记忆写入被压住，训练自己决定何时打开。
            #
            # 权重**必须非零**：零权重 + 无偏置只能给出常数门，γ/β 就退化成
            # 一个固定 0.5 的缩放，"逐通道调制写入强度"名存实亡。
            self.gate_gamma = nn.Linear(dim, dim, bias=False)
            self.gate_beta = nn.Linear(dim, dim, bias=False)
            self.gate_gamma_bias = nn.Parameter(torch.full((dim,), -2.0))
            self.gate_beta_bias = nn.Parameter(torch.full((dim,), -2.0))
            with torch.no_grad():
                nn.init.normal_(self.gate_gamma.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.gate_beta.weight, mean=0.0, std=0.02)
        if out_gate:
            self.out_gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # --- 长程振荡基（多头谱基：coef (M, G, head_dim)）--------------------
        rho0, theta0 = _init_oscillator_grid(num_bases, decay_max)
        # θ 钳制下界在构造时冻结（跟随 decay_max，见 sgc.theta_clamp_floor）
        self._theta_floor = theta_clamp_floor(decay_max)
        self.raw_rho = nn.Parameter(_logit(rho0))
        self.raw_theta = nn.Parameter(_logit(theta0 / math.pi))
        G, hd = self.num_heads, self.head_dim
        self.coef = nn.Parameter(torch.randn(num_bases, G, hd) * 0.02)

        # 逐通道瞬时项
        self.feed = nn.Parameter(torch.ones(dim))

        # 短程 depthwise conv
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=local_kernel,
                                    groups=dim, bias=False)
        with torch.no_grad():
            self.local_conv.weight.zero_()
            self.local_conv.weight[:, 0, -1] = 1.0   # identity at init

        self.out_scale = nn.Parameter(torch.ones(1))

        # --- 四层闸门：L2 证据门 / L3 不确定门（与单头谱基分支同一份数学）------
        # 修复 DREAMING_CMT §6 的 G4 缺口：γ/β 双门分支此前没有闸门，
        # 导致「闸门能力」随分支选择而变。现在两条分支共用 gates.py 的实现。
        self.evidence_gate = bool(evidence_gate)
        self.uncertainty_gate = bool(uncertainty_gate)
        self.gate_eps = eps
        if self.evidence_gate:
            self.evidence_gain = nn.Parameter(torch.tensor(1.0))
            self.evidence_pos_gain = nn.Parameter(torch.tensor(0.0))
            self.evidence_bias = nn.Parameter(torch.tensor(0.0))
        if self.uncertainty_gate:
            self.uncertainty_logit = nn.Parameter(torch.zeros(1))

    # -- 核 ------------------------------------------------------------------
    def oscillator_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(rho, theta) 各 (M,)，rho 在 (0,1)，theta 在 [floor, π−1e-3]。"""
        rho = torch.sigmoid(self.raw_rho)
        # θ 参数级钳制（P1，与 sgc.SpectralGlobalConv 同口径）：命题 5 误差界
        # 含 1/|sinθ_m|，下界构造时冻结（跟随 decay_max），上界 π−1e-3 绝对界。
        theta = (torch.sigmoid(self.raw_theta) * math.pi
                 ).clamp(self._theta_floor, THETA_CLAMP_HI)
        return rho, theta

    def build_kernel(self, length: int, device=None, dtype=None) -> torch.Tensor:
        """materialise 因果核库 K 形状 (length, D)。

        多头谱基：psi (L, M) @ coef.t() (M, G*hd) -> (L, G*hd)，再 reshape 回
        (L, D)（G*hd == D），与单头版同一全通道核广播。
        """
        rho, theta = self.oscillator_params()
        device = device if device is not None else self.coef.device
        dtype = dtype if dtype is not None else self.coef.dtype
        tau = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)
        log_rho = torch.log(rho.to(dtype)).unsqueeze(1)
        psi = torch.exp(log_rho * tau) * torch.sin((tau + 1.0) * theta.to(dtype).unsqueeze(1))
        C = self.coef.to(dtype)                                   # (M, G, hd)
        C = C.view(C.shape[0], C.shape[1] * C.shape[2])           # (M, D)
        kernel = C.t() @ psi                                        # (D, M) @ (M, L)
        return kernel.t().contiguous()                              # (L, D)

    # -- 门 -------------------------------------------------------------------
    def _content_gating(self, h: torch.Tensor) -> torch.Tensor:
        """u = in_proj(h) · [content_gate] · gamma(h) · beta(h)，逐位置逐通道。

        γ/β 的偏置是独立参数（``gate_gamma_bias`` / ``gate_beta_bias``），
        取 -2.0 使初始门 ≈ 0.12（写入受抑），但门本身**随输入变化**。
        FFT（``forward``）与 IIR（``step``）都走这一条路径 → 双形式逐位一致。
        """
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))
        if self.gate_ab:
            u = u * torch.sigmoid(self.gate_gamma(h) + self.gate_gamma_bias)
            u = u * torch.sigmoid(self.gate_beta(h) + self.gate_beta_bias)
        return u

    # -- 前向 -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, L, D = x.shape
        h = self.norm(x)
        u = self._content_gating(h)
        kernel = self.build_kernel(L, device=u.device, dtype=u.dtype)
        long = fft_causal_conv(u, kernel, self.fft_dtype)
        local = self._local_conv_full(u)

        if self.evidence_gate:
            long = long * self._evidence_weights(u)
        if self.uncertainty_gate:
            long = long * self._uncertainty_weights(long, local)

        y = long + local + self.feed * u
        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        return residual + self.out_scale * out

    # -- L2 / L3 闸门（委托给 gates.py，两条分支共用）--------------------------
    def _evidence_weights(self, u: torch.Tensor) -> torch.Tensor:
        """L2 证据门：因果前缀平均能量 → 逐位置权重 (B, L, 1)。"""
        return evidence_weights(u, self.evidence_gain,
                                self.evidence_pos_gain, self.evidence_bias)

    def _uncertainty_weights(self, long: torch.Tensor,
                             local: torch.Tensor) -> torch.Tensor:
        """L3 不确定门：长短路径分歧 → 逐位置阻尼 (B, L, 1)。"""
        return uncertainty_weights(long, local, self.uncertainty_logit,
                                   self.gate_eps)

    def _local_conv_full(self, u: torch.Tensor) -> torch.Tensor:
        K = self.local_kernel
        if K == 1:
            return u * self.local_conv.weight.view(1, 1, -1)
        up = F.pad(u.transpose(1, 2), (K - 1, 0))
        return F.conv1d(up, self.local_conv.weight, groups=self.dim).transpose(1, 2)

    # -- 流式 -----------------------------------------------------------------
    def init_state(self, batch: int, device, dtype) -> LayerStreamState:
        """IIR 二阶递推状态：y1/y2 (B, M, G, head_dim)（总元素 M*D，与单头相同）。

        递推状态按 recurrent_dtype 分配：低精度计算时退化为 fp32，避免 bf16/fp16
        下 IIR 递推逐 token 累积舍入误差。conv_buf 为短 FIR 历史、按计算精度。
        """
        M = self.num_bases
        G, hd = self.num_heads, self.head_dim
        rd = recurrent_dtype(dtype)
        y1 = torch.zeros(batch, M, G, hd, device=device, dtype=rd)
        y2 = torch.zeros(batch, M, G, hd, device=device, dtype=rd)
        buf = None
        if self.local_kernel > 1:
            buf = torch.zeros(batch, self.local_kernel - 1, self.dim,
                              device=device, dtype=dtype)
        energy_sum = init_energy_sum(batch, device) if self.evidence_gate else None
        return LayerStreamState(y1=y1, y2=y2, conv_buf=buf, length=0,
                                energy_sum=energy_sum)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state: LayerStreamState) -> Tuple[torch.Tensor, LayerStreamState]:
        """单 token O(1) 计算 / O(1) 状态；数值与整段 forward 逐位一致。

        递推在状态的 dtype（recurrent_dtype）中进行：低精度计算时状态已是 fp32，
        因此 IIR 振荡递推始终在 fp32 进行，输出 nxt 再 cast 回计算精度与局部路径
        混合。fp32 计算时状态即 fp32，行为与原实现逐位一致。
        """
        squeeze = x_t.dim() == 2
        if squeeze:
            x_t = x_t.unsqueeze(1)
        B, one, D = x_t.shape
        assert one == 1, "step() consumes exactly one position"
        G, hd = self.num_heads, self.head_dim

        # 递推 dtype 取自持久状态（低精度计算时为 fp32）。
        rd = state.y1.dtype

        h = self.norm(x_t)
        u = self._content_gating(h)                       # (B, 1, D)

        rho, theta = self.oscillator_params()
        a = (2.0 * rho * torch.cos(theta)).to(rd)         # (M,)
        b = (-(rho ** 2)).to(rd)
        c = torch.sin(theta).to(rd)

        um = u.squeeze(1).unsqueeze(1).to(rd)             # (B, 1, D) in rd
        # 多头谱基：状态 (B, M, G, hd)；递推逐 (M, G*hd) 通道（G*hd==D）
        y_new = a.view(1, self.num_bases, 1, 1) * state.y1 \
              + b.view(1, self.num_bases, 1, 1) * state.y2 \
              + c.view(1, self.num_bases, 1, 1) * um.reshape(B, 1, G, hd)
        state.y2, state.y1 = state.y1, y_new

        C = self.coef.to(rd)                                              # (M, G, hd)
        # y_new (B, M, G, hd) -> (B, M, G*hd)；与 C (M, G*hd) 逐 (M, G*hd) 求和 -> (B, D)
        M_ = self.num_bases
        y_new_m = y_new.reshape(B, M_, G * hd)                            # (B, M, D)
        C_m = C.reshape(M_, G * hd)                                       # (M, D)
        nxt = torch.einsum("bmd,md->bd", y_new_m, C_m)                   # (B, D)
        nxt = nxt.unsqueeze(1).to(u.dtype)                               # (B, 1, D) 回到计算精度
        loc, state.conv_buf = depthwise_causal_conv(u, self.local_conv.weight, state.conv_buf)

        # L2 / L3 闸门的流式对偶形式：统计量都只用过去，逐位与整段一致
        # （门控权重在 fp32 态里算，乘回前 cast 回计算精度——递推态 fp32、
        #   I/O 与计算 bf16，否则低精度路径被提升回 fp32）
        if self.evidence_gate:
            w, state.energy_sum = evidence_weight_step(
                u, state.energy_sum, state.length,
                self.evidence_gain, self.evidence_pos_gain, self.evidence_bias)
            nxt = nxt * w.to(nxt.dtype)
        if self.uncertainty_gate:
            nxt = nxt * uncertainty_weights(nxt, loc, self.uncertainty_logit,
                                            self.gate_eps).to(nxt.dtype)
        y = nxt + loc + self.feed * u

        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        state.length += 1
        y_t = x_t + self.out_scale * out
        return (y_t.squeeze(1) if squeeze else y_t), state

    # -- 内省 -----------------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, num_bases={self.num_bases}, heads={self.num_heads}, "
                f"local_kernel={self.local_kernel}, gate_ab={self.gate_ab}, "
                f"content_gate={self.content_gate}, out_gate={self.out_gate}, "
                f"gates=L2:{self.evidence_gate}/L3:{self.uncertainty_gate}")

    def streaming_state_bytes(self, batch: int, dtype=torch.float32) -> int:
        """流式状态字节数：(2*M*D + (K-1)*D) * batch * itemsize（多核不改，与单头同）。

        递推状态（y1/y2）按 recurrent_dtype 计字节：低精度计算时退化为 fp32。
        启用 L2 证据门时额外加 ``batch`` 个 fp32 的前缀能量和
        （常驻、与序列长度无关）。
        """
        itemsize = torch.empty(0, dtype=dtype).element_size()
        rd_itemsize = torch.empty(0, dtype=recurrent_dtype(dtype)).element_size()
        n = 2 * self.num_bases * self.dim * batch * rd_itemsize
        n += max(self.local_kernel - 1, 0) * self.dim * batch * itemsize
        nbytes = n
        if self.evidence_gate:
            nbytes += batch * torch.empty(0, dtype=torch.float32).element_size()
        return nbytes

    def effective_kernel_rank(self, length: int) -> int:
        with torch.no_grad():
            K = self.build_kernel(length)
            return int(torch.linalg.matrix_rank(K.t().float()))


# =============================================================================
# 参数量解析（bit-exact）
# =============================================================================
def analytic_gated_spectral_params(
    dim: int, num_bases: int = 64, num_heads: int = 1,
    content_gate: bool = True, out_gate: bool = False,
    gate_ab: bool = True, local_kernel: int = 4,
    norm: str = "rmsnorm",
    evidence_gate: bool = False, uncertainty_gate: bool = False,  # P0-4
) -> int:
    """:class:`GatedSpectralConv2` 的精确参数量（不实例化）。

    每一项（bias 全部为 False）：

    * ``in_proj``:            ``D^2``
    * ``gate_proj``（content_gate）:      ``D^2``
    * ``gate_gamma`` + ``gate_beta``（gate_ab）: ``2 * D^2``
    * ``gate_gamma_bias`` + ``gate_beta_bias``（gate_ab）: ``2 * D``
    * ``out_gate_proj``（out_gate）:      ``D^2``
    * ``out_proj``:           ``D^2``
    * ``raw_rho`` + ``raw_theta``:        ``2 * M``
    * ``coef``（多头谱基 (M, G, hd)）:    ``M * G * hd == M * D``
    * ``feed``:               ``D``
    * ``local_conv``（depthwise）:        ``D * K``
    * ``out_scale``:          ``1``
    * 前置 norm（rmsnorm 口径）:          ``D``

    由 ``tests/test_selective.py::test_analytic_params_bit_exact`` 守护。
    """
    M, G, D = num_bases, num_heads, dim
    hd = D // G
    n = D * D                                  # in_proj
    if content_gate:
        n += D * D
    if gate_ab:
        n += 2 * D * D                         # gate_gamma + gate_beta
        n += 2 * D                             # gate_gamma_bias + gate_beta_bias
    if out_gate:
        n += D * D
    n += D * D                                  # out_proj
    n += 2 * M                                 # raw_rho + raw_theta
    n += M * G * hd                             # coef (M, G, hd) == M*D
    n += D                                      # feed
    n += D * local_kernel                       # local_conv
    n += 1                                      # out_scale
    if evidence_gate:
        n += 3                                  # evidence_gain / pos_gain / bias
    if uncertainty_gate:
        n += 1                                  # uncertainty_logit
    if norm == "rmsnorm":
        n += D                                  # norm weight
    else:
        n += 2 * D                              # layernorm weight + bias
    return n


__all__ = [
    "GatedSpectralConv2",
    "analytic_gated_spectral_params",
]
