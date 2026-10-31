# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""谱振流 mixer —— 谱振流网络 (SRFN) 的混合器。

名字的来处
----------
原来的 ``SpectralGlobalConv``（谱全局卷积）只有"卷积"这一面：核是固定的，
对任何输入都用同一组衰减振荡基。加了 SSM 之后，模型的记忆**变得可以被输入
选择**——同一个基在这一步可以记得久一点，下一步就衰减掉。这是"流"而不是
"卷积"，所以整个模型改称：

* 中文：**谱振流网络**
* 英文：Spectral Resonance Flow Network (**SRFN**)
* mixer 名：``srf``

三条支路，各管一段
------------------
1. **selective SSM**（``SelectiveSpectralSSM``）：慢变、可选择的长期记忆。
   M 个阻尼振荡基，步长 Delta 由输入预测——这就是"选择记多久"。
2. **depthwise 因果卷积**：快变、精确的局部模式（1~3 个 token 的邻接）。
   SSM 的 Delta 再灵活也有最小步长，局部细节交给卷积更省。
3. **两层门控**（证据 / 不确定性）：沿用原 SGC 的设计。它们的作用不是加
   表达力，而是**抑制**——上下文没有可用信号时把长程支路关掉，宁可答不出
   也不要编。这个"知道自己不知道"的能力是小模型最缺的，不能因为换了
   mixer 就丢掉。

与 ``SpectralGlobalConv`` 的关系
--------------------------------
不是叠加，是**替代**。谱核（FFT 全局卷积）和 SSM 的振荡基是同一族数学对象：
前者是后者的 time-invariant 卷积形式。两个都留着等于用两种算法算同一个量，
白白多花一份算力。所以：

* ``mixer="sgc"``  —— 旧路径，FFT + 固定核，完全保留（历史实验可复现）
* ``mixer="srf"``  —— 新路径，selective SSM，本篇

代价：FFT 的 O(L log L) 换成了 chunk 内 Toeplitz 的 O(L*C)，C=32 时约
+17% MACs。换来的是逐 token 的 selectivty，以及不再依赖 FFT（CPU 上短序列
的 FFT 反而比直接卷积慢，实测见 ``docs/SRFN.md``）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gates import evidence_weight_step, evidence_weights, uncertainty_weights
from .sgc import LayerStreamState, depthwise_causal_conv, make_norm, recurrent_dtype
from .ssm import SelectiveSpectralSSM, SSMState


class SpectralResonanceFlow(nn.Module):
    """谱振流混合器：selective 状态空间 + 局部卷积 + 证据/不确定性门。

    Args:
        dim: 通道宽度 D。
        num_bases: 振荡基个数 M（= SSM 的状态维度）。
        num_heads: Delta 的头数 H（每个头内的通道共享步长）。
        chunk_size: chunked scan 的块长 C。
        decay_max: 最长衰减时间常数，用于初始化 rho 的跨度。
        local_kernel: depthwise 因果卷积宽度 K。
        content_gate / out_gate: 内容门控（沿用原 SGC）。
        evidence_gate / uncertainty_gate: 两层抑制门（沿用原 SGC）。
        norm: ``rmsnorm`` | ``layernorm``。
        eps: norm 的 epsilon。
    """

    def __init__(
        self,
        dim: int,
        num_bases: int = 16,
        num_heads: int = 1,          # 保留签名兼容；SSM 现在用 per-token 标量 Delta
        chunk_size: int = 32,
        decay_max: float = 1024.0,
        local_kernel: int = 4,
        content_gate: bool = True,
        out_gate: bool = False,
        evidence_gate: bool = True,
        uncertainty_gate: bool = True,
        norm: str = "rmsnorm",
        eps: float = 1e-6,
        use_mixer_stream: bool = False,
    ) -> None:
        super().__init__()
        if local_kernel < 1:
            raise ValueError("local_kernel must be >= 1")
        self.dim = dim
        self.local_kernel = local_kernel
        self.content_gate = content_gate
        self.out_gate = out_gate
        self.evidence_gate = evidence_gate
        self.uncertainty_gate = uncertainty_gate

        self.norm = make_norm(norm, dim, eps=eps)

        self.in_proj = nn.Linear(dim, dim, bias=False)
        if content_gate:
            self.gate_proj = nn.Linear(dim, dim, bias=False)
        if out_gate:
            self.out_gate_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.ssm = SelectiveSpectralSSM(
            dim, num_bases=num_bases,
            chunk_size=chunk_size, decay_max=decay_max, eps=eps)

        self.local_conv = nn.Conv1d(dim, dim, kernel_size=local_kernel,
                                    groups=dim, bias=False)
        with torch.no_grad():
            self.local_conv.weight.zero_()
            self.local_conv.weight[:, 0, -1] = 1.0   # identity at init

        self.out_scale = nn.Parameter(torch.ones(1))

        # 标量门参数：形状由结构决定（能量越高越开），只有强度可学。
        # 用标量而非 Linear，避免被上层 apply(_init_weights) 重置成常数值。
        # v1.2 闸门 opt-in：门关闭时不创建参数（与 analytic_params 口径一致）。
        if evidence_gate:
            self.evidence_gain = nn.Parameter(torch.tensor(1.0))
            self.evidence_pos_gain = nn.Parameter(torch.tensor(0.0))
            self.evidence_bias = nn.Parameter(torch.tensor(0.0))
        if uncertainty_gate:
            self.uncertainty_logit = nn.Parameter(torch.zeros(1))
        self.gate_eps = 1e-6
        # 融合流式热路径开关（仅影响流式 step）：开启且 ops.is_available() 时，
        # step 下沉到 C++/CUDA 融合算子（见 simplcore.ops.srf_stream）；否则走
        # 纯 PyTorch step。默认关，保持 A/B 对照与历史实验可复现。
        self.use_mixer_stream = bool(use_mixer_stream)

    # -- parallel forward -----------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        long = self.ssm(u)                       # (B, L, D) selective 路径
        local = self._local_conv_full(u)

        if self.evidence_gate:
            long = long * evidence_weights(u, self.evidence_gain,
                                           self.evidence_pos_gain,
                                           self.evidence_bias)
        if self.uncertainty_gate:
            long = long * uncertainty_weights(long, local,
                                              self.uncertainty_logit,
                                              self.gate_eps)

        y = long + local
        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        return residual + self.out_scale * out

    def _local_conv_full(self, u: torch.Tensor) -> torch.Tensor:
        K = self.local_kernel
        if K == 1:
            return u * self.local_conv.weight.view(1, 1, -1)
        up = F.pad(u.transpose(1, 2), (K - 1, 0))
        return F.conv1d(up, self.local_conv.weight, groups=self.dim).transpose(1, 2)

    # -- streaming ------------------------------------------------------------
    def init_state(self, batch: int, device, dtype) -> LayerStreamState:
        ssm_state = self.ssm.init_state(batch, device, dtype)
        buf = None
        if self.local_kernel > 1:
            buf = torch.zeros(batch, self.local_kernel - 1, self.dim,
                              device=device, dtype=dtype)
        # 证据门的 O(1) 运行能量和只在门开启时分配（与字节账目严格一致）。
        energy_sum = None
        if self.evidence_gate:
            energy_sum = torch.zeros(batch, 1, device=device, dtype=torch.float32)
        # 持久保存 ssm_state：step 复用它，scratch 缓冲因此跨 token 持续存在，
        # 流式循环每 token 不再重建 SSMState 与别名（见 step 注释）。
        return LayerStreamState(y1=ssm_state.h_re, y2=ssm_state.h_im,
                                conv_buf=buf, length=0, energy_sum=energy_sum,
                                ssm_state=ssm_state)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor,
             state: LayerStreamState) -> Tuple[torch.Tensor, LayerStreamState]:
        """一个 token，O(1) 状态与算力。

        注意状态复用 ``LayerStreamState`` 的两个字段：``y1``/``y2`` 存 SSM 的
        实/虚部（形状 (B,H,M,dh)，长度无关）。这样 ``SGCStreamState`` 的
        detach / num_bytes 不用为两种 mixer 各写一份。

        融合热路径：``use_mixer_stream`` 开启且扩展就绪时，下沉到 C++/CUDA 融合
        算子（见 ``simplcore.ops.srf_stream``）。融合内部恒 fp32 执行复数 SSM 递推；
        bf16/fp16 输入在 Python 层归一化后下沉、输出 cast 回原精度；扩展未构建则
        优雅回退下方纯 PyTorch 路径。
        """
        if self.use_mixer_stream:
            from ...ops import srf_stream
            if srf_stream.is_available():
                return srf_stream.srf_stream_step(self, x_t, state)

        squeeze = x_t.dim() == 2
        if squeeze:
            x_t = x_t.unsqueeze(1)
        h = self.norm(x_t)
        u = self.in_proj(h)
        if self.content_gate:
            u = u * torch.sigmoid(self.gate_proj(h))

        # 复用 init_state 持久保存的 ssm_state（首次若为空则惰性重建别名）：
        #   - 跨 token 持续存在，step 内部的 scratch 缓冲因此真正复用，流式循环
        #     每 token 不再分配 (B,M,D) 中间量、也不再重建 SSMState；
        #   - SelectiveSpectralSSM.step 原地 copy_ 更新 h_re/h_im，而它们就是
        #     state.y1/state.y2 的别名，无需把结果再拷回。
        if state.ssm_state is None:
            state.ssm_state = SSMState(h_re=state.y1, h_im=state.y2,
                                      length=state.length)
        long, state.ssm_state = self.ssm.step(u, state.ssm_state)
        # SSM 递推在 fp32 状态中进行，输出 long 为 fp32；cast 回计算精度再与
        # 局部路径 / 闸门混合，维持 I/O 与计算精度一致（状态本身保持 fp32 高精度）。
        long = long.to(u.dtype)

        local, state.conv_buf = depthwise_causal_conv(u, self.local_conv.weight,
                                                      state.conv_buf)

        if self.evidence_gate:
            w, state.energy_sum = evidence_weight_step(
                u, state.energy_sum, state.length,
                self.evidence_gain, self.evidence_pos_gain, self.evidence_bias)
            # 门控权重在 fp32 态里算，乘回前 cast 回计算精度（与 SGC.step 同约束：
            # 递推态 fp32、I/O 与计算 bf16，否则低精度路径被提升回 fp32）
            long = long * w.to(long.dtype)
        if self.uncertainty_gate:
            long = long * uncertainty_weights(long, local,
                                              self.uncertainty_logit,
                                              self.gate_eps).to(long.dtype)

        y = long + local
        out = self.out_proj(y)
        if self.out_gate:
            out = out * torch.sigmoid(self.out_gate_proj(h))
        state.length += 1
        out = x_t + self.out_scale * out
        return (out.squeeze(1) if squeeze else out), state

    # -- introspection --------------------------------------------------------
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, bases={self.ssm.num_bases}, "
                f"chunk={self.ssm.chunk_size}, "
                f"local_kernel={self.local_kernel}, "
                f"content_gate={self.content_gate}, out_gate={self.out_gate}, "
                f"evidence_gate={self.evidence_gate}, "
                f"uncertainty_gate={self.uncertainty_gate}")

    def streaming_state_bytes(self, batch: int, dtype=torch.float32) -> int:
        itemsize = torch.empty(0, dtype=dtype).element_size()
        # SSM 递推状态恒按高精度计字节（低精度计算时 fp32）。
        rd_itemsize = torch.empty(0, dtype=recurrent_dtype(dtype)).element_size()
        n = 2 * self.ssm.num_bases * self.dim * batch * rd_itemsize
        n += max(self.local_kernel - 1, 0) * self.dim * batch * itemsize
        extra = batch * 4 if self.evidence_gate else 0   # energy 恒 fp32
        return n + extra
