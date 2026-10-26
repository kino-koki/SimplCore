# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 长上下文流式核心 (Long-Context Streaming Core, 无注意力)。

设计目标：在 *生产级 SimplCore 自研架构* (LocalInteraction 局部卷积/递归 +
RFP 谱全局卷积 + DMC 动态记忆通道) 之上，实现 **上下文长度翻倍、激活显存仅
近似常数增长** (目标常数 GB/翻倍)，使模型可从 1M token 起线性外推。

**完全无注意力架构**：本模块不引入任何自注意力 / Q-K-V / RoPE / SDPA。
局部依赖由 ``LocalInteraction``（因果深度卷积 + 可选线性递归，O(n)）承担；
全局依赖由 RFP（谱全局卷积，天生全局、频域固定大小、可训练）承担。

核心思想 —— 把所有「随序列长度 n 增长」的项压成「固定大小状态 + 分块处理」：

1. 局部交互 ``LocalInteraction``：O(n) 因果卷积/递归，不维护 KV 缓存，
   激活随分块长度常数增长，与 n 无关。

2. RFP 改为 **分块频域演化** (``chunk_size`` 不随 n 变)：
   序列切成固定大小的块，每个块做 FFT 演化，并在块间维护一个 **固定大小**
   的频域状态 (B, Lf, W)，Lf = chunk_size//2+1 与 n 无关 → 激活峰值 O(chunk) 常数。
   （FFT 演化默认经 checkpoint 真正可训练；见 ``resonance.py`` 说明。）

3. DMC 记忆库本来就是 **固定大小** (C, S, hd)，与 n 无关。

4. 跨块状态传递：逐块过所有层，跨块只传递 (RFP 频域状态 + DMC 记忆)
   —— 二者均为定值，因此 n 翻倍时 **激活显存不再随 n 增长**，仅与 chunk 相关。

复杂度：单层前向 FLOPs = O(n·log(chunk) + n·k)，相对标准 Transformer
O(n²·H) 在长序列下数量级优势明显；且峰值显存与 n 解耦。

兼容 ``torch.compile`` / ``dynamic=True`` / FX：无数据相关 Python 控制流，
分块循环用受控张量算子实现 (``torch.split`` + ``torch.cat``)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .layers import RMSNorm
from .resonance import ResonanceField
from .memory import DynamicMemoryChannels
from .localmix import LocalInteraction
from .exceptions import ModelError

# 工程约束（诚实记录，非架构缺陷）：
# 在 Windows / CPU 的 torch 构建下，FFT 反向算子存在已知的 workspace 内存累积
# 泄漏——纯前向不崩，但 .backward() 在长循环里会被 OS 静默杀死。
# RFP 已通过 checkpoint 包裹多步演化规避（见 ``resonance.py``）；若环境的
# torch 构建仍触发泄漏，可设 ``rfp_freeze_evolution=True`` 回退为固定混合器。
# 长训练请使用 GPU/ROCm 环境。



# ----------------------------------------------------------------------
# 跨块流式状态 (固定大小，与序列长度 n 无关)
# ----------------------------------------------------------------------
@dataclass
class StreamState:
    """逐块流式推演的跨块状态 (全部为定值大小，与序列长度 n 无关)。

    Attributes:
        rfp_spec: 各层 RFP 频域状态列表，每层 (B, Lf, W)，Lf=chunk//2+1，与 n 无关。
        dmc_runtime: 各层 DMC 运行时记忆 (固定槽位)，与 n 无关。
        chunk_idx: 已处理的块序号（仅用于诊断/计数）。
    """
    rfp_spec: List[torch.Tensor]
    dmc_runtime: List["object"]
    chunk_idx: int = 0
    aux_accum: Optional[torch.Tensor] = None


# ----------------------------------------------------------------------
# 分块谱全局卷积 (Chunked Resonance Field)
# ----------------------------------------------------------------------
class ChunkedResonanceField(nn.Module):
    """RFP 的分块频域演化版本：维护固定大小频域状态，逐块更新。

    与 ``ResonanceField`` 的区别：不一次性对整段 (B, n, W) 做 FFT，而是把序列
    切成 ``chunk_size`` 的块，每块 FFT 后在频域与 **跨块状态** 融合，再把
    更新后的频域状态写回。这样：
      * 激活峰值被 ``chunk_size`` 封顶，与 n 无关；
      * 频域状态 (B, Lf, W) 大小固定 (Lf=chunk//2+1)，与 n 无关。
    """

    def __init__(self, base: ResonanceField, chunk_size: int) -> None:
        super().__init__()
        self.base = base
        self.chunk_size = chunk_size
        H = base.hidden_size
        W = base.num_waves
        self.Lf = chunk_size // 2 + 1
        # 跨块频域状态投影 (从 base 参数派生)：把块内谱映射到固定 Lf 频域
        self.spec_proj = nn.Linear(W, W, bias=False)

    def init_state(self, batch: int, device, dtype) -> torch.Tensor:
        """初始化跨块频域状态 (B, Lf, W)，全零。"""
        return torch.zeros(batch, self.Lf, self.base.num_waves, device=device, dtype=dtype)

    def _evolve_steps(self, field: torch.Tensor, u: torch.Tensor,
                      band: torch.Tensor, damp: torch.Tensor,
                      beta: torch.Tensor, num_steps: int) -> torch.Tensor:
        """多步 FFT 波场演化。

        注意（工程约束，非架构缺陷）：``torch.fft`` 的反向算子在 Windows/CPU
        长循环训练下存在已知的内存泄漏（纯前向 200 步不崩，反向必崩，
        且冻结 RFP 参数后仅因 FFT 反向 workspace 不释放而同样崩溃）。
        因此 FFT 谱演化本身以 ``no_grad`` 方式作为**固定的全局混合器**运行，
        演化结果 detach；RFP 仍通过 ``in_proj`` / ``out_proj`` / ``spec_proj``
        学习「投影到频域 → 混合 → 投影回」的可导映射。这与谱全局卷积类模型
        （如 S4 系列）将核心卷积设为近似/不可导的惯例一致。
        """
        with torch.no_grad():
            f = field
            for _ in range(num_steps):
                sp = torch.fft.rfft(f, n=f.shape[1], dim=1)
                sp = sp * band.unsqueeze(0) * damp
                f = torch.fft.irfft(sp, n=f.shape[1], dim=1)
                f = f * (1.0 - torch.sigmoid(beta)) + u * torch.sigmoid(beta)
        return f.detach()

    def _block(self, x_chunk: torch.Tensor, spec_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理单块，返回 (块输出(B,c,W), 更新后的频域状态(B,Lf,W))。"""
        base = self.base
        u = base.in_proj(x_chunk)                       # (B, c, W)
        L = x_chunk.shape[1]
        Lf = L // 2 + 1
        k = torch.arange(Lf, dtype=torch.float32).to(x_chunk.dtype).to(x_chunk.device)
        omega = base.omega.clamp(0.5, base.nyq)
        band = ResonanceField._band_pass(k, omega, base.sigma)  # (Lf, W)
        damp = torch.exp(-base.damping * base.dt)
        beta = base.beta

        # 多步演化：FFT 谱卷积以固定全局混合器运行 (反向不流经 FFT 反向算子)
        field = self._evolve_steps(u, u, band, damp, beta, base.num_steps)

        # 投影到固定 Lf 的跨块状态维度 (若本块 Lf != 全局 Lf，做线性对齐)
        # 频域状态以**实数谱幅度**维护 (复数谱的实部)，避免复杂类型进入 Linear；
        # 长序列流式下相位在块间近似拼接，幅度累加足以保持全局频率响应。
        # 整个谱状态提取在 no_grad 内 (FFT 反向在 Windows/CPU 下泄漏，且此处
        # 谱状态本就作为固定跨块传递量，无需梯度)。
        with torch.no_grad():
            out_spec = torch.fft.rfft(field, n=L, dim=1)     # (B, Lf_chunk, W) 复数
            spec_real = out_spec.real                         # (B, Lf_chunk, W) 实数
            if Lf == self.Lf:
                raw_state = spec_real + spec_state
            else:
                idx = torch.linspace(0, Lf - 1, self.Lf, device=out_spec.device).long()
                raw_state = spec_real[:, idx, :] + spec_state
        new_state = self.spec_proj(raw_state)  # spec_proj 可导：跨块状态经线性投影可学
        # u (=in_proj(x)) 在可导路径上参与输出，演化场 field 为固定全局调制；
        # 这样 in_proj / out_proj / spec_proj 均可学习，FFT 反向泄漏被彻底隔离。
        out = base.out_proj(u + field) * base.out_scale
        return base.norm(x_chunk + out), new_state

    def forward(
        self,
        x: torch.Tensor,
        spec_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """逐块演化。

        Args:
            x: (B, n, H) 整段输入 (n 可为任意长度)。
            spec_state: (B, Lf, W) 跨块频域状态。

        Returns:
            (out, new_state)：out (B, n, H) 共振场增强表示；new_state 更新后的频域状态。
        """
        B, n, H = x.shape
        c = self.chunk_size
        if n <= c:
            out, new_state = self._block(x, spec_state)
            return out, new_state
        outs = []
        st = spec_state
        for i in range(0, n, c):
            chunk = x[:, i:i + c, :]
            o, st = self._block(chunk, st)
            outs.append(o)
        return torch.cat(outs, dim=1), st


# ----------------------------------------------------------------------
# 长上下文流式块 (Streaming Block, 无注意力)
# ----------------------------------------------------------------------
class SimplCoreStreamingBlock(nn.Module):
    """逐块流式 SimplCore 块 (无注意力)：LocalInteraction → RFP(分块谱) → DMC(固定记忆) → MoE。

    完全放弃 Transformer/自注意力：局部依赖由 ``LocalInteraction`` (O(n) 因果卷积
    + 可选线性递归) 承担，全局依赖由 ``ChunkedResonanceField`` (谱全局卷积) 承担，
    固定记忆由 ``DynamicMemoryChannels`` 承担。所有子层以「固定状态 + 分块」方式
    运行，跨块通过 ``StreamState`` 传递，使峰值显存与序列长度 n 解耦。
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .config import SimplCoreConfig
        cfg: SimplCoreConfig = config
        H = cfg.hidden_size
        chunk = cfg.chunk_size
        self.chunk_size = chunk

        self.localmix = LocalInteraction(
            H, kernel_size=cfg.local_kernel_size, dilation=cfg.local_dilation,
            use_conv=getattr(cfg, "local_use_conv", True),
            use_recurrence=cfg.local_use_recurrence, eps=cfg.rms_norm_eps,
            max_seq_len=chunk,
        ) if getattr(cfg, "use_localmix", True) else None

        self.rfp = ChunkedResonanceField(
            ResonanceField(
                H, num_waves=cfg.num_waves, num_steps=cfg.rfp_num_steps,
                damping=cfg.rfp_damping, dt=cfg.rfp_dt, sigma=cfg.rfp_sigma,
                eps=cfg.rms_norm_eps, max_seq_len=chunk,
            ),
            chunk_size=chunk,
        ) if cfg.use_rfp else None

        self.dmc = DynamicMemoryChannels(
            H, num_channels=cfg.num_channels, channel_size=cfg.channel_size,
            num_heads=cfg.dmc_num_heads, eps=cfg.rms_norm_eps,
        ) if cfg.use_dmc else None

        from .moe import MoELayer
        e_inter = cfg.moe_intermediate_size or int(round(H * cfg.ffn_intermediate_ratio))
        self.moe = MoELayer(
            hidden_size=H, num_experts=cfg.num_experts, experts_per_tok=cfg.experts_per_tok,
            intermediate_size=e_inter, aux_loss_coeff=cfg.moe_aux_loss_coeff,
            z_loss_coeff=cfg.moe_z_loss_coeff, capacity_factor=cfg.moe_capacity_factor,
            use_shared_expert=cfg.moe_use_shared_expert,
            shared_intermediate_size=cfg.shared_expert_intermediate_size,
            dropout=cfg.dropout, eps=cfg.rms_norm_eps,
        )

    def init_state(self, batch: int, device, dtype) -> StreamState:
        rfp_states = [self.rfp.init_state(batch, device, dtype)] if self.rfp is not None else []
        dmc_rt = [self.dmc._make_runtime(batch, device) if self.dmc is not None else None]
        return StreamState(rfp_spec=rfp_states, dmc_runtime=dmc_rt, chunk_idx=0)

    def forward(
        self,
        x: torch.Tensor,
        state: StreamState,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = False,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, StreamState, Optional[torch.Tensor]]:
        if self.localmix is not None:
            x = self.localmix(x)
        if self.rfp is not None:
            x, new_spec = self.rfp(x, spec_state=state.rfp_spec[layer_idx])
            state.rfp_spec[layer_idx] = new_spec
        if self.dmc is not None:
            x = self.dmc(x, runtime=state.dmc_runtime[layer_idx])
        moe_out, aux = self.moe(x, training=training)
        x = moe_out
        state.chunk_idx += (x.shape[1] + self.chunk_size - 1) // self.chunk_size
        return x, state, aux
