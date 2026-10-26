# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 核心块 (SimplCoreBlock) - PyTorch 实现.

每个 SimplCoreBlock 顺序包含四个自研子组件, 每个自带 LayerNorm 与残差连接:

  Token ──► SGC ──► DMC ──► CCR ──► FFN ──► Next
  Input   O(nlogn)  O(1)mem  Adaptive  GELU   Layer

  1. SGC (Spectral Global Convolution, 谱全局卷积; 自研二次开发版本):
     用 FFT 计算拉普拉斯算子, 在波场中按扩散方程演化, 复杂度 O(n log n).
  2. DMC (Dynamic Memory Channel, 动态记忆通道):
     内容寻址记忆库 (B, C, S, D), 多头读取 + NTM 风格门控写入.
  3. CCR (Conditional Compute Router, 条件计算路由):
     每个 token 动态选择不同计算深度的路径, top-k 加权融合.
  4. FFN (Feed-Forward Network):
     标准 2 层 MLP + GELU.

为避免依赖自研引擎版本 (simplcore_modules.py), 本文件内联定义全部子组件,
直接使用 torch / torch.nn 实现.

接口:
  block(x, memory_state=None) -> (output, new_memory)
    x:            (B, L, D)
    memory_state: (B, C, S, D) 或 None
    output:       (B, L, D)
    new_memory:   (B, C, S, D)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F

try:  # 包内导入 (推荐)
    from .config import SimplCoreConfig
except ImportError:  # 直接以脚本运行时, 补全包路径后用绝对导入回退
    import os
    import sys

    _ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from simplcore.architecture.config import SimplCoreConfig  # type: ignore


# ======================================================================
# 1. 谱混合层 (旧实现, 仅保留兼容; 新实现见 spectral_global_conv.SpectralGlobalConv)
# ======================================================================
class _LegacySpectralMixer(nn.Module):
    """谱混合层 (旧实现; 自研二次开发版为 SpectralGlobalConv).

    用 FFT 全局卷积实现全局信息交互, 单层复杂度 O(n log n) (旧实现):
      1. 投影输入到 W 个波场通道.
      2. 用 FFT 做带通滤波 (围绕共振频率 omega 的高斯窗).
      3. 按扩散方程 F <- F + dt*(alpha*lap - beta*F + gamma*I) 演化 T 步.
      4. 投影回特征维度.

    Args:
        dim: 特征维度 D.
        num_waves: 波场通道数 W.
        damping: 衰减系数初值.
        num_steps: 场演化步数 T.
        dt: 离散化时间步长.
        sigma: 带通高斯窗带宽.
        eps: LayerNorm 数值稳定常数.
    """

    def __init__(
        self,
        dim: int,
        num_waves: int = 8,
        damping: float = 0.1,
        num_steps: int = 3,
        dt: float = 0.1,
        sigma: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_waves > 0
        self.dim = dim
        self.num_waves = num_waves
        self.num_steps = num_steps
        self.dt = dt
        self.sigma = sigma

        # 投影: D <-> W
        self.in_proj = nn.Linear(dim, num_waves, bias=True)
        self.out_proj = nn.Linear(num_waves, dim, bias=True)

        # 可学习物理参数
        self.omega_raw = nn.Parameter(torch.randn(num_waves) * 0.3)
        # softplus(log(expm1(d))) == d, 使初值 alpha == damping
        init_alpha = math.log(math.expm1(max(damping, 1e-6)))
        self.alpha_raw = nn.Parameter(torch.full((num_waves,), init_alpha))
        self.gamma = nn.Parameter(torch.ones(num_waves) * 0.5)
        self.beta = nn.Parameter(torch.tensor([damping]))
        self.out_scale = nn.Parameter(torch.ones(1))

        self.norm = nn.LayerNorm(dim, eps=eps)

    # ---- 物理参数 (非负) ----
    @property
    def omega(self) -> torch.Tensor:
        return F.softplus(self.omega_raw)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.alpha_raw)

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """沿序列维度用 FFT 计算一维拉普拉斯算子 ∇²F (numpy 前向, 非梯度).

        field: (B, L, W) -> (B, L, W)
        """
        fld = field.data if isinstance(field, torch.Tensor) else np.asarray(field)
        L = fld.shape[1]
        freqs = np.fft.fftfreq(L, d=1.0) * (2.0 * math.pi)
        k_sq = freqs ** 2  # (L,)
        field_fft = np.fft.fft(fld, axis=1)               # 复数 (B, L, W)
        lap_fft = field_fft * (-k_sq).reshape(1, -1, 1)   # ∇² 在频域 = -k²
        lap = np.fft.ifft(lap_fft, axis=1).real           # (B, L, W)
        return torch.tensor(lap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """谱混合前向 (历史实现; numpy FFT 前向, 返回非梯度张量).

        注: 自研后端无 FFT 算子, 频域计算以 numpy 完成; 真正可训练实现见
        spectral_global_conv.SpectralGlobalConv (SGC)。
        """
        residual = x
        h = self.norm(x)                                  # (B, L, D)
        B, L, _ = h.shape

        I_field = self.in_proj(h).data                    # (B, L, W) numpy

        # 带通滤波: 围绕 omega 的高斯窗 (numpy)
        freqs = np.fft.fftfreq(L, d=1.0) * (2.0 * math.pi)  # (L,)
        freqs_abs = freqs.reshape(-1, 1)                  # (L, 1)
        omega = self.omega.data.reshape(1, -1)            # (1, W)
        band = np.exp(
            -((freqs_abs - omega) ** 2) / (2.0 * self.sigma ** 2)
        )                                                 # (L, W)

        F_field = np.fft.ifft(
            np.fft.fft(I_field, axis=1) * band[None, :, :], axis=1
        ).real                                            # (B, L, W)

        # 扩散演化
        alpha = self.alpha.data.reshape(1, 1, -1)         # (1, 1, W)
        gamma = self.gamma.data.reshape(1, 1, -1)         # (1, 1, W)
        beta = float(self.beta.data.flatten()[0])         # 标量
        for _ in range(self.num_steps):
            lap = self._laplacian(torch.tensor(F_field)).data
            F_field = F_field + self.dt * (
                alpha * lap - beta * F_field + gamma * I_field
            )

        out = self.out_proj(torch.tensor(F_field)).data    # (B, L, D)
        final = (residual.data + (self.out_scale.data * out)).astype(np.float32)
        return torch.tensor(final)

    def resonance_complexity_analysis(
        self, seq_len: int, batch_size: int = 1
    ) -> dict:
        """与本层计算复杂度并与标准注意力对比."""
        n, d, w = seq_len, self.dim, self.num_waves
        T = self.num_steps
        fft_cost = 5.0 * n * math.log2(max(n, 2))
        proj_cost = 2.0 * n * d * w
        sgc_flops = (proj_cost + T * (fft_cost + n * w)) * batch_size
        attn_flops = 2.0 * n * n * d * batch_size
        return {
            "seq_len": n,
            "sgc_flops": sgc_flops,
            "attn_flops": attn_flops,
            "speedup_vs_attention": attn_flops / max(sgc_flops, 1.0),
            "sgc_big_o": "O(n log n)",
            "attn_big_o": "O(n^2)",
        }


# ======================================================================
# 2. DMC - 动态记忆通道 (内容寻址读写)
# ======================================================================
class _DynamicMemoryChannel(nn.Module):
    """动态记忆通道 (Dynamic Memory Channel).

    维护记忆库 (B, C, S, D):
      - C: 通道数, S: 每 channel 的 slot 数, D: 特征维度.
      - 读: 多头内容寻址 (query -> memory slots 的注意力).
      - 写: NTM 风格门控 (先擦除 erase 再添加 add).
    相对序列长度 n 的复杂度为 O(1) (记忆大小固定).

    Args:
        dim: 特征维度 D.
        num_channels: 通道数 C.
        channel_size: 每 channel 的 slot 数 S.
        num_heads: 读取头数.
        eps: LayerNorm 数值稳定常数.
    """

    def __init__(
        self,
        dim: int,
        num_channels: int = 64,
        channel_size: int = 128,
        num_heads: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_channels > 0 and channel_size > 0
        assert num_heads > 0 and dim % num_heads == 0
        self.dim = dim
        self.num_channels = num_channels
        self.channel_size = channel_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.write_value_proj = nn.Linear(dim, dim, bias=True)
        self.output_proj = nn.Linear(dim, dim, bias=True)
        self.write_gate = nn.Linear(dim, num_channels, bias=True)

        self.channel_weights = nn.Parameter(torch.ones(num_channels))
        self.norm = nn.LayerNorm(dim, eps=eps)

    def _init_memory(self, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
        """初始化记忆库 (小随机, 与 ref 同设备/ dtype)."""
        return torch.randn(
            batch_size, self.num_channels, self.channel_size, self.dim,
            device=ref.device, dtype=ref.dtype,
        ) * 0.02

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向: 对整段序列做读取 + 写入, 返回 (输出, 更新后的记忆).

        Args:
            x: (B, L, D)
            memory_state: (B, C, S, D) 或 None (None 时初始化).

        Returns:
            output:     (B, L, D)
            new_memory: (B, C, S, D)
        """
        residual = x
        B, L, D = x.shape
        H, hd = self.num_heads, self.head_dim

        if memory_state is None:
            memory = self._init_memory(B, x)
        else:
            memory = memory_state

        # ---------- 读: 多头内容寻址 (向量化, 一次处理整段序列) ----------
        q = self.query_proj(x).view(B, L, H, hd)              # (B, L, H, hd)
        k = self.key_proj(memory).view(
            B, self.num_channels, self.channel_size, H, hd
        )                                                     # (B, C, S, H, hd)
        scores = torch.einsum("blhd,bcshd->blcs", q, k) / math.sqrt(hd)
        weights = torch.softmax(scores, dim=-1)               # (B, L, C, S)

        mem_v = memory.view(
            B, self.num_channels, self.channel_size, H, hd
        )                                                     # (B, C, S, H, hd)
        readout = torch.einsum("blcs,bcshd->blchd", weights, mem_v)
        readout = readout.reshape(B, L, self.num_channels, D)  # (B, L, C, D)

        cw = torch.softmax(self.channel_weights, dim=0)       # (C,)
        read_agg = torch.einsum("c,blcd->bld", cw, readout)   # (B, L, D)
        out = self.output_proj(read_agg)                      # (B, L, D)
        output = residual + self.norm(out)

        # ---------- 写: NTM 风格门控 (erase + add) ----------
        wk = self.query_proj(x).view(B, L, H, hd)             # 复用 query 投影作写键
        mem_k = self.key_proj(memory).view(
            B, self.num_channels, self.channel_size, H, hd
        )
        w_scores = torch.einsum("blhd,bcshd->blcs", wk, mem_k) / math.sqrt(hd)
        w_weights = torch.softmax(w_scores, dim=-1)           # (B, L, C, S)
        gate = torch.sigmoid(self.write_gate(x))              # (B, L, C)
        wv = self.write_value_proj(x)                         # (B, L, D)

        # erase: (B, C, S) = sum_L gate * w_weights, 截断到 [0,1]
        erase = (gate.unsqueeze(-1) * w_weights).sum(dim=1)   # (B, C, S)
        erase = erase.clamp(0.0, 1.0)
        # add: (B, C, S, D) = sum_L gate * w_weights * wv, 按序列长度归一
        add = torch.einsum("blc,blcs,bld->bcsd", gate, w_weights, wv) / max(L, 1)

        new_memory = memory * (1.0 - erase.unsqueeze(-1)) + add
        new_memory = new_memory.clamp(-10.0, 10.0)            # 稳定记忆幅值

        return output, new_memory


# ======================================================================
# 3. CCR - 条件计算路由器 (动态深度路径)
# ======================================================================
class _LegacyConditionalRouter(nn.Module):
    """条件计算路由器 (Conditional Compute Router).

    每个 token 被路由到 P 条计算路径, 各路径对应不同计算深度
    (从 min_depth 到 max_depth 均匀分布). 取 top-k 路径加权融合.

    Args:
        dim: 特征维度.
        num_paths: 候选路径数 P.
        min_depth: 最小计算深度.
        max_depth: 最大计算深度.
        topk: top-k 路由的 k.
        dropout: 路由 logits dropout.
        eps: LayerNorm 数值稳定常数.
    """

    def __init__(
        self,
        dim: int,
        num_paths: int = 4,
        min_depth: int = 2,
        max_depth: int = 8,
        topk: int = 2,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_paths > 0
        assert 1 <= min_depth <= max_depth
        assert 1 <= topk <= num_paths
        self.dim = dim
        self.num_paths = num_paths
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.topk = topk
        self.dropout = dropout

        self.router = nn.Linear(dim, num_paths, bias=True)
        self.transform = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.out_norm = nn.LayerNorm(dim, eps=eps)
        self.temperature = nn.Parameter(torch.ones(1))

        self.path_depths: list = self._compute_path_depths()

    def _compute_path_depths(self) -> list:
        """将路径均匀映射到 [min_depth, max_depth]."""
        if self.num_paths == 1:
            return [self.min_depth]
        depths = []
        for p in range(self.num_paths):
            d = self.min_depth + round(
                p * (self.max_depth - self.min_depth) / (self.num_paths - 1)
            )
            depths.append(int(d))
        return depths

    def route(self, x: torch.Tensor) -> torch.Tensor:
        """计算每个 token 对各路径的路由概率 (B, L, P)."""
        logits = self.router(x)
        if self.training and self.dropout > 0:
            logits = F.dropout(logits, p=self.dropout, training=True)
        t = self.temperature.clamp(min=1e-3)
        return torch.softmax(logits / t, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)                                     # (B, L, D)
        B, L, D = h.shape

        probs = self.route(h)                                # (B, L, P)

        # 计算各深度状态 (深度 0..max_depth), 复用同一 transform
        max_d = max(self.path_depths)
        depth_states = {0: h}
        current = h
        for d in range(1, max_d + 1):
            current = current + F.gelu(self.transform(current))
            depth_states[d] = current

        path_states = torch.stack(
            [depth_states[d] for d in self.path_depths], dim=2
        )                                                    # (B, L, P, D)

        if self.topk >= self.num_paths:
            out = torch.einsum("blp,blpd->bld", probs, path_states)
        else:
            topk_probs, topk_idx = probs.topk(self.topk, dim=-1)   # (B, L, k)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)     # (B, L, k, D)
            topk_states = path_states.gather(2, idx)               # (B, L, k, D)
            out = torch.einsum("blk,blkd->bld", topk_probs, topk_states)

        out = self.out_norm(out)
        return residual + out

    def expected_depth(self, x: torch.Tensor) -> torch.Tensor:
        """每个 token 的期望计算深度 (B, L)."""
        probs = self.route(x)
        depths = torch.tensor(
            self.path_depths, dtype=x.dtype, device=x.device
        )
        return (probs * depths).sum(dim=-1)


# ======================================================================
# 4. FFN - 前馈网络 (2 层 MLP + GELU)
# ======================================================================
class _FeedForward(nn.Module):
    """标准前馈网络: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, dim: int, intermediate: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, intermediate)
        self.fc2 = nn.Linear(intermediate, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ======================================================================
# SimplCoreBlock - 核心块
# ======================================================================
class SimplCoreBlock(nn.Module):
    """SimplCore 核心块: SGC -> DMC -> CCR -> FFN.

    每个子组件自带 LayerNorm 与残差连接; DMC 额外维护并返回记忆状态.

    Args:
        config: SimplCoreConfig 配置.
        gradient_checkpointing: 是否对 FFN 子层启用梯度检查点 (省显存).
    """

    def __init__(
        self,
        config: Optional["SimplCoreConfig"] = None,
        gradient_checkpointing: bool = False,
        *,
        dim: Optional[int] = None,
        num_heads: int = 4,
        num_waves: int = 8,
        num_channels: int = 64,
        channel_size: int = 128,
        num_paths: int = 4,
        min_depth: int = 2,
        max_depth: int = 8,
        topk: int = 2,
        ffn_intermediate_ratio: float = 4.0,
        sgc_num_steps: int = 3,
        sgc_damping: float = 0.1,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if config is None:
            if dim is None:
                raise ValueError("必须提供 config 或 dim。")
            config = SimplCoreConfig(
                vocab_size=1000,
                hidden_size=dim,
                num_layers=1,
                num_waves=num_waves,
                num_channels=num_channels,
                channel_size=channel_size,
                dmc_num_heads=num_heads,
                num_paths=num_paths,
                min_depth=min_depth,
                max_depth=max_depth,
                ccr_topk=topk,
                ffn_intermediate_ratio=ffn_intermediate_ratio,
                max_sequence_length=64,
                num_modalities=3,
                modality_dims=(dim, dim, dim),
                dropout=dropout,
                sgc_num_steps=sgc_num_steps,
                sgc_damping=sgc_damping,
                rms_norm_eps=eps,
                use_rollout=False,
            )
        self.config = config
        dim = config.hidden_size
        eps = config.rms_norm_eps

        self.sgc = _LegacySpectralMixer(  # 旧实现; 新版可替换为 SpectralGlobalConv
            dim=dim,
            num_waves=config.num_waves,
            damping=config.sgc_damping,
            num_steps=config.sgc_num_steps,
            dt=config.sgc_dt,
            sigma=config.sgc_sigma,
            eps=eps,
        )
        self.dmc = _DynamicMemoryChannel(
            dim=dim,
            num_channels=config.num_channels,
            channel_size=config.channel_size,
            num_heads=config.dmc_num_heads,
            eps=eps,
        )
        self.ccr = _LegacyConditionalRouter(
            dim=dim,
            num_paths=config.num_paths,
            min_depth=config.min_depth,
            max_depth=config.max_depth,
            topk=config.ccr_topk,
            dropout=config.ccr_dropout,
            eps=eps,
        )

        self.ffn_norm = nn.LayerNorm(dim, eps=eps)
        self.ffn = _FeedForward(
            dim=dim,
            intermediate=int(round(config.ffn_intermediate_ratio * dim)),
            dropout=config.dropout,
        )
        self.ffn_dropout = nn.Dropout(config.dropout)

        self.gradient_checkpointing = gradient_checkpointing

    def _ffn_forward(self, h: torch.Tensor) -> torch.Tensor:
        """FFN 子层 (含 norm), 供梯度检查点调用."""
        return self.ffn(self.ffn_norm(h))

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: (B, L, D) 输入隐状态.
            memory_state: (B, C, S, D) 或 None, DMC 记忆状态.

        Returns:
            output: (B, L, D) 输出隐状态.
            new_memory: (B, C, S, D) 更新后的记忆状态.
        """
        # 1. 谱混合层 (旧实现; 内部 LayerNorm + 残差)
        h = self.sgc(x)
        # 2. DMC (内部 LayerNorm + 残差, 返回新记忆)
        h, new_memory = self.dmc(h, memory_state)
        # 3. CCR (内部 LayerNorm + 残差)
        h = self.ccr(h)
        # 4. FFN (LayerNorm + 残差)
        if self.gradient_checkpointing and self.training:
            try:
                ffn_out = torch.utils.checkpoint.checkpoint(
                    self._ffn_forward, h, use_reentrant=False
                )
            except TypeError:  # 旧版 PyTorch 无 use_reentrant 参数
                ffn_out = torch.utils.checkpoint.checkpoint(self._ffn_forward, h)
            h = h + self.ffn_dropout(ffn_out)
        else:
            h = h + self.ffn_dropout(self.ffn(self.ffn_norm(h)))
        return h, new_memory

    def count_parameters(self) -> int:
        """统计本块参数量 (去重)."""
        total = 0
        seen = set()
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("SimplCoreBlock 测试")
    print("=" * 60)

    cfg = SimplCoreConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_waves=4,
        num_channels=4,
        channel_size=8,
        num_paths=4,
        min_depth=1,
        max_depth=4,
        ffn_intermediate_ratio=2.0,
        max_sequence_length=64,
        num_modalities=3,
        modality_dims=(64, 64, 64),
        tie_word_embeddings=True,
        dropout=0.0,
        sgc_num_steps=2,
        sgc_damping=0.1,
        dmc_num_heads=4,
        ccr_topk=2,
        ccr_dropout=0.0,
        use_rollout=False,
    )

    block = SimplCoreBlock(cfg, gradient_checkpointing=False)
    print(f"单块参数量: {block.count_parameters():,}")

    # ---- Test 1: 基本前向 (无记忆) ----
    print("\n[Test 1] 基本前向 (memory_state=None)")
    x = torch.randn(2, 16, 64, requires_grad=True)
    out, mem = block(x, None)
    print(f"  input={tuple(x.shape)} -> output={tuple(out.shape)}, "
          f"memory={tuple(mem.shape)}")
    assert out.shape == (2, 16, 64)
    assert mem.shape == (2, 4, 8, 64)
    print("  [PASS]")

    # ---- Test 2: 记忆状态传递 ----
    print("\n[Test 2] 记忆状态传递")
    out2, mem2 = block(out.detach(), mem)
    print(f"  传入 memory={tuple(mem.shape)} -> new memory={tuple(mem2.shape)}")
    assert mem2.shape == mem.shape
    print("  [PASS]")

    # ---- Test 3: 梯度流 ----
    print("\n[Test 3] 梯度流验证")
    loss = out.pow(2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters()
    )
    assert has_grad, "块内无梯度!"
    n_with_grad = sum(
        1 for p in block.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  有梯度的参数: {n_with_grad}/{len(list(block.parameters()))}")
    print("  [PASS]")

    # ---- Test 4: 复杂度分析 ----
    print("\n[Test 4] 谱混合层复杂度分析")
    analysis = block.rfp.resonance_complexity_analysis(seq_len=4096)
    print(f"  seq=4096: SpectralMixer={analysis['spectral_mixer_flops']:.3g} FLOPs, "
          f"Attn={analysis['attn_flops']:.3g} FLOPs, "
          f"加速比={analysis['speedup_vs_attention']:.1f}x")
    assert analysis["speedup_vs_attention"] > 1.0
    print("  [PASS]")

    # ---- Test 5: 期望深度 ----
    print("\n[Test 5] CCR 期望深度")
    block.eval()
    ed = block.tar.expected_depth(x.detach())
    print(f"  expected_depth shape={tuple(ed.shape)}, mean={ed.mean():.3f}")
    assert ed.shape == (2, 16)
    print("  [PASS]")

    # ---- Test 6: 梯度检查点模式 ----
    print("\n[Test 6] 梯度检查点模式")
    block_ckpt = SimplCoreBlock(cfg, gradient_checkpointing=True).train()
    x2 = torch.randn(2, 16, 64, requires_grad=True)
    out_ckpt, _ = block_ckpt(x2, None)
    out_ckpt.pow(2).mean().backward()
    has_grad_ckpt = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in block_ckpt.parameters()
    )
    assert has_grad_ckpt, "检查点模式下无梯度!"
    print("  [PASS]")

    print("\n" + "=" * 60)
    print("SimplCoreBlock smoke test PASSED")
    print("=" * 60)
