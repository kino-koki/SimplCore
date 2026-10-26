# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""L1 Gated DeltaNet 线性循环层 (TriMemory 中程记忆)。

核心递推 (每头状态 S ∈ R^{dv×dk})::

    S ← diag(γ_t) · S                    (逐值维衰减门, γ∈(0,1)^dv)
    e_t = v_t − S @ k̂_t                   (delta 规则: 先读出旧关联, 取残差)
    S ← S + β_t · e_t ⊗ k̂_t               (β∈(0,1) 写入门, k̂ 为 L2 归一化键)
    o_t = S @ q_t                         (检索)

性质：
  * **单步显存/算力 O(dk·dv)，与已读上下文长度无关** —— 这是"单步 O(1)"的来源；
  * 训练态完全可微 (纯 PyTorch 循环, 向量化于 batch×heads 维度)；
  * 生产环境应替换为 Triton chunked-scan 内核 (WY 表示, 参考 FLA 库
    ``chunk_gated_delta_rule``)，本实现作为**参考语义**与 CPU/CI 基准。

跨块状态传递：``forward_chunked`` 在块边界将状态 detach (TBPTT)，
训练长序列时激活显存只与块长有关，与总长度无关。
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import RMSNorm


class DeltaState:
    """DeltaNet 推理期运行时状态容器 (不进入计算图)。"""

    def __init__(self, batch: int, heads: int, dv: int, dk: int,
                 device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.S = torch.zeros(batch, heads, dv, dk, device=device, dtype=dtype)

    def nbytes(self) -> int:
        return self.S.numel() * self.S.element_size()

    def reset(self) -> None:
        self.S.zero_()


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet 线性循环层 (pre-norm + 残差内置)。

    Args:
        hidden_size: 模型维度 H。
        num_heads: 循环头数；dk = dv = H // num_heads。
        eps: RMSNorm 稳定常数。
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除。")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # dk = dv

        self.norm = RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)     # 写入门
        self.gamma_proj = nn.Linear(hidden_size, hidden_size, bias=True)  # 衰减门 (per dv)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    # ------------------------------------------------------------------
    def init_state(self, batch: int, device: torch.device,
                   dtype: torch.dtype = torch.float32) -> DeltaState:
        return DeltaState(batch, self.num_heads, self.head_dim, self.head_dim, device, dtype)

    # ------------------------------------------------------------------
    def _project(self, x: torch.Tensor):
        """x: (B, L, H) -> q/k/v (B, heads, L, d)，beta (B, heads, L, 1, 1)，
        gamma (B, heads, L, dv, 1)。"""
        B, L, H = x.shape
        nh, d = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, L, nh, d).transpose(1, 2)
        k = self.k_proj(x).view(B, L, nh, d).transpose(1, 2)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)             # delta 规则要求归一化键
        v = self.v_proj(x).view(B, L, nh, d).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).view(B, L, nh, 1, 1).transpose(1, 2)
        # log-decay 参数化: gamma = exp(-softplus(a)) ∈ (0,1)
        gamma = torch.exp(-F.softplus(self.gamma_proj(x))).view(B, L, nh, d, 1).transpose(1, 2)
        return q, k, v, beta, gamma

    # ------------------------------------------------------------------
    def _scan(
        self,
        x: torch.Tensor,
        S0: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """循环扫描主体。返回 (残差融合输出 (B,L,H), 最终状态 S (B,heads,dv,dk))。"""
        B, L, H = x.shape
        h = self.norm(x)
        q, k, v, beta, gamma = self._project(h)

        if S0 is not None:
            S = S0
        else:
            S = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim,
                            device=x.device, dtype=h.dtype)

        outs = []
        for t in range(L):
            S = S * gamma[:, :, t]                                  # 衰减
            err = v[:, :, t] - torch.einsum("bhdk,bhk->bhd", S, k[:, :, t])
            S = S + beta[:, :, t] * err.unsqueeze(-1) * k[:, :, t].unsqueeze(-2)
            outs.append(torch.einsum("bhdk,bhk->bhd", S, q[:, :, t]))

        o = torch.stack(outs, dim=2)                                # (B, heads, L, dv)
        o = o.transpose(1, 2).reshape(B, L, H)
        return x + self.o_proj(o), S

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        S0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """全量可微前向。

        Args:
            x: (B, L, H)。
            S0: 可选初始状态 (B, heads, dv, dk)，用于跨块延续。

        Returns:
            (output, S_final)：output 为 (B, L, H)；S_final 参与计算图。
        """
        torch._assert(x.dim() == 3, "GatedDeltaNet 期望 3D 输入 (B, L, H)")
        return self._scan(x, S0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_runtime(self, x: torch.Tensor, state: DeltaState) -> torch.Tensor:
        """推理前向：原地更新 ``state`` (detached)，返回 (B, L, H)。"""
        out, S = self._scan(x, state.S)
        state.S.copy_(S.detach())
        return out

    # ------------------------------------------------------------------
    def forward_chunked(
        self,
        x: torch.Tensor,
        chunk_size: int,
        detach_between_chunks: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分块前向 (TBPTT)：块内全可微，块间状态可选 detach。

        Returns:
            (output, S_final_detached)：output (B, L, H) 块内可微；
            最终状态恒为 detached，供下一段序列继续使用。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正。")
        L = x.shape[1]
        outs = []
        S: Optional[torch.Tensor] = None
        for start in range(0, L, chunk_size):
            seg = x[:, start:start + chunk_size]
            out, S = self._scan(seg, S)
            outs.append(out)
            if detach_between_chunks:
                S = S.detach()
        return torch.cat(outs, dim=1), (S.detach() if S is not None else S)

    def extra_repr(self) -> str:
        return f"H={self.hidden_size}, heads={self.num_heads}, d={self.head_dim}"
