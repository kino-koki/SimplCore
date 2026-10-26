# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore MoE 层 (Mixture-of-Experts)。

设计要点 (对齐 PyTorch 2.1 图模式约束)：
  * 纯张量算子实现，无数据相关 Python 控制流，可被 ``torch.compile`` / FX 追踪；
  * token 按 top-k 路由到专家，采用**容量调度 (capacity dispatch)** 实现真实
    的 FLOP 节省 (每 token 仅激活 k/E 的 FFN 计算量)，而非朴素全专家计算；
  * 负载均衡损失 (Switch/Mixtral 风格) + router z-loss，抑制路由坍缩；
  * 可选共享专家 (shared expert) 稳定训练。

参数量口径：每个专家 = SwiGLU FFN = 3·H·E_inter；路由 = H·num_experts。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, init_weights


class ExpertFFN(nn.Module):
    """单个 MoE 专家 (SwiGLU 前馈，无 bias，中间维度显式给定)。"""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
        self.fc2 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.fc3 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc2(x)
        act = F.silu(self.fc1(x)) * gate
        act = self.dropout(act)
        return self.fc3(act)


class SharedExpert(nn.Module):
    """共享专家：所有 token 都经过，提升训练稳定性 (DeepSeek-MoE 风格)。"""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = ExpertFFN(hidden_size, intermediate_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoELayer(nn.Module):
    """Mixture-of-Experts 层 (pre-norm + 残差内置)。

    Args:
        hidden_size: 模型隐层维度 H。
        num_experts: 路由专家数量 E。
        experts_per_tok: 每 token 激活专家数 k (top-k)。
        intermediate_size: 每专家 FFN 中间维度 (SwiGLU)。
        aux_loss_coeff: 负载均衡损失系数。
        z_loss_coeff: router z-loss 系数 (抑制 router logits 过大)。
        capacity_factor: 每专家容量 = ceil(factor · (B·L) / E)；>1 允许溢出缓冲。
        use_shared_expert: 是否启用共享专家。
        shared_intermediate_size: 共享专家中间维度。
        dropout: 专家内 dropout。
        eps: RMSNorm 稳定常数。
        fixed_capacity: 固定容量 (torch.compile 动态形状下可避免 symint one_hot)。
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        experts_per_tok: int = 2,
        intermediate_size: int = 14336,
        aux_loss_coeff: float = 0.01,
        z_loss_coeff: float = 1e-4,
        capacity_factor: float = 1.25,
        use_shared_expert: bool = False,
        shared_intermediate_size: int = 0,
        dropout: float = 0.0,
        eps: float = 1e-6,
        fixed_capacity: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts 必须 >= 1，得到 {num_experts}。")
        if not (1 <= experts_per_tok <= num_experts):
            raise ValueError(f"experts_per_tok({experts_per_tok}) 必须在 [1, {num_experts}]。")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.experts_per_tok = experts_per_tok
        self.intermediate_size = intermediate_size
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.capacity_factor = capacity_factor
        self.fixed_capacity = fixed_capacity

        self.norm = RMSNorm(hidden_size, eps=eps)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertFFN(hidden_size, intermediate_size, dropout=dropout) for _ in range(num_experts)]
        )
        self.shared_expert = (
            SharedExpert(hidden_size, shared_intermediate_size or intermediate_size, dropout=dropout)
            if use_shared_expert else None
        )
        self.apply(lambda m: init_weights(m, std=0.02))

    # ------------------------------------------------------------------
    def _capacity(self, B: int, L: int) -> int:
        if self.fixed_capacity is not None:
            return self.fixed_capacity
        # 每专家容量需考虑 top-k：平均每个专家接收 (k·B·L)/E 个 token
        return max(1, int(math.ceil(self.capacity_factor * self.experts_per_tok * B * L / self.num_experts)))

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, training: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """MoE 前向。

        Args:
            x: (B, L, H) 输入 (已是 block 残差前的表示)。
            training: 是否返回辅助损失。

        Returns:
            (output, aux_loss)：output 为残差叠加后的 (B, L, H)；aux_loss 仅在
            training=True 时非 None。
        """
        torch._assert(x.dim() == 3, "MoELayer 期望 3D 输入 (B, L, H)")
        h = self.norm(x)
        B, L, H = h.shape
        E = self.num_experts
        k = self.experts_per_tok

        router_logits = self.router(h)                       # (B, L, E)
        probs = torch.softmax(router_logits, dim=-1)         # (B, L, E)
        weights, indices = torch.topk(probs, k, dim=-1)      # (B, L, k)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        capacity = self._capacity(B, L)

        # 容量调度：将每个 (token, k-slot) 映射到专家 e 的槽位 slot∈[0,capacity)
        # 关键：位置需在「展平的 (B·L·k) 序列」上做累计，确保同一专家内各 token
        # 获得唯一槽位 (而非在 k 维累计，否则不同专家的选择会抢占同一位置)。
        oh = F.one_hot(indices, num_classes=E).to(h.dtype)   # (B, L, k, E)
        oh_flat = oh.reshape(B * L * k, E)                   # (B·L·k, E) 每行一个选择
        pos_flat = torch.cumsum(oh_flat, dim=0) - 1.0        # 累计计数，-1 表示未选中
        pos = pos_flat.reshape(B, L, k, E).permute(0, 1, 3, 2)  # (B, L, E, k)
        valid = (pos >= 0) & (pos < capacity)                # (B, L, E, k)
        # 被丢弃 token 的 slot 可能 >= capacity，one_hot 会越界；先夹到 [0, cap-1]
        # (无效项最终被 valid 掩为 0，不影响结果)。
        slot = pos.clamp(min=0, max=capacity - 1).to(torch.long)  # (B, L, E, k)
        slot_onehot = F.one_hot(slot, num_classes=capacity)  # (B, L, E, k, capacity)
        # dispatch (B, L, E, capacity)：0/1 表示 token 是否进入该槽
        dispatch = (oh.permute(0, 1, 3, 2).unsqueeze(-1) * slot_onehot * valid.unsqueeze(-1)).sum(dim=3)
        # combine (B, L, E, capacity)：对应槽位的路由权重
        w_exp = weights.unsqueeze(2).expand(B, L, E, k)      # (B, L, E, k)
        combine = (w_exp * oh.permute(0, 1, 3, 2) * valid).unsqueeze(-1) * slot_onehot
        combine = combine.sum(dim=3)                          # (B, L, E, capacity)

        # 收集每个专家的 token：expert_inputs (B, E, capacity, H)
        expert_inputs = torch.einsum("blec,blh->bech", dispatch, h)
        expert_outs = torch.stack(
            [self.experts[e](expert_inputs[:, e]) for e in range(E)], dim=1
        )                                                    # (B, E, capacity, H)
        combined = torch.einsum("blec,bech->blh", combine, expert_outs)  # (B, L, H)

        out = combined
        if self.shared_expert is not None:
            out = out + self.shared_expert(h)
        out = out + x                                        # 残差

        aux = None
        if training:
            aux = self._load_balance_loss(probs, indices) + self._z_loss(router_logits)
        return out, aux

    # ------------------------------------------------------------------
    def _load_balance_loss(self, probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Switch/Mixtral 负载均衡损失：E · Σ_i f_i · P_i。"""
        E = self.num_experts
        mask = F.one_hot(indices, num_classes=E).sum(dim=2)  # (B, L, E) 各槽计数
        f = mask.float().mean(dim=(0, 1))                    # (E,) 路由占比
        P = probs.mean(dim=(0, 1))                           # (E,) 平均概率
        aux = E * torch.sum(f * P)
        return self.aux_loss_coeff * aux

    def _z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """router z-loss：抑制 router logits 过大导致的概率尖锐。"""
        lse = torch.logsumexp(router_logits, dim=-1)         # (B, L)
        return self.z_loss_coeff * (lse * lse).mean()

    def extra_repr(self) -> str:
        return (
            f"E={self.num_experts}, k={self.experts_per_tok}, "
            f"inter={self.intermediate_size}, cap_factor={self.capacity_factor}"
        )
