# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 工具函数"""
import simplcore.tensor as torch
import simplcore.tensor.nn as nn
import math


class RMSNorm(nn.Module):
    """RMSNorm (OpenMythos风格)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 500000.0):
    """预计算RoPE频率 (实数版本，DirectML兼容)

    返回 [max_len, dim//2, 2] 的实数张量，最后一维为 [cos, sin]。
    替代原版 torch.polar 复数实现，避免 DirectML 不支持 ComplexFloat。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [max_len, dim//2]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [max_len, dim//2, 2]


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """应用RoPE (实数版本，DirectML兼容)

    使用旋转矩阵替代复数乘法:
      x1' = x1*cos(θ) - x2*sin(θ)
      x2' = x1*sin(θ) + x2*cos(θ)

    数学上等价于原版 complex multiplication by e^(iθ)。
    freqs_cis shape: [max_seq_len, max_head_dim//2, 2]
    """
    # x shape: [B, T, n_heads, head_dim] → pairs [B, T, n_heads, head_dim//2, 2]
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)

    # x_pairs has 5 dims; seq_len is dim -4, head_dim_half is dim -2
    seq_len = x_pairs.shape[-4]
    head_dim_half = x_pairs.shape[-2]

    # 截取到所需的序列长度和维度
    freqs = freqs_cis[:seq_len, :head_dim_half, :]  # [seq_len, head_dim//2, 2]
    cos = freqs[..., 0].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    sin = freqs[..., 1].unsqueeze(0).unsqueeze(2)

    x1 = x_pairs[..., 0]  # [B, T, n_heads, head_dim//2]
    x2 = x_pairs[..., 1]

    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def get_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """因果掩码 (在CPU上创建后移至设备, 兼容DirectML)"""
    return torch.triu(
        torch.full((seq_len, seq_len), -1e4),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0).to(device)
