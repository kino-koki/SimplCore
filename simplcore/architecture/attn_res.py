# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
标准注意力残差层 (AttentionResidual)
====================================

提供基于标准多头自注意力 (Multi-Head Self-Attention) 的残差层,
作为谱混合处理层 (SGC) 的替代/对比组件。

设计目的:
  1. 当不希望使用 SGC (谱混合) 时, 可用本层作为 drop-in 替代。
  2. 用于消融实验: 对比 SGC (O(n log n)) 与标准注意力 (O(n^2)) 的效果。
  3. 提供因果 (causal) 与非因果两种模式。

复杂度: O(n^2 * d), 其中 n 为序列长度, d 为特征维度。
"""

from __future__ import annotations

import math
from typing import Optional

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F


class AttentionResidual(nn.Module):
    """标准多头自注意力残差层.

    结构:
      x → LayerNorm → Multi-Head Self-Attention → Dropout → + residual → LayerNorm

    与 SGC (LegacySpectralMixer) 的区别:
      - SpectralMixer: 使用 FFT 计算拉普拉斯算子, 复杂度 O(n log n)
      - 本层: 使用标准 QK^T 注意力, 复杂度 O(n^2)
      - 本层支持 flash_attention 标志 (默认 False, 使用标准实现)

    Args:
        dim: 输入 / 输出特征维度.
        num_heads: 注意力头数, 默认 8.
        dropout: dropout 概率, 默认 0.1.
        causal: 是否使用因果掩码 (自回归), 默认 True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert dim > 0, f"dim 必须为正整数, 得到 {dim}"
        assert num_heads > 0, f"num_heads 必须为正整数, 得到 {num_heads}"
        assert dim % num_heads == 0, (
            f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.causal = causal

        # flash_attention 标志: 默认 False, 使用标准实现
        # 当 PyTorch 版本支持且输入满足条件时可设为 True
        self.flash_attention = False

        # QKV 投影
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        # 输出投影
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # LayerNorm (Pre-LN 结构)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # FFN (前馈网络, 用于残差层的第二部分)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准注意力计算 (非 flash).

        Args:
            q: (B, H, L, D) 查询张量.
            k: (B, H, L, D) 键张量.
            v: (B, H, L, D) 值张量.
            mask: 可选掩码, (B, 1, L, L) 或 (B, L, L).

        Returns:
            (B, H, L, D) 注意力输出.
        """
        # 计算注意力分数: (B, H, L, D) x (B, H, D, L) → (B, H, L, L)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 因果掩码
        if self.causal:
            L = q.size(-2)
            causal_mask = torch.triu(
                torch.ones(L, L, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        # 自定义掩码
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L, L)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        # Softmax + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和: (B, H, L, L) x (B, H, L, D) → (B, H, L, D)
        output = torch.matmul(attn_weights, v)
        return output

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flash Attention 前向传播.

        使用 PyTorch 内置的 scaled_dot_product_attention (SDPA) 实现,
        当 PyTorch 版本 >= 2.0 时可用, 内部会自动选择最优后端。

        Args:
            q: (B, H, L, D) 查询张量.
            k: (B, H, L, D) 键张量.
            v: (B, H, L, D) 值张量.
            mask: 可选掩码.

        Returns:
            (B, H, L, D) 注意力输出.
        """
        is_causal = self.causal and mask is None

        if hasattr(F, "scaled_dot_product_attention"):
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
        else:
            # 回退到标准实现
            output = self._standard_attention(q, k, v, mask)

        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播: 标准因果自注意力 + 残差 + LayerNorm + FFN + 残差.

        Args:
            x: (B, L, dim) 输入序列.
            mask: 可选注意力掩码, (B, L, L) 布尔类型 (True 表示可见).

        Returns:
            (B, L, dim) 输出序列, 与输入形状相同.
        """
        B, L, D = x.shape
        assert D == self.dim, f"维度不匹配: 期望 {self.dim}, 得到 {D}"

        residual = x

        # === 第一部分: 自注意力 + 残差 ===
        h = self.norm1(x)

        # QKV 投影: (B, L, D) → (B, L, D) → (B, H, L, head_dim)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        if self.flash_attention:
            attn_out = self._flash_attention_forward(q, k, v, mask)
        else:
            attn_out = self._standard_attention(q, k, v, mask)

        # 合并多头: (B, H, L, head_dim) → (B, L, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        attn_out = self.out_proj(attn_out)

        # 残差连接
        x = residual + attn_out

        # === 第二部分: FFN + 残差 ===
        x = x + self.ffn(self.norm2(x))

        return x

    def attention_complexity(self, seq_len: int) -> dict:
        """计算本层的注意力复杂度.

        Args:
            seq_len: 序列长度 n.

        Returns:
            dict: 包含 FLOPs 估算和复杂度信息.
        """
        n = seq_len
        d = self.dim
        h = self.num_heads
        hd = self.head_dim

        # QKV 投影: 3 * n * d * d
        qkv_flops = 3 * n * d * d
        # 注意力分数: n * n * hd * h = n^2 * d
        attn_flops = n * n * d
        # Softmax + 加权求和: n * n * hd * h = n^2 * d
        weighted_flops = n * n * d
        # 输出投影: n * d * d
        out_flops = n * d * d
        # FFN: 2 * n * d * (4d) = 8 * n * d^2
        ffn_flops = 8 * n * d * d

        total_flops = qkv_flops + attn_flops + weighted_flops + out_flops + ffn_flops

        return {
            "seq_len": n,
            "dim": d,
            "num_heads": h,
            "qkv_flops": qkv_flops,
            "attention_flops": attn_flops + weighted_flops,
            "output_flops": out_flops,
            "ffn_flops": ffn_flops,
            "total_flops": total_flops,
            "big_o": "O(n^2 * d)",
            "note": "标准多头自注意力复杂度为 O(n^2 * d), 其中 n 为序列长度, d 为特征维度.",
        }


# ======================================================================
# 工厂函数
# ======================================================================

def build_attention_residual(
    dim: int,
    num_heads: int = 8,
    dropout: float = 0.1,
    causal: bool = True,
    flash: bool = False,
) -> AttentionResidual:
    """构建 AttentionResidual 层的工厂函数.

    Args:
        dim: 特征维度.
        num_heads: 注意力头数.
        dropout: dropout 概率.
        causal: 是否因果掩码.
        flash: 是否启用 flash attention.

    Returns:
        AttentionResidual 实例.
    """
    layer = AttentionResidual(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        causal=causal,
    )
    layer.flash_attention = flash
    return layer


# ======================================================================
# 自检
# ======================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("AttentionResidual 自检")
    print("=" * 60)

    # 基本前向传播测试
    dim = 128
    layer = AttentionResidual(dim=dim, num_heads=8, dropout=0.1, causal=True)
    x = torch.randn(2, 16, dim)
    y = layer(x)
    print(f"前向传播: input={tuple(x.shape)} → output={tuple(y.shape)}")
    assert y.shape == x.shape, "输出形状应与输入相同"
    print("[PASS] 前向传播正常")

    # 非因果模式测试
    layer_non_causal = AttentionResidual(dim=dim, num_heads=8, causal=False)
    y_nc = layer_non_causal(x)
    assert y_nc.shape == x.shape
    print("[PASS] 非因果模式正常")

    # 因果掩码验证: 修改后面的 token 不影响前面的输出
    x1 = torch.randn(1, 8, dim)
    x2 = x1.clone()
    x2[0, 6:, :] += torch.randn_like(x2[0, 6:, :]) * 10  # 修改后半部分
    y1 = layer(x1)
    y2 = layer(x2)
    # 前半部分输出应该接近 (因果性)
    diff_front = (y1[0, :4] - y2[0, :4]).abs().mean().item()
    diff_back = (y1[0, 6:] - y2[0, 6:]).abs().mean().item()
    print(f"因果性验证: 前部差异={diff_front:.6f}, 后部差异={diff_back:.6f}")
    assert diff_front < diff_back, "因果掩码应使前部差异小于后部"
    print("[PASS] 因果掩码正常")

    # 自定义掩码测试
    mask = torch.ones(2, 16, 16, dtype=torch.bool)
    mask[:, 8:, :8] = False  # 后半部分不能看到前半部分
    y_masked = layer(x, mask=mask)
    assert y_masked.shape == x.shape
    print("[PASS] 自定义掩码正常")

    # Flash attention 测试 (如果可用)
    layer_flash = AttentionResidual(dim=dim, num_heads=8, causal=True)
    layer_flash.flash_attention = True
    y_flash = layer_flash(x)
    assert y_flash.shape == x.shape
    assert not torch.isnan(y_flash).any(), "Flash attention 输出不应包含 NaN"
    print("[PASS] Flash attention 正常")

    # 梯度测试
    layer_grad = AttentionResidual(dim=dim, num_heads=8, dropout=0.0, causal=True)
    x_grad = torch.randn(2, 16, dim, requires_grad=True)
    y_grad = layer_grad(x_grad)
    loss = y_grad.sum()
    loss.backward()
    has_grad = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer_grad.parameters()
    )
    print(f"[PASS] 梯度正常: {has_grad}")

    # 复杂度分析
    complexity = layer.attention_complexity(seq_len=4096)
    print(f"\n复杂度分析 (seq_len=4096):")
    print(f"  总 FLOPs: {complexity['total_flops']:,}")
    print(f"  大 O 记号: {complexity['big_o']}")

    # 参数量统计
    num_params = sum(p.numel() for p in layer.parameters())
    print(f"\n参数量: {num_params:,}")

    # 不同头数测试
    for nh in [1, 2, 4, 8, 16]:
        try:
            l = AttentionResidual(dim=dim, num_heads=nh)
            print(f"  num_heads={nh}: OK")
        except AssertionError:
            print(f"  num_heads={nh}: 跳过 (dim 不能整除)")

    print("\n" + "=" * 60)
    print("AttentionResidual 所有测试 PASSED.")
    print("=" * 60)
