# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
视觉编码器模块 (ViT 架构)
========================
将图像编码为视觉 token 序列, 供多模态主模型消费。

核心组件:
  1. PatchEmbed:             图像分 patch + 线性投影
     (224/16 -> 14x14=196 patches; 448/16 -> 28x28=784 patches)
     采用 reshape+permute+Linear 实现 (数学上等价于 kernel=stride=patch 的 Conv2d),
     兼容自研 NumPy 后端 (无需 Conv2d 算子)。
  2. MultiHeadSelfAttention: 标准多头自注意力 (双向, 无因果掩码)
  3. FeedForward:            两层 MLP, GELU 激活
  4. VisionTransformerBlock: ViT 编码器层 = MSA + FFN (Pre-Norm, LayerNorm)
  5. VisionEncoder:          完整 ViT 编码器
     - patch embedding + CLS token + 可学习位置编码
     - N 层 VisionTransformerBlock
     - 最终 LayerNorm + 线性投影到主模型维度 (vis_hidden_dim -> dim)
     - 支持 448x448 / 224x224 等任意 (patch_size 整数倍) 输入分辨率
       (位置编码通过双线性插值自适应, 仅用 matmul 实现, 双后端兼容)

接口:
  forward(images: [B, C, H, W]) -> visual_tokens: [B, num_patches, dim]

底层张量库为 ``simplcore.tensor`` (双后端: 自研 NumPy 引擎 / PyTorch),
完全脱离对 PyTorch 的硬依赖, 与 SimplCore Mix 其余模块保持一致。
"""
from __future__ import annotations

from typing import Optional

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from .config import MultimodalConfig


# ============================================================
# Patch Embedding (Conv2d-free)
# ============================================================
class PatchEmbed(nn.Module):
    """图像分 patch + 线性投影

    将 [B, C, H, W] 图像切分为 (H/P)*(W/P) 个不重叠 patch, 经线性投影
    映射到 ``embed_dim`` 维, 输出 [B, num_patches, embed_dim]。

    数学上等价于 ``Conv2d(C, embed_dim, kernel_size=P, stride=P)``,
    但仅用 reshape + permute + Linear 实现, 因此在自研 NumPy 后端
    (不提供 Conv2d) 上亦可运行。

    Args:
        image_size: 输入图像边长 (正方形, 须被 patch_size 整除)。
        patch_size: patch 边长。
        channels:   图像通道数 (RGB=3)。
        embed_dim:  patch 嵌入维度 (ViT 内部维度)。
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        assert image_size > 0 and patch_size > 0, (
            f"image_size({image_size}) 与 patch_size({patch_size}) 必须为正"
        )
        assert image_size % patch_size == 0, (
            f"image_size({image_size}) 必须能被 patch_size({patch_size}) 整除"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # 线性投影: 展平的 patch 像素 -> embed_dim
        self.proj = nn.Linear(channels * patch_size * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算

        Args:
            x: [B, C, H, W] 图像张量 (H, W 须被 patch_size 整除,
               不要求等于 image_size, 支持可变分辨率)。
        Returns:
            [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        P = self.patch_size
        gh, gw = H // P, W // P

        # [B, C, H, W] -> [B, C, gh, P, gw, P]
        # H 维拆成 (gh, P), W 维拆成 (gw, P), 即按 patch 网格切分
        x = x.reshape(B, C, gh, P, gw, P)
        # -> [B, gh, gw, C, P, P]  (patch 网格在前, 通道/像素在后)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # -> [B, gh*gw, C*P*P]
        x = x.reshape(B, gh * gw, C * P * P)
        # 线性投影到 embed_dim
        x = self.proj(x)
        return x


# ============================================================
# 多头自注意力 (ViT 专用, 双向)
# ============================================================
class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力 (双向, 无因果掩码)

    Args:
        dim:     输入/输出维度。
        n_heads: 注意力头数。
        dropout: 注意力权重 dropout 概率。
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert n_heads > 0, "注意力头数必须大于 0"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim > 0, (
            f"dim({dim}) 必须大于等于 n_heads({n_heads})"
        )
        # inner_dim 可小于 dim (当 dim 不能被 n_heads 整除时向下取整)
        self.inner_dim = self.n_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # Q/K/V 独立投影 (双后端兼容, 与 mix/attention.py 风格一致)
        self.wq = nn.Linear(dim, self.inner_dim, bias=False)
        self.wk = nn.Linear(dim, self.inner_dim, bias=False)
        self.wv = nn.Linear(dim, self.inner_dim, bias=False)
        # 输出投影
        self.wo = nn.Linear(self.inner_dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算

        Args:
            x: [B, T, dim]
        Returns:
            [B, T, dim]
        """
        B, T, _ = x.shape

        # [B, T, dim] -> [B, T, inner_dim] -> [B, n_heads, T, head_dim]
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数 [B, n_heads, T, T]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和 [B, n_heads, T, head_dim] -> 合并头 [B, T, inner_dim]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, self.inner_dim)
        return self.proj_drop(self.wo(out))


# ============================================================
# FFN (前馈网络, GELU 激活)
# ============================================================
class FeedForward(nn.Module):
    """两层前馈网络: dim -> hidden_dim -> dim, 使用 GELU 激活

    Args:
        dim:       输入/输出维度。
        hidden_dim:中间扩展维度。
        dropout:   dropout 概率。
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.drop(h)


# ============================================================
# VisionTransformerBlock (Pre-Norm)
# ============================================================
class VisionTransformerBlock(nn.Module):
    """ViT 编码器层: Pre-Norm 多头自注意力 + Pre-Norm FFN

    结构::
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Args:
        dim:       模型维度。
        n_heads:   注意力头数。
        mlp_ratio: FFN 中间层相对 dim 的倍率。
        dropout:   dropout 概率。
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算

        Args:
            x: [B, T, dim]
        Returns:
            [B, T, dim]
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 位置编码双线性插值 (双后端兼容, 仅用 matmul 实现)
# ============================================================
def _resample_matrix(old: int, new: int) -> torch.Tensor:
    """构造 1D 线性插值重采样矩阵

    返回矩阵 R: [new, old], 满足 ``o = R @ s`` 将长度 ``old`` 的信号
    重采样为长度 ``new``。采用 tent 核 (线性插值核):

        R[i, j] = max(0, 1 - |src_i - j|),  src_i = (i + 0.5) * old / new - 0.5

    (half-pixel 中心, align_corners=False), 行归一化到和为 1。
    仅使用 arange / 算术 / abs / clamp / sum, 双后端通用。

    Args:
        old: 源长度。
        new: 目标长度。
    Returns:
        [new, old] 重采样矩阵。
    """
    if old == new:
        return torch.eye(old)
    scale = old / new
    # 输出索引 -> 源坐标 (half-pixel)
    src = (torch.arange(new).float() + 0.5) * scale - 0.5  # [new]
    j_idx = torch.arange(old).float()                        # [old]
    # 距离矩阵 [new, old]
    dist = (src.unsqueeze(1) - j_idx.unsqueeze(0)).abs()
    R = (1.0 - dist).clamp(min=0.0)                          # [new, old]
    # 行归一化 (边界情形保证权重和为 1)
    R = R / R.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return R


# ============================================================
# VisionEncoder
# ============================================================
class VisionEncoder(nn.Module):
    """完整 ViT 视觉编码器

    处理流程:
        1. PatchEmbed: [B, C, H, W] -> [B, num_patches, vis_hidden_dim]
        2. 预置 CLS token + 叠加可学习位置编码
           (输入分辨率与配置不一致时, 对 patch 位置编码做 2D 双线性插值)
        3. N 层 VisionTransformerBlock
        4. 最终 LayerNorm, 去掉 CLS, 保留 patch tokens
        5. 线性投影: vis_hidden_dim -> dim (主模型隐藏维度)

    Args:
        cfg: MultimodalConfig 配置对象
    """

    def __init__(self, cfg: MultimodalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.vis_image_size
        self.patch_size = cfg.vis_patch_size
        self.channels = cfg.vis_channels
        self.embed_dim = cfg.vis_hidden_dim
        self.n_layers = cfg.vis_n_layers
        self.n_heads = cfg.vis_n_heads
        self.mlp_ratio = cfg.vis_mlp_ratio

        # ---- Patch Embedding ----
        self.patch_embed = PatchEmbed(
            image_size=self.image_size,
            patch_size=self.patch_size,
            channels=self.channels,
            embed_dim=self.embed_dim,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches

        # ---- CLS token (可学习) ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # ---- 位置编码 (CLS + patches, 可学习) ----
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        nn.init.normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(cfg.vis_dropout)

        # ---- N 层 ViT 编码器 ----
        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    dim=self.embed_dim,
                    n_heads=self.n_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=cfg.vis_dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        # ---- 最终归一化 ----
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # ---- 投影到主模型维度 ----
        self.proj = nn.Linear(self.embed_dim, cfg.dim)
        self.proj_drop = nn.Dropout(cfg.vis_projection_dropout)

    # ------------------------------------------------------------------
    def _interpolate_pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        """将位置编码双线性插值到 [gh, gw] patch 网格

        保留 CLS 位置编码不变, 仅对 patch 部分做 2D 双线性插值。
        通过重采样矩阵 + matmul 实现 (``R_h @ grid @ R_w^T``), 双后端通用。

        Args:
            gh, gw: 目标 patch 网格高/宽。
        Returns:
            [1, 1 + gh*gw, embed_dim] 插值后的位置编码。
        """
        pe = self.pos_embed                       # [1, N0+1, D]
        cls_pe = pe[:, :1, :]                     # [1, 1, D]
        patch_pe = pe[:, 1:, :]                   # [1, N0, D]
        g0 = self.grid_size
        D = self.embed_dim

        # [1, N0, D] -> [1, g0, g0, D] -> [D, g0, g0]
        patch_pe = patch_pe.reshape(1, g0, g0, D)
        patch_pe = patch_pe.permute(0, 3, 1, 2)   # [1, D, g0, g0]
        patch_pe = patch_pe.reshape(D, g0, g0)    # [D, g0, g0]

        # 重采样矩阵: R_h[g0_new, g0_old] / R_w[gw_new, gw_old]
        R_h = _resample_matrix(g0, gh)            # [gh, g0]
        R_w = _resample_matrix(g0, gw)            # [gw, g0]
        # R_h @ grid @ R_w^T  (broadcasting 批量 matmul)
        #   [gh, g0] @ [D, g0, g0] -> [D, gh, g0]
        #   @ [g0, gw]            -> [D, gh, gw]
        patch_pe = torch.matmul(R_h, patch_pe)               # [D, gh, g0]
        patch_pe = torch.matmul(patch_pe, R_w.transpose(0, 1))  # [D, gh, gw]

        # [D, gh, gw] -> [1, gh*gw, D]
        patch_pe = patch_pe.reshape(1, D, gh, gw)
        patch_pe = patch_pe.permute(0, 2, 3, 1)   # [1, gh, gw, D]
        patch_pe = patch_pe.reshape(1, gh * gw, D)

        return torch.cat([cls_pe, patch_pe], dim=1)  # [1, 1+gh*gw, D]

    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向计算

        Args:
            images: [B, C, H, W] 图像张量。H, W 须被 patch_size 整除,
                    可与配置的 image_size 不同 (如 448 或 224),
                    位置编码会自适应插值。
        Returns:
            [B, num_patches, dim] 视觉 token 序列 (不含 CLS)。
        """
        B = images.shape[0]

        # 1. Patch embedding: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(images)
        _, N, _ = x.shape
        # 由 patch 数反推网格尺寸 (正方形输入)
        gh = int(round(N ** 0.5))
        gw = N // gh

        # 2. CLS token
        cls = self.cls_token.expand(B, -1, -1)    # [B, 1, embed_dim]
        x = torch.cat([cls, x], dim=1)            # [B, num_patches+1, embed_dim]

        # 3. 位置编码 (必要时插值)
        if gh != self.grid_size or gw != self.grid_size:
            pos = self._interpolate_pos_embed(gh, gw)
        else:
            pos = self.pos_embed
        x = x + pos
        x = self.pos_drop(x)

        # 4. N 层 ViT 编码器
        for block in self.blocks:
            x = block(x)

        # 5. 最终归一化, 去掉 CLS
        x = self.norm(x)
        x = x[:, 1:, :]                           # [B, num_patches, embed_dim]

        # 6. 投影到主模型维度
        x = self.proj(x)
        x = self.proj_drop(x)
        return x                                  # [B, num_patches, dim]


__all__ = [
    "PatchEmbed",
    "MultiHeadSelfAttention",
    "FeedForward",
    "VisionTransformerBlock",
    "VisionEncoder",
    "_resample_matrix",
]


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 小型配置以便快速自检
    cfg = MultimodalConfig(
        vis_image_size=224,
        vis_patch_size=16,
        vis_channels=3,
        vis_hidden_dim=128,
        vis_n_layers=2,
        vis_n_heads=4,
        vis_mlp_ratio=4.0,
        vis_dropout=0.0,
        vis_projection_dropout=0.0,
        dim=256,
    )

    print("=" * 60)
    print("VisionEncoder 自检")
    print("=" * 60)

    enc = VisionEncoder(cfg)

    # 1) 224x224 输入
    img224 = torch.randn(2, 3, 224, 224)
    out224 = enc(img224)
    print(f"224x224: {tuple(img224.shape)} -> {tuple(out224.shape)}")
    assert out224.shape == (2, 196, 256), f"形状错误: {out224.shape}"

    # 2) 448x448 输入 (位置编码插值)
    img448 = torch.randn(2, 3, 448, 448)
    out448 = enc(img448)
    print(f"448x448: {tuple(img448.shape)} -> {tuple(out448.shape)}")
    assert out448.shape == (2, 784, 256), f"形状错误: {out448.shape}"

    # 3) 梯度流
    img_g = torch.randn(1, 3, 224, 224)
    out_g = enc(img_g)
    out_g.pow(2).mean().backward()
    ok_grad = enc.proj.weight.grad is not None
    print(f"梯度流: {'OK' if ok_grad else 'FAIL'}")
    assert ok_grad

    # 4) 参数量
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"参数量: {n_params:,}")

    print("\n[PASS] 视觉编码器 (ViT) 测试全部通过.")
