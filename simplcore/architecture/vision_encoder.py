# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""视觉 Transformer 编码器 (SimplCoreViTEncoder).

将图像 / 视频编码为统一潜状态序列 (B, L, H), 供 SimplCore 核心处理.

核心组件:
  1. Patch Embedding: 用 Conv2d 将图像切分为 patch 并线性嵌入.
  2. CLS Token: 可学习的全局表示 token.
  3. Positional Embedding: 可学习的位置编码 (patch + CLS).
  4. Transformer Encoder: 标准多头自注意力 + FFN (pre-norm).
  5. Projection: 将内部嵌入维度投影到 output_dim.

接口:
  - encode_image(image):  (B, C, H, W)        -> (B, num_patches, output_dim)
  - encode_video(video):  (B, T, C, H, W)     -> (B, T*num_patches, output_dim)

其中 num_patches = (img_size / patch_size) ** 2, 例如 224/16 -> 14 -> 196.

工厂函数:
  - vit_tiny  : 6 层, 8 头 (小而快)
  - vit_small : 12 层, 12 头
  - vit_base  : 18 层, 16 头
"""
from __future__ import annotations

import math
from typing import Optional

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F


class SimplCoreViTEncoder(nn.Module):
    """SimplCore 视觉 Transformer 编码器.

    Args:
        img_size: 输入图像尺寸 (正方形).
        patch_size: patch 尺寸.
        in_channels: 输入通道数 (RGB=3).
        output_dim: 输出潜状态维度 (最终投影维度).
        num_layers: Transformer 层数.
        num_heads: 多头注意力头数.
        mlp_ratio: FFN 隐藏层相对 embed_dim 的倍率.
        dropout: dropout 概率.
        vit_dim: 内部嵌入维度别名 (兼容 pipeline.py 的 ``SimplCoreViTEncoder(vit_dim=...)``
                 调用); 为 None 时内部嵌入维度取 output_dim.
        vit_depth: num_layers 别名.
        vit_heads: num_heads 别名.

    说明:
        内部嵌入维度 ``embed_dim`` 默认等于 ``output_dim`` (便于与下游潜空间对齐);
        若传入 ``vit_dim`` 则使用 ``vit_dim`` 作为内部嵌入维度, 再通过线性投影映射到
        ``output_dim``. ``embed_dim`` 会被自动调整到 ``num_heads`` 的整数倍.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        output_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        vit_dim: Optional[int] = None,
        vit_depth: Optional[int] = None,
        vit_heads: Optional[int] = None,
    ) -> None:
        super().__init__()

        # 兼容 pipeline.py 的别名参数 (vit_depth / vit_heads 覆盖默认值)
        if vit_depth is not None:
            num_layers = vit_depth
        if vit_heads is not None:
            num_heads = vit_heads

        assert img_size > 0 and patch_size > 0
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) 必须能被 patch_size ({patch_size}) 整除"
        )
        assert num_heads > 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_patches = (img_size // patch_size) ** 2

        # 内部嵌入维度: 优先使用 vit_dim, 否则取 output_dim
        embed_dim = vit_dim if vit_dim is not None else output_dim
        # 保证 embed_dim 可被 num_heads 整除 (向上取整到最近的整数倍)
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + 1) * num_heads
        self.embed_dim = embed_dim

        # ---- Patch Embedding: Conv2d 等价于 patch 切分 + 线性投影 ----
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ---- CLS token (可学习) ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ---- 位置编码 (可学习, patch + CLS) ----
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        # ---- Transformer Encoder (pre-norm) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # ---- 最终投影到 output_dim ----
        if embed_dim != output_dim:
            self.proj: nn.Module = nn.Linear(embed_dim, output_dim)
        else:
            self.proj = nn.Identity()

        # ---- 时序位置编码 (视频用, 可学习) ----
        # 最大帧数 64, 超出 / 不足时通过插值适配任意 T.
        self.max_temporal_len = 64
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_temporal_len, 1, output_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    # ------------------------------------------------------------------
    def _add_temporal_pos(
        self, frame_latents: torch.Tensor, T: int
    ) -> torch.Tensor:
        """为视频帧潜状态添加时序位置编码.

        Args:
            frame_latents: (B, T, num_patches, output_dim)
            T: 帧数

        Returns:
            (B, T, num_patches, output_dim) 添加了时序位置编码的张量.
        """
        if T <= 0:
            return frame_latents
        tpe = self.temporal_pos_embed  # (1, max_T, 1, dim)
        if T != self.max_temporal_len:
            # 沿时序维度插值到 T
            tpe = tpe.permute(0, 3, 1, 2)            # (1, dim, max_T, 1)
            tpe = F.interpolate(
                tpe, size=(T, 1), mode="bilinear", align_corners=False
            )
            tpe = tpe.permute(0, 2, 3, 1)            # (1, T, 1, dim)
        return frame_latents + tpe

    # ------------------------------------------------------------------
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """编码图像为 patch token 序列.

        Args:
            image: (B, C, H, W) 图像张量.

        Returns:
            (B, num_patches, output_dim) 所有 patch token 的潜状态 (不含 CLS).
        """
        B = image.shape[0]
        x = self.patch_embed(image)              # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)         # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)           # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed                   # 加位置编码
        x = self.pos_drop(x)

        x = self.transformer(x)                  # (B, num_patches+1, embed_dim)
        x = self.norm(x)

        patch_tokens = x[:, 1:, :]               # 去掉 CLS, 保留 patch tokens
        out = self.proj(patch_tokens)            # (B, num_patches, output_dim)
        return out

    # ------------------------------------------------------------------
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """编码视频为 patch token 序列.

        对每一帧独立编码, 再沿序列维度拼接; 并叠加可学习的时序位置编码以区分帧序.

        Args:
            video: (B, T, C, H, W) 视频帧序列.

        Returns:
            (B, T*num_patches, output_dim) 拼接所有帧 patch 的潜状态.
        """
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        frame_latents = self.encode_image(frames)          # (B*T, num_patches, output_dim)
        frame_latents = frame_latents.view(
            B, T, self.num_patches, self.output_dim
        )

        # 添加时序位置编码 (区分不同帧)
        frame_latents = self._add_temporal_pos(frame_latents, T)

        # 沿序列维度拼接: (B, T*num_patches, output_dim)
        latent = frame_latents.reshape(B, T * self.num_patches, self.output_dim)
        return latent

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """默认前向等价于 :meth:`encode_image`."""
        return self.encode_image(image)


# ----------------------------------------------------------------------
# 工厂函数
# ----------------------------------------------------------------------
def vit_tiny(
    output_dim: int = 768,
    image_size: int = 224,
    img_size: Optional[int] = None,
    patch_size: int = 16,
    in_channels: int = 3,
    dropout: float = 0.0,
) -> "SimplCoreViTEncoder":
    """ViT-Tiny 配置: 6 层, 8 头 (小而快)."""
    size = image_size if img_size is None else img_size
    return SimplCoreViTEncoder(
        img_size=size,
        patch_size=patch_size,
        in_channels=in_channels,
        output_dim=output_dim,
        num_layers=6,
        num_heads=8,
        dropout=dropout,
    )


def vit_small(
    output_dim: int = 768,
    image_size: int = 224,
    img_size: Optional[int] = None,
    patch_size: int = 16,
    in_channels: int = 3,
    dropout: float = 0.0,
) -> "SimplCoreViTEncoder":
    """ViT-Small 配置: 12 层, 12 头."""
    size = image_size if img_size is None else img_size
    return SimplCoreViTEncoder(
        img_size=size,
        patch_size=patch_size,
        in_channels=in_channels,
        output_dim=output_dim,
        num_layers=12,
        num_heads=12,
        dropout=dropout,
    )


def vit_base(
    output_dim: int = 768,
    image_size: int = 224,
    img_size: Optional[int] = None,
    patch_size: int = 16,
    in_channels: int = 3,
    dropout: float = 0.0,
) -> "SimplCoreViTEncoder":
    """ViT-Base 配置: 18 层, 16 头."""
    size = image_size if img_size is None else img_size
    return SimplCoreViTEncoder(
        img_size=size,
        patch_size=patch_size,
        in_channels=in_channels,
        output_dim=output_dim,
        num_layers=18,
        num_heads=16,
        dropout=dropout,
    )


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 60)
    print("SimplCoreViTEncoder 自检")
    print("=" * 60)

    # 1) vit_tiny 图像编码: (2,3,224,224) -> (2,196,64)
    enc = vit_tiny(output_dim=64)
    img = torch.randn(2, 3, 224, 224)
    latent = enc.encode_image(img)
    print(f"图像编码: {tuple(img.shape)} -> {tuple(latent.shape)}")
    assert latent.shape == (2, 196, 64), f"形状错误: {latent.shape}"

    # 2) 视频编码: (2,4,3,224,224) -> (2,4*196,64)
    video = torch.randn(2, 4, 3, 224, 224)
    vlatent = enc.encode_video(video)
    print(f"视频编码: {tuple(video.shape)} -> {tuple(vlatent.shape)}")
    assert vlatent.shape == (2, 4 * 196, 64), f"形状错误: {vlatent.shape}"

    # 3) 梯度流
    img2 = torch.randn(1, 3, 224, 224, requires_grad=True)
    out2 = enc.encode_image(img2)
    out2.pow(2).mean().backward()
    has_grad = img2.grad is not None and img2.grad.abs().sum() > 0
    print(f"梯度流: {'OK' if has_grad else 'FAIL'}")
    assert has_grad

    # 4) pipeline.py 兼容构造 (vit_dim / vit_depth / vit_heads 别名)
    enc2 = SimplCoreViTEncoder(
        vit_dim=192, vit_depth=12, vit_heads=3, output_dim=256
    )
    l2 = enc2.encode_image(torch.randn(1, 3, 224, 224))
    print(f"pipeline 兼容 (vit_dim=192): {tuple(l2.shape)}")
    assert l2.shape == (1, 196, 256)

    # 5) 工厂函数参数量
    for name, fn in [("tiny", vit_tiny), ("small", vit_small), ("base", vit_base)]:
        n_params = sum(p.numel() for p in fn(output_dim=128).parameters())
        print(f"vit_{name}(output_dim=128): {n_params:,} 参数")

    print("\n[PASS] 视觉编码器 (ViT) 测试全部通过.")
