# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""多模态路由 (ModalityRouter) 与轻量视觉编码器 (SimplCoreViTEncoder)。

将文本 / 图像 / 视频统一投影到共享潜空间 (B, L, H)，并附加可学习模态类型嵌入，
供 SimplCore 核心统一处理。视觉编码使用紧凑 ViT (patch embed + 无注意力块 + CLS 池化)，
无注意力块由 LocalInteraction (局部) 与 ResonanceField (全局谱卷积) 构成，
与主干完全一致的"零自注意力"路线。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .layers import RMSNorm, FFN, init_weights
from .localmix import LocalInteraction
from .resonance import ResonanceField
from .exceptions import ModalityError


# ----------------------------------------------------------------------
# 轻量 ViT (完全无注意力：局部卷积/递归 + 谱全局卷积 替代自注意力)
# ----------------------------------------------------------------------
class _ViTBlock(nn.Module):
    """原生无注意力视觉块：LocalInteraction (局部) + ResonanceField (全局)。

    替代原始 ViT 中的自注意力：局部邻域混合由因果深度卷积+门控线性递归承担，
    全局 patch 间信息交互由谱全局卷积 (RFP) 承担；二者与 FFN 残差连接。
    整个编码器不含 Q/K/V/RoPE/SDPA。
    """

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int, dropout: float, eps: float) -> None:
        super().__init__()
        self.local = LocalInteraction(hidden_size, kernel_size=7, dilation=1,
                                      use_recurrence=True, eps=eps)
        self.rfp = ResonanceField(hidden_size, num_waves=8, num_steps=2,
                                   max_seq_len=max_seq_len)
        self.ffn_norm = RMSNorm(hidden_size, eps=eps)
        self.ffn = FFN(hidden_size, intermediate_ratio=4.0, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local(x)
        x = self.rfp(x)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SimplCoreViTEncoder(nn.Module):
    """紧凑视觉编码器：输出 (B, num_patches+1, output_dim)。"""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        output_dim: int = 768,
        in_channels: int = 3,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除。")
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        max_seq = self.num_patches + 1

        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_ViTBlock(hidden_size, num_heads, max_seq, dropout, eps) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)
        self.apply(lambda m: init_weights(m, std=0.02))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        torch._assert(pixel_values.dim() == 4 and pixel_values.shape[1] == self.in_channels,
                      f"图像应为 (B, {self.in_channels}, H, W)")
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values)              # (B, H, gh, gw)
        x = x.flatten(2).transpose(1, 2)                # (B, num_patches, H)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.shape[1] + 1, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.proj(x)  # (B, num_patches+1, output_dim)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.forward(pixel_values)
        return out[:, 1:, :]  # 去掉 CLS，保留 patch 序列

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        torch._assert(video.dim() == 5, "视频应为 (B, T, 3, H, W)")
        B, T, C, H, W = video.shape
        flat = video.reshape(B * T, C, H, W)
        lat = self.encode_image(flat)            # (B*T, n_patch, D)
        return lat.reshape(B, T * lat.shape[1], lat.shape[2])


# ----------------------------------------------------------------------
# 模态路由
# ----------------------------------------------------------------------
class ModalityRouter(nn.Module):
    """多模态统一路由。"""

    MODALITY_NAMES = ["text", "image", "video", "action"]

    def __init__(
        self,
        hidden_size: int,
        num_modalities: Optional[int] = None,
        use_vision: bool = False,
        image_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_vision = use_vision
        # 与 SimplCoreConfig.num_modalities 对齐；缺省沿用内置模态词表长度。
        # 内置 forward_text/image/video 仅使用索引 0/1/2，故表大小 >= 3 即可。
        self.num_modalities = int(num_modalities) if num_modalities is not None else len(self.MODALITY_NAMES)
        self.modality_embeddings = nn.Parameter(torch.randn(self.num_modalities, hidden_size) * 0.02)
        if use_vision:
            self.vision_encoder = SimplCoreViTEncoder(
                image_size=image_size, patch_size=patch_size,
                hidden_size=hidden_size, num_layers=4, num_heads=max(1, hidden_size // 64),
                output_dim=hidden_size,
            )
        else:
            self.vision_encoder = None
        self.fusion_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        torch._assert(text_embeds.dim() == 3, "文本嵌入应为 (B, L, H)")
        mod = self.modality_embeddings[0].view(1, 1, self.hidden_size)
        return self.fusion_proj(text_embeds + mod)

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.vision_encoder is None:
            raise NotImplementedError("视觉编码器未启用 (use_vision=False)。")
        latent = self.vision_encoder.encode_image(image)
        mod = self.modality_embeddings[1].view(1, 1, self.hidden_size)
        return self.fusion_proj(latent + mod)

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        if self.vision_encoder is None:
            raise NotImplementedError("视觉编码器未启用 (use_vision=False)。")
        latent = self.vision_encoder.encode_video(video)
        mod = self.modality_embeddings[2].view(1, 1, self.hidden_size)
        return self.fusion_proj(latent + mod)

    def forward(
        self,
        text_embeds: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latents, modality_ids = [], []
        if text_embeds is not None:
            latents.append(self.forward_text(text_embeds))
            modality_ids.append(torch.zeros(text_embeds.shape[1], device=text_embeds.device))
        if image is not None:
            lat = self.forward_image(image)
            latents.append(lat)
            modality_ids.append(torch.ones(lat.shape[1], device=image.device))
        if video is not None:
            lat = self.forward_video(video)
            latents.append(lat)
            modality_ids.append(torch.full((lat.shape[1],), 2, device=video.device))
        if not latents:
            raise ModalityError("至少需要一种模态输入。")
        joint = torch.cat(latents, dim=1)
        mask = torch.cat(modality_ids, dim=0).unsqueeze(0).expand(joint.shape[0], -1)
        return {"latent": joint, "modality_mask": mask.long()}
