# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""模态路由 (ModalityRouter).

将文本、图像、视频、动作等不同模态的输入统一转发到 SimplCore 核心.
核心思想:
  1. 各模态独立编码为统一格式 (B, L, H).
  2. 按模态类型添加可学习的类型嵌入 (modality embedding).
  3. 拼接到一起, 形成统一的潜状态序列, 送给 SimplCore 核心处理.

Step1: 实现文本路由 (直接转发现有 embed 层的输出).
Step2: 实现视觉路由 (调用 VisionEncoder 并拼接模态嵌入).

架构示意图:
  Text Input     Image Input    Video Input
      ↓               ↓              ↓
  TextEmbed     VisionEncoder   VisionEncoder
      ↓               ↓              ↓
  (B,L_text,H)   (B,L_img,H)    (B,L_vid,H)
      ↓               ↓              ↓
  +mod_embed    +mod_embed      +mod_embed
      ↓               ↓              ↓
      └───────────────┼──────────────┘
                      ↓
                Conv 1x1 投影
                      ↓
              SimplCore 核心
"""

from __future__ import annotations

from typing import Dict, Optional

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

from SimplCore.architecture.vision_encoder import (
    SimplCoreViTEncoder,
    vit_tiny,
)

# ModalityRouter 默认使用 ViT-Tiny 作为视觉编码器
_DEFAULT_VIT = vit_tiny


class ModalityRouter(nn.Module):
    """模态路由 - 统一处理多模态输入.

    Args:
        hidden_size: 潜状态维度.
        num_modalities: 支持的模态数量 (text, image, video, action).
        use_vision: 是否启用视觉模态 (Step2 启用).
    """

    # 模态类型到索引的映射
    MODALITY_NAMES = ["text", "image", "video", "action"]

    def __init__(
        self,
        hidden_size: int,
        use_vision: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_vision = use_vision

        # 为每种模态学习一个可区分的嵌入向量
        self.modality_embeddings = nn.Parameter(
            torch.randn(len(self.MODALITY_NAMES), hidden_size) * 0.02
        )

        # 视觉编码器 (Step2: 使用真实 ViT)
        if use_vision:
            self.vision_encoder: SimplCoreViTEncoder = _DEFAULT_VIT(
                output_dim=hidden_size,
            )
        else:
            self.vision_encoder = None

        # 模态融合: 1x1 Conv 投影, 可选的跨模态融合层
        self.fusion_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_text(
        self, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """路由文本模态.

        Args:
            text_embeds: (B, L, H) 文本嵌入 (来自 embed 层).

        Returns:
            (B, L, H) 添加了模态嵌入的文本潜状态.
        """
        mod_embed = self.modality_embeddings[0].view(1, 1, self.hidden_size)
        return self.fusion_proj(text_embeds + mod_embed)

    def forward_image(
        self, image: torch.Tensor
    ) -> torch.Tensor:
        """路由图像模态 (Step2).

        Args:
            image: (B, C, H, W) 图像.

        Returns:
            (B, L_img, H) 添加了模态嵌入的视觉潜状态.
        """
        if self.vision_encoder is None:
            raise NotImplementedError("视觉编码器未启用 (Step1 暂不支持图像输入).")
        latent = self.vision_encoder.encode_image(image)
        mod_embed = self.modality_embeddings[1].view(1, 1, self.hidden_size)
        return self.fusion_proj(latent + mod_embed)

    def forward_video(
        self, video: torch.Tensor
    ) -> torch.Tensor:
        """路由视频模态 (Step2).

        Args:
            video: (B, T, C, H, W) 视频帧.

        Returns:
            (B, L_vid, H) 添加了模态嵌入的视觉潜状态.
        """
        if self.vision_encoder is None:
            raise NotImplementedError("视觉编码器未启用 (Step1 暂不支持视频输入).")
        latent = self.vision_encoder.encode_video(video)
        mod_embed = self.modality_embeddings[2].view(1, 1, self.hidden_size)
        return self.fusion_proj(latent + mod_embed)

    def forward(
        self,
        text_embeds: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """路由所有输入的模态到统一潜空间.

        Args:
            text_embeds: (B, L, H) 文本嵌入 (可选).
            image: (B, C, H, W) 图像 (可选).
            video: (B, T, C, H, W) 视频 (可选).

        Returns:
            dict: {
              "latent":  (B, L_total, H) 拼接后的联合潜状态,
              "modality_mask": (B, L_total) 模态类型索引,
            }
        """
        latents = []
        modality_ids = []

        if text_embeds is not None:
            latents.append(self.forward_text(text_embeds))
            modality_ids.append(
                torch.zeros(text_embeds.shape[1], device=text_embeds.device)
            )

        if image is not None:
            img_latent = self.forward_image(image)
            latents.append(img_latent)
            modality_ids.append(
                torch.ones(img_latent.shape[1], device=image.device)
            )

        if video is not None:
            vid_latent = self.forward_video(video)
            latents.append(vid_latent)
            modality_ids.append(
                torch.full((vid_latent.shape[1],), 2, device=video.device)
            )

        if len(latents) == 0:
            raise ValueError("至少需要一种模态输入.")

        joint_latent = torch.cat(latents, dim=1)
        joint_mask = torch.cat(modality_ids, dim=0).unsqueeze(0).expand(
            joint_latent.shape[0], -1
        )

        return {"latent": joint_latent, "modality_mask": joint_mask.long()}


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ModalityRouter 文本路由测试
    router_text = ModalityRouter(hidden_size=256, use_vision=False)
    text = torch.randn(2, 16, 256)
    out = router_text(text_embeds=text)
    print(f"文本路由: {text.shape} → {out['latent'].shape}")
    print(f"模态掩码: {out['modality_mask'].shape}")
    print("[PASS] ModalityRouter 文本路由正常.")

    # ModalityRouter 视觉路由测试 (Step2)
    router_vision = ModalityRouter(hidden_size=64, use_vision=True)
    img = torch.randn(2, 3, 224, 224)
    out_v = router_vision(image=img)
    print(f"视觉路由: {img.shape} → {out_v['latent'].shape}")
    print(f"模态掩码: {out_v['modality_mask'].shape}")

    # 多模态联合路由 (文本 + 图像)
    out_multi = router_vision(
        text_embeds=torch.randn(2, 8, 64),
        image=torch.randn(2, 3, 224, 224),
    )
    print(f"多模态路由: 文本+图像 → {out_multi['latent'].shape}")
    print(f"联合模态掩码: {out_multi['modality_mask'].shape}")
    print("[PASS] ModalityRouter 视觉+多模态路由正常.")


