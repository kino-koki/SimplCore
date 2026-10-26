# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
多模态配置系统
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MultimodalConfig:
    """多模态架构配置

    涵盖视觉编码、判别器、视频生成和跨模态融合的所有超参数。
    """

    # ── 通用 ──
    dim: int = 2048  # 主模型隐藏维度 (与 SimplCoreMixConfig.dim 对齐)
    multimodal_enabled: bool = True

    # ── 视觉编码器 (ViT) ──
    vis_enabled: bool = True
    vis_image_size: int = 224          # 输入图像分辨率
    vis_patch_size: int = 16           # patch 大小
    vis_channels: int = 3              # RGB
    vis_hidden_dim: int = 1024         # ViT 内部维度
    vis_n_layers: int = 12             # ViT 层数
    vis_n_heads: int = 16              # ViT 注意力头数
    vis_mlp_ratio: float = 4.0         # FFN 扩展比
    vis_dropout: float = 0.0
    # 输出: 将 vis_hidden_dim 投影到 dim
    vis_projection_dropout: float = 0.1

    # ── 判别器 (Dis) ──
    dis_enabled: bool = True
    dis_hidden_dim: int = 768
    dis_n_layers: int = 6
    dis_n_heads: int = 8
    dis_dropout: float = 0.1
    # 判别器输出维度 (用于质量评分和对齐损失)
    dis_output_dim: int = 1            # 真假判别
    dis_feature_dim: int = 256         # 特征提取维度 (用于对比学习)

    # ── 跨模态融合 ──
    fusion_enabled: bool = True
    fusion_n_layers: int = 4           # 融合层数
    fusion_n_heads: int = 16
    fusion_dropout: float = 0.1
    # 对齐温度 (对比学习)
    fusion_contrastive_temperature: float = 0.07
    # 视觉 token 插入位置: "prepend" | "interleave" | "cross_attention"
    fusion_mode: str = "prepend"

    # ── 视频生成 ──
    video_enabled: bool = True
    video_latent_dim: int = 4          # VAE 潜空间维度
    video_latent_size: tuple = (32, 32)  # 潜空间空间分辨率
    video_max_frames: int = 16         # 最大帧数
    video_frame_size: int = 256        # 帧分辨率
    video_diffusion_steps: int = 1000  # 扩散步数 (训练)
    video_diffusion_sampling_steps: int = 50  # 采样步数
    video_unet_dim: int = 320          # UNet 基础维度
    video_unet_dim_mults: tuple = (1, 2, 4, 4)  # UNet 维度乘数
    video_unet_n_layers: int = 2       # 每层 block 数
    video_temporal_heads: int = 8      # 时序注意力头数
    video_dropout: float = 0.1
    # 文本条件维度 (从主模型获取)
    video_text_dim: int = 2048

    # ── 模态路由 ──
    router_enabled: bool = True
    router_num_modalities: int = 4     # text / image / video / audio
    router_hidden_dim: int = 512
    router_dropout: float = 0.1

    # ── 音频 (预留) ──
    audio_enabled: bool = False
    audio_sample_rate: int = 16000
    audio_n_mels: int = 80
