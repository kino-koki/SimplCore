# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore Mix 多模态架构
========================
支持: 视觉理解 (Vis) + 判别器 (Dis) + 视频生成 + 跨模态融合

模块组成:
  - VisionEncoder:   ViT架构视觉编码器, 将图像编码为token序列
  - Discriminator:   GAN式判别器, 用于生成质量评估和对齐训练
  - CrossModalFusion: 视觉-文本跨模态融合层
  - VideoGenerator:   扩散模型+时序建模的视频生成器
  - ModalityRouter:   多模态输入路由器
  - MultimodalConfig: 多模态配置
"""
from .config import MultimodalConfig
from .vision_encoder import VisionEncoder
from .discriminator import Discriminator
from .cross_modal import CrossModalFusion
from .video_generator import VideoGenerator
from .modality_router import ModalityRouter

__all__ = [
    "MultimodalConfig",
    "VisionEncoder",
    "Discriminator",
    "CrossModalFusion",
    "VideoGenerator",
    "ModalityRouter",
]
