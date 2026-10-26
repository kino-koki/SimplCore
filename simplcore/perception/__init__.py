# -*- coding: utf-8 -*-
"""SimplCore · 感知模块.

从摄像头电信号到 SimplCore 潜状态的端到端视觉管道.

模块层级:
  摄像头 (硬件) → CameraSource (捕获) → VisualStream (编码+时序) 
  → PerceptionPipeline (融合+推理) → SimplCore 核心

多模态扩展:
  视觉 → MultimodalEncoder → ModalityAttention → MultimodalPerception
  音频 → AudioPerception → 多模态融合
  环境 → EnvironmentPerception → 多模态融合

依赖:
  - opencv-python (摄像头捕获)
  - torch (ViT 推理)
"""

from SimplCore.perception.camera_source import CameraSource
from SimplCore.perception.visual_stream import VisualStream
from SimplCore.perception.pipeline import PerceptionPipeline

from SimplCore.perception.multimodal_fusion import (
    ModalityType,
    PerceptionState,
    MultimodalEncoder,
    ModalityAttention,
    PerceptionMemory,
    MultimodalPerception,
    AudioPerception,
    EnvironmentPerception,
)

__all__ = [
    "CameraSource",
    "VisualStream",
    "PerceptionPipeline",
    "ModalityType",
    "PerceptionState",
    "MultimodalEncoder",
    "ModalityAttention",
    "PerceptionMemory",
    "MultimodalPerception",
    "AudioPerception",
    "EnvironmentPerception",
]
