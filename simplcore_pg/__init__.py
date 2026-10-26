# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore 生产级实现 (PyTorch 2.1，完全无自注意力)。

公开 API (全部架构内算子均不含 Q/K/V/RoPE/SDPA)：
  * config.SimplCoreConfig —— 类型化配置与预设；
  * model.SimplCoreModel / model_moe.SimplCoreMoEModel —— 完整模型；
  * model_stream.SimplCoreStreamingModel —— 长上下文流式模型；
  * block.SimplCoreBlock / model_moe.SimplCoreMoEBlock —— 计算单元；
  * checkpointing.save_pretrained / from_pretrained —— 权重安全保存/加载管线
    (AES-256-GCM 加密 / RBAC 鉴权 / 内存金丝雀保护 / 指纹防篡改)；
  * localmix.LocalInteraction —— 原生局部交互层 (因果卷积+线性递归)；
  * resonance.ResonanceField —— 谱全局卷积 (RFP，全局感受野)；
  * memory.DynamicMemoryChannels —— 动态记忆通道 (DMC)；
  * routing.TopologyAwareRouter —— 拓扑感知路由；
  * multimodal.ModalityRouter / SimplCoreViTEncoder —— 多模态 (同样无注意力)；
  * exceptions.* —— 异常体系与张量健康工具；
  * fx_passes / compile / benchmarks —— 图优化、编译封装与基准。
"""
from __future__ import annotations

from .config import SimplCoreConfig
from .exceptions import (
    SimplCoreError, ConfigError, ModelError, MemoryStateError, RolloutError,
    GradientCheckpointError, ModalityError, StackingError, LossGuardError,
    check_tensor_health, handle_gradient_clipping, format_exception,
)
from .layers import RMSNorm, FFN
from .resonance import ResonanceField
from .memory import DynamicMemoryChannels, MemoryState
from .routing import TopologyAwareRouter
from .localmix import LocalInteraction
from .block import SimplCoreBlock
from .model import SimplCoreModel
from .model_moe import SimplCoreMoEModel, SimplCoreMoEBlock
from .model_stream import SimplCoreStreamingModel
from .moe import MoELayer, ExpertFFN, SharedExpert
from .multimodal import ModalityRouter, SimplCoreViTEncoder
from .checkpointing import save_pretrained, from_pretrained, CheckpointMixin
from .compile import available_backends, compile_model
from . import fx_passes, compile, benchmarks, distributed

__all__ = [
    "SimplCoreConfig",
    "SimplCoreModel", "SimplCoreMoEModel", "SimplCoreStreamingModel",
    "SimplCoreBlock", "SimplCoreMoEBlock",
    "LocalInteraction", "ResonanceField", "DynamicMemoryChannels", "MemoryState",
    "TopologyAwareRouter",
    "MoELayer", "ExpertFFN", "SharedExpert",
    "ModalityRouter", "SimplCoreViTEncoder",
    "RMSNorm", "FFN",
    "save_pretrained", "from_pretrained", "CheckpointMixin",
    "available_backends", "compile_model",
    "SimplCoreError", "ConfigError", "ModelError", "MemoryStateError",
    "RolloutError", "GradientCheckpointError", "ModalityError",
    "StackingError", "LossGuardError",
    "check_tensor_health", "handle_gradient_clipping", "format_exception",
    "fx_passes", "compile", "benchmarks", "distributed",
]

__version__ = "1.0.0"
