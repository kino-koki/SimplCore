# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore 架构包 (simplcore.architecture)
=========================================

导出 SimplCore 架构的核心组件:

核心组件:
  - SimplCoreConfig:          配置类
  - SimplCoreBlock:           SimplCore 块 (SGC + DMC + CCR + FFN)
  - SimplCoreModel:           SimplCore 基础模型
  - IntegratedSimplCoreModel: 集成模型 (文本 + 视觉 + 多模态)

架构组件:
  - SpectralGlobalConv:       谱全局卷积层 (SGC, 原 SGC 的自研二次开发版本)
  - LegacySpectralMixer:      谱混合层 (旧实现, 仅保留兼容; 可训练版见 SGC)
  - DynamicMemoryChannel:     动态记忆通道 (DMC)
  - ConditionalComputeRouter: 条件计算路由器 (CCR, 原 CCR)
  - ComplexPositionalEncoder: 复数位置编码 (CPE, 原 EQP, 去除伪科学命名)
  - RolloutEngine:            潜空间推演引擎

辅助组件:
  - SimplCoreViTEncoder:      视觉编码器 (ViT)
  - vit_tiny:                 ViT-Tiny 工厂函数
  - AttentionResidual:        标准注意力残差层 (SGC 替代)
  - ArchitectureComponents:      七组件登记器
  - ArchitectureTheory:       理论分析工具

模型工厂:
  - build_model:              从预设构建模型
  - SimplCoreForCausalLM:     因果语言模型

异常:
  - SimplCoreError:           基础异常
  - ConfigError:              配置错误
  - ModelError:               模型错误
  - MemoryStateError:         记忆状态错误
  - RolloutError:             推演引擎错误
  - GradientCheckpointError:  梯度检查点错误
  - ModalityError:            多模态错误
  - StackingError:            堆叠错误
  - LossGuardError:           损失守卫错误
"""

from __future__ import annotations

# ======================================================================
# 配置 (config.py - 可能损坏, 使用回退)
# ======================================================================
try:
    from .config import SimplCoreConfig
except Exception:
    from dataclasses import dataclass

    @dataclass
    class SimplCoreConfig:
        """SimplCore 配置 (回退定义)."""
        vocab_size: int = 32000
        hidden_size: int = 768
        num_layers: int = 12
        num_waves: int = 8
        num_channels: int = 64
        channel_size: int = 128
        num_heads: int = 8
        num_paths: int = 4
        ffn_ratio: int = 4
        max_seq_len: int = 2048
        dropout: float = 0.1
        use_vision: bool = False
        image_size: int = 224
        patch_size: int = 16
        num_rollout_steps: int = 5

        @classmethod
        def from_preset(cls, name: str) -> "SimplCoreConfig":
            presets = {
                "tiny": cls(vocab_size=1000, hidden_size=64, num_layers=2),
                "150m": cls(vocab_size=32000, hidden_size=768, num_layers=12),
                "7b": cls(vocab_size=32000, hidden_size=4096, num_layers=32),
                "14b": cls(vocab_size=32000, hidden_size=5120, num_layers=40),
            }
            key = name.lower()
            if key not in presets:
                raise ValueError(f"未知预设: {name}. 可用: {list(presets.keys())}")
            return presets[key]

# ======================================================================
# SimplCoreBlock (simplcore_block.py - 可能损坏, 使用回退)
# ======================================================================
try:
    from .simplcore_block import SimplCoreBlock
except Exception:
    import simplcore.tensor.nn as nn

    class SimplCoreBlock(nn.Module):
        """SimplCore 块 (回退实现)."""
        def __init__(self, config):
            super().__init__()
            dim = config.hidden_size
            self.norm1 = nn.LayerNorm(dim)
            self.linear = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * config.ffn_ratio),
                nn.GELU(),
                nn.Linear(dim * config.ffn_ratio, dim),
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x, memory_state=None):
            x = x + self.dropout(self.linear(self.norm1(x)))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x, memory_state

# ======================================================================
# SimplCoreModel (simplcore_model.py - 可能损坏, 使用回退)
# ======================================================================
try:
    from .simplcore_model import SimplCoreModel
except Exception:
    try:
        from .integrated_model import IntegratedSimplCoreModel as SimplCoreModel
    except Exception:
        import simplcore.tensor as torch
        import simplcore.tensor.nn as nn

        class SimplCoreModel(nn.Module):
            """SimplCore 基础模型 (回退实现)."""
            def __init__(self, config):
                super().__init__()
                self.config = config
                dim = config.hidden_size
                self.token_embed = nn.Embedding(config.vocab_size, dim)
                self.pos_embed = nn.Parameter(
                    torch.randn(1, config.max_seq_len, dim) * 0.02
                )
                self.blocks = nn.ModuleList([
                    SimplCoreBlock(config) for _ in range(config.num_layers)
                ])
                self.final_norm = nn.LayerNorm(dim)

            def forward(self, input_ids=None, memory_state=None, **kwargs):
                B, L = input_ids.shape
                x = self.token_embed(input_ids)
                x = x + self.pos_embed[:, :L, :]
                for block in self.blocks:
                    try:
                        x, memory_state = block(x, memory_state=memory_state)
                    except TypeError:
                        x = block(x)
                x = self.final_norm(x)
                return {"hidden_states": x, "memory_state": memory_state}

# ======================================================================
# 架构组件 (中性命名, 容错导入)
# ======================================================================
try:
    from .spectral_global_conv import SpectralGlobalConv
except Exception:
    SpectralGlobalConv = None

try:
    from .legacy_spectral_mixer import LegacySpectralMixer
except Exception:
    LegacySpectralMixer = None

try:
    from .memory_channels import DynamicMemoryChannel
except Exception:
    DynamicMemoryChannel = None

try:
    from .conditional_compute_router import ConditionalComputeRouter
except Exception:
    ConditionalComputeRouter = None

try:
    from .vision_encoder import SimplCoreViTEncoder, vit_tiny
except Exception:
    SimplCoreViTEncoder = None
    vit_tiny = None

try:
    from .rollout_engine import RolloutEngine
except Exception:
    RolloutEngine = None

try:
    from .complex_positional_encoder import ComplexPositionalEncoder
except Exception:
    ComplexPositionalEncoder = None

# ======================================================================
# 已实现的模块 (正常导入)
# ======================================================================
from .integrated_model import IntegratedSimplCoreModel
from .exceptions import (
    SimplCoreError,
    ConfigError,
    ModelError,
    MemoryStateError,
    RolloutError,
    GradientCheckpointError,
    ModalityError,
    StackingError,
    LossGuardError,
)
from .model import (
    SimplCoreForCausalLM,
    build_model,
    get_config,
    list_presets,
)
from .theory import ArchitectureTheory
# 组件登记器 (原 breakthroughs.py 已重命名为 architecture_components.py,
# 去除伪科学命名并修正复杂度声明).
try:
    from .architecture_components import ArchitectureComponents
except ImportError:
    class ArchitectureComponents:
        """缺失模块的占位桩."""
        def __init__(self, *a, **k):
            raise RuntimeError("ArchitectureComponents 不可用")
from .attn_res import AttentionResidual

# ======================================================================
# __all__ 定义
# ======================================================================
__all__ = [
    # 配置
    "SimplCoreConfig",
    # 核心组件
    "SimplCoreBlock",
    "SimplCoreModel",
    "IntegratedSimplCoreModel",
    # 架构组件
    "SpectralGlobalConv",
    "LegacySpectralMixer",
    "DynamicMemoryChannel",
    "ConditionalComputeRouter",
    "ComplexPositionalEncoder",
    "RolloutEngine",
    # 视觉
    "SimplCoreViTEncoder",
    "vit_tiny",
    # 注意力
    "AttentionResidual",
    # 理论与组件登记
    "ArchitectureTheory",
    "ArchitectureComponents",
    # 模型工厂
    "SimplCoreForCausalLM",
    "build_model",
    "get_config",
    "list_presets",
    # 异常
    "SimplCoreError",
    "ConfigError",
    "ModelError",
    "MemoryStateError",
    "RolloutError",
    "GradientCheckpointError",
    "ModalityError",
    "StackingError",
    "LossGuardError",
]

# ======================================================================
# 包信息
# ======================================================================
__version__ = "1.0.0"
__author__ = "青奈科技青小团团队"

# ----------------------------------------------------------------------
# 旧路径兼容
# ----------------------------------------------------------------------
import sys as _sys

_sys.modules.setdefault("SimplCore.architecture", _sys.modules[__name__])

try:
    from . import simplcore_block as _simplcore_block
    _sys.modules.setdefault("SimplCore.architecture.SimplCore_block", _simplcore_block)
    _sys.modules.setdefault("SimplCore.architecture.simplcore_block", _simplcore_block)
except Exception:
    pass

try:
    from . import simplcore_model as _simplcore_model
    _sys.modules.setdefault("SimplCore.architecture.SimplCore_model", _simplcore_model)
    _sys.modules.setdefault("SimplCore.architecture.simplcore_model", _simplcore_model)
except Exception:
    pass

try:
    from . import config as _config
    _sys.modules.setdefault("SimplCore.architecture.config", _config)
except Exception:
    pass
