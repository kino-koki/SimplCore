# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""多模态感知融合系统 (Multimodal Perception Fusion).

核心功能:
1. 多模态数据处理（视觉、音频、文本）
2. 模态间注意力融合
3. 感知状态管理
4. 环境上下文建模

与现有模块的集成:
- 扩展PerceptionPipeline支持多模态
- 为cognition模块提供多模态输入
- 为dialogue模块提供环境上下文
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

from ..engine.tensor import SimpTensor, randn, zeros, softmax, sigmoid_tensor, tanh, gelu
from ..engine.nn import Module, Linear, LayerNorm


class ModalityType(Enum):
    """模态类型"""
    VISION = "vision"
    AUDIO = "audio"
    TEXT = "text"
    TACTILE = "tactile"
    ENVIRONMENT = "environment"


class PerceptionState(Enum):
    """感知状态"""
    DETECTING = "detecting"
    RECOGNIZING = "recognizing"
    UNDERSTANDING = "understanding"
    INTEGRATING = "integrating"


class MultimodalEncoder(Module):
    """多模态编码器"""
    
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        
        self.vision_proj = Linear(dim, dim)
        self.register_module("vision_proj", self.vision_proj)
        
        self.audio_proj = Linear(dim, dim)
        self.register_module("audio_proj", self.audio_proj)
        
        self.text_proj = Linear(dim, dim)
        self.register_module("text_proj", self.text_proj)
        
        self.tactile_proj = Linear(dim, dim)
        self.register_module("tactile_proj", self.tactile_proj)
        
        self.env_proj = Linear(dim, dim)
        self.register_module("env_proj", self.env_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
    
    def encode_vision(self, visual_feature: SimpTensor) -> SimpTensor:
        """编码视觉特征"""
        return tanh(self.vision_proj(visual_feature))
    
    def encode_audio(self, audio_feature: SimpTensor) -> SimpTensor:
        """编码音频特征"""
        return tanh(self.audio_proj(audio_feature))
    
    def encode_text(self, text_feature: SimpTensor) -> SimpTensor:
        """编码文本特征"""
        return tanh(self.text_proj(text_feature))
    
    def encode_tactile(self, tactile_feature: SimpTensor) -> SimpTensor:
        """编码触觉特征"""
        return tanh(self.tactile_proj(tactile_feature))
    
    def encode_environment(self, env_feature: SimpTensor) -> SimpTensor:
        """编码环境特征"""
        return tanh(self.env_proj(env_feature))


class ModalityAttention(Module):
    """模态间注意力融合"""
    
    def __init__(self, dim: int = 64, num_modalities: int = 5):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        
        self.query_proj = Linear(dim, dim)
        self.register_module("query_proj", self.query_proj)
        
        self.key_proj = Linear(dim, dim)
        self.register_module("key_proj", self.key_proj)
        
        self.value_proj = Linear(dim, dim)
        self.register_module("value_proj", self.value_proj)
        
        self.output_proj = Linear(dim, dim)
        self.register_module("output_proj", self.output_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
    
    def forward(
        self,
        features: List[SimpTensor],
        modality_types: List[ModalityType]
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """模态注意力融合"""
        if not features:
            return zeros(self.dim), {"attention_weights": [], "num_modalities": 0}
        
        batch_size = len(features)
        
        keys = []
        values = []
        for feat in features:
            keys.append(tanh(self.key_proj(feat))._data)
            values.append(tanh(self.value_proj(feat))._data)
        
        keys_np = np.stack(keys, axis=0)
        values_np = np.stack(values, axis=0)
        
        query = tanh(self.query_proj(features[0]))._data
        
        scores = np.dot(query, keys_np.T) / np.sqrt(self.dim)
        attention_weights = softmax(SimpTensor(scores))._data
        
        weighted_sum = np.dot(attention_weights, values_np)
        output = gelu(self.output_proj(SimpTensor(weighted_sum)))
        output = self.norm(output)
        
        modality_weights = {}
        for i, modality in enumerate(modality_types):
            modality_weights[modality.value] = float(attention_weights[i])
        
        info = {
            "attention_weights": [round(float(w), 4) for w in attention_weights],
            "modality_weights": modality_weights,
            "num_modalities": batch_size,
            "dominant_modality": modality_types[np.argmax(attention_weights)].value,
        }
        
        return output, info


class PerceptionMemory(Module):
    """感知记忆"""
    
    def __init__(self, dim: int = 64, max_frames: int = 100):
        super().__init__()
        self.dim = dim
        self.max_frames = max_frames
        
        self.frames: List[Dict[str, Any]] = []
        self.frame_proj = Linear(dim, dim)
        self.register_module("frame_proj", self.frame_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
    
    def add_frame(self, fused_feature: SimpTensor, modality_info: Dict[str, Any]):
        """添加感知帧"""
        frame = {
            "feature": fused_feature,
            "modality_info": modality_info,
            "timestamp": np.datetime64('now'),
        }
        self.frames.append(frame)
        
        if len(self.frames) > self.max_frames:
            self.frames = self.frames[-self.max_frames:]
    
    def get_recent_frames(self, num_frames: int = 10) -> List[Dict[str, Any]]:
        """获取最近帧"""
        return self.frames[-num_frames:]
    
    def get_temporal_feature(self, num_frames: int = 5) -> SimpTensor:
        """获取时序特征"""
        recent = self.get_recent_frames(num_frames)
        if not recent:
            return zeros(self.dim)
        
        features = [frame["feature"]._data for frame in recent]
        stacked = np.stack(features, axis=0)
        
        temporal = np.mean(stacked, axis=0)
        output = tanh(self.frame_proj(SimpTensor(temporal)))
        output = self.norm(output)
        
        return output
    
    def reset(self):
        """重置记忆"""
        self.frames = []


class MultimodalPerception(Module):
    """多模态感知系统"""
    
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        
        self.encoder = MultimodalEncoder(dim)
        self.register_module("encoder", self.encoder)
        
        self.attention = ModalityAttention(dim)
        self.register_module("attention", self.attention)
        
        self.memory = PerceptionMemory(dim)
        self.register_module("memory", self.memory)
        
        self.output_proj = Linear(dim * 2, dim)
        self.register_module("output_proj", self.output_proj)
        
        self.state: PerceptionState = PerceptionState.DETECTING
        
        self._active_modalities: List[ModalityType] = []
        self._modality_features: Dict[str, SimpTensor] = {}
    
    def add_modality(
        self,
        modality_type: ModalityType,
        feature: SimpTensor
    ):
        """添加模态特征"""
        self._active_modalities.append(modality_type)
        
        if modality_type == ModalityType.VISION:
            encoded = self.encoder.encode_vision(feature)
        elif modality_type == ModalityType.AUDIO:
            encoded = self.encoder.encode_audio(feature)
        elif modality_type == ModalityType.TEXT:
            encoded = self.encoder.encode_text(feature)
        elif modality_type == ModalityType.TACTILE:
            encoded = self.encoder.encode_tactile(feature)
        elif modality_type == ModalityType.ENVIRONMENT:
            encoded = self.encoder.encode_environment(feature)
        else:
            encoded = feature
        
        self._modality_features[modality_type.value] = encoded
    
    def clear_modalities(self):
        """清除模态特征"""
        self._active_modalities = []
        self._modality_features = {}
    
    def fuse(self) -> Tuple[SimpTensor, Dict[str, Any]]:
        """融合多模态特征"""
        if not self._modality_features:
            return zeros(self.dim), {"state": self.state.value, "num_modalities": 0}
        
        features = list(self._modality_features.values())
        modalities = self._active_modalities
        
        fused, attention_info = self.attention(features, modalities)
        
        self.memory.add_frame(fused, attention_info)
        
        temporal_feature = self.memory.get_temporal_feature()
        
        combined = SimpTensor(np.concatenate([fused._data, temporal_feature._data]))
        output = gelu(self.output_proj(combined))
        
        self.state = PerceptionState.INTEGRATING
        
        info = {
            "state": self.state.value,
            "num_modalities": len(self._active_modalities),
            "active_modalities": [m.value for m in self._active_modalities],
            "attention_weights": attention_info["attention_weights"],
            "modality_weights": attention_info["modality_weights"],
            "dominant_modality": attention_info["dominant_modality"],
            "memory_frames": len(self.memory.frames),
        }
        
        return output, info
    
    def forward(
        self,
        modalities: Dict[ModalityType, SimpTensor]
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """前向传播"""
        self.clear_modalities()
        
        for modality_type, feature in modalities.items():
            self.add_modality(modality_type, feature)
        
        return self.fuse()
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "state": self.state.value,
            "active_modalities": [m.value for m in self._active_modalities],
            "memory_frames": len(self.memory.frames),
        }
    
    def reset(self):
        """重置"""
        self.clear_modalities()
        self.memory.reset()
        self.state = PerceptionState.DETECTING


class AudioPerception(Module):
    """音频感知模块"""
    
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        
        self.spectrogram_proj = Linear(dim, dim)
        self.register_module("spectrogram_proj", self.spectrogram_proj)
        
        self.pitch_proj = Linear(dim, dim)
        self.register_module("pitch_proj", self.pitch_proj)
        
        self.energy_proj = Linear(dim, dim)
        self.register_module("energy_proj", self.energy_proj)
        
        self.output_proj = Linear(dim * 3, dim)
        self.register_module("output_proj", self.output_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
    
    def process_audio(self, audio_data: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理音频数据"""
        spectrogram = tanh(self.spectrogram_proj(audio_data))
        pitch = tanh(self.pitch_proj(audio_data))
        energy = tanh(self.energy_proj(audio_data))
        
        combined = SimpTensor(np.concatenate([spectrogram._data, pitch._data, energy._data]))
        output = gelu(self.output_proj(combined))
        output = self.norm(output)
        
        info = {
            "audio_feature_dim": audio_data.shape,
            "has_audio": True,
        }
        
        return output, info
    
    def forward(self, audio_data: SimpTensor) -> Tuple[SimpTensor, Dict[str, Any]]:
        return self.process_audio(audio_data)


class EnvironmentPerception(Module):
    """环境感知模块"""
    
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        
        self.context_proj = Linear(dim, dim)
        self.register_module("context_proj", self.context_proj)
        
        self.temperature_proj = Linear(1, dim)
        self.register_module("temperature_proj", self.temperature_proj)
        
        self.light_proj = Linear(1, dim)
        self.register_module("light_proj", self.light_proj)
        
        self.output_proj = Linear(dim * 3, dim)
        self.register_module("output_proj", self.output_proj)
        
        self.norm = LayerNorm(dim)
        self.register_module("norm", self.norm)
    
    def process_environment(
        self,
        context_feature: SimpTensor,
        temperature: float = 25.0,
        light_level: float = 1.0
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        """处理环境数据"""
        context = tanh(self.context_proj(context_feature))
        
        temp_feature = tanh(self.temperature_proj(SimpTensor(np.array([temperature]))))
        light_feature = tanh(self.light_proj(SimpTensor(np.array([light_level]))))
        
        combined = SimpTensor(np.concatenate([context._data, temp_feature._data, light_feature._data]))
        output = gelu(self.output_proj(combined))
        output = self.norm(output)
        
        info = {
            "temperature": temperature,
            "light_level": light_level,
            "has_environment": True,
        }
        
        return output, info
    
    def forward(
        self,
        context_feature: SimpTensor,
        temperature: float = 25.0,
        light_level: float = 1.0
    ) -> Tuple[SimpTensor, Dict[str, Any]]:
        return self.process_environment(context_feature, temperature, light_level)


if __name__ == "__main__":
    dim = 64
    print("=" * 60)
    print("多模态感知融合系统测试")
    print("=" * 60)
    
    print("\n1. 测试多模态编码器:")
    encoder = MultimodalEncoder(dim)
    
    visual_feat = randn(dim)
    audio_feat = randn(dim)
    text_feat = randn(dim)
    
    encoded_vision = encoder.encode_vision(visual_feat)
    encoded_audio = encoder.encode_audio(audio_feat)
    encoded_text = encoder.encode_text(text_feat)
    
    print(f"  视觉编码: {encoded_vision.shape}")
    print(f"  音频编码: {encoded_audio.shape}")
    print(f"  文本编码: {encoded_text.shape}")
    
    print("\n2. 测试模态注意力:")
    attention = ModalityAttention(dim)
    
    features = [encoded_vision, encoded_audio, encoded_text]
    modalities = [ModalityType.VISION, ModalityType.AUDIO, ModalityType.TEXT]
    
    fused, info = attention(features, modalities)
    print(f"  融合输出: {fused.shape}")
    print(f"  注意力权重: {info['attention_weights']}")
    print(f"  主导模态: {info['dominant_modality']}")
    
    print("\n3. 测试多模态感知系统:")
    perception = MultimodalPerception(dim)
    
    modalities_input = {
        ModalityType.VISION: visual_feat,
        ModalityType.AUDIO: audio_feat,
        ModalityType.TEXT: text_feat,
    }
    
    output, info = perception(modalities_input)
    print(f"  输出形状: {output.shape}")
    print(f"  状态: {info['state']}")
    print(f"  活动模态: {info['active_modalities']}")
    print(f"  记忆帧数: {info['memory_frames']}")
    
    print("\n4. 测试音频感知:")
    audio_perception = AudioPerception(dim)
    audio_data = randn(dim)
    audio_output, audio_info = audio_perception(audio_data)
    print(f"  输出形状: {audio_output.shape}")
    print(f"  音频信息: {audio_info}")
    
    print("\n5. 测试环境感知:")
    env_perception = EnvironmentPerception(dim)
    env_output, env_info = env_perception(randn(dim), temperature=22.5, light_level=0.7)
    print(f"  输出形状: {env_output.shape}")
    print(f"  环境信息: {env_info}")
    
    print("\n" + "=" * 60)
    print("所有测试 PASSED!")
    print("=" * 60)
