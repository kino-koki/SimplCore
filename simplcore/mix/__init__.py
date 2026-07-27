# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""
SimplCore Mix - 融合架构
整合 SimplCore Java组件、SimplCore Python引擎、OpenMythos RDT

模块:
- config: 配置系统
- model: 主模型架构
- tokenizer: 自研BPE分词器
- data_loader: 数据加载器
- training: 训练脚本
- evaluation: 评估系统
- inference: 推理引擎
- export: 模型导出
"""
from .config import SimplCoreMixConfig, mix_tiny_text, mix_150m_text, mix_1b, mix_7b, mix_33b
from .model import SimplCoreMix
from .blocks import (
    PreludeBlock,
    RecurrentBlock,
    CodaBlock,
    LTIInjection,
    LoRAAdapter,
    ACTHalting,
    loop_index_embedding,
)
from .attention import RFPAttention, RFPMLAttention
from .moe import TARMoE, Expert
from .memory import DynamicMemoryChannel
from .tokenizer import BPETokenizer, CharTokenizer
from .data_loader import (
    TextDataset,
    MemoryMappedDataset,
    StreamingTextDataset,
    PretokenizedDataset,
    build_data_loader,
    pretokenize_corpus,
)
from .evaluation import Evaluator
from .inference import InferenceEngine
from .export import (
    export_pytorch_checkpoint,
    export_safetensors,
    export_raw_weights,
    export_for_java,
    export_onnx,
)

__version__ = "1.0.0"

__all__ = [
    # 配置
    "SimplCoreMixConfig",
    "mix_tiny_text",
    "mix_150m_text",
    "mix_1b",
    "mix_7b",
    "mix_33b",
    # 模型
    "SimplCoreMix",
    # 核心Block (三家融合)
    "PreludeBlock",
    "RecurrentBlock",
    "CodaBlock",
    "LTIInjection",
    "LoRAAdapter",
    "ACTHalting",
    "loop_index_embedding",
    # 注意力 (SimplCore RFP)
    "RFPAttention",
    "RFPMLAttention",
    # MoE (SimplCore TAR)
    "TARMoE",
    "Expert",
    # 记忆 (SimplCore DMC)
    "DynamicMemoryChannel",
    # 分词器
    "BPETokenizer",
    "CharTokenizer",
    # 数据
    "TextDataset",
    "MemoryMappedDataset",
    "StreamingTextDataset",
    "PretokenizedDataset",
    "build_data_loader",
    "pretokenize_corpus",
    # 评估
    "Evaluator",
    # 推理
    "InferenceEngine",
    # 导出
    "export_pytorch_checkpoint",
    "export_safetensors",
    "export_raw_weights",
    "export_for_java",
    "export_onnx",
]
