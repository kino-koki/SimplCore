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
from .config import (
    SimplCoreMixConfig,
    mix_tiny_text,
    mix_150m_text,
    mix_7b_text,
    mix_14b_text,
    mix_1b,
    mix_7b,
    mix_33b,
)
from .core import (
    SCBackend,
    SCDropout,
    SCEmbedding,
    SCFunctional,
    SCLinear,
    SCModule,
    SCModuleList,
    SCParameter,
    SCReLU,
    SCRMSNorm,
    SCSequential,
    SCSigmoid,
)
from .model import SimplCoreMix
from .baseline import BaselineTransformer
from .ablation import AblationConfig, AblationStudy, run_ablation_smoke
from .mska import MultiScaleKernelAttention
from .fusion import (
    StructuredGatedIntervention,
    SimplCoreGatedIntervention,
    GatedCompression,
    LowRankProjection,
    ResidualAdaptationModule,
)
from .frontier import SCMilestoneSpec
from .runtime import SCRuntimeProfile
from .readiness import ReadinessCheck, build_7b_readiness_report, check_python311
from .calibration import build_calibration_plan
from .optim import SCAdamW, SCOptimizer
from .blocks import (
    PreludeBlock,
    RecurrentBlock,
    CodaBlock,
    LTIInjection,
    LoRAAdapter,
    ACTHalting,
    loop_index_embedding,
)
from .attention import ModulatedBasisAttention, ModulatedBasisMLAttention
from .moe import PrototypeMoE, Expert
from .memory import WorkingMemoryBuffer
from .tokenizer import BPETokenizer, CharTokenizer
from .data_loader import (
    SCDataLoader,
    TextDataset,
    MemoryMappedDataset,
    StreamingTextDataset,
    PretokenizedDataset,
    build_data_loader,
    pretokenize_corpus,
)
from .evaluation import Evaluator, FactualEvaluator
from .inference import InferenceEngine
from .export import (
    export_pytorch_checkpoint,
    export_safetensors,
    export_raw_weights,
    export_for_java,
    export_java26_manifest,
    export_jdk24_manifest,
    export_onnx,
)

__version__ = "1.0.0"

__all__ = [
    # 配置
    "SimplCoreMixConfig",
    "mix_tiny_text",
    "mix_150m_text",
    "mix_7b_text",
    "mix_14b_text",
    "mix_1b",
    "mix_7b",
    "mix_33b",
    # 自研核心抽象
    "SCBackend",
    "SCDropout",
    "SCEmbedding",
    "SCFunctional",
    "SCLinear",
    "SCModule",
    "SCModuleList",
    "SCParameter",
    "SCReLU",
    "SCRMSNorm",
    "SCSequential",
    "SCSigmoid",
    # SimplCore深度融合组件
    "SimplCoreGatedIntervention",
    "StructuredGatedIntervention",
    "GatedCompression",
    "LowRankProjection",
    "ResidualAdaptationModule",
    "SCMilestoneSpec",
    # 跨运行时与优化器
    "SCRuntimeProfile",
    "ReadinessCheck",
    "build_7b_readiness_report",
    "check_python311",
    "build_calibration_plan",
    "SCOptimizer",
    "SCAdamW",
    # 模型
    "SimplCoreMix",
    "BaselineTransformer",
    # 消融实验
    "AblationConfig",
    "AblationStudy",
    "run_ablation_smoke",
    # 核心Block (三家融合)
    "PreludeBlock",
    "RecurrentBlock",
    "CodaBlock",
    "LTIInjection",
    "LoRAAdapter",
    "ACTHalting",
    "loop_index_embedding",
    # 注意力 (SimplCore MBA / MSKA)
    "ModulatedBasisAttention",
    "ModulatedBasisMLAttention",
    "MultiScaleKernelAttention",
    # MoE (SimplCore CCR)
    "PrototypeMoE",
    "Expert",
    # 记忆 (SimplCore DMC)
    "WorkingMemoryBuffer",
    # 分词器
    "BPETokenizer",
    "CharTokenizer",
    # 数据
    "SCDataLoader",
    "TextDataset",
    "MemoryMappedDataset",
    "StreamingTextDataset",
    "PretokenizedDataset",
    "build_data_loader",
    "pretokenize_corpus",
    # 评估
    "Evaluator",
    "FactualEvaluator",
    # 推理
    "InferenceEngine",
    # 导出
    "export_pytorch_checkpoint",
    "export_safetensors",
    "export_raw_weights",
    "export_for_java",
    "export_java26_manifest",
    "export_jdk24_manifest",
    "export_onnx",
]
