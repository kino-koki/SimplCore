# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore —— 非注意力主干架构 (non-attention backbone).

本包是 **SimplCore** 的正式生产实现: 一个彻底脱离 Transformer/softmax 自注意力
的序列建模架构。SGC(谱全局卷积, ``simplcore.core.sgc``)只是 SimplCore 内部的一个
mixer 模块, 不是整个架构的名字 —— 架构叫 **SimplCore**, 它下面可以有多种 mixer
(SGC / SRFN / GatedSpectralConv2), 全部都是非注意力的。

为什么不是 Transformer (核心论点, 详见 `docs/THEORETICAL_EDGE.md`)
-------------------------------------------------------------------
1. **O(1) 常驻流式状态**, 不随序列长度 L 增长的 KV cache。
   位置路径(long-range)用 **M 个阻尼振荡基** 参数化因果全局卷积核:

       psi_m[tau] = rho_m^tau * sin((tau + 1) * theta_m),      tau >= 0
       K[tau, d]  = w_d * 1{tau = 0} + sum_m C[m, d] * psi_m[tau]

   位置由因果卷积核本身编码, **没有位置嵌入表**。

2. **对偶形式 (dual form, bit-exact)**:
     * 并行/训练形式: y[t] = sum_{s<=t} K[t-s] * u[s], 用 FFT 计算, O(L log L);
     * 流式/推理形式: 二阶 IIR 递推(每个基 2 个状态)
           y_m[t] = a_m * y_m[t-1] + b_m * y_m[t-2] + sin(theta_m) * u[t]
           a_m = 2 rho_m cos(theta_m),  b_m = -rho_m^2
       常驻状态 2*M*D 个浮点, **与序列长度 L 无关**, 推理为 O(1)/token。
   两者在 fp32 下数值一致(实测 max abs err < 4e-5), 见 `tests/test_mixer_math.py`。

3. **内容路由 (selective)**: 引入输入相关门控 `g = sigmoid(W_g * norm(x))`,
   作用在卷积输入上: `u = W_in * norm(x) * g`。有效核随内容变化
   (GLA/GSS 家族做法), 直接针对"位置相对核无法按内容路由"的瓶颈,
   且不引入任何 Q/K/V softmax 注意力、不改变 O(L log L) / O(1) 的复杂度阶。

SimplCore 相对传统 Transformer 的理论优势: 无 KV cache 线性膨胀、无位置嵌入表、
FFT O(L log L) 训练 / IIR O(1) 推理双形态、对偶形式 bit-exact 等价。
数值后端: 纯 PyTorch (autograd, 支持 bf16/fp16 autocast, 可 device-agnostic)。
"""
from __future__ import annotations

from ._compat_aliases import install as _install_compat_aliases
from ._compat_aliases import lab_getattr as _lab_getattr
_install_compat_aliases()

from .model.config import (
    SimplCoreConfig,
    SIMPLCORE_PRESETS,
    get_preset,
    MODEL_NAME,
    MODEL_CODENAME,
)
from .arch.sgc import SpectralGlobalConv, SGCStreamState
from .arch.snn_sgc import SpikingSpectralGlobalConv
# SRFN（谱振流网络）：SGC 的 SSM 化版本。mixer="srf" 时启用。
from .arch.srf import SpectralResonanceFlow
from .arch.ssm import SelectiveSpectralSSM, SSMState
from .memory.memory import (
    CompressedMemoryBank,
    MemoryAwareObjective,
    MemoryCompressor,
    MemoryItem,
    TextKeyEncoder,
    TrainableMemoryCompressor,
    memory_aware_loss,
    self_match_scores,
    top_aggregate,
    train_compressor_step,
)
from .dialogue.chat import (
    SPECIAL_TOKENS,
    ChatTemplate,
    SFTDataset,
    SpecialTokenCodec,
    build_memory_encoder,
    build_sft_dataset,
    collate_sft,
    load_memory_encoder,
    pack_sequences,
)
from .dialogue.dialogue_runtime import (
    ChatSession,
    DialogMemory,
    MemoryHit,
    Reply,
)
from .training.sft import (
    SFTConfig,
    SFTResult,
    build_optimizer,
    evaluate_loss,
    fairness_report,
    train_sft,
)
from .training.ab import (
    PairedOutcome,
    ab_report,
    bootstrap_diff_ci,
    mcnemar,
    param_efficiency,
)
from .dialogue.dialogue_eval import (
    BASIC_DIALOGUE_SUITE,
    MEMORY_DIALOGUE_SUITE,
    Check,
    DialogueReport,
    DialogueTask,
    evaluate_dialogue,
    generate_reply,
    memory_ablation,
    memory_projection_report,
)
from .eval.scaling import (
    CostLedger,
    budget_table,
    cost_ledger,
    embedding_share,
    largest_config_for_flops,
    non_embedding_params,
    plan_budget,
    tokens_for_budget,
    train_flops_per_token,
)
from .memory.dream import (
    ReflectionConfig,
    ReflectionLoop,
    ReflectionReport,
    ReflectionStage,
    EvalScores,
    ExperienceBuffer,
    GateVerdict,
    Insight,
    TrainExample,
    Trajectory,
    default_consolidator,
    default_synthesizer,
    default_verifier,
    paired_bootstrap_ci,
    regression_gate,
)
from .arch.abstain import (
    AbstentionHead,
    AbstentionState,
    AbstentionDecision,
    GroundedPolicy,
    GroundedResult,
    GroundedAnswerer,
    abstention_loss,
    grounded_probability,
    should_abstain,
    calibrate_temperature,
    evidence_aware_decision,
    fit_temperature,
    select_threshold,
    expected_calibration_error,
)
from .arch.grounding import (
    Claim,
    Binding,
    EvidenceIndex,
    EvidenceSpan,
    attribution_coverage,
    bind_claim,
    bind_claims,
    bigram_coverage_of,
    coverage_of,
    grounded_labels,
    numbers_supported,
    polarity_matches,
    split_claims,
    unsupported_claims,
)
from .model.model import SimplCoreBlock, SimplCoreModel, SimplCoreForCausalLM
from .memory.sidecar import (BlockGistIndex, PointerHead, SidecarArchive,
                             SidecarRetriever, sidecar_ledger)
from .model.multimodal import (AudioAdapter, ImageAdapter,
                               SimplCoreMultimodal)
from .model.physics_world import (PhysicsConstraints, PhysicsEvent,
                                  PhysicsWorldModel)
from .eval.world_probe import WorldModelProbe, WorldProbeReport
from .arch.selective import GatedSpectralConv2, analytic_gated_spectral_params
from .eval.claims_audit import Check, format_table, run_all
from .model.hf_bridge import (
    MigrationReport,
    init_simplcore_from_hf,
    migrate_embeddings_and_ffn,
    sgc_to_hf_eval_bridge,
    simplcore_stream_state_for_eval,
)
from .model.tokenization import (
    Tokenizer,
    CharTokenizer,
    ByteTokenizer,
    BPETokenizer,
    make_codec,
)
from .arch.accel import AdaptiveDepth, ComputeLedger
from .dialogue.reasoning import IterativeRefiner, ReasoningTrace, stabilized
from .model.nonlinear import (
    LatentRefiner,
    LatentRefineCell,
    HaltingHead,
    RefineLedger,
)
from .memory.smm import (
    SimplModelMemory,
    SMMConfig,
    SMMLedger,
    MemoryTier,
)
from .training.envlearn import (
    WeightBudget,
    AdapterBlock,
    LearnedAdapter,
    ForwardAdapter,
    EnvTrajectory,
    RolloutResult,
    Environment,
    LocalCodeEnvironment,
    StageReport,
    EnvironmentLearner,
    summarize_failures,
)

from .eval.extrapolation import (
    extrapolation_probe,
    decay_ablation,
    sgc_model_factory,
    ExtrapolationResult,
    DecayPoint,
)
from .backend.backend_contract import (
    BackendContract,
    BackendOp,
    ConformanceCheck,
    DEFAULT_CONTRACT,
    SIMPLCORE_BACKEND_OPS,
    conformance_checks,
)
from .backend.conformance_runner import (
    BackendAdapter,
    ConformanceReport,
    ConformanceResult,
    TorchReferenceAdapter,
    run_conformance,
)
from .model.adapter_model import (
    AdapterModel,
    RecordingAdapter,
)
from .training.trainer import (
    ContextCurriculum,
    CurriculumStage,
    StreamingTrainer,
    TrainLedger,
)
from .eval.equal_budget import (
    SyntheticWindowSource,
    TSConfig,
    TransformerBaseline,
    WindowStream,
    match_ts_layers,
    paired_eval,
    run_comparison,
    ts_train_flops_per_token,
)
from .eval.longctx_eval import (
    NeedleCorpus,
    NeedleTrial,
    run_longctx_eval,
    score_trial,
)
from .eval.code_passk_eval import (
    GeneratorSpec,
    make_problems_small,
    pass_at_k,
    run_code_passk,
)
from .eval.throughput_eval import (
    TSDecoder,
    analytic_decode_ledger,
    fit_slope,
    run_throughput_eval,
    summarize_latency,
)
from .training.phase1_train import (
    CorpusBatches,
    SyntheticCorpusSpec,
    make_synthetic_corpus,
    run_phase1,
)
from .backend.backend_bridge import (
    ABI_SPEC,
    CtypesBackendAdapter,
    FakeCLib,
    load_backend,
    symbol_report,
)
from .eval.paper_audit import run_audit
from .deploy.deploy_budget import (
    DeployRow,
    DEPLOY_LADDER,
    DEPLOY_HBM_CAP_1M,
    DEPLOY_DOUBLING_CAP,
    deploy_bytes,
    deployment_report,
    deploy_verdict,
)
from .deploy.campaign import (
    CampaignPlan,
    DoublingRow,
    Gate,
    Phase,
    ShowcaseItem,
    CONTEXT_LADDER,
    HBM_MI250_BYTES,
    PHASE0_MAX_DAYS,
    PHASE1_MAX_DAYS,
    PHASE2_MAX_DAYS,
    RETROSPECT_MAX_DAYS,
    TOTAL_BUDGET_DAYS,
    build_plan,
    claim_verdict,
    doubling_report,
)
from .eval.mathproof import (
    ProofResult,
    run_all as run_math_proofs,
)

__all__ = [
    "SimplCoreConfig",
    "SidecarArchive",
    "SidecarRetriever",
    "BlockGistIndex",
    "PointerHead",
    "sidecar_ledger",
    "SimplCoreMultimodal",
    "ImageAdapter",
    "AudioAdapter",
    "WorldModelProbe",
    "WorldProbeReport",
    "PhysicsWorldModel",
    "PhysicsEvent",
    "PhysicsConstraints",
    "SIMPLCORE_PRESETS",
    "get_preset",
    "MODEL_NAME",
    "MODEL_CODENAME",
    "SpectralGlobalConv",
    "SpikingSpectralGlobalConv",
    "SGCStreamState",
    # 谱振流网络（SRFN）：selective SSM + 谱核 + 推理 O(1) 状态
    "SelectiveSpectralSSM",
    "SSMState",
    "SpectralResonanceFlow",
    # 四层闸门（L2 证据门 / L3 不确定门在 mixer 内，L4 拒答门在输出侧）
    "AbstentionHead",
    "AbstentionState",
    "AbstentionDecision",
    "GroundedPolicy",
    "GroundedResult",
    "GroundedAnswerer",
    "abstention_loss",
    "grounded_probability",
    "should_abstain",
    "calibrate_temperature",
    "evidence_aware_decision",
    "fit_temperature",
    "select_threshold",
    "expected_calibration_error",
    # 证据绑定层（L4 拒答门的标签来源：逐字归因 + 数字/极性核对）
    "Claim",
    "Binding",
    "EvidenceIndex",
    "EvidenceSpan",
    "attribution_coverage",
    "bind_claim",
    "bind_claims",
    "bigram_coverage_of",
    "coverage_of",
    "grounded_labels",
    "numbers_supported",
    "polarity_matches",
    "split_claims",
    "unsupported_claims",
    # CMT 压缩记忆库（连续知识学习）
    "CompressedMemoryBank",
    "MemoryAwareObjective",
    "MemoryCompressor",
    "MemoryItem",
    "TextKeyEncoder",
    "TrainableMemoryCompressor",
    "memory_aware_loss",
    "train_compressor_step",
    "self_match_scores",
    "top_aggregate",
    # Dreaming 自进化循环
    "ReflectionConfig",
    "ReflectionLoop",
    "ReflectionReport",
    "ReflectionStage",
    # Dreaming 的默认钩子（自进化链的"真的会做事"的那一环）
    "default_consolidator",
    "default_synthesizer",
    "default_verifier",
    # 对话数据层（30M 对话能力的地基：特殊 token / 模板 / 掩码 / 打包）
    "SPECIAL_TOKENS",
    "ChatTemplate",
    "build_memory_encoder",
    "load_memory_encoder",
    "SFTDataset",
    "SpecialTokenCodec",
    "build_sft_dataset",
    "collate_sft",
    "pack_sequences",
    # 对话评测（规则可判，无评审模型）
    "BASIC_DIALOGUE_SUITE",
    "Check",
    "DialogueReport",
    "DialogueTask",
    "evaluate_dialogue",
    "generate_reply",
    "MEMORY_DIALOGUE_SUITE",
    "memory_ablation",
    "memory_projection_report",
    # 对话运行时：记忆 + 四态闸门串成一条可用管线
    "ChatSession",
    "DialogMemory",
    "MemoryHit",
    "Reply",
    # 可复用 SFT 训练器（A/B 两侧共用同一实现）
    "SFTConfig",
    "SFTResult",
    "build_optimizer",
    "evaluate_loss",
    "fairness_report",
    "train_sft",
    # A/B 统计（配对检验 + 参数效率）
    "PairedOutcome",
    "ab_report",
    "bootstrap_diff_ci",
    "mcnemar",
    "param_efficiency",
    # 参数预算与算力账本
    "CostLedger",
    "budget_table",
    "cost_ledger",
    "embedding_share",
    "largest_config_for_flops",
    "non_embedding_params",
    "plan_budget",
    "tokens_for_budget",
    "train_flops_per_token",
    "EvalScores",
    "ExperienceBuffer",
    "GateVerdict",
    "Insight",
    "TrainExample",
    "Trajectory",
    "paired_bootstrap_ci",
    "regression_gate",
    "SimplCoreBlock",
    "SimplCoreModel",
    "SimplCoreForCausalLM",
    # 门控谱卷积 mixer（GLA/Mamba2 风格通道门，非注意力）
    "GatedSpectralConv2",
    "analytic_gated_spectral_params",
    # 理论优势的可执行验证（把 THEORETICAL_EDGE.md 的每条主张变成断言）
    "Check",
    "run_all",
    "format_table",
    # 无词表模式（可插拔编解码器）
    "Tokenizer",
    "CharTokenizer",
    "ByteTokenizer",
    "BPETokenizer",
    "make_codec",
    # Transformer / HF 互操作桥接（只搬运安全组件，不引入注意力）
    "MigrationReport",
    "init_simplcore_from_hf",
    "migrate_embeddings_and_ffn",
    "sgc_to_hf_eval_bridge",
    "simplcore_stream_state_for_eval",
    # 长上下文外推 + 衰减消融工具
    "extrapolation_probe",
    "decay_ablation",
    "sgc_model_factory",
    "ExtrapolationResult",
    "DecayPoint",
    # 思考加速（逐 token 自适应深度）
    "AdaptiveDepth",
    # 非线性思考（latent 迭代精化 + ACT 式停止头）
    "LatentRefiner",
    "LatentRefineCell",
    "HaltingHead",
    "RefineLedger",
    # Simpl Model Memory（最大自主学习整合架构）
    "SimplModelMemory",
    "SMMConfig",
    "SMMLedger",
    "MemoryTier",
    "ComputeLedger",
    # 深度推理（迭代式思考控制器）
    "IterativeRefiner",
    "ReasoningTrace",
    "stabilized",
    # 环境学习（权重预算 + 分阶段蒸馏）
    "WeightBudget",
    "AdapterBlock",
    "LearnedAdapter",
    "ForwardAdapter",
    "EnvTrajectory",
    "RolloutResult",
    "Environment",
    "LocalCodeEnvironment",
    "StageReport",
    "EnvironmentLearner",
    "summarize_failures",
    # 自研 CUDA/C++ 后端算子契约（运行时 = 自研后端，PyTorch 退役出战役路径）
    "BackendContract",
    "BackendAdapter",
    "TorchReferenceAdapter",
    "ConformanceReport",
    "ConformanceResult",
    "run_conformance",
    "AdapterModel",
    "RecordingAdapter",
    "StreamingTrainer",
    "ContextCurriculum",
    "CurriculumStage",
    "TrainLedger",
    # 等预算对照（协议 X-equal-budget 的可执行 harness）
    "SyntheticWindowSource",
    "TSConfig",
    "TransformerBaseline",
    "WindowStream",
    "match_ts_layers",
    "paired_eval",
    "run_comparison",
    "ts_train_flops_per_token",
    # 长上下文检索评测（X-longctx-retrieval + X-stream-consistency）
    "NeedleCorpus",
    "NeedleTrial",
    "run_longctx_eval",
    "score_trial",
    # 代码 pass@k 对照（X-code-passk：真实执行判分 + 无偏估计）
    "GeneratorSpec",
    "make_problems_small",
    "pass_at_k",
    "run_code_passk",
    # 解码吞吐对照（X-throughput：O(1) 流式 vs KV cache，延迟斜率 + 解析账）
    "TSDecoder",
    "analytic_decode_ledger",
    "fit_slope",
    "run_throughput_eval",
    "summarize_latency",
    # Phase 1 训练执行器（packed 语料 → 课程窗口流 → StreamingTrainer）
    "CorpusBatches",
    "SyntheticCorpusSpec",
    "make_synthetic_corpus",
    "run_phase1",
    # 后端 C-ABI 桥接层（真实 CUDA 后端 drop-in 接口）
    "ABI_SPEC",
    "CtypesBackendAdapter",
    "FakeCLib",
    "load_backend",
    "symbol_report",
    # 论文/战役账目交叉审计
    "run_audit",
    "BackendOp",
    "ConformanceCheck",
    "DEFAULT_CONTRACT",
    "SIMPLCORE_BACKEND_OPS",
    "conformance_checks",
    # 部署侧 HBM 账本（1M ≤ 2GB，翻倍增量 ≤ 1GB —— 结构性为 0）
    "DeployRow",
    "DEPLOY_LADDER",
    "DEPLOY_HBM_CAP_1M",
    "DEPLOY_DOUBLING_CAP",
    "deploy_bytes",
    "deployment_report",
    "deploy_verdict",
    # 1B→3B 战役计划（硬停闸门 + HBM 账本 + 真实输出验收协议）
    "CampaignPlan",
    "ShowcaseItem",
    "TOTAL_BUDGET_DAYS",
    "DoublingRow",
    "Gate",
    "Phase",
    "CONTEXT_LADDER",
    "HBM_MI250_BYTES",
    "PHASE1_MAX_DAYS",
    "PHASE2_MAX_DAYS",
    "build_plan",
    "claim_verdict",
    "doubling_report",
    "ProofResult",
    "run_math_proofs",
]



# ---------------------------------------------------------------------------
# 0.1-α 命名降调前的旧公共名（pickle / 历史脚本兼容；新代码用新名）
# ---------------------------------------------------------------------------
DeepReasoner = IterativeRefiner
NonlinearThinker = LatentRefiner
NonlinearThinkCell = LatentRefineCell
ThinkLedger = RefineLedger
ThinkAccelerator = AdaptiveDepth
DreamingLoop = ReflectionLoop
DreamConfig = ReflectionConfig
DreamReport = ReflectionReport
DreamStage = ReflectionStage

__all__ += ["DeepReasoner", "NonlinearThinker", "NonlinearThinkCell",
            "ThinkLedger", "ThinkAccelerator", "DreamingLoop",
            "DreamConfig", "DreamReport", "DreamStage"]

__version__ = "0.1-α"

def __getattr__(name: str):
    # lab 旧名（jitrl / conditions_ledger / pilot）的属性级惰性解析：
    # pyproject.toml 规定 core 产品线不得反向 import lab，因此门面不再
    # re-export lab 符号；旧平铺路径仅在显式访问时按需加载 simplcore.lab.*。
    try:
        return _lab_getattr(name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
