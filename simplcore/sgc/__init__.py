# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC —— 内容门控的因果谱全局卷积 (Content-Gated Causal Spectral Global Convolution).

本包是 SGC 的正式生产实现, 取代实验脚本 `verify_sgc/sgc_pytorch.py` 中的原型。

核心设计(经 `docs/SGC_SPEC.md` 论证, 并由 `tests/test_mixer_math.py` 数值验证)
--------------------------------------------------------------------------------
位置路径(long-range)用 **M 个阻尼振荡基** 参数化因果全局卷积核:

    psi_m[tau] = rho_m^tau * sin((tau + 1) * theta_m),      tau >= 0
    K[tau, d]  = w_d * 1{tau = 0} + sum_m C[m, d] * psi_m[tau]

该核有一个**对偶形式**, 这是 SGC 相对 SGC v2 的关键改进:

  * 并行/训练形式: y[t] = sum_{s<=t} K[t-s] * u[s], 用 FFT 计算, O(L log L);
  * 流式/推理形式: 二阶 IIR 递推(每个基 2 个状态)
        y_m[t] = a_m * y_m[t-1] + b_m * y_m[t-2] + sin(theta_m) * u[t]
        a_m = 2 rho_m cos(theta_m),  b_m = -rho_m^2
    常驻状态 2*M*D 个浮点, **与序列长度 L 无关**。

两者在 fp32 下数值一致(实测 max abs err < 4e-5), 见 `tests/test_mixer_math.py`。

修复了 SGC v2 的三个已证实缺陷
------------------------------------------------
1. **感受野被静默截断**: v2 用 `low[:L] @ up` 构核, 当 L > max_len 时不报错、
   不警告, 只是退化成宽度 max_len 的固定窗卷积, 全局性丢失。SGC 的核由
   递推/闭式定义, 长度不受 max_len 限制, 且 `max_len` 只作为初始化尺度先验。
2. **时间基秩瓶颈**: v2 的核是任意 rank-r 子空间(r 默认 8), 即整条 L 长的核
   最多只有 8 个独立时间基函数 —— 这是它在高熵语料上无法按内容路由、转而记忆化
   的结构性原因。SGC 用带衰减先验的 M 基字典(M 默认 64)+ 有界局部 FIR 修正,
   既有长程平滑结构, 也能表示尖锐的局部结构。
3. **核谱每步重算**: v2 每次前向都重新对核做 rfft(实测占卷积耗时约 15%)。
   本实现把可复用的量分离, 并对不需要梯度的推理路径提供缓存。

内容路由(selective)
--------------------
引入输入相关的门控 `g = sigmoid(W_g * norm(x))`, 作用在卷积输入上:
`u = W_in * norm(x) * g`。这样有效核随内容变化(GLA/GSS 家族的做法),
直接针对"位置相对核无法按内容路由"的已报告瓶颈, 且不引入任何 Q/K/V softmax
注意力、不改变 O(L log L) / O(1) 的复杂度阶。

数值后端: 纯 PyTorch (autograd, 支持 bf16/fp16 autocast, 可 device-agnostic)。
"""
from __future__ import annotations

from .config import SGCConfig, SGC_PRESETS, get_preset
from .mixer import SpectralGlobalConv, SGCStreamState
from .memory import (
    CompressedMemoryBank,
    MemoryAwareObjective,
    MemoryCompressor,
    MemoryItem,
    self_match_scores,
    top_aggregate,
)
from .dream import (
    DreamConfig,
    DreamingLoop,
    DreamReport,
    DreamStage,
    ExperienceBuffer,
    Insight,
    TrainExample,
    Trajectory,
)
from .abstain import (
    AbstentionHead,
    abstention_loss,
    grounded_probability,
    should_abstain,
    calibrate_temperature,
)
from .model import SGCBlock, SGCModel, SGCForCausalLM
from .hybrid import (
    SlidingWindowAttention,
    HybridBlock,
    HybridModel,
    HybridLayerState,
    analytic_hybrid_params,
)
from .attention import GQASlidingWindowAttention, analytic_gqa_attn_params
from .attention_yarn import GQASlidingWindowAttentionYaRN, analytic_gqa_yarn_attn_params
from .selective import GatedSpectralConv2, analytic_gated_spectral_params
from .rope_scales import RopeScaler, SlidingWindowAttentionYaRN, make_rope_tables
from .hf_bridge import (
    MigrationReport,
    init_sgc_from_hf,
    migrate_embeddings_and_ffn,
    attention_to_sgc_warmstart,
    attention_gqa_warmstart,
    sgc_to_hf_eval_bridge,
    sgc_state_to_pseudo_kv,
)
from .tokenization import (
    Tokenizer,
    CharTokenizer,
    ByteTokenizer,
    BPETokenizer,
    make_codec,
)
from .extrapolation import (
    extrapolation_probe,
    decay_ablation,
    sgc_model_factory,
    ExtrapolationResult,
    DecayPoint,
)

__all__ = [
    "SGCConfig",
    "SGC_PRESETS",
    "get_preset",
    "SpectralGlobalConv",
    "SGCStreamState",
    # 四层闸门（L2 证据门 / L3 不确定门在 mixer 内，L4 拒答门在输出侧）
    "AbstentionHead",
    "abstention_loss",
    "grounded_probability",
    "should_abstain",
    "calibrate_temperature",
    # CMT 压缩记忆库（连续知识学习）
    "CompressedMemoryBank",
    "MemoryAwareObjective",
    "MemoryCompressor",
    "MemoryItem",
    "self_match_scores",
    "top_aggregate",
    # Dreaming 自进化循环
    "DreamConfig",
    "DreamingLoop",
    "DreamReport",
    "DreamStage",
    "ExperienceBuffer",
    "Insight",
    "TrainExample",
    "Trajectory",
    "SGCBlock",
    "SGCModel",
    "SGCForCausalLM",
    # SGC 混合架构（全局谱卷积 + 局部滑窗注意力）
    "SlidingWindowAttention",
    "HybridBlock",
    "HybridModel",
    "HybridLayerState",
    "analytic_hybrid_params",
    # SGC Hybrid 高效注意力 + 门控增强 mixer + 位置缩放外推
    "GQASlidingWindowAttention",
    "analytic_gqa_attn_params",
    "GQASlidingWindowAttentionYaRN",
    "analytic_gqa_yarn_attn_params",
    "GatedSpectralConv2",
    "analytic_gated_spectral_params",
    "RopeScaler",
    "SlidingWindowAttentionYaRN",
    "make_rope_tables",
    # SGC Hybrid 无词表模式（可插拔编解码器）
    "Tokenizer",
    "CharTokenizer",
    "ByteTokenizer",
    "BPETokenizer",
    "make_codec",
    # Transformer / HF 互操作桥接
    "MigrationReport",
    "init_sgc_from_hf",
    "migrate_embeddings_and_ffn",
    "attention_to_sgc_warmstart",
    "attention_gqa_warmstart",
    "sgc_to_hf_eval_bridge",
    "sgc_state_to_pseudo_kv",
    # 长上下文外推 + 衰减消融工具
    "extrapolation_probe",
    "decay_ablation",
    "sgc_model_factory",
    "ExtrapolationResult",
    "DecayPoint",
]

__version__ = "2.1.0"
