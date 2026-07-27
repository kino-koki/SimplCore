# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore Mix 配置系统"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SimplCoreMixConfig:
    """SimplCore Mix 超参数配置

    融合 OpenMythos RDT + SimplCore 组件
    """
    # 核心维度
    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2

    # 注意力类型: "rfp_mla" | "rfp_gqa"
    # RFP = Resonance Field Processor (SimplCore创新)
    attn_type: str = "rfp_mla"
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128

    # RFP 共振场参数 (SimplCore)
    n_bases: int = 64
    freq_scale_init: float = 1.0
    phase_shift_init: float = 0.0
    rfp_temperature: float = 1.0

    # MoE + TAR 拓扑路由 (SimplCore + OpenMythos)
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4
    expert_dim: int = 512
    tar_topk: int = 4
    tar_use_confidence: bool = True
    tar_uncertainty_threshold: float = 0.3

    # DMC 动态记忆通道 (SimplCore)
    n_memory_channels: int = 8
    memory_dim: int = 512
    memory_decay_init: float = 0.95
    memory_gate_init: float = 0.5
    dmc_use_associative: bool = True
    dmc_assoc_mem_size: int = 64

    # CIN 因果干预 (SimplCore)
    cin_threshold: float = 0.5
    cin_use_confounder: bool = True

    # SPM 自发可塑性 (SimplCore)
    spm_enabled: bool = True
    spm_learning_rate: float = 0.001
    spm_gamma: float = 0.01

    # Parcae 稳定性控制 (OpenMythos)
    stability_method: str = "parcae"  # "parcae" | "none"
    dt_init: float = 0.01
    act_threshold: float = 0.99

    # 认知模块集成
    cognitive_enabled: bool = True
    cognitive_dim: int = 512
    reasoning_mode: str = "balanced"  # fast/deep/balanced/creative

    # 深度LoRA适配 (OpenMythos LoRAAdapter)
    lora_rank: int = 16
    loop_lora_enabled: bool = True

    # LTI注入参数 (OpenMythos增强版)
    # log_A 和 log_dt 为可学习参数，保证谱半径 < 1
    lti_log_dt_init: float = 0.0
    lti_B_init: float = 0.1

    # 循环索引嵌入 (OpenMythos)
    # 将循环索引编码为正弦信号注入隐藏状态前loop_dim个通道
    loop_dim_frac: float = 0.125  # dim的1/8用于loop index embedding

    # 权重绑定 (OpenMythos)
    weight_tying: bool = True

    # RoPE
    rope_theta: float = 500000.0

    # Dropout
    dropout: float = 0.0

    # 生成
    max_output_tokens: int = 4096

    # 反幻觉
    anti_hallucination: bool = True
    ah_threshold: float = 0.4


def mix_1b() -> SimplCoreMixConfig:
    """1B参数配置"""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=8,
        prelude_layers=2,
        coda_layers=2,
        n_experts=32,
        n_shared_experts=2,
        n_experts_per_tok=2,
        expert_dim=1024,
        n_bases=32,
        n_memory_channels=4,
        memory_dim=256,
        lora_rank=8,
    )


def mix_7b() -> SimplCoreMixConfig:
    """7B参数配置"""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=16,
        prelude_layers=4,
        coda_layers=4,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        n_bases=64,
        n_memory_channels=8,
        memory_dim=512,
        lora_rank=16,
    )


def mix_33b() -> SimplCoreMixConfig:
    """33B参数配置"""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=32768,
        max_loop_iters=24,
        prelude_layers=6,
        coda_layers=6,
        n_experts=128,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=4096,
        n_bases=128,
        n_memory_channels=16,
        memory_dim=1024,
        lora_rank=32,
    )


def mix_tiny_text() -> SimplCoreMixConfig:
    """Tiny pure-text config for CPU-safe tests."""
    return SimplCoreMixConfig(
        vocab_size=1024,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=128,
        n_bases=8,
        n_memory_channels=2,
        memory_dim=32,
        lora_rank=4,
        cognitive_enabled=False,
        anti_hallucination=False,
    )


def mix_150m_text() -> SimplCoreMixConfig:
    """Approximately 150M pure-text config; exact count is verified by script."""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=768,
        n_heads=12,
        n_kv_heads=4,
        max_seq_len=2048,
        max_loop_iters=6,
        prelude_layers=2,
        coda_layers=2,
        n_experts=16,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=1024,
        n_bases=24,
        n_memory_channels=4,
        memory_dim=192,
        lora_rank=8,
        cognitive_enabled=False,
        anti_hallucination=False,
    )
