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

    # 注意力类型: "mba_mla" | "mba_gqa" | "mska"
    # MBA = Modulated Basis Attention (SimplCore创新)
    # MSKA = Multi-Scale Kernel Attention (SimplCore核心创新)
    attn_type: str = "mba_mla"
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128

    # MBA 基向量场参数 (SimplCore)
    n_basis_vectors: int = 64
    freq_scale_init: float = 1.0
    phase_shift_init: float = 0.0
    attn_temperature: float = 1.0

    # MoE + CCR 拓扑路由 (SimplCore + OpenMythos)
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4
    expert_dim: int = 512
    proto_topk: int = 4
    proto_use_confidence: bool = True
    proto_low_sim_threshold: float = 0.3

    # DMC 工作记忆缓冲 (SimplCore)
    n_memory_channels: int = 8
    memory_dim: int = 512
    memory_decay_init: float = 0.95
    memory_gate_init: float = 0.5
    wm_use_associative: bool = True
    wm_assoc_mem_size: int = 64

    # CIN 因果干预 — SGI 结构化门控干预 (SimplCore)
    cin_threshold: float = 0.5
    cin_use_confounder: bool = True
    cin_n_groups: int = 8           # 分组变量数量
    cin_top_k_parents: int = 3      # 每个变量的父节点上限
    cin_scale: float = 0.1          # 残差缩放

    # SPM Hebbian在线适应 (SimplCore)
    hebbian_enabled: bool = True
    hebbian_lr: float = 0.001
    hebbian_decay: float = 0.01

    # MSKA 多尺度核注意力参数 (SimplCore核心创新)
    mska_n_bands: int = 4           # 频带数量
    mska_kernel_rank: int = 16      # 双线性核的秩
    mska_scheduling_steps: int = 1  # 循环精化步数 (1=关闭)

    # Parcae 稳定性控制 (OpenMythos)
    stability_method: str = "parcae"  # "parcae" | "none"
    dt_init: float = 0.01
    act_threshold: float = 0.99

    # 推理模块集成
    reasoning_enabled: bool = True
    reasoning_dim: int = 512
    reasoning_mode: str = "balanced"  # fast/deep/balanced/creative
    iterative_refinement_enabled: bool = False
    max_refinement_loops: int = 16
    web_search_enabled: bool = False

    # 训练省显存能力
    gradient_checkpointing: bool = False

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

    # 置信度门控
    confidence_gate_enabled: bool = True
    confidence_threshold: float = 0.4


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
        n_basis_vectors=32,
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
        n_basis_vectors=64,
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
        n_basis_vectors=128,
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
        n_basis_vectors=8,
        n_memory_channels=2,
        memory_dim=32,
        lora_rank=4,
        reasoning_enabled=False,
        confidence_gate_enabled=False,
    )


# ============================================================
# 统一配置入口
# ============================================================

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
        n_basis_vectors=24,
        n_memory_channels=4,
        memory_dim=192,
        lora_rank=8,
        reasoning_enabled=False,
        confidence_gate_enabled=False,
        gradient_checkpointing=True,
        iterative_refinement_enabled=True,
        max_refinement_loops=8,
        web_search_enabled=True,
    )


def mix_3b_text() -> SimplCoreMixConfig:
    """3B pure-text config — 生产级中等规模配置。"""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=3584,
        n_heads=28,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=12,
        prelude_layers=3,
        coda_layers=3,
        n_experts=48,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1792,
        n_basis_vectors=48,
        n_memory_channels=6,
        memory_dim=448,
        lora_rank=12,
        reasoning_enabled=True,
        confidence_gate_enabled=True,
        gradient_checkpointing=True,
        iterative_refinement_enabled=True,
        max_refinement_loops=16,
        web_search_enabled=True,
    )


def mix_7b_text() -> SimplCoreMixConfig:
    """7B pure-text target config for parameter stacking."""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=16,
        prelude_layers=6,
        coda_layers=6,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        n_basis_vectors=64,
        n_memory_channels=8,
        memory_dim=512,
        lora_rank=16,
        # 纯文本 causal LM 路线: 主推理开关关闭;
        # 独立的 web_search / iterative_refinement 能力保持开启 (供后续路线启用)
        reasoning_enabled=False,
        confidence_gate_enabled=False,
        gradient_checkpointing=True,
        iterative_refinement_enabled=True,
        max_refinement_loops=24,
        web_search_enabled=True,
    )


def mix_14b_text() -> SimplCoreMixConfig:
    """14B pure-text target config for post-7B expansion."""
    return SimplCoreMixConfig(
        vocab_size=32000,
        dim=5120,
        n_heads=40,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=20,
        prelude_layers=8,
        coda_layers=8,
        n_experts=96,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=2560,
        n_basis_vectors=96,
        n_memory_channels=12,
        memory_dim=640,
        lora_rank=24,
        reasoning_enabled=False,
        confidence_gate_enabled=False,
    )

_CONFIG_REGISTRY = {
    "tiny": mix_tiny_text,
    "150m": mix_150m_text,
    "3b": mix_3b_text,
    "1b": mix_1b,
    "7b": mix_7b,
    "7b_text": mix_7b_text,
    "14b": mix_14b_text,
    "33b": mix_33b,
}


def get_config(name: str = "150m") -> SimplCoreMixConfig:
    """根据名称获取预定义配置。

    支持的名称: tiny / 150m / 3b / 1b / 7b / 7b_text / 14b / 33b
    """
    key = name.strip().lower()
    if key not in _CONFIG_REGISTRY:
        raise ValueError(
            f"未知配置名称: {name}\n"
            f"可选: {', '.join(_CONFIG_REGISTRY.keys())}"
        )
    return _CONFIG_REGISTRY[key]()


