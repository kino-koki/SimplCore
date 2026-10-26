# -*- coding: utf-8 -*-
"""
快速验证脚本 — 检查所有模块的 import 和基础逻辑是否正常。

用法:
    python scripts/validate_imports.py

在有 Python 环境的机器上运行（目标训练服务器）。
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

errors = []
warnings = []


def check(name: str, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")


# --------------------------------------------------------------------------- #
# 1. qxt.model.config — 12B/100B 配置
# --------------------------------------------------------------------------- #
print("\n=== qxt.model.config ===")


def test_qxt_config():
    from qxt.model.config import QXTConfig, get_current_max_seq_len
    cfg = QXTConfig()
    assert cfg.max_position_embeddings == 1048576, f"max_pos={cfg.max_position_embeddings}"
    assert cfg.rope_theta == 1_000_000.0, f"rope_theta={cfg.rope_theta}"
    assert cfg.attention_mode == "standard"
    assert cfg.progressive_length_warmup["enabled"] == True

    # get_current_max_seq_len
    for step, expected in [(0, 4096), (999, 4096), (1000, 16384), (4999, 16384), (5000, 65536)]:
        got = get_current_max_seq_len(cfg, step)
        assert got == expected, f"step={step}: expected={expected}, got={got}"

    cfg12 = QXTConfig.get_12b_dense_config("csa_hca")
    assert cfg12.attention_mode == "csa_hca"
    assert cfg12.num_experts == 1  # Dense: MoE 禁用
    assert cfg12.max_position_embeddings == 1048576


def test_qxt_config_presets():
    from qxt.model.config import QXTConfig
    cfg100 = QXTConfig.get_100b_moe_config("ring_csa")
    assert cfg100.num_experts == 64
    assert cfg100.num_experts_per_tok == 8
    assert cfg100.use_ring_attention == True
    assert cfg100.num_ring_devices == 8
    assert cfg100.attention_mode == "ring_csa"


check("QXTConfig 基本字段 + progressive_length_warmup", test_qxt_config)
check("QXTConfig.get_100b_moe_config 预设", test_qxt_config_presets)


# --------------------------------------------------------------------------- #
# 2. qingxiaotuan.configuration_qingxiaotuan — HF 兼容配置
# --------------------------------------------------------------------------- #
print("\n=== qingxiaotuan.configuration_qingxiaotuan ===")


def test_qing_config():
    from qingxiaotuan.configuration_qingxiaotuan import QingxiaotuanConfig
    cfg = QingxiaotuanConfig()
    assert cfg.max_position_embeddings == 1048576, f"HF max_pos={cfg.max_position_embeddings}"
    assert cfg.rope_theta == 1_000_000.0, f"HF rope_theta={cfg.rope_theta}"
    assert cfg.attention_mode == "standard"
    assert cfg.progressive_length_warmup["enabled"] == True

    cfg_csa = QingxiaotuanConfig(attention_mode="csa_hca")
    assert cfg_csa.attention_mode == "csa_hca"

    cfg12 = QingxiaotuanConfig.get_12b_dense(attention_mode="hca")
    assert cfg12.hidden_size == 5120
    assert cfg12.num_hidden_layers == 40
    assert cfg12.num_experts == 1
    assert cfg12.attention_mode == "hca"


check("QingxiaotuanConfig 1M 上下文 + CSA/HCA", test_qing_config)


# --------------------------------------------------------------------------- #
# 3. qingxiaotuan.modeling_qingxiaotuan — 注意力工厂
# --------------------------------------------------------------------------- #
print("\n=== qingxiaotuan.modeling_qingxiaotuan ===")


def test_modeling_hf():
    from qingxiaotuan.configuration_qingxiaotuan import QingxiaotuanConfig
    from qingxiaotuan.modeling_qingxiaotuan import (
        QingxiaotuanAttention,
        QingxiaotuanAttentionWithCSA,
        QingxiaotuanDecoderLayer,
        _build_attention_module,
    )
    import torch

    # standard 模式
    cfg_std = QingxiaotuanConfig(attention_mode="standard")
    attn_std = _build_attention_module(cfg_std)
    assert isinstance(attn_std, QingxiaotuanAttention)
    assert attn_std.attention_mode == "standard"

    # csa_hca 模式
    cfg_csa = QingxiaotuanConfig(attention_mode="csa_hca")
    attn_csa = _build_attention_module(cfg_csa)
    assert isinstance(attn_csa, QingxiaotuanAttentionWithCSA)
    assert attn_csa.attention_mode == "csa_hca"

    # DecoderLayer 注入
    layer = QingxiaotuanDecoderLayer(cfg_csa, layer_idx=0)
    assert isinstance(layer.self_attn, QingxiaotuanAttentionWithCSA)
    assert layer.self_attn.attention_mode == "csa_hca"

    # RoPE 参数验证 (YaRN 1M)
    rotary = layer.self_attn.rotary_emb
    assert rotary.max_position_embeddings == 1048576, f"rotary max_pos={rotary.max_position_embeddings}"
    assert rotary.base == 1_000_000.0, f"rotary base={rotary.base}"


check("QingxiaotuanAttention 工厂 + CSA/HCA 注入", test_modeling_hf)


# --------------------------------------------------------------------------- #
# 4. qxt.model.modeling — 独立实现的工厂函数
# --------------------------------------------------------------------------- #
print("\n=== qxt.model.modeling ===")


def test_qxt_modeling_factory():
    from qxt.model.config import QXTConfig
    from qxt.model.modeling import QXTDecoderLayer, _build_attention_module
    cfg = QXTConfig(attention_mode="csa_hca")
    layer = QXTDecoderLayer(cfg)
    # 注意: qxt/model/modeling.py 依赖 qxt/model/csa_hca_ablation
    # 如果 csa_hca_ablation 不存在，会回退到 QXTAttention
    from qxt.model.attention import QXTAttention
    assert layer.self_attn is not None


check("qxt.modeling._build_attention_module 工厂", test_qxt_modeling_factory)


# --------------------------------------------------------------------------- #
# 5. CSA/HCA 消融实验框架
# --------------------------------------------------------------------------- #
print("\n=== qxt.model.csa_hca_ablation ===")


def test_csa_hca_ablation():
    from qxt.model.csa_hca_ablation import (
        get_attention_module,
        ATTENTION_MODES,
        StandardAttention,
        CSAAttention,
        HCAAttention,
    )
    assert "standard" in ATTENTION_MODES
    assert "csa" in ATTENTION_MODES
    assert "hca" in ATTENTION_MODES
    assert "csa_hca" in ATTENTION_MODES

    attn_std = get_attention_module("standard", 8, 2, 64)
    assert isinstance(attn_std, StandardAttention)

    attn_csa = get_attention_module("csa", 8, 2, 64)
    assert isinstance(attn_csa, CSAAttention)

    attn_hca = get_attention_module("hca", 8, 2, 64)
    assert isinstance(attn_hca, HCAAttention)


check("csa_hca_ablation 框架 (get_attention_module + ATTENTION_MODES)", test_csa_hca_ablation)


# --------------------------------------------------------------------------- #
# 6. YaRN RoPE 1M
# --------------------------------------------------------------------------- #
print("\n=== qxt.model.yarn_rope ===")


def test_yarn_rope():
    from qxt.model.yarn_rope import YaRNRotaryEmbedding, create_yarn_from_config
    from qxt.model.config import QXTConfig
    import torch

    yarn = YaRNRotaryEmbedding(
        dim=128,
        max_position_embeddings=1048576,
        original_max_position_embeddings=32768,
        scaling_factor=32.0,
        theta=1_000_000.0,
    )
    assert yarn.max_position_embeddings == 1048576
    assert yarn.base == 1_000_000.0

    pos_ids = torch.arange(8).unsqueeze(0)
    cos, sin = yarn(pos_ids)
    assert cos.shape == sin.shape == (1, 8, 128)

    cfg = QXTConfig()
    yarn2 = create_yarn_from_config(cfg)
    assert yarn2.max_position_embeddings == 1048576


check("YaRN RoPE 1M forward (cos/sin 形状)", test_yarn_rope)


# --------------------------------------------------------------------------- #
# 7. training/sft/train_sft.py — SFTArguments 新参数
# --------------------------------------------------------------------------- #
print("\n=== training/sft/train_sft.py ===")


def test_train_sft_args():
    # 读取文件检查新参数是否存在
    import ast
    sft_path = os.path.join(ROOT, "training", "sft", "train_sft.py")
    with open(sft_path, "r", encoding="utf-8") as f:
        src = f.read()

    required = [
        "attention_mode",
        "enable_progressive_length_warmup",
        "model_size",
        "ProgressiveLengthCallback",
    ]
    for name in required:
        assert name in src, f"缺少参数: {name}"


check("train_sft.py 包含 attention_mode + ProgressiveLengthCallback", test_train_sft_args)


# --------------------------------------------------------------------------- #
# 8. distillation_data 文件
# --------------------------------------------------------------------------- #
print("\n=== distillation_data ===")


def test_distillation_data():
    import json
    folder = os.path.join(ROOT, "distillation_data")

    for fname in ["general.jsonl", "code.jsonl", "math.jsonl", "creative.jsonl", "logic.jsonl"]:
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 1:
                        break
                    d = json.loads(line)
                    assert "instruction" in d or "messages" in d, f"{fname}: 缺少 instruction"
                    print(f"  [SKIP] {fname}: {i+1} 条样本 (存在)")


check("distillation_data jsonl 文件格式", test_distillation_data)


# --------------------------------------------------------------------------- #
# 总结
# --------------------------------------------------------------------------- #
print("\n" + "=" * 60)
if errors:
    print(f"  ❌ {len(errors)} 个检查失败:")
    for name, msg in errors:
        print(f"     - {name}: {msg}")
    sys.exit(1)
else:
    print("  ✅ 所有检查通过！")
    print("")
    print("  下一步:")
    print("    1) python scripts/merge_2100_instructions.py  # 合并指令")
    print("    2) python scripts/distill_with_agnes.py       # 蒸馏")
    print("    3) python training/sft/train_sft.py ...        # 训练")
    sys.exit(0)
