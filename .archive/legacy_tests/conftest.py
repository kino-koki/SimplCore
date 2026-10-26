"""
青小团测试套件 - pytest配置和fixture定义
提供所有测试模块共享的fixture
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
MODEL_CONFIGS_DIR = os.path.join(FIXTURES_DIR, "model_configs")
SAMPLE_DATA_DIR = os.path.join(FIXTURES_DIR, "sample_data")


@pytest.fixture(scope="session")
def small_qingxiaotuan_config():
    from qingxiaotuan import QingxiaotuanConfig
    return QingxiaotuanConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        num_experts=4,
        top_k=2,
        expert_intermediate_size=512,
        shared_expert_intermediate_size=1024,
        moe_layer_interval=1,
        moe_aux_loss_coef=0.01,
        moe_z_loss_coef=0.001,
        initializer_range=0.02,
        torch_dtype="float32",
        use_flash_attention=False,
        attention_sink_tokens=2,
        sliding_window_size=None,
        kv_cache_quantization=False,
        gradient_checkpointing_steps=1,
        segmented_loss_chunk_size=512,
    )


@pytest.fixture(scope="session")
def mid_qingxiaotuan_config():
    from qingxiaotuan import QingxiaotuanConfig
    return QingxiaotuanConfig(
        vocab_size=152064,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        rope_scaling={"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 1024,
                      "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 0.707},
        num_experts=8,
        top_k=2,
        expert_intermediate_size=1024,
        shared_expert_intermediate_size=4096,
        moe_layer_interval=2,
        moe_aux_loss_coef=0.01,
        moe_z_loss_coef=0.001,
        initializer_range=0.02,
        torch_dtype="float32",
        use_flash_attention=False,
        attention_sink_tokens=4,
        sliding_window_size=512,
        kv_cache_quantization=True,
        gradient_checkpointing_steps=2,
        segmented_loss_chunk_size=2048,
    )


@pytest.fixture
def dummy_inputs():
    return {
        "input_ids": torch.randint(0, 1000, (2, 64)),
        "attention_mask": torch.ones(2, 64, dtype=torch.long),
        "labels": torch.randint(0, 1000, (2, 64)),
    }


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp(prefix="qingxiaotuan_test_")
    yield path
    shutil.rmtree(path)


@pytest.fixture
def sample_instruction_path():
    return os.path.join(SAMPLE_DATA_DIR, "sample_instructions.jsonl")


@pytest.fixture
def sample_pretrain_path():
    return os.path.join(SAMPLE_DATA_DIR, "sample_pretrain.jsonl")


@pytest.fixture
def sample_conversation_path():
    return os.path.join(SAMPLE_DATA_DIR, "sample_conversations.jsonl")


@pytest.fixture
def sample_code_path():
    return os.path.join(SAMPLE_DATA_DIR, "sample_code.jsonl")


@pytest.fixture
def sample_reasoning_path():
    return os.path.join(SAMPLE_DATA_DIR, "sample_reasoning.jsonl")


@pytest.fixture
def model_configs_dir():
    return MODEL_CONFIGS_DIR


@pytest.fixture
def sample_data_dir():
    return SAMPLE_DATA_DIR


@pytest.fixture
def yarn_config():
    from qingxiaotuan import QingxiaotuanConfig
    return QingxiaotuanConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=524288,
        rope_theta=500000.0,
        rope_scaling={"type": "yarn", "factor": 64.0, "original_max_position_embeddings": 8192,
                      "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 0.707},
        num_experts=4,
        top_k=2,
        expert_intermediate_size=512,
        shared_expert_intermediate_size=1024,
        moe_layer_interval=1,
        use_flash_attention=False,
        attention_sink_tokens=4,
    )


@pytest.fixture
def flash_attn_config():
    from qingxiaotuan import QingxiaotuanConfig
    return QingxiaotuanConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        use_flash_attention=True,
        attention_sink_tokens=0,
        sliding_window_size=None,
    )


@pytest.fixture
def moe_config():
    from qingxiaotuan import QingxiaotuanConfig
    return QingxiaotuanConfig(
        vocab_size=32000,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        top_k=2,
        expert_intermediate_size=512,
        shared_expert_intermediate_size=1024,
        moe_layer_interval=1,
        use_flash_attention=False,
    )


@pytest.fixture(scope="session")
def small_qingxiaotuan_model(small_qingxiaotuan_config):
    from qingxiaotuan import QingxiaotuanForCausalLM
    return QingxiaotuanForCausalLM(small_qingxiaotuan_config)


@pytest.fixture(scope="session")
def small_moe_model(moe_config):
    from qingxiaotuan import QingxiaotuanForCausalLM
    return QingxiaotuanForCausalLM(moe_config)


@pytest.fixture
def loaded_json_configs(model_configs_dir):
    configs = {}
    for fname in os.listdir(model_configs_dir):
        if fname.endswith(".json"):
            path = os.path.join(model_configs_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                configs[fname] = json.load(f)
    return configs


@pytest.fixture
def batch_inputs():
    return {
        "input_ids": torch.randint(0, 1000, (4, 32)),
        "attention_mask": torch.ones(4, 32, dtype=torch.long),
        "labels": torch.randint(0, 1000, (4, 32)),
    }


@pytest.fixture
def long_sequence_inputs():
    return {
        "input_ids": torch.randint(0, 1000, (1, 8192)),
        "attention_mask": torch.ones(1, 8192, dtype=torch.long),
        "labels": torch.randint(0, 1000, (1, 8192)),
    }
