"""
青小团配置模块测试 - 覆盖QingxiaotuanConfig的所有功能
"""

import os
import json
import pickle
import copy
import pytest
import torch
from qingxiaotuan import QingxiaotuanConfig


class TestQingxiaotuanConfigDefaults:

    def test_default_config_creation(self):
        config = QingxiaotuanConfig()
        assert config.vocab_size == 152064
        assert config.hidden_size == 7168
        assert config.intermediate_size == 20480
        assert config.num_hidden_layers == 48
        assert config.num_attention_heads == 56
        assert config.num_key_value_heads == 8
        assert config.hidden_act == "silu"
        assert config.max_position_embeddings == 524288
        assert config.rms_norm_eps == 1e-6

    def test_default_rope_scaling_yarn(self):
        config = QingxiaotuanConfig()
        rs = config.rope_scaling
        assert rs["type"] == "yarn"
        assert rs["factor"] == 64.0
        assert rs["original_max_position_embeddings"] == 8192
        assert rs["beta_fast"] == 32
        assert rs["beta_slow"] == 1

    def test_default_moe_params(self):
        config = QingxiaotuanConfig()
        assert config.num_experts == 32
        assert config.top_k == 2
        assert config.expert_intermediate_size == 2560
        assert config.shared_expert_intermediate_size == 20480
        assert config.moe_layer_interval == 2
        assert config.moe_aux_loss_coef == 0.01
        assert config.moe_z_loss_coef == 0.001

    def test_default_flash_attention(self):
        config = QingxiaotuanConfig()
        assert config.use_flash_attention is False

    def test_default_attention_sink(self):
        config = QingxiaotuanConfig()
        assert config.attention_sink_tokens == 4

    def test_default_sliding_window(self):
        config = QingxiaotuanConfig()
        assert config.sliding_window_size is None

    def test_default_kv_cache_quantization(self):
        config = QingxiaotuanConfig()
        assert config.kv_cache_quantization is False

    def test_default_gradient_checkpointing(self):
        config = QingxiaotuanConfig()
        assert config.gradient_checkpointing_steps == 12

    def test_default_tie_word_embeddings(self):
        config = QingxiaotuanConfig()
        assert config.tie_word_embeddings is True

    def test_default_initializer_range(self):
        config = QingxiaotuanConfig()
        assert config.initializer_range == 0.02

    def test_default_torch_dtype(self):
        config = QingxiaotuanConfig()
        assert config.torch_dtype == "float16"


class TestQingxiaotuanConfigParameterSetting:

    def test_set_all_parameters(self):
        config = QingxiaotuanConfig(
            vocab_size=64000,
            hidden_size=4096,
            intermediate_size=12288,
            num_hidden_layers=12,
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_act="gelu",
            max_position_embeddings=32768,
            rms_norm_eps=1e-5,
            rope_theta=25000.0,
            rope_scaling=None,
            use_cache=False,
            tie_word_embeddings=False,
            num_experts=16,
            top_k=4,
            expert_intermediate_size=1024,
            shared_expert_intermediate_size=4096,
            moe_layer_interval=4,
            moe_aux_loss_coef=0.05,
            moe_z_loss_coef=0.005,
            initializer_range=0.01,
            torch_dtype="bfloat16",
            use_flash_attention=True,
            attention_sink_tokens=8,
            sliding_window_size=2048,
            kv_cache_quantization=True,
            gradient_checkpointing_steps=6,
            segmented_loss_chunk_size=8192,
        )
        assert config.vocab_size == 64000
        assert config.hidden_size == 4096
        assert config.intermediate_size == 12288
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 4
        assert config.hidden_act == "gelu"
        assert config.max_position_embeddings == 32768
        assert config.rms_norm_eps == 1e-5
        assert config.rope_theta == 25000.0
        assert config.rope_scaling is None
        assert config.use_cache is False
        assert config.tie_word_embeddings is False
        assert config.num_experts == 16
        assert config.top_k == 4
        assert config.expert_intermediate_size == 1024
        assert config.shared_expert_intermediate_size == 4096
        assert config.moe_layer_interval == 4
        assert config.moe_aux_loss_coef == 0.05
        assert config.moe_z_loss_coef == 0.005
        assert config.initializer_range == 0.01
        assert config.torch_dtype == "bfloat16"
        assert config.use_flash_attention is True
        assert config.attention_sink_tokens == 8
        assert config.sliding_window_size == 2048
        assert config.kv_cache_quantization is True
        assert config.gradient_checkpointing_steps == 6
        assert config.segmented_loss_chunk_size == 8192

    def test_partial_parameter_update(self):
        config = QingxiaotuanConfig(hidden_size=5120, num_hidden_layers=40)
        assert config.hidden_size == 5120
        assert config.num_hidden_layers == 40
        assert config.vocab_size == 152064


class TestQingxiaotuanConfigSerialization:

    def test_to_dict_roundtrip(self):
        config1 = QingxiaotuanConfig(hidden_size=2048, num_hidden_layers=6)
        d = config1.to_dict()
        config2 = QingxiaotuanConfig(**d)
        assert config1.hidden_size == config2.hidden_size
        assert config1.num_hidden_layers == config2.num_hidden_layers
        assert config1.use_flash_attention == config2.use_flash_attention
        assert config1.rope_scaling == config2.rope_scaling

    def test_to_json_string(self):
        config = QingxiaotuanConfig(hidden_size=4096)
        json_str = config.to_json_string()
        parsed = json.loads(json_str)
        assert parsed["hidden_size"] == 4096
        assert parsed["model_type"] == "qingxiaotuan"

    def test_from_pretrained_config(self):
        config = QingxiaotuanConfig()
        config_dict = config.to_dict()
        restored = QingxiaotuanConfig.from_dict(config_dict)
        assert restored.hidden_size == config.hidden_size

    def test_pickle_serialization(self):
        config = QingxiaotuanConfig(hidden_size=2048)
        data = pickle.dumps(config)
        restored = pickle.loads(data)
        assert restored.hidden_size == 2048
        assert restored.num_hidden_layers == 48

    def test_to_pretrained_and_load(self, temp_dir):
        config = QingxiaotuanConfig(hidden_size=1024, num_hidden_layers=3)
        config.save_pretrained(temp_dir)
        loaded = QingxiaotuanConfig.from_pretrained(temp_dir)
        assert loaded.hidden_size == 1024
        assert loaded.num_hidden_layers == 3

    def test_to_variant_configs_exist(self):
        assert QingxiaotuanConfig(vocab_size=32000, hidden_size=2048)


class TestQingxiaotuanConfig512K:

    def test_512k_max_position_embeddings(self):
        config = QingxiaotuanConfig(max_position_embeddings=524288)
        assert config.max_position_embeddings == 524288

    def test_512k_with_large_hidden(self):
        config = QingxiaotuanConfig(
            max_position_embeddings=524288,
            hidden_size=7168,
            num_hidden_layers=48,
        )
        assert config.max_position_embeddings == 524288


class TestQingxiaotuanConfigYaRN:

    def test_yarn_config_creation(self):
        rope_scaling = {
            "type": "yarn",
            "factor": 64.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 0.707,
        }
        config = QingxiaotuanConfig(rope_scaling=rope_scaling)
        assert config.rope_scaling["type"] == "yarn"
        assert config.rope_scaling["factor"] == 64.0

    def test_yarn_with_factor_128(self):
        config = QingxiaotuanConfig(
            rope_scaling={"type": "yarn", "factor": 128.0,
                          "original_max_position_embeddings": 8192}
        )
        assert config.rope_scaling["factor"] == 128.0

    def test_no_rope_scaling(self):
        config = QingxiaotuanConfig(rope_scaling=None)
        assert config.rope_scaling is None


class TestQingxiaotuanConfigMoE:

    def test_moe_config_variations(self):
        config = QingxiaotuanConfig(num_experts=64, top_k=8)
        assert config.num_experts == 64
        assert config.top_k == 8

    def test_moe_layer_interval_variations(self):
        config = QingxiaotuanConfig(moe_layer_interval=1)
        assert config.moe_layer_interval == 1
        config2 = QingxiaotuanConfig(moe_layer_interval=3)
        assert config2.moe_layer_interval == 3

    def test_moe_loss_coefficients(self):
        config = QingxiaotuanConfig(moe_aux_loss_coef=0.0, moe_z_loss_coef=0.0)
        assert config.moe_aux_loss_coef == 0.0
        assert config.moe_z_loss_coef == 0.0


class TestQingxiaotuanConfigFlashAttention:

    def test_flash_attn_enabled(self):
        config = QingxiaotuanConfig(use_flash_attention=True)
        assert config.use_flash_attention is True

    def test_flash_attn_disabled(self):
        config = QingxiaotuanConfig(use_flash_attention=False)
        assert config.use_flash_attention is False


class TestQingxiaotuanConfigAttentionSink:

    def test_attention_sink_setting(self):
        config = QingxiaotuanConfig(attention_sink_tokens=16)
        assert config.attention_sink_tokens == 16

    def test_attention_sink_zero(self):
        config = QingxiaotuanConfig(attention_sink_tokens=0)
        assert config.attention_sink_tokens == 0


class TestQingxiaotuanConfigValidation:

    def test_any_config_is_valid(self):
        QingxiaotuanConfig()
        QingxiaotuanConfig(hidden_size=2048)
        QingxiaotuanConfig(hidden_size=4096, num_hidden_layers=12)

    def test_config_with_extra_kwargs(self):
        config = QingxiaotuanConfig(some_extra_param=42)
        assert config.hidden_size == 7168


class TestQingxiaotuanConfigComparison:

    def test_config_equal(self):
        c1 = QingxiaotuanConfig(hidden_size=2048)
        c2 = QingxiaotuanConfig(hidden_size=2048)
        assert c1.hidden_size == c2.hidden_size

    def test_config_not_equal(self):
        c1 = QingxiaotuanConfig(hidden_size=2048)
        c2 = QingxiaotuanConfig(hidden_size=4096)
        assert c1.hidden_size != c2.hidden_size

    def test_model_type_is_qingxiaotuan(self):
        config = QingxiaotuanConfig()
        assert config.model_type == "qingxiaotuan"
        assert hasattr(config, "model_type")
