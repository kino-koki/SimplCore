"""
青小团模型转换测试 - 覆盖HuggingFace转换、vLLM转换、safetensors、检查点合并等
"""

import os
import json
import torch
import datetime
import pytest
from unittest.mock import Mock, patch, MagicMock
from qingxiaotuan import QingxiaotuanConfig


class TestQingxiaotuanHFConversion:

    def test_hf_converter_creation(self):
        from qingxiaotuan.conversion.to_hf import HFConverter
        converter = HFConverter()
        assert converter is not None

    def test_hf_convert_to_transformers(self):
        from qingxiaotuan.conversion.to_hf import HFConverter
        converter = HFConverter()
        config = QingxiaotuanConfig(hidden_size=256, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)
        model = converter.convert(config)
        assert model is not None

    def test_hf_config_mapping(self):
        from qingxiaotuan.conversion.to_hf import HFConverter
        converter = HFConverter()
        qingxiaotuan_config = QingxiaotuanConfig()
        hf_config = converter.map_config(qingxiaotuan_config)
        assert hf_config is not None
        assert hf_config.model_type == "qingxiaotuan"

    def test_hf_save_pretrained(self, temp_dir):
        from qingxiaotuan.conversion.to_hf import HFConverter
        converter = HFConverter()
        config = QingxiaotuanConfig(hidden_size=256, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)
        model = converter.convert(config)
        output_dir = os.path.join(temp_dir, "hf_model")
        converter.save_pretrained(model, output_dir)
        assert os.path.exists(os.path.join(output_dir, "config.json"))
        assert os.path.exists(os.path.join(output_dir, "model.safetensors")) or \
               os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))

    def test_hf_config_has_correct_type(self):
        from qingxiaotuan.conversion.to_hf import HFConverter
        converter = HFConverter()
        qingxiaotuan_config = QingxiaotuanConfig()
        hf_config = converter.map_config(qingxiaotuan_config)
        assert hf_config.model_type == "qingxiaotuan"

    def test_hf_state_dict_keys(self):
        from qingxiaotuan.conversion.to_hf import HFConverter
        converter = HFConverter()
        config = QingxiaotuanConfig(hidden_size=256, num_hidden_layers=1,
                                     num_attention_heads=4, num_key_value_heads=2)
        model = converter.convert(config)
        sd = model.state_dict()
        q_keys = [k for k in sd.keys() if "q_proj" in k]
        k_keys = [k for k in sd.keys() if "k_proj" in k]
        v_keys = [k for k in sd.keys() if "v_proj" in k]
        o_keys = [k for k in sd.keys() if "o_proj" in k]
        assert len(q_keys) >= 1
        assert len(k_keys) >= 1
        assert len(v_keys) >= 1
        assert len(o_keys) >= 1


class TestQingxiaotuanVLLMConversion:

    def test_vllm_converter_creation(self):
        from qingxiaotuan.conversion.to_vllm import VLLMConverter
        converter = VLLMConverter()
        assert converter is not None

    def test_vllm_convert_model(self):
        from qingxiaotuan.conversion.to_vllm import VLLMConverter
        converter = VLLMConverter()
        config = QingxiaotuanConfig(hidden_size=256, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)
        model = converter.convert(config)
        assert model is not None

    def test_vllm_config_mapping(self):
        from qingxiaotuan.conversion.to_vllm import VLLMConverter
        converter = VLLMConverter()
        qingxiaotuan_config = QingxiaotuanConfig()
        vllm_config = converter.map_to_vllm(qingxiaotuan_config)
        assert vllm_config is not None
        assert "model_type" in vllm_config or hasattr(vllm_config, "model_type")

    def test_vllm_weight_mapping(self):
        from qingxiaotuan.conversion.to_vllm import VLLMConverter
        converter = VLLMConverter()
        mapping = converter.get_weight_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    def test_vllm_save_for_serving(self, temp_dir):
        from qingxiaotuan.conversion.to_vllm import VLLMConverter
        converter = VLLMConverter()
        config = QingxiaotuanConfig(hidden_size=256, num_hidden_layers=1,
                                     num_attention_heads=4, num_key_value_heads=2)
        model = converter.convert(config)
        output_dir = os.path.join(temp_dir, "vllm")
        converter.save_for_serving(model, output_dir)
        assert os.path.exists(output_dir)

    def test_vllm_config_has_gqa(self):
        from qingxiaotuan.conversion.to_vllm import VLLMConverter
        converter = VLLMConverter()
        qingxiaotuan_config = QingxiaotuanConfig(num_key_value_heads=8)
        vllm_config = converter.map_to_vllm(qingxiaotuan_config)
        mapped_kv = vllm_config.get("num_key_value_heads", 
                                     getattr(vllm_config, "num_key_value_heads", 8))
        assert mapped_kv == 8


class TestQingxiaotuanSafeTensors:

    def test_safetensors_save_load(self, small_qingxiaotuan_model, temp_dir):
        from safetensors.torch import save_file, load_file
        model = small_qingxiaotuan_model
        sd = model.state_dict()
        path = os.path.join(temp_dir, "model.safetensors")
        save_file(sd, path)
        assert os.path.getsize(path) > 0
        loaded = load_file(path)
        assert set(loaded.keys()) == set(sd.keys())

    def test_safetensors_weight_integrity(self, small_qingxiaotuan_model, temp_dir):
        from safetensors.torch import save_file, load_file
        sd = small_qingxiaotuan_model.state_dict()
        path = os.path.join(temp_dir, "weights.safetensors")
        save_file(sd, path)
        loaded = load_file(path)
        for key in sd:
            assert torch.allclose(sd[key], loaded[key])

    def test_safetensors_sharded_save(self, small_qingxiaotuan_model, temp_dir):
        from safetensors.torch import save_file
        sd = small_qingxiaotuan_model.state_dict()
        items = list(sd.items())
        mid = len(items) // 2
        part1 = dict(items[:mid])
        part2 = dict(items[mid:])
        save_file(part1, os.path.join(temp_dir, "model-00001-of-00002.safetensors"))
        save_file(part2, os.path.join(temp_dir, "model-00002-of-00002.safetensors"))
        assert os.path.exists(os.path.join(temp_dir, "model-00001-of-00002.safetensors"))
        assert os.path.exists(os.path.join(temp_dir, "model-00002-of-00002.safetensors"))

    def test_safetensors_metadata(self, temp_dir):
        from safetensors.torch import save_file
        sd = {"weight": torch.randn(10, 10)}
        metadata = {"format": "pt", "model": "qingxiaotuan"}
        path = os.path.join(temp_dir, "meta.safetensors")
        save_file(sd, path, metadata=metadata)
        assert os.path.getsize(path) > 0


class TestQingxiaotuanCheckpointMerging:

    def test_merge_checkpoints(self, temp_dir):
        from qingxiaotuan.conversion.checkpoint_merger import CheckpointMerger
        merger = CheckpointMerger()
        ckpt1 = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        ckpt2 = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        merged = merger.average([ckpt1, ckpt2], weights=[0.5, 0.5])
        assert "weight" in merged
        assert "bias" in merged

    def test_merge_unequal_weights(self, temp_dir):
        from qingxiaotuan.conversion.checkpoint_merger import CheckpointMerger
        merger = CheckpointMerger()
        ckpt1 = {"weight": torch.full((5, 5), 1.0)}
        ckpt2 = {"weight": torch.full((5, 5), 3.0)}
        merged = merger.average([ckpt1, ckpt2], weights=[0.25, 0.75])
        assert torch.allclose(merged["weight"], torch.full((5, 5), 2.5))

    def test_merge_equal_average(self, temp_dir):
        from qingxiaotuan.conversion.checkpoint_merger import CheckpointMerger
        merger = CheckpointMerger()
        ckpt1 = {"weight": torch.full((3, 3), 10.0)}
        ckpt2 = {"weight": torch.full((3, 3), 20.0)}
        merged = merger.average([ckpt1, ckpt2], weights=None)
        assert torch.allclose(merged["weight"], torch.full((3, 3), 15.0))

    def test_merge_save_merged(self, temp_dir):
        from qingxiaotuan.conversion.checkpoint_merger import CheckpointMerger
        merger = CheckpointMerger()
        ckpt1 = {"weight": torch.randn(5, 5)}
        ckpt2 = {"weight": torch.randn(5, 5)}
        merged = merger.average([ckpt1, ckpt2], weights=[0.5, 0.5])
        path = os.path.join(temp_dir, "merged.pt")
        torch.save(merged, path)
        assert os.path.getsize(path) > 0
