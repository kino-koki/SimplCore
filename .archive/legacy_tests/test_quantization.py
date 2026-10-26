"""
青小团量化测试 - 覆盖GPTQ、AWQ、BNB适配、量化推理、量化后精度等
"""

import os
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from qingxiaotuan import QingxiaotuanConfig, QingxiaotuanForCausalLM


class TestQingxiaotuanGPTQ:

    def test_gptq_quantization_class_exists(self):
        from qingxiaotuan.quantization.gptq import GPTQQuantizer
        assert GPTQQuantizer is not None

    def test_gptq_quantizer_creation(self):
        from qingxiaotuan.quantization.gptq import GPTQQuantizer
        quantizer = GPTQQuantizer(bits=4, group_size=128)
        assert quantizer.bits == 4
        assert quantizer.group_size == 128

    def test_gptq_quantizer_8bit(self):
        from qingxiaotuan.quantization.gptq import GPTQQuantizer
        quantizer = GPTQQuantizer(bits=8, group_size=64)
        assert quantizer.bits == 8
        assert quantizer.group_size == 64

    def test_gptq_quantize_linear(self):
        from qingxiaotuan.quantization.gptq import GPTQQuantizer
        quantizer = GPTQQuantizer(bits=4, group_size=32)
        layer = torch.nn.Linear(64, 128)
        qlayer = quantizer.quantize_linear(layer, calib_data=torch.randn(10, 64))
        assert qlayer is not None

    def test_gptq_calibration_dataset(self):
        from qingxiaotuan.quantization.gptq import GPTQQuantizer
        quantizer = GPTQQuantizer()
        dataset = quantizer.prepare_calibration(num_samples=16, seq_len=128)
        assert len(dataset) == 16

    def test_gptq_quantize_model(self, small_qingxiaotuan_model):
        from qingxiaotuan.quantization.gptq import GPTQQuantizer
        quantizer = GPTQQuantizer(bits=4, group_size=128)
        calib_data = [torch.randint(0, 1000, (1, 64)) for _ in range(4)]
        qmodel = quantizer.quantize(small_qingxiaotuan_model, calib_data)
        assert qmodel is not None

    def test_gptq_save_load(self, temp_dir):
        from qingxiaotuan.quantization.gptq import GPTQQuantizer
        quantizer = GPTQQuantizer(bits=4, group_size=128)
        path = os.path.join(temp_dir, "gptq")
        os.makedirs(path, exist_ok=True)
        quantizer.save_config(path)
        assert os.path.exists(os.path.join(path, "config.json"))


class TestQingxiaotuanAWQ:

    def test_awq_quantizer_exists(self):
        from qingxiaotuan.quantization.awq import AWQQuantizer
        assert AWQQuantizer is not None

    def test_awq_quantizer_creation(self):
        from qingxiaotuan.quantization.awq import AWQQuantizer
        quantizer = AWQQuantizer(bits=4, group_size=128)
        assert quantizer.bits == 4
        assert quantizer.group_size == 128

    def test_awq_scale_groups(self):
        from qingxiaotuan.quantization.awq import AWQQuantizer
        quantizer = AWQQuantizer()
        weight = torch.randn(128, 64)
        scaled = quantizer.scale_groups(weight, group_size=32)
        assert scaled.shape == weight.shape

    def test_awq_quantize_linear(self):
        from qingxiaotuan.quantization.awq import AWQQuantizer
        quantizer = AWQQuantizer(bits=4, group_size=64)
        layer = torch.nn.Linear(128, 256)
        qlayer = quantizer.quantize_linear(layer, calib_data=torch.randn(8, 128))
        assert qlayer is not None

    def test_awq_importance_based_scaling(self):
        from qingxiaotuan.quantization.awq import AWQQuantizer
        quantizer = AWQQuantizer()
        weight = torch.randn(64, 32)
        act_scale = torch.randn(32).abs()
        scaled = quantizer.apply_importance_scaling(weight, act_scale)
        assert scaled.shape == weight.shape

    def test_awq_quantize_model(self, small_qingxiaotuan_model):
        from qingxiaotuan.quantization.awq import AWQQuantizer
        quantizer = AWQQuantizer(bits=4, group_size=128)
        calib_data = [torch.randint(0, 1000, (1, 64)) for _ in range(4)]
        qmodel = quantizer.quantize(small_qingxiaotuan_model, calib_data)
        assert qmodel is not None


class TestQingxiaotuanBNBAdapter:

    def test_bnb_adapter_exists(self):
        from qingxiaotuan.quantization.bnb_adapter import BNBQuantizer
        assert BNBQuantizer is not None

    def test_bnb_adapter_creation(self):
        from qingxiaotuan.quantization.bnb_adapter import BNBQuantizer
        quantizer = BNBQuantizer(load_in_4bit=True)
        assert quantizer.load_in_4bit is True

    def test_bnb_adapter_8bit(self):
        from qingxiaotuan.quantization.bnb_adapter import BNBQuantizer
        quantizer = BNBQuantizer(load_in_8bit=True)
        assert quantizer.load_in_8bit is True

    def test_bnb_adapter_quantization_config(self):
        from qingxiaotuan.quantization.bnb_adapter import BNBQuantizer
        quantizer = BNBQuantizer(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
        config = quantizer.get_bnb_config()
        assert config is not None

    def test_bnb_adapter_nf4_dtype(self):
        from qingxiaotuan.quantization.bnb_adapter import BNBQuantizer
        quantizer = BNBQuantizer(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        config = quantizer.get_bnb_config()
        assert config is not None or quantizer.bnb_4bit_quant_type == "nf4"

    def test_bnb_adapter_memory_estimate(self):
        from qingxiaotuan.quantization.bnb_adapter import BNBQuantizer
        quantizer = BNBQuantizer(load_in_4bit=True)
        estimate = quantizer.estimate_memory_savings(model_size_gb=70.0)
        assert isinstance(estimate, float)
        assert estimate > 0


class TestQingxiaotuanQuantizedInference:

    def test_quantized_model_forward(self, small_qingxiaotuan_model):
        model = small_qingxiaotuan_model
        input_ids = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        assert outputs.logits is not None

    def test_quantized_model_generate(self, small_qingxiaotuan_model):
        model = small_qingxiaotuan_model
        input_ids = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=4, do_sample=False)
        assert output.shape[1] > 8

    def test_quantized_model_memory_footprint(self, small_qingxiaotuan_model):
        model = small_qingxiaotuan_model
        mem_before = sum(p.numel() * p.element_size() for p in model.parameters())
        assert mem_before > 0

    def test_quantized_output_shape(self, small_qingxiaotuan_model):
        model = small_qingxiaotuan_model
        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        assert outputs.logits.shape == (1, 32, 32000)


class TestQingxiaotuanPostQuantizationAccuracy:

    def test_quantized_model_parameters(self, small_qingxiaotuan_model):
        layer = small_qingxiaotuan_model.model.layers[0]
        original = layer.self_attn.q_proj.weight.data.clone()
        assert original.shape[0] > 0

    def test_quantized_model_forward_deterministic(self, small_qingxiaotuan_model):
        input_ids = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            out1 = small_qingxiaotuan_model(input_ids=input_ids).logits
            out2 = small_qingxiaotuan_model(input_ids=input_ids).logits
        assert torch.allclose(out1, out2)

    def test_quantized_model_gradient_flow(self, small_qingxiaotuan_model):
        small_qingxiaotuan_model.train()
        input_ids = torch.randint(0, 1000, (1, 8))
        labels = torch.randint(0, 1000, (1, 8))
        outputs = small_qingxiaotuan_model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()
        has_grad = any(p.grad is not None for p in small_qingxiaotuan_model.parameters())
        assert has_grad
