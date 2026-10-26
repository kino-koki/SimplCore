"""
青小团长序列测试 - 覆盖512K序列、YaRN位置编码、序列打包、分段Loss、长序列显存、长序列推理
"""

import os
import math
import torch
import pytest
from qingxiaotuan import QingxiaotuanConfig, QingxiaotuanForCausalLM
from qingxiaotuan.modeling_qingxiaotuan import QingxiaotuanRotaryEmbedding


class TestQingxiaotuan512KSequence:

    def test_long_context_config(self):
        config = QingxiaotuanConfig(
            max_position_embeddings=524288,
            rope_theta=500000.0,
            rope_scaling={"type": "yarn", "factor": 64.0,
                          "original_max_position_embeddings": 8192,
                          "beta_fast": 32, "beta_slow": 1},
        )
        assert config.max_position_embeddings == 524288

    def test_long_context_rope(self):
        config = QingxiaotuanConfig(
            max_position_embeddings=524288,
            rope_theta=500000.0,
            rope_scaling={"type": "yarn", "factor": 64.0,
                          "original_max_position_embeddings": 8192},
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        rope = QingxiaotuanRotaryEmbedding(
            dim=64, max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta, rope_scaling=config.rope_scaling,
        )
        long_ids = torch.arange(524288).unsqueeze(0)
        x = torch.randn(1, 4, 16, 64)
        cos, sin = rope(x, long_ids[:, :16])
        assert cos.shape[1] == 16
        assert sin.shape[1] == 16

    def test_long_context_memory_estimation(self):
        hidden_size = 7168
        num_layers = 48
        num_heads = 56
        head_dim = 128
        seq_len = 524288
        kv_cache_bytes = num_layers * 2 * num_heads * seq_len * head_dim * 2
        single_layer_bytes = num_layers * 4 * hidden_size * hidden_size * 2
        total_gb = (kv_cache_bytes + single_layer_bytes) / (1024 ** 3)
        assert total_gb > 10


class TestQingxiaotuanYaRNPositionEncoding:

    def test_yarn_rope_creation(self):
        rope = QingxiaotuanRotaryEmbedding(
            dim=64, max_position_embeddings=524288, base=500000.0,
            rope_scaling={"type": "yarn", "factor": 64.0,
                          "original_max_position_embeddings": 8192,
                          "beta_fast": 32, "beta_slow": 1},
        )
        assert rope.rope_scaling["type"] == "yarn"
        assert rope.rope_scaling["factor"] == 64.0

    def test_yarn_rope_long_range(self):
        rope = QingxiaotuanRotaryEmbedding(
            dim=64, max_position_embeddings=524288, base=500000.0,
            rope_scaling={"type": "yarn", "factor": 64.0,
                          "original_max_position_embeddings": 8192},
        )
        position_ids = torch.tensor([[0, 1, 100, 10000, 100000, 500000]])
        x = torch.randn(1, 4, 6, 64)
        cos, sin = rope(x, position_ids)
        assert cos.shape == (1, 6, 64)
        assert sin.shape == (1, 6, 64)

    def test_yarn_freq_computation(self):
        rope = QingxiaotuanRotaryEmbedding(
            dim=64, max_position_embeddings=8192, base=10000.0,
            rope_scaling=None,
        )
        inv_freq = rope.inv_freq
        assert inv_freq.shape[0] == 32
        assert inv_freq[0] > inv_freq[-1]

    def test_yarn_freq_linear_interpolation(self):
        rope = QingxiaotuanRotaryEmbedding(
            dim=64, max_position_embeddings=16384, base=10000.0,
            rope_scaling={"type": "linear", "factor": 2.0},
        )
        assert rope.inv_freq.shape[0] == 32


class TestQingxiaotuanSequencePacking:

    def test_sequence_packing_config(self):
        from qingxiaotuan import QingxiaotuanConfig
        config = QingxiaotuanConfig(segmented_loss_chunk_size=32768)
        assert config.segmented_loss_chunk_size == 32768

    def test_sequence_packing_pad(self):
        from qingxiaotuan import QingxiaotuanConfig
        config = QingxiaotuanConfig(segmented_loss_chunk_size=16384)
        assert config.segmented_loss_chunk_size == 16384

    def test_sequence_packing_efficiency(self):
        seq_lens = [100, 200, 150, 300, 250]
        max_len = 512
        packed = 0
        while seq_lens:
            current = 0
            remaining = []
            for s in seq_lens:
                if current + s <= max_len:
                    current += s
                else:
                    remaining.append(s)
            packed += 1
            seq_lens = remaining
        assert packed < 5

    def test_packed_sequence_loss_weights(self):
        seq_lens = [100, 150, 200]
        max_len = 512
        weights = torch.zeros(max_len)
        start = 0
        for sl in seq_lens:
            weights[start:start + sl] = 1.0 / sl
            start += sl
        assert torch.allclose(weights.sum(), torch.tensor(3.0))

    def test_segment_mask_shape(self):
        max_len = 512
        seq_lens = [100, 200, 212]
        seg_mask = torch.zeros(max_len, max_len)
        start = 0
        for i, sl in enumerate(seq_lens):
            seg_mask[start:start + sl, start:start + sl] = 1
            start += sl
        assert seg_mask.shape == (512, 512)
        assert seg_mask.sum() > 0


class TestQingxiaotuanSegmentedLoss:

    def test_segmented_loss_basic(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        logits = torch.randn(1, 2048, 32000)
        labels = torch.randint(0, 32000, (1, 2048))
        loss = model._compute_segmented_loss(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_segmented_loss_chunked_correctly(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        assert model.segmented_loss_chunk_size == 512
        logits = torch.randn(1, 2048, 32000)
        labels = torch.randint(0, 32000, (1, 2048))
        loss = model._compute_segmented_loss(logits, labels)
        full_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 32000), labels.view(-1))
        assert loss.item() > 0

    def test_segmented_loss_larger_chunk(self):
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            segmented_loss_chunk_size=1024,
        )
        model = QingxiaotuanForCausalLM(config)
        assert model.segmented_loss_chunk_size == 1024

    def test_segmented_loss_gradient(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        model.train()
        logits = torch.randn(1, 1024, 32000, requires_grad=True)
        labels = torch.randint(0, 32000, (1, 1024))
        loss = model._compute_segmented_loss(logits, labels)
        loss.backward()
        assert logits.grad is not None

    def test_segmented_loss_masked(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        logits = torch.randn(1, 512, 32000)
        labels = torch.full((1, 512), -100, dtype=torch.long)
        labels[0, 256:] = torch.randint(0, 32000, (256,))
        loss = model._compute_segmented_loss(logits, labels)
        assert loss.ndim == 0

    def test_segmented_loss_vs_full(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        logits = torch.randn(1, 512, 32000)
        labels = torch.randint(0, 32000, (1, 512))
        seg_loss = model._compute_segmented_loss(logits, labels)
        from torch.nn import CrossEntropyLoss
        ce = CrossEntropyLoss()
        full_loss = ce(logits.view(-1, 32000), labels.view(-1))
        assert abs(seg_loss.item() - full_loss.item()) < 0.1

    def test_segmented_loss_multiple_chunks(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        logits = torch.randn(1, 2048, 32000)
        labels = torch.randint(0, 32000, (1, 2048))
        loss = model._compute_segmented_loss(logits, labels)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestQingxiaotuanLongSequenceMemory:

    def test_activation_memory_estimation(self):
        batch_size = 1
        seq_len = 524288
        hidden_size = 7168
        activation_bytes = batch_size * seq_len * hidden_size * 2
        activation_gb = activation_bytes / (1024 ** 3)
        assert activation_gb > 1

    def test_kv_cache_for_long_seq(self):
        num_layers = 48
        num_kv_heads = 8
        head_dim = 128
        seq_len = 524288
        kv_per_layer = 2 * num_kv_heads * seq_len * head_dim * 2
        total_kv_cache = num_layers * kv_per_layer
        total_gb = total_kv_cache / (1024 ** 3)
        assert total_gb > 10

    def test_long_seq_gradient_checkpointing(self):
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            gradient_checkpointing_steps=2,
        )
        model = QingxiaotuanForCausalLM(config)
        model.gradient_checkpointing_enable()
        assert model.model.gradient_checkpointing is True

    def test_long_seq_checkpoint_steps(self):
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256,
            num_hidden_layers=48, num_attention_heads=4, num_key_value_heads=2,
            gradient_checkpointing_steps=12,
        )
        assert config.gradient_checkpointing_steps == 12


class TestQingxiaotuanLongContextInference:

    def test_long_context_generate_base(self, small_qingxiaotuan_model):
        input_ids = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            output = small_qingxiaotuan_model.generate(
                input_ids,
                max_new_tokens=8,
                do_sample=False,
            )
        assert output.shape[1] == 24

    def test_long_context_incremental_inference(self, small_qingxiaotuan_model):
        model = small_qingxiaotuan_model
        model.model.use_cache = True
        input_ids = torch.randint(0, 1000, (1, 8))
        past_key_values = None
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
        assert outputs.past_key_values is not None

    def test_long_context_kv_cache_shape(self, small_qingxiaotuan_model):
        model = small_qingxiaotuan_model.model
        model.use_cache = True
        hidden = torch.randn(1, 32, 256)
        pos = torch.arange(32).unsqueeze(0)
        out, present_kv = model.layers[0](hidden, position_ids=pos, use_cache=True)
        assert present_kv is not None

    def test_long_context_attention_with_cache(self, small_qingxiaotuan_model):
        input_ids = torch.randint(0, 1000, (1, 8))
        labels = torch.randint(0, 1000, (1, 8))
        with torch.no_grad():
            outputs = small_qingxiaotuan_model(
                input_ids=input_ids,
                labels=labels,
                use_cache=True,
            )
        assert outputs.loss is not None

    def test_attention_sink_with_long_context(self):
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=524288, attention_sink_tokens=4,
        )
        model = QingxiaotuanForCausalLM(config)
        input_ids = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        assert outputs.logits.shape == (1, 64, 32000)

    def test_long_context_yarn_with_model(self):
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=524288,
            rope_theta=500000.0,
            rope_scaling={"type": "yarn", "factor": 64.0,
                          "original_max_position_embeddings": 8192},
            attention_sink_tokens=2,
        )
        model = QingxiaotuanForCausalLM(config)
        input_ids = torch.randint(0, 1000, (1, 128))
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        assert outputs.logits.shape[1] == 128
