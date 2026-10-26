"""
青小团注意力机制测试 - 覆盖FlashAttention、Attention Sink、Sliding Window、KV Cache量化等
"""

import pytest
import torch
from qingxiaotuan.modeling_qingxiaotuan import QingxiaotuanAttention


class TestQingxiaotuanFlashAttention:

    def test_flash_attn_availability_check(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        available = attn._check_flash_attn()
        assert available is not None

    def test_flash_attn_flag_in_config(self, flash_attn_config):
        assert flash_attn_config.use_flash_attention is True

    def test_flash_attn_enabled_creation(self, flash_attn_config):
        attn = QingxiaotuanAttention(flash_attn_config)
        assert attn.use_flash_attention is True

    def test_flash_attn_forward_shape(self, flash_attn_config):
        attn = QingxiaotuanAttention(flash_attn_config)
        hidden = torch.randn(1, 8, 256)
        pos = torch.arange(8).unsqueeze(0)
        out, _ = attn(hidden, position_ids=pos)
        assert out.shape == (1, 8, 256)


class TestQingxiaotuanStandardAttention:

    def test_standard_attention_forward(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(2, 16, 256)
        pos = torch.arange(16).unsqueeze(0).expand(2, -1)
        out, _ = attn(hidden, position_ids=pos)
        assert out.shape == (2, 16, 256)

    def test_standard_attention_causal_mask(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(1, 4, 256)
        pos = torch.arange(4).unsqueeze(0)
        out1, _ = attn(hidden, position_ids=pos)
        out2, _ = attn(hidden, position_ids=pos)
        assert torch.allclose(out1, out2)

    def test_attention_head_dim(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        assert attn.head_dim == 64

    def test_attention_o_proj_shape(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        assert attn.o_proj.weight.shape == (256, 256)


class TestQingxiaotuanAttentionSink:

    def test_attention_sink_enabled(self, small_qingxiaotuan_config):
        assert small_qingxiaotuan_config.attention_sink_tokens == 2

    def test_attention_sink_zero(self):
        from qingxiaotuan import QingxiaotuanConfig
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, attention_sink_tokens=0,
        )
        attn = QingxiaotuanAttention(config)
        assert attn.attention_sink_tokens == 0

    def test_attention_sink_forward_with_sink(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(1, 16, 256)
        pos = torch.arange(16).unsqueeze(0)
        out, _ = attn(hidden, position_ids=pos)
        assert out.shape == (1, 16, 256)

    def test_attention_sink_larger_sink(self):
        from qingxiaotuan import QingxiaotuanConfig
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, attention_sink_tokens=16,
        )
        attn = QingxiaotuanAttention(config)
        hidden = torch.randn(1, 32, 256)
        pos = torch.arange(32).unsqueeze(0)
        out, _ = attn(hidden, position_ids=pos)
        assert out.shape == (1, 32, 256)

    def test_attention_sink_mask_shape(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        attn.training = True
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 16, 64)
        scores = torch.matmul(q, k.transpose(2, 3)) / 8.0
        masked = attn._apply_attention_sink_mask(
            scores, 8, 16, scores.device, scores.dtype
        )
        assert masked.shape == (1, 4, 8, 16)

    def test_attention_sink_preserves_first_tokens(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        attn.training = True
        q = torch.randn(1, 4, 8, 64)
        k = torch.randn(1, 4, 16, 64)
        scores = torch.ones(1, 4, 8, 16)
        masked = attn._apply_attention_sink_mask(scores, 8, 16, scores.device, scores.dtype)
        assert not torch.any(torch.isinf(masked[:, :, :, :2]))


class TestQingxiaotuanSlidingWindow:

    def test_sliding_window_enabled(self, mid_qingxiaotuan_config):
        attn = QingxiaotuanAttention(mid_qingxiaotuan_config)
        assert attn.sliding_window_size == 512

    def test_sliding_window_disabled(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        assert attn.sliding_window_size is None

    def test_sliding_window_forward(self, mid_qingxiaotuan_config):
        attn = QingxiaotuanAttention(mid_qingxiaotuan_config)
        attn.training = False
        hidden = torch.randn(1, 32, 1024)
        pos = torch.arange(32).unsqueeze(0)
        out, _ = attn(hidden, position_ids=pos)
        assert out.shape == (1, 32, 1024)

    def test_sliding_window_falls_back(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        attn.training = False
        attn.sliding_window_size = 512
        q = torch.randn(1, 4, 16, 64)
        k = torch.randn(1, 4, 8, 64)
        v = torch.randn(1, 4, 8, 64)
        result = attn._sliding_window_attention(q, k, v, None)
        assert result.shape == (1, 4, 16, 64)


class TestQingxiaotuanKVCacheQuantization:

    def test_kv_cache_quantization_flag(self, mid_qingxiaotuan_config):
        assert mid_qingxiaotuan_config.kv_cache_quantization is True

    def test_kv_cache_quantize_shape(self, mid_qingxiaotuan_config):
        attn = QingxiaotuanAttention(mid_qingxiaotuan_config)
        k = torch.randn(1, 4, 16, 64)
        v = torch.randn(1, 4, 16, 64)
        (k_q, k_scale), (v_q, v_scale) = attn._quantize_kv_cache(k, v)
        assert k_q.dtype == torch.int8
        assert v_q.dtype == torch.int8
        assert k_scale.shape == v_scale.shape

    def test_kv_cache_dequantize_shape(self, mid_qingxiaotuan_config):
        attn = QingxiaotuanAttention(mid_qingxiaotuan_config)
        k_q = torch.randint(-127, 127, (1, 4, 16, 64), dtype=torch.int8)
        v_q = torch.randint(-127, 127, (1, 4, 16, 64), dtype=torch.int8)
        k_scale = torch.randn(1, 4, 16, 1).abs().clamp(min=1e-5)
        v_scale = torch.randn(1, 4, 16, 1).abs().clamp(min=1e-5)
        k_dq, v_dq = attn._dequantize_kv_cache((k_q, k_scale), (v_q, v_scale))
        assert k_dq.shape == (1, 4, 16, 64)
        assert v_dq.shape == (1, 4, 16, 64)

    def test_kv_cache_quantization_disabled(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        k = torch.randn(1, 4, 16, 64)
        v = torch.randn(1, 4, 16, 64)
        result_k, result_v = attn._quantize_kv_cache(k, v)
        assert result_k is k
        assert result_v is v

    def test_kv_cache_dequantization_disabled(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        k = torch.randn(1, 4, 16, 64)
        v = torch.randn(1, 4, 16, 64)
        result_k, result_v = attn._dequantize_kv_cache(k, v)
        assert result_k is k
        assert result_v is v

    def test_kv_cache_quantization_scale(self, mid_qingxiaotuan_config):
        attn = QingxiaotuanAttention(mid_qingxiaotuan_config)
        k = torch.randn(1, 4, 16, 64) * 10
        v = torch.randn(1, 4, 16, 64) * 10
        (k_q, k_scale), (v_q, v_scale) = attn._quantize_kv_cache(k, v)
        assert k_scale.min().item() > 0
        assert v_scale.min().item() > 0


class TestQingxiaotuanLongSequenceAttention:

    def test_long_sequence_forward(self):
        from qingxiaotuan import QingxiaotuanConfig
        config = QingxiaotuanConfig(
            vocab_size=32000, hidden_size=256, intermediate_size=1024,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=16384,
        )
        attn = QingxiaotuanAttention(config)
        hidden = torch.randn(1, 4096, 256)
        pos = torch.arange(4096).unsqueeze(0)
        out, _ = attn(hidden, position_ids=pos)
        assert out.shape == (1, 4096, 256)

    def test_attention_with_attention_mask_2d(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(1, 8, 256)
        pos = torch.arange(8).unsqueeze(0)
        mask = torch.ones(1, 8, dtype=torch.long)
        out, _ = attn(hidden, attention_mask=mask, position_ids=pos)
        assert out.shape == (1, 8, 256)

    def test_attention_with_padding_mask(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(1, 8, 256)
        pos = torch.arange(8).unsqueeze(0)
        mask = torch.ones(1, 8, dtype=torch.long)
        mask[0, 6:] = 0
        out, _ = attn(hidden, attention_mask=mask, position_ids=pos)
        assert out.shape == (1, 8, 256)

    def test_attention_output_is_deterministic(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(1, 8, 256)
        pos = torch.arange(8).unsqueeze(0)
        torch.manual_seed(42)
        out1, _ = attn(hidden, position_ids=pos)
        torch.manual_seed(42)
        out2, _ = attn(hidden, position_ids=pos)
        assert torch.allclose(out1, out2)
