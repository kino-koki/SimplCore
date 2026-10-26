"""
青小团模型核心模块测试 - 覆盖RMSNorm、RoPE、注意力、MLP、MoE、DecoderLayer、完整模型
"""

import pytest
import torch
import torch.nn as nn
from qingxiaotuan import QingxiaotuanConfig, QingxiaotuanForCausalLM
from qingxiaotuan.modeling_qingxiaotuan import (
    QingxiaotuanRMSNorm,
    QingxiaotuanRotaryEmbedding,
    QingxiaotuanAttention,
    QingxiaotuanMLP,
    QingxiaotuanMoE,
    QingxiaotuanDecoderLayer,
    QingxiaotuanModel,
    rotate_half,
    apply_rotary_pos_emb,
)


class TestQingxiaotuanRMSNorm:

    def test_rmsnorm_creation(self):
        rms = QingxiaotuanRMSNorm(hidden_size=256, eps=1e-6)
        assert rms.weight.shape == (256,)
        assert rms.variance_epsilon == 1e-6

    def test_rmsnorm_forward_shape(self):
        rms = QingxiaotuanRMSNorm(hidden_size=256)
        x = torch.randn(2, 32, 256)
        out = rms(x)
        assert out.shape == (2, 32, 256)

    def test_rmsnorm_normalization(self):
        rms = QingxiaotuanRMSNorm(hidden_size=64, eps=1e-6)
        x = torch.randn(4, 16, 64)
        out = rms(x)
        mean_square = out.pow(2).mean(-1)
        assert torch.allclose(mean_square, torch.ones_like(mean_square), atol=1e-5)

    def test_rmsnorm_weight_effect(self):
        rms = QingxiaotuanRMSNorm(hidden_size=32)
        rms.weight.data.fill_(2.0)
        x = torch.ones(1, 5, 32)
        out = rms(x)
        assert torch.allclose(out[0, 0], torch.full((32,), 2.0), atol=1e-5)


class TestQingxiaotuanRoPE:

    def test_rotary_embedding_creation(self):
        rope = QingxiaotuanRotaryEmbedding(dim=64, max_position_embeddings=8192, base=10000.0)
        assert rope.dim == 64
        assert rope.max_position_embeddings == 8192
        assert rope.base == 10000.0

    def test_rotary_embedding_forward(self):
        rope = QingxiaotuanRotaryEmbedding(dim=64)
        x = torch.randn(2, 8, 16, 64)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
        cos, sin = rope(x, position_ids)
        assert cos.shape == (2, 16, 64)
        assert sin.shape == (2, 16, 64)

    def test_yarn_rope_creation(self):
        rope = QingxiaotuanRotaryEmbedding(
            dim=64, max_position_embeddings=524288, base=500000.0,
            rope_scaling={"type": "yarn", "factor": 64.0,
                          "original_max_position_embeddings": 8192,
                          "beta_fast": 32, "beta_slow": 1}
        )
        x = torch.randn(1, 4, 8, 64)
        position_ids = torch.arange(8).unsqueeze(0)
        cos, sin = rope(x, position_ids)
        assert cos.shape == (1, 8, 64)

    def test_rotate_half_function(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        rotated = rotate_half(x)
        assert rotated.shape == (1, 4)
        assert torch.allclose(rotated[0], torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    def test_apply_rotary_pos_emb(self):
        q = torch.randn(2, 4, 16, 64)
        k = torch.randn(2, 4, 16, 64)
        cos = torch.randn(2, 16, 64)
        sin = torch.randn(2, 16, 64)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestQingxiaotuanGQAAttention:

    def test_attention_creation(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        assert attn.num_heads == 4
        assert attn.num_key_value_heads == 2
        assert attn.num_key_value_groups == 2
        assert attn.head_dim == 64

    def test_attention_projection_layers(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        assert isinstance(attn.q_proj, nn.Linear)
        assert isinstance(attn.k_proj, nn.Linear)
        assert isinstance(attn.v_proj, nn.Linear)
        assert isinstance(attn.o_proj, nn.Linear)
        assert attn.q_proj.weight.shape == (256, 256)
        assert attn.k_proj.weight.shape == (128, 256)
        assert attn.v_proj.weight.shape == (128, 256)

    def test_attention_forward_shape(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(2, 16, 256)
        position_ids = torch.arange(16).unsqueeze(0).expand(2, -1)
        output, past_kv = attn(hidden, position_ids=position_ids)
        assert output.shape == (2, 16, 256)
        assert past_kv is None

    def test_attention_with_cache(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(1, 8, 256)
        position_ids = torch.arange(8).unsqueeze(0)
        output, past_kv = attn(hidden, position_ids=position_ids, use_cache=True)
        assert output.shape == (1, 8, 256)
        assert past_kv is not None
        assert len(past_kv) == 2

    def test_attention_with_past_kv(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        h1 = torch.randn(1, 4, 256)
        pos1 = torch.arange(4).unsqueeze(0)
        _, past_kv = attn(h1, position_ids=pos1, use_cache=True)
        h2 = torch.randn(1, 1, 256)
        pos2 = torch.tensor([[4]])
        output, past_kv2 = attn(h2, position_ids=pos2, past_key_value=past_kv, use_cache=True)
        assert output.shape == (1, 1, 256)
        assert past_kv2 is not None

    def test_gqa_repeat_interleave(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        assert attn.num_key_value_groups == 2

    def test_attention_backward(self, small_qingxiaotuan_config):
        attn = QingxiaotuanAttention(small_qingxiaotuan_config)
        hidden = torch.randn(1, 8, 256, requires_grad=True)
        position_ids = torch.arange(8).unsqueeze(0)
        output, _ = attn(hidden, position_ids=position_ids)
        loss = output.sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.shape == (1, 8, 256)


class TestQingxiaotuanSwiGLUMLP:

    def test_mlp_creation(self, small_qingxiaotuan_config):
        mlp = QingxiaotuanMLP(small_qingxiaotuan_config)
        assert isinstance(mlp.gate_proj, nn.Linear)
        assert isinstance(mlp.up_proj, nn.Linear)
        assert isinstance(mlp.down_proj, nn.Linear)

    def test_mlp_forward_shape(self, small_qingxiaotuan_config):
        mlp = QingxiaotuanMLP(small_qingxiaotuan_config)
        x = torch.randn(2, 16, 256)
        out = mlp(x)
        assert out.shape == (2, 16, 256)

    def test_mlp_swiglu_activation(self, small_qingxiaotuan_config):
        mlp = QingxiaotuanMLP(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        gate_out = mlp.gate_proj(x)
        silu_out = nn.functional.silu(gate_out)
        up_out = mlp.up_proj(x)
        combined = silu_out * up_out
        assert combined.shape == (1, 4, 1024)

    def test_mlp_backward(self, small_qingxiaotuan_config):
        mlp = QingxiaotuanMLP(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256, requires_grad=True)
        out = mlp(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestQingxiaotuanMoE:

    def test_moe_creation(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        assert moe.num_experts == 4
        assert moe.top_k == 2
        assert len(moe.expert_gate_proj) == 4
        assert len(moe.expert_up_proj) == 4
        assert len(moe.expert_down_proj) == 4

    def test_moe_forward(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 8, 256)
        out, aux_loss, z_loss, router_logits = moe(x)
        assert out.shape == (1, 8, 256)
        assert aux_loss.ndim == 0
        assert z_loss.ndim == 0
        assert router_logits.shape == (8, 4)

    def test_moe_aux_loss(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(2, 16, 256)
        _, aux_loss, _, _ = moe(x)
        assert aux_loss.item() > 0

    def test_moe_z_loss(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(2, 16, 256)
        _, _, z_loss, _ = moe(x)
        assert z_loss.item() > 0

    def test_moe_shared_expert(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        assert hasattr(moe, 'shared_expert_gate_proj')
        assert hasattr(moe, 'shared_expert_down_proj')
        assert hasattr(moe, 'shared_expert_gate')

    def test_moe_backward(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256, requires_grad=True)
        out, _, _, _ = moe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_moe_router_logits(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        _, _, _, router_logits = moe(x)
        assert router_logits.shape == (4, 4)
        probs = torch.softmax(router_logits, dim=-1)
        assert torch.allclose(probs.sum(-1), torch.ones(4))


class TestQingxiaotuanDecoderLayer:

    def test_dense_decoder_layer(self, small_qingxiaotuan_config):
        layer = QingxiaotuanDecoderLayer(small_qingxiaotuan_config, layer_idx=0)
        assert hasattr(layer, 'input_layernorm')
        assert hasattr(layer, 'self_attn')
        assert isinstance(layer.mlp, QingxiaotuanMLP)

    def test_moe_decoder_layer(self, small_qingxiaotuan_config):
        layer = QingxiaotuanDecoderLayer(small_qingxiaotuan_config, layer_idx=1)
        assert isinstance(layer.mlp, QingxiaotuanMoE)

    def test_decoder_layer_forward_dense(self, small_qingxiaotuan_config):
        layer = QingxiaotuanDecoderLayer(small_qingxiaotuan_config, layer_idx=0)
        x = torch.randn(1, 8, 256)
        position_ids = torch.arange(8).unsqueeze(0)
        outputs = layer(x, position_ids=position_ids)
        hidden = outputs[0]
        assert hidden.shape == (1, 8, 256)
        assert outputs[2] is None
        assert outputs[3] is None

    def test_decoder_layer_forward_moe(self, small_qingxiaotuan_config):
        layer = QingxiaotuanDecoderLayer(small_qingxiaotuan_config, layer_idx=1)
        x = torch.randn(1, 8, 256)
        position_ids = torch.arange(8).unsqueeze(0)
        outputs = layer(x, position_ids=position_ids)
        assert outputs[0].shape == (1, 8, 256)
        assert outputs[2] is not None

    def test_decoder_layer_with_cache(self, small_qingxiaotuan_config):
        layer = QingxiaotuanDecoderLayer(small_qingxiaotuan_config, layer_idx=0)
        x = torch.randn(1, 4, 256)
        position_ids = torch.arange(4).unsqueeze(0)
        outputs = layer(x, position_ids=position_ids, use_cache=True)
        assert outputs[1] is not None

    def test_decoder_layer_backward(self, small_qingxiaotuan_config):
        layer = QingxiaotuanDecoderLayer(small_qingxiaotuan_config, layer_idx=0)
        x = torch.randn(1, 4, 256, requires_grad=True)
        position_ids = torch.arange(4).unsqueeze(0)
        outputs = layer(x, position_ids=position_ids)
        loss = outputs[0].sum()
        loss.backward()
        assert x.grad is not None


class TestQingxiaotuanModel:

    def test_model_creation(self, small_qingxiaotuan_config):
        model = QingxiaotuanModel(small_qingxiaotuan_config)
        assert model.embed_tokens.num_embeddings == 32000
        assert len(model.layers) == 2

    def test_model_forward(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model.model(
            input_ids=dummy_inputs["input_ids"],
            attention_mask=dummy_inputs["attention_mask"],
        )
        assert outputs.last_hidden_state.shape == (2, 64, 256)

    def test_model_forward_no_return_dict(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model.model(
            input_ids=dummy_inputs["input_ids"],
            return_dict=False,
        )
        assert len(outputs) >= 1

    def test_model_with_position_ids(self, small_qingxiaotuan_model, dummy_inputs):
        position_ids = torch.arange(64).unsqueeze(0).expand(2, -1)
        outputs = small_qingxiaotuan_model.model(
            input_ids=dummy_inputs["input_ids"],
            position_ids=position_ids,
        )
        assert outputs.last_hidden_state.shape == (2, 64, 256)

    def test_model_output_hidden_states(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model.model(
            input_ids=dummy_inputs["input_ids"],
            output_hidden_states=True,
        )
        assert outputs.hidden_states is not None
        assert len(outputs.hidden_states) == 3

    def test_model_moe_outputs(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model.model(
            input_ids=dummy_inputs["input_ids"],
        )
        assert hasattr(outputs, "moe_aux_losses")
        assert hasattr(outputs, "moe_z_losses")

    def test_model_gradient_checkpointing(self, small_qingxiaotuan_model, dummy_inputs):
        small_qingxiaotuan_model.model.gradient_checkpointing = True
        small_qingxiaotuan_model.model.train()
        outputs = small_qingxiaotuan_model.model(
            input_ids=dummy_inputs["input_ids"],
        )
        assert outputs.last_hidden_state.shape == (2, 64, 256)

    def test_model_backward(self, small_qingxiaotuan_model, dummy_inputs):
        small_qingxiaotuan_model.model.train()
        outputs = small_qingxiaotuan_model.model(
            input_ids=dummy_inputs["input_ids"],
        )
        loss = outputs.last_hidden_state.sum()
        loss.backward()
        assert small_qingxiaotuan_model.model.embed_tokens.weight.grad is not None


class TestQingxiaotuanForCausalLM:

    def test_causallm_creation(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        assert model.lm_head.weight.shape == (32000, 256)
        assert model.segmented_loss_chunk_size == 512

    def test_causallm_forward_logits(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model(
            input_ids=dummy_inputs["input_ids"],
        )
        assert outputs.logits.shape == (2, 64, 32000)

    def test_causallm_with_loss(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model(
            input_ids=dummy_inputs["input_ids"],
            labels=dummy_inputs["labels"],
        )
        assert outputs.loss is not None
        assert outputs.loss.ndim == 0

    def test_causallm_no_loss_without_labels(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model(
            input_ids=dummy_inputs["input_ids"],
        )
        assert outputs.loss is None

    def test_causallm_tied_weights(self, small_qingxiaotuan_model):
        assert hasattr(small_qingxiaotuan_model, '_tied_weights_keys')

    def test_causallm_get_set_embeddings(self, small_qingxiaotuan_model):
        embed = small_qingxiaotuan_model.get_input_embeddings()
        assert embed is not None
        new_embed = nn.Embedding(32000, 256)
        small_qingxiaotuan_model.set_input_embeddings(new_embed)
        assert small_qingxiaotuan_model.get_input_embeddings() is new_embed

    def test_causallm_get_set_output_embeddings(self, small_qingxiaotuan_model):
        head = small_qingxiaotuan_model.get_output_embeddings()
        assert head is not None

    def test_causallm_get_decoder(self, small_qingxiaotuan_model):
        decoder = small_qingxiaotuan_model.get_decoder()
        assert decoder is small_qingxiaotuan_model.model

    def test_prepare_inputs_for_generation(self, small_qingxiaotuan_model, dummy_inputs):
        prepared = small_qingxiaotuan_model.prepare_inputs_for_generation(
            input_ids=dummy_inputs["input_ids"],
            attention_mask=dummy_inputs["attention_mask"],
        )
        assert "input_ids" in prepared
        assert "attention_mask" in prepared

    def test_prepare_inputs_with_past(self, small_qingxiaotuan_model, dummy_inputs):
        past = [(torch.randn(1, 2, 4, 64), torch.randn(1, 2, 4, 64)) for _ in range(2)]
        prepared = small_qingxiaotuan_model.prepare_inputs_for_generation(
            input_ids=dummy_inputs["input_ids"],
            past_key_values=past,
        )
        assert prepared["input_ids"].shape[1] == 1


class TestQingxiaotuanBatchInference:

    def test_batch_forward(self, small_qingxiaotuan_model, batch_inputs):
        outputs = small_qingxiaotuan_model(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
        )
        assert outputs.logits.shape[0] == 4

    def test_batch_with_loss(self, small_qingxiaotuan_model, batch_inputs):
        outputs = small_qingxiaotuan_model(
            input_ids=batch_inputs["input_ids"],
            labels=batch_inputs["labels"],
        )
        assert outputs.loss is not None

    def test_batch_different_length(self, small_qingxiaotuan_model):
        input_ids = torch.randint(0, 1000, (3, 48))
        labels = torch.randint(0, 1000, (3, 48))
        outputs = small_qingxiaotuan_model(input_ids=input_ids, labels=labels)
        assert outputs.loss is not None


class TestQingxiaotuanGradientCheckpointing:

    def test_gradient_checkpointing_flag(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        model.gradient_checkpointing_enable()
        assert model.model.gradient_checkpointing is True

    def test_gradient_checkpointing_disable(self, small_qingxiaotuan_config):
        model = QingxiaotuanForCausalLM(small_qingxiaotuan_config)
        model.gradient_checkpointing_disable()
        assert model.model.gradient_checkpointing is False


class TestQingxiaotuanKVCache:

    def test_kv_cache_shape(self, small_qingxiaotuan_model, dummy_inputs):
        model = small_qingxiaotuan_model.model
        for layer in model.layers:
            out = layer(
                dummy_inputs["input_ids"].float(),
                position_ids=torch.arange(64).unsqueeze(0).expand(2, -1),
                use_cache=True,
            )
            present_kv = out[1]
            assert present_kv is not None

    def test_kv_cache_update(self, small_qingxiaotuan_model, dummy_inputs):
        model = small_qingxiaotuan_model.model
        hidden = dummy_inputs["input_ids"].float()
        pos_ids = torch.arange(64).unsqueeze(0).expand(2, -1)
        out1 = model.layers[0](hidden, position_ids=pos_ids, use_cache=True)
        kv1 = out1[1]
        hidden2 = torch.randn(2, 1, 256)
        pos2 = torch.tensor([[64], [64]])
        out2 = model.layers[0](hidden2, position_ids=pos2, past_key_value=kv1, use_cache=True)
        assert out2[1] is not None


class TestQingxiaotuanMoEForward:

    def test_moe_model_forward(self, small_qingxiaotuan_model, dummy_inputs):
        outputs = small_qingxiaotuan_model.model(input_ids=dummy_inputs["input_ids"])
        aux = outputs.get("moe_aux_losses", None)
        assert aux is not None
        if len(aux) > 0:
            assert aux[0].ndim == 0
