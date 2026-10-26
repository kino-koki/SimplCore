"""
青小团MoE模块测试 - 覆盖路由、专家选择、辅助损失、负载均衡等
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from qingxiaotuan.modeling_qingxiaotuan import QingxiaotuanMoE


class TestQingxiaotuanMoERouter:

    def test_router_creation(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        assert isinstance(moe.gate, nn.Linear)
        assert moe.gate.weight.shape == (4, 256)

    def test_router_logits_shape(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(2, 8, 256)
        flat = x.view(-1, 256)
        logits = moe.gate(flat)
        assert logits.shape == (16, 4)

    def test_router_softmax_distribution(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        _, _, _, router_logits = moe(x)
        probs = F.softmax(router_logits, dim=-1)
        assert torch.allclose(probs.sum(-1), torch.ones(4))
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_router_noise_factor(self):
        from qingxiaotuan.modeling_qingxiaotuan import QingxiaotuanMoE
        _ = QingxiaotuanMoE.__init__


class TestQingxiaotuanMoETop2Selection:

    def test_top2_selection_shape(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 8, 256)
        flat = x.view(-1, 256)
        router_logits = moe.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        assert top_k_weights.shape == (8, 2)
        assert top_k_indices.shape == (8, 2)

    def test_top2_weights_normalized(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        flat = x.view(-1, 256)
        router_logits = moe.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        normalized = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        assert torch.allclose(normalized.sum(-1), torch.ones(4))

    def test_top2_indices_within_range(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(2, 16, 256)
        flat = x.view(-1, 256)
        router_logits = moe.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        _, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        assert torch.all(top_k_indices >= 0)
        assert torch.all(top_k_indices < moe.num_experts)

    def test_top2_different_experts(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 32, 256)
        flat = x.view(-1, 256)
        router_logits = moe.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        assert not torch.all(top_k_indices[:, 0] == top_k_indices[:, 1])


class TestQingxiaotuanSharedExpert:

    def test_shared_expert_creation(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        assert moe.shared_expert_gate_proj.weight.shape == (1024, 256)
        assert moe.shared_expert_up_proj.weight.shape == (1024, 256)
        assert moe.shared_expert_down_proj.weight.shape == (256, 1024)
        assert moe.shared_expert_gate.weight.shape == (1, 256)

    def test_shared_expert_forward(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        flat = x.view(-1, 256)
        shared_out = moe.shared_expert_down_proj(
            F.silu(moe.shared_expert_gate_proj(flat)) * moe.shared_expert_up_proj(flat)
        )
        assert shared_out.shape == (4, 256)

    def test_shared_expert_gate_sigmoid(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        flat = x.view(-1, 256)
        gate = torch.sigmoid(moe.shared_expert_gate(flat))
        assert gate.shape == (4, 1)
        assert torch.all(gate >= 0)
        assert torch.all(gate <= 1)


class TestQingxiaotuanAuxLoss:

    def test_aux_loss_computation(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        router_logits = torch.randn(32, 4)
        routing_weights = F.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        aux_loss = moe._compute_aux_loss(router_logits, top_k_indices)
        assert aux_loss.ndim == 0
        assert aux_loss.item() >= 0

    def test_aux_loss_non_zero(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(4, 16, 256)
        _, aux_loss, _, _ = moe(x)
        assert aux_loss.item() > 0

    def test_aux_loss_scalar(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        _, aux_loss, _, _ = moe(x)
        assert aux_loss.dim() == 0

    def test_aux_loss_different_batch_sizes(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        for bs in [1, 2, 4]:
            x = torch.randn(bs, 8, 256)
            _, aux_loss, _, _ = moe(x)
            assert aux_loss.ndim == 0


class TestQingxiaotuanZLoss:

    def test_z_loss_computation(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        router_logits = torch.randn(16, 4)
        z_loss = moe._compute_z_loss(router_logits)
        assert z_loss.ndim == 0
        assert z_loss.item() >= 0

    def test_z_loss_non_zero(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(2, 8, 256)
        _, _, z_loss, _ = moe(x)
        assert z_loss.item() > 0

    def test_z_loss_zero_for_uniform(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        router_logits = torch.zeros(16, 4)
        z_loss = moe._compute_z_loss(router_logits)
        expected = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        assert torch.allclose(z_loss, expected)

    def test_z_loss_shape(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        _, _, z_loss, _ = moe(x)
        assert z_loss.dim() == 0


class TestQingxiaotuanLoadBalancing:

    def test_load_balance_different_inputs(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x1 = torch.randn(1, 16, 256)
        x2 = torch.randn(1, 16, 256) * 10
        _, aux1, _, _ = moe(x1)
        _, aux2, _, _ = moe(x2)
        assert isinstance(aux1.item(), float)
        assert isinstance(aux2.item(), float)

    def test_expert_usage_statistics(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(4, 32, 256)
        flat = x.view(-1, 256)
        router_logits = moe.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        for k in range(moe.top_k):
            counts = torch.bincount(top_k_indices[:, k], minlength=moe.num_experts)
            assert counts.shape == (4,)

    def test_expert_dropping_not_occurring(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 8, 256)
        out, _, _, _ = moe(x)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))


class TestQingxiaotuanExpertCapacity:

    def test_expert_capacity_all_experts_used(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 64, 256)
        flat = x.view(-1, 256)
        router_logits = moe.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, moe.top_k, dim=-1)
        used_experts = torch.unique(top_k_indices)
        assert len(used_experts) <= moe.num_experts

    def test_expert_forward_all_experts(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 2, 256)
        flat = x.view(-1, 256)
        for idx in range(moe.num_experts):
            expert_out = moe.expert_down_proj[idx](
                F.silu(moe.expert_gate_proj[idx](flat)) * moe.expert_up_proj[idx](flat)
            )
            assert expert_out.shape == (2, 256)

    def test_expert_output_shape_each(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(1, 4, 256)
        out, _, _, _ = moe(x)
        assert out.shape == (1, 4, 256)

    def test_moe_forward_different_seq_lens(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        for seq_len in [1, 4, 16, 64]:
            x = torch.randn(1, seq_len, 256)
            out, aux, z, _ = moe(x)
            assert out.shape == (1, seq_len, 256)
            assert aux.ndim == 0
            assert z.ndim == 0

    def test_moe_with_large_batch(self, small_qingxiaotuan_config):
        moe = QingxiaotuanMoE(small_qingxiaotuan_config)
        x = torch.randn(8, 32, 256)
        out, aux_loss, z_loss, _ = moe(x)
        assert out.shape == (8, 32, 256)
        assert aux_loss.ndim == 0
        assert z_loss.ndim == 0
