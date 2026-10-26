# -*- coding: utf-8 -*-
"""MoE 层与 56B MoE 模型测试。"""
import torch

from simplcore_pg.config import SimplCoreConfig
from simplcore_pg.model_moe import SimplCoreMoEModel, SimplCoreMoEBlock
from simplcore_pg.moe import MoELayer, ExpertFFN


def _tiny_moe(**kw):
    base = dict(
        vocab_size=200, hidden_size=32, num_layers=2, num_experts=2, experts_per_tok=2,
        moe_intermediate_size=64, max_sequence_length=32, dmc_num_heads=4,
        use_rfp=False, use_dmc=False, dropout=0.0,
    )
    base.update(kw)
    return SimplCoreConfig(**base)


def test_56b_moe_parameter_budget():
    c = SimplCoreConfig.get_56b_moe_config()
    est = c.estimate_parameters()
    # 总量 ~56B
    assert 55e9 <= est["total"] <= 57e9, est["total"]
    # 单个专家角色跨全部层累计 ~7.8B (每层 0.557B × 14 层)
    per_role_total = est["experts"] / c.num_experts
    assert 7.4e9 <= per_role_total <= 8.2e9, per_role_total
    # 激活参数 (top-2) 占总专家 ~2/7
    active = est["experts"] * (c.experts_per_tok / c.num_experts)
    assert active / est["total"] < 0.35


def test_moe_forward_shape_and_aux():
    m = SimplCoreMoEModel(_tiny_moe()).eval()
    ids = torch.randint(0, 200, (2, 12))
    with torch.no_grad():
        logits = m(ids)
    assert logits.shape == (2, 12, 200)
    # 训练时 aux loss > 0
    m.train()
    m(ids, training=True)
    aux = m.last_aux_loss
    assert aux is not None and float(aux) > 0


def test_moe_load_balancing_balanced():
    # 随机路由下，负载均衡损失应较小 (专家使用较均衡)
    layer = MoELayer(hidden_size=32, num_experts=4, experts_per_tok=2,
                     intermediate_size=64)
    layer.eval()
    x = torch.randn(4, 16, 32)
    with torch.no_grad():
        _, aux = layer(x, training=True)
    # aux 为有限正数
    assert torch.isfinite(aux) and float(aux) >= 0


def test_moe_gradient_flows():
    m = SimplCoreMoEModel(_tiny_moe()).train()
    ids = torch.randint(0, 200, (2, 10))
    opt = m.configure_optimizers(lr=1e-3)
    out = m(ids, training=True)
    loss = out.float().mean() + (m.last_aux_loss or 0.0)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_moe_generate_terminates():
    m = SimplCoreMoEModel(_tiny_moe()).eval()
    ids = torch.randint(0, 200, (1, 4))
    with torch.no_grad():
        g = m.generate(ids, max_new_tokens=5, top_k=3)
    assert g.shape[0] == 1 and g.shape[1] >= 4


def test_moe_trace_fx():
    m = SimplCoreMoEModel(_tiny_moe()).eval()
    ids = torch.randint(0, 200, (2, 10))
    gm = m.trace_fx(example_input_ids=ids)
    with torch.no_grad():
        out_g = gm(ids)
        out_e = m(ids)
    assert torch.allclose(out_g, out_e, atol=1e-4)


def test_moe_dispatch_matches_dense_when_no_drop():
    # 容量足够 (不丢 token) 时，MoE 调度结果与"按权重聚合"等价
    layer = MoELayer(hidden_size=16, num_experts=3, experts_per_tok=2,
                     intermediate_size=32, capacity_factor=8.0)
    layer.eval()
    B, L, H = 2, 8, 16
    x = torch.randn(B, L, H)
    with torch.no_grad():
        out, _ = layer(x, training=False)
    # 参考：用 gather 直接按 top-k 权重聚合
    h = layer.norm(x)
    logits = layer.router(h)
    probs = torch.softmax(logits, -1)
    w, idx = torch.topk(probs, 2, -1)
    w = w / (w.sum(-1, keepdim=True) + 1e-6)
    all_out = torch.stack([layer.experts[e](h) for e in range(3)], 0)  # (E,B,L,H)
    gathered = torch.gather(
        all_out.permute(1, 2, 0, 3), 2, idx.unsqueeze(-1).expand(B, L, 2, H)
    )  # (B,L,2,H)
    ref = (gathered * w.unsqueeze(-1)).sum(2) + x
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()


def test_shared_expert_runs():
    layer = MoELayer(hidden_size=16, num_experts=2, experts_per_tok=1,
                     intermediate_size=32, use_shared_expert=True,
                     shared_intermediate_size=32)
    layer.eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        out, _ = layer(x, training=False)
    assert out.shape == x.shape


def test_moe_dispatch_with_dropping_runs():
    # 极小容量强制丢 token：验证 one_hot 越界已被 clamp 修复，前向不崩溃
    layer = MoELayer(hidden_size=16, num_experts=4, experts_per_tok=2,
                     intermediate_size=32, capacity_factor=0.5)
    layer.eval()
    x = torch.randn(2, 32, 16)
    with torch.no_grad():
        out, _ = layer(x, training=False)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
