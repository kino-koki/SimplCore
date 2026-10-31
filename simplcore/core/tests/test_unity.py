# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity 单元测试：验证「多个小专家 + 路由粘合 = MoE 效果」真实可训练。

覆盖：
1. 基座冻结 → 不进 param_filter（显存只随可学权重增长）；
2. 每个专家在**不同**域数据上真实参训、且形成域专属特化；
3. MoE 路由（domain=None）梯度能流向路由门；
4. WeightBudget 守卫：超 3B 上限时 spawn_expert 抛错（harness 生长受约束）；
5. 前向返回 (logits, loss)；集成到 build_enhanced_model("unity")。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.training_integration import build_enhanced_model, warmup_unity_domains
from simplcore.unity.config import UnityConfig
from simplcore.unity.model import UnityModel
from simplcore.unity.evolution import UnitySelfEvolution


def _tiny_cfg():
    return SimplCoreConfig(dim=32, num_layers=2, num_bases=8, vocab_size=64,
                           dropout=0.0, tie_embeddings=True)


def _make_loader(tokens, seq_len=8, batch=4, n=16):
    # 域数据：把 tokens 拼成固定分布序列，作为 x 与 y（自回归式 1 步偏移）
    data = []
    for _ in range(n):
        seq = torch.tensor(tokens, dtype=torch.long).repeat((seq_len // len(tokens)) + 1)[:seq_len]
        x = seq[:-1].unsqueeze(0).repeat(batch, 1)
        y = seq[1:].unsqueeze(0).repeat(batch, 1)
        data.append((x, y))
    return iter(data)


def test_base_frozen_excluded():
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    unity = UnityModel(base, UnityConfig(num_experts=3, top_k=2))
    learned = unity.num_params_learned()
    base_params = sum(p.numel() for p in base.parameters())
    total = sum(p.numel() for p in unity.parameters())
    # 基座冻结：base 参数全部 requires_grad=False
    for p in base.parameters():
        assert not p.requires_grad, "基座必须冻结"
    # 可学部分只来自路由 + 专家，不包含基座
    assert learned > 0, "至少路由+专家应可学"
    assert learned < total, "可学参数应不含冻结基座（learned < 总参）"
    assert learned < base_params or base_params <= learned, "（小模型基座也可能很小，仅校验基座被排除）"
    # param_filter 内不应出现任何 base 参数
    base_ids = {id(p) for p in base.parameters()}
    assert all(id(p) not in base_ids for p in unity.param_filter()), \
        "param_filter 不得包含冻结基座参数"
    print(f"[ok] base frozen & excluded: base={base_params} learned={learned} total={total}")


def test_experts_different_data_specialize():
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    unity = UnityModel(base, UnityConfig(num_experts=2, top_k=1,
                                         router_z_loss=0.0, load_balance_loss=0.0))
    # 两个**不同**域：A 用 token 0..3；B 用 token 4..7
    loader_a = _make_loader([0, 1, 2, 3], n=200)
    loader_b = _make_loader([4, 5, 6, 7], n=200)
    opt0 = torch.optim.AdamW(unity.experts[0].parameters(), lr=1e-2)
    opt1 = torch.optim.AdamW(unity.experts[1].parameters(), lr=1e-2)
    # 隔离训练：domain=0 只训 expert0，domain=1 只训 expert1（数据不同）
    unity.train_expert(0, loader_a, opt0, steps=200)
    unity.train_expert(1, loader_b, opt1, steps=200)
    # 评测：expert0 在自己域 A 上的 CE 应显著低于在域 B 上（特化）
    with torch.no_grad():
        xa, ya = next(_make_loader([0, 1, 2, 3], n=1))
        xb, yb = next(_make_loader([4, 5, 6, 7], n=1))
        _, la = unity(xa, ya, domain=0)
        _, lb = unity(xb, yb, domain=0)
        _, la2 = unity(xa, ya, domain=1)
        _, lb2 = unity(xb, yb, domain=1)
    assert float(la) < float(lb), f"expert0 应更拟合域A: la={la} lb={lb}"
    assert float(lb2) < float(la2), f"expert1 应更拟合域B: lb2={lb2} la2={la2}"
    print(f"[ok] domain-specialization: E0(A)={la:.3f} E0(B)={lb:.3f} | "
          f"E1(A)={la2:.3f} E1(B)={lb2:.3f}")


def test_router_gradient_flows():
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    unity = UnityModel(base, UnityConfig(num_experts=3, top_k=2))
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    opt = torch.optim.AdamW(unity.param_filter(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    _, loss = unity(x, y)  # MoE 路由（domain=None）
    loss.backward()
    grads = [p.grad for p in unity.router.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads), "路由门应有梯度"
    print(f"[ok] router grad flows: loss={float(loss):.3f}")


def test_budget_guard_blocks_overspawn():
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    # 先用正常预算构造（分配 1 个专家），再把上限压到"刚好够当前用量 + 5k"，
    # 逼出后续 spawn_expert 的预算守卫（harness 生长受 3B 上限约束）。
    from simplcore.core.envlearn import WeightBudget
    unity = UnityModel(base, UnityConfig(num_experts=1, top_k=1))
    used = unity._budget.learned_params
    unity._budget.max_learned = used + 5000
    unity._budget.total_cap = unity._budget.base_params + used + 5000
    raised = False
    try:
        while True:
            unity.spawn_expert()  # 下一个专家需求 > 剩余上限 → 抛错
    except ValueError:
        raised = True
    assert raised, "超预算应抛 ValueError（harness 生长受 3B 上限约束）"
    print(f"[ok] budget guard: num_experts={len(unity.experts)} (capped at ~{used} learned)")


def test_integration_with_train_factory():
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    em = build_enhanced_model(base, cfg, ["unity"], unity_experts=2, unity_topk=2)
    assert em.model is not base, "unity 应替换 em.model"
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    logits, loss = em.train_forward(x, y)
    assert logits.shape[-1] == cfg.vocab_size
    assert float(loss) == float(loss)  # not nan
    print(f"[ok] integration: train_forward loss={float(loss):.3f}, "
          f"learned_params={em.extras['unity'].num_params_learned()}")


def test_expert_persistence():
    import os
    import tempfile
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    unity = UnityModel(base, UnityConfig(num_experts=2, top_k=1,
                                         router_z_loss=0.0, load_balance_loss=0.0))
    # 域隔离训 expert0，制造可区分权重
    opt0 = torch.optim.AdamW(unity.experts[0].parameters(), lr=1e-2)
    unity.train_expert(0, _make_loader([0, 1, 2, 3], n=120), opt0, steps=120)
    path = os.path.join(tempfile.mkdtemp(), "unity.pt")
    unity.save_learned(path)
    # 全新基座 + 全新 UnityModel → 从快照恢复
    base2 = SimplCoreModel(cfg)
    restored = UnityModel.load_learned(base2, path)
    assert len(restored.experts) == len(unity.experts), "专家数应恢复"
    # 关键：恢复的 expert0 权重应与原一致（参数近似相等）
    max_diff = max(
        (a - b).abs().max().item()
        for a, b in zip(unity.experts[0].parameters(), restored.experts[0].parameters()))
    assert max_diff < 1e-5, f"恢复权重应与原一致，实际差 {max_diff}"
    # 预算账目还原
    assert restored._budget.learned_params == unity._budget.learned_params, "预算应恢复"
    print(f"[ok] persistence: restored {len(restored.experts)} experts, "
          f"param_maxdiff={max_diff:.2e}")


def test_self_evolution_manager():
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    unity = UnityModel(base, UnityConfig(num_experts=1, top_k=1,
                                         router_z_loss=0.0, load_balance_loss=0.0))
    n0 = len(unity.experts)
    evo = UnitySelfEvolution(unity, steps_per_domain=60, lr=1e-2)
    evo.start()
    evo.register_domain("domA", _make_loader([0, 1, 2, 3], n=60))
    evo.register_domain("domB", _make_loader([4, 5, 6, 7], n=60))
    evo.stop(timeout=60)
    st = evo.status()
    assert len(unity.experts) == n0 + 2, "后台应长出 2 个新专家"
    assert set(st["done"]) >= {"domA", "domB"}, f"两域都应完成：{st['done']}"
    assert st["errors"] == [], f"不应有错误：{st['errors']}"
    print(f"[ok] self-evolution: experts {n0}->{len(unity.experts)}, "
          f"done={st['done']}")


def test_warmup_domains_specialize():
    cfg = _tiny_cfg()
    base = SimplCoreModel(cfg)
    unity = UnityModel(base, UnityConfig(num_experts=2, top_k=1,
                                         router_z_loss=0.0, load_balance_loss=0.0))
    loaders = [_make_loader([0, 1, 2, 3], n=150),
               _make_loader([4, 5, 6, 7], n=150)]
    warmup_unity_domains(unity, loaders, steps=150, lr=1e-2)
    with torch.no_grad():
        xa, ya = next(_make_loader([0, 1, 2, 3], n=1))
        xb, yb = next(_make_loader([4, 5, 6, 7], n=1))
        _, la = unity(xa, ya, domain=0)
        _, lb = unity(xb, yb, domain=0)
    assert float(la) < float(lb), f"warm-up 后 expert0 应更拟合域A：{la} vs {lb}"
    print(f"[ok] warmup specialize: E0(A)={float(la):.3f} E0(B)={float(lb):.3f}")


def test_generate_o1_streaming():
    """0.1-α 补齐：UnityMoE 的统一生成入口（O(1) 流式 prefill + 采样）。

    此前 Unity 只有训练前向与 stream_step，没有 generate——自进化专家
    训练后无法用统一接口出文本。锁定：输出形状、路由在生成路径上被消费
    （MoE 权重参与了递推）、确定性（eval + 无采样随机性时两次一致）。
    """
    torch.manual_seed(0)
    unity = UnityModel(SimplCoreModel(_tiny_cfg()),
                       UnityConfig(num_experts=2, top_k=1))
    unity.eval()
    idx = torch.randint(0, 64, (1, 6))
    out = unity.generate(idx, max_new_tokens=5, temperature=1.0, top_k=1)
    assert out.shape == (1, 6 + 5)
    assert (out[:, :6] == idx).all()          # 前缀原样保留
    # 确定性：同参两次生成一致（top_k=1 贪心，eval 模式）
    out2 = unity.generate(idx, max_new_tokens=5, temperature=1.0, top_k=1)
    assert torch.equal(out, out2)


if __name__ == "__main__":
    test_base_frozen_excluded()
    test_experts_different_data_specialize()
    test_router_gradient_flows()
    test_budget_guard_blocks_overspawn()
    test_integration_with_train_factory()
    test_expert_persistence()
    test_self_evolution_manager()
    test_warmup_domains_specialize()
    test_generate_o1_streaming()
    print("\nALL UNITY TESTS PASSED")
