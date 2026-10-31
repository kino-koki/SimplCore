# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SGC-R（谱专家路由变体）的数值正确性测试。

守护 docstring 中的架构声明，强度与 ``test_mixer_math.py`` 一致：

* 分组并行形式与 O(L²) 显式卷积参考一致（含 per-group 路由加权）；
* FFT（整段并行）形式与 IIR（O(1) 流式）形式是同一算子；
* 分块 ``forward_chunk`` 与整段 ``forward`` 在 fp32 误差内逐位一致；
* 流式状态不随序列长度增长，且字节账目与 sgc 逐字节相同；
* 严格因果（含路由决策的因果性）；
* 全参数（含 route_proj、rho/theta）收到梯度；straight-through 生效；
* switch 负载均衡 aux 有限且非负；
* 注册表/配置/解析参数账目/纯洁性四层接线一致。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.sgc import (RoutedSpectralGlobalConv, SpectralGlobalConv,
                                fft_causal_conv)
from simplcore.core.purity import assert_architecture_pure


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def explicit_causal_conv(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Direct O(L^2) causal convolution, used as ground truth."""
    B, L, D = u.shape
    y = torch.zeros_like(u)
    for t in range(L):
        for s in range(t + 1):
            y[:, t, :] += u[:, s, :] * kernel[t - s].unsqueeze(0)
    return y


def _random_routed(dim=16, num_bases=8, route_groups=2, route_topk=1,
                   local_kernel=4, seed=0, **kw) -> RoutedSpectralGlobalConv:
    torch.manual_seed(seed)
    m = RoutedSpectralGlobalConv(
        dim=dim, num_bases=num_bases, local_kernel=local_kernel,
        route_groups=route_groups, route_topk=route_topk, **kw)
    with torch.no_grad():
        m.coef.normal_(0, 0.3)
        m.in_proj.weight.normal_(0, 0.2)
        m.out_proj.weight.normal_(0, 0.2)
        m.local_conv.weight.normal_(0, 0.3)
        m.route_proj.weight.normal_(0, 0.5)   # 强路由对比度，保证各组都被用到
    return m


# ---------------------------------------------------------------------------
# 1. grouped parallel form == explicit O(L^2) reference
# ---------------------------------------------------------------------------
def test_grouped_long_matches_explicit():
    """长程通路 = Σ_g explicit_conv(u·w_g, K_g)（路由加权在注入端）。"""
    m = _random_routed()
    m.eval()
    L = 24
    x = torch.randn(2, L, m.dim)
    with torch.no_grad():
        h = m.norm(x)
        u = m.in_proj(h) * torch.sigmoid(m.gate_proj(h))
        w, _ = m._route_weights(h)
        ref = None
        for g, sel in m._group_slices:
            u_g = u * w[..., g: g + 1]
            k_g = m._group_kernel(sel, L, device=u.device, dtype=u.dtype)
            part = explicit_causal_conv(u_g, k_g)
            ref = part if ref is None else ref + part
        got = m._grouped_long(u, w, L)
    err = (got - ref).abs().max().item()
    assert err < 1e-4, f"grouped long mismatch {err}"


def test_mixer_output_matches_explicit():
    """整条 mixer 输出与显式参考一致（长程按组显式卷积 + 局部 + 瞬时 + 门控）。"""
    m = _random_routed()
    m.eval()
    L = 24
    x = torch.randn(2, L, m.dim)
    with torch.no_grad():
        h = m.norm(x)
        u = m.in_proj(h) * torch.sigmoid(m.gate_proj(h))
        w, _ = m._route_weights(h)
        long = None
        for g, sel in m._group_slices:
            u_g = u * w[..., g: g + 1]
            k_g = m._group_kernel(sel, L, device=u.device, dtype=u.dtype)
            part = explicit_causal_conv(u_g, k_g)
            long = part if long is None else long + part
        local = m._local_conv_full(u)
        if m.evidence_gate:
            long = long * m._evidence_weights(u)
        if m.uncertainty_gate:
            long = long * m._uncertainty_weights(long, local)
        manual = long + local + m.feed * u
        manual = m.out_proj(manual)
        expected = x + m.out_scale * manual
        got = m(x)
    err = (got - expected).abs().max().item()
    assert err < 1e-4, f"mixer output mismatch {err}"


# ---------------------------------------------------------------------------
# 2. the dual form: FFT (parallel) == IIR (streaming)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L", [1, 4, 17, 64, 129])
def test_streaming_matches_parallel(L):
    m = _random_routed(num_bases=8, route_groups=2, route_topk=1, seed=3)
    m.eval()
    x = torch.randn(2, L, m.dim)
    with torch.no_grad():
        y_par = m(x)
        st = m.init_state(2, x.device, x.dtype)
        outs = []
        for t in range(L):
            y_t, st = m.step(x[:, t], st)
            outs.append(y_t)
        y_seq = torch.stack(outs, dim=1)
    err = (y_par - y_seq).abs().max().item()
    assert err < 1e-4, f"parallel/streaming mismatch {err}"


@pytest.mark.parametrize("topk", [1, 2])
def test_streaming_matches_parallel_topk(topk):
    m = _random_routed(num_bases=8, route_groups=4, route_topk=topk, seed=7)
    m.eval()
    L = 40
    x = torch.randn(2, L, m.dim)
    with torch.no_grad():
        y_par = m(x)
        st = m.init_state(2, x.device, x.dtype)
        outs = []
        for t in range(L):
            y_t, st = m.step(x[:, t], st)
            outs.append(y_t)
        y_seq = torch.stack(outs, dim=1)
    err = (y_par - y_seq).abs().max().item()
    assert err < 1e-4, f"topk={topk} parallel/streaming mismatch {err}"


def test_chunked_matches_full():
    """分块 forward_chunk 与整段 forward 在 fp32 误差内逐位一致。"""
    m = _random_routed(num_bases=8, route_groups=2, route_topk=1, seed=11)
    m.eval()
    L, T = 23, 7
    x = torch.randn(1, L, m.dim)
    with torch.no_grad():
        y_full = m(x)
        outs = []
        st = m.init_state(1, x.device, x.dtype)
        for s in range(0, L, T):
            yc, st = m.forward_chunk(x[:, s: s + T], st)
            outs.append(yc)
        y_chunk = torch.cat(outs, dim=1)
    err = (y_full - y_chunk).abs().max().item()
    assert err < 1e-4, f"chunked/full mismatch {err}"


def test_streaming_state_is_constant_in_length():
    m = _random_routed(num_bases=8, route_groups=2)
    sizes = []
    for L in (8, 128, 2048):
        st = m.init_state(1, torch.device("cpu"), torch.float32)
        x = torch.randn(1, 1, m.dim)
        with torch.no_grad():
            for _ in range(min(L, 32)):
                _, st = m.step(x, st)
        sizes.append(st.y1.numel() + st.y2.numel()
                     + (st.conv_buf.numel() if st.conv_buf is not None else 0))
    assert len(set(sizes)) == 1, f"state grew with length: {sizes}"
    # 流式状态字节账目与未路由的 sgc **逐字节相同**（路由只门控输入注入，
    # 不携带任何随长度增长或随专家数增长的状态载体）。
    ref = SpectralGlobalConv(dim=m.dim, num_bases=m.num_bases,
                             local_kernel=m.local_kernel)
    assert m.streaming_state_bytes(1) == ref.streaming_state_bytes(1)


# ---------------------------------------------------------------------------
# 3. causality and receptive field
# ---------------------------------------------------------------------------
def test_mixer_is_strictly_causal():
    m = _random_routed()
    m.eval()
    L = 32
    x = torch.randn(1, L, m.dim)
    with torch.no_grad():
        y1 = m(x)
        x2 = x.clone()
        x2[:, L // 2:, :] = torch.randn(1, L - L // 2, m.dim) * 100
        y2 = m(x2)
    delta = (y1[:, : L // 2] - y2[:, : L // 2]).abs().max().item()
    assert delta < 1e-5, f"future leaks into the past: {delta}"


def test_route_decision_is_causal():
    """路由只依赖当前 token：扰动未来位置不改变过去的注入权重。"""
    m = _random_routed()
    m.eval()
    L = 16
    x = torch.randn(1, L, m.dim)
    with torch.no_grad():
        w1, _ = m._route_weights(m.norm(x))
        x2 = x.clone()
        x2[:, 8:, :] = torch.randn(1, L - 8, m.dim) * 50
        w2, _ = m._route_weights(m.norm(x2))
    err = (w1[:, :8] - w2[:, :8]).abs().max().item()
    assert err < 1e-6, f"route weights leak from the future: {err}"


def test_receptive_field_is_not_capped():
    """无 max_len 截断：脉冲串保证至少一个位置被路由到慢衰减组，
    其影响必须传导到最后一位。注意 SGC-R 的感受野是**内容相关**的
    （deformable 语义）：单脉冲可能恰被路由到快组而完全衰减（rho^L→0），
    这是算子语义，不是截断。"""
    m = _random_routed(num_bases=8, route_groups=2, seed=5)
    m.eval()
    L = 512
    x = torch.zeros(1, L, m.dim)
    with torch.no_grad():
        y1 = m(x)
        x2 = x.clone()
        # 随机脉冲：常数脉冲经 norm/proj 后 h 全同 → 路由决策全同（可能
        # 全落快组）；随机脉冲让 8 个位置的路由分散，至少其一命中慢组。
        torch.manual_seed(123)
        pulses = torch.randn(8, m.dim) * 5.0
        x2[:, ::64, :] = pulses
        y2 = m(x2)
    influence = (y1[:, -1] - y2[:, -1]).abs().max().item()
    assert influence > 1e-6, "slow-basis influence does not reach position L-1"


def test_state_memory_survives_unrouted_periods():
    """reusable top-k 的 SGC 语义：专家被切换掉后，其记忆在状态里齐次
    衰减、不丢失，仍通过全基 einsum 贡献输出（与父类共享此性质）。"""
    m = _random_routed(num_bases=8, route_groups=2, route_topk=1, seed=5)
    m.eval()
    per = m.num_bases // m.route_groups
    slow = slice(per, m.num_bases)             # 组 1 = 慢基
    st = m.init_state(1, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        st.y1[:, slow, :] = 0.5                # 模拟"该专家曾被激活"
        zero = torch.zeros(1, 1, m.dim)
        ys = []
        for _ in range(32):                    # 零输入：无任何新注入
            y_t, st = m.step(zero, st)
            ys.append(y_t)
        y_with = torch.cat(ys, dim=1)
        # 对照：把慢组的输出系数置零 —— 差异即"慢组残留记忆的贡献"
        coef_ref = m.coef.clone()
        m.coef[slow, :] = 0.0
        st2 = m.init_state(1, torch.device("cpu"), torch.float32)
        st2.y1[:, slow, :] = 0.5
        ys2 = []
        for _ in range(32):
            y_t, st2 = m.step(zero, st2)
            ys2.append(y_t)
        y_without = torch.cat(ys2, dim=1)
        m.coef.copy_(coef_ref)
    delta = (y_with - y_without).abs().max().item()
    assert delta > 1e-6, f"unrouted expert memory is lost: {delta}"


def test_topk_sparsity_pattern():
    """注入权重每位置至多 k 个非零组，且组内归一化（和 ≤ 1）。"""
    m = _random_routed(num_bases=8, route_groups=4, route_topk=2)
    m.eval()
    h = torch.randn(3, 12, m.dim)
    with torch.no_grad():
        w, soft = m._route_weights(h)
    nonzero = (w > 0).sum(-1)
    assert (nonzero <= 2).all(), f"more than topk groups active: {nonzero}"
    # eval 模式下 w == hard：非零项是 renormalized softmax 的 top-2
    sums = w.sum(-1)
    assert (sums <= 1.0 + 1e-5).all() and (sums >= 0.5 - 1e-5).all(), sums


# ---------------------------------------------------------------------------
# 4. gradients
# ---------------------------------------------------------------------------
def test_all_parameters_receive_gradient():
    m = _random_routed(num_bases=8, route_groups=4, route_topk=2, seed=2)
    L = 16
    x = torch.randn(2, L, m.dim)
    m(x).pow(2).mean().backward()
    missing = []
    for name, p in m.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            missing.append(name)
    assert not missing, f"parameters without gradient: {missing}"


def test_straight_through_reaches_router():
    """前向 hard（稀疏注入），反向梯度经 dense softmax 流到 route_proj。"""
    m = _random_routed(num_bases=8, route_groups=4, route_topk=1, seed=4)
    m.train()
    L = 16
    x = torch.randn(2, L, m.dim)
    y = m(x)
    with torch.no_grad():
        _, soft = m._route_weights(m.norm(x))
        frac_active = (soft > 0).float().mean().item()   # dense softmax 全为正
    assert frac_active == 1.0, "sanity: dense softmax should be all-positive"
    y.pow(2).mean().backward()
    assert m.route_proj.weight.grad is not None
    assert m.route_proj.weight.grad.abs().sum() > 0, \
        "straight-through failed: router got no gradient"


def test_switch_aux_finite_and_nonneg():
    m = _random_routed(num_bases=8, route_groups=4, route_topk=2)
    m.eval()
    with torch.no_grad():
        m(torch.randn(2, 12, m.dim))
    aux = m.last_route_aux
    assert aux is not None and torch.isfinite(aux) and aux.item() >= 0.0
    weighted = m.router_aux_loss()
    assert weighted.item() == pytest.approx(m.route_aux * aux.item(), rel=1e-6)


# ---------------------------------------------------------------------------
# 5. model level + wiring
# ---------------------------------------------------------------------------
def _cfg(**kw) -> SimplCoreConfig:
    kw.setdefault("sgc_route_groups", 2)
    kw.setdefault("sgc_route_topk", 1)
    return SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                           local_kernel=4, mixer="sgc-r", **kw)


def test_model_forward_backward():
    cfg = _cfg()
    m = SimplCoreModel(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 24))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 24, cfg.vocab_size)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


def test_streaming_generation_matches_full_forward():
    """模型级：整段生成与 O(1) 流式生成的贪婪解码一致。"""
    torch.manual_seed(0)
    cfg = _cfg(dropout=0.0)
    m = SimplCoreModel(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        a = m.generate(prompt, max_new_tokens=6, temperature=0.0, use_state=True)
        b = m.generate(prompt, max_new_tokens=6, temperature=0.0, use_state=False)
    assert torch.equal(a, b), f"streaming {a.tolist()} != full {b.tolist()}"


def test_analytic_params_matches_real_model():
    from simplcore.core.config import analytic_params
    cfg = _cfg(sgc_route_groups=4, sgc_route_topk=2)
    assert analytic_params(cfg) == SimplCoreModel(cfg).num_params()
    # 路由器参数 D·G + G 相对 sgc 的增量被公式捕获（每层一个 mixer）
    base = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                           local_kernel=4, mixer="sgc")
    assert analytic_params(cfg) == analytic_params(base) \
        + cfg.num_layers * (cfg.dim * cfg.sgc_route_groups
                            + cfg.sgc_route_groups)


def test_routed_differs_from_sgc():
    """同 seed 下路由变体与 sgc 是不同的算子（结构 + 输出）。"""
    torch.manual_seed(9)
    cfg = dict(vocab_size=64, dim=32, num_layers=1, num_bases=8, local_kernel=4)
    mr = SimplCoreModel(SimplCoreConfig(mixer="sgc-r", sgc_route_groups=2,
                                        sgc_route_topk=1, **cfg))
    ms = SimplCoreModel(SimplCoreConfig(mixer="sgc", **cfg))
    assert mr.num_params() == ms.num_params() + 32 * 2 + 2
    torch.manual_seed(1)
    idx = torch.randint(0, 64, (1, 32))
    with torch.no_grad():
        a = mr(idx)
        b = ms(idx)
    assert (a - b).abs().max().item() > 1e-6, "routed output identical to sgc"


def test_routed_purity():
    """sgc-r 通过三层纯洁性守卫（配置 + 实例 + 注册表/源码）。"""
    cfg = _cfg()
    assert_architecture_pure(config=cfg, model=SimplCoreModel(cfg))


def test_invalid_route_config_rejected():
    with pytest.raises(ValueError):
        SimplCoreConfig(vocab_size=64, dim=32, num_layers=1, num_bases=8,
                        mixer="sgc-r", sgc_route_groups=3).validate()  # 8 % 3
    with pytest.raises(ValueError):
        SimplCoreConfig(vocab_size=64, dim=32, num_layers=1, num_bases=8,
                        mixer="sgc-r", sgc_route_groups=4,
                        sgc_route_topk=5).validate()