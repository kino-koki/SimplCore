# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""simplcore/mix 因果性守护测试 (2026-10 审计第 0.2 节修复).

三个已定位的非因果算子:
  1. cin.py  上下文自注意力 softmax(q@k^T) 无因果掩码
  2. memory.py  x.mean(dim=1) 全序列均值
  3. blocks.py  Hebbian 全局均值 x.mean(dim=1) / out.mean(dim=1)

全部修复为因果前缀统计后, 本文件用"只扰动最后一个 token"的协议
做端到端守护: 前 L-1 个位置的输出必须逐位不变 (浮点误差 < 1e-4)。
扰动用 cat 拼接实现 (NumPy 引擎的 setitem 广播受限, 见 tensor.py)。

运行:
  python -m pytest tests_simplcore_mix/test_causality.py -q
"""
from __future__ import annotations

import pytest

import simplcore.tensor as st


def _tiny_cfg():
    """可快速在 CPU 上构建的 Mix 配置 (复用 mix_tiny_text)。"""
    from simplcore.mix.config import get_config

    return get_config("tiny")


def _perturb_last(x):
    """只扰动最后一个 token: 用随机张量替换, 保持前 L-1 行不变。"""
    B, L, D = x.shape
    prefix = x[:, : L - 1, :]
    last = st.randn(B, 1, D) * 100
    return st.cat([prefix, last], dim=1)


# ---------------------------------------------------------------------------
# 1. 组件级因果性
# ---------------------------------------------------------------------------
def test_cin_context_attn_is_causal():
    """cin: 扰动最后一个 token, 前 L-1 位置的 context residual 不变。"""
    from simplcore.mix.cin import StructuredGatedIntervention

    st.manual_seed(0)
    m = StructuredGatedIntervention(dim=64, n_groups=4).eval()
    L = 16
    x = st.randn(1, L, 64)
    with st.no_grad():
        y1 = m(x, x)
        x2 = _perturb_last(x)
        y2 = m(x2, x2)
    delta = (y1[:, : L - 1] - y2[:, : L - 1]).abs().max().item()
    assert delta < 1e-4, f"cin leaks the future: {delta}"


def test_memory_update_is_causal():
    """memory: 扰动最后一个 token, gate/decay 的前 L-1 行逐位不变。"""
    from simplcore.mix.memory import WorkingMemoryBuffer
    from simplcore.mix.config import get_config

    st.manual_seed(0)
    cfg = get_config("tiny")
    m = WorkingMemoryBuffer(cfg).train()
    L = 16
    x = st.randn(2, L, cfg.dim)
    with st.no_grad():
        mem1, g1, d1 = m._update_memories(x, state=None)
        x2 = _perturb_last(x)
        mem2, g2, d2 = m._update_memories(x2, state=None)
    # 因果前缀守护: gate/decay 逐位置 [B, T, C], 第 t 行只依赖 s <= t。
    # 只扰动末位时, 前 L-1 行必须逐位不变。
    delta_g = (g1[:, : L - 1, :] - g2[:, : L - 1, :]).abs().max().item()
    delta_d = (d1[:, : L - 1, :] - d2[:, : L - 1, :]).abs().max().item()
    assert delta_g < 1e-4, f"memory gate leaks the future: {delta_g}"
    assert delta_d < 1e-4, f"memory decay leaks the future: {delta_d}"


def test_hebbian_update_is_causal():
    """blocks: 扰动最后一个 token, 前 T-1 行的塑料输出逐位不变。"""
    from simplcore.mix.blocks import RecurrentBlock
    from simplcore.mix.config import get_config

    st.manual_seed(0)
    cfg = get_config("tiny")
    rb = RecurrentBlock(cfg).train()
    B, T, D = 2, 8, cfg.dim
    x = st.randn(B, T, D)
    out = st.randn(B, T, D)
    with st.no_grad():
        r1, s1 = rb._apply_hebbian_update(x, out, hebb_state=None)
        x2 = _perturb_last(x)
        out2 = _perturb_last(out)
        r2, s2 = rb._apply_hebbian_update(x2, out2, hebb_state=None)
    delta = (r1[:, : T - 1] - r2[:, : T - 1]).abs().max().item()
    assert delta < 1e-4, f"hebbian update leaks the future: {delta}"


# ---------------------------------------------------------------------------
# 2. 端到端 Mix 因果性
# ---------------------------------------------------------------------------
def test_mix_model_is_causal():
    """端到端: 只扰动最后一个 token, 前 L-1 位置的 logits 逐位不变。"""
    from simplcore.mix.model import SimplCoreMix

    st.manual_seed(0)
    cfg = _tiny_cfg()
    model = SimplCoreMix(cfg).eval()
    L = 32
    idx = st.randint(0, cfg.vocab_size, (1, L))
    with st.no_grad():
        y1 = model(idx, n_loops=1)
        idx2 = st.cat([idx[:, : L - 1], (idx[:, :1] + 7) % cfg.vocab_size], dim=1)
        y2 = model(idx2, n_loops=1)
    delta = (y1[:, : L - 1] - y2[:, : L - 1]).abs().max().item()
    assert delta < 1e-4, f"SimplCoreMix leaks the future: {delta}"


def test_mix_generate_is_deterministic():
    """生成路径 (KV 缓存) 同 prompt 同参数必须逐位一致。"""
    from simplcore.mix.model import SimplCoreMix

    st.manual_seed(0)
    cfg = _tiny_cfg()
    model = SimplCoreMix(cfg).eval()
    L = 8
    idx = st.randint(0, cfg.vocab_size, (1, L))
    with st.no_grad():
        a = model.generate(idx, max_new_tokens=4, temperature=0.0, n_loops=1)
        b = model.generate(idx, max_new_tokens=4, temperature=0.0, n_loops=1)
    assert a.shape == b.shape
    delta = (a - b).abs().max().item()
    assert delta == 0.0, f"generate is not deterministic: {delta}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
