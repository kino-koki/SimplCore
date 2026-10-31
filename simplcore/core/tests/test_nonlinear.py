# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""非线性思考模块（nonlinear.py）的测试。"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
from simplcore.core.nonlinear import (
    HaltingHead,
    LatentRefineCell,
    LatentRefiner,
)


def _tiny():
    torch.manual_seed(0)
    cfg = SimplCoreConfig(vocab_size=64, dim=32, num_layers=2, num_bases=8,
                          decay_max=128, local_kernel=3, num_heads=2,
                          ffn_ratio=2.0, mixer="sgc")
    return SimplCoreModel(cfg).eval()


def _run_stream(step_fn, model, idx, B=1):
    state = model.init_stream(B)
    logits = None
    for t in range(idx.shape[1]):
        logits, state = step_fn(idx[:, t], state)
    return logits, state


def test_passthrough_bit_identical_to_model():
    """max_iters=0 时与 SimplCoreModel.stream_step 逐位一致（theory 守卫）。"""
    m = _tiny()
    th = LatentRefiner(m, max_iters=0).eval()
    idx = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        lg_th, _ = _run_stream(th.stream_step, m, idx)
        lg_plain, _ = _run_stream(m.stream_step, m, idx)
    assert torch.equal(lg_th, lg_plain)
    assert th.ledger.iters_run == 0


def test_disabled_flag_is_passthrough():
    m = _tiny()
    th = LatentRefiner(m, max_iters=3, enabled=False).eval()
    idx = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        lg_th, _ = _run_stream(th.stream_step, m, idx)
        lg_plain, _ = _run_stream(m.stream_step, m, idx)
    assert torch.equal(lg_th, lg_plain)


def test_thinking_changes_logits_and_is_finite():
    """思考步（cell 初始为恒等 + halting=0.5 → 至少迭代一次后非恒等）

    注意：down 零初始化使第一步 cell 输出与输入相同；但 halting(z)=0.5 <
    threshold=0.6 时会走满 max_iters=2。恒等两次仍相等——所以本测试给
    cell.down 一个非零扰动，验证"思考确实改变 logits"的机制路径。
    """
    m = _tiny()
    th = LatentRefiner(m, max_iters=2, halt_threshold=0.6).eval()
    with torch.no_grad():
        th.cell.down.weight.add_(0.01)          # 打破恒等（模拟已训练）
    idx = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        lg_th, _ = _run_stream(th.stream_step, m, idx)
        lg_plain, _ = _run_stream(m.stream_step, m, idx)
    assert torch.isfinite(lg_th).all()
    assert not torch.equal(lg_th, lg_plain)
    assert th.ledger.iters_run > 0
    assert th.ledger.speedup <= 1.0 + 1e-9      # 全跑满时 speedup<=1


def test_stream_bytes_unchanged():
    """思考不新增流式状态。"""
    m = _tiny()
    th = LatentRefiner(m, max_iters=2)
    assert th.stream_bytes(1) == m.stream_bytes(1)


def test_training_loss_backward_and_ponder():
    """并行训练路径：CE + ponder 可反传，梯度穿过 cell 与 halting。"""
    m = _tiny()
    th = LatentRefiner(m, max_iters=3, ponder_weight=0.1).train()
    idx = torch.randint(0, 64, (2, 12))
    targets = torch.randint(0, 64, (2, 12))
    logits, loss = th(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    g_cell = th.cell.down.weight.grad
    g_halt = th.halting.proj.weight.grad
    assert g_cell is not None and torch.isfinite(g_cell).all()
    assert g_halt is not None and g_halt.abs().sum() > 0   # ponder 项推 halting


def test_cell_zero_init_is_identity():
    """down 零初始化 → 单步 cell 输出等于输入（恒等）。"""
    torch.manual_seed(1)
    cell = LatentRefineCell(dim=16)
    z = torch.randn(2, 5, 16)
    with torch.no_grad():
        assert torch.equal(cell(z), z)


def test_halting_head_init_is_half():
    torch.manual_seed(1)
    h = HaltingHead(dim=8)
    p = h(torch.randn(4, 8))
    assert torch.allclose(p, torch.full_like(p, 0.5))


def test_purity_no_attention():
    m = _tiny()
    th = LatentRefiner(m, max_iters=2)
    assert th.attention_like_modules() == []


def test_invalid_args():
    m = _tiny()
    with pytest.raises(ValueError):
        LatentRefiner(m, max_iters=-1)
    with pytest.raises(ValueError):
        LatentRefiner(m, halt_threshold=0.0)


def test_generate_matches_passthrough_when_disabled():
    m = _tiny()
    th = LatentRefiner(m, max_iters=0).eval()
    idx = torch.randint(0, 64, (2, 8))
    torch.manual_seed(7)
    g1 = th.generate(idx.clone(), max_new_tokens=6, temperature=0, top_k=1)
    torch.manual_seed(7)
    g2 = m.generate(idx.clone(), max_new_tokens=6, temperature=0, top_k=1)
    assert torch.equal(g1, g2)
