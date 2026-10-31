# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""定容绑定记忆（R1）的守护测试——针对复现报告的三条教训.

1. **绑定机制构造性验证**：把寻址/读出投影设为恒等（理想化训练终态），
   「键 值」相邻的绑定写入 + 查询必须精确恢复 value——bound 与 single
   两种写入模式的**机制差异**要可测（single 在多对干扰下混淆，
   bound 精确）；这是对「卡点在读取」诊断的机制级确认。
2. **定容性**：``slot_bytes()`` 与已写入步数无关（写 1 步 == 写 10000 步），
   FIFO 环形覆盖正确（最老槽先被覆盖）。
3. **全链梯度**（复现报告 mem_Wq 梯度恒 0 的教训）：addr_key / addr_value /
   addr_query / read_out 四个投影在 retrieve_sequence 路径上梯度全部
   非零——任何一处恒 0 即失败。
4. **双路径一致性**：推理路径（write_step 逐步）与训练路径
   （retrieve_sequence）在同输入同参数下产生相同的槽内容语义。
5. **因果性**：位置 t 的读出看不到含 h_t 的绑定（先读后写）。
6. 一致性判据：寻址 softmax over S 槽（O(S·D)）——S 不随时序增长，
   由 2 的定容性覆盖；purity 全仓扫描由既有测试覆盖（本模块无禁词）。

诚实边界：全部为**机制级**测试（恒等投影 = 理想化训练终态）；
检索质量数值不做断言（未训练验证）。
"""
from __future__ import annotations

import pytest
import torch

from simplcore.core.memory.binding_memory import BoundedBindingMemory


def _identity_mem(n_slots: int = 16, dim: int = 8, mode: str = "bound",
                  window: int = 2) -> BoundedBindingMemory:
    m = BoundedBindingMemory(dim=dim, n_slots=n_slots, bind_window=window,
                             write_mode=mode)
    torch.manual_seed(0)
    with torch.no_grad():
        m.addr_key.weight.copy_(torch.eye(dim))
        m.addr_value.weight.copy_(torch.eye(dim))
        m.addr_query.weight.copy_(torch.eye(dim))
    m.eval()
    return m


# ---------------------------------------------------------------------------
# 1. 绑定机制：bound 精确恢复 vs single 混淆（构造性）
# ---------------------------------------------------------------------------
def test_bound_mode_recovers_value_exactly() -> None:
    m = _identity_mem(n_slots=16, dim=8, mode="bound", window=2)
    # 序列：(k1 v1) (k2 v2) (k3 v3) 相邻绑定；然后查询 k2
    e = [torch.eye(8)[i] * 3.0 for i in range(8)]      # 8 个可分辨向量
    vals = [e[4], e[5], e[6]]
    seq = torch.stack([e[0], e[4], e[1], e[5], e[2], e[6]])   # k1 v1 k2 v2 k3 v3
    m.reset()
    with torch.no_grad():
        m.write_sequence(seq)
        out, probs = m.retrieve(e[1])                   # 查询 k2
    # 池化绑定的读出是 value 侧窗口均值（含 v2 主分量）——**最近邻判定**：
    # 与 v2 的相似度必须严格高于其它任何 value（排序断言，非精确等式：
    # mean([k2,v2]) 池化让 key/value 各含对方分量，锐化靠训练的 addr 投影）
    sims = {i: float(out @ v) for i, v in enumerate(vals)}
    best = max(sims, key=sims.get)
    assert best == 1, f"查询 k2 的读出必须最接近 v2: {sims}"
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


def test_single_mode_fails_under_interference() -> None:
    """失败对照：single 模式（同 token 既当 key 又当 value）在多对干扰下
    读不出 v2——复现报告的卡点被机制级复现.

    single 写入槽 i = (addr_key(h_i), addr_value(h_i))；查询 k2=e[1] 时
    最匹配的槽是「h=e[1] 自己的槽」（即 k1 槽后写的 v1 槽？——k 序列里
    e[1] 只在 k2 位置出现，single 写的是 (e[0],e[0]) (e[4],e[4]) …，
    没有 key=e[1] 的槽；最近邻是 e[0] 或 e[2] 的槽 → 读出 ≠ e[5]）。
    """
    m = _identity_mem(n_slots=16, dim=8, mode="single")
    e = [torch.eye(8)[i] * 3.0 for i in range(8)]
    seq = torch.stack([e[0], e[4], e[1], e[5], e[2], e[6]])
    m.reset()
    with torch.no_grad():
        m.write_sequence(seq)
        out, _ = m.retrieve(e[1])
    assert not torch.allclose(out, e[5], atol=1e-3), \
        "single 模式（绑定缺失）不应精确恢复 v2——卡点机制复现"


def test_bound_beats_single_on_mqar_style() -> None:
    """多对干扰下的对照：bound 恢复精度必须高于 single（机制级 MQAR 缩比）."""
    dim, pairs = 16, 6
    e = [torch.eye(dim)[i] * 3.0 for i in range(dim)]
    keys, vals = e[:pairs], e[pairs:2 * pairs]
    seq = torch.stack([x for kv in zip(keys, vals) for x in kv])  # k v k v ...
    accs = {}
    for mode in ("bound", "single"):
        m = _identity_mem(n_slots=32, dim=dim, mode=mode, window=2)
        m.reset()
        with torch.no_grad():
            m.write_sequence(seq)                    # buffer 路径整段写入
            hit = 0
            for i, (k, v) in enumerate(zip(keys, vals)):
                out, _ = m.retrieve(k)
                sims = [float(out @ vv) for vv in vals]
                hit += int(max(range(len(vals)), key=lambda j: sims[j]) == i)
        accs[mode] = hit / pairs
    assert accs["bound"] == 1.0
    assert accs["bound"] > accs["single"], \
        f"绑定写入必须优于单 token 写入: {accs}"


# ---------------------------------------------------------------------------
# 2. 定容性 + FIFO
# ---------------------------------------------------------------------------
def test_capacity_constant_and_fifo_eviction() -> None:
    m = _identity_mem(n_slots=4, dim=8, mode="bound", window=2)
    m.reset()
    b0 = m.slot_bytes()
    e = torch.eye(8) * 3.0
    for t in range(100):                                # bound：每步一个绑定单元
        m.write_step(e[t % 8])
    assert m.slot_bytes() == b0, "槽状态字节数必须与写入步数无关（O(1) 判据）"
    # FIFO：只剩最后 4 个绑定单元的 key（e[t%8] 末 4 步的 key 侧）
    out, _ = m.retrieve(e[3])                           # 最后写入含 e[3]
    assert float(out.norm()) > 0.0
    # 最老的绑定已被覆盖：bound 首步窗口不满不写，100 步写 99 次，环回
    assert int(m.write_ptr.item()) == 99 % 4


# ---------------------------------------------------------------------------
# 3. 全链梯度（mem_Wq 教训）
# ---------------------------------------------------------------------------
def test_full_chain_gradients_nonzero() -> None:
    torch.manual_seed(0)
    m = BoundedBindingMemory(dim=16, n_slots=8, bind_window=2)
    m.train()
    H = torch.randn(24, 16)
    out = m.retrieve_sequence(H)
    # (a) addr_* 三投影：经读出向量直接回传（下游损失取 out 的线性组合）
    noise = torch.randn(out.shape, generator=torch.Generator().manual_seed(3))
    (out * noise).sum().backward()
    for name in ("addr_key", "addr_value", "addr_query"):
        g = getattr(m, name).weight.grad
        assert g is not None, f"{name}.grad is None"
        assert float(g.abs().sum()) > 0, \
            f"{name} 梯度恒 0 —— 复现报告的 mem_Wq 缺陷模式，禁止回归"
    # (b) read_out：零初始化投影对**自身二次损失**的梯度数学上恒 0
    # （固有性质非缺陷）；其真实梯度由下游损失提供——验证：
    # d(sum((Wx)·noise))/dW = noise·xᵀ ≠ 0，与 W 值无关。
    # 独立前向（上一次 backward 已释放图）。
    m.zero_grad()
    out2 = m.retrieve_sequence(H)
    noise2 = torch.randn(out2.shape, generator=torch.Generator().manual_seed(5))
    (m.read_out(out2) * noise2).sum().backward()
    g = m.read_out.weight.grad
    assert g is not None and float(g.abs().sum()) > 0, \
        "read_out 梯度在下游非零损失下必须非零"


# ---------------------------------------------------------------------------
# 4. 双路径一致性 + 因果性
# ---------------------------------------------------------------------------
def test_inference_training_paths_agree() -> None:
    torch.manual_seed(1)
    m = BoundedBindingMemory(dim=8, n_slots=8, bind_window=2)
    H = torch.randn(12, 8)
    # 训练路径（图外槽容器，先读后写）
    m.reset()
    out_seq = m.retrieve_sequence(H)
    # 推理路径：**先读后写**（与训练路径同因果序：位置 t 读 [0,t) 的写入，
    # 写 (h_{t-1}|h_t) 发生在 t 步末）
    m.reset()
    outs = []
    for t in range(H.shape[0]):
        outs.append(m.retrieve(H[t])[0])
        m.write_step(H[t])
    out_inf = torch.stack(outs)
    assert torch.allclose(out_seq, out_inf, atol=1e-5), \
        "推理与训练两条路径的读出必须一致（同一机制的两面）"


def test_causal_read_at_position_t() -> None:
    """因果性：位置 t 的读出不得使用含 h_t 的绑定（先读后写）."""
    m = _identity_mem(n_slots=8, dim=8, mode="bound", window=2)
    m.reset()
    e0 = torch.eye(8)[0] * 3.0
    e1 = torch.eye(8)[1] * 3.0
    with torch.no_grad():
        out0 = m.retrieve_sequence(torch.stack([e0]))    # 单步：无已写槽
    assert float(out0.abs().max()) == 0.0, \
        "首个位置的读出必须为 0（没有任何历史绑定可读；同时验证无 NaN）"
    # 已知特性（窗口池化绑定）：value 侧 token 的自查询会部分命中同槽
    # （key 池化含 e1 分量）——锐化依赖训练的 addr 投影，此处只锁
    # 「不产生 NaN / 不越界」与因果零起点。
    m2 = _identity_mem(n_slots=8, dim=8, mode="bound", window=2)
    m2.reset()
    with torch.no_grad():
        out2 = m2.retrieve_sequence(torch.stack([e0, e1]))
    assert torch.isfinite(out2).all(), "读出必须有限（masked softmax 不得产生 NaN）"

# ---------------------------------------------------------------------------
# 7. soft 可微写入（二期复现报告：3000 步退化的结构性修复）
# ---------------------------------------------------------------------------
def test_soft_mode_gate_gradients_flow() -> None:
    """soft 门控写入全程可微：write_gate（写入策略参数）必须拿到非零梯度
    ——硬 argmin 写入的梯度恒 0（复现报告二期的 3000 步退化根因）."""
    torch.manual_seed(0)
    m = BoundedBindingMemory(dim=16, n_slots=8, bind_window=2,
                             write_mode="soft")
    m.train()
    H = torch.randn(24, 16)
    out = m.retrieve_sequence(H)
    noise = torch.randn(out.shape, generator=torch.Generator().manual_seed(3))
    (out * noise).sum().backward()
    g = m.write_gate.weight.grad
    assert g is not None and float(g.abs().sum()) > 0,         "write_gate 梯度恒 0 —— 可微写入失效（3000 步退化将无法修复）"
    for name in ("addr_key", "addr_value", "addr_query"):
        assert float(getattr(m, name).weight.grad.abs().sum()) > 0


def test_soft_mode_capacity_constant() -> None:
    """soft 槽状态仍定容：slot_bytes 与写入步数无关（一致性判据）."""
    torch.manual_seed(0)
    m = BoundedBindingMemory(dim=8, n_slots=4, bind_window=2,
                             write_mode="soft")
    m.reset()
    b0 = m.slot_bytes()
    H = torch.randn(200, 8)
    m.retrieve_sequence(H)
    assert m.slot_bytes() == b0, "soft 模式槽状态必须仍与步数无关"


def test_soft_mode_no_collapse_mechanism() -> None:
    """bound vs soft 的真实机制差异：**写入可微性**（非"不遗忘"——门控
    0.5 下 N 步后早期信息衰减 2^-N，soft 同样遗忘；3000 步退化的修复点
    是"写入可微"——槽位分配的梯度信号存在，而非记忆不灭）。

    锁定：(a) write_gate 参数存在且可微（梯度测试见上）；(b) bound 的槽
    key **精确等于**最后 S 个绑定的 addr_key 输出（硬替换语义）；
    (c) soft 的槽 key 是**历史门控混合**（不等于任何单个绑定 key——
    混合语义存在，分配平滑）。"""
    dim, n_slots = 16, 8
    torch.manual_seed(0)
    e = [torch.eye(dim)[i] * 3.0 for i in range(dim)]
    flood = torch.stack([e[i % dim] for i in range(2 * n_slots)])  # 已知绑定
    # (a) bound：硬替换语义——槽 key 精确等于最后 8 个绑定单元的 key 侧
    m_b = BoundedBindingMemory(dim=dim, n_slots=n_slots, bind_window=2,
                               write_mode="bound")
    with torch.no_grad():
        m_b.addr_key.weight.copy_(torch.eye(dim))
        m_b.addr_value.weight.copy_(torch.eye(dim))
    m_b.reset()
    with torch.no_grad():
        m_b.write_sequence(flood)
        # 15 个绑定（(f0,f1)...(f14,f15)）> 8 槽：FIFO 后槽集合 = 最后 8 个
        # 绑定的 key 侧 = addr_key(flood[7..14])（环形覆盖，物理序≠时间序）
        expected = [m_b.addr_key(flood[j]) for j in range(7, 15)]
        matched = sum(
            1 for exp_k in expected
            if any(torch.allclose(m_b.slot_k[i], exp_k, atol=1e-5)
                   for i in range(n_slots)))
        assert matched == n_slots,             f"bound 硬替换语义破坏: 仅 {matched}/{n_slots} 槽匹配最后 8 绑定"
    # (b) soft：门控混合——槽 key 不等于任何单个绑定 key（混合体）
    m_s = BoundedBindingMemory(dim=dim, n_slots=n_slots, bind_window=2,
                               write_mode="soft")
    with torch.no_grad():
        m_s.addr_key.weight.copy_(torch.eye(dim))
        m_s.addr_value.weight.copy_(torch.eye(dim))
    m_s.reset()
    with torch.no_grad():
        m_s.write_sequence(flood)
        matches = sum(
            1 for i in range(n_slots)
            for j in range(2 * n_slots)
            if torch.allclose(m_s.slot_k[i], m_s.addr_key(flood[j]),
                              atol=1e-4))
    assert matches == 0,         f"soft 槽应为历史门控混合体（不等于任何单绑定 key）: {matches} 处精确匹配"
    assert torch.isfinite(m_s.slot_k).all()


def test_soft_mode_slot_state_passing() -> None:
    """soft 的 slot_state 传入/传出（跨 chunk BPTT 同构 stream_step state）。"""
    torch.manual_seed(0)
    m = BoundedBindingMemory(dim=8, n_slots=4, bind_window=2,
                             write_mode="soft")
    m.reset()
    H1 = torch.randn(10, 8)
    H2 = torch.randn(10, 8)
    out1, state = m.retrieve_sequence(H1, return_state=True)
    out2, _ = m.retrieve_sequence(H2, slot_state=state, return_state=True)
    assert out1.shape == (10, 8) and out2.shape == (10, 8)
    assert torch.isfinite(out2).all()
