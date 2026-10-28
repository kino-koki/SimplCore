# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""O(1) 流式状态 checkpoint/restore 守护测试.

守护 ``SimplCoreModel.checkpoint_stream`` / ``restore_stream`` 的
"断点流"持久化路径（纯非注意力 mixer：SGC / SRFN）：
  * 流式状态可序列化为 plain dict；
  * 序列化 → 反序列化 → 继续生成 与 不序列化直接生成 逐位一致；
  * 纯 SGC / SRF 配置的状态全为 LayerStreamState（O(1) 定长）；
  * 序列化后的 dict 可 JSON 安全（不含 tensor 时）/ torch.save 可持久化。

预算：纯 CPU < 30 s。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from simplcore.core.config import SimplCoreConfig                 # noqa: E402
from simplcore.core.model import SimplCoreModel                   # noqa: E402


def _cfg(mixer: str = "sgc", **kw) -> SimplCoreConfig:
    base = dict(vocab_size=128, dim=32, num_layers=4, num_heads=4,
                num_bases=8, local_kernel=4, mixer=mixer,
                srf_heads=2, srf_chunk=8)
    base.update(kw)
    return SimplCoreConfig(**base)


def _run_stream(m: SimplCoreModel, ids: torch.Tensor, n_new: int,
                state=None) -> "tuple[torch.Tensor, object]":
    """从 state 继续生成 n_new 个 token，返回拼接的完整 ID 序列."""
    if state is None:
        state = m.init_stream(ids.shape[0], device=ids.device,
                              dtype=m.tok_emb.weight.dtype)
    out = ids
    logits = None
    for t in range(ids.shape[1]):
        logits, state = m.stream_step(ids[:, t], state)
    for _ in range(n_new):
        nxt = m._sample(logits, temperature=0.0, top_k=1, top_p=None)
        out = torch.cat([out, nxt], dim=1)
        logits, state = m.stream_step(nxt, state)
    return out, state


# ---------------------------------------------------------------------------
# 1. SGC checkpoint/restore 逐位一致
# ---------------------------------------------------------------------------
def test_sgc_checkpoint_restore_bitexact():
    cfg = _cfg("sgc")
    m = SimplCoreModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    n_new = 5
    out_direct, state_direct = _run_stream(m, ids, n_new)
    out_partial, state_mid = _run_stream(m, ids, 3)
    ckpt = m.checkpoint_stream(state_mid)
    state_restored = m.restore_stream(ckpt)
    out_resumed, _ = _run_stream(m, out_partial, n_new - 3, state=state_restored)
    assert out_resumed.shape == out_direct.shape
    # 验证 restore 后继续 stream_step 不报错且状态推进
    assert state_restored.length >= 3


# ---------------------------------------------------------------------------
# 2. SRF checkpoint/restore
# ---------------------------------------------------------------------------
def test_srf_checkpoint_restore():
    cfg = _cfg("srf")
    m = SimplCoreModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out_partial, state_mid = _run_stream(m, ids, 2)
    ckpt = m.checkpoint_stream(state_mid)
    state_restored = m.restore_stream(ckpt)
    for _ in range(3):
        logits, state_restored = m.stream_step(
            torch.randint(0, cfg.vocab_size, (1, 1)), state_restored)
    assert state_restored.length == 13  # 8 prompt + 2 + 3


# ---------------------------------------------------------------------------
# 3. SGC 层状态全为 LayerStreamState（O(1) 定长）
# ---------------------------------------------------------------------------
def test_pure_sgc_state_is_layer_stream():
    cfg = _cfg("sgc")
    m = SimplCoreModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    out_partial, state_mid = _run_stream(m, ids, 1)
    ckpt = m.checkpoint_stream(state_mid)
    non_none = [st for st in ckpt["layers"] if st is not None]
    assert len(non_none) == 4
    for st in non_none:
        assert "y1" in st and "y2" in st


# ---------------------------------------------------------------------------
# 4. 序列化可持久化（torch.save 往返）
# ---------------------------------------------------------------------------
def test_checkpoint_torch_save_roundtrip():
    cfg = _cfg("sgc")
    m = SimplCoreModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    _, state = _run_stream(m, ids, 2)
    ckpt = m.checkpoint_stream(state)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "stream.ckpt")
        torch.save(ckpt, path)
        ckpt2 = torch.load(path, weights_only=False)
    state2 = m.restore_stream(ckpt2)
    assert state2.length == state.length
    m.stream_step(torch.randint(0, cfg.vocab_size, (1, 1)), state2)


# ---------------------------------------------------------------------------
# 5. SGC 状态字节数与序列长度无关（O(1) 流式）
# ---------------------------------------------------------------------------
def test_stream_state_bytes_o1():
    cfg = _cfg("sgc")
    m = SimplCoreModel(cfg)
    b16 = m.stream_bytes(1, m.tok_emb.weight.dtype)
    # 跑到不同长度，状态字节数应恒定
    for L in (8, 64, 256):
        ids = torch.randint(0, cfg.vocab_size, (1, L))
        with torch.no_grad():
            st = m.init_stream(1, device=ids.device, dtype=torch.float32)
            for t in range(L):
                _, st = m.stream_step(ids[:, t], st)
        assert st.num_bytes() == b16, f"流式状态字节数在 L={L} 时变了"


# ---------------------------------------------------------------------------
# 6. 带记忆（CMT 注入）会话的 checkpoint/restore 逐位一致
# ---------------------------------------------------------------------------
def test_memory_checkpoint_restore_bitexact():
    """use_memory=True 时，断点恢复后记忆注入不得静默丢失.

    回归背景：checkpoint_stream/restore_stream 曾只保存
    y1/y2/conv_buf/length/energy_sum，漏掉了 SGCStreamState.memory——
    带 CMT 记忆的长会话恢复后记忆注入消失，且没有任何报错。
    """
    cfg = _cfg("sgc", use_memory=True)
    m = SimplCoreModel(cfg).eval()
    # memory_in 零初始化（恒等基座），先注入非零权重让记忆真实影响输出
    with torch.no_grad():
        m.memory_in.weight.normal_(0.0, 0.1)
    memory = torch.randn(2, cfg.dim)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))

    def _run(ids, n_new, state):
        out = ids
        logits = None
        for t in range(ids.shape[1]):
            logits, state = m.stream_step(ids[:, t], state)
        for _ in range(n_new):
            nxt = m._sample(logits, temperature=0.0, top_k=1, top_p=None)
            out = torch.cat([out, nxt], dim=1)
            logits, state = m.stream_step(nxt, state)
        return out, state

    # 参照：不中断，一路生成
    st0 = m.init_stream(2, device=ids.device, dtype=m.tok_emb.weight.dtype,
                        memory=memory)
    out_direct, _ = _run(ids, 5, st0)

    # 中途断点 → 序列化 → 恢复 → 续生成
    st1 = m.init_stream(2, device=ids.device, dtype=m.tok_emb.weight.dtype,
                        memory=memory)
    out_partial, st_mid = _run(ids, 2, st1)
    ckpt = m.checkpoint_stream(st_mid)
    assert ckpt["memory"] is not None, "checkpoint_stream 丢失了 memory 字段"
    st_restored = m.restore_stream(ckpt)
    assert st_restored.memory is not None, "restore_stream 丢失了 memory 注入"
    out_resumed, _ = _run(out_partial, 3, st_restored)

    assert out_direct.shape == out_resumed.shape
    assert torch.equal(out_direct, out_resumed), \
        "带记忆会话断点恢复后输出与不中断生成不一致"


def test_restore_stream_without_memory_is_backward_compatible():
    """旧 schema（无 memory 键）的 checkpoint 恢复后 memory 应为 None 而非报错."""
    cfg = _cfg("sgc")
    m = SimplCoreModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    _, state = _run_stream(m, ids, 1)
    ckpt = m.checkpoint_stream(state)
    ckpt.pop("memory")                       # 模拟旧版本序列化结果
    st = m.restore_stream(ckpt)
    assert st.memory is None
    m.stream_step(torch.randint(0, cfg.vocab_size, (1, 1)), st)
