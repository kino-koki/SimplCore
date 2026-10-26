# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""潜空间推演引擎 (RolloutEngine) 与世界模型预测头 (WorldModelHead).

RolloutEngine:
    在潜空间对输入潜状态进行多步迭代推演. 每步调用一个 SimplCoreBlock
    (或任意 nn.Module), 共享参数地反复精炼潜状态, 类似 "潜空间扩散 /
    思维链推演". 支持固定步数推演与自适应提前停止 (基于潜状态变化幅度),
    以及 ``detach_each_step`` 截断 BPTT 以节省显存.

    典型用法::

        engine = RolloutEngine(block, max_steps=8, adaptive=False)
        out, memory_state, states = engine(
            latent, num_steps=4, return_all_states=True
        )
        # states[0] = 初始潜状态, states[i] = 第 i 步后的潜状态

WorldModelHead:
    将推演后的潜状态解码为世界模型预测三路输出:
      - logits     : token 级别 logits     (B, L, vocab_size)
      - reward     : 标量奖励预测          (B, L, 1)
      - continuity : 连续性分数 in [0, 1]  (B, L, 1)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn


class RolloutEngine(nn.Module):
    """潜空间多步推演引擎.

    Args:
        block: SimplCoreBlock 或任意 nn.Module. 调用约定为
            ``block(latent, memory_state)``, 返回值可为:
              * 单张量 ``latent`` (记忆状态不变), 或
              * 元组 ``(latent, memory_state)``.
        max_steps: 最大推演步数 (同时是 ``num_steps=None`` 时的默认步数,
            以及自适应模式下的步数上限).
        adaptive: 是否启用自适应提前停止.
        threshold: 自适应停止阈值. 相邻两步潜状态的相对 L2 变化小于该值时停止.
        detach_each_step: 每步是否 detach 输入 (截断 BPTT, 节省显存).
    """

    def __init__(
        self,
        block: nn.Module,
        max_steps: int = 8,
        adaptive: bool = False,
        threshold: float = 1e-4,
        detach_each_step: bool = False,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1, 得到 {max_steps}")
        self.block = block
        self.max_steps = max_steps
        self.adaptive = adaptive
        self.threshold = threshold
        self.detach_each_step = detach_each_step

    # ------------------------------------------------------------------
    def _step(
        self,
        latent: torch.Tensor,
        memory_state: Optional[object],
    ) -> Tuple[torch.Tensor, Optional[object]]:
        """执行单步推演.

        兼容 block 返回单张量或 ``(latent, memory_state)`` 元组.
        """
        result = self.block(latent, memory_state)
        if isinstance(result, (tuple, list)):
            new_latent = result[0]
            new_mem = result[1] if len(result) > 1 else memory_state
        else:
            new_latent = result
            new_mem = memory_state
        return new_latent, new_mem

    # ------------------------------------------------------------------
    def forward(
        self,
        latent: torch.Tensor,
        memory_state: Optional[object] = None,
        num_steps: Optional[int] = None,
        return_all_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object], List[torch.Tensor]]:
        """多步推演.

        Args:
            latent: (B, L, H) 初始潜状态.
            memory_state: 跨序列记忆状态 (透传给 block).
            num_steps: 推演步数; ``None`` 时取 ``max_steps``
                (自适应模式下 ``max_steps`` 作为上限).
            return_all_states: 是否收集所有中间状态. 无论取值如何, 始终返回
                ``states`` 列表; 该参数保留用于 API 兼容.

        Returns:
            ``(final_latent, final_memory_state, states)`` 三元组, 其中
            ``states[0]`` 为初始潜状态, ``states[i]`` 为第 ``i`` 步后的潜状态.
        """
        if num_steps is None:
            steps = self.max_steps
        else:
            steps = min(int(num_steps), self.max_steps)

        states: List[torch.Tensor] = [latent]
        cur = latent
        mem = memory_state

        for _ in range(steps):
            nxt, mem = self._step(cur, mem)

            # 自适应提前停止: 相邻步潜状态相对变化 < threshold
            if self.adaptive:
                with torch.no_grad():
                    denom = cur.float().norm() + 1e-8
                    change = (nxt - cur).float().norm() / denom
                if change.item() < self.threshold:
                    cur = nxt
                    states.append(cur)
                    break

            # 截断 BPTT: 每步将输入从计算图分离 (省显存, 仅保留单步梯度)
            if self.detach_each_step:
                cur = nxt.detach().requires_grad_(cur.requires_grad)
            else:
                cur = nxt
            states.append(cur)

        return cur, mem, states

    # ------------------------------------------------------------------
    def adaptive_rollout(
        self,
        latent: torch.Tensor,
        threshold: float = 1e-4,
        memory_state: Optional[object] = None,
        return_all_states: bool = True,
    ) -> Tuple[torch.Tensor, Optional[object], List[torch.Tensor], int]:
        """自适应推演: 根据潜状态变化幅度自动决定停止时机.

        Args:
            latent: (B, L, H) 初始潜状态.
            threshold: 停止阈值.
            memory_state: 跨序列记忆状态.
            return_all_states: 是否返回所有中间状态.

        Returns:
            ``(final_latent, final_memory_state, states, num_steps_taken)``,
            其中 ``num_steps_taken = len(states) - 1`` 为实际推演步数.
        """
        old_threshold = self.threshold
        old_adaptive = self.adaptive
        self.threshold = threshold
        self.adaptive = True
        try:
            cur, mem, states = self.forward(
                latent,
                memory_state=memory_state,
                num_steps=self.max_steps,
                return_all_states=return_all_states,
            )
        finally:
            self.threshold = old_threshold
            self.adaptive = old_adaptive
        return cur, mem, states, len(states) - 1

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """统计可训练参数总量 (含 block)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WorldModelHead(nn.Module):
    """世界模型预测头.

    将潜状态解码为三路预测:
      - logits:     token 级别 logits     (B, L, vocab_size)
      - reward:     每个 token 的标量奖励 (B, L, 1)
      - continuity: 连续性分数 in [0, 1]  (B, L, 1)

    Args:
        dim: 潜状态维度.
        vocab_size: 词表大小.
        hidden_dim: 预测头隐藏维度 (None 时取 max(dim, 128)).
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        if hidden_dim is None:
            hidden_dim = max(dim, 128)

        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.logits_head = nn.Linear(dim, vocab_size)
        self.reward_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continuity_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向解码.

        Args:
            latent: (B, L, H) 潜状态.

        Returns:
            ``(logits, reward, continuity)``.
        """
        h = self.norm(latent)
        logits = self.logits_head(h)                          # (B, L, vocab)
        reward = self.reward_head(h)                          # (B, L, 1)
        continuity = torch.sigmoid(self.continuity_head(h))   # (B, L, 1) in [0,1]
        return logits, reward, continuity


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
class _DummyBlock(nn.Module):
    """自检用占位 block: 残差式精炼潜状态 (会逐步收敛)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        latent: torch.Tensor,
        memory_state: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        h = self.norm(latent)
        out = latent + 0.1 * torch.tanh(self.fc(h))
        return out, memory_state


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 60)
    print("RolloutEngine 自检")
    print("=" * 60)

    dim = 32
    block = _DummyBlock(dim)

    # 1) 固定步数: states 长度 = 1 + num_steps
    engine = RolloutEngine(block, max_steps=6, adaptive=False)
    x = torch.randn(2, 16, dim)
    out, ms, states = engine(x, num_steps=4, return_all_states=True)
    print(f"固定步数: {tuple(x.shape)} -> {tuple(out.shape)}, states={len(states)}")
    assert out.shape == (2, 16, dim)
    assert len(states) == 5  # 1 初始 + 4 步

    # 2) 梯度流
    x2 = torch.randn(2, 16, dim, requires_grad=True)
    out2, _, _ = engine(x2, num_steps=3)
    out2.sum().backward()
    has_grad = x2.grad is not None and x2.grad.abs().sum() > 0
    print(f"梯度流: {'OK' if has_grad else 'FAIL'}")
    assert has_grad

    # 3) 自适应提前停止
    engine_adp = RolloutEngine(block, max_steps=8, adaptive=True, threshold=1e-3)
    out3, _, states3 = engine_adp(x, return_all_states=True)
    print(f"自适应: 实际步数={len(states3) - 1} (上限 8)")
    assert (len(states3) - 1) <= 8

    # 4) detach_each_step 模式 (截断 BPTT)
    engine_det = RolloutEngine(
        block, max_steps=6, adaptive=False, detach_each_step=True
    )
    x4 = torch.randn(2, 16, dim, requires_grad=True)
    out4, _, _ = engine_det(x4, num_steps=4)
    out4.sum().backward()
    block_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in engine_det.block.parameters()
    )
    print(f"detach 模式反向: block 梯度 = {block_grad}")
    assert block_grad

    # 5) adaptive_rollout 方法
    _, _, _, n_taken = engine_adp.adaptive_rollout(x, threshold=1e-3)
    print(f"adaptive_rollout: 停止于 {n_taken} 步")
    assert n_taken <= 8

    # 6) count_parameters
    print(f"count_parameters: {engine.count_parameters()}")

    # 7) WorldModelHead
    head = WorldModelHead(dim=64, vocab_size=256)
    lat = torch.randn(2, 16, 64)
    logits, reward, cont = head(lat)
    print(
        f"WorldModelHead: logits={tuple(logits.shape)}, "
        f"reward={tuple(reward.shape)}, cont={tuple(cont.shape)}"
    )
    assert logits.shape == (2, 16, 256)
    assert reward.shape == (2, 16, 1)
    assert cont.shape == (2, 16, 1)
    assert (cont >= 0).all() and (cont <= 1).all()

    print(f"[PASS] RolloutEngine 创建和推演正常.")
