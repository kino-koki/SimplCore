# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""动态记忆通道 (Dynamic Memory Channel, DMC).

核心思想:
  内容寻址记忆, 复杂度 O(1) (相对于序列长度 n)。
  维护固定大小记忆库 (B, C, S, D), 其中 C 为通道数, S 为每通道 slot 数,
  通过多头注意力进行读取, NTM 风格门控写入 (先擦除后添加)。

工作机制:
  - read(query): 多头注意力从 query 读取记忆槽位, 返回各通道聚合读出;
  - write(key, value): NTM 风格门控写入 — 先擦除后添加;
  - forward(x): 逐步处理序列, 每步 read + write, 输出经残差 + LayerNorm;
  - reset_state / get_state / set_state / detach_state: 用于 BPTT 的记忆状态管理。
"""

from __future__ import annotations

import math
from typing import Optional

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F


class DynamicMemoryChannel(nn.Module):
    """动态记忆通道 (Dynamic Memory Channel).

    维护固定大小记忆库 (B, C, S, D), 通过内容寻址读写,
    记忆复杂度 O(1) (相对于序列长度 n)。

    Args:
        dim: 特征维度 D.
        num_channels: 记忆通道数 C (并行记忆库数量).
        channel_size: 每个通道的 slot 数 S.
        num_heads: 读取/写入注意力头数 (须能整除 dim, 否则回退为 1).
    """

    def __init__(
        self,
        dim: int,
        num_channels: int = 64,
        channel_size: int = 128,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_channels > 0 and channel_size > 0
        self.dim = dim
        self.num_channels = num_channels
        self.channel_size = channel_size
        # 头数须能整除 dim, 否则回退为单头
        self.num_heads = (
            num_heads if (num_heads > 0 and dim % num_heads == 0) else 1
        )
        self.head_dim = dim // self.num_heads

        # 投影层
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.write_value_proj = nn.Linear(dim, dim, bias=True)
        self.output_proj = nn.Linear(dim, dim, bias=True)
        # 写入门 (每个通道一个门)
        self.write_gate = nn.Linear(dim, num_channels, bias=True)

        # 跨通道聚合权重 (可学习)
        self.channel_weights = nn.Parameter(torch.ones(num_channels))

        self.norm = nn.LayerNorm(dim)

        # 记忆库状态 (非参数, 不进入 state_dict)
        self._memory: Optional[torch.Tensor] = None
        self._batch_size: int = 0

    # ------------------------------------------------------------------
    # 记忆状态管理
    # ------------------------------------------------------------------
    def reset_state(self, batch_size: int) -> None:
        """重置记忆库, 为新序列做准备.

        以小随机值初始化记忆, 保证起始记忆接近"空白"。

        Args:
            batch_size: 批次大小 B.
        """
        device = self.channel_weights.device
        self._memory = torch.randn(
            batch_size, self.num_channels, self.channel_size, self.dim,
            device=device,
        ) * 0.02
        self._batch_size = batch_size

    def get_state(self) -> Optional[torch.Tensor]:
        """获取当前记忆库状态.

        Returns:
            (B, C, S, D) 记忆张量, 或未初始化时为 None.
        """
        return self._memory

    def set_state(self, memory: Optional[torch.Tensor]) -> None:
        """设置记忆库状态.

        Args:
            memory: (B, C, S, D) 记忆张量, 或 None.
        """
        self._memory = memory
        if memory is not None:
            self._batch_size = memory.shape[0]

    def detach_state(self) -> None:
        """截断 BPTT: 将记忆从计算图分离.

        在跨段训练时调用, 防止梯度沿记忆回传过远。
        """
        if self._memory is not None:
            self._memory = self._memory.detach()

    def _ensure_memory(self, batch_size: int) -> torch.Tensor:
        """惰性确保记忆库已按批次大小分配."""
        if self._memory is None or self._batch_size != batch_size:
            self.reset_state(batch_size)
        return self._memory  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 读写操作
    # ------------------------------------------------------------------
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """基于 query 的多头注意力读取.

        Args:
            query: (B, D) 查询向量.

        Returns:
            (B, C, D) 各通道聚合读出.
        """
        B = query.shape[0]
        memory = self._ensure_memory(B)  # (B, C, S, D)
        H, hd = self.num_heads, self.head_dim

        q = self.query_proj(query).reshape(B, H, hd)  # (B, H, hd)
        keys = self.key_proj(memory).reshape(
            B, self.num_channels, self.channel_size, H, hd
        )  # (B, C, S, H, hd)

        # 注意力打分: query 到各 slot
        scores = torch.einsum("bhd,bcshd->bchs", q, keys) / math.sqrt(hd)
        weights = F.softmax(scores, dim=-1)  # (B, C, S)

        # 加权聚合记忆值
        mem_view = memory.reshape(B, self.num_channels, self.channel_size, H, hd)
        readout = torch.einsum("bchs,bcshd->bchd", weights, mem_view)  # (B, C, H, hd)
        return readout.reshape(B, self.num_channels, self.dim)

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """NTM 风格门控写入 (先擦除后添加).

        计算注意力权重, 经门控后擦除对应 slot 再写入新值。

        Args:
            key: (B, D) 写入键.
            value: (B, D) 写入值.
        """
        B = key.shape[0]
        memory = self._ensure_memory(B)  # (B, C, S, D)
        H, hd = self.num_heads, self.head_dim

        k = self.query_proj(key).reshape(B, H, hd)  # (B, H, hd)
        v = self.write_value_proj(value)  # (B, D)

        # 写入门 (每通道一个)
        gate = torch.sigmoid(self.write_gate(key))  # (B, C)

        keys = self.key_proj(memory).reshape(
            B, self.num_channels, self.channel_size, H, hd
        )
        scores = torch.einsum("bhd,bcshd->bchs", k, keys) / math.sqrt(hd)
        # 先对每个 head 在 slot 维归一化，再跨 head 平均，得到每个 slot 的写入强度
        w = F.softmax(scores, dim=-1).mean(dim=2)  # (B, C, S)

        # 广播: 门 × slot 写入强度 × 值
        g = gate.reshape(B, self.num_channels, 1, 1)        # (B, C, 1, 1)
        w_exp = w.reshape(B, self.num_channels, self.channel_size, 1)  # (B, C, S, 1)
        v_exp = v.reshape(B, 1, 1, self.dim)                # (B, 1, 1, D)

        erase = g * w_exp  # (B, C, S, 1)
        # 先擦除再添加: M ← M ⊙(1 − erase) + erase ⊙ v
        new_memory = memory * (1.0 - erase) + erase * v_exp
        self._memory = new_memory.clamp(-10.0, 10.0)

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """处理输入序列: 对每步执行 read + write, 返回融合记忆的输出.

        Args:
            x: (B, L, D) 输入序列.

        Returns:
            (B, L, D) 输出序列 (含残差 + LayerNorm).
        """
        residual = x
        B, L, D = x.shape
        assert D == self.dim, f"dim 不匹配: 期望 {self.dim}, 得到 {D}"

        outputs = []
        for t in range(L):
            xt = x[:, t, :]  # (B, D)
            readout = self.read(xt)  # (B, C, D)
            self.write(xt, xt)
            # 跨通道加权聚合
            cw = F.softmax(self.channel_weights, dim=0)  # (C,)
            agg = torch.einsum("c,bcd->bd", cw, readout)  # (B, D)
            out_t = self.output_proj(agg)  # (B, D)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # (B, L, D)
        return residual + self.norm(output)


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("DynamicMemoryChannel 测试")
    print("=" * 60)

    dmc = DynamicMemoryChannel(dim=32, num_channels=4, channel_size=8, num_heads=2)
    x = torch.randn(2, 10, 32, requires_grad=True)
    y = dmc(x)
    print(f"输入: {tuple(x.shape)}  输出: {tuple(y.shape)}")
    assert y.shape == (2, 10, 32)
    assert not torch.isnan(y).any(), "输出含 NaN"

    # 梯度流验证
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in dmc.parameters()
    )
    print(f"梯度流通畅: {has_grad}")
    assert has_grad, "参数无梯度"

    # 记忆状态管理
    state = dmc.get_state()
    print(f"记忆状态: {tuple(state.shape)}")
    assert state.shape == (2, 4, 8, 32)

    # detach_state: 截断 BPTT
    dmc.detach_state()
    assert dmc.get_state().requires_grad is False, "detach_state 未生效"

    # set_state / reset_state
    dmc.reset_state(3)
    assert dmc.get_state().shape == (3, 4, 8, 32)
    snapshot = dmc.get_state().clone()
    dmc.set_state(snapshot)
    assert torch.allclose(dmc.get_state(), snapshot)

    print("DMC smoke test PASSED")
