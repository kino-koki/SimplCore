# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""条件计算路由 (Conditional Compute Routing, CCR).

核心思想:
  每个 token 动态选择计算路径, 不同路径对应不同计算深度。
  通过路由网络对若干候选路径产生 logits, 经温度缩放的 softmax 得到路径概率,
  再对各路径 (不同深度) 的状态做加权融合 (支持 top-k 稀疏路由)。

机制:
  1. 路由网络 (Linear) 产生 num_paths 条路径的 logits;
  2. 每条路径均匀映射到 [min_depth, max_depth] 的计算深度;
  3. 通过反复施加变换层得到各深度状态;
  4. 用 softmax 概率对 (top-k) 路径状态加权融合;
  5. 残差连接 + LayerNorm。温度为可学习标量。
"""

from __future__ import annotations

from typing import List

import numpy as np

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.nn import functional as F
from simplcore.tensor.tensor import _add_grad


class ConditionalComputeRouter(nn.Module):
    """条件计算路由器 (Conditional Compute Router).

    每个 token 动态选择计算路径, 不同路径对应不同计算深度。

    Args:
        dim: 特征维度 D.
        num_paths: 候选路径数 P (每条路径对应一个计算深度).
        min_depth: 最小计算深度.
        max_depth: 最大计算深度.
        topk: top-k 路由的 k (k <= num_paths).
        dropout: 路由 logits 的 dropout 概率.
    """

    def __init__(
        self,
        dim: int,
        num_paths: int = 4,
        min_depth: int = 2,
        max_depth: int = 8,
        topk: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim > 0 and num_paths > 0
        assert 1 <= min_depth <= max_depth
        assert 1 <= topk <= num_paths
        self.dim = dim
        self.num_paths = num_paths
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.topk = topk
        self.dropout_p = dropout

        # 路由网络: 产生各路径 logits
        self.router = nn.Linear(dim, num_paths, bias=True)
        # 共享变换层: 反复施加以获得不同深度的状态
        self.transform = nn.Linear(dim, dim, bias=True)
        # 各路径对应的计算深度
        self.path_depths: List[int] = self._compute_path_depths()

        self.norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

        # 可学习温度 (缩放 softmax)
        self.temperature = nn.Parameter(torch.ones(1))

    def _compute_path_depths(self) -> List[int]:
        """将路径均匀映射到 [min_depth, max_depth] 区间.

        Returns:
            长度为 num_paths 的深度列表 (整数)。
        """
        if self.num_paths == 1:
            return [self.min_depth]
        depths: List[int] = []
        for p in range(self.num_paths):
            d = self.min_depth + round(
                p * (self.max_depth - self.min_depth) / (self.num_paths - 1)
            )
            depths.append(int(d))
        return depths

    def route(self, x: torch.Tensor) -> torch.Tensor:
        """计算每个 token 对各路径的路由概率.

        Args:
            x: (..., D) 输入特征.

        Returns:
            (..., P) 路径概率分布 (沿最后一维求和为 1)。
        """
        logits = self.router(x)
        if self.training and self.dropout_p > 0:
            logits = F.dropout(logits, p=self.dropout_p, training=self.training)
        # 温度缩放 (温度下限防止除零)
        t = self.temperature.clamp(min=1e-3)
        return F.softmax(logits / t, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """路由 token 到不同计算深度路径并加权融合.

        Args:
            x: (B, L, D) 输入序列.

        Returns:
            (B, L, D) 输出序列 (含残差 + LayerNorm).
        """
        residual = x
        h = self.norm(x)
        B, L, D = h.shape

        probs = self.route(h)  # (B, L, P)

        # 计算所有深度状态: 复用同一变换层反复施加
        max_d = max(self.path_depths)
        depth_states = {0: h}
        current = h
        for d in range(1, max_d + 1):
            current = current + F.gelu(self.transform(current))
            depth_states[d] = current

        # 收集各路径对应深度的状态
        path_states = torch.stack(
            [depth_states[d] for d in self.path_depths], dim=2
        )  # (B, L, P, D)

        if self.topk >= self.num_paths:
            # 全路径加权融合
            out = (probs.unsqueeze(-1) * path_states).sum(dim=2)
        else:
            # top-k 稀疏路由: 只融合概率最高的 k 条路径
            topk_probs, topk_idx = torch.topk(
                probs, k=self.topk, dim=-1
            )  # (B, L, k)
            # 重归一化 top-k 概率
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            # 沿路径维 (dim=2) 按 topk_idx 收集状态 (自研后端无 torch.gather, 用循环等价实现)
            Bs, Ls, ks = topk_idx.shape
            topk_states = torch.zeros((Bs, Ls, ks, D), dtype=path_states.dtype)
            idx_data = topk_idx.data.astype("int64")
            ps_data = path_states.data
            out_buf = np.zeros((Bs, Ls, ks, D), dtype=ps_data.dtype)
            for b in range(Bs):
                for l in range(Ls):
                    for j in range(ks):
                        p = int(idx_data[b, l, j])
                        out_buf[b, l, j, :] = ps_data[b, l, p, :]
            topk_states = path_states._make_out(
                out_buf.astype(ps_data.dtype), "ccr_gather", (path_states,)
            )
            # 反向: 把 topk_states 的梯度按 idx 加回 path_states (稀疏 scatter-add)
            def _backward():
                if topk_states.grad is None:
                    return
                g = np.asarray(topk_states.grad, dtype=ps_data.dtype)
                grad_ps = np.zeros_like(ps_data)
                for b in range(Bs):
                    for l in range(Ls):
                        for j in range(ks):
                            p = int(idx_data[b, l, j])
                            grad_ps[b, l, p, :] += g[b, l, j, :]
                # 通过 path_states 的 _add_grad (需拿到父张量)
                _add_grad(path_states, grad_ps)
            topk_states._backward = _backward
            out = (topk_probs.unsqueeze(-1) * topk_states).sum(dim=2)  # (B, L, D)

        out = self.out_norm(out)
        return residual + out

    def expected_depth(self, x: torch.Tensor) -> torch.Tensor:
        """计算每个 token 的期望计算深度.

        Args:
            x: (B, L, D) 输入序列.

        Returns:
            (B, L) 每个 token 的期望计算深度 (概率加权平均)。
        """
        probs = self.route(x)  # (B, L, P)
        depths = torch.tensor(
            self.path_depths, dtype=torch.float32, device=x.device
        )  # (P,)
        return (probs * depths).sum(dim=-1)  # (B, L)


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("ConditionalComputeRouter 测试")
    print("=" * 60)

    router = ConditionalComputeRouter(dim=32, num_paths=4, min_depth=2, max_depth=8, topk=2)
    x = torch.randn(2, 16, 32, requires_grad=True)
    y = router(x)
    print(f"输入: {tuple(x.shape)}  输出: {tuple(y.shape)}")
    assert y.shape == (2, 16, 32)
    assert not bool(np.isnan(y.data).any()), "输出含 NaN"

    # 路由概率
    probs = router.route(x)
    print(f"路由概率: {tuple(probs.shape)}, 每位置求和: {probs.sum(dim=-1).mean().item():.4f}")
    assert probs.shape == (2, 16, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 16), atol=1e-5)

    # 期望深度
    ed = router.expected_depth(x)
    print(f"期望计算深度 (均值): {ed.mean().item():.3f}, 路径深度: {router.path_depths}")
    assert ed.shape == (2, 16)
    assert router.min_depth - 1e-3 <= ed.mean().item() <= router.max_depth + 1e-3

    # 梯度流验证
    loss = (y ** 2).mean()
    loss.backward()
    has_grad = any(
        p.grad is not None and float(np.abs(p.grad).sum()) > 0
        for p in router.parameters()
    )
    print(f"梯度流通畅: {has_grad}")
    assert has_grad, "参数无梯度"

    print("CCR smoke test PASSED")
