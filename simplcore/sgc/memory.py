# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""CMT 式压缩记忆库 —— SGC 原生落地。

CMT（*Compression Memory Training*, AAAI'25）的三阶段：**编码(压缩) → 存储 → 检索与聚合**。
它的关键取舍是**基座权重冻结**，新知识进外部记忆库，因此天然没有灾难性遗忘。

原文在**注意力层**注入 KV。SGC 没有注意力、也没有 KV 缓存（流式状态是
``2·M·D`` 的 O(1) 振荡器态），因此本模块把注入点改为 **SGC 残差流上的加性记忆项**
—— 这是 SGC 原生的等价物，且**不引入任何 softmax 注意力**。

沿用 CMT 的三项技术：

* **self-matching**   —— :func:`self_match_scores`，查询与记忆键归一化点积
* **top-aggregation** —— :meth:`CompressedMemoryBank.aggregate`，按相似度加权求和
* **memory-aware objective** —— :class:`MemoryAwareObjective`，只对"记忆相对基座的增量"计损失

诚实边界：本模块只负责**记忆的编码/存储/检索/聚合**。"记忆内容对不对"由
Dreaming 循环的 no-regression gate 负责（``simplcore/sgc/dream.py``），
不在此处保证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# self-matching / top-aggregation
# ---------------------------------------------------------------------------
def self_match_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """CMT 的 self-matching：查询向量与记忆键做归一化点积（余弦）。

    Args:
        query: (B, D) 查询向量。
        keys:  (N, D) 记忆键。
    Returns:
        (B, N) 相似度分数。
    """
    q = F.normalize(query.float(), dim=-1)
    k = F.normalize(keys.float(), dim=-1)
    return q @ k.t()


def top_aggregate(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    top_k: int = 4,
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CMT 的 top-aggregation：取 top-k 条记忆，按相似度做 softmax 加权求和。

    Args:
        query: (B, D)
        keys: (N, D)
        values: (N, D)
        top_k: 取前 k 条。
        temperature: softmax 温度，越小越尖锐。
    Returns:
        ``(aggregated (B, D), scores (B, N))`` —— scores 是全量相似度，
        便于调用方做"检索置信度"判断（例如低于阈值就不用记忆）。
    """
    scores = self_match_scores(query, keys)
    n = keys.shape[0]
    k = max(1, min(int(top_k), n))
    top_scores, top_idx = scores.topk(k, dim=-1)
    weights = torch.softmax(top_scores / max(temperature, 1e-6), dim=-1)   # (B, k)
    picked = values[top_idx]                                              # (B, k, D)
    aggregated = (weights.unsqueeze(-1) * picked).sum(dim=1)              # (B, D)
    return aggregated, scores


# ---------------------------------------------------------------------------
# memory-aware objective
# ---------------------------------------------------------------------------
class MemoryAwareObjective(nn.Module):
    """CMT 的 memory-aware objective：只对"有记忆 vs 无记忆"的**增量**计损失。

    原文用对比式 logit 集成来放大记忆的贡献。这里给出等价但更轻的形式：
    先分别算有记忆/无记忆的 logits，取差值作为增量 logits，再算交叉熵。
    梯度因此集中在"记忆带来的改变"上，而不是模型本已会的东西。

    Args:
        base_loss: 基础损失函数，默认交叉熵。签名为 ``(logits, targets) -> scalar``。
    """

    def __init__(self, base_loss=None) -> None:
        super().__init__()
        self.base_loss = base_loss or (
            lambda logits, targets: F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100,
            )
        )

    def forward(
        self,
        logits_with_memory: torch.Tensor,
        logits_without_memory: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits_with_memory: (B, L, V)
            logits_without_memory: (B, L, V)
            targets: (B, L)
        """
        delta = logits_with_memory - logits_without_memory
        return self.base_loss(delta, targets)


# ---------------------------------------------------------------------------
# 压缩器
# ---------------------------------------------------------------------------
class MemoryCompressor(nn.Module):
    """默认压缩器：SGC 前向 → 末端若干隐状态 → 投影为 k 个 soft memory token。

    CMT 原文用 causal Transformer decoder 把长文档压成 k 个 soft token。
    这里用 SGC 自己的隐状态做同样的事：因果卷积天然把整段上下文压进每个位置，
    末端位置因此已携带全文摘要，取末端 k 个位置即得一组 soft token。

    注意：这是**近似**而非等价 —— 原文的压缩器是可训练的独立模块，
    本实现是训练无关的读出。若要复现 CMT 的效果，应把本类替换为可训练压缩器。
    """

    def __init__(self, dim: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, model, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model: 任意提供 ``forward(idx) -> logits`` 的因果模型。
            ids: (B, L) token id。
        Returns:
            (B, k, D) soft memory token。
        """
        _, hidden = model(ids, return_hidden=True)          # (B, L, D)
        tail = hidden[:, -self.k:, :]                       # (B, k, D)
        if tail.shape[1] < self.k:                          # 序列比 k 短时补齐
            pad = hidden[:, :1, :].expand(-1, self.k - tail.shape[1], -1)
            tail = torch.cat([tail, pad], dim=1)
        return self.proj(tail)


# ---------------------------------------------------------------------------
# 记忆库
# ---------------------------------------------------------------------------
@dataclass
class MemoryItem:
    """一条压缩记忆。"""
    key: torch.Tensor          # (D,) 检索键
    value: torch.Tensor        # (D,) 注入用值
    meta: Dict[str, Any] = field(default_factory=dict)


class CompressedMemoryBank:
    """CMT 的存储阶段：定容的压缩记忆库。

    定容 + LRU 驱逐（不是 CMT 原文的做法，CMT 的记忆库无上限）。
    这里的定容是为了让在线系统的显存/内存占用可预测；驱逐策略为
    "最不常被检索"优先，因为未被检索过的记忆对线上收益最低。
    """

    def __init__(self, dim: int, capacity: int = 4096) -> None:
        self.dim = dim
        self.capacity = capacity
        self._keys: List[torch.Tensor] = []
        self._values: List[torch.Tensor] = []
        self._meta: List[Dict[str, Any]] = []
        self._hits: List[int] = []

    # -- 写入 ---------------------------------------------------------------
    def add(self, keys: torch.Tensor, values: torch.Tensor,
            meta: Optional[Sequence[Dict[str, Any]]] = None) -> None:
        """批量写入。

        Args:
            keys: (N, D)
            values: (N, D)
            meta: 可选，每个条目一份元信息。
        """
        if keys.shape[0] != values.shape[0]:
            raise ValueError("keys and values must have the same leading dim")
        for i in range(keys.shape[0]):
            self._keys.append(keys[i].detach().to(torch.float32))
            self._values.append(values[i].detach().to(torch.float32))
            self._meta.append(dict(meta[i]) if meta is not None else {})
            self._hits.append(0)
        self._evict()

    def _evict(self) -> None:
        while len(self._keys) > self.capacity:
            victim = min(range(len(self._hits)), key=lambda i: self._hits[i])
            self._keys.pop(victim)
            self._values.pop(victim)
            self._meta.pop(victim)
            self._hits.pop(victim)

    # -- 检索 ---------------------------------------------------------------
    def retrieve(self, query: torch.Tensor, top_k: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索最相似的 top-k 条记忆。

        Args:
            query: (B, D)
            top_k: 取前 k 条。
        Returns:
            ``(values (B, k, D), scores (B, k))``；库为空时返回空张量。
        """
        if not self._keys:
            b = query.shape[0]
            return (query.new_zeros((b, 0, self.dim)),
                    query.new_zeros((b, 0)))
        keys = torch.stack(self._keys)          # (N, D)
        values = torch.stack(self._values)      # (N, D)
        scores = self_match_scores(query, keys)  # (B, N)
        k = max(1, min(int(top_k), keys.shape[0]))
        top_scores, top_idx = scores.topk(k, dim=-1)
        picked = values[top_idx]                # (B, k, D)
        # 命中计数（LRU 驱逐用）
        for idx in top_idx.reshape(-1).tolist():
            self._hits[idx] += 1
        return picked, top_scores

    def aggregate(self, query: torch.Tensor, top_k: int = 4,
                  temperature: float = 0.07) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索 + top-aggregation，返回可直接注入模型的记忆向量。

        Returns:
            ``(memory (B, D), scores (B, N))``
        """
        if not self._keys:
            return query.new_zeros((query.shape[0], self.dim)), query.new_zeros((query.shape[0], 0))
        keys = torch.stack(self._keys)
        values = torch.stack(self._values)
        return top_aggregate(query, keys, values, top_k=top_k, temperature=temperature)

    # -- 状态 ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._keys)

    @property
    def keys(self) -> torch.Tensor:
        return torch.stack(self._keys) if self._keys else torch.zeros(0, self.dim)

    @property
    def values(self) -> torch.Tensor:
        return torch.stack(self._values) if self._values else torch.zeros(0, self.dim)

    def metas(self) -> List[Dict[str, Any]]:
        return list(self._meta)

    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._keys),
            "capacity": self.capacity,
            "hits_total": int(sum(self._hits)),
        }
