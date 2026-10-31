# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-

"""经验缓冲：在线轨迹入流、离线采样回放。"""
from __future__ import annotations


from .types import ReflectionStage, Trajectory

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch




# ---------------------------------------------------------------------------
# 经验回放池
# ---------------------------------------------------------------------------
class ExperienceBuffer:
    """优先级经验回放池。

    连续学习的三项精华之一：**回放**。按价值采样而非均匀采样，
    让有限的固化预算花在最有信息量的经验上。

    优先级默认取 ``score``（事后评分）；若未评分则用
    ``1 - confidence``（模型自己越不确信的越值得反思），
    再用 ``1 - grounded``（没有证据支撑的越值得反思）加权。
    """

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self._items: List[Trajectory] = []

    def add(self, traj: Trajectory) -> None:
        self._items.append(traj)
        if len(self._items) > self.capacity:
            # 丢弃最旧且价值最低的
            worst = min(range(len(self._items)),
                        key=lambda i: (self._items[i].score, -i))
            self._items.pop(worst)

    def extend(self, trajs: Sequence[Trajectory]) -> None:
        for t in trajs:
            self.add(t)

    def __len__(self) -> int:
        return len(self._items)

    def priority(self, traj: Trajectory) -> float:
        p = traj.score
        if p <= 0.0:
            p = (1.0 - float(traj.confidence)) + (0.0 if traj.grounded else 0.5)
        return float(p)

    def sample(self, n: int, strategy: str = "priority") -> List[Trajectory]:
        """采样 n 条经验。

        Args:
            n: 采样条数。
            strategy: ``priority``（按价值，带随机性）/ ``recent``（最新）/
                ``uniform``（均匀）。
        """
        if not self._items:
            return []
        n = min(n, len(self._items))
        if strategy == "recent":
            return list(self._items[-n:])
        if strategy == "uniform":
            idx = torch.randperm(len(self._items))[:n].tolist()
            return [self._items[i] for i in idx]
        # priority：softmax 加权无放回采样
        prio = torch.tensor([self.priority(t) for t in self._items], dtype=torch.float32)
        w = torch.softmax(prio * 4.0, dim=0)
        idx = torch.multinomial(w, n, replacement=False).tolist()
        return [self._items[i] for i in idx]

    def stats(self) -> Dict[str, Any]:
        if not self._items:
            return {"size": 0, "capacity": self.capacity}
        return {
            "size": len(self._items),
            "capacity": self.capacity,
            "grounded_ratio": sum(1 for t in self._items if t.grounded) / len(self._items),
            "mean_confidence": sum(t.confidence for t in self._items) / len(self._items),
        }

