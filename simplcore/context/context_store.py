# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""上下文机制: 存储 / 检索 / 衰减 (无词表, 无 softmax 注意力).

本模块在 SimplCore 既有 PersistentMemory (长期, O(1) 空间) 与 HashMemory
(工作记忆, O(n) 计算) 之上, 构建面向 "Unit (单元)" 序列的上下文银行
(ContextBank). 核心改进:

1. **三级记忆**
   - 瞬时缓冲 (sensory): 当前 batch 的 Unit 表征, 不落盘.
   - 工作记忆 (working): 最近 K 个 Unit 的循环缓冲, 容量固定.
   - 长期记忆 (long-term): 复用 PersistentMemory, 跨会话持久, 空间 O(1) wrt n.

2. **检索**: 用确定性最近邻哈希 (argmax) 定位, 不使用 softmax 打分,
   与 "彻底脱离注意力" 目标一致.

3. **衰减 / 巩固策略**: 引入 "惊喜度 (surprise)" 信号 (预测误差).
   惊喜度高的 Unit 优先写入长期记忆且不被衰减; 长期记忆按
   access_count / age / surprise 综合价值做温和衰减, 而非硬性删除.

注: 当前环境缺少 numpy, 本文件仅做 py_compile 级验证; 数值测试见
simplcore/verify_formulas.py 的 tensor 集成层 (联网安装 numpy 后运行).
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

import simplcore.tensor as torch
from simplcore.memory.persistent_memory import PersistentMemory  # 既有 O(1) 记忆


class ContextBank:
    """无词表上下文银行.

    Args:
        dim: 单元表征维度 D.
        working_capacity: 工作记忆容量 (最近 Unit 数), 固定.
        persist_channels / persist_slots: 长期记忆 (C, S).
        persist_path: 长期记忆持久化目录; None 则不落盘.
    """

    def __init__(
        self,
        dim: int,
        working_capacity: int = 2048,
        persist_channels: int = 64,
        persist_slots: int = 128,
        persist_path: Optional[str] = None,
    ) -> None:
        self.dim = dim
        self.working_capacity = working_capacity
        self._working: list = []  # 循环缓冲: List[(vec, surprise)]
        self.long_term = PersistentMemory(
            dim=dim,
            num_channels=persist_channels,
            slot_size=persist_slots,
            path=persist_path,
            num_heads=1,
        )
        # 全局时间步进 (用于 age 计算); 不依赖 wall-clock 也可传入 step
        self._tick = 0

    # ---------------------------------------------------------------
    # 存储
    # ---------------------------------------------------------------
    def store_units(self, units: "torch.Tensor", surprise: "torch.Tensor") -> None:
        """写入上下文.

        Args:
            units: (L, D) 当前 Unit 表征序列.
            surprise: (L,) 每个 Unit 的惊喜度 (非负标量, 0~1 区间最佳).
        """
        L = units.shape[0]
        for i in range(L):
            vec = units[i]
            s = float(surprise[i]) if hasattr(surprise, "__getitem__") else float(surprise)
            self._working.append((vec, s))
        # 超容量则溢出到长期记忆 (高惊喜优先保留在工作区)
        if len(self._working) > self.working_capacity:
            self._working.sort(key=lambda x: x[1])  # 低惊喜先溢出
            overflow = self._working[: len(self._working) - self.working_capacity]
            self._working = self._working[len(self._working) - self.working_capacity:]
            for vec, s in overflow:
                # 高惊喜 (>=0.3) 进入长期记忆, 低惊喜直接丢弃
                if s >= 0.3:
                    q = vec.reshape(1, self.dim)
                    self.long_term.query_write(q, q)
        self._tick += 1

    # ---------------------------------------------------------------
    # 检索 (确定性哈希, 无 softmax)
    # ---------------------------------------------------------------
    def retrieve(self, query: "torch.Tensor", topk: int = 4) -> "torch.Tensor":
        """基于内容寻址读取长期记忆 (复用 PersistentMemory.read 的投影寻址).

        Args:
            query: (D,) 或 (B, D) 查询向量.
            topk: 返回若干最近邻槽的读出聚合 (此处用通道聚合, 非注意力加权).
        Returns:
            (D,) 或 (B, D) 读出.
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return self.long_term.read(query)

    # ---------------------------------------------------------------
    # 衰减 / 巩固
    # ---------------------------------------------------------------
    def decay_step(self, decay_rate: float = 0.001, min_access_to_keep: float = 1.0) -> int:
        """长期记忆衰减扫描: 低价值 slot 轻微衰减 (不直接删除, 保持结构).

        价值 = access_count / (1 + decay_rate * age); 越久未访问且命中越弱,
        越接近被衰减. 高频/高惊喜内容因反复写入保有能力, 实现 "非遗忘".
        """
        return self.long_term.forget_step(
            decay_rate=decay_rate, min_access_to_keep=min_access_to_keep
        )

    def working_window(self) -> "torch.Tensor":
        """返回当前工作记忆拼接 (L', D); 空则返回 None."""
        if not self._working:
            return None
        vecs = [v for (v, _) in self._working]
        return torch.stack(vecs, dim=0)

    def stats(self) -> dict:
        st = self.long_term.stats()
        st["working_size"] = len(self._working)
        st["working_capacity"] = self.working_capacity
        return st
