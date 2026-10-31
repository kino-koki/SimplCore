# Copyright 2026 (C) 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplUnity（MoA）配置：多个小专家 + 路由门粘合出 MoE 效果。

设计目标（用户需求）：
* 冻结 SGC 基座，在之上粘合 N 个「小参数专家」，每个专家**训练数据必须不同**（域专属）；
* 路由门做 token 级 top-k 选择，加权求和 → 等效 MoE 的动态容量；
* 所有可学参数（路由 + 专家）经 ``WeightBudget`` 计入 2B/3B 上限；
* 支持 ``spawn_expert``：模型在真实环境学到新域并把经验写进权重时，后台长出新专家
  （模型冻结、程序/harness 生长 —— 与 Harness Learning 精神一致，规则驱动版）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class UnityConfig:
    """Mixture-of-Adapters 的超参。

    ``expert_hidden`` / ``router_hidden`` 为 0 时在建模型时按 ``2*dim`` / ``dim`` 解析。
    """

    num_experts: int = 4
    expert_hidden: int = 0          # 0 -> 2*dim
    expert_layers: int = 2
    top_k: int = 2
    router_hidden: int = 0          # 0 -> dim
    capacity_factor: float = 1.25
    router_z_loss: float = 1e-3     # 路由 logit 正则（防熵崩）
    load_balance_loss: float = 1e-2  # 专家利用率均衡正则
    expert_dropout: float = 0.0

    def resolve(self, dim: int) -> "ResolvedUnityConfig":
        return ResolvedUnityConfig(
            num_experts=max(1, int(self.num_experts)),
            expert_hidden=self.expert_hidden or 2 * dim,
            expert_layers=max(1, int(self.expert_layers)),
            top_k=min(int(self.top_k), max(1, int(self.num_experts))),
            router_hidden=self.router_hidden or dim,
            capacity_factor=float(self.capacity_factor),
            router_z_loss=float(self.router_z_loss),
            load_balance_loss=float(self.load_balance_loss),
            expert_dropout=float(self.expert_dropout),
            dim=dim,
        )


@dataclass
class ResolvedUnityConfig:
    num_experts: int
    expert_hidden: int
    expert_layers: int
    top_k: int
    router_hidden: int
    capacity_factor: float
    router_z_loss: float
    load_balance_loss: float
    expert_dropout: float
    dim: int

#: 各主干预设的 Unity 默认配置。2B 主干只学基础能力，域能力由后台专家承载；
#: 4 个专家 × ~50M（dim=2048, hidden=4096, 2 层）≈ 201M 可学参数，
#: 落在 WeightBudget 的 2B 可学 / 3B 总上限内。
UNITY_PRESETS = {
    "tuan-2b": UnityConfig(num_experts=4, top_k=2, expert_layers=2),
    "tuan-1b": UnityConfig(num_experts=4, top_k=2, expert_layers=2),
}

DEFAULT_UNITY = UnityConfig()


def for_preset(preset: Optional[str]) -> UnityConfig:
    """按主干预设取 Unity 配置；未知预设回落到默认（不静默改变预算口径）。"""
    if preset and preset in UNITY_PRESETS:
        return UNITY_PRESETS[preset]
    return DEFAULT_UNITY
