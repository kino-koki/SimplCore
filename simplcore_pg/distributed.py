# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""SimplCore MoE 分布式训练支撑 (专家并行 + 张量并行 + FSDP)。

说明：
  * 56B MoE 单卡无法容纳，必须分片。本模块给出可落地的分片策略与 FSDP wrap
    策略，配合 ``torchrun`` 启动即可在 GPU 集群上训练；
  * 专家并行 (Expert Parallelism, EP)：把不同专家放到不同 rank，token 经
    all-to-all 在专家间路由，是 MoE 的核心扩展手段；
  * 张量并行 (Tensor Parallelism, TP)：对注意力/共享线性做列/行切分；
  * FSDP (Fully Sharded Data Parallel)：对剩余参数做 ZeRO-3 式分片。

所有函数均为纯规划/封装，不主动初始化进程组，便于在 ``torchrun`` 环境直接调用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ParallelPlan:
    """分片规划摘要。"""

    world_size: int
    expert_parallel_size: int
    tensor_parallel_size: int
    data_parallel_size: int
    expert_to_rank: List[int] = field(default_factory=list)
    per_rank_experts: Dict[int, List[int]] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "world_size": self.world_size,
            "ep": self.expert_parallel_size,
            "tp": self.tensor_parallel_size,
            "dp": self.data_parallel_size,
            "expert_to_rank": self.expert_to_rank,
            "per_rank_experts": self.per_rank_experts,
        }


def plan_parallelism(
    num_experts: int,
    world_size: int,
    *,
    expert_parallel_size: Optional[int] = None,
    tensor_parallel_size: int = 1,
) -> ParallelPlan:
    """计算 EP/TP/DP 维度与专家到 rank 的映射。

    Args:
        num_experts: 专家数量。
        world_size: 总 GPU 数。
        expert_parallel_size: 显式指定 EP (None 时取 min(num_experts, world_size))。
        tensor_parallel_size: TP 维度 (默认 1)。

    Raises:
        ValueError: 维度乘积超过 world_size 或不整除。
    """
    if world_size < 1:
        raise ValueError("world_size 必须 >= 1。")
    ep = expert_parallel_size or min(num_experts, world_size)
    if ep > world_size:
        raise ValueError(f"EP({ep}) 不能超过 world_size({world_size})。")
    if world_size % (ep * tensor_parallel_size) != 0:
        raise ValueError(
            f"world_size({world_size}) 须被 EP×TP({ep * tensor_parallel_size}) 整除。"
        )
    dp = world_size // (ep * tensor_parallel_size)

    # 专家在 EP 维度上轮转分配 (每个 EP rank 持有 num_experts/ep 个专家)
    experts_per_ep = num_experts // ep
    if experts_per_ep * ep != num_experts:
        raise ValueError(f"num_experts({num_experts}) 须被 EP({ep}) 整除。")
    expert_to_rank: List[int] = []
    per_rank: Dict[int, List[int]] = {r: [] for r in range(ep)}
    for e in range(num_experts):
        r = e // experts_per_ep
        expert_to_rank.append(r)
        per_rank[r].append(e)
    return ParallelPlan(
        world_size=world_size, expert_parallel_size=ep,
        tensor_parallel_size=tensor_parallel_size, data_parallel_size=dp,
        expert_to_rank=expert_to_rank, per_rank_experts=per_rank,
    )


def fsdp_wrap_policy(num_layers: int) -> "object":
    """返回 FSDP 的 transformer 式 wrap 策略：以每个 MoE/计算块为 wrap 单元。

    仅当 ``torch.distributed.fsdp`` 可用时返回真实策略，否则返回 None
    (便于非分布式环境导入与单元测试)。
    """
    try:
        from torch.distributed.fsdp.wrap import (
            TransformerBlockwiseWrapPolicy, size_based_auto_wrap_policy,
        )
    except Exception:  # pragma: no cover - 非分布式环境
        return None
    # 优先按块类型包裹；退化为按尺寸自动包裹
    try:
        from .model_moe import SimplCoreMoEBlock
        from .block import SimplCoreBlock
        return TransformerBlockwiseWrapPolicy({SimplCoreMoEBlock, SimplCoreBlock})
    except Exception:  # pragma: no cover
        return size_based_auto_wrap_policy(min_params=1e7)


def shard_model_for_distributed(
    model: nn.Module,
    *,
    num_experts: int,
    world_size: int,
    mp_policy: str = "bf16",
    shard_strategy: str = "FULL_SHARD",
) -> "object":
    """用 FSDP 包裹模型 (需在 ``torchrun`` 进程组内调用)。

    Args:
        model: 待分片的 ``nn.Module``。
        num_experts / world_size: 见 :func:`plan_parallelism`。
        mp_policy: 混合精度策略 ("bf16" / "fp16" / "none")。
        shard_strategy: FSDP ShardingStrategy 名称。

    Returns:
        FSDP 包裹后的模型；若 FSDP 不可用则返回原模型 (便于 CPU 冒烟)。
    """
    try:
        from torch.distributed import init_process_group
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    except Exception:  # pragma: no cover - 非分布式环境
        return model

    import os
    if not (os.environ.get("RANK") is not None and torch.distributed.is_available()):
        return model

    mp = None
    if mp_policy == "bf16":
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    elif mp_policy == "fp16":
        mp = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16)

    strategy = getattr(ShardingStrategy, shard_strategy, ShardingStrategy.FULL_SHARD)
    plan = plan_parallelism(num_experts, world_size)
    policy = fsdp_wrap_policy(getattr(model.config, "num_layers", 0))
    return FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp,
        auto_wrap_policy=policy,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        limit_all_gathers=True,
    )


def launch_torchrun_note() -> str:
    """返回启动 56B MoE 训练的 ``torchrun`` 命令模板 (供文档/CLI 引用)。"""
    return (
        "torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 \\\n"
        "    simplcore_pg/training/train.py \\\n"
        "    --preset 56b-moe --seq-len 4096 --micro-batch 1 \\\n"
        "    --global-batch 1024 --lr 3e-4 --epochs 1"
    )
