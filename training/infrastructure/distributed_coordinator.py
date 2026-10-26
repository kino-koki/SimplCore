"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

distributed_coordinator.py - 分布式协调模块
多节点任务协调、通信拓扑管理、梯度同步策略、流水线并行调度
"""

import os
import sys
import json
import time
import math
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import traceback

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.DistributedCoordinator")


class TrainingMode(Enum):
    """训练模式"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    HYBRID = "hybrid"


class CommunicationTopology(Enum):
    """通信拓扑"""
    RING = "ring"
    TREE = "tree"
    STAR = "star"
    ALL_REDUCE = "all_reduce"
    HIERARCHICAL = "hierarchical"


class ParallelStrategy(Enum):
    """并行策略"""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    TENSOR_PARALLEL = "tensor_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"


@dataclass
class PipelineStage:
    """流水线阶段"""
    stage_id: int = 0
    node_id: str = ""
    device_ids: List[int] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    micro_batches: int = 1
    compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    memory_usage_gb: float = 0.0


@dataclass
class CommunicationStats:
    """通信统计"""
    all_reduce_time_ms: float = 0.0
    broadcast_time_ms: float = 0.0
    scatter_time_ms: float = 0.0
    gather_time_ms: float = 0.0
    send_recv_time_ms: float = 0.0
    total_communication_time_ms: float = 0.0
    data_transferred_mb: float = 0.0
    gradient_sync_time_ms: float = 0.0
    param_sync_time_ms: float = 0.0


class DistributedCoordinator:
    """
    青小团分布式协调器

    管理多节点任务协调、通信拓扑、梯度同步、流水线并行和数据并行负载均衡
    """

    def __init__(
        self,
        model_name: str = "青小团",
        local_rank: int = 0,
        world_size: int = 1,
        node_rank: int = 0,
        num_nodes: int = 1,
        training_mode: TrainingMode = TrainingMode.SYNCHRONOUS,
        communication_topology: CommunicationTopology = CommunicationTopology.ALL_REDUCE,
        parallel_strategy: ParallelStrategy = ParallelStrategy.DATA_PARALLEL,
        gradient_sync_every_n_steps: int = 1,
        pipeline_parallel_stages: int = 1,
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        mixed_precision: str = "bf16",
        gradient_accumulation_steps: int = 1,
        enable_communication_profiling: bool = True,
    ):
        self.model_name = model_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.node_rank = node_rank
        self.num_nodes = num_nodes
        self.training_mode = training_mode
        self.communication_topology = communication_topology
        self.parallel_strategy = parallel_strategy
        self.gradient_sync_every_n_steps = gradient_sync_every_n_steps
        self.pipeline_parallel_stages = pipeline_parallel_stages
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_communication_profiling = enable_communication_profiling

        self.pipeline_stages: List[PipelineStage] = []
        self.communication_stats = CommunicationStats()
        self._comm_lock = threading.Lock()

        self._gradient_queue: List[Dict[str, torch.Tensor]] = []
        self._step_count = 0
        self._async_gradient_thread: Optional[threading.Thread] = None
        self._shutdown_flag = False

        self._load_balancing_enabled = True
        self._node_capacities: Dict[str, float] = {}

        self._pipeline_schedule: List[List[int]] = []

        self._init_distributed()

        if self.local_rank == 0:
            logger.info(f"青小团分布式协调器初始化完成")
            logger.info(f"青奈科技 - 模式={training_mode.value}, 拓扑={communication_topology.value}, "
                       f"并行={parallel_strategy.value}")

    def _init_distributed(self):
        """初始化分布式环境"""
        if not dist.is_initialized():
            return

        if self.local_rank == 0:
            logger.info(
                f"青小团 - 分布式环境: world_size={self.world_size}, "
                f"nodes={self.num_nodes}, local_rank={self.local_rank}"
            )

    def set_training_mode(self, mode: TrainingMode):
        """切换训练模式"""
        self.training_mode = mode
        logger.info(f"青小团 - 切换训练模式: {mode.value}")

    def should_sync_gradients(self, step: int) -> bool:
        """判断是否需要同步梯度"""
        if self.training_mode == TrainingMode.ASYNCHRONOUS:
            return False
        if self.training_mode == TrainingMode.SYNCHRONOUS:
            return True
        return step % self.gradient_sync_every_n_steps == 0

    def sync_gradients(
        self,
        model: torch.nn.Module,
        loss: Optional[torch.Tensor] = None,
    ) -> float:
        """
        同步梯度

        Args:
            model: 模型
            loss: loss值

        Returns:
            同步耗时(ms)
        """
        if not dist.is_initialized() or self.world_size <= 1:
            return 0.0

        start = time.time()

        try:
            if self.communication_topology == CommunicationTopology.ALL_REDUCE:
                avg_grad_norm = self._all_reduce_gradients(model)
            elif self.communication_topology == CommunicationTopology.RING:
                avg_grad_norm = self._ring_all_reduce_gradients(model)
            elif self.communication_topology == CommunicationTopology.HIERARCHICAL:
                avg_grad_norm = self._hierarchical_all_reduce_gradients(model)
            else:
                avg_grad_norm = self._all_reduce_gradients(model)

            sync_time = (time.time() - start) * 1000

            with self._comm_lock:
                self.communication_stats.gradient_sync_time_ms += sync_time
                self.communication_stats.total_communication_time_ms += sync_time

            return sync_time

        except Exception as e:
            logger.error(f"青小团 - 梯度同步失败: {e}")
            return 0.0

    def _all_reduce_gradients(self, model: torch.nn.Module) -> float:
        """All-Reduce 梯度同步"""
        total_norm = 0.0
        grad_buffers = []

        for param in model.parameters():
            if param.grad is not None:
                grad_buffers.append(param.grad)
                total_norm += param.grad.norm().item() ** 2

        if grad_buffers:
            for grad in grad_buffers:
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                grad.div_(self.world_size)

        return math.sqrt(total_norm)

    def _ring_all_reduce_gradients(self, model: torch.nn.Module) -> float:
        """Ring All-Reduce 梯度同步"""
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return 0.0

        chunks = []
        chunk_size = max(1, len(params) // self.world_size)
        for i in range(0, len(params), chunk_size):
            chunk = params[i:i + chunk_size]
            chunks.append(chunk)

        world_size = self.world_size
        rank = dist.get_rank()

        for r in range(world_size - 1):
            send_rank = (rank - r - 1) % world_size
            recv_rank = (rank + r + 1) % world_size

            send_chunk = chunks[(rank - r) % len(chunks)]
            recv_chunk = chunks[(rank - r - 1) % len(chunks)]

            if send_chunk and recv_chunk:
                send_tensor = torch.cat([p.grad.flatten() for p in send_chunk])
                recv_tensor = torch.zeros_like(send_tensor)

                req = dist.isend(send_tensor, dst=send_rank)
                dist.irecv(recv_tensor, src=recv_rank).wait()
                req.wait()

                offset = 0
                for p in recv_chunk:
                    size = p.grad.numel()
                    p.grad.data.add_(recv_tensor[offset:offset + size].reshape(p.grad.shape))
                    offset += size

        total_norm = sum(p.grad.norm().item() ** 2 for p in params if p.grad is not None)
        for p in params:
            p.grad.div_(world_size)

        return math.sqrt(total_norm)

    def _hierarchical_all_reduce_gradients(self, model: torch.nn.Module) -> float:
        """分层 All-Reduce 梯度同步（节点内+节点间）"""
        if self.num_nodes <= 1:
            return self._all_reduce_gradients(model)

        total_norm = 0.0
        params = [p for p in model.parameters() if p.grad is not None]

        intra_node_group = None
        inter_node_group = None

        try:
            intra_node_group = dist.new_group(
                ranks=[i for i in range(self.world_size)
                       if i // (self.world_size // self.num_nodes) == self.node_rank]
            )
        except Exception:
            pass

        if params:
            for param in params:
                total_norm += param.grad.norm().item() ** 2

        return math.sqrt(total_norm)

    def broadcast_parameters(self, model: torch.nn.Module, src_rank: int = 0):
        """广播模型参数"""
        if not dist.is_initialized():
            return

        start = time.time()

        for param in model.parameters():
            dist.broadcast(param.data, src=src_rank)

        broadcast_time = (time.time() - start) * 1000

        with self._comm_lock:
            self.communication_stats.broadcast_time_ms += broadcast_time
            self.communication_stats.total_communication_time_ms += broadcast_time

        logger.info(f"青小团 - 参数广播完成: {broadcast_time:.1f}ms")

    def setup_pipeline_parallelism(
        self,
        model_layers: List[str],
        devices_per_stage: Optional[List[List[int]]] = None,
    ):
        """
        设置流水线并行

        Args:
            model_layers: 模型层名称列表
            devices_per_stage: 每个阶段的设备分配
        """
        if self.parallel_strategy != ParallelStrategy.PIPELINE_PARALLEL:
            return

        num_stages = min(self.pipeline_parallel_stages, len(model_layers))
        layers_per_stage = math.ceil(len(model_layers) / num_stages)

        self.pipeline_stages = []
        for i in range(num_stages):
            start_idx = i * layers_per_stage
            end_idx = min((i + 1) * layers_per_stage, len(model_layers))

            stage = PipelineStage(
                stage_id=i,
                node_id=f"node_{i % self.num_nodes}",
                device_ids=devices_per_stage[i] if devices_per_stage and i < len(devices_per_stage) else [i],
                layers=model_layers[start_idx:end_idx],
                micro_batches=self.gradient_accumulation_steps,
            )
            self.pipeline_stages.append(stage)

        self._build_pipeline_schedule()

        if self.local_rank == 0:
            logger.info(f"青小团 - 流水线并行设置: {num_stages} 阶段")
            for stage in self.pipeline_stages:
                logger.info(f"  阶段 {stage.stage_id}: {len(stage.layers)} 层")

    def _build_pipeline_schedule(self):
        """构建流水线调度计划"""
        num_stages = len(self.pipeline_stages)
        micro_batches = self.gradient_accumulation_steps
        total_timesteps = num_stages + micro_batches - 1

        schedule = []
        for t in range(total_timesteps):
            timestep_ops = []
            for stage_idx in range(num_stages):
                mb_idx = t - stage_idx
                if 0 <= mb_idx < micro_batches:
                    timestep_ops.append(stage_idx)
            schedule.append(timestep_ops)

        self._pipeline_schedule = schedule

    def get_pipeline_schedule(self) -> List[List[int]]:
        """获取流水线调度计划"""
        return self._pipeline_schedule

    def balance_data_parallel_load(self, node_capacities: Dict[str, float]):
        """
        数据并行负载均衡

        Args:
            node_capacities: 各节点容量 {node_id: capacity_ratio}
        """
        if not self._load_balancing_enabled:
            return

        self._node_capacities = node_capacities
        total_capacity = sum(node_capacities.values())

        if total_capacity <= 0:
            return

        normalized = {
            node_id: cap / total_capacity
            for node_id, cap in node_capacities.items()
        }

        if self.local_rank == 0:
            logger.info(f"青小团 - 数据并行负载均衡:")
            for node_id, ratio in normalized.items():
                logger.info(f"  {node_id}: {ratio:.2%}")

        return normalized

    def get_optimal_batch_size_per_node(self, global_batch_size: int) -> Dict[str, int]:
        """计算每个节点的最优batch size"""
        if not self._node_capacities:
            total = self.num_nodes
            return {f"node_{i}": global_batch_size // total
                    for i in range(self.num_nodes)}

        total_capacity = sum(self._node_capacities.values())
        batch_sizes = {}
        remaining = global_batch_size

        for node_id, capacity in sorted(
            self._node_capacities.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if node_id == list(self._node_capacities.keys())[-1]:
                batch_sizes[node_id] = remaining
            else:
                batch_size = int(global_batch_size * capacity / total_capacity)
                batch_sizes[node_id] = batch_size
                remaining -= batch_size

        return batch_sizes

    def monitor_communication(self) -> CommunicationStats:
        """获取通信统计"""
        return self.communication_stats

    def reset_communication_stats(self):
        """重置通信统计"""
        with self._comm_lock:
            self.communication_stats = CommunicationStats()

    def optimize_transmission(
        self,
        tensor: torch.Tensor,
        dst_rank: int,
        tag: int = 0,
    ) -> float:
        """
        优化的节点间数据传输

        Args:
            tensor: 要传输的张量
            dst_rank: 目标rank
            tag: 消息标签

        Returns:
            传输耗时(ms)
        """
        if not dist.is_initialized():
            return 0.0

        start = time.time()

        if tensor.numel() > 1024 * 1024:
            tensor = tensor.contiguous()

        dist.send(tensor, dst=dst_rank, tag=tag)

        transfer_time = (time.time() - start) * 1000

        with self._comm_lock:
            data_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
            self.communication_stats.data_transferred_mb += data_mb
            self.communication_stats.send_recv_time_ms += transfer_time
            self.communication_stats.total_communication_time_ms += transfer_time

        return transfer_time

    def all_gather_expert_distribution(
        self,
        local_expert_usage: Dict[int, float],
    ) -> Dict[int, List[float]]:
        """
        收集所有节点的专家负载分布

        Args:
            local_expert_usage: 本地专家使用率

        Returns:
            所有节点的专家使用率
        """
        if not dist.is_initialized():
            return {k: [v] for k, v in local_expert_usage.items()}

        gathered = [{} for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_expert_usage)

        result: Dict[int, List[float]] = {}
        for node_data in gathered:
            for expert_id, usage in node_data.items():
                if expert_id not in result:
                    result[expert_id] = []
                result[expert_id].append(usage)

        return result

    def get_coordinator_status(self) -> Dict[str, Any]:
        """获取协调器状态"""
        return {
            "company": "青奈科技",
            "model": self.model_name,
            "training_mode": self.training_mode.value,
            "topology": self.communication_topology.value,
            "parallel_strategy": self.parallel_strategy.value,
            "world_size": self.world_size,
            "num_nodes": self.num_nodes,
            "pipeline_stages": len(self.pipeline_stages),
            "gradient_sync_interval": self.gradient_sync_every_n_steps,
            "mixed_precision": self.mixed_precision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "communication_stats": asdict(self.communication_stats),
            "load_balancing_enabled": self._load_balancing_enabled,
        }

    def shutdown(self):
        """关闭分布式协调器"""
        self._shutdown_flag = True
        logger.info("青小团 - 分布式协调器已关闭")
        logger.info(f"青奈科技 - 通信统计: {asdict(self.communication_stats)}")


class GradientSyncManager:
    """
    梯度同步管理器
    负责高效的梯度同步策略，支持异步和同步模式
    """

    def __init__(
        self,
        world_size: int,
        sync_mode: str = "sync",
        overlap_communication: bool = True,
        compression: bool = False,
    ):
        self.world_size = world_size
        self.sync_mode = sync_mode
        self.overlap_communication = overlap_communication
        self.compression = compression
        self._sync_time_history: List[float] = []

    def sync(self, model: torch.nn.Module) -> float:
        """同步梯度"""
        start = time.time()

        if self.sync_mode == "sync":
            self._sync_gradients(model)
        else:
            self._async_gradients(model)

        sync_time = (time.time() - start) * 1000
        self._sync_time_history.append(sync_time)

        return sync_time

    def _sync_gradients(self, model: torch.nn.Module):
        """同步模式梯度同步"""
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return

        grad_buffers = [p.grad for p in params]
        for grad in grad_buffers:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)

    def _async_gradients(self, model: torch.nn.Module):
        """异步模式梯度同步"""
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return

        handles = []
        for param in params:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            handles.append(handle)

        for param in params:
            param.grad.div_(self.world_size)

    def get_avg_sync_time(self) -> float:
        """获取平均同步时间"""
        if not self._sync_time_history:
            return 0.0
        return float(np.mean(self._sync_time_history[-100:]))


class PipelineScheduler:
    """
    流水线并行调度器
    负责微批次调度和流水线气泡优化
    """

    def __init__(self, num_stages: int, num_microbatches: int):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.schedule = self._build_1f1b_schedule()

    def _build_1f1b_schedule(self) -> List[List[Tuple[int, str]]]:
        """构建1F1B调度方案"""
        schedule = []
        num_warmup = self.num_stages - 1

        for i in range(num_warmup):
            timestep = []
            for s in range(min(i + 1, self.num_stages)):
                mb = i - s
                if 0 <= mb < self.num_microbatches:
                    timestep.append((s, "forward"))
            schedule.append(timestep)

        steady_steps = self.num_microbatches - num_warmup
        for i in range(steady_steps):
            timestep = []
            for s in range(self.num_stages):
                mb = num_warmup + i - s
                if 0 <= mb < self.num_microbatches:
                    timestep.append((s, "forward" if "forward" else "backward"))
            schedule.append(timestep)

        num_cooldown = self.num_stages - 1
        for i in range(num_cooldown):
            timestep = []
            for s in range(self.num_stages - i - 1, self.num_stages):
                mb = steady_steps + num_warmup + i - s
                if 0 <= mb < self.num_microbatches:
                    timestep.append((s, "backward"))
            schedule.append(timestep)

        return schedule

    def compute_bubble_ratio(self) -> float:
        """计算流水线气泡比例"""
        total_ops = len(self.schedule) * self.num_stages
        useful_ops = self.num_stages * self.num_microbatches * 2
        return max(0.0, 1.0 - useful_ops / total_ops)

    def get_schedule(self) -> List[List[Tuple[int, str]]]:
        return self.schedule
