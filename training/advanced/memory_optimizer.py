"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

memory_optimizer.py - 显存优化模块
激活值检查点、内存offload（参数/优化器状态→CPU）、
内存碎片整理、显存预算管理、KV Cache优化、
MoE专家缓存优化、显存使用预测
"""

import os
import sys
import math
import time
import logging
import threading
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Set, Iterator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.MemoryOptimizer")


class OffloadLocation(Enum):
    CPU = "cpu"
    DISK = "disk"


class CheckpointMode(Enum):
    FULL = "full"
    SELECTIVE = "selective"
    NONE = "none"


@dataclass
class MemoryBudget:
    total_gb: float = 0.0
    reserved_gb: float = 2.0
    model_gb: float = 0.0
    optimizer_gb: float = 0.0
    activation_gb: float = 0.0
    kv_cache_gb: float = 0.0
    moe_cache_gb: float = 0.0
    available_gb: float = 0.0
    peak_gb: float = 0.0
    usage_percent: float = 0.0


@dataclass
class MemoryStats:
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    cached_bytes: int = 0
    peak_allocated_bytes: int = 0
    fragmentation_percent: float = 0.0
    offloaded_bytes: int = 0
    checkpoint_saved_bytes: int = 0
    kv_cache_saved_bytes: int = 0
    moe_cache_hit_rate: float = 0.0
    total_allocations: int = 0


class OffloadBuffer:
    def __init__(self, max_cpu_gb: float = 64.0):
        self.max_cpu_bytes = int(max_cpu_gb * 1024 ** 3)
        self.buffer: Dict[str, torch.Tensor] = OrderedDict()
        self.current_bytes = 0

    def offload(self, key: str, tensor: torch.Tensor) -> bool:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if self.current_bytes + tensor_bytes > self.max_cpu_bytes:
            self._evict(tensor_bytes)
        cpu_tensor = tensor.cpu().pin_memory()
        self.buffer[key] = cpu_tensor
        self.current_bytes += tensor_bytes
        return True

    def load(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        if key not in self.buffer:
            return None
        cpu_tensor = self.buffer.pop(key)
        self.current_bytes -= cpu_tensor.numel() * cpu_tensor.element_size()
        return cpu_tensor.to(device, non_blocking=True)

    def _evict(self, needed_bytes: int):
        while self.current_bytes + needed_bytes > self.max_cpu_bytes and self.buffer:
            _, tensor = self.buffer.popitem(last=False)
            self.current_bytes -= tensor.numel() * tensor.element_size()
            del tensor

    def clear(self):
        self.buffer.clear()
        self.current_bytes = 0


class QingxiaotuanMemoryOptimizer:
    """
    青小团显存优化器
    支持激活值检查点、内存offload、碎片整理、显存预算管理、
    KV Cache优化、MoE专家缓存优化、显存使用预测

    针对70B MoE模型的显存瓶颈，优化激活值显存和专家缓存
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_mode: CheckpointMode = CheckpointMode.SELECTIVE,
        selective_checkpoint_layers: Optional[List[str]] = None,
        enable_offload: bool = True,
        offload_location: OffloadLocation = OffloadLocation.CPU,
        offload_max_cpu_gb: float = 64.0,
        enable_kv_cache_optimization: bool = True,
        kv_cache_quantization: bool = True,
        kv_cache_bits: int = 8,
        enable_moe_cache: bool = True,
        moe_cache_capacity: int = 128,
        memory_fragmentation_compact: bool = True,
        compact_interval: int = 100,
        memory_budget_gb: float = 80.0,
        enable_memory_prediction: bool = True,
    ):
        self.model = model
        self.checkpoint_mode = checkpoint_mode
        self.selective_checkpoint_layers = selective_checkpoint_layers or []
        self.enable_offload = enable_offload
        self.offload_location = offload_location
        self.offload_max_cpu_gb = offload_max_cpu_gb
        self.enable_kv_cache_optimization = enable_kv_cache_optimization
        self.kv_cache_quantization = kv_cache_quantization
        self.kv_cache_bits = kv_cache_bits
        self.enable_moe_cache = enable_moe_cache
        self.moe_cache_capacity = moe_cache_capacity
        self.memory_fragmentation_compact = memory_fragmentation_compact
        self.compact_interval = compact_interval
        self.memory_budget_gb = memory_budget_gb
        self.enable_memory_prediction = enable_memory_prediction

        self.offload_buffer = OffloadBuffer(max_cpu_gb=offload_max_cpu_gb)
        self.moe_expert_cache: Dict[str, OrderedDict] = {}
        self.kv_cache_stats: Dict[str, Any] = {}
        self.stats = MemoryStats()
        self.budget = MemoryBudget(total_gb=memory_budget_gb)
        self._step_counter = 0
        self._activation_handles: List[Any] = []
        self._checkpointed_modules: List[nn.Module] = []
        self._memory_history: deque = deque(maxlen=1000)

        self._setup_checkpointing()
        self._init_moe_cache()
        logger.info(f"显存优化器初始化完成: checkpoint={checkpoint_mode.value}, "
                    f"offload={'启用' if enable_offload else '禁用'}, "
                    f"KV Cache优化={'启用' if enable_kv_cache_optimization else '禁用'}")

    def _setup_checkpointing(self):
        if self.checkpoint_mode == CheckpointMode.NONE:
            return

        for name, module in self.model.named_modules():
            if self.checkpoint_mode == CheckpointMode.FULL:
                if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    self._checkpointed_modules.append(module)
            elif self.checkpoint_mode == CheckpointMode.SELECTIVE:
                if any(pattern in name for pattern in self.selective_checkpoint_layers):
                    self._checkpointed_modules.append(module)
                elif any(k in name.lower() for k in ["attention", "mlp", "ffn", "expert"]):
                    self._checkpointed_modules.append(module)

        if self.checkpointed_modules:
            logger.info(f"激活值检查点已设置: {len(self.checkpointed_modules)} 个模块")

    def _init_moe_cache(self):
        for name, module in self.model.named_modules():
            if "expert" in name.lower() or "moe" in name.lower():
                self.moe_expert_cache[name] = OrderedDict()

    @property
    def checkpointed_modules(self) -> List[nn.Module]:
        return self._checkpointed_modules

    def enable_checkpointing(self, module: nn.Module):
        if module not in self._checkpointed_modules:
            self._checkpointed_modules.append(module)

    def offload_tensor(self, key: str, tensor: torch.Tensor) -> bool:
        if not self.enable_offload:
            return False
        success = self.offload_buffer.offload(key, tensor)
        if success:
            self.stats.offloaded_bytes += tensor.numel() * tensor.element_size()
        return success

    def load_tensor(self, key: str, device: torch.device) -> Optional[torch.Tensor]:
        return self.offload_buffer.load(key, device)

    def offload_optimizer_state(self, optimizer: torch.optim.Optimizer):
        if not self.enable_offload:
            return
        offloaded_count = 0
        for param_group in optimizer.param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    key = f"optimizer_state_{id(p)}"
                    self.offload_buffer.offload(key, p.grad)
                    offloaded_count += 1
        if offloaded_count > 0:
            logger.debug(f"优化器状态offload: {offloaded_count} 个张量")

    def compact_memory(self):
        if not self.memory_fragmentation_compact:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("显存碎片整理完成")

    def optimize_kv_cache(self, cache_k: torch.Tensor, cache_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_kv_cache_optimization:
            return cache_k, cache_v

        if self.kv_cache_quantization and self.kv_cache_bits < 16:
            original_dtype = cache_k.dtype
            if original_dtype == torch.float32:
                scale_k = cache_k.abs().max() / (2 ** (self.kv_cache_bits - 1) - 1)
                scale_v = cache_v.abs().max() / (2 ** (self.kv_cache_bits - 1) - 1)
                quant_k = torch.round(cache_k / scale_k).clamp(
                    -(2 ** (self.kv_cache_bits - 1)),
                    2 ** (self.kv_cache_bits - 1) - 1
                ).to(torch.int8)
                quant_v = torch.round(cache_v / scale_v).clamp(
                    -(2 ** (self.kv_cache_bits - 1)),
                    2 ** (self.kv_cache_bits - 1) - 1
                ).to(torch.int8)
                self.stats.kv_cache_saved_bytes += (cache_k.numel() * 4 - quant_k.numel() * 1)
                return quant_k, quant_v

        return cache_k, cache_v

    def restore_kv_cache(self, quant_k: torch.Tensor, quant_v: torch.Tensor,
                         scale_k: torch.Tensor, scale_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if quant_k.dtype == torch.int8:
            k_float = quant_k.float() * scale_k
            v_float = quant_v.float() * scale_v
            return k_float, v_float
        return quant_k, quant_v

    def get_moe_expert_cache(self, expert_name: str, input_key: str) -> Optional[torch.Tensor]:
        if not self.enable_moe_cache:
            return None
        cache = self.moe_expert_cache.get(expert_name)
        if cache is None:
            return None
        if input_key in cache:
            cache.move_to_end(input_key)
            self.stats.moe_cache_hit_rate = 0.9 * self.stats.moe_cache_hit_rate + 0.1
            return cache[input_key]
        self.stats.moe_cache_hit_rate = 0.9 * self.stats.moe_cache_hit_rate + 0.0
        return None

    def set_moe_expert_cache(self, expert_name: str, input_key: str, output: torch.Tensor):
        if not self.enable_moe_cache:
            return
        cache = self.moe_expert_cache.get(expert_name)
        if cache is None:
            return
        cache[input_key] = output.cpu()
        while len(cache) > self.moe_cache_capacity:
            cache.popitem(last=False)

    def clear_moe_cache(self, expert_name: Optional[str] = None):
        if expert_name is not None:
            cache = self.moe_expert_cache.get(expert_name)
            if cache is not None:
                cache.clear()
        else:
            for cache in self.moe_expert_cache.values():
                cache.clear()
        logger.info(f"MoE专家缓存已清除: {expert_name or '全部'}")

    def predict_memory_usage(self, batch_size: int, seq_length: int, num_layers: int) -> MemoryBudget:
        if not self.enable_memory_prediction:
            return self.budget

        bytes_per_param = 2
        model_params = sum(p.numel() for p in self.model.parameters())
        model_gb = model_params * bytes_per_param / (1024 ** 3)

        optimizer_gb = model_gb * 3

        hidden_size = 7168
        num_heads = 64
        head_dim = hidden_size // num_heads
        kv_cache_per_layer = 2 * batch_size * seq_length * num_heads * head_dim * 2
        kv_cache_gb = kv_cache_per_layer * num_layers / (1024 ** 3)
        if self.kv_cache_quantization:
            kv_cache_gb *= self.kv_cache_bits / 16

        activation_gb = batch_size * seq_length * hidden_size * 4 * 34 / (1024 ** 3)
        if self.checkpoint_mode != CheckpointMode.NONE:
            activation_gb *= 0.3

        moe_cache_gb = self.moe_cache_capacity * num_layers * 0.5

        total_gb = model_gb + optimizer_gb + activation_gb + kv_cache_gb + moe_cache_gb

        self.budget = MemoryBudget(
            total_gb=total_gb,
            reserved_gb=2.0,
            model_gb=model_gb,
            optimizer_gb=optimizer_gb,
            activation_gb=activation_gb,
            kv_cache_gb=kv_cache_gb,
            moe_cache_gb=moe_cache_gb,
            available_gb=self.memory_budget_gb - total_gb,
            usage_percent=(total_gb / self.memory_budget_gb) * 100,
        )
        return self.budget

    def get_current_memory_usage(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.stats.allocated_bytes = int(allocated * 1024 ** 3)
        self.stats.reserved_bytes = int(reserved * 1024 ** 3)
        self.stats.peak_allocated_bytes = int(peak * 1024 ** 3)
        self._memory_history.append(allocated)
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "peak_gb": peak,
            "free_gb": self.memory_budget_gb - allocated,
        }

    def analyze_fragmentation(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        if reserved == 0:
            return 0.0
        frag = 1.0 - (allocated / reserved)
        self.stats.fragmentation_percent = frag * 100
        return frag * 100

    def optimize_memory_layout(self):
        self.analyze_fragmentation()
        if self.stats.fragmentation_percent > 20:
            logger.info(f"碎片率 {self.stats.fragmentation_percent:.1f}%，执行碎片整理")
            self.compact_memory()

    def apply_gradient_checkpointing(self):
        for module in self._checkpointed_modules:
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
            elif hasattr(module, "activation_checkpointing"):
                module.activation_checkpointing = True
        logger.info(f"梯度检查点已应用于 {len(self._checkpointed_modules)} 个模块")

    def estimate_safe_batch_size(self, target_usage: float = 0.9) -> int:
        current = self.get_current_memory_usage()
        if not current:
            return 1
        allocated_gb = current.get("allocated_gb", 1)
        if allocated_gb < 1:
            return 1
        scaling_factor = (self.memory_budget_gb * target_usage) / allocated_gb
        return max(1, int(scaling_factor))

    def step(self):
        self._step_counter += 1
        self.get_current_memory_usage()

        if self._step_counter % self.compact_interval == 0:
            self.optimize_memory_layout()

        if self._step_counter % 50 == 0:
            usage = self.get_current_memory_usage()
            frag = self.analyze_fragmentation()
            logger.debug(f"[显存] allocated={usage.get('allocated_gb', 0):.1f}GB, "
                        f"peak={usage.get('peak_gb', 0):.1f}GB, "
                        f"frag={frag:.1f}%")

    def get_memory_stats(self) -> MemoryStats:
        return self.stats

    def get_budget(self) -> MemoryBudget:
        return self.budget

    def state_dict(self) -> Dict[str, Any]:
        usage = self.get_current_memory_usage()
        return {
            "checkpoint_mode": self.checkpoint_mode.value,
            "kv_cache_quantization": self.kv_cache_quantization,
            "kv_cache_bits": self.kv_cache_bits,
            "moe_cache_capacity": self.moe_cache_capacity,
            "stats": {
                "offloaded_bytes": self.stats.offloaded_bytes,
                "checkpoint_saved_bytes": self.stats.checkpoint_saved_bytes,
                "kv_cache_saved_bytes": self.stats.kv_cache_saved_bytes,
                "moe_cache_hit_rate": self.stats.moe_cache_hit_rate,
            },
            "current_memory": usage,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        checkpoint_mode = state_dict.get("checkpoint_mode", "selective")
        self.checkpoint_mode = CheckpointMode(checkpoint_mode)
        self.kv_cache_quantization = state_dict.get("kv_cache_quantization", True)
        self.kv_cache_bits = state_dict.get("kv_cache_bits", self.kv_cache_bits)
        self.moe_cache_capacity = state_dict.get("moe_cache_capacity", self.moe_cache_capacity)

        stats_data = state_dict.get("stats", {})
        if stats_data:
            self.stats.offloaded_bytes = stats_data.get("offloaded_bytes", 0)
            self.stats.kv_cache_saved_bytes = stats_data.get("kv_cache_saved_bytes", 0)
            self.stats.moe_cache_hit_rate = stats_data.get("moe_cache_hit_rate", 0.0)

        logger.info("显存优化器状态已恢复")

    def print_memory_summary(self):
        usage = self.get_current_memory_usage()
        budget_pred = self.predict_memory_usage(1, 524288, 80)
        frag = self.analyze_fragmentation()

        summary = [
            "=" * 60,
            "青小团显存优化 - 显存信息摘要",
            "=" * 60,
            f"显存预算: {self.memory_budget_gb:.0f}GB",
            f"检查点模式: {self.checkpoint_mode.value}",
            f"KV Cache量化: {'启用' if self.kv_cache_quantization else '禁用'} ({self.kv_cache_bits}bits)",
            f"MoE专家缓存: {'启用' if self.enable_moe_cache else '禁用'} (容量={self.moe_cache_capacity})",
            f"内存Offload: {'启用' if self.enable_offload else '禁用'} (最大CPU={self.offload_max_cpu_gb:.0f}GB)",
            f"---",
            f"当前分配: {usage.get('allocated_gb', 0):.1f}GB",
            f"当前预留: {usage.get('reserved_gb', 0):.1f}GB",
            f"峰值: {usage.get('peak_gb', 0):.1f}GB",
            f"碎片率: {frag:.1f}%",
            f"---",
            f"预测总显存: {budget_pred.total_gb:.1f}GB",
            f"模型参数: {budget_pred.model_gb:.1f}GB",
            f"优化器状态: {budget_pred.optimizer_gb:.1f}GB",
            f"激活值: {budget_pred.activation_gb:.1f}GB",
            f"KV Cache: {budget_pred.kv_cache_gb:.1f}GB",
            f"MoE缓存: {budget_pred.moe_cache_gb:.1f}GB",
            f"可用: {budget_pred.available_gb:.1f}GB",
            f"使用率: {budget_pred.usage_percent:.1f}%",
            f"---",
            f"Offload数据量: {self.stats.offloaded_bytes / 1024 / 1024 / 1024:.2f}GB",
            f"KV Cache节省: {self.stats.kv_cache_saved_bytes / 1024 / 1024 / 1024:.2f}GB",
            f"MoE缓存命中率: {self.stats.moe_cache_hit_rate:.4f}",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
