"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

gradient_optimizer.py - 梯度优化模块
梯度压缩（TopK/随机/量化）、梯度延迟补偿、梯度累积（12步）、
梯度裁剪（按范数/按值）、梯度噪声注入、MoE梯度优化
"""

import os
import sys
import math
import time
import random
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.GradientOptimizer")


class CompressionMethod(Enum):
    TOPK = "topk"
    RANDOM = "random"
    QUANTIZATION = "quantization"
    NONE = "none"


class ClipMethod(Enum):
    NORM = "norm"
    VALUE = "value"
    BOTH = "both"


@dataclass
class GradientStats:
    total_steps: int = 0
    grad_norm_mean: float = 0.0
    grad_norm_max: float = 0.0
    grad_norm_min: float = float("inf")
    compression_ratio: float = 0.0
    compression_savings: int = 0
    noise_std: float = 0.0
    clipped_count: int = 0
    sparse_grad_ratio: float = 0.0
    moe_grad_sparsity: float = 0.0
    accumulation_count: int = 0
    delay_steps: int = 0


@dataclass
class GradientAccumulator:
    buffer: Dict[str, torch.Tensor] = field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 12
    loss_sum: float = 0.0


class QingxiaotuanGradientOptimizer:
    """
    青小团梯度优化器
    支持梯度压缩（TopK/随机/量化）、延迟补偿、梯度累积（默认12步）、
    梯度裁剪（按范数/按值）、梯度噪声注入、MoE稀疏梯度优化

    针对70B MoE模型的稀疏梯度特性，优化专家路由的梯度通信效率
    """

    def __init__(
        self,
        model: nn.Module,
        compression_method: CompressionMethod = CompressionMethod.NONE,
        compression_ratio: float = 0.01,
        clip_method: ClipMethod = ClipMethod.NORM,
        clip_value: float = 1.0,
        clip_norm: float = 1.0,
        accumulation_steps: int = 12,
        noise_std: float = 0.0,
        noise_decay: float = 0.999,
        delay_compensation: bool = False,
        delay_alpha: float = 0.6,
        moe_grad_optimization: bool = True,
        moe_grad_scale: float = 0.5,
        quantization_bits: int = 8,
    ):
        self.model = model
        self.compression_method = compression_method
        self.compression_ratio = compression_ratio
        self.clip_method = clip_method
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.accumulation_steps = accumulation_steps
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.delay_compensation_enabled = delay_compensation
        self.delay_alpha = delay_alpha
        self.moe_grad_optimization = moe_grad_optimization
        self.moe_grad_scale = moe_grad_scale
        self.quantization_bits = quantization_bits

        self.accumulator = GradientAccumulator(max_steps=accumulation_steps)
        self.delay_buffer: Dict[str, torch.Tensor] = {}
        self.moe_param_names: List[str] = []
        self.stats = GradientStats()
        self._step_counter = 0
        self._detect_moe_params()

    def _detect_moe_params(self):
        for name, param in self.model.named_parameters():
            if "expert" in name.lower() or "moe" in name.lower() or "gate" in name.lower() or "router" in name.lower():
                self.moe_param_names.append(name)
        if self.moe_param_names:
            logger.info(f"检测到 {len(self.moe_param_names)} 个MoE参数，启用MoE梯度优化")

    def accumulate_gradient(self, loss: torch.Tensor, retain_graph: bool = False):
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward(retain_graph=retain_graph)
        self.accumulator.loss_sum += loss.item()
        self.accumulator.step_count += 1
        self.stats.accumulation_count = self.accumulator.step_count

    def is_accumulation_ready(self) -> bool:
        return self.accumulator.step_count >= self.accumulation_steps

    def step(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        self._step_counter += 1
        self.stats.total_steps += 1

        if self.moe_grad_optimization:
            self._apply_moe_grad_scaling()

        if self.delay_compensation_enabled:
            self._apply_delay_compensation()

        grad_norm = self._clip_gradients()

        if self.noise_std > 0:
            self._apply_gradient_noise()

        compressed_info = {}
        if self.compression_method != CompressionMethod.NONE:
            compressed_info = self._compress_gradients()

        optimizer.step()
        optimizer.zero_grad()
        self.accumulator.step_count = 0
        self.accumulator.loss_sum = 0.0

        if self.noise_decay < 1.0:
            self.noise_std *= self.noise_decay

        step_info = {
            "grad_norm": grad_norm,
            "clipped": self.stats.clipped_count > 0,
            "accumulation_steps": self.accumulation_steps,
            "noise_std": self.noise_std,
            "compression_ratio": compressed_info.get("ratio", 0.0),
            "compression_savings": compressed_info.get("savings", 0),
            "moe_grad_sparsity": self.stats.moe_grad_sparsity,
        }
        return step_info

    def _clip_gradients(self) -> float:
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if not parameters:
            return 0.0

        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm)

        self.stats.grad_norm_mean = 0.9 * self.stats.grad_norm_mean + 0.1 * total_norm
        self.stats.grad_norm_max = max(self.stats.grad_norm_max, total_norm)
        self.stats.grad_norm_min = min(self.stats.grad_norm_min, total_norm)

        should_clip = False
        if self.clip_method in (ClipMethod.NORM, ClipMethod.BOTH) and self.clip_norm > 0:
            if total_norm > self.clip_norm:
                clip_coef = self.clip_norm / (total_norm + 1e-6)
                for p in parameters:
                    p.grad.data.mul_(clip_coef)
                should_clip = True
                logger.debug(f"梯度按范数裁剪: {total_norm:.4f} -> {self.clip_norm:.4f}")

        if self.clip_method in (ClipMethod.VALUE, ClipMethod.BOTH) and self.clip_value > 0:
            for p in parameters:
                clipped = p.grad.data.clamp_(-self.clip_value, self.clip_value)
                if not should_clip:
                    max_abs = p.grad.data.abs().max().item()
                    if max_abs > self.clip_value:
                        should_clip = True

        if should_clip:
            self.stats.clipped_count += 1

        return total_norm

    def _apply_gradient_noise(self):
        with torch.no_grad():
            for p in self.model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * self.noise_std
                    p.grad.data.add_(noise)

    def _apply_moe_grad_scaling(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if "expert" in name.lower() or "moe" in name.lower():
                param.grad.data.mul_(self.moe_grad_scale)
            elif "gate" in name.lower() or "router" in name.lower():
                param.grad.data.mul_(self.moe_grad_scale * 1.5)

        moe_grads = []
        for name in self.moe_param_names:
            for n, p in self.model.named_parameters():
                if n == name and p.grad is not None:
                    moe_grads.append(p.grad.data.abs().mean().item())
        if moe_grads:
            sparsity = sum(1 for g in moe_grads if g < 1e-6) / max(1, len(moe_grads))
            self.stats.moe_grad_sparsity = 0.9 * self.stats.moe_grad_sparsity + 0.1 * sparsity

    def _apply_delay_compensation(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name in self.delay_buffer:
                delayed_grad = self.delay_buffer[name]
                compensation = delayed_grad * self.delay_alpha
                param.grad.data.add_(compensation)
            self.delay_buffer[name] = param.grad.data.clone()

    def _compress_gradients(self) -> Dict[str, Any]:
        total_elements = 0
        total_compressed = 0

        for p in self.model.parameters():
            if p.grad is None:
                continue
            total_elements += p.grad.numel()
            original_grad = p.grad.data.clone()

            if self.compression_method == CompressionMethod.TOPK:
                compressed = self._topk_compress(p.grad.data, self.compression_ratio)
            elif self.compression_method == CompressionMethod.RANDOM:
                compressed = self._random_compress(p.grad.data, self.compression_ratio)
            elif self.compression_method == CompressionMethod.QUANTIZATION:
                compressed = self._quantize_compress(p.grad.data, self.quantization_bits)
            else:
                compressed = p.grad.data

            non_zero = (compressed != 0).sum().item()
            total_compressed += non_zero
            p.grad.data.copy_(compressed)

        ratio = 1.0 - (total_compressed / max(1, total_elements))
        self.stats.compression_ratio = 0.9 * self.stats.compression_ratio + 0.1 * ratio
        self.stats.compression_savings += total_elements - total_compressed

        return {
            "ratio": ratio,
            "savings": total_elements - total_compressed,
            "total_elements": total_elements,
            "compressed_elements": total_compressed,
        }

    def _topk_compress(self, grad: torch.Tensor, ratio: float) -> torch.Tensor:
        if ratio >= 1.0:
            return grad
        original_shape = grad.shape
        flat_grad = grad.flatten()
        k = max(1, int(flat_grad.numel() * ratio))
        values, indices = torch.topk(flat_grad.abs(), k, sorted=False)
        compressed = torch.zeros_like(flat_grad)
        compressed[indices] = flat_grad[indices]
        return compressed.view(original_shape)

    def _random_compress(self, grad: torch.Tensor, ratio: float) -> torch.Tensor:
        if ratio >= 1.0:
            return grad
        mask = torch.rand_like(grad) < ratio
        compressed = grad * mask
        compressed = compressed * (1.0 / max(ratio, 1e-6))
        return compressed

    def _quantize_compress(self, grad: torch.Tensor, bits: int) -> torch.Tensor:
        if bits >= 32:
            return grad
        grad_norm = grad.norm()
        if grad_norm < 1e-8:
            return grad
        normalized = grad / grad_norm
        levels = 2 ** (bits - 1)
        quantized = torch.round(normalized * levels) / levels
        return quantized * grad_norm

    def reset_accumulation(self):
        self.accumulator.step_count = 0
        self.accumulator.loss_sum = 0.0
        self.accumulator.buffer.clear()

    def reset_delay_buffer(self):
        self.delay_buffer.clear()
        self.stats.delay_steps = 0
        logger.info("梯度延迟缓冲区已重置")

    def set_compression(self, method: CompressionMethod, ratio: float = 0.01):
        self.compression_method = method
        self.compression_ratio = ratio
        logger.info(f"梯度压缩方法已设置为: {method.value}, 压缩率: {ratio}")

    def set_noise(self, noise_std: float, noise_decay: float = 0.999):
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        logger.info(f"梯度噪声注入已设置: std={noise_std}, decay={noise_decay}")

    def set_moe_grad_scale(self, scale: float):
        self.moe_grad_scale = scale
        logger.info(f"MoE梯度缩放系数已设置为: {scale}")

    def get_gradient_stats(self) -> GradientStats:
        return self.stats

    def compute_gradient_sparsity(self) -> float:
        total_zeros = 0
        total_elements = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_zeros += (p.grad.data.abs() < 1e-8).sum().item()
                total_elements += p.grad.numel()
        if total_elements == 0:
            return 0.0
        return total_zeros / total_elements

    def compute_moe_gradient_sparsity(self) -> float:
        moe_zeros = 0
        moe_elements = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                is_moe = any(k in name.lower() for k in ["expert", "moe", "gate", "router"])
                if is_moe:
                    moe_zeros += (param.grad.data.abs() < 1e-8).sum().item()
                    moe_elements += param.grad.numel()
        if moe_elements == 0:
            return 0.0
        return moe_zeros / moe_elements

    def get_gradient_norm_per_layer(self) -> Dict[str, float]:
        norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.data.norm(2).item()
        return norms

    def analyze_gradient_distribution(self, bins: int = 10) -> Dict[str, Any]:
        all_grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                all_grads.append(p.grad.data.abs().flatten())
        if not all_grads:
            return {}
        all_grads = torch.cat(all_grads)
        mean_val = all_grads.mean().item()
        std_val = all_grads.std().item()
        max_val = all_grads.max().item()
        min_val = all_grads.min().item()
        median_val = all_grads.median().item()
        percentile_90 = torch.quantile(all_grads, 0.9).item()
        percentile_99 = torch.quantile(all_grads, 0.99).item()
        return {
            "mean": mean_val,
            "std": std_val,
            "max": max_val,
            "min": min_val,
            "median": median_val,
            "p90": percentile_90,
            "p99": percentile_99,
            "total_elements": all_grads.numel(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "accumulation_steps": self.accumulation_steps,
            "noise_std": self.noise_std,
            "noise_decay": self.noise_decay,
            "compression_method": self.compression_method.value,
            "compression_ratio": self.compression_ratio,
            "clip_norm": self.clip_norm,
            "clip_value": self.clip_value,
            "stats": {
                "total_steps": self.stats.total_steps,
                "grad_norm_mean": self.stats.grad_norm_mean,
                "clipped_count": self.stats.clipped_count,
                "compression_ratio": self.stats.compression_ratio,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.accumulation_steps = state_dict.get("accumulation_steps", self.accumulation_steps)
        self.noise_std = state_dict.get("noise_std", self.noise_std)
        self.noise_decay = state_dict.get("noise_decay", self.noise_decay)
        self.compression_ratio = state_dict.get("compression_ratio", self.compression_ratio)
        self.clip_norm = state_dict.get("clip_norm", self.clip_norm)
        self.clip_value = state_dict.get("clip_value", self.clip_value)
        method = state_dict.get("compression_method", "none")
        self.compression_method = CompressionMethod(method)

        stats_data = state_dict.get("stats", {})
        if stats_data:
            self.stats.total_steps = stats_data.get("total_steps", 0)
            self.stats.grad_norm_mean = stats_data.get("grad_norm_mean", 0.0)
            self.stats.clipped_count = stats_data.get("clipped_count", 0)
            self.stats.compression_ratio = stats_data.get("compression_ratio", 0.0)

        logger.info("梯度优化器状态已恢复")

    def print_gradient_summary(self):
        dist_info = self.analyze_gradient_distribution()
        sparsity = self.compute_gradient_sparsity()
        moe_sparsity = self.compute_moe_gradient_sparsity()

        summary = [
            "=" * 60,
            "青小团梯度优化 - 梯度信息摘要",
            "=" * 60,
            f"梯度范数均值: {self.stats.grad_norm_mean:.4f}",
            f"梯度范数最大: {self.stats.grad_norm_max:.4f}",
            f"梯度范数最小: {self.stats.grad_norm_min:.4f}",
            f"全局梯度稀疏度: {sparsity:.4f}",
            f"MoE梯度稀疏度: {moe_sparsity:.4f}",
            f"裁剪次数: {self.stats.clipped_count}",
            f"累积步数: {self.accumulator.step_count}/{self.accumulation_steps}",
            f"压缩方法: {self.compression_method.value}",
            f"噪声标准差: {self.noise_std:.6f}",
            f"---",
            f"梯度均值: {dist_info.get('mean', 0):.6f}",
            f"梯度中位数: {dist_info.get('median', 0):.6f}",
            f"梯度P90: {dist_info.get('p90', 0):.6f}",
            f"梯度P99: {dist_info.get('p99', 0):.6f}",
            f"梯度元素总数: {dist_info.get('total_elements', 0)}",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
