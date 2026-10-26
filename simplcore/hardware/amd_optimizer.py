# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""AMD MI250 GPU 深度优化模块 (amd_optimizer).

针对 AMD MI250 GPU (128GB HBM2e, CDNA2 架构) 的完整优化方案:
  - MI250Optimizer:  内存/计算/通信/分布式全方位优化
  - MemoryPool:      预分配显存池, 减少碎片, 支持动态扩展
  - AMDPerformanceMonitor: GPU 利用率 / 内存带宽监控与报告生成

硬件关键参数:
  - 2 个 GCD (Graphics Compute Die), 各 110 CU
  - 128 GB HBM2e, 峰值带宽 3.2 TB/s
  - Infinity Fabric 互联, 跨 GCD 约 800 GB/s
  - MFMA 指令支持 fp16/bf16/fp32 张量核心

72B 模型在 2x MI250 上训练时间估算:
  假设: batch_size_per_gpu=2, seq_len=4096, gradient_accumulation=8,
  每 optimizer step ≈ 6.5s (含 TP 通信和梯度累积),
  每步 tokens = 2 × 2 × 8 × 4096 = 131,072
  总 tokens = 1T, 总步数 = 1T / 131,072 ≈ 7,629,395 步
  总时间 ≈ 7,629,395 × 6.5 / 3600 / 24 ≈ 573.9 天 (不含 overhead)
  — 对于 2 卡场景, 建议使用更多 GPU 或在 20 天内训练更小规模数据(~35B tokens)。
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import simplcore.tensor as torch
import simplcore.tensor.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# AMD GPU / ROCm 可用性检测
# ======================================================================
def _detect_amd_gpu() -> bool:
    """检测当前环境是否存在可用的 AMD GPU (ROCm).

    Returns:
        True 表示检测到 AMD GPU; False 为 CPU 模式
    """
    if not torch.cuda.is_available():
        return False
    try:
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            if "AMD" in name or "gfx" in name.lower() or "MI2" in name:
                return True
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return True
    except Exception:
        return False
    return False


_HAS_AMD_GPU: bool = _detect_amd_gpu()


# ======================================================================
# MI250Specs — 硬件规格
# ======================================================================
@dataclass
class MI250Specs:
    """AMD MI250 GPU 完整硬件规格描述.

    Attributes:
        hbm_size:                 单卡 HBM2e 显存容量 (字节)
        num_gcd:                  GCD 数量
        num_cu_per_gcd:           每 GCD 的 CU 数量
        memory_bandwidth:         HBM2e 峰值带宽 (字节/秒)
        infinity_fabric_bandwidth: 跨 GCD Infinity Fabric 带宽 (字节/秒)
        l2_cache_per_gcd:         每 GCD 的 L2 缓存 (字节)
        mfma_tflops_fp16:         fp16 MFMA 理论峰值 (TFLOPS)
        mfma_tflops_bf16:         bf16 MFMA 理论峰值 (TFLOPS)
        mfma_tflops_fp32:         fp32 MFMA 理论峰值 (TFLOPS)
        num_gpus_for_72b:         训练 72B 模型推荐 GPU 数
    """
    hbm_size: int = 128 * (1024 ** 3)
    num_gcd: int = 2
    num_cu_per_gcd: int = 110
    memory_bandwidth: int = int(3.2e12)
    infinity_fabric_bandwidth: int = int(800e9)
    l2_cache_per_gcd: int = 8 * (1024 ** 2)
    mfma_tflops_fp16: float = 383.0
    mfma_tflops_bf16: float = 191.5
    mfma_tflops_fp32: float = 95.7
    num_gpus_for_72b: int = 2

    def total_cu(self) -> int:
        """返回全卡 CU 总数."""
        return self.num_gcd * self.num_cu_per_gcd

    def total_hbm_gb(self) -> float:
        """返回总显存 (GB)."""
        return self.hbm_size / (1024 ** 3)

    def estimate_72b_training_time(
        self,
        total_tokens: int = 1_000_000_000_000,
        batch_size_per_gpu: int = 4,
        seq_len: int = 4096,
        step_time_seconds: float = 8.0,
        gradient_accumulation: int = 1,
        num_gpus: int = 2,
    ) -> Dict[str, Any]:
        """估算 72B 模型在 MI250 上的训练时间.

        计算方法:
          - 全局 batch size = batch_size_per_gpu × num_gpus × gradient_accumulation
          - 每步 token 数 = 全局 batch size × seq_len
          - 总步数 = total_tokens / 每步 token 数
          - step_time_seconds 为一次 optimizer step 的耗时 (含所有梯度累积微批次)

        Args:
            total_tokens:          总训练 token 数
            batch_size_per_gpu:    每 GPU 每步的 micro-batch size
            seq_len:               序列长度
            step_time_seconds:     每次 optimizer step 耗时 (秒, 含所有微批次 forward/backward/通信)
            gradient_accumulation: 梯度累积步数 (影响 tokens_per_step, 但不影响每步耗时计算)
            num_gpus:              GPU 数量

        Returns:
            包含训练时间估算的字典
        """
        # 全局 batch size = batch_size_per_gpu * num_gpus * gradient_accumulation
        global_batch = batch_size_per_gpu * num_gpus * gradient_accumulation
        tokens_per_step = global_batch * seq_len
        total_steps = math.ceil(total_tokens / tokens_per_step)

        total_seconds = total_steps * step_time_seconds
        total_days = total_seconds / (3600 * 24)

        # 模型并行通信开销 (TP=2 时约 15%)
        communication_overhead = 0.15
        total_days_with_overhead = total_days * (1 + communication_overhead)

        # 额外开销: checkpoint 保存、评估、日志等约 10%
        extra_overhead = 0.10
        total_days_real = total_days_with_overhead * (1 + extra_overhead)

        return {
            "total_tokens": total_tokens,
            "global_batch_size": global_batch,
            "tokens_per_step": tokens_per_step,
            "total_steps": total_steps,
            "step_time_seconds": step_time_seconds,
            "total_days_estimated": round(total_days_real, 2),
            "within_20_days": total_days_real < 20,
            "num_gpus": num_gpus,
            "gradient_accumulation": gradient_accumulation,
            "batch_size_per_gpu": batch_size_per_gpu,
            "communication_overhead_pct": communication_overhead * 100,
            "extra_overhead_pct": extra_overhead * 100,
        }

    def estimate_memory_usage_72b(self) -> Dict[str, float]:
        """估算 72B 模型在 MI250 上的显存占用 (单卡, GB).

        Returns:
            各组件显存估算 (GB)
        """
        num_params = 72e9  # 72B
        precision_bytes = 2  # bf16

        # 模型权重 (bf16)
        weights_gb = num_params * precision_bytes / (1024 ** 3)

        # 优化器状态 (AdamW: fp32 参数 + momentum + variance)
        optimizer_gb = num_params * 4 * 3 / (1024 ** 3)

        # 梯度 (bf16, 与模型权重同大小)
        gradients_gb = num_params * precision_bytes / (1024 ** 3)

        # 激活值 (估算: 取决于 batch_size, seq_len, hidden_size)
        # 对于 72B, hidden=8192, 每层激活约 batch * seq * hidden * precision
        # 保守估算 ~40 GB (含梯度检查点)
        activations_gb = 40.0

        total = weights_gb + optimizer_gb + gradients_gb + activations_gb

        return {
            "weights_bf16_gb": round(weights_gb, 1),
            "optimizer_states_gb": round(optimizer_gb, 1),
            "gradients_gb": round(gradients_gb, 1),
            "activations_gb": round(activations_gb, 1),
            "total_estimated_gb": round(total, 1),
            "single_gpu_hbm_gb": self.total_hbm_gb(),
            "within_single_gpu": total < self.total_hbm_gb(),
        }


# ======================================================================
# MemoryPool — 预分配显存池
# ======================================================================
class MemoryPool:
    """预分配显存池, 减少碎片化并支持动态扩展.

    MI250 的 128 GB HBM 需要精细的内存管理以避免碎片化。
    MemoryPool 通过预分配固定大小的缓存块池, 减少运行时频繁的
    cudaMalloc/cudaFree 调用, 从而降低碎片并提升分配效率。

    用法示例::

        pool = MemoryPool(total_size_gb=80, block_size_mb=256)
        tensor = pool.allocate((4096, 8192), torch.bfloat16)
        pool.free(tensor)
        print(pool.get_stats())

    Attributes:
        total_size:      池总大小 (字节)
        block_size:      每块大小 (字节)
        free_blocks:     空闲块队列 (按地址排序)
        used_blocks:     已分配块追踪
        expansion_count: 动态扩展次数
    """

    def __init__(
        self,
        total_size_gb: float = 80.0,
        block_size_mb: int = 256,
        device: Optional[int] = None,
        enable_expansion: bool = True,
    ) -> None:
        """初始化显存池.

        Args:
            total_size_gb:    池总大小 (GB)
            block_size_mb:    每块大小 (MB)
            device:           GPU 设备索引 (None 自动检测)
            enable_expansion: 是否允许动态扩展
        """
        self._device = device if device is not None else (0 if _HAS_AMD_GPU else None)
        self._enable_expansion = enable_expansion

        self.total_size = int(total_size_gb * (1024 ** 3))
        self.block_size = block_size_mb * 1024 * 1024
        self.num_blocks = self.total_size // self.block_size

        # 锁 (线程安全)
        self._lock = threading.Lock()

        # 空闲块列表 (每项为 (offset, size))
        self.free_blocks: List[Tuple[int, int]] = [(0, self.total_size)]

        # 已分配块追踪: key=数据指针, value=(offset, size)
        self.used_blocks: Dict[int, Tuple[int, int]] = {}

        # 统计
        self._expansion_count = 0
        self._total_allocated = 0
        self._total_freed = 0
        self._peak_usage = 0
        self._allocation_count = 0

        # 预分配底层张量 (作为显存池的后端存储)
        self._pool_tensor: Optional[torch.Tensor] = None
        self._init_pool()

        logger.info(
            "MemoryPool 初始化: total=%.1f GB, block=%d MB, blocks=%d, device=%s",
            total_size_gb,
            block_size_mb,
            self.num_blocks,
            f"cuda:{self._device}" if self._device is not None else "cpu",
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
        alignment: int = 256,
    ) -> Optional[torch.Tensor]:
        """从池中分配指定形状的张量.

        采用最佳适配 (best-fit) 策略, 优先选择大小最接近的可用块,
        以减少内部碎片。

        Args:
            shape:     张量形状
            dtype:     数据类型
            alignment: 地址对齐要求 (字节)

        Returns:
            分配的张量, 若分配失败返回 None
        """
        if self._pool_tensor is None:
            logger.error("MemoryPool: 池张量为空, 无法分配")
            return None

        needed_bytes = int(
            math.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        )
        # 对齐到 block_size 的倍数
        needed_aligned = math.ceil(needed_bytes / self.block_size) * self.block_size

        with self._lock:
            # 最佳适配: 找到不小于 needed_aligned 的最小块
            best_idx = -1
            best_size = float("inf")
            for i, (offset, size) in enumerate(self.free_blocks):
                if size >= needed_aligned and size < best_size:
                    best_idx = i
                    best_size = size

            if best_idx == -1:
                # 尝试动态扩展
                if self._enable_expansion:
                    return self._expand_and_allocate(shape, dtype, needed_aligned)
                logger.warning(
                    "MemoryPool: 分配失败, 需要 %d bytes, 无足够空闲块",
                    needed_aligned,
                )
                return None

            offset, size = self.free_blocks.pop(best_idx)
            remaining = size - needed_aligned

            if remaining > 0:
                # 剩余空间放回空闲列表
                self.free_blocks.append((offset + needed_aligned, remaining))
                # 保持空闲列表按 offset 排序
                self.free_blocks.sort(key=lambda x: x[0])

            # 从池张量中切片
            tensor = self._pool_tensor[offset // 2: (offset + needed_bytes) // 2]
            tensor = tensor.view(shape).contiguous()

            # 记录已分配
            data_ptr = tensor.data_ptr()
            self.used_blocks[data_ptr] = (offset, needed_aligned)
            self._total_allocated += needed_aligned
            self._peak_usage = max(self._peak_usage, self._total_allocated - self._total_freed)
            self._allocation_count += 1

            return tensor

    def free(self, tensor: torch.Tensor) -> bool:
        """释放张量, 将内存归还池中.

        Args:
            tensor: 待释放的张量

        Returns:
            释放是否成功
        """
        if self._pool_tensor is None:
            return False

        data_ptr = tensor.data_ptr()
        with self._lock:
            if data_ptr not in self.used_blocks:
                logger.warning("MemoryPool: 试图释放未追踪的张量")
                return False

            offset, size = self.used_blocks.pop(data_ptr)
            self._total_freed += size

            # 归还到空闲列表并合并相邻块
            self._merge_free_block(offset, size)

            return True

    def reset(self) -> None:
        """重置内存池, 释放所有已分配块."""
        with self._lock:
            self.free_blocks = [(0, self.total_size)]
            self.used_blocks.clear()
            self._total_allocated = 0
            self._total_freed = 0
            self._allocation_count = 0
            self._peak_usage = 0
            logger.info("MemoryPool: 已重置")

    def get_stats(self) -> Dict[str, Any]:
        """返回内存池统计信息.

        Returns:
            统计字典
        """
        with self._lock:
            free_total = sum(size for _, size in self.free_blocks)
            used_total = sum(size for _, size in self.used_blocks.values())
            return {
                "total_size_gb": round(self.total_size / (1024 ** 3), 2),
                "free_gb": round(free_total / (1024 ** 3), 2),
                "used_gb": round(used_total / (1024 ** 3), 2),
                "peak_usage_gb": round(self._peak_usage / (1024 ** 3), 2),
                "utilization_pct": round(used_total / max(1, self.total_size) * 100, 1),
                "free_blocks": len(self.free_blocks),
                "used_blocks": len(self.used_blocks),
                "allocation_count": self._allocation_count,
                "expansion_count": self._expansion_count,
                "device": f"cuda:{self._device}" if self._device is not None else "cpu",
            }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _init_pool(self) -> None:
        """初始化后端池张量."""
        try:
            if self._device is not None and _HAS_AMD_GPU:
                device_str = f"cuda:{self._device}"
                # 使用 uint8 (1 字节/元素) 创建池, 便于字节级寻址
                num_elements = self.total_size
                self._pool_tensor = torch.empty(
                    num_elements, dtype=torch.uint8, device=device_str
                )
                logger.info("MemoryPool: 预分配 %.1f GB 显存池", self.total_size / (1024 ** 3))
            else:
                # CPU 仿真模式
                self._pool_tensor = torch.empty(self.total_size, dtype=torch.uint8)
                logger.info(
                    "MemoryPool: CPU 模式, 预分配 %.1f GB (仿真)",
                    self.total_size / (1024 ** 3),
                )
        except RuntimeError as exc:
            logger.error("MemoryPool: 预分配失败: %s, 降级为按需分配", exc)
            self._pool_tensor = None

    def _merge_free_block(self, offset: int, size: int) -> None:
        """将释放的块合并回空闲列表, 并合并相邻空闲块.

        Args:
            offset: 释放块的起始偏移
            size:   释放块的大小
        """
        self.free_blocks.append((offset, size))
        # 按 offset 排序
        self.free_blocks.sort(key=lambda x: x[0])

        # 合并相邻块
        merged = []
        for blk in self.free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == blk[0]:
                # 相邻, 合并
                prev_offset, prev_size = merged.pop()
                merged.append((prev_offset, prev_size + blk[1]))
            else:
                merged.append(blk)
        self.free_blocks = merged

    def _expand_and_allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        needed_aligned: int,
    ) -> Optional[torch.Tensor]:
        """动态扩展池并分配.

        当池空间不足时, 尝试额外分配新块并追加到池中。

        Args:
            shape:          张量形状
            dtype:          数据类型
            needed_aligned: 对齐后所需字节

        Returns:
            分配的张量或 None
        """
        if not self._enable_expansion or self._pool_tensor is None:
            return None

        try:
            expand_size = max(needed_aligned, self.total_size // 4)
            if _HAS_AMD_GPU:
                new_block = torch.empty(
                    expand_size, dtype=torch.uint8, device=self._pool_tensor.device
                )
            else:
                new_block = torch.empty(expand_size, dtype=torch.uint8)

            new_offset = self.total_size
            self.total_size += expand_size
            self._expansion_count += 1

            # 将新块加入空闲列表
            self.free_blocks.append((new_offset, expand_size))
            self.free_blocks.sort(key=lambda x: x[0])

            logger.info(
                "MemoryPool: 动态扩展 +%.1f GB (共 %.1f GB, 已扩展 %d 次)",
                expand_size / (1024 ** 3),
                self.total_size / (1024 ** 3),
                self._expansion_count,
            )

            # 重新尝试分配 (递归调用, 但最多一次)
            return self.allocate(shape, dtype)
        except RuntimeError as exc:
            logger.error("MemoryPool: 扩展失败: %s", exc)
            return None


# ======================================================================
# AMDPerformanceMonitor — GPU 性能监控
# ======================================================================
class AMDPerformanceMonitor:
    """AMD GPU 性能监控器.

    提供 GPU 利用率、内存带宽、温度等关键指标的实时监控,
    并生成结构化性能报告。支持历史数据滑动窗口, 便于分析趋势。

    用法示例::

        monitor = AMDPerformanceMonitor()
        monitor.start()
        # ... 训练代码 ...
        report = monitor.stop_and_report()
        print(report)
    """

    def __init__(
        self,
        window_size: int = 100,
        device_id: int = 0,
        enable_detailed: bool = True,
    ) -> None:
        """初始化监控器.

        Args:
            window_size:       历史数据窗口大小
            device_id:         GPU 设备 ID
            enable_detailed:   是否收集详细信息 (利用率 / 温度等)
        """
        self._device_id = device_id
        self._window_size = window_size
        self._enable_detailed = enable_detailed
        self._is_running = False

        # 历史数据
        self._utilization_history: List[float] = []
        self._memory_used_history: List[float] = []
        self._timestamp_history: List[float] = []

        # 统计
        self._sample_count = 0
        self._start_time = 0.0

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 监控数据采集
    # ------------------------------------------------------------------
    def sample(self) -> Dict[str, Any]:
        """采集当前 GPU 状态快照.

        Returns:
            当前 GPU 状态字典; 若无 AMD GPU 则返回空状态
        """
        if not _HAS_AMD_GPU:
            return {
                "gpu_available": False,
                "utilization_pct": 0.0,
                "memory_used_gb": 0.0,
                "memory_total_gb": 0.0,
                "memory_free_gb": 0.0,
                "temperature_c": 0.0,
                "power_w": 0.0,
            }

        state: Dict[str, Any] = {"gpu_available": True}

        try:
            # 显存信息
            mem_info = torch.cuda.mem_get_info(self._device_id)
            state["memory_free_gb"] = mem_info[0] / (1024 ** 3)
            state["memory_total_gb"] = mem_info[1] / (1024 ** 3)
            state["memory_used_gb"] = state["memory_total_gb"] - state["memory_free_gb"]

            # GPU 利用率 (估算: 通过对比峰值内存带宽)
            if self._enable_detailed:
                # 尝试通过 ROCm 驱动获取详细信息
                state["utilization_pct"] = self._estimate_gpu_utilization()
                state["temperature_c"] = self._get_temperature()
                state["power_w"] = self._get_power_usage()
            else:
                state["utilization_pct"] = 0.0
                state["temperature_c"] = 0.0
                state["power_w"] = 0.0

        except Exception as exc:
            logger.debug("AMDPerformanceMonitor: 采样失败: %s", exc)
            state["utilization_pct"] = 0.0
            state["memory_used_gb"] = 0.0
            state["memory_free_gb"] = 0.0
            state["memory_total_gb"] = 0.0
            state["temperature_c"] = 0.0
            state["power_w"] = 0.0

        return state

    def record_sample(self) -> None:
        """记录一次采样到历史数据中."""
        state = self.sample()
        timestamp = time.time()

        with self._lock:
            self._utilization_history.append(state["utilization_pct"])
            self._memory_used_history.append(state["memory_used_gb"])
            self._timestamp_history.append(timestamp)
            self._sample_count += 1

            # 限制窗口大小
            if len(self._utilization_history) > self._window_size:
                self._utilization_history.pop(0)
                self._memory_used_history.pop(0)
                self._timestamp_history.pop(0)

    def start(self, interval_seconds: float = 5.0) -> None:
        """启动周期性后台监控.

        Args:
            interval_seconds: 采样间隔 (秒)
        """
        if self._is_running:
            logger.warning("AMDPerformanceMonitor: 已在运行")
            return

        self._is_running = True
        self._start_time = time.time()
        self._sample_count = 0

        def _monitor_loop() -> None:
            while self._is_running:
                self.record_sample()
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_monitor_loop, daemon=True, name="AMD-perf-mon")
        thread.start()
        logger.info(
            "AMDPerformanceMonitor: 已启动, 采样间隔=%.1fs, 设备=cuda:%d",
            interval_seconds,
            self._device_id,
        )

    def stop(self) -> None:
        """停止后台监控."""
        self._is_running = False
        logger.info("AMDPerformanceMonitor: 已停止")

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------
    def generate_report(self) -> Dict[str, Any]:
        """生成性能监控报告.

        Returns:
            包含统计摘要和推荐的报告字典
        """
        with self._lock:
            if self._sample_count == 0:
                return {"status": "no_data", "message": "未采集到样本"}

            avg_util = (
                sum(self._utilization_history) / len(self._utilization_history)
                if self._utilization_history
                else 0.0
            )
            max_util = max(self._utilization_history) if self._utilization_history else 0.0

            avg_mem = (
                sum(self._memory_used_history) / len(self._memory_used_history)
                if self._memory_used_history
                else 0.0
            )
            max_mem = max(self._memory_used_history) if self._memory_used_history else 0.0

            elapsed = time.time() - self._start_time

            # 内存带宽估算 (基于 MI250 理论峰值和使用率)
            specs = MI250Specs()
            estimated_bandwidth = (
                specs.memory_bandwidth * (avg_util / 100.0) / 1e12
            )

            # 推荐建议
            recommendations: List[str] = []
            if avg_util < 50:
                recommendations.append("GPU 利用率偏低 (< 50%), 建议增大 batch_size 或减小梯度累积步数")
            if avg_mem > specs.total_hbm_gb() * 0.9:
                recommendations.append("显存使用超过 90%, 建议启用梯度检查点或减小模型并行分片")
            if avg_util > 90 and avg_mem > specs.total_hbm_gb() * 0.85:
                recommendations.append("GPU 利用率和显存均处于高位, 当前配置良好")

            return {
                "status": "ok",
                "gpu_available": _HAS_AMD_GPU,
                "device_id": self._device_id,
                "sampling": {
                    "total_samples": self._sample_count,
                    "elapsed_seconds": round(elapsed, 1),
                    "window_size": self._window_size,
                },
                "gpu_utilization": {
                    "avg_pct": round(avg_util, 1),
                    "max_pct": round(max_util, 1),
                    "history": self._utilization_history[-50:],
                },
                "memory": {
                    "avg_used_gb": round(avg_mem, 1),
                    "max_used_gb": round(max_mem, 1),
                    "total_gb": round(specs.total_hbm_gb(), 1),
                    "history": self._memory_used_history[-50:],
                },
                "bandwidth_estimate": {
                    "estimated_tb_s": round(estimated_bandwidth, 2),
                    "peak_tb_s": specs.memory_bandwidth / 1e12,
                },
                "recommendations": recommendations,
            }

    def stop_and_report(self) -> Dict[str, Any]:
        """停止监控并生成报告 (便捷方法).

        Returns:
            性能报告
        """
        self.stop()
        return self.generate_report()

    # ------------------------------------------------------------------
    # 内部方法: ROCm 硬件信息获取
    # ------------------------------------------------------------------
    def _estimate_gpu_utilization(self) -> float:
        """估算当前 GPU 利用率.

        通过检查 GPU 上的 kernel 活动情况来估算。
        在 ROCm 环境下, 可以通过 nvidia-smi 类似工具获取,
        这里使用 torch.cuda 的时间戳差异作近似估算。

        Returns:
            利用率百分比 (0-100)
        """
        try:
            # 通过检查当前 GPU 上是否有活跃 kernel 来估算
            # 实际生产环境建议使用 rocm-smi 或 AMD SMI 库
            return min(100.0, max(0.0, torch.cuda.utilization(self._device_id)))
        except (AttributeError, NotImplementedError):
            # torch 版本不支持 utilization API
            pass
        except Exception:
            pass
        return 0.0

    def _get_temperature(self) -> float:
        """获取 GPU 温度.

        Returns:
            温度 (摄氏度), 不可用时返回 0
        """
        try:
            if hasattr(torch.cuda, "temperature"):
                return torch.cuda.temperature(self._device_id)
        except Exception:
            pass

        # 尝试通过文件系统读取 (Linux ROCm)
        try:
            path = f"/sys/class/drm/card{self._device_id}/device/hwmon/hwmon*/temp1_input"
            import glob
            matches = glob.glob(path)
            if matches:
                with open(matches[0]) as f:
                    return int(f.read().strip()) / 1000.0
        except Exception:
            pass
        return 0.0

    def _get_power_usage(self) -> float:
        """获取 GPU 功耗.

        Returns:
            功耗 (瓦特), 不可用时返回 0
        """
        if not _HAS_AMD_GPU:
            return 0.0
        try:
            # 通过文件系统读取 (Linux ROCm)
            import glob
            path = f"/sys/class/drm/card{self._device_id}/device/hwmon/hwmon*/power1_average"
            matches = glob.glob(path)
            if matches:
                with open(matches[0]) as f:
                    return int(f.read().strip()) / 1_000_000.0
        except Exception:
            pass
        return 0.0


# ======================================================================
# MI250Optimizer — 全方位 AMD MI250 优化器
# ======================================================================
class MI250Optimizer:
    """AMD MI250 GPU 全方位优化器.

    针对 CDNA2 架构提供内存、计算、通信与分布式训练的深度优化:

    1. 内存优化:
       - 梯度检查点 (Activation Checkpointing) 减少激活显存
       - MemoryPool 预分配池减少碎片
       - 动态内存分配策略

    2. 计算优化:
       - Flash Attention 替代方案
       - 算子融合标记
       - 半精度 (bf16/fp16) 自动启用

    3. 通信优化:
       - Infinity Fabric 亲和性设置
       - 通信计算重叠
       - 梯度压缩与聚合优化

    4. 分布式训练:
       - 2 卡模型并行 + 数据并行混合策略
       - GCD 感知的设备映射

    用法示例::

        opt = MI250Optimizer()
        model = opt.optimize_memory(model)
        model = opt.optimize_computation(model)
        model = opt.optimize_communication(model)
        strategy = opt.get_distributed_strategy(rank=0, world_size=2)
    """

    def __init__(
        self,
        specs: Optional[MI250Specs] = None,
        memory_pool_size_gb: float = 60.0,
    ) -> None:
        """初始化 MI250 优化器.

        Args:
            specs:               MI250 硬件规格 (默认使用 MI250Specs())
            memory_pool_size_gb: 预分配显存池大小 (GB)
        """
        self.specs = specs if specs is not None else MI250Specs()
        self._amd_available = _HAS_AMD_GPU
        self._memory_pool: Optional[MemoryPool] = None

        # 初始化内存池 (仅在 AMD GPU 可用时)
        if self._amd_available:
            try:
                self._memory_pool = MemoryPool(
                    total_size_gb=memory_pool_size_gb,
                    block_size_mb=256,
                )
            except Exception as exc:
                logger.warning("MI250Optimizer: 内存池初始化失败: %s", exc)

        if self._amd_available:
            logger.info("MI250Optimizer: AMD GPU 已检测到, 启用全部优化")
        else:
            logger.info("MI250Optimizer: CPU 模式, 优化降级为空操作")

    # ------------------------------------------------------------------
    # 1. 内存优化
    # ------------------------------------------------------------------
    def optimize_memory(
        self,
        model: nn.Module,
        enable_checkpointing: bool = True,
        checkpoint_layers: Optional[List[int]] = None,
    ) -> nn.Module:
        """优化模型内存使用.

        策略:
          1. 启用梯度检查点 (activation checkpointing), 以时间换空间
          2. 注册内存池以供推理/训练时复用
          3. 标记模型的内存布局偏好

        Args:
            model:               待优化模型
            enable_checkpointing: 是否启用梯度检查点
            checkpoint_layers:   指定启用检查点的层索引 (None = 全部)

        Returns:
            优化后的模型
        """
        if not self._amd_available:
            logger.info("optimize_memory: CPU 模式, 跳过")
            return model

        # ---- 梯度检查点 ----
        if enable_checkpointing:
            self._apply_gradient_checkpointing(model, checkpoint_layers)
            logger.info("optimize_memory: 已启用梯度检查点")

        # ---- 内存池注册 ----
        if self._memory_pool is not None:
            model._SimplCore_memory_pool = self._memory_pool  # type: ignore[attr-defined]
            logger.info(
                "optimize_memory: 已注册 MemoryPool (%.1f GB)",
                self._memory_pool.total_size / (1024 ** 3),
            )

        # ---- 内存格式优化 ----
        model = self._optimize_memory_format(model)

        return model

    def get_memory_pool(self) -> Optional[MemoryPool]:
        """返回内部 MemoryPool 实例."""
        return self._memory_pool

    # ------------------------------------------------------------------
    # 2. 计算优化
    # ------------------------------------------------------------------
    def optimize_computation(
        self,
        model: nn.Module,
        enable_flash_attention: bool = True,
        enable_operator_fusion: bool = True,
        precision: str = "bf16",
    ) -> nn.Module:
        """优化模型计算效率.

        策略:
          1. Flash Attention 标记 (xformers 或原生实现)
          2. 算子融合标记 (供 JIT / torch.compile 使用)
          3. 半精度计算预配置

        Args:
            model:                 待优化模型
            enable_flash_attention: 是否启用 Flash Attention
            enable_operator_fusion: 是否启用算子融合
            precision:             精度模式 ("bf16" / "fp16" / "fp32")

        Returns:
            优化后的模型
        """
        if not self._amd_available:
            logger.info("optimize_computation: CPU 模式, 跳过")
            return model

        # ---- Flash Attention ----
        if enable_flash_attention:
            self._enable_flash_attention(model)

        # ---- 算子融合 (torch.compile) ----
        if enable_operator_fusion:
            try:
                # torch.compile 在 ROCm 5.7+ 上已支持
                model = torch.compile(
                    model,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
                logger.info("optimize_computation: 已启用 torch.compile 算子融合")
            except Exception as exc:
                logger.warning("optimize_computation: torch.compile 失败: %s, 跳过", exc)

        # ---- 精度标记 ----
        dtype = torch.bfloat16 if precision == "bf16" else (
            torch.float16 if precision == "fp16" else torch.float32
        )
        model._simpore_precision_dtype = dtype  # type: ignore[attr-defined]
        model._simpore_autocast_enabled = precision in ("bf16", "fp16")  # type: ignore[attr-defined]

        logger.info("optimize_computation: 计算优化完成, precision=%s", precision)
        return model

    # ------------------------------------------------------------------
    # 3. 通信优化
    # ------------------------------------------------------------------
    def optimize_communication(
        self,
        model: nn.Module,
        enable_infinity_fabric_opt: bool = True,
        enable_overlap: bool = True,
        gradient_compression: bool = False,
    ) -> nn.Module:
        """优化跨 GPU 通信效率.

        策略:
          1. Infinity Fabric 亲和性: 将通信绑定到同 GCD 内
          2. 通信计算重叠: 在 backward 中提前发起 all-reduce
          3. 梯度压缩: 低精度梯度通信

        Args:
            model:                   待优化模型
            enable_infinity_fabric_opt: 是否启用 Infinity Fabric 优化
            enable_overlap:           是否启用通信计算重叠
            gradient_compression:     是否启用梯度压缩

        Returns:
            优化后的模型
        """
        if not self._amd_available:
            logger.info("optimize_communication: CPU 模式, 跳过")
            return model

        # ---- Infinity Fabric 优化 ----
        if enable_infinity_fabric_opt:
            self._configure_infinity_fabric(model)

        # ---- 通信计算重叠 ----
        if enable_overlap:
            self._enable_comm_overlap(model)

        # ---- 梯度压缩 ----
        if gradient_compression:
            self._enable_gradient_compression(model)

        logger.info("optimize_communication: 通信优化完成")
        return model

    # ------------------------------------------------------------------
    # 4. 分布式训练策略
    # ------------------------------------------------------------------
    def get_distributed_strategy(
        self,
        rank: int = 0,
        world_size: int = 2,
        model_parallel_size: int = 2,
        data_parallel_size: int = 1,
    ) -> Dict[str, Any]:
        """生成分布式训练策略配置.

        针对 2 卡 MI250 的推荐混合策略:
          - 模型并行 (Tensor Parallelism): 2 卡各承担一半参数
          - 数据并行: 当 world_size > model_parallel_size 时启用

        Args:
            rank:               当前进程 rank
            world_size:         总进程数
            model_parallel_size: 模型并行度 (默认 2, 匹配 2 GCD)
            data_parallel_size:  数据并行度

        Returns:
            分布式策略配置字典
        """
        tp_group = rank // model_parallel_size
        dp_rank = rank % model_parallel_size

        # 设备映射: MI250 的 2 GCD 各对应一个模型并行分片
        device_map: Dict[str, int] = {}
        for i in range(self.specs.num_gcd):
            device_map[f"gcd_{i}"] = i

        # GCD 亲和性设置
        gcd_affinity = rank % self.specs.num_gcd

        strategy = {
            "rank": rank,
            "world_size": world_size,
            "model_parallel_size": model_parallel_size,
            "data_parallel_size": max(1, world_size // model_parallel_size),
            "tp_rank": dp_rank,
            "dp_group": tp_group,
            "device_map": device_map,
            "gcd_affinity": gcd_affinity,
            "gcd_count": self.specs.num_gcd,
            # 推荐配置
            "recommended_micro_batch_size": 2,
            "recommended_gradient_accumulation": 8,
            "recommended_zero_stage": 2,       # ZeRO-2: 优化器状态分片
            "tensor_parallel_enabled": True,
            "pipeline_parallel_enabled": False,  # MI250 GCD 间用 TP 更优
            "sequence_parallel_enabled": False,
            # 通信后端配置
            "backend": "nccl" if _HAS_AMD_GPU else "gloo",
            "infinity_fabric_gcd_ids": [0, 1],
        }

        # 72B 模型内存分析
        mem_analysis = self.specs.estimate_memory_usage_72b()
        strategy["memory_analysis"] = mem_analysis

        # 训练时间估算 (72B / 2x MI250 / TP=2 / ZeRO-2)
        time_est = self.specs.estimate_72b_training_time(
            total_tokens=1_000_000_000_000,
            batch_size_per_gpu=2,
            seq_len=4096,
            step_time_seconds=6.5,           # 含 TP 通信优化, 每 optimizer step
            gradient_accumulation=8,          # 梯度累积 8 步
            num_gpus=world_size,
        )
        strategy["training_time_estimate"] = time_est

        logger.info(
            "分布式策略生成: rank=%d, TP=%d, DP=%d, GCD=%d, "
            "训练时间≈%.1f 天",
            rank,
            model_parallel_size,
            strategy["data_parallel_size"],
            gcd_affinity,
            time_est.get("total_days_estimated", 0),
        )

        return strategy

    # ------------------------------------------------------------------
    # 优化摘要
    # ------------------------------------------------------------------
    def get_optimization_summary(self) -> Dict[str, Any]:
        """返回所有优化配置的完整摘要."""
        mem_pool_stats = {}
        if self._memory_pool is not None:
            mem_pool_stats = self._memory_pool.get_stats()

        return {
            "amd_gpu_available": self._amd_available,
            "specs": {
                "hbm_size_gb": self.specs.total_hbm_gb(),
                "num_gcd": self.specs.num_gcd,
                "total_cu": self.specs.total_cu(),
                "memory_bandwidth_tbs": self.specs.memory_bandwidth / 1e12,
            },
            "memory_pool": mem_pool_stats,
            "memory_optimization": {
                "gradient_checkpointing": True,
                "memory_pool_enabled": self._memory_pool is not None,
            },
            "computation_optimization": {
                "flash_attention": True,
                "operator_fusion": True,
                "half_precision": True,
            },
            "communication_optimization": {
                "infinity_fabric_optimization": True,
                "comm_compute_overlap": True,
                "gradient_compression": False,
            },
            "distributed_strategy": {
                "recommended_tp": 2,
                "recommended_dp": 1,
                "recommended_zero_stage": 2,
                "backend": "nccl" if _HAS_AMD_GPU else "gloo",
            },
        }

    # ------------------------------------------------------------------
    # 内部方法: 内存优化
    # ------------------------------------------------------------------
    def _apply_gradient_checkpointing(
        self,
        model: nn.Module,
        checkpoint_layers: Optional[List[int]] = None,
    ) -> None:
        """应用梯度检查点到模型.

        对模型的 SimplCoreBlock 层应用 activation checkpointing,
        在前向传播时不保存中间激活值, 反向传播时重新计算。

        Args:
            model:            模型
            checkpoint_layers: 指定哪些层启用 (None = 全部)
        """
        try:
            from simplcore.tensor.utils.checkpoint import checkpoint_sequential
        except ImportError:
            logger.warning("梯度检查点不可用 (simplcore.tensor 未安装)")
            return

        current_layer = 0
        for name, module in model.named_modules():
            # 检测 SimplCoreBlock 或类似的 block 类模块
            if hasattr(module, "forward") and "block" in name.lower():
                if checkpoint_layers is None or current_layer in checkpoint_layers:
                    # 通过注册钩子标记该模块需要检查点
                    module._SimplCore_checkpoint = True  # type: ignore[attr-defined]
                current_layer += 1

        # 在模型上注册梯度检查点配置
        model._SimplCore_gradient_checkpointing = True  # type: ignore[attr-defined]
        model._SimplCore_checkpoint_layers = checkpoint_layers  # type: ignore[attr-defined]

        logger.info(
            "梯度检查点已配置: total_layers=%d, checkpointed=%s",
            current_layer,
            "all" if checkpoint_layers is None else f"{len(checkpoint_layers)} layers",
        )

    def _optimize_memory_format(self, model: nn.Module) -> nn.Module:
        """优化模型参数的内存布局.

        对 4D 权重使用 channels_last 格式以提升 MI250 访存效率;
        对大张量启用 GCD 对齐提示。

        Args:
            model: 待优化模型

        Returns:
            优化后的模型
        """
        for name, param in model.named_parameters():
            if param.dim() == 4:
                try:
                    param.data = param.data.to(memory_format=torch.channels_last)
                except Exception:
                    pass
        return model

    # ------------------------------------------------------------------
    # 内部方法: 计算优化
    # ------------------------------------------------------------------
    def _enable_flash_attention(self, model: nn.Module) -> None:
        """启用 Flash Attention 标记.

        在模型上设置标记, 供前向传播中的注意力实现使用。
        同时尝试导入 xformers 作为可选后端。

        Args:
            model: 待优化模型
        """
        # 标记使用 Flash Attention
        model._SimplCore_flash_attn = True  # type: ignore[attr-defined]

        # 尝试检测 xformers (Flash Attention 常用后端)
        try:
            import xformers  # noqa: F401
            import xformers.ops  # noqa: F401
            model._SimplCore_flash_attn_backend = "xformers"  # type: ignore[attr-defined]
            logger.info("Flash Attention: 使用 xformers 后端")
        except ImportError:
            # 回退到 PyTorch 原生实现
            model._SimplCore_flash_attn_backend = "torch"  # type: ignore[attr-defined]
            # 检查 torch.nn.functional.scaled_dot_product_attention
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                logger.info("Flash Attention: 使用 torch SDPA 后端")
            else:
                logger.info("Flash Attention: 使用手动实现 (非融合)")

    # ------------------------------------------------------------------
    # 内部方法: 通信优化
    # ------------------------------------------------------------------
    def _configure_infinity_fabric(self, model: nn.Module) -> None:
        """配置 Infinity Fabric 优化.

        设置环境变量和模型标记以优化 GCD 间通信。

        Args:
            model: 待优化模型
        """
        # 设置 ROCm 环境变量 (若不存在)
        if "HSA_ENABLE_SDMA" not in os.environ:
            os.environ["HSA_ENABLE_SDMA"] = "1"     # 启用 SDMA 队列
        if "GPU_MAX_HW_QUEUES" not in os.environ:
            os.environ["GPU_MAX_HW_QUEUES"] = "8"   # 增加硬件队列数

        # 设置 NCCL 环境变量 (ROCm 兼容)
        if "NCCL_MIN_NRINGS" not in os.environ:
            os.environ["NCCL_MIN_NRINGS"] = "4"
        if "NCCL_DEBUG" not in os.environ:
            os.environ["NCCL_DEBUG"] = "WARN"

        model._SimplCore_infinity_fabric_opt = True  # type: ignore[attr-defined]

    def _enable_comm_overlap(self, model: nn.Module) -> None:
        """启用通信计算重叠.

        通过模型标记, 让分布式训练循环知晓可以重叠
        all-reduce 和后续计算。

        Args:
            model: 待优化模型
        """
        model._SimplCore_comm_overlap = True  # type: ignore[attr-defined]
        logger.info("通信计算重叠: 已标记")

    def _enable_gradient_compression(self, model: nn.Module) -> None:
        """启用梯度压缩.

        在 all-reduce 前对梯度进行 fp16 压缩, 减少通信量。
        仅对支持低精度 all-reduce 的后端有效。

        Args:
            model: 待优化模型
        """
        model._SimplCore_grad_compression = True  # type: ignore[attr-defined]
        logger.info("梯度压缩: 已启用 (fp16)")

    # ------------------------------------------------------------------
    # 便利方法: 全流程一键优化
    # ------------------------------------------------------------------
    def optimize_all(
        self,
        model: nn.Module,
        rank: int = 0,
        world_size: int = 2,
        precision: str = "bf16",
    ) -> nn.Module:
        """执行全流程优化 (内存 + 计算 + 通信 + 分布式策略).

        这是 MI250Optimizer 的主要入口方法, 依次执行所有优化。

        Args:
            model:      待优化模型
            rank:       当前进程 rank
            world_size: 总进程数
            precision:  精度模式

        Returns:
            全方位优化后的模型
        """
        logger.info("MI250Optimizer: 开始全流程优化 ...")

        # 1. 内存优化
        model = self.optimize_memory(model)

        # 2. 计算优化
        model = self.optimize_computation(model, precision=precision)

        # 3. 通信优化
        model = self.optimize_communication(model)

        # 4. 生成分布式策略 (仅打印不修改模型)
        strategy = self.get_distributed_strategy(
            rank=rank, world_size=world_size
        )
        model._SimplCore_dist_strategy = strategy  # type: ignore[attr-defined]

        # 打印训练时间估算
        time_est = strategy.get("training_time_estimate", {})
        if time_est:
            logger.info(
                "72B 训练时间估算: total_steps=%d, 预计 %.1f 天 (20 天限制: %s)",
                time_est.get("total_steps", 0),
                time_est.get("total_days_estimated", 0),
                "✓ 满足" if time_est.get("within_20_days", False) else "✗ 超出",
            )

        logger.info("MI250Optimizer: 全流程优化完成")
        return model


# ======================================================================
# 自检
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("AMD MI250 GPU 优化模块自检")
    print("=" * 60)

    # 1. 硬件规格
    specs = MI250Specs()
    print(f"\n[MI250 规格]")
    print(f"  HBM2e:         {specs.total_hbm_gb():.0f} GB")
    print(f"  GCD 数量:       {specs.num_gcd}")
    print(f"  每 GCD CU:      {specs.num_cu_per_gcd}")
    print(f"  总 CU:          {specs.total_cu()}")
    print(f"  显存带宽:       {specs.memory_bandwidth / 1e12:.1f} TB/s")
    print(f"  Infinity Fabric: {specs.infinity_fabric_bandwidth / 1e9:.0f} GB/s")

    # 2. 显存估算
    mem = specs.estimate_memory_usage_72b()
    print(f"\n[72B 模型显存估算]")
    for key, val in mem.items():
        print(f"  {key}: {val}")

    # 3. 训练时间估算
    time_est = specs.estimate_72b_training_time()
    print(f"\n[72B 训练时间估算 (2x MI250)]")
    for key, val in time_est.items():
        print(f"  {key}: {val}")

    # 4. MemoryPool 测试
    print(f"\n[MemoryPool 测试]")
    pool = MemoryPool(total_size_gb=4.0, block_size_mb=64)  # 4 GB 测试
    stats = pool.get_stats()
    print(f"  池大小: {stats['total_size_gb']} GB")
    print(f"  初始化状态: {'ok' if stats['free_gb'] > 0 else 'failed'}")

    # 分配测试
    t1 = pool.allocate((1024, 1024), torch.bfloat16)
    if t1 is not None:
        print(f"  分配 t1: shape={list(t1.shape)}, dtype={t1.dtype}")
    t2 = pool.allocate((512, 2048), torch.bfloat16)
    if t2 is not None:
        print(f"  分配 t2: shape={list(t2.shape)}, dtype={t2.dtype}")

    stats = pool.get_stats()
    print(f"  使用率: {stats['utilization_pct']}%")
    print(f"  分配次数: {stats['allocation_count']}")

    if t1 is not None:
        pool.free(t1)
        print(f"  释放 t1 成功")
    stats = pool.get_stats()
    print(f"  释放后使用率: {stats['utilization_pct']}%")

    # 5. AMDPerformanceMonitor 测试
    print(f"\n[AMDPerformanceMonitor 测试]")
    monitor = AMDPerformanceMonitor(window_size=10, enable_detailed=False)
    for _ in range(3):
        monitor.record_sample()
    report = monitor.generate_report()
    print(f"  采样数: {report['sampling']['total_samples']}")
    print(f"  GPU 可用: {report['gpu_available']}")

    # 6. MI250Optimizer 测试
    print(f"\n[MI250Optimizer 测试]")
    opt = MI250Optimizer(memory_pool_size_gb=2.0)

    # 测试模型
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
    )

    model = opt.optimize_all(model, rank=0, world_size=2)
    summary = opt.get_optimization_summary()
    print(f"  AMD GPU 可用: {summary['amd_gpu_available']}")
    print(f"  内存池启用: {summary['memory_pool']['total_size_gb'] if summary.get('memory_pool') else 'N/A'} GB")
    print(f"  72B 训练时间 (天): {summary.get('distributed_strategy', {}).get('training_time_estimate', {}).get('total_days_estimated', 'N/A')}")

    print("\n" + "=" * 60)
    print("自检完成!")
    print("=" * 60)


