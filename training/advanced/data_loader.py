"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

data_loader.py - 数据加载模块
512K长序列数据加载、预取（prefetch）+缓存、动态批处理、
数据混洗（分布式）、异步数据加载、数据分布统计、性能分析
"""

import os
import sys
import math
import time
import random
import logging
import threading
import queue
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator, Iterable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import multiprocessing as mp

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.DataLoader")


@dataclass
class DataStats:
    total_samples: int = 0
    total_tokens: int = 0
    batch_count: int = 0
    avg_seq_length: float = 0.0
    max_seq_length: int = 0
    min_seq_length: int = float("inf")
    avg_batch_tokens: float = 0.0
    load_time_ms: float = 0.0
    prefetch_hit_rate: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    skipped_samples: int = 0


@dataclass
class DataDistribution:
    seq_length_histogram: Dict[int, int] = field(default_factory=dict)
    token_count: int = 0
    unique_sequences: int = 0
    avg_tokens_per_sample: float = 0.0
    length_bins: List[int] = field(default_factory=lambda: [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288])


class PrefetchWorker(threading.Thread):
    def __init__(self, data_source, prefetch_queue, prefetch_size=4):
        super().__init__(daemon=True)
        self.data_source = data_source
        self.prefetch_queue = prefetch_queue
        self.prefetch_size = prefetch_size
        self.running = True
        self._iterator = iter(data_source)

    def run(self):
        while self.running:
            try:
                if self.prefetch_queue.qsize() < self.prefetch_size:
                    batch = next(self._iterator)
                    self.prefetch_queue.put(batch, timeout=5)
                else:
                    time.sleep(0.01)
            except StopIteration:
                self._iterator = iter(self.data_source)
            except Exception as e:
                logger.error(f"预取线程异常: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False


class QingxiaotuanDataLoader:
    """
    青小团数据加载器
    支持512K长序列数据加载、预取+缓存、动态批处理、
    数据混洗（分布式）、异步数据加载、数据分布统计、性能分析

    针对70B MoE模型的长序列训练场景优化数据加载效率
    """

    def __init__(
        self,
        dataset: Iterable,
        batch_size: int = 1,
        max_seq_length: int = 524288,
        shuffle: bool = True,
        shuffle_buffer_size: int = 10000,
        drop_last: bool = True,
        prefetch_size: int = 4,
        cache_size: int = 1000,
        num_workers: int = 2,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        dynamic_batching: bool = True,
        max_batch_tokens: int = 524288,
        pack_sequences: bool = True,
        enable_profiling: bool = True,
        timeout: float = 60.0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.drop_last = drop_last
        self.prefetch_size = prefetch_size
        self.cache_size = cache_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.dynamic_batching = dynamic_batching
        self.max_batch_tokens = max_batch_tokens
        self.pack_sequences = pack_sequences
        self.enable_profiling = enable_profiling
        self.timeout = timeout

        self.stats = DataStats()
        self.distribution = DataDistribution()
        self._cache: Dict[str, Any] = {}
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=prefetch_size * 2)
        self._prefetch_worker: Optional[PrefetchWorker] = None
        self._epoch = 0
        self._batch_index = 0
        self._shuffle_buffer: List[Any] = []
        self._local_indices: List[int] = []
        self._load_times: deque = deque(maxlen=100)
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

        self._setup_distributed()
        self._start_prefetch()
        logger.info(f"数据加载器初始化完成: batch_size={batch_size}, "
                    f"max_seq_len={max_seq_length}, "
                    f"prefetch={prefetch_size}, workers={num_workers}, "
                    f"dynamic_batching={'启用' if dynamic_batching else '禁用'}")

    def _setup_distributed(self):
        if self.distributed and dist.is_initialized():
            if self.world_size <= 1:
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
        total_samples = len(self.dataset) if hasattr(self.dataset, "__len__") else 0
        if total_samples > 0 and self.distributed:
            indices = list(range(total_samples))
            self._local_indices = indices[self.rank:self.world_size]
            logger.info(f"分布式数据加载: rank={self.rank}, world={self.world_size}, "
                        f"local_samples={len(self._local_indices)}")

    def _start_prefetch(self):
        if self.prefetch_size > 0 and not self._prefetch_worker:
            self._prefetch_worker = PrefetchWorker(
                self._data_generator(),
                self._prefetch_queue,
                self.prefetch_size,
            )
            self._prefetch_worker.start()

    def _data_generator(self) -> Iterator[Any]:
        while True:
            batch = self._load_batch()
            if batch is not None:
                yield batch

    def _load_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        samples = []
        for _ in range(self.batch_size):
            sample = self._get_next_sample()
            if sample is None:
                break
            samples.append(sample)

        if not samples:
            return None

        if len(samples) < self.batch_size and self.drop_last:
            return None

        if self.dynamic_batching:
            return self._dynamic_batch(samples)
        else:
            return self._collate_batch(samples)

    def _get_next_sample(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            while len(self._shuffle_buffer) < self.shuffle_buffer_size:
                try:
                    sample = self._read_next_sample()
                    if sample is None:
                        break
                    self._shuffle_buffer.append(sample)
                except Exception as e:
                    logger.warning(f"读取样本异常: {e}")
                    continue

            if not self._shuffle_buffer:
                return None

            if self.shuffle:
                idx = self._rng.randint(0, len(self._shuffle_buffer) - 1)
                self._shuffle_buffer[-1], self._shuffle_buffer[idx] = \
                    self._shuffle_buffer[idx], self._shuffle_buffer[-1]

            return self._shuffle_buffer.pop()

    def _read_next_sample(self) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(self.dataset, "__getitem__"):
                if self.distributed and self._local_indices:
                    idx = self._local_indices[self._batch_index % len(self._local_indices)]
                else:
                    idx = self._batch_index % len(self.dataset) if hasattr(self.dataset, "__len__") else 0
                self._batch_index += 1
                if hasattr(self.dataset, "__len__") and self._batch_index >= len(self.dataset):
                    self._epoch += 1
                    self._batch_index = 0
                    if self.distributed and self._local_indices:
                        self._rng.shuffle(self._local_indices)
                    return None
                return self.dataset[idx]
            else:
                return next(self.dataset)
        except (StopIteration, IndexError):
            self._epoch += 1
            self._batch_index = 0
            return None

    def _dynamic_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        token_counts = [s.get("input_ids", s.get("tokens", [])).shape[0] if torch.is_tensor(
            s.get("input_ids", s.get("tokens", []))) else len(s.get("input_ids", s.get("tokens", [])))
                        for s in samples]

        total_tokens = sum(token_counts)
        while total_tokens > self.max_batch_tokens and len(samples) > 1:
            samples.pop()
            token_counts.pop()
            total_tokens = sum(token_counts)

        return self._collate_batch(samples)

    def _collate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        keys = samples[0].keys() if samples else []

        if self.pack_sequences:
            all_input_ids = []
            all_labels = []
            all_attention_mask = []
            for sample in samples:
                input_ids = sample.get("input_ids", sample.get("tokens"))
                if torch.is_tensor(input_ids):
                    all_input_ids.append(input_ids)
                else:
                    all_input_ids.append(torch.tensor(input_ids))
                if "labels" in sample:
                    labels = sample["labels"]
                    if torch.is_tensor(labels):
                        all_labels.append(labels)
                    else:
                        all_labels.append(torch.tensor(labels))
                if "attention_mask" in sample:
                    mask = sample["attention_mask"]
                    if torch.is_tensor(mask):
                        all_attention_mask.append(mask)
                    else:
                        all_attention_mask.append(torch.tensor(mask))

            batch["input_ids"] = torch.cat(all_input_ids, dim=0).unsqueeze(0)
            if all_labels:
                batch["labels"] = torch.cat(all_labels, dim=0).unsqueeze(0)
            if all_attention_mask:
                batch["attention_mask"] = torch.cat(all_attention_mask, dim=0).unsqueeze(0)

            batch["seq_lengths"] = torch.tensor([len(ids) for ids in all_input_ids])
            batch["packed"] = True
        else:
            for key in keys:
                if key in ("input_ids", "tokens", "labels", "attention_mask"):
                    tensors = []
                    for sample in samples:
                        val = sample.get(key)
                        if torch.is_tensor(val):
                            tensors.append(val)
                        elif val is not None:
                            tensors.append(torch.tensor(val))
                    if tensors:
                        max_len = max(t.shape[0] for t in tensors) if tensors else 0
                        padded = torch.zeros(len(tensors), max_len, dtype=tensors[0].dtype)
                        for i, t in enumerate(tensors):
                            padded[i, :t.shape[0]] = t
                        batch[key] = padded

        seq_len = batch["input_ids"].shape[1] if "input_ids" in batch else 0
        self.stats.total_samples += len(samples)
        self.stats.total_tokens += seq_len
        self.stats.batch_count += 1
        self.stats.avg_seq_length = 0.9 * self.stats.avg_seq_length + 0.1 * seq_len
        self.stats.max_seq_length = max(self.stats.max_seq_length, seq_len)
        self.stats.min_seq_length = min(self.stats.min_seq_length, seq_len)

        return batch

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        start_time = time.time()

        if self._prefetch_worker is not None and self._prefetch_worker.is_alive():
            try:
                batch = self._prefetch_queue.get(timeout=self.timeout)
                self.stats.prefetch_hit_rate = 0.9 * self.stats.prefetch_hit_rate + 0.1
            except queue.Empty:
                self.stats.prefetch_hit_rate = 0.9 * self.stats.prefetch_hit_rate + 0.0
                batch = self._load_batch()
        else:
            batch = self._load_batch()

        if batch is None:
            raise StopIteration

        elapsed_ms = (time.time() - start_time) * 1000
        self._load_times.append(elapsed_ms)
        self.stats.load_time_ms = np.mean(self._load_times)

        if self.enable_profiling and self.stats.batch_count % 100 == 0:
            self._log_performance()

        return batch

    def __len__(self) -> int:
        if hasattr(self.dataset, "__len__"):
            total = len(self.dataset)
            if self.distributed:
                total = total // self.world_size
            return total // self.batch_size
        return 0

    def _log_performance(self):
        avg_time = np.mean(self._load_times) if self._load_times else 0
        tokens_per_sec = self.stats.total_tokens / (max(1, sum(self._load_times)) * 1000)
        self.stats.throughput_tokens_per_sec = tokens_per_sec
        logger.info(
            f"[数据加载] batch={self.stats.batch_count}, "
            f"avg_load={avg_time:.1f}ms, "
            f"avg_seq_len={self.stats.avg_seq_length:.0f}, "
            f"tokens/sec={tokens_per_sec:.0f}, "
            f"prefetch_hit={self.stats.prefetch_hit_rate:.2f}, "
            f"epoch={self._epoch}"
        )

    def analyze_data_distribution(self) -> DataDistribution:
        self.distribution = DataDistribution()
        total_len = 0
        count = 0

        for i in range(min(1000, len(self.dataset)) if hasattr(self.dataset, "__len__") else 0):
            sample = self.dataset[i]
            seq_len = sample.get("input_ids", sample.get("tokens")).shape[0] if torch.is_tensor(
                sample.get("input_ids", sample.get("tokens"))) else len(sample.get("input_ids", sample.get("tokens"), []))
            total_len += seq_len
            count += 1
            for bin_size in self.distribution.length_bins:
                if seq_len <= bin_size:
                    self.distribution.seq_length_histogram[bin_size] = \
                        self.distribution.seq_length_histogram.get(bin_size, 0) + 1
                    break

        self.distribution.token_count = total_len
        self.distribution.unique_sequences = count
        self.distribution.avg_tokens_per_sample = total_len / max(1, count)

        logger.info("数据分布分析完成")
        return self.distribution

    def get_data_stats(self) -> DataStats:
        return self.stats

    def get_distribution(self) -> DataDistribution:
        return self.distribution

    def shutdown(self):
        if self._prefetch_worker is not None:
            self._prefetch_worker.stop()
            self._prefetch_worker = None
        logger.info("数据加载器已关闭")

    def reset(self):
        self.shutdown()
        self._epoch = 0
        self._batch_index = 0
        self._shuffle_buffer.clear()
        self.stats = DataStats()
        self._start_prefetch()
        logger.info("数据加载器已重置")

    def print_data_summary(self):
        summary = [
            "=" * 60,
            "青小团数据加载 - 数据信息摘要",
            "=" * 60,
            f"总样本数: {self.stats.total_samples}",
            f"总Token数: {self.stats.total_tokens:,}",
            f"Batch数: {self.stats.batch_count}",
            f"平均序列长度: {self.stats.avg_seq_length:.0f}",
            f"最大序列长度: {self.stats.max_seq_length}",
            f"最小序列长度: {self.stats.min_seq_length}",
            f"---",
            f"预取命中率: {self.stats.prefetch_hit_rate:.2f}",
            f"缓存命中率: {self.stats.cache_hit_rate:.2f}",
            f"加载延迟: {self.stats.load_time_ms:.1f}ms",
            f"吞吐量: {self.stats.throughput_tokens_per_sec:.0f} tokens/sec",
            f"---",
            f"动态批处理: {'启用' if self.dynamic_batching else '禁用'}",
            f"序列打包: {'启用' if self.pack_sequences else '禁用'}",
            f"分布式: {'启用' if self.distributed else '禁用'}",
            f"工作线程: {self.num_workers}",
            f"预取大小: {self.prefetch_size}",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
