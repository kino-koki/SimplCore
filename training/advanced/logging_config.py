"""
青小团 70B MoE 高级训练工具与优化器
青奈科技 (c) 2026

logging_config.py - 日志配置模块
结构化日志、日志级别动态控制、日志轮转、
分布式日志收集、训练指标格式化输出、日志分析工具
"""

import os
import sys
import json
import time
import math
import logging
import logging.handlers
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, TextIO
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.LoggingConfig")


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class OutputFormat(Enum):
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    log_level: LogLevel = LogLevel.INFO
    output_format: OutputFormat = OutputFormat.STRUCTURED
    max_file_size_mb: int = 100
    backup_count: int = 10
    enable_console: bool = True
    enable_file: bool = True
    enable_json_file: bool = True
    enable_distributed: bool = True
    log_interval: int = 10
    metrics_format: str = "detailed"


@dataclass
class TrainingMetrics:
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    gpu_memory_allocated_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0
    gpu_utilization: float = 0.0
    moe_aux_loss: float = 0.0
    moe_load_balancing_loss: float = 0.0
    epoch: int = 0
    timestamp: str = ""
    rank: int = 0


class StructuredFormatter(logging.Formatter):
    def __init__(self, fmt: Optional[str] = None, output_format: OutputFormat = OutputFormat.STRUCTURED):
        super().__init__(fmt)
        self.output_format = output_format

    def format(self, record: logging.LogRecord) -> str:
        if self.output_format == OutputFormat.JSON:
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if hasattr(record, "metrics"):
                log_data["metrics"] = record.metrics
            if record.exc_info:
                log_data["exception"] = traceback.format_exception(*record.exc_info)
            return json.dumps(log_data, ensure_ascii=False)

        elif self.output_format == OutputFormat.STRUCTURED:
            timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            level = record.levelname.ljust(8)
            logger_name = record.name.split(".")[-1].ljust(20)
            message = record.getMessage()
            return f"[{timestamp}] {level} | {logger_name} | {message}"

        return super().format(record)


class QingxiaotuanLoggingConfig:
    """
    青小团日志配置器
    支持结构化日志、日志级别动态控制、日志轮转、
    分布式日志收集、训练指标格式化输出、日志分析工具

    针对70B MoE模型的分布式训练场景，提供统一的日志管理体系
    """

    def __init__(
        self,
        config: Optional[LoggingConfig] = None,
        name: str = "青小团",
        rank: int = 0,
    ):
        self.config = config or LoggingConfig()
        self.name = name
        self.rank = rank
        self._log_dir = Path(self.config.log_dir)
        self._loggers: Dict[str, logging.Logger] = {}
        self._metrics_history: deque = deque(maxlen=1000)
        self._handler_ids: List[int] = []

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1

        self._setup_logging()
        logger.info(f"日志配置初始化完成: level={self.config.log_level.name}, "
                    f"format={self.config.output_format.value}, "
                    f"rank={self.rank}/{self.world_size}")

    def _setup_logging(self):
        self._log_dir.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level.value)

        if self.config.enable_console:
            self._add_console_handler()

        if self.config.enable_file:
            self._add_file_handler()

        if self.config.enable_json_file:
            self._add_json_file_handler()

    def _add_console_handler(self):
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                return
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.config.log_level.value)
        formatter = StructuredFormatter(output_format=self.config.output_format)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    def _add_file_handler(self):
        log_file = self._log_dir / f"qingxiaotuan_rank{self.rank}.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count,
            encoding="utf-8",
        )
        handler.setLevel(self.config.log_level.value)
        formatter = StructuredFormatter(output_format=OutputFormat.TEXT)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        self._handler_ids.append(id(handler))

    def _add_json_file_handler(self):
        json_log_file = self._log_dir / f"qingxiaotuan_rank{self.rank}_metrics.jsonl"
        handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        formatter = StructuredFormatter(output_format=OutputFormat.JSON)
        handler.setFormatter(formatter)
        json_logger = logging.getLogger("青小团.Metrics")
        json_logger.addHandler(handler)
        json_logger.setLevel(logging.INFO)
        self._handler_ids.append(id(handler))

    def set_level(self, level: LogLevel):
        self.config.log_level = level
        logging.getLogger().setLevel(level.value)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level.value)
        logger.info(f"日志级别已切换为: {level.name}")

    def get_logger(self, name: str) -> logging.Logger:
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(f"{self.name}.{name}")
        return self._loggers[name]

    def log_metrics(self, metrics: TrainingMetrics, logger_name: str = "青小团.Metrics"):
        self._metrics_history.append(metrics)
        log_record = logging.LogRecord(
            name=logger_name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=None,
            exc_info=None,
        )
        log_record.metrics = asdict(metrics)

        formatted = self._format_metrics(metrics)
        standard_logger = logging.getLogger(logger_name)
        standard_logger.info(formatted)

    def _format_metrics(self, metrics: TrainingMetrics) -> str:
        if self.config.metrics_format == "minimal":
            return (f"[Step {metrics.step}] loss={metrics.loss:.4f}, "
                    f"lr={metrics.learning_rate:.2e}, "
                    f"tok/s={metrics.throughput_tokens_per_sec:.0f}")

        elif self.config.metrics_format == "detailed":
            return (
                f"[Step {metrics.step}/{metrics.epoch}] "
                f"loss={metrics.loss:.4f} | "
                f"lr={metrics.learning_rate:.2e} | "
                f"grad_norm={metrics.grad_norm:.4f} | "
                f"throughput={metrics.throughput_tokens_per_sec:.0f} tok/s | "
                f"GPU_mem={metrics.gpu_memory_allocated_gb:.1f}GB | "
                f"moe_aux={metrics.moe_aux_loss:.4f} | "
                f"moe_lb={metrics.moe_load_balancing_loss:.4f}"
            )

        return (f"[Step {metrics.step}] loss={metrics.loss:.4f}")

    def log_training_start(self, config: Dict[str, Any]):
        start_msg = [
            "=" * 70,
            f"青小团 70B MoE 训练启动 (Rank {self.rank}/{self.world_size})",
            "=" * 70,
            f"启动时间: {datetime.now().isoformat()}",
            f"训练配置:",
        ]
        for key, value in config.items():
            start_msg.append(f"  {key}: {value}")
        start_msg.append("=" * 70)
        for line in start_msg:
            logger.info(line)

    def log_training_end(self, total_steps: int, total_time: float, best_loss: float):
        end_msg = [
            "=" * 70,
            f"青小团 70B MoE 训练完成 (Rank {self.rank}/{self.world_size})",
            "=" * 70,
            f"完成时间: {datetime.now().isoformat()}",
            f"总步数: {total_steps}",
            f"总用时: {total_time:.2f}s ({total_time / 3600:.2f}h)",
            f"最佳loss: {best_loss:.4f}",
            "=" * 70,
        ]
        for line in end_msg:
            logger.info(line)

    def get_recent_metrics(self, n: int = 10) -> List[TrainingMetrics]:
        return list(self._metrics_history)[-n:]

    def analyze_metrics(self) -> Dict[str, Any]:
        if not self._metrics_history:
            return {}
        losses = [m.loss for m in self._metrics_history]
        grad_norms = [m.grad_norm for m in self._metrics_history if m.grad_norm > 0]
        throughputs = [m.throughput_tokens_per_sec for m in self._metrics_history if m.throughput_tokens_per_sec > 0]
        gpu_mems = [m.gpu_memory_allocated_gb for m in self._metrics_history]

        return {
            "loss": {
                "min": min(losses),
                "max": max(losses),
                "mean": np.mean(losses),
                "last": losses[-1],
                "trend": "下降" if len(losses) > 10 and losses[-1] < losses[0] else "波动",
            },
            "grad_norm": {
                "mean": np.mean(grad_norms) if grad_norms else 0,
                "max": max(grad_norms) if grad_norms else 0,
            },
            "throughput": {
                "mean": np.mean(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0,
            },
            "gpu_memory": {
                "mean": np.mean(gpu_mems) if gpu_mems else 0,
                "peak": max(gpu_mems) if gpu_mems else 0,
            },
            "total_steps": len(self._metrics_history),
        }

    def export_metrics_jsonl(self, output_path: Optional[str] = None):
        if output_path is None:
            output_path = str(self._log_dir / f"metrics_export_rank{self.rank}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for metrics in self._metrics_history:
                f.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")
        logger.info(f"训练指标已导出至: {output_path}")

    def get_metrics_dataframe(self) -> "pd.DataFrame":
        try:
            import pandas as pd
            records = [asdict(m) for m in self._metrics_history]
            return pd.DataFrame(records)
        except ImportError:
            logger.warning("需要安装 pandas 来生成 DataFrame")
            return None

    def cleanup(self):
        for handler in logging.getLogger().handlers[:]:
            if id(handler) in self._handler_ids:
                handler.close()
                logging.getLogger().removeHandler(handler)
        self._handler_ids.clear()
        logger.info("日志处理器已清理")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "log_level": self.config.log_level.name,
                "output_format": self.config.output_format.value,
                "log_interval": self.config.log_interval,
                "metrics_format": self.config.metrics_format,
            },
            "total_metrics": len(self._metrics_history),
            "rank": self.rank,
            "world_size": self.world_size,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        config_data = state_dict.get("config", {})
        if config_data:
            log_level = config_data.get("log_level", "INFO")
            self.config.log_level = LogLevel[log_level]
            self.set_level(self.config.log_level)
        logger.info("日志配置状态已恢复")

    def print_logging_summary(self):
        analysis = self.analyze_metrics()
        summary = [
            "=" * 60,
            "青小团日志配置 - 日志信息摘要",
            "=" * 60,
            f"日志级别: {self.config.log_level.name}",
            f"输出格式: {self.config.output_format.value}",
            f"日志目录: {self._log_dir}",
            f"日志轮转: {self.config.max_file_size_mb}MB x {self.config.backup_count}",
            f"Rank: {self.rank}/{self.world_size}",
            f"---",
            f"记录的指标数: {len(self._metrics_history)}",
            f"Loss范围: [{analysis.get('loss', {}).get('min', 0):.4f}, "
            f"{analysis.get('loss', {}).get('max', 0):.4f}]",
            f"Loss趋势: {analysis.get('loss', {}).get('trend', 'N/A')}",
            f"平均吞吐: {analysis.get('throughput', {}).get('mean', 0):.0f} tok/s",
            f"峰值显存: {analysis.get('gpu_memory', {}).get('peak', 0):.1f}GB",
            "=" * 60,
        ]
        for line in summary:
            logger.info(line)
        return "\n".join(summary)
