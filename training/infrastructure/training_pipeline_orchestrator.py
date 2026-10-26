"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

training_pipeline_orchestrator.py - 训练管线编排器
完整的训练生命周期管理，多阶段训练编排
"""

import os
import sys
import json
import time
import math
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import traceback

import torch
import numpy as np
import yaml

from .checkpoint_manager import (
    CheckpointManager, AsyncCheckpointManager, DistributedCheckpointManager
)
from .training_monitor import (
    TrainingMonitor, MonitorMetrics, GPUMonitor, MoELoadBalancer
)
from .experiment_tracker import ExperimentTracker
from .fault_tolerance import FaultTolerance, NodeInfo, NodeStatus
from .distributed_coordinator import (
    DistributedCoordinator, TrainingMode, CommunicationTopology, ParallelStrategy
)
from .profile_analyzer import ProfileAnalyzer

logger = logging.getLogger("青小团.TrainingPipelineOrchestrator")


class TrainingStage(Enum):
    """训练阶段"""
    PRETRAIN = "pretrain"
    SFT = "sft"
    RLHF = "rlhf"
    CONTINUOUS_PRETRAIN = "continuous_pretrain"


class PipelineState(Enum):
    """管线状态"""
    INITIALIZING = "initializing"
    ENV_CHECK = "env_check"
    DATA_PREP = "data_prep"
    MODEL_INIT = "model_init"
    DIST_SETUP = "dist_setup"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class PipelineConfig:
    """青小团训练管线配置"""
    company: str = "青奈科技"
    model_name: str = "青小团"
    experiment_name: str = ""
    output_dir: str = "./training_output"
    seed: int = 42

    # Model
    model_type: str = "moe"
    hidden_size: int = 7168
    num_layers: int = 61
    num_attention_heads: int = 64
    num_experts: int = 64
    top_k: int = 8
    moe_intermediate_size: int = 2048
    shared_intermediate_size: int = 1024
    vocab_size: int = 151936
    max_position_embeddings: int = 131072

    # Training
    batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 2000
    total_steps: int = 100000
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = True
    use_flash_attention: bool = True

    # Distributed
    world_size: int = 8
    num_nodes: int = 1
    data_parallel_size: int = 8
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    training_mode: str = "synchronous"
    communication_topology: str = "all_reduce"

    # Infrastructure
    checkpoint_every_n_steps: int = 1000
    checkpoint_every_n_hours: float = 2.0
    keep_last_n_checkpoints: int = 5
    keep_best_k_checkpoints: int = 3
    max_checkpoints: int = 20
    enable_async_checkpoint: bool = True

    monitor_use_tensorboard: bool = True
    monitor_use_wandb: bool = False
    monitor_loss_spike_threshold: float = 10.0
    monitor_grad_norm_threshold: float = 100.0

    enable_fault_tolerance: bool = True
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 30.0
    enable_auto_restart: bool = True
    max_restart_attempts: int = 3
    enable_elastic_training: bool = False

    # Data
    train_data_path: str = ""
    eval_data_path: str = ""
    data_cache_dir: str = "./data_cache"
    num_workers: int = 8
    prefetch_factor: int = 2

    # Stages
    stages: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TrainingPipelineOrchestrator:
    """
    青小团训练管线编排器

    编排所有基础设施组件，管理完整的训练生命周期：
    配置加载 -> 环境检查 -> 数据准备 -> 模型初始化 -> 分布式设置 ->
    训练循环 -> 监控收集 -> 检查点管理 -> 故障恢复 -> 训练结束处理
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_provider: Optional[Callable] = None,
        data_provider: Optional[Callable] = None,
        local_rank: int = 0,
        node_rank: int = 0,
    ):
        self.config = config
        self.model_provider = model_provider
        self.data_provider = data_provider
        self.local_rank = local_rank
        self.node_rank = node_rank

        self.output_dir = Path(config.output_dir)
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.metrics_dir = self.output_dir / "metrics"

        if self.local_rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None

        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.monitor: Optional[TrainingMonitor] = None
        self.experiment_tracker: Optional[ExperimentTracker] = None
        self.fault_tolerance: Optional[FaultTolerance] = None
        self.distributed_coordinator: Optional[DistributedCoordinator] = None
        self.profile_analyzer: Optional[ProfileAnalyzer] = None
        self.gpu_monitor: Optional[GPUMonitor] = None
        self.moe_load_balancer: Optional[MoELoadBalancer] = None

        self.state = PipelineState.INITIALIZING
        self.current_stage = TrainingStage.PRETRAIN
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        self._running = False
        self._paused = False
        self._lock = threading.Lock()
        self._pipeline_thread: Optional[threading.Thread] = None

        self._stage_callbacks: Dict[str, List[Callable]] = {
            "on_stage_start": [],
            "on_stage_end": [],
            "on_step": [],
            "on_checkpoint": [],
            "on_failure": [],
            "on_completion": [],
        }

        if self.local_rank == 0:
            logger.info("=" * 60)
            logger.info("青小团训练管线编排器初始化")
            logger.info(f"青奈科技 - 模型: {config.model_name}")
            logger.info(f"青奈科技 - 配置: {config.output_dir}")
            logger.info("=" * 60)

    def _setup_logging(self):
        """设置日志系统"""
        if self.local_rank != 0:
            return

        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        logger.info(f"青小团 - 日志文件: {log_file}")

    def _init_components(self):
        """初始化所有基础设施组件"""
        if self.local_rank == 0:
            logger.info("青小团 - 开始初始化基础设施组件...")

        self.gpu_monitor = GPUMonitor()

        if self.config.moe_config or True:
            self.moe_load_balancer = MoELoadBalancer(
                num_experts=self.config.num_experts
            )

        ckpt_class = AsyncCheckpointManager if self.config.enable_async_checkpoint else CheckpointManager
        self.checkpoint_manager = ckpt_class(
            save_dir=str(self.checkpoint_dir),
            model_name=self.config.model_name,
            save_every_n_steps=self.config.checkpoint_every_n_steps,
            save_every_n_hours=self.config.checkpoint_every_n_hours,
            keep_last_n=self.config.keep_last_n_checkpoints,
            keep_best_k=self.config.keep_best_k_checkpoints,
            max_checkpoints=self.config.max_checkpoints,
            local_rank=self.local_rank,
            world_size=self.config.world_size,
        )

        self.monitor = TrainingMonitor(
            log_dir=str(self.log_dir),
            model_name=self.config.model_name,
            use_tensorboard=self.config.monitor_use_tensorboard,
            use_wandb=self.config.monitor_use_wandb,
            loss_spike_threshold=self.config.monitor_loss_spike_threshold,
            grad_norm_threshold=self.config.monitor_grad_norm_threshold,
            local_rank=self.local_rank,
        )

        self.experiment_tracker = ExperimentTracker(
            storage_dir=str(self.output_dir / "experiments"),
            experiment_name=self.config.experiment_name or f"{self.config.model_name}_训练",
            local_rank=self.local_rank,
        )
        self.experiment_tracker.update_config({
            "model": {
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "num_experts": self.config.num_experts,
                "top_k": self.config.top_k,
            },
            "training": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "total_steps": self.config.total_steps,
                "mixed_precision": self.config.mixed_precision,
            },
            "distributed": {
                "world_size": self.config.world_size,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "pipeline_parallel_size": self.config.pipeline_parallel_size,
            },
        })

        self.fault_tolerance = FaultTolerance(
            work_dir=str(self.output_dir / "fault_tolerance"),
            model_name=self.config.model_name,
            node_rank=self.node_rank,
            world_size=self.config.world_size,
            enable_heartbeat=self.config.enable_fault_tolerance,
            heartbeat_interval=self.config.heartbeat_interval,
            heartbeat_timeout=self.config.heartbeat_timeout,
            enable_auto_restart=self.config.enable_auto_restart,
            max_restart_attempts=self.config.max_restart_attempts,
            enable_elastic_training=self.config.enable_elastic_training,
        )

        self.distributed_coordinator = DistributedCoordinator(
            model_name=self.config.model_name,
            local_rank=self.local_rank,
            world_size=self.config.world_size,
            node_rank=self.node_rank,
            num_nodes=self.config.num_nodes,
            training_mode=TrainingMode(self.config.training_mode),
            communication_topology=CommunicationTopology(self.config.communication_topology),
            parallel_strategy=ParallelStrategy.DATA_PARALLEL,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        self.profile_analyzer = ProfileAnalyzer(
            model_name=self.config.model_name,
            output_dir=str(self.output_dir / "profiling"),
        )

        self.fault_tolerance.register_on_failure(self._on_node_failure)

        if self.local_rank == 0:
            logger.info("青小团 - 所有基础设施组件初始化完成")
            logger.info("青奈科技 - 检查点管理器 ✓")
            logger.info("青奈科技 - 训练监控器 ✓")
            logger.info("青奈科技 - 实验追踪器 ✓")
            logger.info("青奈科技 - 故障容错器 ✓")
            logger.info("青奈科技 - 分布式协调器 ✓")
            logger.info("青奈科技 - 性能分析器 ✓")

    def set_model_provider(self, provider: Callable):
        """设置模型提供函数"""
        self.model_provider = provider

    def set_data_provider(self, provider: Callable):
        """设置数据提供函数"""
        self.data_provider = provider

    def init_model(self):
        """初始化模型"""
        if self.model_provider is None:
            raise ValueError("青小团 - 未设置模型提供函数")

        self.state = PipelineState.MODEL_INIT
        if self.local_rank == 0:
            logger.info("青小团 - 开始初始化模型...")

        self.model = self.model_provider(self.config)

        if self.local_rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"青小团 - 模型初始化完成")
            logger.info(f"青奈科技 - 总参数量: {total_params:,}")
            logger.info(f"青奈科技 - 可训练参数量: {trainable_params:,}")

    def init_optimizer(self):
        """初始化优化器和学习率调度器"""
        if self.model is None:
            raise ValueError("青小团 - 模型未初始化")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )

        def warmup_cosine_schedule(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            progress = float(step - self.config.warmup_steps) / float(
                max(1, self.config.total_steps - self.config.warmup_steps)
            )
            return max(
                self.config.min_learning_rate / self.config.learning_rate,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_cosine_schedule
        )

        if self.local_rank == 0:
            logger.info(f"青小团 - 优化器初始化完成: {self.config.learning_rate} -> {self.config.min_learning_rate}")

    def init_data(self):
        """初始化数据"""
        if self.data_provider is None:
            raise ValueError("青小团 - 未设置数据提供函数")

        self.state = PipelineState.DATA_PREP
        if self.local_rank == 0:
            logger.info("青小团 - 开始初始化数据...")

        self.train_dataloader, self.eval_dataloader = self.data_provider(self.config)

        if self.local_rank == 0:
            if self.train_dataloader:
                logger.info(f"青小团 - 训练数据加载完成: {len(self.train_dataloader)} 批次")
            if self.eval_dataloader:
                logger.info(f"青小团 - 评估数据加载完成: {len(self.eval_dataloader)} 批次")

    def check_environment(self) -> bool:
        """检查训练环境"""
        self.state = PipelineState.ENV_CHECK
        checks_passed = True

        if self.local_rank == 0:
            logger.info("青小团 - 开始环境检查...")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"青奈科技 - GPU: {gpu_name} x{gpu_count}, 显存: {gpu_memory:.0f}GB")
        else:
            logger.error("青小团 - CUDA不可用")
            checks_passed = False

        try:
            from torch.cuda.amp import autocast
            logger.info("青奈科技 - 混合精度训练支持: 可用")
        except ImportError:
            logger.warning("青奈科技 - 混合精度训练: 不可用")

        if self.config.enable_fault_tolerance:
            logger.info(f"青奈科技 - 故障容错: 已启用 (心跳间隔={self.config.heartbeat_interval}s)")
        else:
            logger.info("青奈科技 - 故障容错: 未启用")

        if self.config.monitor_use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                logger.info("青奈科技 - TensorBoard: 可用")
            except ImportError:
                logger.warning("青奈科技 - TensorBoard: 未安装")

        return checks_passed

    def train(self):
        """启动训练管线"""
        if self._running:
            logger.warning("青小团 - 训练已在运行中")
            return

        self._running = True
        self._pipeline_thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self._pipeline_thread.start()

        if self.local_rank == 0:
            logger.info("青小团 - 训练管线已启动")

    def _run_pipeline(self):
        """运行训练管线主流程"""
        try:
            self.state = PipelineState.INITIALIZING

            if not self.check_environment():
                self.state = PipelineState.FAILED
                logger.error("青小团 - 环境检查失败")
                return

            self._init_components()

            self.experiment_tracker.start_experiment()

            resume_state = self.checkpoint_manager.get_resume_state()
            if resume_state["can_resume"]:
                logger.info(f"青小团 - 检测到断点，将从 step {resume_state['global_step']} 继续训练")
                self.global_step = resume_state["global_step"]
                self.fault_tolerance.load_latest_progress()

            if self.model is None:
                self.init_model()
            if self.optimizer is None:
                self.init_optimizer()
            if self.train_dataloader is None:
                self.init_data()

            if self.config.world_size > 1:
                self.state = PipelineState.DIST_SETUP
                self.distributed_coordinator.broadcast_parameters(self.model)

            self.state = PipelineState.TRAINING
            self._training_loop()

        except Exception as e:
            self.state = PipelineState.FAILED
            logger.error(f"青小团 - 训练管线异常: {e}")
            logger.error(traceback.format_exc())
            self.experiment_tracker.end_experiment(
                status="failed",
                notes=f"异常: {str(e)}",
            )
            self._cleanup()

    def _training_loop(self):
        """训练主循环"""
        stage_start_time = time.time()
        accumulation_loss = 0.0
        accumulation_steps = 0
        optimizer_step_count = 0

        num_batches = len(self.train_dataloader) if self.train_dataloader else float("inf")

        while self.global_step < self.config.total_steps and self._running:
            if self._paused:
                time.sleep(1.0)
                continue

            for batch in self.train_dataloader:
                if not self._running or self._paused:
                    break

                self.profile_analyzer.start_step(self.global_step)

                with torch.cuda.amp.autocast(
                    enabled=self.config.mixed_precision in ("fp16", "bf16"),
                    dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
                ):
                    self.profile_analyzer.mark_forward_start()
                    loss = self._forward_step(batch)
                    self.profile_analyzer.mark_backward_start()

                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                self.profile_analyzer.mark_optimizer_start()

                accumulation_loss += loss.item()
                accumulation_steps += 1

                if accumulation_steps >= self.config.gradient_accumulation_steps:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    if self.distributed_coordinator.should_sync_gradients(self.global_step):
                        comm_time = self.distributed_coordinator.sync_gradients(self.model)
                        self.profile_analyzer.mark_communication(comm_time)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    optimizer_step_count += 1
                    self.global_step += 1
                    avg_loss = accumulation_loss / accumulation_steps
                    current_lr = self.scheduler.get_last_lr()[0]

                    gpu_stats = self.gpu_monitor.get_total_memory_usage() if self.gpu_monitor else {}

                    self._on_training_step(avg_loss, grad_norm, current_lr, gpu_stats, stage_start_time)

                    accumulation_loss = 0.0
                    accumulation_steps = 0

                    if self.global_step >= self.config.total_steps:
                        break

        self._on_training_complete(stage_start_time)

    def _forward_step(self, batch) -> torch.Tensor:
        """前向传播步骤"""
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].cuda()
            labels = batch[1].cuda() if len(batch) > 1 else input_ids
        elif isinstance(batch, dict):
            input_ids = batch["input_ids"].cuda()
            labels = batch.get("labels", input_ids).cuda()
        else:
            input_ids = batch.cuda()
            labels = input_ids

        output = self.model(input_ids=input_ids, labels=labels)
        return output.loss

    def _on_training_step(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        gpu_stats: Dict[str, Any],
        start_time: float,
    ):
        """训练步回调"""
        elapsed = time.time() - start_time
        tokens_per_sec = (self.config.batch_size * self.config.gradient_accumulation_steps
                          * 2048 / max(elapsed / max(1, self.global_step), 1e-8))

        metrics = MonitorMetrics(
            step=self.global_step,
            epoch=self.current_epoch,
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=grad_norm,
            tokens_per_second=tokens_per_sec,
            gpu_utilization=gpu_stats.get("utilization_percent", 0.0),
            memory_used_gb=gpu_stats.get("total_allocated_gb", 0.0),
            memory_total_gb=gpu_stats.get("total_capacity_gb", 0.0),
            batch_time_ms=elapsed / max(1, self.global_step) * 1000,
        )

        self.monitor.log_metrics(metrics)
        self.experiment_tracker.log_metrics_to_experiment(
            step=self.global_step,
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
        )

        self.fault_tolerance.update_training_progress(
            global_step=self.global_step,
            epoch=self.current_epoch,
            loss=loss,
            best_loss=self.best_loss,
            learning_rate=learning_rate,
            loss_history=self.monitor.get_loss_history(),
        )

        self.profile_analyzer.end_step(
            loss=loss,
            gradient_norm=grad_norm,
            tokens_per_second=tokens_per_sec,
            gpu_utilization=gpu_stats.get("utilization_percent", 0.0),
            memory_used_gb=gpu_stats.get("total_allocated_gb", 0.0),
        )

        if loss < self.best_loss:
            self.best_loss = loss

        if self.checkpoint_manager.should_save(self.global_step, loss):
            self._save_checkpoint(loss, learning_rate)

        if self.global_step % 100 == 0:
            self.monitor.update_progress_bar(
                self.global_step,
                self.config.total_steps,
                {"loss": loss, "lr": learning_rate, "tok/s": tokens_per_sec},
            )

        for callback in self._stage_callbacks.get("on_step", []):
            try:
                callback(self.global_step, loss)
            except Exception:
                pass

    def _save_checkpoint(self, loss: float, learning_rate: float):
        """保存检查点"""
        self.state = PipelineState.CHECKPOINTING
        start_time = time.time()

        self.checkpoint_manager.save_checkpoint(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            global_step=self.global_step,
            epoch=self.current_epoch,
            loss=loss,
            learning_rate=learning_rate,
            moe_config={
                "num_experts": self.config.num_experts,
                "top_k": self.config.top_k,
            },
        )

        save_time = (time.time() - start_time) * 1000
        self.profile_analyzer.mark_checkpoint(save_time)

        for callback in self._stage_callbacks.get("on_checkpoint", []):
            try:
                callback(self.global_step)
            except Exception:
                pass

    def _on_training_complete(self, start_time: float):
        """训练完成回调"""
        self.state = PipelineState.COMPLETED
        total_time = time.time() - start_time

        if self.local_rank == 0:
            logger.info("=" * 60)
            logger.info("青小团训练完成！")
            logger.info(f"青奈科技 - 总步数: {self.global_step}")
            logger.info(f"青奈科技 - 最佳Loss: {self.best_loss:.6f}")
            logger.info(f"青奈科技 - 总训练时长: {str(timedelta(seconds=int(total_time)))}")
            logger.info("=" * 60)

        self.experiment_tracker.end_experiment(
            status="completed",
            notes=f"训练完成，共{self.global_step}步，最佳loss={self.best_loss:.6f}",
            result_updates={
                "final_loss": self.monitor.get_current_status().get("smoothed_loss", 0.0),
                "best_loss": self.best_loss,
                "final_step": self.global_step,
                "total_tokens_processed": self.monitor.get_current_status().get("total_tokens_processed", 0),
                "total_training_time_seconds": total_time,
            },
        )

        self._save_checkpoint(self.best_loss, self.scheduler.get_last_lr()[0])

        for callback in self._stage_callbacks.get("on_completion", []):
            try:
                callback()
            except Exception:
                pass

        self._cleanup()

    def _on_node_failure(self, failed_node: NodeInfo):
        """节点故障回调"""
        logger.warning(f"青小团 - 节点故障: {failed_node.node_id}")

        for callback in self._stage_callbacks.get("on_failure", []):
            try:
                callback(failed_node)
            except Exception:
                pass

    def _cleanup(self):
        """清理资源"""
        if self.local_rank == 0:
            logger.info("青小团 - 开始清理资源...")

        try:
            self.checkpoint_manager.save_checkpoint(
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                global_step=self.global_step,
                loss=self.best_loss,
                learning_rate=self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
                force_type="final",
            )
        except Exception:
            pass

        self.monitor.close()
        self.profile_analyzer.close()
        self.fault_tolerance.shutdown()
        self.distributed_coordinator.shutdown()
        self.experiment_tracker.close()
        self.checkpoint_manager.shutdown()

        self._running = False
        self.state = PipelineState.STOPPED

        if self.local_rank == 0:
            logger.info("青小团 - 训练管线已安全关闭")

    def pause(self):
        """暂停训练"""
        self._paused = True
        logger.info("青小团 - 训练已暂停")

    def resume(self):
        """恢复训练"""
        self._paused = False
        logger.info("青小团 - 训练已恢复")

    def stop(self):
        """停止训练"""
        self._running = False
        logger.info("青小团 - 训练停止中...")

    def get_status(self) -> Dict[str, Any]:
        """获取管线状态"""
        monitor_status = self.monitor.get_current_status() if self.monitor else {}
        cluster_health = self.fault_tolerance.get_cluster_health() if self.fault_tolerance else {}

        return {
            "company": "青奈科技",
            "model": self.config.model_name,
            "state": self.state.value,
            "stage": self.current_stage.value,
            "global_step": self.global_step,
            "total_steps": self.config.total_steps,
            "progress": f"{self.global_step / self.config.total_steps * 100:.1f}%" if self.config.total_steps > 0 else "0%",
            "best_loss": self.best_loss if self.best_loss < float("inf") else None,
            "running": self._running,
            "paused": self._paused,
            "experiment": self.experiment_tracker._current_experiment.experiment_name if self.experiment_tracker and self.experiment_tracker._current_experiment else "N/A",
            **monitor_status,
            "cluster": cluster_health,
        }

    def generate_summary_report(self) -> str:
        """生成训练总结报告"""
        monitor_report = self.monitor.generate_training_report() if self.monitor else ""
        profile_report = self.profile_analyzer.generate_profile_report() if self.profile_analyzer else ""
        fault_report = self.fault_tolerance.generate_fault_report() if self.fault_tolerance else ""

        report = f"""
{'='*60}
青小团训练总结报告 - 青奈科技
{'='*60}

训练配置:
  模型: {self.config.model_name}
  隐藏层: {self.config.hidden_size}
  Transformer层数: {self.config.num_layers}
  MoE专家数: {self.config.num_experts}
  学习率: {self.config.learning_rate}
  总步数: {self.config.total_steps}
  并行策略: DP={self.config.data_parallel_size}

训练结果:
  训练状态: {self.state.value}
  最终步数: {self.global_step}
  最优Loss: {self.best_loss:.6f if self.best_loss < float('inf') else 'N/A'}

基础设施报告:
{monitor_report}
{profile_report}
{fault_report}

{'='*60}
青奈科技 - 青小团分布式训练系统
报告时间: {datetime.now().isoformat()}
{'='*60}
"""
        report_path = self.output_dir / "training_summary_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"青小团 - 总结报告已保存: {report_path}")
        return report

    def register_callback(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self._stage_callbacks:
            self._stage_callbacks[event].append(callback)
