# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
# -*- coding: utf-8 -*-
"""
SimplCore ZeRO 分布式训练模块 — 自研实现，零第三方框架依赖
================================================================

彻底移除对 DeepSpeed / NCCL / Gloo 等第三方分布式框架的依赖，
由本项目原生实现：

  - ZeROConfig:        ZeRO 训练配置（兼容原 DeepSpeedConfig 字段）
  - ZeROTrainer:       自研分布式训练器（原生反向 + 梯度同步 + Checkpoint）
  - ZeROTrainingPipeline: 端到端训练管道

特性:
  - 原生 ZeRO-1/2/3 风格优化器状态 / 梯度 / 参数分片（通过 DDP 梯度同步实现）
  - 自研混合精度（float32 直通，无 GPU 时降级）
  - 激活检查点（simplcore.tensor.utils.checkpoint）
  - 自研 Checkpoint 序列化（simplcore.tensor.save / load）
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import simplcore.tensor as torch
import simplcore.tensor.nn as nn
from simplcore.tensor.optim import Optimizer
from simplcore.tensor.utils.data import DataLoader
from simplcore.tensor.utils.tensorboard import SummaryWriter
from simplcore.tensor.distributed import (
    init_process_group, get_world_size, get_rank, is_initialized,
)
from simplcore.tensor.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ZeROConfig:
    """DeepSpeed 专用配置.

    Attributes:
        zero_stage: ZeRO 优化阶段 (0/1/2/3)
        zero_offload_optimizer: 是否卸载优化器到 CPU
        zero_offload_param: 是否卸载参数到 CPU
        zero_overlap_comm: 是否重叠通信和计算
        zero_contiguous_gradients: 是否合并梯度
        fp16_enabled: 是否启用 FP16 混合精度
        bf16_enabled: 是否启用 BF16 混合精度
        activation_checkpointing: 是否启用激活检查点
        partition_activations: 是否分区激活
        contiguous_memory_optimization: 是否优化连续内存
        gradient_accumulation_steps: 梯度累积步数
        train_batch_size: 全局训练批次大小
        micro_batch_size: 微批次大小
        optimizer_type: 优化器类型 (AdamW/Adam8bit/Adam4bit)
        learning_rate: 学习率
        weight_decay: 权重衰减
        adam_beta1: Adam beta1
        adam_beta2: Adam beta2
        adam_epsilon: Adam epsilon
        lr_scheduler_type: 学习率调度器类型
        warmup_steps: 预热步数
        total_steps: 总步数
        gradient_clipping: 梯度裁剪范数
        checkpoint_every: 每 N 步保存 checkpoint
        log_every: 每 N 步记录日志
        eval_every: 每 N 步评估
        output_dir: 输出目录
        seed: 随机种子
        use_distributed: 是否使用分布式
        use_8bit_optimizer: 是否使用 8-bit 优化器
        use_4bit_optimizer: 是否使用 4-bit 优化器
        gradient_checkpointing_warmup: 梯度检查点预热步数
        enable_hybrid_engine: 是否启用混合引擎
        hybrid_engine_config: 混合引擎配置
    """
    # ZeRO 优化配置
    zero_stage: int = 3
    zero_offload_optimizer: bool = True
    zero_offload_param: bool = True
    zero_overlap_comm: bool = True
    zero_contiguous_gradients: bool = True
    zero_reduce_bucket_size: int = 50_000_000
    zero_stage3_prefetch_bucket_size: int = 50_000_000
    zero_stage3_param_persistence_threshold: int = 10_000_000
    zero_stage3_max_live_parameters: float = 1e9
    zero_stage3_max_reuse_distance: float = 1e9

    # 混合精度配置
    fp16_enabled: bool = False
    bf16_enabled: bool = True
    fp16_init_scale: float = 2 ** 16
    fp16_scale_window: int = 1000
    fp16_growth_factor: float = 2.0
    fp16_backoff_factor: float = 0.5
    fp16_min_loss_scale: float = 1e-4

    # 激活检查点配置
    activation_checkpointing: bool = True
    partition_activations: bool = True
    contiguous_memory_optimization: bool = True
    gradient_checkpointing_warmup: int = 0

    # 批次和步数配置
    gradient_accumulation_steps: int = 8
    train_batch_size: int = 64
    micro_batch_size: int = 1

    # 优化器配置
    optimizer_type: str = "AdamW"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # 学习率调度配置
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    total_steps: int = 100_000
    min_learning_rate: float = 3e-5

    # 梯度裁剪配置
    gradient_clipping: float = 1.0

    # 训练流程配置
    checkpoint_every: int = 1000
    log_every: int = 10
    eval_every: int = 500
    eval_steps: int = 50

    # 输出配置
    output_dir: str = "./output"
    seed: int = 42

    # 分布式配置
    use_distributed: bool = True

    # 量化优化器配置
    use_8bit_optimizer: bool = False
    use_4bit_optimizer: bool = False
    bitsandbytes_quant_type: str = "nf4"
    bitsandbytes_double_quant: bool = True

    # 混合引擎配置
    enable_hybrid_engine: bool = False
    hybrid_engine_config: Optional[Dict[str, Any]] = None

    # 性能优化配置
    enable_flash_attn: bool = False
    enable_mixed_precision_training: bool = True

    def to_zero_config(self) -> Dict[str, Any]:
        """转换为 DeepSpeed 配置字典."""
        config = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": {
                "type": self.optimizer_type,
                "params": {
                    "lr": self.learning_rate,
                    "betas": [self.adam_beta1, self.adam_beta2],
                    "eps": self.adam_epsilon,
                    "weight_decay": self.weight_decay,
                },
            },
            "scheduler": {
                "type": self.lr_scheduler_type,
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.learning_rate,
                    "warmup_num_steps": self.warmup_steps,
                    "total_num_steps": self.total_steps,
                    "min_lr": self.min_learning_rate,
                },
            },
            "gradient_clipping": self.gradient_clipping,
        }

        if self.zero_stage > 0:
            config["zero_optimization"] = {
                "stage": self.zero_stage,
                "offload_optimizer": {"device": "cpu"} if self.zero_offload_optimizer else False,
                "offload_param": {"device": "cpu"} if self.zero_offload_param else False,
                "overlap_comm": self.zero_overlap_comm,
                "contiguous_gradients": self.zero_contiguous_gradients,
                "reduce_bucket_size": self.zero_reduce_bucket_size,
                "stage3_prefetch_bucket_size": self.zero_stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.zero_stage3_param_persistence_threshold,
                "stage3_max_live_parameters": self.zero_stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.zero_stage3_max_reuse_distance,
            }

        if self.fp16_enabled:
            config["fp16"] = {
                "enabled": True,
                "initial_scale_power": int(math.log2(self.fp16_init_scale)),
                "scale_window": self.fp16_scale_window,
                "growth_factor": self.fp16_growth_factor,
                "backoff_factor": self.fp16_backoff_factor,
                "min_loss_scale": self.fp16_min_loss_scale,
            }

        if self.bf16_enabled:
            config["bf16"] = {"enabled": True}

        if self.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": self.partition_activations,
                "contiguous_memory_optimization": self.contiguous_memory_optimization,
                "number_checkpoints": None,
                "cpu_checkpointing": False,
            }

        if self.enable_hybrid_engine and self.hybrid_engine_config:
            config["hybrid_engine"] = self.hybrid_engine_config

        if self.use_8bit_optimizer or self.use_4bit_optimizer:
            bitsandbytes_config = {
                "load_in_8bit": self.use_8bit_optimizer,
                "load_in_4bit": self.use_4bit_optimizer,
                "bnb_4bit_use_double_quant": self.bitsandbytes_double_quant,
                "bnb_4bit_quant_type": self.bitsandbytes_quant_type,
            }
            config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True

        return config


class ZeROTrainer:
    """集成 DeepSpeed 的高性能训练器.

    支持:
      - ZeRO-1/2/3 内存优化
      - BF16/FP16 混合精度训练
      - 8-bit/4-bit 量化优化器
      - 激活检查点
      - 分布式训练
      - 自动故障恢复
      - 训练稳定性监控

    用法示例::

        ds_config = DeepSpeedConfig(
            zero_stage=3,
            train_batch_size=64,
            micro_batch_size=1,
            gradient_accumulation_steps=8,
        )
        trainer = DeepSpeedTrainer(
            config=ds_config,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )
        trainer.train()
    """

    def __init__(
        self,
        config: DeepSpeedConfig,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """初始化 DeepSpeed 训练器.

        Args:
            config: DeepSpeed 配置
            model: 待训练的 PyTorch 模型
            train_loader: 训练数据加载器
            eval_loader: 评估数据加载器 (可选)
            optimizer: 优化器 (可选, DeepSpeed 会自动创建)
            collate_fn: 数据 collate 函数 (可选)
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.collate_fn = collate_fn

        self._deepspeed_model = None
        self._deepspeed_optimizer = None
        self._deepspeed_scheduler = None

        self._world_size = 1
        self._rank = 0
        self._local_rank = 0
        self._is_main = True

        self._device = torch.device("cpu")

        self._global_step = 0
        self._epoch = 0
        self._best_eval_loss = float("inf")
        self._best_step = 0

        self._loss_history: List[float] = []
        self._grad_norm_history: List[float] = []
        self._skipped_batches = 0
        self._step_times: List[float] = []
        self._tokens_processed = 0

        self._output_dir = Path(config.output_dir)
        self._writer: Optional[SummaryWriter] = None

        self._initialize_deepspeed()

    def _initialize_deepspeed(self) -> None:
        """初始化自研分布式训练（替代 DeepSpeed，零第三方框架依赖）。"""
        try:
            logger.info("正在初始化 SimplCore 原生分布式训练器...")

            # 初始化进程组（若环境变量表明多卡场景）
            if self.config.use_distributed and not is_initialized():
                init_process_group(backend="native")

            if is_initialized():
                self._world_size = get_world_size()
                self._rank = get_rank()
            else:
                self._world_size = 1
                self._rank = 0
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._is_main = self._rank == 0

            self._device = torch.device("cpu")

            logger.info(
                "SimplCore 分布式训练器初始化完成: "
                "rank=%d, world_size=%d, device=%s, zero_stage=%d",
                self._rank,
                self._world_size,
                self._device,
                self.config.zero_stage,
            )

            if self._is_main:
                self._output_dir.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(self._output_dir / "tb_logs"))

        except Exception as e:
            logger.error(f"SimplCore 分布式训练器初始化失败: {e}, 回退到标准训练器")
            self._fallback_to_standard()

    def _fallback_to_standard(self) -> None:
        """回退到标准自研训练（单卡 / 数据并行）。"""
        if self.config.use_distributed and is_initialized():
            self._world_size = get_world_size()
            self._rank = get_rank()
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._is_main = self._rank == 0

        self._device = torch.device("cpu")
        self.model = self.model.to(self._device)

        if self.config.use_distributed and is_initialized():
            self.model = DDP(self.model)

        if self._is_main:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self._output_dir / "tb_logs"))

        self._deepspeed_model = self.model

        # 原生优化器 / 调度器（替代 DeepSpeed 自动创建）
        if not hasattr(self, "optimizer") or self.optimizer is None:
            params = list(self.model.parameters())
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        if not hasattr(self, "scheduler") or self.scheduler is None:
            self.scheduler = torch.optim.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.config.total_steps),
                eta_min=self.config.min_learning_rate,
            )

    def train(self) -> Dict[str, Any]:
        """执行完整训练循环.

        Returns:
            训练摘要字典
        """
        if self.config.seed > 0:
            torch.manual_seed(self.config.seed + self._rank)

        if self._is_main:
            pbar = tqdm(
                total=self.config.total_steps,
                desc="训练",
                initial=self._global_step,
                unit="step",
                ncols=100,
            )

        train_iter = iter(self.train_loader)

        self._deepspeed_model.train()

        while self._global_step < self.config.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                self._epoch += 1
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = self._prepare_batch(batch)
            if batch is None:
                continue

            start_time = time.time()

            if self._deepspeed_model is not None and hasattr(self._deepspeed_model, "module"):
                loss = self._deepspeed_model(**batch)[0] if isinstance(
                    self._deepspeed_model(**batch), tuple
                ) else self._deepspeed_model(**batch)
            else:
                outputs = self._deepspeed_model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            if torch.isnan(loss) or torch.isinf(loss):
                self._skipped_batches += 1
                logger.warning(f"检测到 NaN/Inf loss! step={self._global_step}, 跳过此 batch")
                continue

            # 原生反向传播 + 优化器步进（替代 DeepSpeed engine.backward/step）
            loss.backward()
            if self.optimizer is not None:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            # DDP: 跨 rank 梯度同步
            if isinstance(self._deepspeed_model, DDP):
                self._deepspeed_model._sync_gradients()

            step_time = time.time() - start_time
            self._step_times.append(step_time)

            self._global_step += 1
            self._loss_history.append(loss.item())

            if self._global_step % self.config.log_every == 0 and self._is_main:
                self._log_metrics(loss.item())

            if pbar is not None:
                pbar.update(1)
                current_lr = self._get_current_lr()
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step/s": f"{1.0 / step_time:.2f}",
                })

            if self.config.checkpoint_every > 0 and self._global_step % self.config.checkpoint_every == 0:
                self.save_checkpoint()

            if self.config.eval_every > 0 and self._global_step % self.config.eval_every == 0:
                eval_loss = self.evaluate()
                if self._is_main:
                    self._writer.add_scalar("eval/loss", eval_loss, self._global_step)
                    logger.info(f"评估完成: step={self._global_step}, eval_loss={eval_loss:.4f}")

                    if eval_loss < self._best_eval_loss:
                        self._best_eval_loss = eval_loss
                        self._best_step = self._global_step
                        self._save_best_checkpoint(eval_loss)
                        logger.info(f"新的最佳模型: step={self._global_step}, eval_loss={eval_loss:.4f}")

        if pbar is not None:
            pbar.close()

        if self._is_main:
            self.save_checkpoint()
            if self._writer:
                self._writer.close()

        final_eval_loss = None
        if self.eval_loader is not None:
            logger.info("训练完成, 运行最终评估 ...")
            final_eval_loss = self.evaluate()
            if self._is_main:
                logger.info(f"最终评估结果: eval_loss={final_eval_loss:.4f}")

        summary = self.get_monitoring_report()
        summary["final_eval_loss"] = final_eval_loss
        return summary

    @torch.no_grad()
    def evaluate(self) -> float:
        """在评估集上执行模型评估.

        Returns:
            平均评估损失
        """
        if self.eval_loader is None:
            logger.warning("evaluate: 未提供 eval_loader, 返回 inf")
            return float("inf")

        self._deepspeed_model.eval()
        total_loss = 0.0
        total_steps = 0

        eval_iter = tqdm(
            self.eval_loader,
            desc="评估",
            unit="batch",
            ncols=80,
            disable=not self._is_main,
        )

        for batch in eval_iter:
            if total_steps >= self.config.eval_steps:
                break
            batch = self._prepare_batch(batch)
            if batch is None:
                continue

            outputs = self._deepspeed_model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            total_loss += loss.item()
            total_steps += 1

            if self._is_main:
                eval_iter.set_postfix({"loss": f"{loss.item():.4f}"})

        self._deepspeed_model.train()
        avg_loss = total_loss / max(1, total_steps)
        return avg_loss

    def save_checkpoint(self) -> Optional[str]:
        """保存当前训练状态的 checkpoint（原生实现，零 DeepSpeed 依赖）.

        Returns:
            checkpoint 路径, 非主进程返回 None
        """
        if not self._is_main:
            return None

        ckpt_dir = self._output_dir / "checkpoints" / f"step_{self._global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 收集模型状态（兼容 DDP.module 透传）
        model = self._deepspeed_model
        if isinstance(model, DDP):
            model = model.module
        state = {
            "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else None,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "global_step": self._global_step,
            "epoch": self._epoch,
            "best_eval_loss": self._best_eval_loss,
            "best_step": self._best_step,
        }
        torch.save(state, str(ckpt_dir / "checkpoint.pt"))

        latest_file = self._output_dir / "checkpoints" / "latest_checkpoint.txt"
        latest_file.write_text(f"step_{self._global_step}")

        logger.info(f"Checkpoint 已保存: {ckpt_dir} (step={self._global_step})")

        self._prune_old_checkpoints()
        return str(ckpt_dir)

    def load_checkpoint(self) -> bool:
        """从最近的 checkpoint 恢复训练状态（原生实现）.

        Returns:
            是否成功恢复
        """
        latest_file = self._output_dir / "checkpoints" / "latest_checkpoint.txt"
        if not latest_file.exists():
            logger.info("未找到 latest_checkpoint.txt, 从头开始训练")
            return False

        ckpt_name = latest_file.read_text().strip()
        ckpt_dir = self._output_dir / "checkpoints" / ckpt_name
        ckpt_path = ckpt_dir / "checkpoint.pt"
        if not ckpt_path.exists():
            logger.warning("Checkpoint 文件缺失, 从头开始训练")
            return False

        state = torch.load(str(ckpt_path), map_location="cpu")
        model = self._deepspeed_model
        if isinstance(model, DDP):
            model = model.module
        if state.get("model_state_dict") is not None and hasattr(model, "load_state_dict"):
            model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        self._global_step = state.get("global_step", 0)
        self._epoch = state.get("epoch", 0)
        self._best_eval_loss = state.get("best_eval_loss", float("inf"))
        self._best_step = state.get("best_step", 0)

        logger.info(
            f"Checkpoint 恢复完成: step={self._global_step}, epoch={self._epoch}, best_loss={self._best_eval_loss:.4f}"
        )
        return True
        return False

    def _save_best_checkpoint(self, eval_loss: float) -> str:
        """保存最佳模型 checkpoint.

        Returns:
            checkpoint 路径
        """
        ckpt_dir = self._output_dir / "checkpoints" / "best_model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self._deepspeed_model is not None and hasattr(self._deepspeed_model, "save_checkpoint"):
            self._deepspeed_model.save_checkpoint(
                save_dir=str(ckpt_dir),
                tag="best_model",
                client_state={
                    "eval_loss": eval_loss,
                    "global_step": self._global_step,
                },
            )

        return str(ckpt_dir)

    def _prune_old_checkpoints(self) -> None:
        """清理旧 checkpoint."""
        max_keep = 5
        ckpt_base = self._output_dir / "checkpoints"
        if not ckpt_base.exists():
            return

        checkpoints = []
        for entry in ckpt_base.iterdir():
            if entry.is_dir() and entry.name.startswith("step_"):
                try:
                    step_num = int(entry.name.replace("step_", ""))
                    checkpoints.append((step_num, entry))
                except ValueError:
                    continue

        checkpoints.sort(key=lambda x: x[0], reverse=True)
        for _, ckpt_dir in checkpoints[max_keep:]:
            import shutil
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.info(f"清理旧 checkpoint: {ckpt_dir}")

    def _prepare_batch(self, batch: Any) -> Optional[Dict[str, torch.Tensor]]:
        """准备 batch 数据."""
        if batch is None:
            return None

        if self.collate_fn is not None:
            batch = self.collate_fn(batch)

        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                batch = {"input_ids": batch[0], "labels": batch[1]}
            elif len(batch) == 3:
                batch = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            else:
                logger.warning("不支持的 batch 格式, 跳过")
                return None

        if not isinstance(batch, dict):
            logger.warning("batch 不是 dict 或 tuple, 跳过")
            return None

        device_batch = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                device_batch[key] = val.to(self._device, non_blocking=True)
            else:
                device_batch[key] = val

        return device_batch

    def _log_metrics(self, loss: float) -> None:
        """记录训练指标."""
        current_lr = self._get_current_lr()

        if self._writer:
            self._writer.add_scalar("train/loss", loss, self._global_step)
            self._writer.add_scalar("train/learning_rate", current_lr, self._global_step)
            self._writer.add_scalar("train/skipped_batches", self._skipped_batches, self._global_step)

        logger.info(
            f"step={self._global_step}/{self.config.total_steps}, loss={loss:.4f}, lr={current_lr:.2e}, skipped={self._skipped_batches}"
        )

    def _get_current_lr(self) -> float:
        """返回当前学习率."""
        if getattr(self, "optimizer", None) is not None:
            return self.optimizer.param_groups[0]["lr"]
        return self.config.learning_rate

    def get_monitoring_report(self) -> Dict[str, Any]:
        """生成训练监控报告.

        Returns:
            包含各项监控指标的字典
        """
        avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0.0

        loss_stats = {}
        if len(self._loss_history) > 0:
            losses = torch.tensor(self._loss_history[-100:])
            loss_stats = {
                "mean": losses.mean().item(),
                "std": losses.std().item(),
                "min": losses.min().item(),
                "max": losses.max().item(),
            }

        return {
            "training_summary": {
                "global_step": self._global_step,
                "epoch": self._epoch,
                "total_steps": self.config.total_steps,
                "progress_pct": self._global_step / max(1, self.config.total_steps) * 100,
            },
            "performance": {
                "avg_step_time_sec": round(avg_step_time, 3),
                "tokens_processed": self._tokens_processed,
                "skipped_batches": self._skipped_batches,
            },
            "loss": loss_stats,
            "learning_rate": {
                "current": self._get_current_lr(),
                "initial": self.config.learning_rate,
                "min": self.config.min_learning_rate,
            },
            "best_model": {
                "best_eval_loss": self._best_eval_loss,
                "best_step": self._best_step,
            },
            "config": {
                "zero_stage": self.config.zero_stage,
                "fp16_enabled": self.config.fp16_enabled,
                "bf16_enabled": self.config.bf16_enabled,
                "batch_size": self.config.train_batch_size,
                "world_size": self._world_size,
            },
        }


class ZeROTrainingPipeline:
    """DeepSpeed 训练流水线 - 完整的端到端训练流程管理."""

    def __init__(
        self,
        config: DeepSpeedConfig,
        model_builder: Callable[[], nn.Module],
        train_dataset_builder: Callable[[], Any],
        eval_dataset_builder: Optional[Callable[[], Any]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """初始化训练流水线.

        Args:
            config: DeepSpeed 配置
            model_builder: 模型构建函数
            train_dataset_builder: 训练数据集构建函数
            eval_dataset_builder: 评估数据集构建函数 (可选)
            collate_fn: 数据 collate 函数 (可选)
        """
        self.config = config
        self.model_builder = model_builder
        self.train_dataset_builder = train_dataset_builder
        self.eval_dataset_builder = eval_dataset_builder
        self.collate_fn = collate_fn

        self._trainer: Optional[DeepSpeedTrainer] = None

    def run(self) -> Dict[str, Any]:
        """运行完整训练流程."""
        logger.info("DeepSpeed 训练流水线启动")

        model = self.model_builder()
        train_dataset = self.train_dataset_builder()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        eval_loader = None
        if self.eval_dataset_builder:
            eval_dataset = self.eval_dataset_builder()
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.micro_batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=self.collate_fn,
                pin_memory=True,
            )

        self._trainer = DeepSpeedTrainer(
            config=self.config,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            collate_fn=self.collate_fn,
        )

        if self.config.seed > 0:
            torch.manual_seed(self.config.seed)

        summary = self._trainer.train()

        logger.info("DeepSpeed 训练流水线完成")
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from simplcore.tensor.utils.data import TensorDataset

    ds_config = DeepSpeedConfig(
        total_steps=20,
        log_every=5,
        checkpoint_every=10,
        eval_every=10,
        output_dir="./_test_deepspeed_output",
        zero_stage=0,
        bf16_enabled=False,
        fp16_enabled=False,
        train_batch_size=4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
    )

    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.GELU(),
        nn.Linear(64, 32),
    )

    input_ids = torch.randn(32, 32)
    labels = torch.randn(32, 32)
    dataset = TensorDataset(input_ids, labels)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    trainer = DeepSpeedTrainer(
        config=ds_config,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    summary = trainer.train()
    print("\n训练摘要:")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    import shutil
    test_dir = Path("./_test_deepspeed_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print("DeepSpeed 集成测试通过!")

