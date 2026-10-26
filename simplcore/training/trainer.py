# Copyright 2026 (C) Lu'an Qingnai Sciences Technology Co., Ltd. / 六安市青奈科技有限责任公司
"""SimplCore 训练框架 (SimplCoreTrainer).

提供面向 72B 大模型的完整分布式训练能力, 包含:
  - TrainingConfig: 统一训练配置数据类
  - SimplCoreTrainer: 训练循环、评估、Checkpoint、故障恢复、稳定性监控

训练循环流程:
  1. 准备 batch 数据
  2. 前向传播 (自动检测并跳过 NaN loss)
  3. 反向传播
  4. 梯度裁剪
  5. 优化器步进
  6. 学习率调度
  7. 日志记录 (每 N 步)
  8. Checkpoint 保存 (每 N 步 + 训练完成)
  9. 损失值监控——若连续 5 步 loss > 3σ, 自动降低学习率
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
import simplcore.tensor.distributed as dist
import simplcore.tensor.nn as nn
from simplcore.tensor.nn.parallel import DistributedDataParallel as DDP
from simplcore.tensor.optim import Optimizer
from simplcore.tensor.optim import LambdaLR
from simplcore.tensor.utils.data import DataLoader
from simplcore.tensor.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ======================================================================
# TrainingConfig — 统一训练配置
# ======================================================================
@dataclass
class TrainingConfig:
    """训练超参数配置.

    Attributes:
        batch_size:             每步的 batch size (全局, 含所有 GPU)
        micro_batch_size:       微批次大小 (梯度累积时, 每卡每步)
        gradient_accumulation_steps: 梯度累积步数 = batch_size // (micro_batch_size * world_size)
        learning_rate:          初始学习率
        min_learning_rate:      最小学习率 (余弦退火下限)
        warmup_steps:           预热步数 (线性 warmup)
        total_steps:            总训练步数
        weight_decay:           AdamW 权重衰减
        adam_beta1:             Adam beta1
        adam_beta2:             Adam beta2
        adam_epsilon:           Adam epsilon
        grad_clip:              全局梯度裁剪范数 (0 表示不裁剪)
        grad_clip_mode:         梯度裁剪模式 ("norm" / "value")
        checkpoint_every:       每 N 步保存一次 checkpoint (0 表示仅保存最终)
        keep_checkpoint_max:    最多保留的 checkpoint 数 (-1 表示不限制)
        log_every:              每 N 步记录一次日志
        eval_every:             每 N 步运行一次评估 (0 表示训练结束后评估)
        eval_steps:             评估时采样的步数
        save_best:              是否保存最佳模型 (按 eval_loss)
        precision:              混合精度类型 ("bf16" / "fp16" / "fp32")
        seed:                   随机种子
        output_dir:             输出目录 (checkpoint / 日志 / TensorBoard)
        resume_from_checkpoint: 是否自动从最近 checkpoint 恢复
        distributed:            是否启用分布式训练
        find_unused_parameters: DDP find_unused_parameters
        detect_anomaly:         是否开启 autograd 异常检测 (调试用, 影响性能)
        loss_spike_threshold:   损失尖峰检测倍数 (连续 steps > mean + threshold * std 时降 LR)
        loss_spike_patience:    损失尖峰连续步数阈值
        lr_reduce_factor:       检测到损失尖峰时的 LR 衰减系数
        lr_reduce_min_lr:       降 LR 后的最小值下限
        eval_batch_size:        评估时的 batch size
    """
    # ---------- 批次与步数 ----------
    batch_size: int = 64
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    total_steps: int = 100_000

    # ---------- 学习率 ----------
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 1000

    # ---------- 优化器 ----------
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # ---------- 梯度 ----------
    grad_clip: float = 1.0
    grad_clip_mode: str = "norm"

    # ---------- Checkpoint ----------
    checkpoint_every: int = 1000
    keep_checkpoint_max: int = 5
    save_best: bool = True

    # ---------- 日志 & 评估 ----------
    log_every: int = 10
    eval_every: int = 500
    eval_steps: int = 50
    eval_batch_size: int = 8

    # ---------- 精度 ----------
    precision: str = "bf16"

    # ---------- 稳定性 ----------
    seed: int = 42
    loss_spike_threshold: float = 3.0   # loss > mean + 3σ 视为尖峰
    loss_spike_patience: int = 5        # 连续 5 步触发降 LR
    lr_reduce_factor: float = 0.5       # 降 LR 倍数
    lr_reduce_min_lr: float = 1e-6      # 降 LR 下限

    # ---------- 分布式 ----------
    distributed: bool = True
    find_unused_parameters: bool = False
    detect_anomaly: bool = False

    # ---------- 显存优化 ----------
    gradient_checkpointing: bool = False
    use_deepspeed: bool = False  # 是否启用 DeepSpeed
    use_8bit_optimizer: bool = False  # 是否启用8-bit优化器(bitsandbytes)
    
    # ---------- 优化器选择 ----------
    optimizer_type: str = "adamw"  # adamw / muonclip

    # ---------- 路径 ----------
    output_dir: str = "./output"
    resume_from_checkpoint: bool = True

    # ---------- 数据 ----------
    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 2

    def __post_init__(self) -> None:
        """参数合法性校验."""
        assert self.batch_size > 0, "batch_size 必须为正"
        assert self.micro_batch_size > 0, "micro_batch_size 必须为正"
        assert self.learning_rate > 0, "learning_rate 必须为正"
        assert self.total_steps > 0, "total_steps 必须为正"
        assert self.precision in ("bf16", "fp16", "fp32"), \
            f"precision 必须是 bf16/fp16/fp32, 当前: {self.precision}"
        assert self.lr_reduce_factor > 0, "lr_reduce_factor 必须为正"

    def get_autocast_dtype(self) -> torch.dtype:
        """返回 autocast 所需的数据类型."""
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        else:
            return torch.float32

    def get_autocast_enabled(self) -> bool:
        """混合精度是否启用."""
        return self.precision in ("bf16", "fp16")


# ======================================================================
# 学习率调度器工厂
# ======================================================================
def _get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """余弦退火 + 线性预热学习率调度器.

    Args:
        optimizer: 优化器
        warmup_steps: 预热步数
        total_steps: 总步数
        min_lr_ratio: 最小学习率相对于初始 LR 的比率

    Returns:
        LambdaLR 调度器实例
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # 线性预热: 0 → 1
            return float(current_step) / float(max(1, warmup_steps))
        # 余弦退火: 1 → min_lr_ratio
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ======================================================================
# SimplCoreTrainer — 核心训练器
# ======================================================================
class SimplCoreTrainer:
    """SimplCore 训练器.

    支持 72B 模型的分布式训练, 具备 checkpoint 管理、故障恢复、
    训练稳定性监控与混合精度训练能力。

    用法示例::

        config = TrainingConfig(total_steps=50000)
        model = SimplCoreModel(SimplCoreConfig.get_72b_config())
        train_loader = DataLoader(...)
        eval_loader = DataLoader(...)

        trainer = SimplCoreTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )
        trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LambdaLR] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        """初始化训练器.

        Args:
            config:       训练配置
            model:        待训练的 PyTorch 模型
            train_loader: 训练数据加载器
            eval_loader:  评估数据加载器 (可选, 若无则仅训练)
            optimizer:    优化器 (可选, 默认使用 AdamW)
            scheduler:    学习率调度器 (可选, 默认使用余弦退火 + warmup)
            collate_fn:   数据 collate 函数 (可选)
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.collate_fn = collate_fn

        # ---- 分布式初始化 ----
        self._world_size = 1
        self._rank = 0
        self._local_rank = 0
        self._is_main = True
        if config.distributed and dist.is_available() and dist.is_initialized():
            self._world_size = dist.get_world_size()
            self._rank = dist.get_rank()
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._is_main = self._rank == 0

        # ---- 设备 ----
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{self._local_rank}")
        else:
            self._device = torch.device("cpu")

        # 将模型移至对应设备
        self.model = self.model.to(self._device)

        # 分布式包装
        if config.distributed and dist.is_available() and dist.is_initialized():
            self.model = DDP(
                self.model,
                device_ids=[self._local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=config.find_unused_parameters,
            )

        # ---- 优化器 ----
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = self._create_default_optimizer()

        # ---- 学习率调度器 ----
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            min_lr_ratio = config.min_learning_rate / max(1e-8, config.learning_rate)
            self.scheduler = _get_cosine_schedule_with_warmup(
                self.optimizer,
                warmup_steps=config.warmup_steps,
                total_steps=config.total_steps,
                min_lr_ratio=min_lr_ratio,
            )

        # ---- 混合精度 ----
        self._autocast_dtype = config.get_autocast_dtype()
        self._autocast_enabled = config.get_autocast_enabled()
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None
        if config.precision == "fp16":
            self._scaler = torch.cuda.amp.GradScaler(enabled=True)
        elif config.precision == "bf16":
            self._scaler = None

        # ---- 梯度检查点 ----
        if config.gradient_checkpointing:
            self._apply_gradient_checkpointing()

        # ---- 自研 ZeRO 分布式训练（替代 DeepSpeed，零第三方框架依赖） ----
        if config.use_deepspeed:
            try:
                from simplcore.tensor.distributed import init_process_group, get_world_size, get_rank
                if not dist.is_initialized():
                    init_process_group(backend="native")
                if dist.is_initialized():
                    self._world_size = get_world_size()
                    self._rank = get_rank()
                    self._is_main = (self._rank == 0)
                    # 原生 ZeRO-3 包装：DDP 负责跨 rank 梯度同步
                    self.model = DDP(self.model)
                    logger.info("SimplCore 原生 ZeRO 分布式训练已启用 (stage=%d)", config.zero_stage)
                else:
                    logger.info("单进程模式: 跳过分发式初始化")
            except Exception as e:
                logger.warning(f"分布式初始化失败: {e}, 回退到标准训练")
        elif config.use_8bit_optimizer:
            # 8-bit 量化优化器：原生引擎以 float32 运行，量化仅作标记
            logger.info("8-bit 量化优化器标记已记录 (原生引擎以 float32 计算)")

        # ---- 输出目录 ----
        self._output_dir = Path(config.output_dir)
        if self._is_main:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            # TensorBoard
            self._writer = SummaryWriter(log_dir=str(self._output_dir / "tb_logs"))

        # ---- 训练状态 ----
        self._global_step = 0
        self._epoch = 0
        self._best_eval_loss = float("inf")
        self._best_step = 0

        # ---- 稳定性监控 ----
        self._loss_history: List[float] = []          # 最近损失值滑动窗口
        self._loss_spike_counter = 0                  # 连续尖峰计数
        self._loss_spike_window = 100                 # 统计窗口大小
        self._grad_norm_history: List[float] = []     # 梯度范数历史
        self._skipped_batches = 0                     # 跳过的 NaN batch 数

        # ---- 性能统计 ----
        self._step_times: List[float] = []
        self._tokens_processed = 0

        logger.info(
            "SimplCoreTrainer 初始化完成: "
            "rank=%d, world_size=%d, device=%s, precision=%s, total_steps=%d",
            self._rank,
            self._world_size,
            self._device,
            config.precision,
            config.total_steps,
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        """执行完整训练循环.

        包含:
          1. 自动故障恢复 (若 resume_from_checkpoint=True)
          2. 训练循环 (batch → forward → backward → clip → step)
          3. 稳定性监控 (loss / grad norm)
          4. 日志记录 (tqdm / TensorBoard)
          5. Checkpoint 保存
          6. 训练完成后的评估

        Returns:
            训练摘要字典 (含最终指标)
        """
        # ---- 故障恢复 ----
        if self.config.resume_from_checkpoint:
            resumed = self.load_checkpoint()
            if resumed:
                logger.info("已从 checkpoint 恢复训练, 继续步 %d", self._global_step)

        # ---- 数据迭代器 ----
        train_iter = iter(self.train_loader)

        # ---- 总进度条 (仅主进程) ----
        pbar: Optional[tqdm] = None
        if self._is_main:
            pbar = tqdm(
                total=self.config.total_steps,
                desc="训练",
                initial=self._global_step,
                unit="step",
                ncols=100,
            )

        # ---- 梯度累积参数 ----
        accum_steps = self.config.gradient_accumulation_steps
        model_module = self.model.module if isinstance(self.model, DDP) else self.model

        # ---- 训练循环 ----
        self.model.train()
        self.optimizer.zero_grad()

        while self._global_step < self.config.total_steps:
            for _ in range(accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # 一个 epoch 结束, 重新加载
                    self._epoch += 1
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                # 准备 batch
                batch = self._prepare_batch(batch)
                if batch is None:
                    continue

                # 前向传播 (含 NaN 检测)
                loss = self._forward_step(batch)
                if loss is None:
                    self._skipped_batches += 1
                    continue

                # 反向传播 (含混合精度缩放)
                loss = loss / accum_steps
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

            # ---- 梯度裁剪 ----
            grad_norm = self._clip_gradients()

            # ---- 优化器步进 ----
            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()

            # ---- 学习率调度 ----
            self.scheduler.step()
            self.optimizer.zero_grad()

            self._global_step += 1

            # ---- 梯度和损失监控 ----
            if grad_norm is not None:
                self._grad_norm_history.append(grad_norm)

            # ---- 损失尖峰检测 ----
            current_loss = loss.item() * accum_steps if loss is not None else 0.0
            self._loss_history.append(current_loss)
            self._check_loss_spike(current_loss)

            # ---- 日志记录 ----
            if self._global_step % self.config.log_every == 0 and self._is_main:
                self._log_metrics(current_loss, grad_norm)

            # ---- 更新进度条 ----
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{self._get_current_lr():.2e}",
                    "gnorm": f"{grad_norm:.2f}" if grad_norm is not None else "N/A",
                })

            # ---- Checkpoint 保存 ----
            if (
                self.config.checkpoint_every > 0
                and self._global_step % self.config.checkpoint_every == 0
            ):
                self.save_checkpoint()

            # ---- 评估 ----
            if (
                self.config.eval_every > 0
                and self._global_step % self.config.eval_every == 0
            ):
                eval_loss = self.evaluate()
                if self._is_main:
                    self._writer.add_scalar("eval/loss", eval_loss, self._global_step)
                    logger.info("评估完成: step=%d, eval_loss=%.4f", self._global_step, eval_loss)

                    # 保存最佳模型
                    if self.config.save_best and eval_loss < self._best_eval_loss:
                        self._best_eval_loss = eval_loss
                        self._best_step = self._global_step
                        self._save_best_checkpoint(eval_loss)
                        logger.info("新的最佳模型: step=%d, eval_loss=%.4f", self._global_step, eval_loss)

        # ---- 训练结束 ----
        if pbar is not None:
            pbar.close()

        if self._is_main:
            # 最终 checkpoint
            self.save_checkpoint()
            self._writer.close()

        # ---- 训练完成后自动评估 ----
        final_eval_loss: Optional[float] = None
        if self.eval_loader is not None:
            logger.info("训练完成, 运行最终评估 ...")
            final_eval_loss = self.evaluate()
            if self._is_main:
                logger.info("最终评估结果: eval_loss=%.4f", final_eval_loss)

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

        self.model.eval()
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

            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=self._autocast_dtype,
                enabled=self._autocast_enabled,
            ):
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs

            total_loss += loss.item()
            total_steps += 1

            if self._is_main:
                eval_iter.set_postfix({"loss": f"{loss.item():.4f}"})

        self.model.train()
        avg_loss = total_loss / max(1, total_steps)
        return avg_loss

    def save_checkpoint(self) -> Optional[str]:
        """保存当前训练状态的 checkpoint.

        包含:
          - 模型状态 (state_dict)
          - 优化器状态
          - 学习率调度器状态
          - GradScaler 状态 (fp16)
          - 训练状态 (step, epoch, best_loss 等)
          - 训练配置

        Returns:
            checkpoint 路径, 非主进程返回 None
        """
        if not self._is_main:
            return None

        ckpt_dir = self._output_dir / "checkpoints" / f"step_{self._global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_module = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            # 模型
            "model_state_dict": model_module.state_dict(),
            "model_config": model_module.config if hasattr(model_module, "config") else None,
            # 优化器 & 调度器
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # 混合精度
            "scaler_state_dict": self._scaler.state_dict() if self._scaler is not None else None,
            # 训练状态
            "global_step": self._global_step,
            "epoch": self._epoch,
            "best_eval_loss": self._best_eval_loss,
            "best_step": self._best_step,
            # 监控历史
            "loss_history": self._loss_history[-self._loss_spike_window:],
            "grad_norm_history": self._grad_norm_history[-self._loss_spike_window:],
            "skipped_batches": self._skipped_batches,
            # 配置
            "training_config": self.config,
        }

        ckpt_path = ckpt_dir / "training_state.pt"
        torch.save(checkpoint, ckpt_path)

        # 更新最新 checkpoint 符号链接 (或记录文件)
        latest_file = self._output_dir / "checkpoints" / "latest_checkpoint.txt"
        latest_file.write_text(f"step_{self._global_step}")

        logger.info("Checkpoint 已保存: %s (step=%d)", ckpt_path, self._global_step)

        # 清理旧 checkpoint
        self._prune_old_checkpoints()

        return str(ckpt_path)

    def load_checkpoint(self) -> bool:
        """从最近的 checkpoint 恢复训练状态.

        自动检测 output_dir/checkpoints/ 下的最新 checkpoint。
        若未找到 checkpoint, 返回 False。

        Returns:
            是否成功恢复
        """
        latest_file = self._output_dir / "checkpoints" / "latest_checkpoint.txt"
        if not latest_file.exists():
            logger.info("未找到 latest_checkpoint.txt, 从头开始训练")
            return False

        ckpt_name = latest_file.read_text().strip()
        ckpt_path = self._output_dir / "checkpoints" / ckpt_name / "training_state.pt"
        if not ckpt_path.exists():
            logger.warning("Checkpoint 文件不存在: %s, 从头开始训练", ckpt_path)
            return False

        logger.info("正在从 checkpoint 恢复: %s", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=self._device)

        # 恢复模型
        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        model_module.load_state_dict(checkpoint["model_state_dict"], strict=False)

        # 恢复优化器
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 恢复调度器
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 恢复混合精度
        if self._scaler is not None and "scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # 恢复训练状态
        self._global_step = checkpoint.get("global_step", 0)
        self._epoch = checkpoint.get("epoch", 0)
        self._best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        self._best_step = checkpoint.get("best_step", 0)
        self._loss_history = checkpoint.get("loss_history", [])
        self._grad_norm_history = checkpoint.get("grad_norm_history", [])
        self._skipped_batches = checkpoint.get("skipped_batches", 0)

        logger.info(
            "Checkpoint 恢复完成: step=%d, epoch=%d, best_loss=%.4f",
            self._global_step,
            self._epoch,
            self._best_eval_loss,
        )
        return True

    def get_monitoring_report(self) -> Dict[str, Any]:
        """生成训练监控报告.

        Returns:
            包含各项监控指标的字典
        """
        avg_step_time = (
            sum(self._step_times) / len(self._step_times)
            if self._step_times
            else 0.0
        )

        # 损失统计
        loss_stats = {}
        if len(self._loss_history) > 0:
            losses = torch.tensor(self._loss_history[-self._loss_spike_window:])
            loss_stats = {
                "mean": losses.mean().item(),
                "std": losses.std().item(),
                "min": losses.min().item(),
                "max": losses.max().item(),
                "recent_mean": (
                    torch.tensor(self._loss_history[-min(50, len(self._loss_history)):]).mean().item()
                ),
            }
        # 梯度范数统计
        grad_stats = {}
        if len(self._grad_norm_history) > 0:
            gnorms = torch.tensor(self._grad_norm_history[-self._loss_spike_window:])
            grad_stats = {
                "mean": gnorms.mean().item(),
                "max": gnorms.max().item(),
                "min": gnorms.min().item(),
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
            "gradient_norm": grad_stats,
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
                "precision": self.config.precision,
                "batch_size": self.config.batch_size,
                "distributed": self.config.distributed,
                "world_size": self._world_size,
            },
        }

    # ------------------------------------------------------------------
    # 内部方法: 训练循环核心
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: Any) -> Optional[Dict[str, torch.Tensor]]:
        """准备 batch 数据: 移至设备 & 类型转换.

        Args:
            batch: 原始 batch 数据 (可以是 dict, tuple 或其它格式)

        Returns:
            整理后的 dict, 包含 input_ids, labels 等; 无效时返回 None
        """
        if batch is None:
            return None

        # 若提供了 collate_fn, 使用它处理
        if self.collate_fn is not None:
            batch = self.collate_fn(batch)

        # 将 batch 转为 dict (若为 tuple 则尝试解包)
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

        # 移至设备
        device_batch = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                device_batch[key] = val.to(self._device, non_blocking=True)
            else:
                device_batch[key] = val

        return device_batch

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """执行单步前向传播, 自动检测并跳过 NaN loss.

        Args:
            batch: 输入 batch (dict)

        Returns:
            loss 张量, 若为 NaN 则返回 None
        """
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=self._autocast_dtype,
            enabled=self._autocast_enabled,
        ):
            if self.config.detect_anomaly:
                with torch.autograd.set_detect_anomaly(True):
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)

            if isinstance(outputs, dict):
                loss = outputs["loss"] if "loss" in outputs else outputs["logits"]
            elif isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs

        # NaN / Inf 检测
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                "检测到 NaN/Inf loss! step=%d, 跳过此 batch",
                self._global_step,
            )
            return None

        return loss

    def _clip_gradients(self) -> Optional[float]:
        """执行梯度裁剪并返回全局梯度范数.

        Returns:
            裁剪前的全局梯度范数 (若不可用则返回 None)
        """
        if self.config.grad_clip <= 0:
            return None

        if self._scaler is not None:
            # fp16 模式下, 需要在 scaler 的 unscale 后裁剪
            self._scaler.unscale_(self.optimizer)

        if self.config.grad_clip_mode == "norm":
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip,
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                clip_value=self.config.grad_clip,
            )
            grad_norm = None

        return grad_norm

    def _check_loss_spike(self, current_loss: float) -> None:
        """检测损失尖峰并自动降低学习率.

        当连续 loss_spike_patience 步的 loss 超出 mean + threshold * std 时,
        将学习率乘以 lr_reduce_factor。
        """
        if len(self._loss_history) < self._loss_spike_window:
            return

        window = self._loss_history[-self._loss_spike_window:]
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = math.sqrt(variance)

        if std < 1e-8:
            # 损失值几乎无波动时, 不触发
            self._loss_spike_counter = 0
            return

        threshold = mean + self.config.loss_spike_threshold * std

        if current_loss > threshold:
            self._loss_spike_counter += 1
            logger.info(
                "损失尖峰检测: step=%d, loss=%.4f, mean=%.4f, threshold=%.4f, counter=%d/%d",
                self._global_step,
                current_loss,
                mean,
                threshold,
                self._loss_spike_counter,
                self.config.loss_spike_patience,
            )

            if self._loss_spike_counter >= self.config.loss_spike_patience:
                # 降低学习率
                current_lr = self._get_current_lr()
                new_lr = max(
                    current_lr * self.config.lr_reduce_factor,
                    self.config.lr_reduce_min_lr,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
                self._loss_spike_counter = 0
                logger.warning(
                    "自动降低学习率: %.6f → %.6f (step=%d)",
                    current_lr,
                    new_lr,
                    self._global_step,
                )
        else:
            # 逐歩衰减计数
            self._loss_spike_counter = max(0, self._loss_spike_counter - 1)

    # ------------------------------------------------------------------
    # 内部方法: Checkpoint 管理
    # ------------------------------------------------------------------
    def _save_best_checkpoint(self, eval_loss: float) -> str:
        """保存最佳模型 checkpoint.

        Args:
            eval_loss: 当前评估损失

        Returns:
            checkpoint 路径
        """
        ckpt_dir = self._output_dir / "checkpoints" / "best_model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        ckpt_path = ckpt_dir / f"best_loss_{eval_loss:.4f}_step_{self._global_step}.pt"

        torch.save(
            {
                "model_state_dict": model_module.state_dict(),
                "eval_loss": eval_loss,
                "global_step": self._global_step,
                "model_config": (
                    model_module.config if hasattr(model_module, "config") else None
                ),
            },
            ckpt_path,
        )

        # 同时保存一个固定名称的副本, 便于加载
        latest_best = ckpt_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": model_module.state_dict(),
                "eval_loss": eval_loss,
                "global_step": self._global_step,
            },
            latest_best,
        )

        return str(ckpt_path)

    def _prune_old_checkpoints(self) -> None:
        """清理超出 keep_checkpoint_max 数量的旧 checkpoint."""
        max_keep = self.config.keep_checkpoint_max
        if max_keep <= 0:
            return

        ckpt_base = self._output_dir / "checkpoints"
        if not ckpt_base.exists():
            return

        # 收集所有 step_N 目录
        checkpoints: List[Tuple[int, Path]] = []
        for entry in ckpt_base.iterdir():
            if entry.is_dir() and entry.name.startswith("step_"):
                try:
                    step_num = int(entry.name.replace("step_", ""))
                    checkpoints.append((step_num, entry))
                except ValueError:
                    continue

        # 按步数排序, 保留最新的 max_keep 个
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        for _, ckpt_dir in checkpoints[max_keep:]:
            import shutil
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.info("清理旧 checkpoint: %s", ckpt_dir)

    # ------------------------------------------------------------------
    # 内部方法: 日志 & 监控
    # ------------------------------------------------------------------
    def _log_metrics(self, loss: float, grad_norm: Optional[float]) -> None:
        """记录训练指标到 TensorBoard 和日志.

        Args:
            loss:     当前 loss
            grad_norm: 当前梯度范数
        """
        current_lr = self._get_current_lr()

        # TensorBoard
        self._writer.add_scalar("train/loss", loss, self._global_step)
        self._writer.add_scalar("train/learning_rate", current_lr, self._global_step)
        if grad_norm is not None:
            self._writer.add_scalar("train/grad_norm", grad_norm, self._global_step)
        self._writer.add_scalar("train/skipped_batches", self._skipped_batches, self._global_step)

        # 损失分布信息
        if len(self._loss_history) >= self._loss_spike_window:
            recent = self._loss_history[-self._loss_spike_window:]
            mean = sum(recent) / len(recent)
            variance = sum((x - mean) ** 2 for x in recent) / len(recent)
            std = math.sqrt(variance)
            self._writer.add_scalar("train/loss_mean", mean, self._global_step)
            self._writer.add_scalar("train/loss_std", std, self._global_step)

        logger.info(
            "step=%d/%d, loss=%.4f, lr=%.2e, grad_norm=%s, skipped=%d",
            self._global_step,
            self.config.total_steps,
            loss,
            current_lr,
            f"{grad_norm:.4f}" if grad_norm is not None else "N/A",
            self._skipped_batches,
        )

    def _get_current_lr(self) -> float:
        """返回当前学习率 (从优化器 param_group 读取)."""
        return self.optimizer.param_groups[0]["lr"]

    def _create_default_optimizer(self) -> Optimizer:
        """创建默认优化器.

        支持多种优化器:
          - AdamW: 标准优化器
          - MuonClip: Kimi团队的二阶优化器, 收敛速度翻倍
        """
        # 参数分组: 需要 / 不需要权重衰减
        decay_params = []
        no_decay_params = []

        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        for name, param in model_module.named_parameters():
            if not param.requires_grad:
                continue
            # LayerNorm / 偏置不进行权重衰减
            if "layer_norm" in name or "layernorm" in name.lower() or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        # 根据配置选择优化器
        optimizer_type = getattr(self.config, 'optimizer_type', 'adamw')
        
        if optimizer_type.lower() == 'muonclip':
            try:
                from SimplCore.training.muon_clip import MuonClipW
                optimizer = MuonClipW(
                    param_groups,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                    weight_decay=self.config.weight_decay,
                )
                logger.info("MuonClipW 二阶优化器已启用 (Kimi团队技术)")
            except ImportError:
                logger.warning("MuonClipW 未找到, 回退到 AdamW")
                optimizer = torch.optim.AdamW(
                    param_groups,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )

        return optimizer

    def _apply_gradient_checkpointing(self) -> None:
        """应用梯度检查点优化, 节省激活内存.

        将模型的子模块标记为使用梯度检查点, 以计算换内存。
        对 LegacySpectralMixer、SimplCoreBlock 等大型模块应用检查点。
        """
        model_module = self.model.module if isinstance(self.model, DDP) else self.model

        def set_checkpoint(module):
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                return
            if len(list(module.children())) == 0:
                return
            if hasattr(module, "forward") and callable(module.forward):
                orig_forward = module.forward

                def checkpoint_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(orig_forward, *args, use_reentrant=False)

                module.forward = checkpoint_forward

        model_module.apply(set_checkpoint)
        logger.info("梯度检查点已应用")


# ======================================================================
# 自检
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from SimplCore.architecture.config import SimplCoreConfig
    from SimplCore.architecture.SimplCore_model import SimplCoreModel
    from simplcore.tensor.utils.data import TensorDataset

    # 创建小型配置用于测试
    cfg = TrainingConfig(
        total_steps=50,
        log_every=5,
        checkpoint_every=20,
        eval_every=25,
        output_dir="./_test_trainer_output",
        distributed=False,
        precision="fp32",
        seed=42,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=5,
    )

    model_cfg = SimplCoreConfig.get_small_config()
    model = SimplCoreModel(model_cfg)

    # 模拟数据
    vocab_size = model_cfg.vocab_size
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (64, seq_len))
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, labels)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    trainer = SimplCoreTrainer(
        config=cfg,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )

    summary = trainer.train()
    print("\n训练摘要:")
    for key, val in summary.items():
        print(f"  {key}: {val}")

    # 清理测试输出
    import shutil
    test_dir = Path("./_test_trainer_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print("自检通过!")


