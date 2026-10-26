"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

checkpoint_manager.py - 检查点管理模块
负责自动保存、加载、淘汰和验证训练检查点
"""

import os
import sys
import json
import time
import shutil
import logging
import threading
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import traceback

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.CheckpointManager")


@dataclass
class CheckpointMetadata:
    """青小团检查点元数据"""
    version: str = "1.0.0"
    company: str = "青奈科技"
    model_name: str = "青小团"
    global_step: int = 0
    epoch: int = 0
    loss: float = float("inf")
    best_loss: float = float("inf")
    optimizer_state: bool = False
    scheduler_state: bool = False
    train_start_time: Optional[str] = None
    save_time: Optional[str] = None
    checkpoint_type: str = "regular"  # regular, best, periodic
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    loss_history: List[float] = field(default_factory=list)
    learning_rate: float = 0.0
    num_params: int = 0
    num_experts: int = 0
    moe_config: Dict[str, Any] = field(default_factory=dict)
    nodes_count: int = 1
    world_size: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    """
    青小团训练检查点管理器
    支持按步数、时间、loss值自动保存，智能淘汰策略，断点续训自动恢复
    """

    def __init__(
        self,
        save_dir: str,
        model_name: str = "青小团",
        save_every_n_steps: int = 1000,
        save_every_n_hours: float = 2.0,
        save_best_on_loss: bool = True,
        keep_last_n: int = 5,
        keep_best_k: int = 3,
        max_checkpoints: int = 20,
        enable_validation: bool = True,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        初始化青小团检查点管理器

        Args:
            save_dir: 检查点保存根目录
            model_name: 模型名称
            save_every_n_steps: 每N步保存一次
            save_every_n_hours: 每N小时保存一次
            save_best_on_loss: 是否在loss最优时保存
            keep_last_n: 保留最近N个检查点
            keep_best_k: 保留最优K个检查点
            max_checkpoints: 最大检查点数量
            enable_validation: 是否启用检查点验证
            local_rank: 本地进程rank
            world_size: 总进程数
        """
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_hours = save_every_n_hours
        self.save_best_on_loss = save_best_on_loss
        self.keep_last_n = keep_last_n
        self.keep_best_k = keep_best_k
        self.max_checkpoints = max_checkpoints
        self.enable_validation = enable_validation
        self.local_rank = local_rank
        self.world_size = world_size

        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.metadata_dir = self.save_dir / "metadata"
        self.tmp_dir = self.save_dir / "tmp_checkpoints"

        self._init_directories()
        self._lock = threading.Lock()

        self.best_loss = float("inf")
        self.best_checkpoint_path: Optional[Path] = None
        self.last_save_time = time.time()
        self.last_save_step = 0

        self._checkpoint_history: List[Dict[str, Any]] = []
        self._loss_history: List[float] = []

        self._validate_on_save = False
        self._shutdown_flag = False

        self._async_save_thread: Optional[threading.Thread] = None
        self._pending_save: Optional[Dict[str, Any]] = None

        if self.local_rank == 0:
            logger.info(f"青小团检查点管理器初始化完成，保存目录: {self.checkpoints_dir}")
            logger.info(f"青奈科技 - 检查点策略: 每{save_every_n_steps}步 / 每{save_every_n_hours}小时 / " + 
                       f"最佳loss保存={save_best_on_loss}")

    def _init_directories(self):
        """初始化青小团目录结构"""
        if self.local_rank == 0:
            for d in [self.checkpoints_dir, self.metadata_dir, self.tmp_dir]:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f"青小团 - 创建目录: {d}")

    def _get_checkpoint_path(self, step: int, ckpt_type: str = "regular") -> Path:
        """获取检查点文件路径"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qingxiaotuan_step_{step}_{ckpt_type}_{timestamp}"
        return self.checkpoints_dir / filename

    def _get_metadata_path(self, step: int) -> Path:
        """获取元数据文件路径"""
        return self.metadata_dir / f"metadata_step_{step}.json"

    def should_save(self, global_step: int, current_loss: float, force: bool = False) -> bool:
        """判断是否需要保存检查点"""
        if force:
            return True

        if self.local_rank != 0:
            return False

        if global_step - self.last_save_step >= self.save_every_n_steps:
            return True

        hours_elapsed = (time.time() - self.last_save_time) / 3600.0
        if hours_elapsed >= self.save_every_n_hours:
            return True

        if self.save_best_on_loss and current_loss < self.best_loss:
            return True

        return False

    def save_checkpoint(
        self,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        global_step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        learning_rate: float = 0.0,
        moe_config: Optional[Dict[str, Any]] = None,
        extra_data: Optional[Dict[str, Any]] = None,
        force_type: Optional[str] = None,
    ) -> Optional[Path]:
        """
        保存青小团训练检查点

        Args:
            model_state_dict: 模型状态字典
            optimizer_state_dict: 优化器状态
            scheduler_state_dict: 学习率调度器状态
            global_step: 全局训练步数
            epoch: 当前epoch
            loss: 当前loss值
            learning_rate: 当前学习率
            moe_config: MoE配置
            extra_data: 额外数据
            force_type: 强制检查点类型 (regular/best/periodic)

        Returns:
            保存路径
        """
        if self.local_rank != 0:
            return None

        is_best = loss < self.best_loss
        ckpt_type = force_type or ("best" if is_best else "regular")

        save_path = self._get_checkpoint_path(global_step, ckpt_type)
        metadata_path = self._get_metadata_path(global_step)

        self._loss_history.append(loss)
        if is_best:
            self.best_loss = loss
            self.best_checkpoint_path = save_path

        metadata = CheckpointMetadata(
            version="1.0.0",
            company="青奈科技",
            model_name=self.model_name,
            global_step=global_step,
            epoch=epoch,
            loss=loss,
            best_loss=self.best_loss,
            optimizer_state=optimizer_state_dict is not None,
            scheduler_state=scheduler_state_dict is not None,
            save_time=datetime.now().isoformat(),
            checkpoint_type=ckpt_type,
            loss_history=self._loss_history[-1000:],
            learning_rate=learning_rate,
            moe_config=moe_config or {},
            world_size=self.world_size,
        )

        checkpoint_data = {
            "metadata": metadata.to_dict(),
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
            "extra": extra_data or {},
        }

        try:
            self._atomic_save(checkpoint_data, save_path, metadata, metadata_path)
            self.last_save_time = time.time()
            self.last_save_step = global_step

            self._checkpoint_history.append({
                "path": str(save_path),
                "step": global_step,
                "loss": loss,
                "type": ckpt_type,
                "time": datetime.now().isoformat(),
            })

            self._apply_eviction_policy()

            if is_best:
                self._save_best_marker(save_path)

            logger.info(
                f"青小团 - 检查点已保存 [{ckpt_type}] "
                f"步数={global_step}, loss={loss:.6f}, path={save_path.name}"
            )

            dist.barrier() if dist.is_initialized() else None
            return save_path

        except Exception as e:
            logger.error(f"青小团 - 检查点保存失败: {e}")
            logger.error(traceback.format_exc())
            return None

    def _atomic_save(
        self,
        data: Dict[str, Any],
        save_path: Path,
        metadata: CheckpointMetadata,
        metadata_path: Path,
    ):
        """原子化保存，防止保存过程中断导致文件损坏"""
        tmp_path = self.tmp_dir / f"tmp_{save_path.name}.pt"
        tmp_metadata = self.tmp_dir / f"tmp_{metadata_path.name}.json"

        torch.save(data, tmp_path)
        with open(tmp_metadata, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

        shutil.move(str(tmp_path), str(save_path))
        shutil.move(str(tmp_metadata), str(metadata_path))

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        加载青小团检查点，支持自动恢复最佳/最新检查点

        Args:
            checkpoint_path: 指定检查点路径，None则自动搜索
            load_optimizer: 是否加载优化器状态
            load_scheduler: 是否加载调度器状态
            map_location: 设备映射

        Returns:
            包含所有加载数据的字典
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None:
            logger.warning("青小团 - 未找到检查点，将从头开始训练")
            return {"metadata": None, "model_state_dict": None}

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"青小团 - 检查点文件不存在: {checkpoint_path}")
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        if self.enable_validation:
            if not self.validate_checkpoint(str(checkpoint_path)):
                logger.error(f"青小团 - 检查点验证失败: {checkpoint_path}")
                fallback = self._find_fallback_checkpoint()
                if fallback:
                    logger.info(f"青小团 - 使用备用检查点: {fallback}")
                    checkpoint_path = Path(fallback)
                else:
                    raise RuntimeError("所有检查点均验证失败")

        try:
            map_location = map_location or ("cuda" if torch.cuda.is_available() else "cpu")
            data = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

            if not load_optimizer:
                data.pop("optimizer_state_dict", None)
            if not load_scheduler:
                data.pop("scheduler_state_dict", None)

            metadata = data.get("metadata", {})
            if metadata:
                self.best_loss = metadata.get("best_loss", float("inf"))
                self._loss_history = metadata.get("loss_history", [])
                self.last_save_step = metadata.get("global_step", 0)

            logger.info(
                f"青小团 - 检查点加载成功: {checkpoint_path.name}, "
                f"步数={metadata.get('global_step', 'N/A')}, "
                f"loss={metadata.get('loss', 'N/A')}"
            )

            return data

        except Exception as e:
            logger.error(f"青小团 - 检查点加载失败: {e}")
            logger.error(traceback.format_exc())
            raise

    def _find_latest_checkpoint(self) -> Optional[str]:
        """寻找最新的有效检查点"""
        if not self.checkpoints_dir.exists():
            return None

        checkpoint_files = sorted(
            self.checkpoints_dir.glob("qingxiaotuan_step_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not checkpoint_files:
            return None

        return str(checkpoint_files[0])

    def _find_fallback_checkpoint(self) -> Optional[str]:
        """寻找备用的有效检查点"""
        checkpoints = sorted(
            self.checkpoints_dir.glob("qingxiaotuan_step_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for cp in checkpoints[1:]:
            try:
                if self.validate_checkpoint(str(cp)):
                    return str(cp)
            except Exception:
                continue

        return None

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        验证检查点完整性

        检查项:
        1. 文件完整性
        2. 元数据完整性
        3. 模型参数可加载
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logger.error(f"青小团 - 检查点文件不存在: {path}")
            return False

        file_size = path.stat().st_size
        if file_size < 1024:
            logger.error(f"青小团 - 检查点文件过小: {file_size} bytes")
            return False

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)

            if "metadata" not in data:
                logger.error("青小团 - 检查点缺少metadata")
                return False

            if "model_state_dict" not in data:
                logger.error("青小团 - 检查点缺少model_state_dict")
                return False

            metadata = data["metadata"]
            required_fields = ["company", "model_name", "global_step"]
            for field in required_fields:
                if field not in metadata:
                    logger.error(f"青小团 - 元数据缺少字段: {field}")
                    return False

            model_dict = data["model_state_dict"]
            param_count = sum(v.numel() for v in model_dict.values() if isinstance(v, torch.Tensor))
            if param_count == 0:
                logger.error("青小团 - 模型参数量为0")
                return False

            logger.info(
                f"青小团 - 检查点验证通过: {path.name}, "
                f"参数量={param_count:,}, 文件大小={file_size / 1024 / 1024:.2f}MB"
            )
            return True

        except Exception as e:
            logger.error(f"青小团 - 检查点验证异常: {e}")
            return False

    def _save_best_marker(self, best_path: Path):
        """保存最佳检查点标记"""
        marker_path = self.checkpoints_dir / "best_checkpoint.txt"
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"青奈科技 - 青小团最佳检查点\n")
            f.write(f"路径: {best_path}\n")
            f.write(f"loss: {self.best_loss:.6f}\n")
            f.write(f"保存时间: {datetime.now().isoformat()}\n")

    def _apply_eviction_policy(self):
        """
        应用检查点淘汰策略

        策略:
        1. 保留最近N个检查点
        2. 保留最优K个检查点
        3. 超过最大数量时自动删除
        """
        all_checkpoints = sorted(
            self.checkpoints_dir.glob("qingxiaotuan_step_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if len(all_checkpoints) <= self.max_checkpoints:
            return

        recent = set(all_checkpoints[:self.keep_last_n])

        if self.save_best_on_loss:
            best_by_loss = sorted(
                [c for c in all_checkpoints if c not in recent],
                key=lambda p: self._get_loss_from_checkpoint(p),
            )[:self.keep_best_k]
            keep = recent | set(best_by_loss)
        else:
            keep = recent

        to_delete = [c for c in all_checkpoints if c not in keep]
        for cp in to_delete:
            try:
                cp.unlink()
                meta_path = self._get_metadata_for_checkpoint(cp)
                if meta_path and meta_path.exists():
                    meta_path.unlink()
                logger.info(f"青小团 - 删除旧检查点: {cp.name}")
            except Exception as e:
                logger.warning(f"青小团 - 删除检查点失败: {cp.name}, {e}")

    def _get_loss_from_checkpoint(self, path: Path) -> float:
        """从检查点文件中读取loss值"""
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            return data.get("metadata", {}).get("loss", float("inf"))
        except Exception:
            return float("inf")

    def _get_metadata_for_checkpoint(self, path: Path) -> Optional[Path]:
        """获取检查点对应的元数据文件路径"""
        metadata_path = self.metadata_dir / f"metadata_{path.name.replace('qingxiaotuan_', '')}.json"
        if metadata_path.exists():
            return metadata_path
        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有可用检查点"""
        checkpoints = sorted(
            self.checkpoints_dir.glob("qingxiaotuan_step_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        result = []
        for cp in checkpoints:
            try:
                metadata = self._extract_metadata(cp)
                result.append({
                    "path": str(cp),
                    "name": cp.name,
                    "size_mb": cp.stat().st_size / 1024 / 1024,
                    "created": datetime.fromtimestamp(cp.stat().st_mtime).isoformat(),
                    "metadata": metadata,
                })
            except Exception as e:
                result.append({
                    "path": str(cp),
                    "name": cp.name,
                    "error": str(e),
                })

        return result

    def _extract_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """从检查点提取元数据"""
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            return data.get("metadata", None)
        except Exception:
            return None

    def get_resume_state(self) -> Dict[str, Any]:
        """
        获取断点续训状态

        Returns:
            {
                "checkpoint_path": str,
                "global_step": int,
                "epoch": int,
                "loss": float,
                "best_loss": float,
                "loss_history": List[float],
                "learning_rate": float,
                "hardware_info": Dict,
            }
        """
        latest = self._find_latest_checkpoint()
        if latest is None:
            return {"can_resume": False}

        try:
            data = self.load_checkpoint(latest, load_optimizer=False, load_scheduler=False)
            metadata = data.get("metadata", {})

            return {
                "can_resume": True,
                "checkpoint_path": latest,
                "global_step": metadata.get("global_step", 0),
                "epoch": metadata.get("epoch", 0),
                "loss": metadata.get("loss", float("inf")),
                "best_loss": metadata.get("best_loss", float("inf")),
                "loss_history": metadata.get("loss_history", []),
                "learning_rate": metadata.get("learning_rate", 0.0),
                "save_time": metadata.get("save_time", ""),
                "checkpoint_type": metadata.get("checkpoint_type", ""),
            }
        except Exception as e:
            logger.error(f"青小团 - 获取恢复状态失败: {e}")
            return {"can_resume": False, "error": str(e)}

    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳检查点路径"""
        marker = self.checkpoints_dir / "best_checkpoint.txt"
        if marker.exists():
            with open(marker, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("路径:"):
                        path_str = line.strip()[3:].strip()
                        if Path(path_str).exists():
                            return path_str

        best_path = self._find_best_by_loss()
        return str(best_path) if best_path else None

    def _find_best_by_loss(self) -> Optional[Path]:
        """通过loss寻找最佳检查点"""
        checkpoints = list(self.checkpoints_dir.glob("qingxiaotuan_step_*"))
        if not checkpoints:
            return None

        best = None
        best_loss = float("inf")
        for cp in checkpoints:
            loss = self._get_loss_from_checkpoint(cp)
            if loss < best_loss:
                best_loss = loss
                best = cp

        return best

    def convert_checkpoint_for_inference(self, checkpoint_path: str, output_path: str):
        """
        将训练检查点转换为推理格式

        移除优化器、调度器等训练相关状态，仅保留模型权重
        """
        logger.info(f"青小团 - 开始转换检查点为推理格式: {checkpoint_path}")
        data = self.load_checkpoint(checkpoint_path, load_optimizer=False, load_scheduler=False)

        inference_data = {
            "metadata": {
                "company": "青奈科技",
                "model_name": "青小团",
                "version": data.get("metadata", {}).get("version", "1.0.0"),
                "converted_at": datetime.now().isoformat(),
                "source_checkpoint": str(Path(checkpoint_path).name),
                "format": "inference",
            },
            "model_state_dict": data["model_state_dict"],
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inference_data, output_path_obj)

        logger.info(f"青小团 - 推理格式检查点已保存: {output_path}")
        return str(output_path_obj)

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """获取检查点摘要统计"""
        all_ckpts = self.list_checkpoints()

        summary = {
            "company": "青奈科技",
            "model_name": self.model_name,
            "total_checkpoints": len(all_ckpts),
            "total_size_gb": sum(c.get("size_mb", 0) for c in all_ckpts) / 1024,
            "best_loss": self.best_loss if self.best_loss < float("inf") else None,
            "latest_step": self.last_save_step,
            "latest_loss": self._loss_history[-1] if self._loss_history else None,
            "checkpoints": all_ckpts[:10],
        }
        return summary

    def cleanup_tmp_files(self):
        """清理临时文件"""
        if self.tmp_dir.exists():
            for f in self.tmp_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            logger.info("青小团 - 临时文件清理完成")

    def shutdown(self):
        """关闭检查点管理器，清理资源"""
        self._shutdown_flag = True
        self.cleanup_tmp_files()
        logger.info("青小团 - 检查点管理器已关闭")


class AsyncCheckpointManager(CheckpointManager):
    """
    异步检查点管理器
    在独立的线程中执行保存操作，不影响训练进程
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._save_queue: List[Dict[str, Any]] = []
        self._queue_lock = threading.Lock()
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
        logger.info("青小团 - 异步检查点保存线程已启动")

    def _save_worker(self):
        """后台保存工作线程"""
        while not self._shutdown_flag:
            try:
                save_data = None
                with self._queue_lock:
                    if self._save_queue:
                        save_data = self._save_queue.pop(0)

                if save_data:
                    super().save_checkpoint(**save_data)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"青小团 - 异步保存线程异常: {e}")
                time.sleep(1.0)

    def save_checkpoint(self, *args, **kwargs):
        """将保存任务加入队列，立即返回"""
        with self._queue_lock:
            self._save_queue.append({
                "model_state_dict": kwargs.get("model_state_dict"),
                "optimizer_state_dict": kwargs.get("optimizer_state_dict"),
                "scheduler_state_dict": kwargs.get("scheduler_state_dict"),
                "global_step": kwargs.get("global_step", 0),
                "epoch": kwargs.get("epoch", 0),
                "loss": kwargs.get("loss", 0.0),
                "learning_rate": kwargs.get("learning_rate", 0.0),
                "moe_config": kwargs.get("moe_config"),
                "extra_data": kwargs.get("extra_data"),
                "force_type": kwargs.get("force_type"),
            })

    def wait_for_pending_saves(self, timeout: float = 300.0):
        """等待所有待处理的保存操作完成"""
        start = time.time()
        while self._save_queue:
            if time.time() - start > timeout:
                logger.warning("青小团 - 等待保存超时")
                break
            time.sleep(0.5)
        logger.info("青小团 - 所有待处理保存操作已完成")


class DistributedCheckpointManager(CheckpointManager):
    """
    分布式检查点管理器
    支持多节点一致的检查点保存和加载
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_rank = kwargs.get("node_rank", 0)
        self._nodes_count = kwargs.get("nodes_count", 1)
        self._consensus_timeout = kwargs.get("consensus_timeout", 60)

    def save_checkpoint(self, *args, **kwargs):
        """分布式一致保存，所有节点确认后保存"""
        if not dist.is_initialized():
            return super().save_checkpoint(*args, **kwargs)

        global_step = kwargs.get("global_step", 0)
        should_save = torch.tensor([1 if self.should_save(global_step, kwargs.get("loss", 0.0)) else 0],
                                    device="cuda")

        dist.all_reduce(should_save, op=dist.ReduceOp.MAX)
        if should_save.item() == 0:
            return None

        dist.barrier()

        if self.local_rank == 0:
            result = super().save_checkpoint(*args, **kwargs)
        else:
            result = None

        dist.broadcast_object_list(
            [result] if result else [None],
            src=0,
        )

        dist.barrier()
        return result if self.local_rank == 0 else None

    def _gather_metadata_from_nodes(self) -> Dict[str, Any]:
        """收集所有节点的元数据"""
        if not dist.is_initialized():
            return {}

        gathered = [{} for _ in range(dist.get_world_size())]
        local_meta = {
            "node_rank": self._node_rank,
            "world_size": self.world_size,
        }
        dist.all_gather_object(gathered, local_meta)

        result = {}
        for i, meta in enumerate(gathered):
            result[f"node_{i}"] = meta

        return result


def get_optimal_checkpoint_interval(
    total_steps: int,
    expected_training_hours: float,
    desired_checkpoints: int = 10,
) -> Tuple[int, float]:
    """
    计算最优检查点保存间隔

    Args:
        total_steps: 总训练步数
        expected_training_hours: 预期训练时长(小时)
        desired_checkpoints: 期望的检查点数量

    Returns:
        (step_interval, hour_interval): 步数间隔和小时间隔
    """
    step_interval = max(100, total_steps // desired_checkpoints)
    hour_interval = max(0.5, expected_training_hours / desired_checkpoints)
    return step_interval, hour_interval
