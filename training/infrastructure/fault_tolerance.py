"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

fault_tolerance.py - 容错机制模块
心跳检测、节点故障检测、自动重启、断点续训
"""

import os
import sys
import json
import time
import math
import logging
import socket
import threading
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import traceback

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger("青小团.FaultTolerance")


class NodeStatus(Enum):
    """节点状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNSTABLE = "unstable"
    FAILED = "failed"
    RECOVERING = "recovering"
    OFFLINE = "offline"


@dataclass
class NodeInfo:
    """青小团节点信息"""
    node_id: str = ""
    node_rank: int = 0
    hostname: str = ""
    ip_address: str = ""
    pid: int = 0
    status: NodeStatus = NodeStatus.HEALTHY
    gpu_count: int = 0
    memory_total_gb: float = 0.0
    last_heartbeat: float = 0.0
    heartbeat_miss_count: int = 0
    current_step: int = 0
    current_loss: float = 0.0
    uptime_seconds: float = 0.0
    role: str = "worker"  # coordinator, worker
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        if "status" in data and isinstance(data["status"], str):
            data["status"] = NodeStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingProgress:
    """青小团训练进度"""
    global_step: int = 0
    epoch: int = 0
    loss: float = float("inf")
    best_loss: float = float("inf")
    total_samples_processed: int = 0
    total_tokens_processed: int = 0
    current_checkpoint: str = ""
    loss_history: List[float] = field(default_factory=list)
    learning_rate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingProgress":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class HeartbeatMonitor:
    """
    青小团心跳监控器
    定期发送和检测心跳，维护节点存活状态
    """

    def __init__(
        self,
        node_id: str,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        max_missed_heartbeats: int = 3,
        local_rank: int = 0,
    ):
        self.node_id = node_id
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.max_missed_heartbeats = max_missed_heartbeats
        self.local_rank = local_rank

        self.nodes: Dict[str, NodeInfo] = {}
        self._running = False
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._on_node_failed: Optional[Callable[[NodeInfo], None]] = None
        self._on_node_recovered: Optional[Callable[[NodeInfo], None]] = None

        self._node_failure_history: List[Dict[str, Any]] = []

        if self.local_rank == 0:
            logger.info(f"青小团心跳监控器初始化: 间隔={heartbeat_interval}s, 超时={heartbeat_timeout}s")

    def register_node(self, node_info: NodeInfo):
        """注册节点"""
        with self._lock:
            self.nodes[node_info.node_id] = node_info
        logger.info(f"青小团 - 注册节点: {node_info.node_id} ({node_info.hostname})")

    def unregister_node(self, node_id: str):
        """注销节点"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = NodeStatus.OFFLINE
                del self.nodes[node_id]
        logger.info(f"青小团 - 注销节点: {node_id}")

    def receive_heartbeat(self, node_id: str, status_data: Dict[str, Any]):
        """接收心跳信号"""
        with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                node.heartbeat_miss_count = 0
                node.status = NodeStatus.HEALTHY

                for k, v in status_data.items():
                    if hasattr(node, k):
                        setattr(node, k, v)

                node.uptime_seconds = time.time() - node.last_heartbeat

    def send_heartbeat(self, node_info: NodeInfo):
        """发送心跳信号（用于协调节点接收）"""
        pass

    def start(self):
        """启动心跳监控"""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        if self.local_rank == 0:
            logger.info("青小团 - 心跳监控已启动")

    def stop(self):
        """停止心跳监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("青小团 - 心跳监控已停止")

    def _monitor_loop(self):
        """心跳监控主循环"""
        while self._running:
            try:
                self._check_heartbeats()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"青小团 - 心跳监控异常: {e}")

    def _check_heartbeats(self):
        """检查所有节点的心跳状态"""
        now = time.time()
        with self._lock:
            for node_id, node in list(self.nodes.items()):
                time_since_heartbeat = now - node.last_heartbeat

                if time_since_heartbeat > self.heartbeat_timeout:
                    node.heartbeat_miss_count += 1

                    if node.heartbeat_miss_count >= self.max_missed_heartbeats:
                        old_status = node.status
                        node.status = NodeStatus.FAILED

                        if old_status != NodeStatus.FAILED:
                            logger.error(
                                f"青小团 - 节点故障检测: {node_id} "
                                f"(host: {node.hostname}), "
                                f"已错过 {node.heartbeat_miss_count} 次心跳"
                            )

                            failure_record = {
                                "node_id": node_id,
                                "hostname": node.hostname,
                                "time": datetime.now().isoformat(),
                                "missed_heartbeats": node.heartbeat_miss_count,
                                "last_known_step": node.current_step,
                            }
                            self._node_failure_history.append(failure_record)

                            if self._on_node_failed:
                                self._on_node_failed(node)

                    elif node.heartbeat_miss_count >= 1:
                        if node.status == NodeStatus.HEALTHY:
                            node.status = NodeStatus.UNSTABLE
                            logger.warning(
                                f"青小团 - 节点不稳定: {node_id}, "
                                f"上次心跳: {time_since_heartbeat:.1f}秒前"
                            )

    def set_on_node_failed_callback(self, callback: Callable[[NodeInfo], None]):
        """设置节点故障回调"""
        self._on_node_failed = callback

    def set_on_node_recovered_callback(self, callback: Callable[[NodeInfo], None]):
        """设置节点恢复回调"""
        self._on_node_recovered = callback

    def get_failed_nodes(self) -> List[NodeInfo]:
        """获取故障节点列表"""
        with self._lock:
            return [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]

    def get_active_nodes(self) -> List[NodeInfo]:
        """获取活跃节点列表"""
        with self._lock:
            return [n for n in self.nodes.values()
                    if n.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED)]

    def get_node_count(self) -> Dict[str, int]:
        """获取各类节点数量"""
        with self._lock:
            counts = {}
            for node in self.nodes.values():
                status_str = node.status.value
                counts[status_str] = counts.get(status_str, 0) + 1
            counts["total"] = len(self.nodes)
            return counts

    def get_failure_history(self) -> List[Dict[str, Any]]:
        """获取故障历史"""
        return self._node_failure_history


class FaultTolerance:
    """
    青小团故障容错系统

    提供心跳检测、节点故障检测、自动恢复、断点续训和弹性训练能力
    """

    def __init__(
        self,
        work_dir: str,
        model_name: str = "青小团",
        node_id: Optional[str] = None,
        node_rank: int = 0,
        world_size: int = 1,
        enable_heartbeat: bool = True,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        enable_auto_restart: bool = True,
        max_restart_attempts: int = 3,
        restart_delay: float = 10.0,
        checkpoint_on_failure: bool = True,
        progress_persist_interval: int = 100,
        enable_elastic_training: bool = False,
    ):
        self.work_dir = Path(work_dir)
        self.model_name = model_name
        self.node_id = node_id or f"node_{socket.gethostname()}_{node_rank}"
        self.node_rank = node_rank
        self.world_size = world_size
        self.enable_heartbeat = enable_heartbeat
        self.enable_auto_restart = enable_auto_restart
        self.max_restart_attempts = max_restart_attempts
        self.restart_delay = restart_delay
        self.checkpoint_on_failure = checkpoint_on_failure
        self.progress_persist_interval = progress_persist_interval
        self.enable_elastic_training = enable_elastic_training

        self.fault_dir = self.work_dir / "fault_tolerance"
        self.progress_dir = self.work_dir / "progress"
        self.recovery_dir = self.work_dir / "recovery"

        if self.node_rank == 0:
            self.fault_dir.mkdir(parents=True, exist_ok=True)
            self.progress_dir.mkdir(parents=True, exist_ok=True)
            self.recovery_dir.mkdir(parents=True, exist_ok=True)

        self.heartbeat_monitor = HeartbeatMonitor(
            node_id=self.node_id,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            local_rank=node_rank,
        )

        self._training_progress = TrainingProgress()
        self._restart_count = 0
        self._recovery_mode = False
        self._shutdown_flag = False
        self._lock = threading.Lock()
        self._node_info = self._create_node_info()

        self._on_failure_callbacks: List[Callable] = []
        self._on_recovery_callbacks: List[Callable] = []
        self._on_node_add_callbacks: List[Callable] = []
        self._on_node_remove_callbacks: List[Callable] = []

        self._setup_signal_handlers()

        if enable_heartbeat and node_rank == 0:
            self.heartbeat_monitor.start()

        if self.node_rank == 0:
            logger.info(f"青小团故障容错系统初始化完成")
            logger.info(f"青奈科技 - 自动重启={enable_auto_restart}, 弹性训练={enable_elastic_training}")

    def _create_node_info(self) -> NodeInfo:
        """创建当前节点信息"""
        return NodeInfo(
            node_id=self.node_id,
            node_rank=self.node_rank,
            hostname=socket.gethostname(),
            ip_address=socket.gethostbyname(socket.gethostname()),
            pid=os.getpid(),
            status=NodeStatus.HEALTHY,
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            role="coordinator" if self.node_rank == 0 else "worker",
        )

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"青小团 - 接收到信号 {signum}，开始安全关闭")
        self.shutdown()

    def update_training_progress(self, **kwargs):
        """
        更新训练进度

        Args:
            global_step: 全局步数
            epoch: 当前epoch
            loss: 当前loss
            best_loss: 最优loss
            total_samples_processed: 总处理样本数
            total_tokens_processed: 总处理token数
            learning_rate: 学习率
        """
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._training_progress, k):
                    if k == "loss_history" and isinstance(v, list):
                        self._training_progress.loss_history = v[-1000:]
                    else:
                        setattr(self._training_progress, k, v)

            self._training_progress.timestamp = datetime.now().isoformat()

            step = self._training_progress.global_step
            if step % self.progress_persist_interval == 0:
                self._persist_progress()

    def _persist_progress(self):
        """持久化训练进度"""
        try:
            progress_file = self.progress_dir / f"progress_step_{self._training_progress.global_step}.json"
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(self._training_progress.to_dict(), f, ensure_ascii=False, indent=2)

            latest_file = self.progress_dir / "latest_progress.json"
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(self._training_progress.to_dict(), f, ensure_ascii=False, indent=2)

            self._cleanup_old_progress()
        except Exception as e:
            logger.warning(f"青小团 - 持久化训练进度失败: {e}")

    def _cleanup_old_progress(self, keep_last: int = 50):
        """清理旧的进度文件"""
        try:
            progress_files = sorted(
                self.progress_dir.glob("progress_step_*.json"),
                key=lambda p: p.stat().st_mtime,
            )
            for f in progress_files[:-keep_last]:
                f.unlink()
        except Exception:
            pass

    def load_latest_progress(self) -> Optional[TrainingProgress]:
        """加载最近的训练进度"""
        latest_file = self.progress_dir / "latest_progress.json"
        if not latest_file.exists():
            return None

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._training_progress = TrainingProgress.from_dict(data)
            logger.info(
                f"青小团 - 已加载断点进度: step={self._training_progress.global_step}, "
                f"loss={self._training_progress.loss:.6f}"
            )
            return self._training_progress
        except Exception as e:
            logger.warning(f"青小团 - 加载进度失败: {e}")
            return None

    def detect_failure(self) -> List[NodeInfo]:
        """检测故障节点"""
        if not self.enable_heartbeat:
            return []

        return self.heartbeat_monitor.get_failed_nodes()

    def handle_failure(self, failed_node: NodeInfo) -> bool:
        """
        处理节点故障

        Args:
            failed_node: 故障节点信息

        Returns:
            是否成功恢复
        """
        logger.warning(
            f"青小团 - 处理节点故障: {failed_node.node_id} "
            f"(host: {failed_node.hostname})"
        )

        if self.checkpoint_on_failure:
            self._save_failure_checkpoint(failed_node)

        for callback in self._on_failure_callbacks:
            try:
                callback(failed_node)
            except Exception as e:
                logger.error(f"青小团 - 故障回调执行失败: {e}")

        if self.enable_auto_restart and self._restart_count < self.max_restart_attempts:
            return self._attempt_restart()
        else:
            logger.error("青小团 - 已达到最大重启次数，停止自动恢复")
            return False

    def _save_failure_checkpoint(self, failed_node: NodeInfo):
        """保存故障检查点"""
        try:
            failure_record = {
                "company": "青奈科技",
                "model": self.model_name,
                "failure_time": datetime.now().isoformat(),
                "failed_node": failed_node.to_dict(),
                "training_progress": self._training_progress.to_dict(),
                "restart_attempt": self._restart_count + 1,
            }

            failure_file = self.fault_dir / f"failure_{failed_node.node_id}_{int(time.time())}.json"
            with open(failure_file, "w", encoding="utf-8") as f:
                json.dump(failure_record, f, ensure_ascii=False, indent=2)

            logger.info(f"青小团 - 故障记录已保存: {failure_file}")
        except Exception as e:
            logger.warning(f"青小团 - 保存故障记录失败: {e}")

    def _attempt_restart(self) -> bool:
        """尝试重启训练"""
        self._restart_count += 1
        self._recovery_mode = True

        logger.info(
            f"青小团 - 尝试重启训练 (第 {self._restart_count}/{self.max_restart_attempts} 次)"
        )

        time.sleep(self.restart_delay)

        latest = self.load_latest_progress()
        if latest is None:
            logger.warning("青小团 - 无可用进度，将从头开始训练")
        else:
            logger.info(
                f"青小团 - 将从 step {latest.global_step} 继续训练"
            )

        self._recovery_mode = False

        for callback in self._on_recovery_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"青小团 - 恢复回调执行失败: {e}")

        return True

    def add_node(self, node_info: NodeInfo) -> bool:
        """
        动态添加节点（弹性训练）

        Args:
            node_info: 新节点信息

        Returns:
            是否成功添加
        """
        if not self.enable_elastic_training:
            logger.warning("青小团 - 弹性训练未启用，无法添加节点")
            return False

        self.heartbeat_monitor.register_node(node_info)
        self.world_size += 1

        for callback in self._on_node_add_callbacks:
            try:
                callback(node_info)
            except Exception as e:
                logger.error(f"青小团 - 添加节点回调失败: {e}")

        logger.info(
            f"青小团 - 添加节点: {node_info.node_id}, "
            f"当前节点数: {self.world_size}"
        )
        return True

    def remove_node(self, node_id: str) -> bool:
        """
        动态移除节点（弹性训练）

        Args:
            node_id: 节点ID

        Returns:
            是否成功移除
        """
        if not self.enable_elastic_training:
            logger.warning("青小团 - 弹性训练未启用，无法移除节点")
            return False

        node = self.heartbeat_monitor.nodes.get(node_id)
        if node:
            for callback in self._on_node_remove_callbacks:
                try:
                    callback(node)
                except Exception as e:
                    logger.error(f"青小团 - 移除节点回调失败: {e}")

        self.heartbeat_monitor.unregister_node(node_id)
        self.world_size = max(1, self.world_size - 1)

        logger.info(
            f"青小团 - 移除节点: {node_id}, "
            f"当前节点数: {self.world_size}"
        )
        return True

    def register_on_failure(self, callback: Callable[[NodeInfo], None]):
        """注册故障回调"""
        self._on_failure_callbacks.append(callback)

    def register_on_recovery(self, callback: Callable):
        """注册恢复回调"""
        self._on_recovery_callbacks.append(callback)

    def register_on_node_add(self, callback: Callable[[NodeInfo], None]):
        """注册节点添加回调"""
        self._on_node_add_callbacks.append(callback)

    def register_on_node_remove(self, callback: Callable[[NodeInfo], None]):
        """注册节点移除回调"""
        self._on_node_remove_callbacks.append(callback)

    def get_training_progress(self) -> TrainingProgress:
        """获取当前训练进度"""
        return self._training_progress

    def is_recovery_mode(self) -> bool:
        """是否处于恢复模式"""
        return self._recovery_mode

    def get_restart_count(self) -> int:
        """获取重启次数"""
        return self._restart_count

    def get_cluster_health(self) -> Dict[str, Any]:
        """获取集群健康状态"""
        node_counts = self.heartbeat_monitor.get_node_count()
        active_nodes = self.heartbeat_monitor.get_active_nodes()

        return {
            "company": "青奈科技",
            "model": self.model_name,
            "coordinator_node": self.node_id,
            "node_status_counts": node_counts,
            "active_nodes": len(active_nodes),
            "total_nodes": node_counts.get("total", 0),
            "restart_count": self._restart_count,
            "recovery_mode": self._recovery_mode,
            "total_failures": len(self.heartbeat_monitor.get_failure_history()),
            "training_step": self._training_progress.global_step,
            "training_loss": self._training_progress.loss,
            "elastic_training": self.enable_elastic_training,
        }

    def wait_for_cluster_ready(
        self,
        min_nodes: int,
        timeout: float = 300.0,
    ) -> bool:
        """
        等待集群就绪

        Args:
            min_nodes: 最小节点数
            timeout: 超时时间(秒)

        Returns:
            是否就绪
        """
        start = time.time()
        while time.time() - start < timeout:
            active = self.heartbeat_monitor.get_active_nodes()
            if len(active) >= min_nodes:
                logger.info(f"青小团 - 集群就绪: {len(active)}/{min_nodes} 节点活跃")
                return True
            time.sleep(5.0)

        logger.error(f"青小团 - 集群就绪超时: 要求{min_nodes}节点，当前{len(active)}节点")
        return False

    def generate_fault_report(self) -> str:
        """生成故障报告"""
        failures = self.heartbeat_monitor.get_failure_history()
        cluster_health = self.get_cluster_health()

        report = f"""
{'='*60}
青小团故障容错报告 - 青奈科技
{'='*60}

集群健康状态:
  总节点数: {cluster_health['total_nodes']}
  活跃节点: {cluster_health['active_nodes']}
  重启次数: {cluster_health['restart_count']}
  恢复模式: {cluster_health['recovery_mode']}
  弹性训练: {cluster_health['elastic_training']}

故障历史:
  总故障次数: {len(failures)}
"""
        for i, failure in enumerate(failures[-10:]):
            report += f"  {i+1}. 节点 {failure['node_id']} "
            report += f"于 {failure['time']} 故障\n"

        report += f"""
当前训练进度:
  步数: {self._training_progress.global_step}
  Loss: {self._training_progress.loss:.6f}
  Best Loss: {self._training_progress.best_loss:.6f}

{'='*60}
青奈科技 - 青小团分布式训练系统
{'='*60}
"""
        return report

    def shutdown(self):
        """安全关闭故障容错系统"""
        if self._shutdown_flag:
            return

        self._shutdown_flag = True

        self._persist_progress()

        self.heartbeat_monitor.stop()

        logger.info("青小团 - 故障容错系统已关闭")
        logger.info(f"青奈科技 - 最终进度: step={self._training_progress.global_step}")
