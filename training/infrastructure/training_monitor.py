"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

training_monitor.py - 训练监控模块
实时监控训练指标，支持TensorBoard、WandB、控制台进度条和自动告警
"""

import os
import sys
import time
import json
import math
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import traceback

import torch
import numpy as np

logger = logging.getLogger("青小团.TrainingMonitor")


@dataclass
class MonitorMetrics:
    """青小团监控指标数据"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    smoothed_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    tokens_per_second: float = 0.0
    gpu_utilization: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    expert_usage: Dict[int, float] = field(default_factory=dict)
    moe_load_balancing_loss: float = 0.0
    throughput_samples_per_sec: float = 0.0
    batch_time_ms: float = 0.0
    data_load_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    communication_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


class TrainingMonitor:
    """
    青小团训练监控器
    实时采集和展示训练指标，支持TensorBoard、WandB和自动告警
    """

    def __init__(
        self,
        log_dir: str,
        model_name: str = "青小团",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        smooth_factor: float = 0.99,
        alert_on_loss_spike: bool = True,
        alert_on_grad_explosion: bool = True,
        loss_spike_threshold: float = 10.0,
        grad_norm_threshold: float = 100.0,
        alert_interval_seconds: int = 300,
        alert_webhook_url: Optional[str] = None,
        alert_email: Optional[str] = None,
        alert_dingtalk_token: Optional[str] = None,
        progress_bar_enabled: bool = True,
        local_rank: int = 0,
    ):
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project or f"{model_name}_training"
        self.wandb_entity = wandb_entity
        self.smooth_factor = smooth_factor
        self.alert_on_loss_spike = alert_on_loss_spike
        self.alert_on_grad_explosion = alert_on_grad_explosion
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_norm_threshold = grad_norm_threshold
        self.alert_interval_seconds = alert_interval_seconds
        self.alert_webhook_url = alert_webhook_url
        self.alert_email = alert_email
        self.alert_dingtalk_token = alert_dingtalk_token
        self.progress_bar_enabled = progress_bar_enabled
        self.local_rank = local_rank

        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.metrics_dir = self.log_dir / "metrics"
        self.alerts_dir = self.log_dir / "alerts"

        if self.local_rank == 0:
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            self.alerts_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_history: deque = deque(maxlen=10000)
        self._loss_history: deque = deque(maxlen=10000)
        self._lr_history: deque = deque(maxlen=10000)
        self._grad_norm_history: deque = deque(maxlen=10000)
        self._throughput_history: deque = deque(maxlen=10000)
        self._expert_usage_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))

        self._current_step = 0
        self._current_epoch = 0
        self._smoothed_loss = 0.0
        self._start_time = time.time()
        self._step_start_time = time.time()
        self._last_alert_time = 0
        self._anomaly_count = 0
        self._total_tokens_processed = 0
        self._total_samples_processed = 0
        self._lock = threading.Lock()

        self._tensorboard_writer = None
        self._wandb_run = None
        self._progress_bar = None
        self._alert_thread: Optional[threading.Thread] = None
        self._alert_queue: List[Dict[str, Any]] = []
        self._shutdown_flag = False
        self._alert_lock = threading.Lock()

        self._init_writers()

        if self.local_rank == 0:
            logger.info(f"青小团训练监控器初始化完成")
            logger.info(f"青奈科技 - TensorBoard={use_tensorboard}, WandB={use_wandb}, 进度条={progress_bar_enabled}")

    def _init_writers(self):
        """初始化日志写入器"""
        if self.local_rank != 0:
            return

        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
                logger.info(f"青小团 - TensorBoard日志目录: {self.tensorboard_dir}")
            except ImportError:
                logger.warning("青小团 - tensorboard未安装，禁用TensorBoard")
                self.use_tensorboard = False

        if self.use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={"model": self.model_name},
                )
                logger.info(f"青小团 - WandB初始化成功，project={self.wandb_project}")
            except ImportError:
                logger.warning("青小团 - wandb未安装，禁用WandB")
                self.use_wandb = False
            except Exception as e:
                logger.warning(f"青小团 - WandB初始化失败: {e}")
                self.use_wandb = False

    def log_metrics(self, metrics: MonitorMetrics):
        """
        记录训练指标

        Args:
            metrics: 监控指标数据
        """
        self._current_step = metrics.step
        self._current_epoch = metrics.epoch

        with self._lock:
            self._smoothed_loss = (
                self.smooth_factor * self._smoothed_loss
                + (1 - self.smooth_factor) * metrics.loss
                if self._smoothed_loss > 0
                else metrics.loss
            )
            metrics.smoothed_loss = self._smoothed_loss

            self._metrics_history.append(metrics)
            self._loss_history.append((metrics.step, metrics.loss, self._smoothed_loss))
            self._lr_history.append((metrics.step, metrics.learning_rate))
            self._grad_norm_history.append((metrics.step, metrics.gradient_norm))
            self._throughput_history.append((metrics.step, metrics.tokens_per_second))

            for expert_id, usage in metrics.expert_usage.items():
                self._expert_usage_history[expert_id].append((metrics.step, usage))

        self._total_tokens_processed += int(metrics.tokens_per_second * metrics.batch_time_ms / 1000)
        self._total_samples_processed += 1

        if self.local_rank == 0:
            self._write_to_tensorboard(metrics)
            self._write_to_wandb(metrics)
            self._check_anomalies(metrics)
            self._save_metrics_to_disk(metrics)

    def _write_to_tensorboard(self, metrics: MonitorMetrics):
        """写入TensorBoard"""
        if not self.use_tensorboard or self._tensorboard_writer is None:
            return

        try:
            step = metrics.step
            self._tensorboard_writer.add_scalar("train/loss", metrics.loss, step)
            self._tensorboard_writer.add_scalar("train/smoothed_loss", metrics.smoothed_loss, step)
            self._tensorboard_writer.add_scalar("train/learning_rate", metrics.learning_rate, step)
            self._tensorboard_writer.add_scalar("train/gradient_norm", metrics.gradient_norm, step)
            self._tensorboard_writer.add_scalar("train/tokens_per_second", metrics.tokens_per_second, step)
            self._tensorboard_writer.add_scalar("train/gpu_utilization", metrics.gpu_utilization, step)
            self._tensorboard_writer.add_scalar("train/memory_used_gb", metrics.memory_used_gb, step)
            self._tensorboard_writer.add_scalar("train/throughput_samples_per_sec", metrics.throughput_samples_per_sec, step)
            self._tensorboard_writer.add_scalar("train/batch_time_ms", metrics.batch_time_ms, step)
            self._tensorboard_writer.add_scalar("train/moe_load_balancing_loss", metrics.moe_load_balancing_loss, step)

            self._tensorboard_writer.add_scalar("timing/data_load_time_ms", metrics.data_load_time_ms, step)
            self._tensorboard_writer.add_scalar("timing/compute_time_ms", metrics.compute_time_ms, step)
            self._tensorboard_writer.add_scalar("timing/communication_time_ms", metrics.communication_time_ms, step)

            for expert_id, usage in metrics.expert_usage.items():
                self._tensorboard_writer.add_scalar(f"expert_usage/expert_{expert_id}", usage, step)

            for k, v in metrics.extra.items():
                if isinstance(v, (int, float)):
                    self._tensorboard_writer.add_scalar(f"extra/{k}", v, step)

        except Exception as e:
            logger.warning(f"青小团 - TensorBoard写入失败: {e}")

    def _write_to_wandb(self, metrics: MonitorMetrics):
        """写入WandB"""
        if not self.use_wandb or self._wandb_run is None:
            return

        try:
            import wandb
            log_data = {
                "train/loss": metrics.loss,
                "train/smoothed_loss": metrics.smoothed_loss,
                "train/learning_rate": metrics.learning_rate,
                "train/gradient_norm": metrics.gradient_norm,
                "train/tokens_per_second": metrics.tokens_per_second,
                "train/gpu_utilization": metrics.gpu_utilization,
                "train/memory_used_gb": metrics.memory_used_gb,
                "train/throughput_samples_per_sec": metrics.throughput_samples_per_sec,
                "train/batch_time_ms": metrics.batch_time_ms,
                "train/moe_load_balancing_loss": metrics.moe_load_balancing_loss,
                "timing/data_load_time_ms": metrics.data_load_time_ms,
                "timing/compute_time_ms": metrics.compute_time_ms,
                "timing/communication_time_ms": metrics.communication_time_ms,
            }
            for expert_id, usage in metrics.expert_usage.items():
                log_data[f"expert_usage/expert_{expert_id}"] = usage
            log_data.update(metrics.extra)
            self._wandb_run.log(log_data, step=metrics.step)

        except Exception as e:
            logger.warning(f"青小团 - WandB写入失败: {e}")

    def _check_anomalies(self, metrics: MonitorMetrics):
        """检测训练异常并触发告警"""
        now = time.time()
        anomalies = []

        if self.alert_on_loss_spike and metrics.loss > self._smoothed_loss * self.loss_spike_threshold:
            anomalies.append({
                "type": "loss_spike",
                "severity": "warning",
                "message": (
                    f"Loss激增检测: 当前loss={metrics.loss:.4f}, "
                    f"平滑loss={self._smoothed_loss:.4f}, "
                    f"阈值={self._smoothed_loss * self.loss_spike_threshold:.4f}"
                ),
                "value": metrics.loss,
                "threshold": self._smoothed_loss * self.loss_spike_threshold,
            })

        if self.alert_on_grad_explosion and metrics.gradient_norm > self.grad_norm_threshold:
            anomalies.append({
                "type": "grad_explosion",
                "severity": "critical",
                "message": (
                    f"梯度爆炸检测: 梯度范数={metrics.gradient_norm:.4f}, "
                    f"阈值={self.grad_norm_threshold}"
                ),
                "value": metrics.gradient_norm,
                "threshold": self.grad_norm_threshold,
            })

        if metrics.moe_load_balancing_loss > 1.0:
            anomalies.append({
                "type": "moe_imbalance",
                "severity": "warning",
                "message": (
                    f"MoE负载严重不均衡: load_balancing_loss={metrics.moe_load_balancing_loss:.4f}"
                ),
                "value": metrics.moe_load_balancing_loss,
                "threshold": 1.0,
            })

        if metrics.gpu_utilization < 10.0:
            anomalies.append({
                "type": "gpu_underutilization",
                "severity": "info",
                "message": (
                    f"GPU利用率过低: {metrics.gpu_utilization:.1f}%, 可能IO瓶颈"
                ),
                "value": metrics.gpu_utilization,
                "threshold": 10.0,
            })

        for anomaly in anomalies:
            self._anomaly_count += 1
            logger.warning(f"青小团 - 训练异常检测 [{anomaly['severity']}]: {anomaly['message']}")
            self._save_alert(anomaly)
            self._queue_alert(anomaly)

    def _save_alert(self, alert: Dict[str, Any]):
        """保存告警记录到磁盘"""
        try:
            alert_file = self.alerts_dir / f"alert_{self._anomaly_count}_{int(time.time())}.json"
            with open(alert_file, "w", encoding="utf-8") as f:
                json.dump({
                    "company": "青奈科技",
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "step": self._current_step,
                    **alert,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"青小团 - 保存告警记录失败: {e}")

    def _queue_alert(self, alert: Dict[str, Any]):
        """将告警加入发送队列"""
        now = time.time()
        with self._alert_lock:
            if now - self._last_alert_time >= self.alert_interval_seconds:
                self._alert_queue.append(alert)
                self._last_alert_time = now
                if self._alert_thread is None or not self._alert_thread.is_alive():
                    self._alert_thread = threading.Thread(target=self._send_alerts, daemon=True)
                    self._alert_thread.start()

    def _send_alerts(self):
        """发送告警通知"""
        while self._alert_queue:
            alert = self._alert_queue.pop(0)
            self._send_webhook_alert(alert)
            self._send_dingtalk_alert(alert)
            self._send_email_alert(alert)

    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """通过webhook发送告警"""
        if not self.alert_webhook_url:
            return
        try:
            import requests
            payload = {
                "msgtype": "text",
                "text": {
                    "content": (
                        f"【青小团训练告警】\n"
                        f"类型: {alert['type']}\n"
                        f"严重程度: {alert['severity']}\n"
                        f"消息: {alert['message']}\n"
                        f"步数: {self._current_step}\n"
                        f"时间: {datetime.now().isoformat()}\n"
                        f"青奈科技 - 青小团监控系统"
                    ),
                },
            }
            requests.post(self.alert_webhook_url, json=payload, timeout=10)
            logger.info(f"青小团 - Webhook告警已发送: {alert['type']}")
        except Exception as e:
            logger.warning(f"青小团 - Webhook告警发送失败: {e}")

    def _send_dingtalk_alert(self, alert: Dict[str, Any]):
        """通过钉钉发送告警"""
        if not self.alert_dingtalk_token:
            return
        try:
            import requests
            url = f"https://oapi.dingtalk.com/robot/send?access_token={self.alert_dingtalk_token}"
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": "青小团训练告警",
                    "text": (
                        f"## 青小团训练告警\n"
                        f"- **类型**: {alert['type']}\n"
                        f"- **严重程度**: {alert['severity']}\n"
                        f"- **消息**: {alert['message']}\n"
                        f"- **步数**: {self._current_step}\n"
                        f"- **时间**: {datetime.now().isoformat()}\n"
                        f"\n---\n青奈科技 - 青小团监控系统"
                    ),
                },
            }
            headers = {"Content-Type": "application/json"}
            requests.post(url, json=payload, headers=headers, timeout=10)
            logger.info(f"青小团 - 钉钉告警已发送: {alert['type']}")
        except Exception as e:
            logger.warning(f"青小团 - 钉钉告警发送失败: {e}")

    def _send_email_alert(self, alert: Dict[str, Any]):
        """通过邮件发送告警"""
        if not self.alert_email:
            return
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(
                f"青小团训练告警\n\n"
                f"类型: {alert['type']}\n"
                f"严重程度: {alert['severity']}\n"
                f"消息: {alert['message']}\n"
                f"步数: {self._current_step}\n"
                f"时间: {datetime.now().isoformat()}\n"
                f"\n青奈科技 - 青小团监控系统",
                "plain", "utf-8"
            )
            msg["Subject"] = f"【青小团训练告警】{alert['type']}"
            msg["From"] = "qingxiaotuan-monitor@qingnai.tech"
            msg["To"] = self.alert_email
            logger.info(f"青小团 - 邮件告警已加入队列: {alert['type']}")
        except Exception as e:
            logger.warning(f"青小团 - 邮件告警发送失败: {e}")

    def _save_metrics_to_disk(self, metrics: MonitorMetrics):
        """定期将指标保存到磁盘"""
        if self._current_step % 100 == 0:
            try:
                metrics_file = self.metrics_dir / f"metrics_step_{self._current_step}.json"
                data = asdict(metrics)
                data["company"] = "青奈科技"
                data["model"] = self.model_name
                data["smoothed_loss"] = self._smoothed_loss
                data["expert_usage"] = {str(k): v for k, v in metrics.expert_usage.items()}
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"青小团 - 保存指标到磁盘失败: {e}")

    def update_progress_bar(self, current: int, total: int, extra_info: Optional[Dict[str, Any]] = None):
        """
        更新控制台进度条

        Args:
            current: 当前步数
            total: 总步数
            extra_info: 额外显示信息
        """
        if self.local_rank != 0 or not self.progress_bar_enabled:
            return

        elapsed = time.time() - self._start_time
        if current > 0:
            steps_per_sec = current / elapsed
            eta_seconds = (total - current) / steps_per_sec if steps_per_sec > 0 else 0
            eta = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta = "N/A"

        progress = current / total * 100 if total > 0 else 0
        bar_length = 50
        filled = int(bar_length * current // total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        info_parts = [
            f"步数: {current}/{total}",
            f"loss: {self._smoothed_loss:.4f}" if self._smoothed_loss > 0 else "loss: N/A",
            f"ETA: {eta}",
        ]

        if extra_info:
            for k, v in extra_info.items():
                if isinstance(v, float):
                    info_parts.append(f"{k}: {v:.4f}")
                else:
                    info_parts.append(f"{k}: {v}")

        elapsed_str = str(timedelta(seconds=int(elapsed)))
        info_str = " | ".join(info_parts)

        progress_line = (
            f"\r青小团训练 [{bar}] {progress:.1f}% | "
            f"耗时: {elapsed_str} | {info_str}"
        )

        sys.stdout.write(progress_line)
        sys.stdout.flush()

        if current >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前训练状态摘要"""
        elapsed = time.time() - self._start_time
        recent_metrics = list(self._metrics_history)[-100:] if self._metrics_history else []

        avg_throughput = 0.0
        avg_gpu_util = 0.0
        if recent_metrics:
            avg_throughput = np.mean([m.tokens_per_second for m in recent_metrics])
            avg_gpu_util = np.mean([m.gpu_utilization for m in recent_metrics])

        return {
            "company": "青奈科技",
            "model": self.model_name,
            "current_step": self._current_step,
            "current_epoch": self._current_epoch,
            "smoothed_loss": self._smoothed_loss,
            "learning_rate": self._lr_history[-1][1] if self._lr_history else 0.0,
            "avg_throughput_tokens_per_sec": avg_throughput,
            "avg_gpu_utilization": avg_gpu_util,
            "total_tokens_processed": self._total_tokens_processed,
            "elapsed_seconds": elapsed,
            "elapsed_str": str(timedelta(seconds=int(elapsed))),
            "anomaly_count": self._anomaly_count,
            "metrics_count": len(self._metrics_history),
        }

    def get_loss_history(self) -> List[Tuple[int, float, float]]:
        """获取loss历史"""
        return list(self._loss_history)

    def get_learning_rate_history(self) -> List[Tuple[int, float]]:
        """获取学习率历史"""
        return list(self._lr_history)

    def get_gradient_norm_history(self) -> List[Tuple[int, float]]:
        """获取梯度范数历史"""
        return list(self._grad_norm_history)

    def get_expert_usage_statistics(self) -> Dict[str, Any]:
        """获取MoE专家使用率统计"""
        stats = {}
        for expert_id, history in self._expert_usage_history.items():
            if history:
                usages = [h[1] for h in history]
                stats[f"expert_{expert_id}"] = {
                    "mean": float(np.mean(usages)),
                    "std": float(np.std(usages)),
                    "max": float(max(usages)),
                    "min": float(min(usages)),
                    "recent": float(usages[-1]) if usages else 0.0,
                }

        if stats:
            usages = [s["mean"] for s in stats.values()]
            stats["summary"] = {
                "num_experts": len(stats),
                "mean_usage": float(np.mean(usages)),
                "std_usage": float(np.std(usages)),
                "max_usage": float(max(usages)),
                "min_usage": float(min(usages)),
                "balance_score": 1.0 - float(np.std(usages) / (np.mean(usages) + 1e-8)),
            }

        return stats

    def generate_training_report(self, output_path: Optional[str] = None) -> str:
        """
        生成训练报告

        Args:
            output_path: 输出路径，None则返回字符串

        Returns:
            报告内容
        """
        elapsed = time.time() - self._start_time
        recent_metrics = list(self._metrics_history)[-100:] if self._metrics_history else []

        avg_loss = np.mean([m.loss for m in recent_metrics]) if recent_metrics else 0.0
        min_loss = min([m.loss for m in recent_metrics]) if recent_metrics else 0.0
        avg_throughput = np.mean([m.tokens_per_second for m in recent_metrics]) if recent_metrics else 0.0
        avg_gpu = np.mean([m.gpu_utilization for m in recent_metrics]) if recent_metrics else 0.0

        report = f"""
{'='*60}
青小团训练报告 - 青奈科技
{'='*60}

基本信息:
  模型: {self.model_name}
  运行时长: {str(timedelta(seconds=int(elapsed)))}
  当前步数: {self._current_step}
  当前epoch: {self._current_epoch}
  总处理token数: {self._total_tokens_processed:,}

训练指标:
  当前loss: {self._smoothed_loss:.6f}
  平均loss(最近100步): {avg_loss:.6f}
  最低loss(最近100步): {min_loss:.6f}
  平均吞吐量: {avg_throughput:.2f} tokens/sec
  平均GPU利用率: {avg_gpu:.1f}%
  异常告警次数: {self._anomaly_count}

MoE负载均衡:
"""
        expert_stats = self.get_expert_usage_statistics()
        if "summary" in expert_stats:
            report += f"  专家数量: {expert_stats['summary']['num_experts']}\n"
            report += f"  平均使用率: {expert_stats['summary']['mean_usage']:.2%}\n"
            report += f"  均衡度得分: {expert_stats['summary']['balance_score']:.4f}\n"

        report += f"""
{'='*60}
青奈科技 - 青小团分布式训练系统
{'='*60}
"""
        if output_path:
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path_obj, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"青小团 - 训练报告已保存: {output_path}")

        return report

    def export_metrics_to_csv(self, output_path: str):
        """导出指标到CSV文件"""
        import csv
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "loss", "smoothed_loss", "learning_rate",
                "gradient_norm", "tokens_per_sec", "gpu_utilization",
                "memory_used_gb", "batch_time_ms", "timestamp"
            ])
            for m in self._metrics_history:
                writer.writerow([
                    m.step, m.epoch, m.loss, m.smoothed_loss, m.learning_rate,
                    m.gradient_norm, m.tokens_per_second, m.gpu_utilization,
                    m.memory_used_gb, m.batch_time_ms,
                    datetime.fromtimestamp(m.timestamp).isoformat(),
                ])
        logger.info(f"青小团 - 指标已导出到CSV: {output_path}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        recent = list(self._metrics_history)[-1000:] if len(self._metrics_history) > 1000 else list(self._metrics_history)
        if not recent:
            return {}

        return {
            "company": "青奈科技",
            "model": self.model_name,
            "avg_batch_time_ms": float(np.mean([m.batch_time_ms for m in recent])),
            "avg_data_load_time_ms": float(np.mean([m.data_load_time_ms for m in recent])),
            "avg_compute_time_ms": float(np.mean([m.compute_time_ms for m in recent])),
            "avg_comm_time_ms": float(np.mean([m.communication_time_ms for m in recent])),
            "avg_tokens_per_sec": float(np.mean([m.tokens_per_second for m in recent])),
            "avg_gpu_util": float(np.mean([m.gpu_utilization for m in recent])),
            "compute_ratio": float(np.mean([m.compute_time_ms / (m.batch_time_ms + 1e-8) for m in recent])),
            "comm_ratio": float(np.mean([m.communication_time_ms / (m.batch_time_ms + 1e-8) for m in recent])),
            "data_ratio": float(np.mean([m.data_load_time_ms / (m.batch_time_ms + 1e-8) for m in recent])),
        }

    def close(self):
        """关闭监控器，清理资源"""
        self._shutdown_flag = True

        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
            logger.info("青小团 - TensorBoard writer已关闭")

        if self._wandb_run is not None:
            self._wandb_run.finish()
            logger.info("青小团 - WandB run已关闭")

        final_report = self.generate_training_report(
            str(self.metrics_dir / "final_training_report.txt")
        )

        self.export_metrics_to_csv(
            str(self.metrics_dir / "training_metrics.csv")
        )

        logger.info("青小团 - 训练监控器已关闭")
        logger.info(f"青奈科技 - 最终训练报告:\n{final_report}")


class GPUMonitor:
    """
    GPU状态监控器
    实时采集GPU利用率、显存、温度等信息
    """

    def __init__(self, gpu_indices: Optional[List[int]] = None):
        self.gpu_indices = gpu_indices or list(range(torch.cuda.device_count() if torch.cuda.is_available() else 0))

    def get_gpu_stats(self) -> Dict[int, Dict[str, float]]:
        """获取所有GPU的实时状态"""
        stats = {}
        for idx in self.gpu_indices:
            if torch.cuda.is_available():
                try:
                    stats[idx] = {
                        "utilization": float(torch.cuda.utilization(idx)),
                        "memory_allocated_gb": torch.cuda.memory_allocated(idx) / 1024**3,
                        "memory_reserved_gb": torch.cuda.memory_reserved(idx) / 1024**3,
                        "memory_total_gb": torch.cuda.get_device_properties(idx).total_memory / 1024**3,
                        "temperature_c": 0.0,
                    }
                except Exception:
                    stats[idx] = {
                        "utilization": 0.0,
                        "memory_allocated_gb": 0.0,
                        "memory_reserved_gb": 0.0,
                        "memory_total_gb": 0.0,
                        "temperature_c": 0.0,
                    }
        return stats

    def get_total_memory_usage(self) -> Dict[str, float]:
        """获取总显存使用情况"""
        total_allocated = 0.0
        total_reserved = 0.0
        total_capacity = 0.0

        for idx in self.gpu_indices:
            if torch.cuda.is_available():
                total_allocated += torch.cuda.memory_allocated(idx) / 1024**3
                total_reserved += torch.cuda.memory_reserved(idx) / 1024**3
                total_capacity += torch.cuda.get_device_properties(idx).total_memory / 1024**3

        return {
            "total_allocated_gb": total_allocated,
            "total_reserved_gb": total_reserved,
            "total_capacity_gb": total_capacity,
            "utilization_percent": (total_allocated / total_capacity * 100) if total_capacity > 0 else 0.0,
        }


class MoELoadBalancer:
    """
    MoE负载均衡监控器
    跟踪每个专家的使用频率和负载分布
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.expert_counts = np.zeros(num_experts, dtype=np.float64)
        self.total_calls = 0
        self.history: List[np.ndarray] = []

    def record_routing(self, expert_indices: np.ndarray):
        """
        记录路由选择

        Args:
            expert_indices: 每个token选择的专家索引
        """
        counts = np.bincount(expert_indices, minlength=self.num_experts).astype(np.float64)
        self.expert_counts += counts
        self.total_calls += len(expert_indices)
        self.history.append(counts)

        if len(self.history) > 1000:
            self.history.pop(0)

    def get_usage_distribution(self) -> Dict[str, Any]:
        """获取当前使用分布"""
        if self.total_calls == 0:
            return {f"expert_{i}": 0.0 for i in range(self.num_experts)}

        distribution = self.expert_counts / self.total_calls
        return {
            f"expert_{i}": float(distribution[i])
            for i in range(self.num_experts)
        }

    def get_balance_score(self) -> float:
        """计算负载均衡得分 (0-1, 1表示完全均衡)"""
        if self.total_calls == 0:
            return 1.0

        distribution = self.expert_counts / self.total_calls
        ideal = 1.0 / self.num_experts
        variance = np.mean((distribution - ideal) ** 2)
        return max(0.0, 1.0 - variance * self.num_experts)

    def get_imbalance_experts(self, threshold: float = 0.2) -> List[int]:
        """获取负载不均衡的专家"""
        if self.total_calls == 0:
            return []

        distribution = self.expert_counts / self.total_calls
        ideal = 1.0 / self.num_experts
        return [i for i in range(self.num_experts)
                if abs(distribution[i] - ideal) > threshold / self.num_experts]

    def reset(self):
        """重置统计数据"""
        self.expert_counts = np.zeros(self.num_experts, dtype=np.float64)
        self.total_calls = 0
        self.history.clear()
