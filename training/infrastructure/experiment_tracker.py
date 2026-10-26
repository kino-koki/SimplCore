"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

experiment_tracker.py - 实验追踪模块
记录实验配置、版本管理、结果比较和自动生成报告
"""

import os
import sys
import json
import time
import uuid
import logging
import copy
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import traceback

import numpy as np

logger = logging.getLogger("青小团.ExperimentTracker")


@dataclass
class ExperimentConfig:
    """青小团实验配置"""
    experiment_id: str = ""
    experiment_name: str = ""
    company: str = "青奈科技"
    model_name: str = "青小团"
    created_at: str = ""
    updated_at: str = ""
    status: str = "created"  # created, running, completed, failed, aborted
    tags: List[str] = field(default_factory=list)
    description: str = ""

    # Model architecture
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training hyperparameters
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Distributed training config
    distributed_config: Dict[str, Any] = field(default_factory=dict)
    
    # Infrastructure config
    infrastructure_config: Dict[str, Any] = field(default_factory=dict)
    
    # Optimizer config
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    
    # Learning rate scheduler
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    
    # MoE specific config
    moe_config: Dict[str, Any] = field(default_factory=dict)
    
    # Extra config
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentResult:
    """青小团实验结果"""
    experiment_id: str = ""
    company: str = "青奈科技"
    final_loss: float = float("inf")
    best_loss: float = float("inf")
    final_step: int = 0
    total_steps: int = 0
    total_epochs: int = 0
    total_tokens_processed: int = 0
    total_training_time_seconds: float = 0.0
    avg_throughput_tokens_per_sec: float = 0.0
    avg_gpu_utilization: float = 0.0
    peak_memory_gb: float = 0.0
    anomaly_count: int = 0
    loss_history: List[float] = field(default_factory=list)
    learning_rate_history: List[float] = field(default_factory=list)
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ExperimentTracker:
    """
    青小团实验追踪器
    管理实验的全生命周期，支持配置记录、版本管理、结果比较和报告生成
    """

    def __init__(
        self,
        storage_dir: str,
        experiment_name: Optional[str] = None,
        auto_save: bool = True,
        local_rank: int = 0,
    ):
        self.storage_dir = Path(storage_dir)
        self.auto_save = auto_save
        self.local_rank = local_rank

        self.experiments_dir = self.storage_dir / "experiments"
        self.reports_dir = self.storage_dir / "reports"
        self.comparisons_dir = self.storage_dir / "comparisons"

        if self.local_rank == 0:
            self.experiments_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            self.comparisons_dir.mkdir(parents=True, exist_ok=True)

        self._current_experiment: Optional[ExperimentConfig] = None
        self._current_result: Optional[ExperimentResult] = None
        self._experiments_cache: Dict[str, ExperimentConfig] = {}
        self._results_cache: Dict[str, ExperimentResult] = {}
        self._lock = threading.Lock()

        self._load_existing_experiments()

        if experiment_name:
            self.create_experiment(experiment_name)

        if self.local_rank == 0:
            logger.info(f"青小团实验追踪器初始化完成，存储目录: {self.storage_dir}")
            logger.info(f"青奈科技 - 已加载 {len(self._experiments_cache)} 个实验")

    def _load_existing_experiments(self):
        """加载已有的实验数据"""
        if not self.experiments_dir.exists():
            return

        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                config_file = exp_dir / "config.json"
                result_file = exp_dir / "result.json"

                if config_file.exists():
                    try:
                        with open(config_file, "r", encoding="utf-8") as f:
                            config = ExperimentConfig.from_dict(json.load(f))
                        self._experiments_cache[config.experiment_id] = config
                    except Exception as e:
                        logger.warning(f"青小团 - 加载实验配置失败 {config_file}: {e}")

                if result_file.exists():
                    try:
                        with open(result_file, "r", encoding="utf-8") as f:
                            result = ExperimentResult.from_dict(json.load(f))
                        self._results_cache[result.experiment_id] = result
                    except Exception as e:
                        logger.warning(f"青小团 - 加载实验结果失败 {result_file}: {e}")

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        创建新实验

        Args:
            name: 实验名称
            description: 实验描述
            tags: 实验标签
            config: 实验配置

        Returns:
            实验ID
        """
        experiment_id = f"qxt_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._current_experiment = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=name,
            company="青奈科技",
            model_name="青小团",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            status="created",
            tags=tags or [],
            description=description,
        )

        if config:
            self._current_experiment.model_config = config.get("model_config", {})
            self._current_experiment.training_config = config.get("training_config", {})
            self._current_experiment.data_config = config.get("data_config", {})
            self._current_experiment.distributed_config = config.get("distributed_config", {})
            self._current_experiment.infrastructure_config = config.get("infrastructure_config", {})
            self._current_experiment.optimizer_config = config.get("optimizer_config", {})
            self._current_experiment.scheduler_config = config.get("scheduler_config", {})
            self._current_experiment.moe_config = config.get("moe_config", {})
            self._current_experiment.extra_config = config.get("extra_config", {})

        self._experiments_cache[experiment_id] = self._current_experiment
        self._current_result = ExperimentResult(experiment_id=experiment_id)

        if self.auto_save:
            self._save_experiment()

        if self.local_rank == 0:
            logger.info(f"青小团 - 创建实验: {name} (ID: {experiment_id})")
            logger.info(f"青奈科技 - 实验标签: {tags}")

        return experiment_id

    def start_experiment(self):
        """标记实验开始"""
        if self._current_experiment is None:
            logger.warning("青小团 - 没有活跃的实验可启动")
            return

        with self._lock:
            self._current_experiment.status = "running"
            self._current_experiment.updated_at = datetime.now().isoformat()

        if self.auto_save:
            self._save_experiment()

        logger.info(f"青小团 - 实验开始: {self._current_experiment.experiment_name}")

    def end_experiment(
        self,
        status: str = "completed",
        notes: str = "",
        result_updates: Optional[Dict[str, Any]] = None,
    ):
        """
        结束实验

        Args:
            status: 结束状态 (completed/failed/aborted)
            notes: 实验备注
            result_updates: 结果更新
        """
        if self._current_experiment is None:
            logger.warning("青小团 - 没有活跃的实验可结束")
            return

        with self._lock:
            self._current_experiment.status = status
            self._current_experiment.updated_at = datetime.now().isoformat()

            if self._current_result:
                self._current_result.notes = notes
                if result_updates:
                    for k, v in result_updates.items():
                        if hasattr(self._current_result, k):
                            setattr(self._current_result, k, v)

        if self.auto_save:
            self._save_experiment()
            self._save_result()

        logger.info(
            f"青小团 - 实验结束: {self._current_experiment.experiment_name}, "
            f"状态: {status}"
        )

    def update_config(self, config_updates: Dict[str, Any]):
        """
        更新实验配置

        Args:
            config_updates: 配置更新
        """
        if self._current_experiment is None:
            logger.warning("青小团 - 没有活跃的实验可更新")
            return

        with self._lock:
            for section, values in config_updates.items():
                config_field = f"{section}_config"
                if hasattr(self._current_experiment, config_field):
                    current = getattr(self._current_experiment, config_field)
                    if isinstance(current, dict) and isinstance(values, dict):
                        current.update(values)
                    else:
                        setattr(self._current_experiment, config_field, values)

            self._current_experiment.updated_at = datetime.now().isoformat()

        if self.auto_save:
            self._save_experiment()

    def update_result(self, result_updates: Dict[str, Any]):
        """
        更新实验结果

        Args:
            result_updates: 结果更新
        """
        if self._current_result is None:
            return

        with self._lock:
            for k, v in result_updates.items():
                if hasattr(self._current_result, k):
                    if k == "loss_history" and isinstance(v, list):
                        self._current_result.loss_history = v
                    elif k == "learning_rate_history" and isinstance(v, list):
                        self._current_result.learning_rate_history = v
                    else:
                        setattr(self._current_result, k, v)

        if self.auto_save and len(self._current_result.loss_history) % 100 == 0:
            self._save_result()

    def log_metrics_to_experiment(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        **extra_metrics,
    ):
        """记录指标到当前实验"""
        if self._current_result is None:
            return

        with self._lock:
            self._current_result.loss_history.append(loss)
            self._current_result.learning_rate_history.append(learning_rate)
            self._current_result.final_step = step

            if loss < self._current_result.best_loss:
                self._current_result.best_loss = loss

            self._current_result.final_loss = loss

            for k, v in extra_metrics.items():
                if k not in ("loss_history", "learning_rate_history"):
                    self._current_result.extra[k] = v

    def add_tag(self, tag: str):
        """添加标签到当前实验"""
        if self._current_experiment is None:
            return
        with self._lock:
            if tag not in self._current_experiment.tags:
                self._current_experiment.tags.append(tag)
                self._current_experiment.updated_at = datetime.now().isoformat()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """获取实验配置"""
        return self._experiments_cache.get(experiment_id)

    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """获取实验结果"""
        return self._results_cache.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "created_at",
        limit: int = 50,
    ) -> List[ExperimentConfig]:
        """
        列出实验

        Args:
            status: 按状态过滤
            tags: 按标签过滤
            sort_by: 排序字段
            limit: 返回数量

        Returns:
            实验列表
        """
        experiments = list(self._experiments_cache.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        if tags:
            experiments = [
                e for e in experiments
                if any(tag in e.tags for tag in tags)
            ]

        if sort_by == "created_at":
            experiments.sort(key=lambda e: e.created_at, reverse=True)
        elif sort_by == "updated_at":
            experiments.sort(key=lambda e: e.updated_at, reverse=True)
        elif sort_by == "name":
            experiments.sort(key=lambda e: e.experiment_name)

        return experiments[:limit]

    def search_experiments(self, query: str) -> List[ExperimentConfig]:
        """搜索实验"""
        query = query.lower()
        results = []
        for exp in self._experiments_cache.values():
            if (
                query in exp.experiment_name.lower()
                or query in exp.description.lower()
                or query in exp.experiment_id.lower()
                or any(query in tag.lower() for tag in exp.tags)
            ):
                results.append(exp)
        return results

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        比较多个实验

        Args:
            experiment_ids: 实验ID列表
            metrics: 要比较的指标

        Returns:
            比较结果
        """
        default_metrics = [
            "final_loss", "best_loss", "final_step", "total_steps",
            "total_training_time_seconds", "avg_throughput_tokens_per_sec",
            "avg_gpu_utilization", "peak_memory_gb", "anomaly_count",
        ]
        metrics = metrics or default_metrics

        comparison = {
            "company": "青奈科技",
            "model": "青小团",
            "comparison_time": datetime.now().isoformat(),
            "num_experiments": len(experiment_ids),
            "experiments": [],
            "summary": {},
        }

        for exp_id in experiment_ids:
            config = self.get_experiment(exp_id)
            result = self.get_result(exp_id)

            if config is None:
                continue

            exp_data = {
                "experiment_id": exp_id,
                "experiment_name": config.experiment_name,
                "status": config.status,
                "tags": config.tags,
                "created_at": config.created_at,
                "config_summary": {
                    "model": config.model_config,
                    "training": {k: v for k, v in config.training_config.items()
                                if k in ["batch_size", "learning_rate", "num_epochs", "warmup_steps"]},
                    "moe": {k: v for k, v in config.moe_config.items()
                           if k in ["num_experts", "top_k", "expert_capacity"]},
                },
            }

            if result:
                exp_data["results"] = {
                    metric: getattr(result, metric, None)
                    for metric in metrics
                }

            comparison["experiments"].append(exp_data)

        if comparison["experiments"]:
            results_only = [
                e["results"] for e in comparison["experiments"]
                if "results" in e and e["results"].get("best_loss") is not None
            ]
            if results_only:
                comparison["summary"] = {
                    "best_overall_loss": min(
                        r.get("best_loss", float("inf")) for r in results_only
                    ),
                    "best_experiment": comparison["experiments"][
                        min(
                            range(len(results_only)),
                            key=lambda i: results_only[i].get("best_loss", float("inf")),
                        )
                    ]["experiment_name"],
                    "avg_best_loss": float(np.mean([
                        r.get("best_loss", float("inf")) for r in results_only
                    ])),
                }

        comparison_path = self.comparisons_dir / f"comparison_{len(experiment_ids)}exps_{int(time.time())}.json"
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        return comparison

    def generate_report(self, experiment_id: Optional[str] = None) -> str:
        """
        生成实验报告

        Args:
            experiment_id: 实验ID，None使用当前实验

        Returns:
            报告内容(Markdown格式)
        """
        if experiment_id:
            config = self.get_experiment(experiment_id)
            result = self.get_result(experiment_id)
        else:
            config = self._current_experiment
            result = self._current_result

        if config is None:
            return "未找到实验数据"

        report = f"""# 青小团实验报告

**公司**: 青奈科技
**模型**: {config.model_name}
**实验名称**: {config.experiment_name}
**实验ID**: {config.experiment_id}
**状态**: {config.status}
**创建时间**: {config.created_at}
**标签**: {', '.join(config.tags) if config.tags else '无'}

---

## 1. 实验配置

### 模型架构
"""
        for k, v in config.model_config.items():
            report += f"- **{k}**: {v}\n"

        report += "\n### 训练超参数\n"
        for k, v in config.training_config.items():
            report += f"- **{k}**: {v}\n"

        report += "\n### 分布式配置\n"
        for k, v in config.distributed_config.items():
            report += f"- **{k}**: {v}\n"

        report += "\n### MoE配置\n"
        for k, v in config.moe_config.items():
            report += f"- **{k}**: {v}\n"

        if result:
            report += f"""
---

## 2. 实验结果

| 指标 | 值 |
|------|-----|
| 最终Loss | {result.final_loss:.6f} |
| 最优Loss | {result.best_loss:.6f} |
| 最终步数 | {result.final_step} |
| 总步数 | {result.total_steps} |
| 总处理Token数 | {result.total_tokens_processed:,} |
| 训练时长 | {result.total_training_time_seconds:.1f}秒 ({result.total_training_time_seconds/3600:.2f}小时) |
| 平均吞吐量 | {result.avg_throughput_tokens_per_sec:.2f} tokens/sec |
| 平均GPU利用率 | {result.avg_gpu_utilization:.1f}% |
| 峰值显存 | {result.peak_memory_gb:.1f} GB |
| 异常告警次数 | {result.anomaly_count} |
"""

            if result.loss_history:
                recent_losses = result.loss_history[-100:]
                report += f"\n### Loss统计(最近100步)\n"
                report += f"- 当前: {recent_losses[-1]:.6f}\n"
                report += f"- 平均: {float(np.mean(recent_losses)):.6f}\n"
                report += f"- 最低: {float(np.min(recent_losses)):.6f}\n"
                report += f"- 最高: {float(np.max(recent_losses)):.6f}\n"

            if result.eval_metrics:
                report += "\n### 评估指标\n"
                for k, v in result.eval_metrics.items():
                    report += f"- **{k}**: {v}\n"

        if result and result.notes:
            report += f"\n## 3. 备注\n\n{result.notes}\n"

        report += f"""

---

*报告生成时间: {datetime.now().isoformat()}*
*青奈科技 - 青小团分布式训练系统*
"""

        if experiment_id:
            report_path = self.reports_dir / f"report_{experiment_id}.md"
        else:
            report_path = self.reports_dir / f"report_current.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"青小团 - 实验报告已生成: {report_path}")
        return report

    def duplicate_experiment(
        self,
        source_id: str,
        new_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        复制实验

        Args:
            source_id: 源实验ID
            new_name: 新实验名称
            config_overrides: 配置覆盖

        Returns:
            新实验ID
        """
        source = self.get_experiment(source_id)
        if source is None:
            logger.error(f"青小团 - 源实验不存在: {source_id}")
            return None

        config = source.to_dict()
        config.pop("experiment_id", None)
        config.pop("experiment_name", None)
        config.pop("created_at", None)
        config.pop("updated_at", None)
        config.pop("status", None)

        if config_overrides:
            for section, values in config_overrides.items():
                config_key = f"{section}_config"
                if config_key in config and isinstance(config[config_key], dict):
                    config[config_key].update(values)

        return self.create_experiment(
            name=new_name,
            description=f"复制自 {source.experiment_name}",
            tags=source.tags,
            config=config,
        )

    def _save_experiment(self):
        """保存实验配置到磁盘"""
        if self._current_experiment is None:
            return

        try:
            exp_dir = self.experiments_dir / self._current_experiment.experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            config_file = exp_dir / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self._current_experiment.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"青小团 - 保存实验配置失败: {e}")

    def _save_result(self):
        """保存实验结果到磁盘"""
        if self._current_result is None:
            return

        try:
            exp_dir = self.experiments_dir / self._current_result.experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            result_file = exp_dir / "result.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(self._current_result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"青小团 - 保存实验结果失败: {e}")

    def export_experiment_data(self, experiment_id: str, output_dir: str) -> str:
        """导出实验数据到指定目录"""
        config = self.get_experiment(experiment_id)
        result = self.get_result(experiment_id)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if config:
            exp_config_path = output_path / f"{experiment_id}_config.json"
            with open(exp_config_path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

        if result:
            exp_result_path = output_path / f"{experiment_id}_result.json"
            with open(exp_result_path, "w", encoding="utf-8") as f:
                data = result.to_dict()
                data["loss_history"] = result.loss_history
                data["learning_rate_history"] = result.learning_rate_history
                json.dump(data, f, ensure_ascii=False, indent=2)

            if result.loss_history:
                loss_path = output_path / f"{experiment_id}_loss.csv"
                with open(loss_path, "w", encoding="utf-8") as f:
                    f.write("step,loss\n")
                    for i, loss in enumerate(result.loss_history):
                        f.write(f"{i},{loss}\n")

        return str(output_path)

    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        if experiment_id in self._experiments_cache:
            del self._experiments_cache[experiment_id]
        if experiment_id in self._results_cache:
            del self._results_cache[experiment_id]

        exp_dir = self.experiments_dir / experiment_id
        if exp_dir.exists():
            import shutil
            shutil.rmtree(exp_dir)
            logger.info(f"青小团 - 已删除实验: {experiment_id}")
            return True

        return False

    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取实验摘要统计"""
        total = len(self._experiments_cache)
        by_status = {}
        total_training_time = 0.0
        best_losses = []

        for exp in self._experiments_cache.values():
            by_status[exp.status] = by_status.get(exp.status, 0) + 1

        for result in self._results_cache.values():
            total_training_time += result.total_training_time_seconds
            if result.best_loss < float("inf"):
                best_losses.append(result.best_loss)

        return {
            "company": "青奈科技",
            "model": "青小团",
            "total_experiments": total,
            "experiments_by_status": by_status,
            "total_training_time_hours": total_training_time / 3600,
            "avg_best_loss": float(np.mean(best_losses)) if best_losses else None,
            "min_best_loss": float(np.min(best_losses)) if best_losses else None,
        }

    def close(self):
        """关闭实验追踪器"""
        if self._current_experiment and self._current_experiment.status == "running":
            self.end_experiment(status="aborted", notes="实验追踪器关闭")

        logger.info("青小团 - 实验追踪器已关闭")
        logger.info(f"青奈科技 - 实验摘要: {self.get_summary_statistics()}")
