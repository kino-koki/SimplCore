"""
青小团 70B MoE 分布式训练基础设施
青奈科技 (c) 2026

hyperparameter_optimizer.py - 超参数优化模块
网格搜索、随机搜索、贝叶斯优化、学习率曲线搜索
"""

import os
import sys
import json
import time
import math
import random
import logging
import itertools
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import traceback

import numpy as np

logger = logging.getLogger("青小团.HyperparameterOptimizer")


@dataclass
class HyperparameterTrial:
    """青小团超参数试验"""
    trial_id: int = 0
    company: str = "青奈科技"
    params: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterTrial":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SearchStrategy:
    """搜索策略基类"""
    def __init__(self, param_space: Dict[str, List[Any]]):
        self.param_space = param_space

    def suggest(self) -> Dict[str, Any]:
        raise NotImplementedError

    def update(self, trial: HyperparameterTrial):
        raise NotImplementedError

    def get_best_params(self) -> Dict[str, Any]:
        raise NotImplementedError


class GridSearch(SearchStrategy):
    """网格搜索"""

    def __init__(self, param_space: Dict[str, List[Any]]):
        super().__init__(param_space)
        self._combinations = list(itertools.product(*param_space.values()))
        self._param_names = list(param_space.keys())
        self._index = 0
        self._trials: List[HyperparameterTrial] = []

    def suggest(self) -> Dict[str, Any]:
        if self._index >= len(self._combinations):
            return {}
        combination = self._combinations[self._index]
        self._index += 1
        return dict(zip(self._param_names, combination))

    def update(self, trial: HyperparameterTrial):
        self._trials.append(trial)

    def get_best_params(self) -> Dict[str, Any]:
        completed = [t for t in self._trials if t.status == "completed" and t.result]
        if not completed:
            return {}
        best = min(completed, key=lambda t: t.result.get("loss", float("inf")))
        return best.params

    def total_trials(self) -> int:
        return len(self._combinations)

    def remaining_trials(self) -> int:
        return max(0, self.total_trials() - self._index)


class RandomSearch(SearchStrategy):
    """随机搜索"""

    def __init__(self, param_space: Dict[str, List[Any]], max_trials: int = 100):
        super().__init__(param_space)
        self.max_trials = max_trials
        self._trials: List[HyperparameterTrial] = []
        self._suggested_count = 0
        self._param_names = list(param_space.keys())

    def suggest(self) -> Dict[str, Any]:
        if self._suggested_count >= self.max_trials:
            return {}
        self._suggested_count += 1
        return {
            name: random.choice(values)
            for name, values in self.param_space.items()
        }

    def update(self, trial: HyperparameterTrial):
        self._trials.append(trial)

    def get_best_params(self) -> Dict[str, Any]:
        completed = [t for t in self._trials if t.status == "completed" and t.result]
        if not completed:
            return {}
        best = min(completed, key=lambda t: t.result.get("loss", float("inf")))
        return best.params


class BayesianOptimization(SearchStrategy):
    """
    贝叶斯优化
    使用Tree Parzen Estimator (TPE) 进行超参数搜索
    """

    def __init__(
        self,
        param_space: Dict[str, List[Any]],
        n_initial_trials: int = 10,
        gamma: float = 0.25,
    ):
        super().__init__(param_space)
        self.n_initial_trials = n_initial_trials
        self.gamma = gamma
        self._trials: List[HyperparameterTrial] = []
        self._param_names = list(param_space.keys())
        self._suggested_count = 0
        self._best_trial: Optional[HyperparameterTrial] = None

    def suggest(self) -> Dict[str, Any]:
        self._suggested_count += 1

        completed = [t for t in self._trials if t.status == "completed" and t.result]
        if len(completed) < self.n_initial_trials:
            return {
                name: random.choice(values)
                for name, values in self.param_space.items()
            }

        params = {}
        for name in self._param_names:
            values = self.param_space[name]
            if len(values) <= 1:
                params[name] = values[0]
                continue

            losses = np.array([t.result.get("loss", float("inf")) for t in completed])
            param_values = np.array([t.params[name] for t in completed])

            threshold = np.percentile(losses, self.gamma * 100)
            l_mask = losses < threshold
            g_mask = ~l_mask

            if np.sum(l_mask) < 1 or np.sum(g_mask) < 1:
                params[name] = random.choice(values)
                continue

            l_values = param_values[l_mask]
            g_values = param_values[g_mask]

            if isinstance(values[0], (int, float)):
                l_mean = np.mean(l_values)
                l_std = max(np.std(l_values), 1e-8)
                g_mean = np.mean(g_values)
                g_std = max(np.std(g_values), 1e-8)

                candidates = np.linspace(min(values), max(values), 100)
                l_probs = self._gaussian_pdf(candidates, l_mean, l_std)
                g_probs = self._gaussian_pdf(candidates, g_mean, g_std)
                ei = l_probs / (g_probs + 1e-8)
                best_idx = np.argmax(ei)
                params[name] = float(candidates[best_idx])
            else:
                l_counts = {}
                for v in l_values:
                    l_counts[v] = l_counts.get(v, 0) + 1
                best_value = max(l_counts, key=l_counts.get)
                params[name] = best_value

        return params

    def _gaussian_pdf(self, x: np.ndarray, mean: float, std: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    def update(self, trial: HyperparameterTrial):
        self._trials.append(trial)
        if trial.status == "completed" and trial.result:
            if self._best_trial is None or trial.result.get("loss", float("inf")) < self._best_trial.result.get("loss", float("inf")):
                self._best_trial = trial

    def get_best_params(self) -> Dict[str, Any]:
        if self._best_trial:
            return self._best_trial.params
        completed = [t for t in self._trials if t.status == "completed" and t.result]
        if not completed:
            return {}
        best = min(completed, key=lambda t: t.result.get("loss", float("inf")))
        return best.params


class HyperparameterOptimizer:
    """
    青小团超参数优化器

    支持网格搜索、随机搜索、贝叶斯优化（optuna）、学习率曲线搜索
    """

    def __init__(
        self,
        param_space: Dict[str, List[Any]],
        search_algorithm: str = "bayesian",
        max_trials: int = 50,
        optimization_metric: str = "loss",
        minimize_metric: bool = True,
        output_dir: str = "./hpo_results",
        n_jobs: int = 1,
        early_stopping_rounds: int = 10,
        early_stopping_threshold: float = 1e-4,
    ):
        self.param_space = param_space
        self.search_algorithm = search_algorithm
        self.max_trials = max_trials
        self.optimization_metric = optimization_metric
        self.minimize_metric = minimize_metric
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stopping_threshold = early_stopping_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._trials: List[HyperparameterTrial] = []
        self._trial_counter = 0
        self._best_result = float("inf") if minimize_metric else float("-inf")
        self._best_params: Dict[str, Any] = {}
        self._no_improvement_rounds = 0
        self._lock = threading.Lock()
        self._is_running = False
        self._objective_function: Optional[Callable] = None

        self._searcher = self._create_searcher()

        if search_algorithm == "optuna":
            try:
                import optuna
                self._optuna_available = True
            except ImportError:
                logger.warning("青小团 - optuna未安装，回退到贝叶斯优化")
                self._optuna_available = False
                self._searcher = BayesianOptimization(param_space)
        else:
            self._optuna_available = False

        logger.info(f"青小团超参数优化器初始化完成")
        logger.info(f"青奈科技 - 算法={search_algorithm}, 最大试验={max_trials}, "
                   f"指标={optimization_metric}")

    def _create_searcher(self) -> SearchStrategy:
        """创建搜索策略"""
        if self.search_algorithm == "grid":
            return GridSearch(self.param_space)
        elif self.search_algorithm == "random":
            return RandomSearch(self.param_space, max_trials=self.max_trials)
        elif self.search_algorithm == "bayesian":
            return BayesianOptimization(self.param_space)
        else:
            return BayesianOptimization(self.param_space)

    def set_objective(self, objective_fn: Callable[[Dict[str, Any]], float]):
        """
        设置目标函数

        Args:
            objective_fn: 接受参数字典，返回loss值
        """
        self._objective_function = objective_fn

    def optimize(
        self,
        objective_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    ) -> Dict[str, Any]:
        """
        执行超参数优化

        Args:
            objective_fn: 目标函数，接受参数字典返回loss值

        Returns:
            最优参数
        """
        if objective_fn:
            self._objective_function = objective_fn

        if self._objective_function is None:
            raise ValueError("青小团 - 未设置目标函数")

        self._is_running = True
        self._start_time = time.time()

        if self._optuna_available and self.search_algorithm == "optuna":
            return self._optimize_with_optuna()

        return self._optimize_with_searcher()

    def _optimize_with_searcher(self) -> Dict[str, Any]:
        """使用内置搜索器进行优化"""
        for trial_idx in range(self.max_trials):
            if not self._is_running:
                break

            params = self._searcher.suggest()
            if not params:
                break

            trial = self._run_trial(params)
            self._searcher.update(trial)

            with self._lock:
                self._trials.append(trial)

            metric_value = trial.result.get(self.optimization_metric, float("inf"))
            improved = (
                metric_value < self._best_result - self.early_stopping_threshold
                if self.minimize_metric
                else metric_value > self._best_result + self.early_stopping_threshold
            )

            if improved:
                self._best_result = metric_value
                self._best_params = params.copy()
                self._no_improvement_rounds = 0
            else:
                self._no_improvement_rounds += 1

            self._log_trial_progress(trial, trial_idx + 1)

            if self._no_improvement_rounds >= self.early_stopping_rounds:
                logger.info(f"青小团 - 提前停止: {self.early_stopping_rounds}轮无改善")
                break

        self._is_running = False
        self._save_results()

        return self._best_params

    def _optimize_with_optuna(self) -> Dict[str, Any]:
        """使用optuna进行优化"""
        try:
            import optuna

            def optuna_objective(trial):
                params = {}
                for name, values in self.param_space.items():
                    if isinstance(values[0], float):
                        params[name] = trial.suggest_float(name, float(min(values)), float(max(values)))
                    elif isinstance(values[0], int):
                        params[name] = trial.suggest_int(name, int(min(values)), int(max(values)))
                    else:
                        params[name] = trial.suggest_categorical(name, values)

                result = self._objective_function(params)
                return result

            study = optuna.create_study(
                direction="minimize" if self.minimize_metric else "maximize",
                study_name="青小团_hpo",
            )

            study.optimize(optuna_objective, n_trials=self.max_trials)

            for i, trial in enumerate(study.trials):
                hpo_trial = HyperparameterTrial(
                    trial_id=i,
                    params=trial.params,
                    result={"loss": trial.value} if trial.value is not None else {},
                    status="completed" if trial.value is not None else "failed",
                )
                self._trials.append(hpo_trial)

            self._best_params = study.best_params
            self._best_result = study.best_value

            self._save_results()
            return study.best_params

        except Exception as e:
            logger.error(f"青小团 - Optuna优化失败: {e}")
            return self._optimize_with_searcher()

    def _run_trial(self, params: Dict[str, Any]) -> HyperparameterTrial:
        """运行单次试验"""
        self._trial_counter += 1
        trial = HyperparameterTrial(
            trial_id=self._trial_counter,
            params=params,
            status="running",
            start_time=datetime.now().isoformat(),
        )

        try:
            start = time.time()
            result = self._objective_function(params)
            duration = time.time() - start

            trial.result = {self.optimization_metric: result}
            trial.status = "completed"
            trial.duration_seconds = duration
            trial.end_time = datetime.now().isoformat()

        except Exception as e:
            trial.status = "failed"
            trial.error = str(e)
            trial.end_time = datetime.now().isoformat()
            logger.error(f"青小团 - 试验 {self._trial_counter} 失败: {e}")

        return trial

    def _log_trial_progress(self, trial: HyperparameterTrial, trial_num: int):
        """记录试验进度"""
        metric = trial.result.get(self.optimization_metric, "N/A")
        if isinstance(metric, float):
            metric_str = f"{metric:.6f}"
        else:
            metric_str = str(metric)

        logger.info(
            f"青小团 - 试验 [{trial_num}/{self.max_trials}] "
            f"ID={trial.trial_id}, "
            f"状态={trial.status}, "
            f"{self.optimization_metric}={metric_str}, "
            f"耗时={trial.duration_seconds:.1f}s"
        )

    def get_trial_results(self) -> List[HyperparameterTrial]:
        """获取所有试验结果"""
        return self._trials

    def get_best_trial(self) -> Optional[HyperparameterTrial]:
        """获取最佳试验"""
        completed = [t for t in self._trials if t.status == "completed" and t.result]
        if not completed:
            return None

        key = lambda t: t.result.get(self.optimization_metric, float("inf"))
        reverse = not self.minimize_metric
        return min(completed, key=key) if not reverse else max(completed, key=key)

    def _save_results(self):
        """保存优化结果"""
        results = {
            "company": "青奈科技",
            "model": "青小团",
            "search_algorithm": self.search_algorithm,
            "max_trials": self.max_trials,
            "optimization_metric": self.optimization_metric,
            "best_params": self._best_params,
            "best_result": self._best_result,
            "total_trials": len(self._trials),
            "completed_trials": sum(1 for t in self._trials if t.status == "completed"),
            "failed_trials": sum(1 for t in self._trials if t.status == "failed"),
            "total_duration_seconds": time.time() - getattr(self, "_start_time", time.time()),
            "trials": [t.to_dict() for t in self._trials],
        }

        results_file = self.output_dir / f"hpo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"青小团 - 超参数优化结果已保存: {results_file}")

    def visualize_results(self, output_dir: Optional[str] = None):
        """
        可视化优化结果

        生成参数重要性、搜索轨迹、参数关系等图表
        """
        out_dir = Path(output_dir or self.output_dir / "visualizations")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            completed = [t for t in self._trials if t.status == "completed" and t.result]
            if not completed:
                logger.warning("青小团 - 无完成试验可供可视化")
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"青小团超参数优化 - {self.search_algorithm}", fontsize=14)

            trial_ids = [t.trial_id for t in completed]
            losses = [t.result.get(self.optimization_metric, 0) for t in completed]

            axes[0, 0].plot(trial_ids, losses, "b-", alpha=0.7)
            axes[0, 0].scatter(trial_ids, losses, c="r", s=20, alpha=0.7)
            axes[0, 0].set_xlabel("试验ID")
            axes[0, 0].set_ylabel(self.optimization_metric)
            axes[0, 0].set_title("搜索轨迹")
            axes[0, 0].grid(True, alpha=0.3)

            durations = [t.duration_seconds for t in completed]
            axes[0, 1].hist(durations, bins=20, alpha=0.7, color="steelblue")
            axes[0, 1].set_xlabel("耗时(秒)")
            axes[0, 1].set_ylabel("频次")
            axes[0, 1].set_title("试验耗时分布")
            axes[0, 1].grid(True, alpha=0.3)

            param_names = list(self.param_space.keys())
            param_importances = self._compute_param_importances(completed)
            if param_importances:
                names = list(param_importances.keys())
                values = list(param_importances.values())
                axes[1, 0].barh(names, values, color="coral", alpha=0.7)
                axes[1, 0].set_xlabel("重要性")
                axes[1, 0].set_title("参数重要性")
                axes[1, 0].grid(True, alpha=0.3)

            if len(param_names) >= 2 and len(completed) > 5:
                p1 = completed[0].params.get(param_names[0], 0)
                p2 = completed[0].params.get(param_names[1], 0)
                if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
                    param_values_1 = [t.params.get(param_names[0], 0) for t in completed]
                    param_values_2 = [t.params.get(param_names[1], 0) for t in completed]
                    sc = axes[1, 1].scatter(param_values_1, param_values_2, c=losses,
                                            cmap="viridis", s=40, alpha=0.7)
                    axes[1, 1].set_xlabel(param_names[0])
                    axes[1, 1].set_ylabel(param_names[1])
                    axes[1, 1].set_title("参数关系")
                    plt.colorbar(sc, ax=axes[1, 1], label=self.optimization_metric)
                    axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = out_dir / "hpo_visualization.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"青小团 - 优化可视化已保存: {plot_path}")

        except ImportError:
            logger.warning("青小团 - matplotlib不可用，跳过可视化")
        except Exception as e:
            logger.warning(f"青小团 - 可视化失败: {e}")

    def _compute_param_importances(
        self,
        trials: List[HyperparameterTrial],
    ) -> Dict[str, float]:
        """计算参数重要性"""
        if len(trials) < 5:
            return {}

        importances = {}
        losses = np.array([t.result.get(self.optimization_metric, float("inf")) for t in trials])

        for name in self.param_space.keys():
            values = [t.params.get(name) for t in trials]
            unique_values = set(values)
            if len(unique_values) <= 1:
                importances[name] = 0.0
                continue

            if isinstance(list(unique_values)[0], (int, float)):
                numeric_values = np.array([float(v) for v in values])
                correlation = np.corrcoef(numeric_values, losses)[0, 1]
                importances[name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                group_losses = {}
                for v, loss in zip(values, losses):
                    if v not in group_losses:
                        group_losses[v] = []
                    group_losses[v].append(loss)
                group_means = [np.mean(g) for g in group_losses.values()]
                importances[name] = float(np.std(group_means)) if len(group_means) > 1 else 0.0

        max_importance = max(importances.values()) if importances else 1.0
        if max_importance > 0:
            importances = {k: v / max_importance for k, v in importances.items()}

        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    def generate_report(self) -> str:
        """生成优化报告"""
        completed = [t for t in self._trials if t.status == "completed" and t.result]
        best = self.get_best_trial()

        report = f"""
# 青小团超参数优化报告

**公司**: 青奈科技
**搜索算法**: {self.search_algorithm}
**优化指标**: {self.optimization_metric}
**总试验数**: {len(self._trials)}
**成功**: {len(completed)}
**失败**: {sum(1 for t in self._trials if t.status == 'failed')}

---

## 最优参数

"""
        if best:
            for k, v in best.params.items():
                report += f"- **{k}**: {v}\n"
            report += f"\n**最优指标值**: {best.result.get(self.optimization_metric, 'N/A'):.6f}\n"

        report += "\n## 参数空间\n"
        for k, v in self.param_space.items():
            report += f"- **{k}**: {v}\n"

        report += """
---

*报告生成时间: {datetime.now().isoformat()}*
*青奈科技 - 青小团分布式训练系统*
"""
        return report

    def stop(self):
        """停止优化"""
        self._is_running = False
        logger.info("青小团 - 超参数优化已停止")
